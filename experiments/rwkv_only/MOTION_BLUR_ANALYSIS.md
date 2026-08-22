# RWKV-Only 运动模糊问题分析与改进方案

## 1. 现象

RWKV-Only ep40 推理 pair20 全图：输出图像在运动区域产生了 **输入中心帧不存在的模糊**。输入中心帧本身是清晰的，模糊是模型处理过程中引入的。

## 2. 根因分析

### 2.1 核心机制：TCA 软对齐的固有限制

TCA 的时序对齐流程：

```
原始 l2(H/2,5帧) ──→ C_omega(softmax注意力) ──→ 增强 l2(H/2,5帧) 按 omega 扭曲 → 加权融合 → F_t_aligned
                                    ↑                                              ↑
                             中心帧Q @ 邻帧K^T / tau                     sum(warped_t × softmax(gate_t))
                                 (ds×ds 低分辨率)                             + center_residual
```

**C_omega 的本质是空间 softmax 注意力矩阵**，不是显式光流位移场：

```python
# TemporalCorrespondence
sim = Q @ K^T / tau           # (B, N, N), N=ds²
omega = softmax(sim)          # 每行是一个概率分布
```

邻帧特征被 warp 的方式是矩阵乘法：
```python
# TemporalAggregation
warped = f_t_nbor @ omega^T   # (B, C, N) — 每个输出像素 = 所有输入像素的加权和
```

**这等价于一个非参数的软扭曲 — 输出每个位置是输入所有位置的凸组合**。

### 2.2 为什么凸组合导致模糊

考虑一个运动像素：中心帧位置 `(i,j)` 对应邻帧位置 `(i',j')`。

| 场景 | C_omega 行为 | Warp 结果 |
|------|-------------|-----------|
| **静止区域** | softmax 峰值集中在正确位置 | 尖锐，等价于点对点拷贝 |
| **运动区域** | 跨帧对应关系不精确，softmax 分散到多个候选位置 | **所有候选位置的加权平均 → 模糊** |
| **遮挡/新出现** | 无正确对应，softmax 趋近均匀分布 | 全局平均 → 严重模糊 |

### 2.3 低分辨率聚合放大模糊

```
TCA 流程:
  l2 (H/2, W/2) → 下采样到 ds≈32  → C_omega (32×32矩阵) → 扭曲 32×32 特征 → 上采样回 H/2
                        ⬇                                   ⬇
                   信息损失                              低分辨率模糊
                                                     上采样后无法恢复细节
```

### 2.4 TCA 输出端的 residual 不足以恢复尖锐度

```python
# TemporalAggregation 最终输出
agg_up = interpolate(softmax_sum, size=(H,W))  # 低分模糊特征上采样
out = norm(agg_up + center_feat)               # + center enhanced features
```

这里的 `center_feat` 是 **经过 MVCShift + SpatialWKV2D + ChannelMix 增强后的中心帧特征**，已不是原始编码特征。residual 连接保留了增强特征的低频结构，但不能恢复被模糊覆盖的高频细节。

### 2.5 卷积融合头无法解耦

```python
# RWKV-Only 融合
fused = concat([l1_feat(sharp), tca_up(blurred), l3_up(coarse)], dim=1)
res_t = head_conv(fused)  # 3层conv: 192→128→64→3
```

`l1_feat` (中心帧，尖锐) 和 `tca_up` (邻帧融合，模糊) 在 channel 维度拼接后通过卷积混合。卷积的 receptive field 有限（3×3×3层），无法全局判断哪些区域该信任 l1 而非 TCA。结果是在运动区域，`tca_up` 的模糊分量 "泄漏" 到输出中。

### 2.6 完整的因果链

```
相机/物体运动
  → 邻帧与中心帧空间错位
    → TemporalCorrespondence 找不到精确对应
      → C_omega softmax 发散 → 多位置加权平均
        → F_t_aligned 在运动区域模糊
          → 上采样到 H 后模糊保留
            → concat 到 192 通道中占 1/3
              → conv fusion 无法完美屏蔽 → **输出模糊**
```

## 3. 与完整三分支模型的对比

完整 F10m5 三分支模型通过以下机制避免此问题：

| 机制 | 模块 | 作用 |
|------|------|------|
| **gamma 约束** | NDPN | `gamma ≤ 0.1` — 时序融合贡献上限 10%，85%+ 来自空间降噪 |
| **detail_residual** | NDPN | `f_enc - F_denoised → 1×1 gate` — 从编码特征补充高频细节 |
| **运动门控** | MCPN | `G_t` — 帧级运动权重，强运动帧被抑制 |
| **逐阶段残差** | SGRF | Stage A→B 梯度隔离，B 的 delta_scale=0.2 限制偏离 |
| **image_center 直连** | SGRF | 原始输入中心帧直接参与重建 |

NDPN 的 temporal_base 机制尤为关键：

```
F_denoised = gamma × F_temporal + (1-gamma) × f_enc_center
     ↑              ↑                        ↑
  γ≤0.1          TCA对齐特征            编码器中心帧(尖锐)
                  (可能模糊)              (100%可信)
```

即使 TCA 模糊，NDPN 也只从中提取 ≤10%，剩余 ≥90% 来自中心帧自身。

## 4. 改进方案

### 方案 A：中心帧优势门控 (★★★ 推荐, 兼容性最佳)

在 TCA 输出端增加可学习的 **per-pixel 门控**，在运动区域自动衰减 TCA 贡献、增强中心帧直连：

```python
# 修改 TemporalAggregation
self.center_gate = nn.Sequential(
    nn.Conv2d(channels * 2, channels, 3, padding=1),  # [center, tca_agg]→gate
    nn.GELU(),
    nn.Conv2d(channels, 1, 3, padding=1),
    nn.Sigmoid(),
)

def forward(self, center_feat, neighbor_feats, C_omega_list):
    # ... existing aggregation ...
    agg_up = interpolate(agg, (H,W))
    alpha = self.center_gate(torch.cat([center_feat, agg_up], dim=1))  # (B,1,H,W)
    out = alpha * center_feat + (1 - alpha) * agg_up
    return self.out_norm(out)
```

**学习信号**：门控在静止区域学 `α→0`（信任 TCA 的多帧降噪），在运动区域学 `α→1`（信任中心帧的清晰度）。

**兼容性**：纯 RWKV-Only 直接适用。三分支模型中，NDPN/MCPN 仍在门控后接收特征，不受影响。

### 方案 B：高频保留残差 (★★☆)

在融合阶段注入中心帧的高频（边缘/纹理）分量，确保细节不被 TCA 模糊淹没：

```python
# 在 RWKVOnlyLLVE 中
self.hf_extract = nn.Conv2d(3, fused_channels, 3, padding=1)  # 从RGB提取高频
self.hf_gate = nn.Sequential(
    nn.Conv2d(fused_channels * 2, fused_channels, 1),
    nn.GELU(),
    nn.Conv2d(fused_channels, 1, 1),
    nn.Sigmoid(),
)

def forward(self, x):
    # ... existing ...
    fused = torch.cat([l1_feat, tca_up, l3_up], dim=1)
    base = self.head(fused)
    
    # 高频注入
    img_center = x[:, T//2]  # (B, 3, H, W)
    hf = self.hf_extract(img_center) - F.avg_pool2d(self.hf_extract(img_center), 3,1,1)
    gate = self.hf_gate(torch.cat([l1_feat, tca_up], dim=1))
    base = base + gate * torch.sigmoid(hf[:, :3])  # attention-gated HF injection
    
    return {"res_t": base}
```

### 方案 C：温度 τ 约束 (★☆☆, 仅限调试)

限制 `TemporalCorrespondence` 的温度 τ 下界，让 softmax 更尖锐：

```python
# 当前: tau = softplus(tau_raw) + 0.05 → 下界0.05, 无上界
# 改为: tau = softplus(tau_raw) + 0.01  # 更低的温度 → 更尖锐的softmax
```

**局限**：低温让 softmax 更 "硬"，但 C_omega @ ds≈32 分辨率本身就丢失了精确的空间位置信息，单靠调 τ 治标不治本。

### 方案 D：多分辨率对齐 (★★☆, 需结构改动)

在 H/2 和 H 各做一次 TCA，H 层 TCA 的 C_omega 在高分辨率下更精确：

```
l1(H) → TCA_lite(H, 小通道16) → F_aligned_H  (精细对齐)
l2(H/2) → TCA(H/2, 64) → F_aligned_l2  (结构对齐)
concat[F_aligned_l2↑, F_aligned_H, l3↑] → head → res_t
```

### 方案 E：显式光流对齐 (★☆☆, 成本高)

用 PWC-Net/Raft 取代 C_omega 软对齐，邻帧按光流 warp 到中心帧坐标系。优点是对运动区域精确，缺点是增加推理延迟和参数量。

## 5. 建议实施顺序

1. **方案 A** 最先实施 — 改动最小，效果在静止区域保持多帧降噪收益，在运动区域自动退化为单帧增强
2. **方案 B** 作为补充 — 高频细节注入可进一步锐化纹理
3. 对三分支全景模型而言，当前 NDPN gamma≤0.1 + detail_residual 已提供等效保护，方案 A 可作为 **额外的安全网** 集成到 TCA 输出端
