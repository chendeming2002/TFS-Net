# RWKV-Only v2 — 概念模型分析与改进路线

## 1. 模型设计

### 1.1 架构图

```
Input: (B, 5, 3, H, W)
  │
  ▼ PyramidEncoder (3-stage FPN)
  │  stage1(l1, H/1, 32ch) → lateral1 → (B,5,64,H,W)
  │  stage2(l2, H/2, 64ch) → lateral2 → (B,5,64,H/2,W/2)
  │  stage3(l3, H/4, 96ch) → lateral3 → (B,5,64,H/4,W/4)
  │
  ├── l1_lat ──── l2_lat ──── l3_lat
  │                │
  │                ▼ TCA (H/2)
  │                │  HaarDWT anchor → MVC-Shift → SpatialWKV2D(4向)
  │                │  → Channel Mix → sace_out
  │                │  → TemporalCorrespondence (cosine sim, ds≈32²)
  │                │     C_omega_list: [4× (B, 1024, 1024)] softmax attention
  │                │  → TemporalAggregation: warp + softmax-weighted fusion
  │                │     F_t_aligned (B,64,H/2,W/2)
  │                │
  │                ├── C_omega_list
  │                │     │ diagonal(4 neighbors)
  │                │     ▼ diag_stack (B,1024,4)
  │                │     ▼ conf_proj: Linear(4→16)→GELU→Linear(16→1)→Sigmoid
  │                │     ▼ conf_map (B,1,32,32) → bilinear↑ → (B,1,H,W)
  │                │     ▼ motion_map = 1 - conf_map
  │                │     ▼
  │                ├── F_t_aligned → bilinear↑ → tca_up (B,64,H,W)
  │                │
  │  l1[:,2](中心帧)                  │
  │     │                              │
  │     ├── proj_l1 → l1_feat          │
  │     │                              │
  │     └── proj_l1_gate → l1_gate ────┤
  │                                     │
  │            motion_map ──────────────┤
  │                                     ▼
  │  tca_gated = tca_up×(1-motion) + l1_gate×motion
  │
  │  l3[:,2](中心帧) → proj_l3 → bilinear↑ → l3_up
  │
  └── concat[l1_feat, tca_gated, l3_up] (B,192,H,W)
        │
        ▼ Head: conv(192→128,3×3)+GELU → conv(128→64,3×3)+GELU → conv(64→3,3×3)+Sigmoid
        │
        ▼ res_t (B,3,H,W)
```

### 1.2 核心机制

**运动门控融合**：从 C_omega 对角线提取空间置信度，生成 motion_map，在运动区域用 l1 单帧特征替代 TCA 多帧融合特征：

```
tca_gated = tca_up × (1 - motion_map) + l1_gate × motion_map
```

- motion→0 (静止): tca_gated ≈ tca_up (多帧降噪)
- motion→1 (运动): tca_gated ≈ l1_gate (单帧尖锐)

### 1.3 关键参数

| 组件 | 参数量 |
|------|:-----:|
| PyramidEncoder | ~280K |
| TCA | ~470K |
| conf_proj | ~2K |
| proj_l1, proj_l1_gate, proj_l3 | ~55K |
| Head | ~82K |
| **总计** | **~0.82M** |

### 1.4 与 v1 的差异

| 机制 | v1 | v2 |
|------|:--:|:--:|
| 融合方式 | concat[l1, tca, l3] 静态 | concat[l1, **tca_gated**, l3] 运动感知 |
| 运动检测 | 无 | C_omega diagonal → conf_proj → motion_map |
| l1 投影 | proj_l1 ×1 | proj_l1 (head用) + proj_l1_gate (门控用) ×2 |
| 参数 | 0.78M | 0.82M (+40K) |

---

## 2. 训练与推理结果

### 2.1 训练曲线

| Epoch | Train Loss | PSNR (256² val) | SSIM |
|:-----:|:----------:|:---------------:|:----:|
| 10 | 0.147 | 19.46 | 0.764 |
| 20 | 0.140 | 19.63 | 0.767 |
| 30 | 0.138 | 19.66 | 0.767 |
| 40 | 0.134 | **20.02** | **0.772** |
| 49 | 0.129 | 19.77 | 0.769 |

vs v1 峰值 (19.80 @ ep30): **+0.22 dB**

### 2.2 单场景推理 (1080×1920 全图, ep49)

| 指标 | pair20 (明亮, 静态) | pair45 (暗光, 运动) |
|------|:--:|:--:|
| PSNR | 26.70 | 15.04 |
| SSIM | 0.915 | 0.679 |
| LPIPS | 0.144 | 0.366 |
| MABD | 0.022 | 0.137 |
| motion_map avg | 0.640 | 0.640 |

### 2.3 问题诊断

**问题 1：motion_map 退化为全局偏置**

两个场景 motion_map 均为 0.640，场景间无分化。根因：
- conf_proj 输入维度仅 4（4 个邻帧的 C_omega 对角线值），表达能力极低
- → 学到的是整个数据集的平均最优值（0.64 = 64%信任 l1, 36%信任 TCA）
- → 丧失了"逐空间位置动态路由"的设计初衷

**问题 2：暗光运动场景 PSNR 极低**

pair45 PSNR=15.04 vs 测试集平均 19.77。原因：
- 无显式光照估计 → 暗区的亮度提升完全依赖 3 层 conv 隐式学习
- 运动 + 低 SNR → TCA 对齐失败 + l1 单帧噪声大 → 两难
- 无 SNR 感知的帧间聚合 → 噪声帧和清晰帧等权参与

**问题 3：单层 conv head 的容量瓶颈**

192ch → 3ch 只有 3 层 conv，无法进行复杂的像素级色调映射。光照不均匀场景（如暗角、局部过曝）需要更大的感受野和非线性变换能力。

---

## 3. 三源解耦分支对问题的评估

### 3.1 三源解耦模型简介

```
Encoder → TCA → [ISPN, NDPN, MCPN] → CXG → SGRF → res_t
              ↑
              三个并行分支：
                ISPN: 光照增强 (gain_map + Zero-DCE曲线)
                NDPN: SNR感知时序去噪 (F_denoised + detail_residual)
                MCPN: 运动对齐 + 去模糊 (G_t 运动门控)
              ↑
              CXG: NDPN↔MCPN 交叉门控
              SGRF: A(提亮) → B(精炼) 两阶段梯度隔离
```

### 3.2 对当前问题的缓解

| 问题 | 三分支缓解机制 | 原理 |
|------|--------------|------|
| motion_map 退化 | NDPN conf_proj (同结构) | 同样退化，不缓解 |
| 暗光运动 | **ISPN** per-pixel gain [0.5,2.0] | 显式亮度补偿，暗区映射独立于运动 |
| 暗光噪声 | **NDPN** SNR 调制 α | 低 SNR 帧自动降权 |
| 运动模糊 | **MCPN** G_t 运动权重 | 强运动帧被抑制 |
| 时序模糊泄漏 | **NDPN** γ≤0.1 + detail_residual | 时序特征贡献上限 10%，编码器直连补充 |
| head 容量瓶颈 | SGRF 两阶段 + StageBlock | 先提亮再精炼，各阶段有独立容量 |

### 3.3 对当前问题的加剧

| 问题 | 三分支加剧机制 | 严重程度 |
|------|--------------|:--:|
| **γ clamp 0.1** | 时序融合贡献被锁死在 10% —— 对暗光场景，多帧平均的降噪收益无法充分释放 | ★★★ |
| **Kendall UW 多 loss** | 15 项辅助损失稀释 L_pix 梯度 → 优化目标偏离 PSNR | ★★★ |
| **SGRF stop_gradient** | StageA→StageB 梯度隔离 → 无法端到端联合优化 → 不同分支之间缺乏反馈 | ★★☆ |
| **CXG 交叉门控** | 额外 ~50K 参数的门控模块，增加了优化难度但边际收益未知 | ★☆☆ |

**核心结论**：三源解耦的**架构思想**（光照/噪声/运动分开处理）正确缓解了简化模型的短板，但 F10m5 的**实现约束**（γ clamp、多 loss、stop_gradient）自缚手脚。实际效果是 **F10m5 ep40=18.27 < v2 ep40=20.02** —— 说明约束的损害超过了分支结构的收益。

---

## 4. 改进方案

### 4.1 简化模型改进 (RWKV-Only 路线)

#### P1: 空间感知 conf_proj (替代当前 Linear 结构) ★★★

当前 conf_proj 是 per-pixel 的 4→1 Linear，无法利用 C_omega 的空间结构。改为：

```python
# 替换 conf_proj: Linear(4,16)→GELU→Linear(16,1)→Sigmoid
# 为空间卷积结构:
self.conf_conv = nn.Sequential(
    nn.Conv2d(num_neighbors, 16, 3, 1, 1),  # 4→16 空间卷积
    nn.GELU(),
    nn.Conv2d(16, 1, 3, 1, 1),
    nn.Sigmoid(),
)

def _build_motion_map(self, C_omega_list, target_size, device):
    # 从 C_omega 对角提取空间图
    diag_maps = []
    for C_t in C_omega_list:
        diag = C_t.diagonal(dim1=-2, dim2=-1)  # (B, N)
        ds = int(diag.shape[-1] ** 0.5)
        diag_maps.append(diag.reshape(B, 1, ds, ds))
    diag_cat = torch.cat(diag_maps, dim=1)  # (B, 4, ds, ds)
    conf_map = self.conf_conv(diag_cat)     # 空间卷积学习
    conf_map = F.interpolate(conf_map, size=target_size, ...)
    return 1.0 - conf_map
```

**预期**：3×3 卷积利用 C_omega 的局部空间结构，能从邻域对角线差异中检测运动边缘，打破全局偏置。

#### P2: 通道级自适应门控 ★★☆

当前 motion_map 是单通道标量，对 64 通道特征施加统一门控。改为通道级：

```python
self.channel_motion_gate = nn.Sequential(
    nn.Conv2d(fused_channels, fused_channels, 3, 1, 1),
    nn.GELU(),
    nn.Conv2d(fused_channels, fused_channels, 1),
    nn.Sigmoid(),
)

# forward
channel_gate = self.channel_motion_gate(l1_gate_feat)  # (B, 64, H, W)
tca_gated = tca_up * (1 - channel_gate) + l1_gate_feat * channel_gate
```

**预期**：不同通道编码不同频率/语义信息，允许低频通道多用 TCA（降噪）、高频通道多用 l1（锐度）。

#### P3: 细节保留残差连接 ★★☆

在 head 输出端增加从输入中心帧的高频注入：

```python
# 残差学习: res_t = head(fused) + α × (input_center - avg_pool(input_center))
hf = image_center - F.avg_pool2d(image_center, 5, 1, 2)
alpha = 0.1  # 或可学习
res_t = head(fused) + alpha * hf
```

**预期**：直接补充被 head 抹平的高频纹理。

#### P4: 多帧权重显式学习 (替代隐式 TCA 融合) ★☆☆

在 l2 层加入显式的 per-frame 加权，与 TCA 输出并行：

```python
frame_weights = self.frame_scorer(l2_lat)  # (B, T, 1, H/2, W/2)
l2_weighted = (l2_lat * frame_weights.softmax(dim=1)).sum(dim=1)
# concat 时加入 l2_weighted 作为额外信息源
fused = concat([l1_feat, tca_gated, l3_up, l2_weighted_up], dim=1)
```

### 4.2 三分支兼容改进

以下方案在概念模型层实现，可直接迁移到 F10m5 的三分支架构而不冲突。

#### T1: 移除 γ clamp，改为可学习的 soft-gate ★★★

```python
# NDPN 中
self.gamma_raw = nn.Parameter(torch.full((1, channels, 1, 1), 0.01))
# 删除: gamma = gamma_raw.clamp(max=0.1)
# 改为: gamma = gamma_raw.sigmoid()  # (0, 1) 光滑可微
```

**预期**：模型自行学习最优的时序融合比例，不再被人为限制。

#### T2: 用 motion_map 调制 NDPN 的 temporal_base ★★☆

当前 NDPN 的 `F_denoised` 对所有空间位置等权融合。加入 motion_map 调制：

```python
# motion_map 从 C_omega 计算（概念模型中已有）
F_denoised = F_denoised * (1 - motion_map) + f_enc_center * motion_map
```

**预期**：NDPN 在运动区域自动回退到单帧降噪。

#### T3: 简化损失函数 ★★☆

```python
# 替换 Kendall UW 15 项 loss 为:
loss = L1(res_t, gt) + 0.2 * (1 - SSIM(res_t, gt))
     + 0.01 * L1(gain_map, 1.0)      # ISPN 正则
     + 0.01 * gamma.pow(2).mean()    # NDPN 正则
```

**预期**：主梯度方向对准 PSNR，辅助损失仅做弱正则。

#### T4: 移除 SGRF stop_gradient ★☆☆

```python
# SGRF Stage A 输出的 img_lit 不再 detach
img_lit = self.brighten_stage(gain_map, image_center)  # 无 .detach()
```

**预期**：Stage B 的梯度能回传修正 Stage A，实现真正的端到端优化。

---

## 5. 实验优先级

| 优先级 | 方案 | 改动量 | 预期收益 | 能否直接验证 |
|:--:|------|:--:|:--:|:--:|
| 1 | P1 空间 conf_proj | ~1K 参数 | 打破 motion_map 全局偏置 | ✓ 单次训练 |
| 2 | P2 通道级门控 | ~50K 参数 | 特征级自适应 | ✓ 单次训练 |
| 3 | P3 残差连接 | ~0 参数 | 高频细节保留 | ✓ 单次训练 |
| 4 | T1 γ clamp 解除 | 0 参数 (需三分支) | 释放时序融合上限 | 需完整 F10m5 |
| 5 | T3 损失简化 | 0 参数 (需三分支) | 梯度对准 PSNR | 需完整 F10m5 |
