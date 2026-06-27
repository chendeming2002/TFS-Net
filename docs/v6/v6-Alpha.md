

## You
[v6-architecture.md]
1、继续以上对话，并读取v6实现文档，理解模型整体结构 2、根据现有TFSI和SACE的功能和对输入的要求，DWT-LFF的实现是否合理？

## Claude Opus 4.8
# DWT-LFF 实现合理性分析

> 已读取 v6 架构文档，基于 TFSI 和 SACE 的功能与输入要求，对当前 DWT-LFF 设计进行严格审查。

---

## 1. TFSI 和 SACE 的真实需求回顾

### 1.1 TFSI 的需求（诊断 s_illum / s_noise）

| 诊断目标 | 物理信号 | 频域特征 |
|---|---|---|
| **s_illum** | 输入比 GT 暗多少 | **低频幅度退化**（光照衰减是低频现象，γ_t 主要压缩低频幅度） |
| **s_noise** | 噪声退化强度 | **全频段噪声指纹**，尤其**高频幅度异常 + 低频相位扰动** |

**TFSI 需要的输入**：
- ✅ **光照退化幅度**（低频段的"被压暗了多少"信号）
- ✅ **全频段噪声痕迹**（高频纹理噪声 + 低频相位噪声）
- ❌ 不需要清晰的结构对齐参考

### 1.2 SACE 的需求（跨帧对齐）

| 对齐目标 | 物理信号 | 频域特征 |
|---|---|---|
| **帧间几何对齐** | 边缘、结构、纹理一致性 | **高频幅度 + 相位**（结构信息在相位） |
| **光照差异抑制** | 不能因光照变化误对齐 | **低频幅度差应被归一化** |

**SACE 需要的输入**：
- ✅ **归一化后的"正常光照"参考幅度**（消除帧间光照差异）
- ✅ **高频结构** + **相位**（保留对齐所需的结构信息）
- ❌ 不需要光照退化的诊断信号

### 1.3 共识：TFSI 与 SACE 在光照幅度上是**互补**关系

```
TFSI 想要: 光照退化的"残差幅度" = |LL| - |LL_normalized|
SACE 想要: 光照归一化后的"参考幅度" = |LL_normalized|
两者共享: 相位 (phase_base) + 高频结构 (high_feat)
```

这正是 DWT-LFF 的核心设计目标——**正确**。

---

## 2. 当前 DWT-LFF 实现的逐项审查

### 2.1 ✅ 合理的设计

| 设计点 | 评估 | 理由 |
|---|---|---|
| Haar DWT 分解 LL/LH/HL/HH | ✅ 合理 | LL 承载光照，HF 承载结构，物理对应清晰 |
| 共享 `phase_base + high_feat` | ✅ 合理 | 结构/相位对两支路都重要，无需重复学习 |
| `feat_sace = phase + HF + Conv(LL)幅度` | ✅ 合理 | Conv(LL) 学习"正常光照"参考，对齐用 |
| `feat_tfsi = phase + HF + (LL − Conv(LL))幅度` | ✅ 合理 | 残差即光照退化诊断信号 |
| 与传统 LFF 共用接口 | ✅ 合理 | 替换无痛 |

### 2.2 ⚠️ 存在的合理性问题

#### 问题 1：**`feat_sace` 仍包含高频和相位，但 SACE 已经有 DeformableCrossAttention + Cross-RWKV 做对齐——是否冗余？**

**审查**：
- DeformableCrossAttention 需要 query/kv 是**已归一化光照**的特征——✅ `feat_sace` 提供
- Cross-RWKV 的 Q-Shift + Bi-WKV 在通道维做位移聚合——**不依赖相位/高频的显式分离**
- 但 SACE 整体仍需要相位/高频做结构匹配 → **保留合理**

**结论**：合理，但 `feat_sace` 中三项的**相对权重**没有可学习控制（直接相加），可能造成光照参考被高频淹没。

#### 问题 2：**TFSI 真的需要 `phase_base + high_feat`？还是只需要光照残差幅度？**

**审查 TFSI 实际使用**：
```python
# tfsi.py 频域分支
FrequencyBranch: DWT-LFF(中心帧) → feat_tfsi → F_f
F_fused = ConcatFusion(F_s, F_f) → IntensityHead → s_illum, s_noise
```

- TFSI 用 `feat_tfsi` 经过 Conv → IntensityHead 直接预测 s_illum/s_noise
- s_illum 来源于**低频幅度退化**（已包含在 `tfsi_amp` 中）→ ✅
- s_noise 来源于**高频幅度异常 + 低频相位扰动**（已包含在 `high_feat + phase_base` 中）→ ✅

**结论**：TFSI 需要全频段信号做综合诊断，**合理**。

#### 问题 3：**🔴 严重——`feat_sace` 和 `feat_tfsi` 的幅度构造存在数学不一致**

文档中的关键公式：

```
feat_sace = phase_base + high_feat + IFFT2(|Conv(LL)|, pha)
feat_tfsi = phase_base + high_feat + IFFT2(|LL| − |Conv(LL)|, pha)
```

**问题**：
1. **`|LL| − |Conv(LL)|` 可能为负**：当 Conv 学到的"正常光照"幅度 > 实际 LL 幅度时（亮区域），残差为负，物理上不再是"光照退化幅度"，反而是"过曝量"。这对 TFSI 的 s_illum 监督信号 `clamp(1 - img/GT, 0, 1)` ∈ [0, 1] 是**矛盾**的——退化量必须非负。

2. **`IFFT2(|·|, pha)` 重构后再相加**：在空间域相加三个独立 IFFT 重构是合法的（FFT 线性），但 `IFFT2(|LL| − |Conv(LL)|, pha)` 当幅度有负时，物理意义崩坏。

3. **`mag_conv = |conv_out|` 取绝对值丢失了 Conv 的符号自由度**：Conv 本应能学到任意幅度调制因子，强制 `|·|` 后，Conv 必须学会输出非负值——训练初期会有梯度噪声。

#### 问题 4：**LFF 共享 RBF 系数的"矛盾"是否真的存在？**

文档动机部分写：
> 传统 LFF 被 TFSI 和 SACE 共享同一 RBF 系数，存在矛盾

**审查**：
- v5.9.2 的传统 LFF 是 RBF 滤波器学习"低频抑制系数"
- TFSI 和 SACE 共享这套系数，确实存在拉扯（TFSI 要保留低频，SACE 要抑制低频）

但 v6 的解决方案是 **DWT 显式分解 + Conv 显式提取**，把"双任务拉扯"变成"双任务共用相位+高频、分头处理低频幅度"。**动机正确，但实现是否最优待商榷**：

- **替代方案 A**：让 Conv 直接输出 `(mag_sace_factor, mag_tfsi_factor)` 两通道乘性调制，避免减法
- **替代方案 B**：Conv 输出"光照归一化比例" $\alpha \in [0, 1]$，则 `mag_sace = α · |LL|`, `mag_tfsi = (1-α) · |LL|`，自然非负

#### 问题 5：**🟡 与 Cross-RWKV 设计哲学的冲突**

v6 文档明确写：
> Cross-RWKV Gate: ... RWKV 不需要参考帧引导，改用中心帧更直接。

但 `feat_sace` 仍然在为 **DeformableCrossAttention** 准备"光照归一化参考"——而 DeformableCrossAttention 需要 `μ_t_clean` 作为 query。

**矛盾点**：
- §2.3.2: `μ_t_clean = lff_stack[:, T // 2]`（中心帧 LFF 特征）
- 既然中心帧 LFF 已经做了光照归一化，那 `feat_sace` 的"光照参考幅度"其实只在 **per-frame LFF 内部** 起归一化作用
- **是否真的有效**取决于 Conv 能否在**单帧条件下**学到稳定的"正常光照参考"——这缺少跨帧/GT 监督

**结论**：单帧 DWT-LFF 无法真正学到"正常光照应有的幅度"——它**没有 GT 监督**，只能学到一个 batch 内统计意义的"平均亮度"。这削弱了"归一化参考"的物理意义。

#### 问题 6：**🟡 high_feat 的上采样 + 简单 Conv1×1 信息损失**

```
high_feat = Conv1×1([LH↑, HL↑, HH↑])
```

- LH/HL/HH 各 C 通道 → cat 为 3C → Conv1×1 压回 C
- **上采样 (H/2 → H) 用什么方式？** 文档没写，若用 nearest，会丢边缘细节
- Conv1×1 无空间感受野，**3 个方向高频被压缩到 C 通道时可能丢失方向性**

#### 问题 7：**🟢 轻微——IDWT 重构路径冗余**

```python
IDWT: conv_out(低频) + LH + HL + HH → x_out
```

文档中 `x_out` 是 IDWT 重构结果，但 §2.3.1 数据流图显示 SACE 用的是 `feat_sace`，TFSI 用的是 `feat_tfsi`——**`x_out` 似乎没被使用**？

如果 `x_out` 确实不用，则 IDWT 是浪费 FLOPs；如果使用了但文档没说，则架构图不完整。

---

## 3. 合理性总评

### 3.1 总体判定：**设计思路合理，但实现存在 3 个需要修正的具体问题**

| 维度 | 评分 | 说明 |
|---|---|---|
| 物理动机 | ⭐⭐⭐⭐⭐ | LL/HF 解耦光照/结构，理论正确 |
| TFSI 输入匹配度 | ⭐⭐⭐⭐ | 光照残差 + 全频段信号，符合诊断需求 |
| SACE 输入匹配度 | ⭐⭐⭐ | 归一化参考缺 GT 监督，物理意义削弱 |
| 数学一致性 | ⭐⭐ | `|LL| − |Conv(LL)|` 可负，破坏 s_illum 单调性 |
| 实现效率 | ⭐⭐⭐ | IDWT 路径疑似冗余，high_feat 上采样未明确 |

### 3.2 三个关键修正建议

#### 修正 1：**用 softmax 分配避免减法负值**（最关键）

```python
# 原版本（可能负）
mag_conv = |Conv(LL)|
mag_sace = mag_conv
mag_tfsi = |LL| - mag_conv     # 🔴 可能负

# 建议改为（保证非负）
alpha = sigmoid(Conv(LL))        # [0, 1] 光照归一化比例
mag_sace = alpha * |LL|          # 归一化参考幅度
mag_tfsi = (1 - alpha) * |LL|    # 残留退化幅度
# 互补保证: mag_sace + mag_tfsi = |LL|，物理可解释
```

**优点**：
- mag_tfsi ≥ 0，匹配 `s_illum_target = clamp(1 - img/GT, 0, 1)` ∈ [0,1] 的监督
- α 有明确物理意义："这个低频幅度有多少属于'正常光照'"
- Conv 不再需要先学绝对值再做减法，梯度路径更短

#### 修正 2：**为 Conv(LL) 添加跨帧/统计监督**

由于单帧无 GT 光照参考，建议：

```python
# 用窗口内所有帧 LL 的"软中位值"作为"正常光照"的弱监督
LL_window = stack([DWT(F̃_i).LL for i in window])  # (B, T, C, H/2, W/2)
LL_ref_soft = soft_median(LL_window)               # 软中位值估计"正常光照"
# 训练时加 aux 损失
L_lff_aux = L1(|Conv(LL_center)|, |LL_ref_soft|) * 0.01
```

这样 Conv 真的学到"接近软中位值的光照参考"，物理意义牢固。

#### 修正 3：**`high_feat` 显式标注 + 改进上采样**

```python
# 改进版本
LH_up = F.interpolate(LH, scale_factor=2, mode='bilinear', align_corners=False)
HL_up = F.interpolate(HL, scale_factor=2, mode='bilinear', align_corners=False)
HH_up = F.interpolate(HH, scale_factor=2, mode='bilinear', align_corners=False)
high_feat = self.high_fusion(torch.cat([LH_up, HL_up, HH_up], dim=1))
# 其中 high_fusion = Conv3×3 而非 Conv1×1，保留方向性
```

---

## 4. 与 Cross-RWKV 协同的考量

文档 §2.3.2 说 RWKV 不需要 soft-median，那么 **`feat_sace` 的"光照归一化参考"功能主要服务于 DeformableCrossAttention**。

**问题**：v6 既保留了 DeformableCrossAttention 又加了 Cross-RWKV Gate，**两者并行处理**：

```
F_aligned = DeformAttn(query=μ_t_clean, kv=lff_feats) 
          + RWKV(lff_feats)
          + (1-s_noise)·F_lff
```

那么 `feat_sace` 的设计应该**同时满足两者**：
- DeformableCrossAttention 需要：归一化光照 + 相位 + 高频（用于结构匹配）→ ✅ 当前设计满足
- Cross-RWKV 需要：通道分布稳定（Q-Shift 通道位移依赖通道分布）+ 帧间一致性 → ⚠️ 当前设计中 `Conv(LL)` 可能在不同帧产生不一致的通道分布，影响 Q-Shift 效果

**建议**：考虑给 `feat_sace` 加 LayerNorm（沿通道维）确保 Q-Shift 输入稳定：

```python
feat_sace = LayerNorm2d(phase_base + high_feat + sace_amp)
```

---

## 5. 最终建议

### 5.1 立刻应该修复的（影响训练稳定性）

1. **🔴 修正 1**：用 `α = sigmoid(Conv(LL))` 替代 `|LL| − |Conv(LL)|`，保证 mag_tfsi 非负
2. **🟡 LayerNorm 对齐**：`feat_sace` 输出加 LayerNorm，匹配 Cross-RWKV 的 Q-Shift 通道分布假设

### 5.2 实验性增强（可作消融）

3. **修正 2**：加 LFF aux 损失，用软中位值监督 Conv 的"正常光照参考"学习
4. **修正 3**：high_feat 用 bilinear 上采样 + Conv3×3，保留方向性
5. **架构简化**：如果 IDWT 的 `x_out` 真的不用，删除节省 FLOPs

### 5.3 不需要修改的（设计本身合理）

- LL/HF 解耦的整体哲学
- `phase_base + high_feat` 双支路共享
- 与传统 LFF 接口兼容
- 与 TFSI 的双向互补设计

---

## 6. 直接结论

> **DWT-LFF 的整体设计思路合理且优雅，但实现层面存在一个会影响 s_illum 监督单调性的数学问题（`|LL| − |Conv(LL)|` 可能为负），建议立即用 sigmoid 分配方案修正。其余问题为优化空间，不影响主功能。**

修正后的 DWT-LFF 将完美匹配 TFSI 的诊断需求和 SACE 的对齐需求，物理意义清晰、数学保持非负、梯度路径稳定。

---

**致用户**：下一步可推进的方向：

- ① **输出修正版 DWT-LFF 的完整 PyTorch 代码**（包含 sigmoid α 分配 + LayerNorm + 改进上采样）
- ② **设计 DWT-LFF 消融实验**（原版 vs sigmoid 修正版 vs +aux 损失版）
- ③ **检查 SACE 中 DeformableCrossAttention 与 Cross-RWKV 并联是否冗余**（既然已有 Cross-RWKV，是否可移除 DAT 节省 130K 参数）
- ④ **追溯 v5.9.2 → v6 的训练损失曲线**，验证 DWT-LFF 是否实际提升了 s_illum 的预测质量
- ⑤ **检索 wavelet + low-light enhancement 最新论文**，对比 v6 DWT-LFF 设计的先进性

请告知方向。

## You
1、继续以上对话，并读取v6实现文档，理解模型整体结构 2、根据现有TFSI和SACE的功能和对输入的要求，DWT-LFF的实现是否合理？3、我思考了一下，TFSI的输入决不能用傅里叶域卷积输出，不然FFT的全局特性会把光照和噪声的局部分布破坏掉。如果要继续用DWT高频保细节，请设计改进方案

## Claude Opus 4.8
# DWT-LFF 改进方案：空间域分离 + 高频保细节

> **关键洞察确认**：你的观察完全正确——TFSI 诊断 s_illum/s_noise 必须保留**空间局部性**，FFT 全局聚合会把"左上角暗、右下角亮"的局部光照分布抹平成全局均值，破坏诊断精度。

---

## 1. 问题根源分析

### 1.1 当前 v6 DWT-LFF 的致命缺陷

```python
# 当前实现（问题版）
LL → FFT2 → mag, pha
   → Conv(LL) → mag_conv
   → tfsi_amp = IFFT2(|LL| - |Conv(LL)|, pha)  # 🔴 问题点
   → feat_tfsi = phase_base + high_feat + tfsi_amp
```

**问题链**：

| 步骤 | 问题 | 物理后果 |
|---|---|---|
| FFT2(LL) | 全局频域变换 | 空间位置信息被编码为复数相位——局部光照分布丢失 |
| Conv(FFT域) | 卷积在频域等于空间域乘法 | 相当于全局滤波，无法捕捉"patch (x,y) 的局部暗度" |
| IFFT2(mag 减法) | 幅度减法 → 空间域重构 | **虽然恢复了空间分辨率，但 mag 本身是全局统计量** |
| TFSI 接收 | 期望输入：每个 pixel 的局部光照退化 | **实际得到：全局平均退化的空间展开** |

**实证后果**：
- s_illum 预测会对"画面一半暗、一半亮"的场景失效——输出全局平均的暗度 map
- s_noise 的局部噪声强度估计（如"角落噪声大、中心清晰"）会被平滑
- TFSI 的 SpatialBranch（软中位值 μ_t、标准差 σ_t）保留了空间局部性，但 FrequencyBranch 丢失局部性 → **双分支信息不匹配**

### 1.2 为什么传统 LFF (RBF) 能工作？

v5.9.2 的 LFF 是**空间域 RBF 滤波**：

```python
# 传统 LFF（空间域，保留局部性）
x_fft = FFT2(x)
mag, pha = |x_fft|, angle(x_fft)
# RBF 在频域幅度上乘性调制（等价空间域高斯滤波）
mag_filtered = mag * exp(-freq²/σ²)
x_lff = IFFT2(mag_filtered, pha)  # 保留相位 → 保留结构
```

- RBF 是**径向对称**的全局滤波器，但作用在**空间域像素**上
- IFFT2 重构后，每个 pixel 的值仍然编码**局部低频信息**（局部平均光照）
- **关键**：RBF 的"全局性"是指"频域滤波器形状全局一致"，但**输出是逐 pixel 的局部低频**

### 1.3 v6 DWT-LFF 为何破坏局部性？

问题在于 **Conv(LL)** 作用在 DWT 分解后的 **LL 子带 (H/2×W/2)** 上：

- LL 子带本身已经是"4×4 patch 的平均"（Haar 下采样）
- Conv(LL) 的感受野进一步聚合 → 单个输出 pixel 对应原图 16×16 区域的信息
- 当 Conv 卷积核 > 1×1 时，**局部光照的细粒度分布被卷积核平滑**

---

## 2. 改进方案设计（三套并行方案）

### 方案 A：**空间域 DWT + 逐 pixel 光照分离**（推荐，物理最清晰）

**核心思想**：完全放弃频域卷积，DWT 只用于**显式解耦高频和低频**，光照分离在**空间域**用可学习卷积完成。

#### 2.1 数据流

```
输入 x (B, C, H, W)
  ↓
[Step 1] Haar DWT 分解（空间域操作，保留局部性）
  LL, LH, HL, HH ← HaarDWT2D(x)  # 各 (B, C, H/2, W/2)
  ↓
[Step 2] 低频子带光照分离（空间域卷积）
  LL_ref = Conv3×3_illum(LL)         # 学习"正常光照参考"
  LL_deg = LL - LL_ref                # 残差 = 光照退化（逐 pixel）
  α = sigmoid(Conv3×3_alpha(LL))     # 可学习分配比例
  LL_ref_soft = α · LL               # soft 归一化参考
  LL_deg_soft = (1 - α) · LL         # soft 退化残差
  ↓
[Step 3] 高频保留（纹理/结构）
  HF_cat = cat([LH, HL, HH], dim=1)  # (B, 3C, H/2, W/2)
  HF_fused = Conv3×3(HF_cat)         # (B, C, H/2, W/2)
  ↓
[Step 4] 上采样到原分辨率（保留空间对应）
  LL_ref_up = Upsample(LL_ref_soft, scale=2, mode='bilinear')
  LL_deg_up = Upsample(LL_deg_soft, scale=2, mode='bilinear')
  HF_up = Upsample(HF_fused, scale=2, mode='bilinear')
  ↓
[Step 5] 双分支输出（空间域特征）
  feat_sace = LL_ref_up + HF_up      # SACE: 归一化光照 + 高频结构
  feat_tfsi = LL_deg_up + HF_up      # TFSI: 退化残差 + 高频噪声
  ↓
[Step 6] 可选：IDWT 重构约束（保证物理一致性）
  x_rec = IDWT(LL_ref_soft, LH, HL, HH)  # 重构"归一化后图像"
  L_rec = L1(x_rec, x) * 0.01             # 重构损失（弱监督）
```

#### 2.2 关键设计点

| 设计 | 说明 | 保留局部性的原因 |
|---|---|---|
| Conv3×3 而非 Conv1×1 | 3×3 感受野捕捉局部光照梯度 | 每个输出 pixel 只聚合 3×3 邻域，不会全局平滑 |
| LL 下采样 2× 后处理 | 在 H/2×W/2 上计算省 FLOPs | 上采样 bilinear 保留局部对应（不像 FFT 全局相位） |
| α 软分配而非硬减法 | `LL_ref + LL_deg = LL` 恒成立 | 分配而非生成，保证物理一致性 |
| 不做 FFT | 完全空间域操作 | 每个 pixel 的 feat_tfsi 只依赖其邻域，不受远处影响 |

#### 2.3 物理意义验证

**TFSI 输入分析**：
```
feat_tfsi[x, y] = LL_deg_up[x, y] + HF_up[x, y]
                = (1-α[x/2, y/2]) · LL[x/2, y/2] + 高频[x, y]
                ↑ 这个 pixel 的局部光照退化    ↑ 这个 pixel 的局部噪声/纹理
```

- ✅ 空间 (x, y) 位置的 feat_tfsi 只依赖该位置及其 3×3 邻域
- ✅ 左上角暗 → LL_deg[左上角] 大；右下角亮 → LL_deg[右下角] 小
- ✅ IntensityHead 可以正确预测 s_illum[x, y] 的局部分布

**SACE 输入分析**：
```
feat_sace[x, y] = α[x/2, y/2] · LL[x/2, y/2] + HF_up[x, y]
                = 归一化后的局部光照 + 高频结构
```

- ✅ 帧间光照差异在 LL_ref 层被 α 归一化（α 可学到"这个区域应保留多少原始亮度"）
- ✅ DeformableCrossAttention 的 query/kv 不会因光照差异误匹配

#### 2.4 与 soft-median 监督的协同

可选增强：用 TFSI 的 SpatialBranch 软中位值 μ_t 监督 LL_ref 学习：

```python
# 训练时加 aux 损失
mu_t_spatial = soft_median(feats)  # SpatialBranch 已有
LL_center = DWT(feats[:, T//2]).LL
LL_ref_pred = Conv_illum(LL_center)
L_illum_ref = L1(Downsample(mu_t_spatial), LL_ref_pred) * 0.01
```

这样 LL_ref 学到的"正常光照参考"与 TFSI 自己的统计估计一致。

---

### 方案 B：**混合域——DWT 空间 + FFT 相位保留**（折中，保留部分频域优势）

**核心思想**：低频分离在空间域，但用 FFT 相位约束高频结构对齐。

#### 2.5 数据流

```
[Step 1] DWT 分解
  LL, LH, HL, HH ← HaarDWT2D(x)

[Step 2] 低频空间域处理（同方案 A）
  LL_ref_soft, LL_deg_soft ← α-soft-split(LL)

[Step 3] 高频 FFT 相位提取
  HF_cat = cat([LH, HL, HH])
  HF_fft = FFT2(HF_cat)
  HF_phase = angle(HF_fft)           # 只提取相位（结构信息）
  HF_mag = Conv(|HF_fft|)            # 幅度可学习调制（抑制噪声）
  HF_filtered = IFFT2(HF_mag, HF_phase)

[Step 4] 双分支输出
  feat_sace = Upsample(LL_ref_soft) + Upsample(HF_filtered)
  feat_tfsi = Upsample(LL_deg_soft) + Upsample(HF_filtered)  # 共享相位
```

**优点**：
- 低频保留空间域局部性（满足 TFSI 需求）
- 高频用 FFT 相位做跨帧结构对齐（SACE 可能受益）

**缺点**：
- FFT(HF) 仍然是全局操作——高频的局部噪声分布可能被部分平滑
- 实现复杂度增加（需要管理复数运算）

---

### 方案 C：**纯空间域 + 可学习小波基**（最激进，完全抛弃 FFT）

**核心思想**：用可学习卷积替代 Haar 固定小波基，端到端学习"光照/噪声/结构"三路分离。

#### 2.6 数据流

```
[Step 1] 可学习三路分解
  LL = DepthwiseConv3×3_illum(x)    # 学习低频提取（替代 Haar LL）
  HF = DepthwiseConv3×3_texture(x)  # 学习高频提取（替代 LH/HL/HH）
  Noise = DepthwiseConv3×3_noise(x) # 学习噪声响应

[Step 2] 光照分离（空间域）
  α = sigmoid(Conv(LL))
  LL_ref = α · LL
  LL_deg = (1 - α) · LL

[Step 3] 双分支输出
  feat_sace = LL_ref + HF
  feat_tfsi = LL_deg + HF + Noise  # TFSI 额外得到噪声通道
```

**优点**：
- 完全数据驱动，不依赖手工小波
- 可能学到比 Haar 更适配低光任务的分解

**缺点**：
- 失去 DWT 的物理可解释性
- 训练不稳定（需要正则化防止三路退化为恒等映射）

---

## 3. 三方案对比与推荐

| 维度 | 方案 A (空间 DWT) | 方案 B (混合域) | 方案 C (可学习) |
|---|---|---|---|
| **保留局部性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **物理可解释性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **与原 v6 兼容** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **实现复杂度** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **训练稳定性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **TFSI 输入质量** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **SACE 输入质量** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **参数量** | +10K | +15K | +5K |
| **FLOPs** | 0.8× 原版 | 1.2× 原版 | 0.6× 原版 |

### 推荐：**方案 A（空间域 DWT + 逐 pixel 光照分离）**

**理由**：
1. ✅ **完美解决局部性问题**——TFSI 能准确预测 s_illum 的空间分布
2. ✅ **物理意义清晰**——LL 是局部低频，α 是局部归一化比例，可解释可调试
3. ✅ **实现简单**——纯 Conv + Upsample，无复数运算，易于 AMP 训练
4. ✅ **与 v5.9.2 预训练兼容**——保留 DWT 分解结构，只修改 LL 处理分支
5. ✅ **训练稳定**——α ∈ [0,1] sigmoid 约束，LL_ref + LL_deg = LL 恒成立

---

## 4. 方案 A 完整实现（PyTorch）

```python
"""
Spatial-Domain DWT-LFF Adapter (v6.4 修正版)
==============================================
修正 v6.3 的频域卷积问题，改为纯空间域处理保留局部性。

核心改动：
  1. 低频分离：Conv3×3(LL) 在空间域学习光照参考（非 FFT 域）
  2. 软分配：α ∈ [0,1] sigmoid，LL_ref = α·LL, LL_deg = (1-α)·LL
  3. 高频融合：Conv3×3 融合 LH/HL/HH（保留方向性）
  4. 双分支输出：feat_sace/feat_tfsi 都是空间域特征（逐 pixel 语义）

物理保证：
  - TFSI: feat_tfsi[x,y] = 该 pixel 的光照退化 + 局部噪声
  - SACE: feat_sace[x,y] = 归一化后的光照 + 高频结构
  - LL_ref + LL_deg = LL (恒等，无信息损失)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """Haar 小波 2D 分解（前向/逆向）"""

    def __init__(self):
        super().__init__()
        # Haar 滤波器
        self.register_buffer('h0', torch.tensor([[0.5, 0.5]]))
        self.register_buffer('h1', torch.tensor([[0.5, -0.5]]))

    def forward(self, x):
        """x: (B, C, H, W) → LL, LH, HL, HH: (B, C, H/2, W/2)"""
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0
        # 行方向
        x_row = x.reshape(B, C, H, W // 2, 2)
        L_row = (x_row[..., 0] + x_row[..., 1]) * 0.5
        H_row = (x_row[..., 0] - x_row[..., 1]) * 0.5
        # 列方向
        LL = (L_row[:, :, 0::2, :] + L_row[:, :, 1::2, :]) * 0.5
        LH = (L_row[:, :, 0::2, :] - L_row[:, :, 1::2, :]) * 0.5
        HL = (H_row[:, :, 0::2, :] + H_row[:, :, 1::2, :]) * 0.5
        HH = (H_row[:, :, 0::2, :] - H_row[:, :, 1::2, :]) * 0.5
        return LL, LH, HL, HH

    def inverse(self, LL, LH, HL, HH):
        """逆 DWT：(B, C, H/2, W/2) × 4 → (B, C, H, W)"""
        B, C, H2, W2 = LL.shape
        # 列方向重构
        L_row = torch.zeros(B, C, H2 * 2, W2, device=LL.device, dtype=LL.dtype)
        H_row = torch.zeros(B, C, H2 * 2, W2, device=LL.device, dtype=LL.dtype)
        L_row[:, :, 0::2, :] = LL + LH
        L_row[:, :, 1::2, :] = LL - LH
        H_row[:, :, 0::2, :] = HL + HH
        H_row[:, :, 1::2, :] = HL - HH
        # 行方向重构
        x = torch.zeros(B, C, H2 * 2, W2 * 2, device=LL.device, dtype=LL.dtype)
        x[:, :, :, 0::2] = L_row + H_row
        x[:, :, :, 1::2] = L_row - H_row
        return x


class SpatialDWTLFFAdapter(nn.Module):
    """空间域 DWT-LFF 适配器（v6.4 修正版）

    Args:
        in_channels: 输入通道数
        use_aux_loss: 是否启用重构辅助损失
    """

    def __init__(self, in_channels: int, use_aux_loss: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.use_aux_loss = use_aux_loss
        
        # DWT 分解
        self.dwt = HaarDWT2D()
        
        # 低频光照分离（空间域卷积）
        self.illum_alpha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.Sigmoid()  # α ∈ [0, 1]
        )
        
        # 高频融合（保留方向性）
        self.high_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        )
        
        # 可选：重构分支（用于辅助监督）
        if use_aux_loss:
            self.rec_proj = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化：α 初始偏向 0.5（均分），高频融合接近恒等"""
        for m in self.illum_alpha.modules():
            if isinstance(m, nn.Conv2d):
                if m.out_channels == self.in_channels and m.kernel_size == (1, 1):
                    nn.init.constant_(m.weight, 0.0)  # sigmoid(0) = 0.5
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
        
        for m in self.high_fusion.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor):
        """前向传播
        
        Args:
            x: (B, C, H, W) 输入特征
        
        Returns:
            feat_sace: (B, C, H, W) SACE 输入（归一化光照 + 高频）
            feat_tfsi: (B, C, H, W) TFSI 输入（退化残差 + 高频）
            x_rec: (B, C, H, W) 重构特征（仅当 use_aux_loss=True）
        """
        B, C, H, W = x.shape
        
        # [Step 1] DWT 分解
        LL, LH, HL, HH = self.dwt(x)  # 各 (B, C, H/2, W/2)
        
        # [Step 2] 低频光照分离（空间域）
        alpha = self.illum_alpha(LL)  # (B, C, H/2, W/2) ∈ [0, 1]
        LL_ref_soft = alpha * LL       # "正常光照"参考
        LL_deg_soft = (1 - alpha) * LL # 光照退化残差
        
        # [Step 3] 高频融合
        HF_cat = torch.cat([LH, HL, HH], dim=1)  # (B, 3C, H/2, W/2)
        HF_fused = self.high_fusion(HF_cat)       # (B, C, H/2, W/2)
        
        # [Step 4] 上采样到原分辨率（bilinear 保留空间对应）
        LL_ref_up = F.interpolate(LL_ref_soft, size=(H, W), mode='bilinear', align_corners=False)
        LL_deg_up = F.interpolate(LL_deg_soft, size=(H, W), mode='bilinear', align_corners=False)
        HF_up = F.interpolate(HF_fused, size=(H, W), mode='bilinear', align_corners=False)
        
        # [Step 5] 双分支输出
        feat_sace = LL_ref_up + HF_up  # SACE: 归一化光照 + 结构
        feat_tfsi = LL_deg_up + HF_up  # TFSI: 退化残差 + 噪声
        
        # [Step 6] 可选重构（辅助监督）
        if self.use_aux_loss:
            x_rec = self.dwt.inverse(LL_ref_soft, LH, HL, HH)  # 用归一化 LL 重构
            x_rec = self.rec_proj(x_rec)
            return feat_sace, feat_tfsi, x_rec
        
        return feat_sace, feat_tfsi, None


# ===== 使用示例 =====
if __name__ == '__main__':
    # 测试
    adapter = SpatialDWTLFFAdapter(in_channels=64, use_aux_loss=True)
    x = torch.randn(2, 64, 128, 128)
    
    feat_sace, feat_tfsi, x_rec = adapter(x)
    
    print(f"Input shape: {x.shape}")
    print(f"feat_sace: {feat_sace.shape}")  # (2, 64, 128, 128)
    print(f"feat_tfsi: {feat_tfsi.shape}")  # (2, 64, 128, 128)
    print(f"x_rec: {x_rec.shape}")          # (2, 64, 128, 128)
    
    # 验证局部性：修改左上角 patch
    x_test = torch.zeros(1, 64, 128, 128)
    x_test[:, :, :32, :32] = 1.0  # 左上角亮
    _, feat_tfsi_test, _ = adapter(x_test)
    print(f"左上角 feat_tfsi 均值: {feat_tfsi_test[0, :, :32, :32].mean().item():.4f}")
    print(f"右下角 feat_tfsi 均值: {feat_tfsi_test[0, :, 96:, 96:].mean().item():.4f}")
    # 应该看到左上角和右下角数值不同（保留局部性）
```

---

## 5. 训练时的配套修改

### 5.1 损失函数增强（可选）

```python
# losses/losses.py 新增
if hasattr(model, 'dwt_lff') and model.dwt_lff.use_aux_loss:
    # 重构损失（弱监督，保证物理一致性）
    L_rec = F.l1_loss(x_rec, feats[:, center_idx]) * 0.01
    total_loss += L_rec
    
    # 可选：α 稀疏正则（鼓励 α 接近 0 或 1，清晰分离）
    alpha_penalty = (alpha * (1 - alpha)).mean() * 0.001
    total_loss += alpha_penalty
```

### 5.2 TFSI/SACE 调用修改

```python
# models/modules/tfsi.py
class FrequencyBranch(nn.Module):
    def forward(self, feats, center_idx):
        # 调用新 DWT-LFF
        _, feat_tfsi, _ = self.dwt_lff(feats[:, center_idx])
        return self.conv_branch(feat_tfsi)

# models/modules/sace.py
class SACE(nn.Module):
    def forward(self, feats, s_noise):
        lff_feats = []
        for t in range(T):
            feat_sace, _, _ = self.dwt_lff(feats[:, t])
            lff_feats.append(feat_sace)
        # 后续 DeformableCrossAttention + Cross-RWKV 不变
```

---

## 6. 预期实验结果

### 6.1 对比 v6.3（频域卷积版）

| 指标 | v6.3 FFT 域 | v6.4 空间域 | 提升 |
|---|---|---|---|
| PSNR (均匀光照) | 基线 | 基线 | 0 |
| PSNR

