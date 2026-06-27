# TFS-Net v6 模型架构设计文档

> 日期：2026-06-27
> 版本：v6.4（空间域 DWT-LFF + Cross-RWKV Gate）
> 训练配置：`configs/v6_rwkv.yaml`，batch=2，30 epoch，参数量 1.169M
> 预训练权重：v5.9.2 latest.pth（fuse 修复 + s_illum 复生 + IFPN 监督）
> 设计参考：`v6-Alpha.md`（空间域修正论证）、`v6-Alpha-corr.md`（IDWT 修正论证）

---

## 1. 概述

v6 在 v5.9.2 基础上引入四个核心改造：

| 改造 | 动机 | 来源 |
|---|---|---|
| **Cross-RWKV Gate** | 将 SACE 帧间注意力从 DAT 风格可变形采样升级为 RWKV 线性 Bi-WKV | Vision-RWKV (ICLR 2025) |
| **空间域 DWT-LFF** | 用小波分离低频（光照/噪声）和高频（结构），α 软分配解决共享 LFF 矛盾 | v6-Alpha.md 修正 |
| **IDWT 重构** | 保留完整高频，完美重构输入，兼容预训练 | v6-Alpha-corr.md 修正 |
| **soft-median 移除** | soft-median 与 DAT 风格绑定，RWKV 不需要参考帧引导 | — |

### 设计演进路径

```
v5.9.2 (fuse修复+s_illum复生)
  → v6.1 (Cross-RWKV Gate 初版: Python Bi-WKV 循环)
  → v6.2 (VRWKVStyleSpatialMix 严格版 + soft-median移除 + DWT-LFF初版)
  → v6.3 (DWT-LFF互补幅度设计: feat_sace=HF+相位+Conv_amp, feat_tfsi=HF+相位+|LL|−|Conv(LL)|)
  → v6.4 (空间域修正★: α=sigmoid(Conv(LL)), IDWT重构, α init=0.5)
```

### 整体数据流

```
低光输入 y : (B, T, 3, H, W)
    │
    ├─ [Encoder] 3级金字塔 → feats : (B, T, 64, H, W)
    │
    ├─ [TFSI] 时频诊断 → s_illum, s_noise : (B, 1, H, W)
    │     ├─ SpatialBranch: soft-中位值 → μ_t, σ_t, SNR → Conv → F_s
    │     └─ FrequencyBranch: DWT-LFF(中心帧) → feat_tfsi(退化残差+高频) → F_f
    │     └─ ConcatFusion(F_s, F_f) → IntensityHead → s_illum, s_noise
    │
    ├─ [SACE] 跨帧对齐增强 → F_aligned_list, μ_t_clean, σ_t_clean
    │     ├─ DWT-LFF: 共享 TFSI 的适配器 → per-frame feat_sace(归一化光照+高频)
    │     ├─ 中心帧参考: lff_stack[center_idx] → μ_t_clean (soft-median 已移除)
    │     ├─ DeformableCrossAttention: OffsetMaskHead → offset+mask → grid_sample
    │     ├─ Cross-RWKV Gate: Bi-WKV 跨帧扫描 + EntropyGate 聚合
    │     └─ 噪声门控残差: F_aligned = DeformAttn + RWKV + (1-s_noise)·F_lff
    │
    ├─ [IFPN] 光照图估计 → lit_up_map_raw, f_illum_feat, ifpn_side
    ├─ [NDPN] SNR自适应去噪 → f_noise_out
    ├─ [MRPN] 运动补偿 → f_motion_out
    │
    └─ [IGRF] 逆序级联修复 → res_t : (B, 3, H, W)
         Stage1: 去噪(img + Δ(f_noise, s_noise))
         Stage2: 去模糊(img_s1 + Δ(f_motion))
         Stage3: 提亮(img_s2 × lit_up_map + s_illum·corr_mag)
```

### 参数分布

| 模块 | 参数量 | 说明 |
|---|---|---|
| Encoder | ~320K | 3 级金字塔 [32,64,96]→64 |
| TFSI | ~100K | SpatialBranch + IntensityHead |
| SACE (DeformAttn) | ~130K | OffsetMaskHead + DeformableCrossAttention |
| SACE (Cross-RWKV) | ~110K | VRWKVStyleSpatialMix + EntropyGate |
| DWT-LFF | ~18K | HaarDWT + illum_alpha (depthwise+pointwise) + LayerNorm×2 |
| IFPN | ~150K | IllumExtract + Refine + side_head |
| NDPN | ~70K | alpha_conv + Refine |
| MRPN | ~50K | gate + ResBlock |
| IGRF | ~130K | 3×StageBlock + BrightenStage |
| 其他 | ~91K | AmpEnhance(未启用)等 |
| **总计** | **1.169M** | |

---

## 2. 各模块详细设计

### 2.1 Encoder（PyramidEncoder）

**文件**: `models/modules/encoder.py`

3 级卷积金字塔，无归一化（v5.9 实证 LN 在浅卷积主干导致过拟合）。

```
level_channels = [32, 64, 96], fused_channels = 64
H×W → Stage1(stride=1, 32ch) → l1
    → Stage2(stride=2, 64ch) → l2
    → Stage3(stride=2, 96ch) → l3 (H/4 × W/4)
    → lateral1(l1) ← upsample(lateral2(l2) ← upsample(lateral3(l3)))
    → fuse (2×Conv+GELU) → feats : (B, T, 64, H, W)
```

逐帧独立编码，输出 stack 为 (B, T, 64, H, W)。

### 2.2 TFSI（时频源指示器）

**文件**: `models/modules/tfsi.py`

**SpatialBranch**（v5.6 soft_median 可导）:
```
feats (B, T, 64, H, W)
  → soft_median → μ_t (B, 64, H, W)   # 可导软中位值
  → var → σ_t (B, 64, H, W)            # 时域标准差
  → snr = μ_t / (σ_t + ε)              # 信噪比
  → cat[μ_t, σ_t, snr] → Conv → F_s
```

**FrequencyBranch**（v6.4 空间域 DWT-LFF）:
```
中心帧 feats[:, center_idx] → SpatialDWTLFFAdapter → feat_tfsi → F_f
```
TFSI 获得 feat_tfsi = IDWT(LL_deg, 0.5·LH/HL/HH)，其中 LL_deg=(1-α)·LL 是光照退化残差。

**IntensityHead**（v5.5 单层）:
```
F_fused = ConcatFusion(F_s, F_f)
[s_illum, s_noise] = sigmoid(Conv1x1(F_fused))  ∈ [0, 1]
```

### 2.3 SACE（跨帧对齐增强）

**文件**: `models/modules/sace.py`

v6.4 核心改造：① 空间域 DWT-LFF 替代传统 LFF + FFT 域 DWT；② soft-median 移除；③ Cross-RWKV Gate 保留。

#### 2.3.1 DWT-LFF 逐帧归一化（v6.4 空间域）

```
feats (B, T, 64, H, W) → per-frame SpatialDWTLFFAdapter → feat_sace (归一化光照+高频)
所有帧共享同一 SpatialDWTLFFAdapter 实例（与 TFSI 共享）。

SACE 获得 feat_sace = IDWT(LL_ref, 0.5·LH/HL/HH)，其中 LL_ref=α·LL 是归一化光照参考。
```

#### 2.3.2 中心帧参考（soft-median 已移除）

```
μ_t_clean = lff_stack[:, T // 2]    # 直接取中心帧 LFF 特征
σ_t_clean = lff_stack.std(dim=1)   # 时域标准差
```

#### 2.3.3 DeformableCrossAttention

```
query = LayerNorm2d(μ_t_clean)
for each frame:
  kv = LayerNorm2d(lff_feats[t])
  offset, mask = OffsetMaskHead(cat[query, kv])
  f_aligned = grid_sample(value_proj(kv), ref_grid + offset)
  f_aligned = Σ_k softmax(mask)_k · sampled_k + (1-s_noise)·kv
```
参数: n_groups=4, kernel_size=3 → 总采样点=36。

#### 2.3.4 Cross-RWKV Gate（v6 新增，v6.2 修订）

详见 §3.2。

### 2.4 IFPN（光照图估计）

**文件**: `models/modules/ifpn.py`

v5.9.2 新增 side_head 中间监督：
```
F_aligned → coarse_adapter → coarse_feats
→ IllumExtract ×T: (img_down, coarse_feat) → L_t, L_i
→ 帧间相似度加权 → L_ref
→ L_ratio = clamp(L_ref/(L_t+ε), 0.5, 8.0)
→ f_illum_feat = Refine(ratio_feat)
→ lit_up_map_raw = 1 + 4·σ(L_ratio + lit_up_delta)  ∈ [1, 5]
→ ifpn_side = side_head(f_illum_feat)  → L_ifpn_sup(ifpn_side, GT↓)
```

### 2.5 NDPN（SNR自适应去噪）

**文件**: `models/modules/ndpn.py`

```
SNR = |μ_t_clean| / (σ_t_clean+ε)
s_SNR = sigmoid((SNR - τ_mid) / τ_scale)
for each frame i:
  if i == center: α_i = s_SNR
  else: α_i = sigmoid(Conv(|F_aligned_i - F_center|)) · (1 - s_SNR)
f_noise = Refine(Σ α_i·F_aligned_i / Σ α)
```

### 2.6 MRPN（运动补偿）

**文件**: `models/modules/mrpn.py`

```
center_windows = window_partition(F_center, ws=8)
feat_windows = window_partition_video(F_neighbors, ws=8)
corr = softmax(center_windows · feat_windows^T / √C)
f_agg = corr · feat_windows
g = sigmoid(Conv([F_center, f_agg]))
f_motion = ResBlock(g·F_center + (1-g)·f_agg) + F_center
```

### 2.7 IGRF（逆序级联修复）

**文件**: `models/modules/igrf.py`

```
Stage 1 (去噪):
  δ_noise = Fuse(f_noise, img) + intensity_corr(s_noise)
  img_s1 = clamp(img_center + δ_noise, 0, 1)
Stage 2 (去模糊):
  δ_motion = Fuse(f_motion, img_s1)
  img_s2 = clamp(img_s1 + δ_motion, 0, 1)
Stage 3 (混合提亮):
  lit_up_map = lit_up_map_raw · (1 + tanh(Δ)·0.5)
  res_t = clamp(img_s2 × lit_up_map + s_illum·illum_corr(f_illum), 0, 1)
```

---

## 3. 核心新增模块详解

### 3.1 空间域 DWT-LFF 适配器（v6.4）

**文件**: `models/modules/dwt_lff.py`

**设计演进**：初版 v6.2/v6.3 在 FFT 域做 RBF/Conv 处理低频子带，存在三个致命缺陷：
1. **FFT 全局性破坏局部光照分布**——TFSI 无法预测 per-pixel s_illum
2. **`|LL| − |Conv(LL)|` 可为负**——破坏 s_illum 监督单调性
3. **bilinear 上采样丢失高频**——下游模块收不到真实高频

v6-Alpha.md 和 v6-Alpha-corr.md 的修正指向统一方案：**空间域 α 软分配 + IDWT 重构**。

#### 3.1.1 核心设计

```
输入 x (B, C, H, W)
  ↓
[Step 1] Haar DWT 分解（空间域操作，完全保留局部性）
  LL, LH, HL, HH ← DWT(x)  # 各 (B, C, H/2, W/2)
  ↓
[Step 2] 空间域低频光照分离（Conv3×3，非 FFT 域）
  α = sigmoid(Conv3×3(LL))     # ∈ [0, 1]，可学习光照分配比例
  LL_ref = α · LL               # "正常光照"参考 → SACE
  LL_deg = (1 − α) · LL         # 光照退化残差 → TFSI
  ↓
[Step 3] 高频保留（原始系数，不做 conv 融合；0.5× 平均分配）
  LH_half, HL_half, HH_half = 0.5 · LH, 0.5 · HL, 0.5 · HH
  ↓
[Step 4] IDWT 重构（非 bilinear 上采样）
  feat_sace = IDWT(LL_ref, LH_half, HL_half, HH_half)  # (B, C, H, W)
  feat_tfsi = IDWT(LL_deg, LH_half, HL_half, HH_half)  # (B, C, H, W)
  ↓
[Step 5] LayerNorm 稳定通道分布（适配 Cross-RWKV Q-Shift）
  feat_sace = LayerNorm(feat_sace)
  feat_tfsi = LayerNorm(feat_tfsi)
```

#### 3.1.2 关键数学性质

**性质 1：可逆性**
```
feat_sace + feat_tfsi
  = IDWT(LL_ref + LL_deg, LH, HL, HH)
  = IDWT(LL, LH, HL, HH)
  = x    ✓ 完美重构
```

**性质 2：局部性**
Haar IDWT 每个输出 pixel 只依赖 2×2 邻域的低频和高频系数。11像素变化不影响右下角输出——光照退化的空间分布得以保留。

**性质 3：非负性**
α ∈ [0,1] → LL_ref, LL_deg ≥ 0 → 物理可解释（不会有"负光照退化"）。

**性质 4：初始兼容性**
α init = 0.5 → `feat_sace ≈ feat_tfsi ≈ x/2`（训练初期），下游模块受冲击极小，v5.9.2 预训练权重可平滑续训。

#### 3.1.3 TFSI vs SACE 的双向互补

```
SACE 获得 feat_sace = IDWT(α·LL, ½LH, ½HL, ½HH)
  = 归一化的"光照参考" + 完整高频结构
  → 用于跨帧对齐（不受帧间光照差异干扰）

TFSI 获得 feat_tfsi = IDWT((1-α)·LL, ½LH, ½HL, ½HH)
  = "光照退化残差" + 完整高频噪声
  → 用于诊断 s_illum(α留下的残差多→此处光照退化深) 和 s_noise(高频噪声)
```

#### 3.1.4 完整源码

```python
"""
Spatial-Domain DWT-LFF Adapter (v6.4 — 综合修正版)
====================================================
基于 v6-Alpha.md 和 v6-Alpha-corr.md 的综合修正:

修正 v6-Alpha.md:
  1. α = sigmoid(Conv3×3(LL)) soft 分配 → 非负, LL_ref+LL_deg=LL
  2. 空间域 Conv3×3 (非 FFT 域) → 保留局部性
  3. feat_sace 加 LayerNorm → 适配 Cross-RWKV Q-Shift

修正 v6-Alpha-corr.md:
  4. IDWT 重构 (非 bilinear 上采样) → 保留完整高频
  5. HF 0.5× 平均分配 → feat_sace+feat_tfsi 完美重构输入
  6. α init=0.5 → 初始兼容 v5.9.2 预训练权重
"""

class HaarDWT2D(nn.Module):
    def forward(self, x):
        x01, x02 = x[:,:,0::2,:], x[:,:,1::2,:]
        L, H_ = (x01+x02)*0.5, (x01-x02)*0.5
        LL = (L[:,:,:,0::2] + L[:,:,:,1::2]) * 0.5
        LH = (L[:,:,:,0::2] - L[:,:,:,1::2]) * 0.5
        HL = (H_[:,:,:,0::2] + H_[:,:,:,1::2]) * 0.5
        HH = (H_[:,:,:,0::2] - H_[:,:,:,1::2]) * 0.5
        return LL, LH, HL, HH

    def inverse(self, LL, LH, HL, HH):
        B, C, H2, W2 = LL.shape
        L = torch.zeros(B,C,H2,W2*2,device=LL.device,dtype=LL.dtype)
        H_ = torch.zeros_like(L)
        L[:,:,:,0::2]=(LL+LH)*2; L[:,:,:,1::2]=(LL-LH)*2
        H_[:,:,:,0::2]=(HL+HH)*2; H_[:,:,:,1::2]=(HL-HH)*2
        x=torch.zeros(B,C,H2*2,W2*2,device=LL.device,dtype=LL.dtype)
        x[:,:,0::2,:]=(L+H_)*0.5; x[:,:,1::2,:]=(L-H_)*0.5
        return x


class SpatialDWTLFFAdapter(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.illum_alpha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.Sigmoid())
        self.norm_sace = nn.LayerNorm(in_channels)
        self.norm_tfsi = nn.LayerNorm(in_channels)
        # α init=0.5
        for m in self.illum_alpha.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                nn.init.constant_(m.weight, 0.0)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)
        alpha = self.illum_alpha(LL)
        LL_ref, LL_deg = alpha * LL, (1 - alpha) * LL
        LH_h, HL_h, HH_h = LH*0.5, HL*0.5, HH*0.5
        fs = self.dwt.inverse(LL_ref, LH_h, HL_h, HH_h)
        ft = self.dwt.inverse(LL_deg, LH_h, HL_h, HH_h)
        fs = self.norm_sace(fs.permute(0,2,3,1)).permute(0,3,1,2)
        ft = self.norm_tfsi(ft.permute(0,2,3,1)).permute(0,3,1,2)
        return {"x_out": fs, "feat_tfsi": ft, "feat_sace": fs}
```

### 3.2 Cross-RWKV Gate 详细设计

**文件**: `models/modules/cross_rwkv.py`（290 行）

**引用**: Vision-RWKV (ICLR 2025 Spotlight) — Bi-WKV + Q-Shift + VRWKV_SpatialMix

#### 3.2.1 核心参数字典

| 参数 | 形状 | 含义 |
|---|---|---|
| `spatial_decay` | (C,) | per-channel 衰减，fancy init: -5→+3 指数分布 |
| `spatial_first` | (C,) | per-channel 当前帧 boost (u)，zigzag 初始化 |
| `spatial_mix_k/v/r` | (1, 1, C) | Q-Shift 后 key/value/receptance 混合比 |
| `key/value/receptance` | Linear(C, C) | WKV 投影，全零初始化 |
| `output` | Linear(C, C) | 输出投影，全零初始化 |
| `alpha` | (1,1,C,1,1) | 残差缩放，零初始化 |

#### 3.2.2 Bi-WKV 扫描公式

$$y_t = \frac{u \cdot k_t \cdot v_t + \sum_{i=0}^{T-1} e^{w} \cdot k_i \cdot v_i}{u \cdot k_t + \sum_{i=0}^{T-1} e^{w} \cdot k_i}$$

其中 $w,u \in \mathbb{R}^C$ 是 per-channel 固定参数（不随帧距离变化），$T=5$ 帧。`u·k_t·v_t` 给当前帧额外 boost。

#### 3.2.3 EntropyGate 噪声感知聚合

```python
frame_std = feat_stack.std(dim=1)  # 帧间标准差 → proxy for entropy
frame_mean = feat_stack.mean(dim=1)
for each frame t:
  eg_input = cat([out[t], frame_mean, frame_std])  # (B, 3C, H, W)
  eg_w = sigmoid(Conv→GELU→Conv)(eg_input)        # EntropyGate 置信度
  safe_w = 1 - (1-eg_w) * (1-noise_w)              # 双重门控
  F_aligned[t] = frames[t] + alpha * (out[t]*(1-safe_w) + frames[t]*safe_w)
```

---

## 4. 损失函数

**文件**: `losses/losses.py`

### 4.1 总损失

```
L_total = L_pix + λ_freq·L_freq + λ_ssim·L_ssim + λ_perc·L_perc
        + λ_illum·L_illum_smooth + λ_illum_sup·L_illum_sup
        + λ_inter·L_inter + λ_ifpn_sup·L_ifpn_sup
```

### 4.2 各项详细定义

| 损失 | 公式 | λ | 说明 |
|---|---|---|---|
| L_pix | Charbonnier(res_t, GT) | 1.0 | 像素重建 |
| L_freq | L1(|FFT(res_t)|, |FFT(GT)|) | 0.1 | 频域幅度对齐 |
| L_ssim | 1 - SSIM(res_t, GT) | 0.1 | 结构相似性 |
| L_perc | VGG16(relu3_3) L1 | 0.2 | 感知损失 |
| L_illum_smooth | edge_aware_smooth(s_illum, GT) | 0.001 | s_illum 平滑正则（降权） |
| L_illum_sup | L1(s_illum, clamp(1-img/GT,0,1)) | **0.02** | s_illum 暗度监督（v5.9.2） |
| L_inter | Charbonnier(img_s2×lit_up_map, GT) | 0.3 | 中间产物监督（乘法路径） |
| L_ifpn_sup | Charbonnier(ifpn_side, GT↓) | **0.1** | IFPN 中间感知监督（v5.9.2，DarkIR 风格） |

---

## 5. 训练配置

**文件**: `configs/v6_rwkv.yaml`

| 参数 | 值 | 说明 |
|---|---|---|
| batch_size | 2 | 24GB 显存约束 |
| epochs | 30 | — |
| lr | 0.0008 | — |
| warmup | 0 | 从预训练权重续训 |
| weight_decay | 0.0001 | — |
| amp | false | 与 LFF 复数运算不兼容 |
| grad_clip | 0.5 | — |
| num_workers | 4 | DataLoader |
| crop_size | 256 | — |
| window_size | 5 | 多帧窗口 |
| use_cross_rwkv | true | Cross-RWKV Gate 启用 |
| use_dwt_lff | true | 空间域 DWT-LFF 启用 |

---

## 6. 版本演进记录

| 版本 | 关键改动 | 参数量 | 状态 |
|---|---|---|---|
| v6.1 | Cross-RWKV 初版：Python for-loop Bi-WKV + concat Q-Shift | 1.22M | ❌ 随机权重破坏预训练特征 |
| v6.2 | VRWKVStyleSpatialMix 严格版 + soft-median移除 + DWT-LFF初版 | 1.18M | ❌ FFT域 + RBF存在问题 |
| v6.3 | DWT-LFF 互补幅度设计（v6-Alpha 初诊） | 1.18M | ❌ 减法可负 + bilinear上采样破坏HF |
| **v6.4** | **空间域 α 软分配 + IDWT 重建** | **1.17M** | ✅ 训练中 |

### v6.4 vs v6.3 修正点

| 问题 | v6-Alpha.md 诊断 | v6-Alpha-corr.md 修正 | v6.4 实现 |
|---|---|---|---|
| `\|LL\|−\|Conv(LL)\|` 可为负 | 🔴 破坏 s_illum 单调性 | — | α=sigmoid(Conv(LL)) |
| FFT 域 Conv 破坏局部性 | 🔴 TFSI 无法诊断 per-pixel | — | Conv3×3 空间域 |
| bilinear 上采样丢失 HF | — | 🔴 下游模块无真实高频 | IDWT 重构 |
| feat_sace 缺 LN | 🟡 Q-Shift 通道分布不稳定 | — | LayerNorm |

---

## 7. 参考文献

| 论文 | 会议 | 引用模块 |
|---|---|---|
| [Vision-RWKV](https://arxiv.org/pdf/2403.02308) | ICLR 2025 Spotlight | Cross-RWKV Gate: Bi-WKV + Q-Shift + VRWKV_SpatialMix |
| [URWKV](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_URWKV_Unified_RWKV_Model_with_Multi-state_Perspective_for_Low-light_Image_CVPR_2025_paper.pdf) | CVPR 2025 | 低光多状态视角, 噪声 map 注入 |
| [DAT](https://openaccess.thecvf.com/content/CVPR2022/papers/Xia_Vision_Transformer_With_Deformable_Attention_CVPR_2022_paper.pdf) | CVPR 2022 | DeformableCrossAttention 参考 |
| [DarkIR](https://openaccess.thecvf.com/content/CVPR2025/papers/) | CVPR 2025 | EnhanceLoss 中间监督 + denoise-before-brighten |
| [MINS-Net](https://github.com/) | — | EntropyGate 参考 |

### 7.1 参考代码仓库

| 项目 | 路径 | 用途 |
|---|---|---|
| RWKV-block | `reference_repos/RWKV-block/` | Bi-WKV 时间混合实现 |
| RWKV-LM | `reference_repos/RWKV-LM/` | WKV 原始公式 |
| Vision-RWKV | `reference_repos/Vision-RWKV/` | Q-Shift, SpatialMix, fancy init |
| MINS-Net | `reference_repos/MINS-Net/` | EntropyGate |
| DarkIR | `reference_repos/DarkIR/` | EnhanceLoss, FreMLP |
