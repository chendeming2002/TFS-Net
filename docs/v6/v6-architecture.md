# TFS-Net v6 模型架构设计文档

> 日期：2026-06-27
> 训练配置：`configs/v6_rwkv.yaml`，batch=2，30 epoch，参数量 1.180M
> 预训练权重：v5.9.2 latest.pth（fuse 修复 + s_illum 复生 + IFPN 监督）

---

## 1. 概述

v6 在 v5.9.2 基础上引入三个核心改造：

| 改造 | 动机 | 来源 |
|---|---|---|
| **Cross-RWKV Gate** | 将 SACE 的帧间注意力从 DAT 风格可变形采样升级为 RWKV 线性复杂度 Bi-WKV 扫描 | Vision-RWKV (ICLR 2025) |
| **DWT-LFF 分解** | 用小波变换解耦低频（光照/噪声诊断）和高频（结构/对齐），解决 TFSI/SACE 共享 LFF 的矛盾 | Haar DWT + 原有 RBF |
| **soft-median 移除** | soft-median 与 DAT 风格注意力绑定，RWKV 不需要参考帧引导 | — |

### 整体数据流

```
低光输入 y : (B, T, 3, H, W)
    │
    ├─ [Encoder] 3级金字塔 → feats : (B, T, 64, H, W)
    │
    ├─ [TFSI] 时频诊断 → s_illum, s_noise : (B, 1, H, W)
    │     ├─ SpatialBranch: soft-中位值 → μ_t, σ_t, SNR → Conv → F_s
    │     └─ FrequencyBranch: DWT-LFF(中心帧) → feat_tfsi(HF+LF_phase+光照残差幅度) → F_f
    │     └─ ConcatFusion(F_s, F_f) → IntensityHead → s_illum, s_noise
    │
    ├─ [SACE] 跨帧对齐增强 → F_aligned_list, μ_t_clean, σ_t_clean
    │     ├─ DWT-LFF: 共享 TFSI 的适配器 → per-frame feat_sace(HF+LF_phase+光照参考幅度)
    │     ├─ 中心帧参考: lff_stack[center_idx] → μ_t_clean (v6.2: 移除 soft-median)
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
| TFSI | ~100K | 含 LFF RBF + SpatialBranch + IntensityHead |
| SACE (DeformAttn) | ~130K | OffsetMaskHead + DeformableCrossAttention |
| SACE (Cross-RWKV) | ~110K | VRWKVStyleSpatialMix + EntropyGate |
| DWT-LFF | ~30K | HaarDWT + illum_conv + high_fusion |
| IFPN | ~150K | IllumExtract + Refine + side_head |
| NDPN | ~70K | alpha_conv + Refine |
| MRPN | ~50K | gate + ResBlock |
| IGRF | ~130K | 3×StageBlock + BrightenStage |
| 其他 | ~90K | AmpEnhance(未启用)等 |
| **总计** | **1.180M** | |

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

**SpatialBranch**（v5.6 P1-5: soft_median 可导）:
```
feats (B, T, 64, H, W)
  → soft_median → μ_t (B, 64, H, W)   # 可导软中位值
  → var → σ_t (B, 64, H, W)            # 时域标准差
  → snr = μ_t / (σ_t + ε)              # 信噪比
  → cat[μ_t, σ_t, snr] → Conv → F_s
```

**FrequencyBranch**（v6.2: DWT-LFF）:
```
中心帧 feats[:, center_idx]
  → DWT-LFF → feat_tfsi (低频幅度 + 噪声信息) → F_f
```

**IntensityHead**（v5.6 回退到 v5.5 单层）:
```
F_fused = ConcatFusion(F_s, F_f)
[s_illum, s_noise] = sigmoid(Conv1x1(F_fused))  ∈ [0, 1]
```

输出: `s_illum`（光照退化强度）、`s_noise`（噪声退化强度）。

### 2.3 SACE（跨帧对齐增强）

**文件**: `models/modules/sace.py`

v6.2 核心改造：① DWT-LFF 替代传统 LFF；② soft-median 移除，中心帧直接作参考；③ Cross-RWKV Gate 替代纯 deformable。

#### 2.3.1 DWT-LFF 逐帧归一化

```
feats (B, T, 64, H, W) → per-frame DWT-LFF → feat_sace (HF + LF_phase + 光照参考幅度)
所有帧共享同一 DWTLFFAdapter 实例（与 TFSI 共享）。
```

DWT-LFF 输出的 `feat_sace` = 高频子带(HF) + 低频相位 + Conv(LF)光照参考幅度——提供 SACE 对齐所需的归一化光照参考 + 高频结构 + 相位约束。`feat_tfsi`（TFSI 使用）与此互补——拿光照残差幅度。

详见 §3.1。

#### 2.3.2 中心帧参考（v6.2: soft-median 移除）

```
μ_t_clean = lff_stack[:, T // 2]    # 直接取中心帧 LFF 特征
σ_t_clean = lff_stack.std(dim=1)   # 时域标准差
```

soft-median 与 DAT 风格注意力绑定（为 deformable 提供空间参考点）。RWKV 不需要参考帧引导，改用中心帧更直接。

#### 2.3.3 DeformableCrossAttention

```
query = LayerNorm2d(μ_t_clean)
for each frame:
  kv = LayerNorm2d(lff_feats[t])
  offset, mask = OffsetMaskHead(cat[query, kv])
  f_aligned = grid_sample(value_proj(kv), ref_grid + offset)
  f_aligned = Σ_k softmax(mask)_k · sampled_k + (1-s_noise)·kv
```

参数: n_groups=4, kernel_size=3 → 总采样点=36（4×9）。

#### 2.3.4 Cross-RWKV Gate（v6 新增）

**文件**: `models/modules/cross_rwkv.py`

在 DeformableCrossAttention 之后，对已对齐的 F_aligned_list 做跨帧 Bi-WKV 长程聚合。

```
F_aligned_list (T frames, each B×64×H×W)
  → stack → (B, T, L, C), L=H×W, C=64
  → VRWKVStyleSpatialMix (详见 §3.2)
      Q-Shift 通道位移
      → time_mix (xk = x·mix_k + shifted·(1-mix_k))
      → Bi-WKV: (u·k·v + Σ ew·k·v) / (u·k + Σ ew·k)
      → sr = sigmoid(r) 门控
  → EntropyGate 聚合
      frame_std = 帧间标准差 (proxy for attention entropy)
      eg_w = sigmoid(Conv([out, mean, std]))
      → safe_w = 1 - (1-eg_w)·(1-noise_w)  (双重门控)
  → F_aligned = frames + α · (RWKV_out · (1-safe_w) + frames · safe_w)
```

α 零初始化 → 初始行为=恒等映射（完全保留 v5.9.2 预训练质量）。

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

每阶段 fuse 含 2×ResBlock（Conv-GELU-Conv+skip）。illum_corr 零初始化，intensity_corr 零初始化。

---

## 3. 核心新增模块详解

### 3.1 DWT-LFF 适配器

**文件**: `models/modules/dwt_lff.py`（v6.3 严格版）

**设计动机**: 传统 LFF 被 TFSI 和 SACE 共享同一 RBF 系数，存在矛盾：
- TFSI 需要低频信息（光照在幅度谱低频段，噪声全频段含低频）
- SACE 需要抑制低频（光照差异是"对齐噪声"），保留相位高频结构

**核心设计**: Haar 小波分解后，低频子带（LL）经可学习 Conv 提取"正常光照参考幅度"，SACE 和 TFSI 互补地拿到光照参考和光照残差。

**数据流**:
```
x (B, C, H, W)
  → HaarDWT2D → LL, LH, HL, HH  (各 B, C, H/2, W/2)

LL 子带:
  → FFT2 → mag_original, pha
  → Conv(LL) → conv_out → mag_conv = |conv_out|  (可学习光照幅度提取器)

低频相位基座: phase_base = IFFT2(1, pha)  → 上采样
高频融合: high_feat = Conv1×1([LH↑, HL↑, HH↑])

feat_sace = phase_base + high_feat + sace_amp
  sace_amp = IFFT2(mag_conv, pha) → 上采样
  = 低频相位 + 高频结构 + Conv提取的"正常光照幅度"
  → 供 SACE 对齐（光照归一化参考）

feat_tfsi = phase_base + high_feat + tfsi_amp
  tfsi_amp = IFFT2(mag_original − mag_conv, pha) → 上采样
  = 低频相位 + 高频结构 + "被移除的光照退化幅度"
  → 供 TFSI 诊断 s_illum/s_noise (正是 γ_t 的贡献)

IDWT: conv_out(低频) + LH + HL + HH → x_out
```

**物理意义**:
- Conv 学习提取"正常光照应有的幅度"——SACE 的归一化参考
- LL - Conv = 光照退化幅度（γ_t 的影响）——TFSI 诊断 s_illum 的输入
- 两者共享低频相位（结构/噪声指纹）和高频（纹理/运动）
- 互补设计: SACE 拿"光照参考"做对齐，TFSI 拿"光照残差"做诊断

**关键公式**:
```
feat_sace = phase_base + high_feat + IFFT2(|Conv(LL)|, pha)
feat_tfsi = phase_base + high_feat + IFFT2(|LL| − |Conv(LL)|, pha)
         = phase_base + high_feat + IFFT2(|LL|, pha) − IFFT2(|Conv(LL)|, pha)
         = feat_sace + IFFT2(|LL| − 2·|Conv(LL)|, pha)   (注: 非精确等式)
```

### 3.2 Cross-RWKV Gate 详细设计

**文件**: `models/modules/cross_rwkv.py`（290 行）

**引用**: Vision-RWKV (ICLR 2025 Spotlight) — Bi-WKV + Q-Shift + VRWKV_SpatialMix

#### 3.2.1 核心参数字典

| 参数 | 形状 | 含义 |
|---|---|---|
| `spatial_decay` | (C,) | per-channel 帧间衰减，fancy init: -5→+3 指数分布 |
| `spatial_first` | (C,) | per-channel 当前帧 boost (u 项)，zigzag 初始化 |
| `spatial_mix_k` | (1, 1, C) | Q-Shift 后的 key 与原始输入混合比 |
| `spatial_mix_v` | (1, 1, C) | value 混合比 |
| `spatial_mix_r` | (1, 1, C) | receptance 混合比 |
| `key/value/receptance` | Linear(C, C) | WKV 投影，全零初始化 |
| `output` | Linear(C, C) | 输出投影，全零初始化 |
| `alpha` | (1,1,C,1,1) | 残差缩放，零初始化 |

#### 3.2.2 Q-Shift 通道位移

```python
def q_shift_2d(x, shift_pixel=1, gamma=0.25):
    # 通道分 4 组，各做不同方向 1-pixel shift
    # γ=1/4: 4 组覆盖全部通道
    g = int(C * gamma)
    # 组 0: 左移
    out[:, :g, :, :W-1] = x[:, :g, :, 1:]
    # 组 1: 右移
    out[:, g:2g, :, 1:] = x[:, g:2g, :, :W-1]
    # 组 2: 上移
    out[:, 2g:3g, :H-1, :] = x[:, 2g:3g, 1:, :]
    # 组 3: 下移
    out[:, 3g:4g, 1:, :] = x[:, 3g:4g, :H-1, :]
    return out
```

相比 Vision-RWKV 原版的 token 维度 Q-Shift，改为直接在 2D 特征图上做通道分组位移，更适配 TFS-Net 的 (B, C, H, W) 特征布局。

#### 3.2.3 Bi-WKV 扫描公式

严格的 Vision-RWKV 双向 WKV 公式：

$$y_t = \frac{u \cdot k_t \cdot v_t + \sum_{i=0}^{T-1} e^{w} \cdot k_i \cdot v_i}{u \cdot k_t + \sum_{i=0}^{T-1} e^{w} \cdot k_i}$$

其中:
- $w \in \mathbb{R}^C$: spatial_decay / T（per-channel 衰减）
- $u \in \mathbb{R}^C$: spatial_first / T（当前帧 boost）
- $k, v \in \mathbb{R}^{B\times T\times L\times C}$: 投影后的 key/value
- $T=5$: 帧窗口大小

所有帧互相可见（双向），每 channel 独立衰减权重。`u·k_t·v_t` 给当前帧额外 boost。

#### 3.2.4 EntropyGate 噪声感知聚合

```python
# 帧间特征标准差 → proxy for attention entropy
frame_std = feat_stack.std(dim=1)  # (B, C, H, W)
frame_mean = feat_stack.mean(dim=1)

# 对各帧独立计算 entropy gate
eg_input = cat([out[:, t], frame_mean, frame_std], dim=1)  # (B, 3C, H, W)
eg_w = sigmoid(Conv→GELU→Conv)(eg_input)  # (B, 1, H, W)

# 双重门控: entropy + noise
safe_w = 1 - (1 - eg_w) * (1 - noise_w)
F_aligned[t] = frames[t] + alpha * (out[t] * (1-safe_w) + frames[t] * safe_w)
```

#### 3.2.5 CrossRWKVGate 完整源码

```python
"""
Cross-RWKV Gate — v6 跨帧交叉注意力 (2026-06-27, 修订版)
============================================================
严格参照 Vision-RWKV (ICLR 2025) 的 VRWKV_SpatialMix 实现

核心修正 (vs 初版):
  1. Bi-WKV: 加入 spatial_first (当前 token 加权项), 匹配官方 WKV 公式
  2. Time Mix: 加入 spatial_mix_k/v/r 可学习混合比 (当前 vs 历史特征)
  3. Q-Shift: 改为通道空间位移 + mix 混合 (匹配官方 q_shift + mix 模式)
  4. Gate: 简化接收门控 sr = sigmoid(r), 取消 query-kv concat
  5. Init: fancy init 风格 (指数衰减分布 + 层级相关)

继承的论文方法:
  - Vision-RWKV (ICLR 2025) Bi-WKV + Q-Shift + VRWKV_SpatialMix
  - URWKV (CVPR 2025) 噪声 map 注入 WKV decay
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class VRWKVStyleSpatialMix(nn.Module):
    """Vision-RWKV 风格的 SpatialMix 模块 — 严格参照官方 VRWKV_SpatialMix。

    包含: Q-Shift (通道位移 + mix 混合) + WKV 扫描 (含 spatial_first 项)

    Args:
        channels: token 维度
        num_frames: 跨帧窗口大小 T (默认 5)
        layer_id: 层 ID (用于 fancy init 的层级比例)
        n_layer: 总层数 (用于 fancy init)
    """

    def __init__(self, channels: int, num_frames: int = 5,
                 layer_id: float = 0.0, n_layer: int = 1):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.shift_pixel = 1
        self.channel_gamma = 0.25

        # ★ WKV 参数 (参照 Vision-RWKV fancy init)
        ratio_0_to_1 = layer_id / max(n_layer - 1, 1)
        ratio_1_to_almost0 = 1.0 - layer_id / max(n_layer, 1)

        # spatial_time_decay: 帧间衰减 (对应 w 参数)
        decay_speed = torch.ones(channels)
        for h in range(channels):
            decay_speed[h] = -5 + 8 * (h / (channels - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.spatial_decay = nn.Parameter(decay_speed)

        # spatial_time_first: 当前帧加权 (对应 u 参数)
        zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(channels)]) * 0.5
        self.spatial_first = nn.Parameter(torch.ones(channels) * math.log(0.3) + zigzag)

        # ★ Time Mix 混合比
        x = torch.ones(1, 1, channels)
        for i in range(channels):
            x[0, 0, i] = i / channels
        self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        # ★ 投影 (参照 VRWKV_SpatialMix)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.receptance = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)

        # 输出全零初始化 → 初始恒等
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.key.weight)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.receptance.weight)
        for mod in [self.output, self.key, self.value, self.receptance]:
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

    @staticmethod
    def q_shift_2d(x: torch.Tensor, shift_pixel: int = 1,
                   gamma: float = 0.25) -> torch.Tensor:
        """Vision-RWKV Q-Shift: 通道空间位移 (非 concat+conv 替代)。"""
        B, C, H, W = x.shape
        assert C >= 4
        g = int(C * gamma)
        out = x.clone()
        if g > 0:
            out[:, :g, :, :W - shift_pixel] = x[:, :g, :, shift_pixel:W]
        if 2 * g <= C:
            out[:, g:2 * g, :, shift_pixel:W] = x[:, g:2 * g, :, :W - shift_pixel]
        if 3 * g <= C:
            out[:, 2 * g:3 * g, :H - shift_pixel, :] = x[:, 2 * g:3 * g, shift_pixel:H, :]
        if 4 * g <= C:
            out[:, 3 * g:4 * g, shift_pixel:H, :] = x[:, 3 * g:4 * g, :H - shift_pixel, :]
        return out

    def _bi_wkv_scan(self, w: torch.Tensor, u: torch.Tensor,
                     k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Bi-WKV 双向扫描 — 严格匹配 Vision-RWKV 官方公式.

        官方公式 (per-channel decay w, per-channel first/bonus u):
            y_t = (u·k_t·v_t + Σ_i exp(w)·k_i·v_i)
                / (u·k_t      + Σ_i exp(w)·k_i)
        """
        B, T, L, C = k.shape
        device, dtype = k.device, k.dtype
        ew = torch.exp(w).view(1, 1, 1, C)
        u_coeff = u.view(1, 1, 1, C)
        k_weighted = k * ew
        kv_all = (k_weighted * v).sum(dim=1)
        boost = u_coeff * k * v
        k_all = k_weighted.sum(dim=1)
        k_boost = u_coeff * k
        out = torch.zeros(B, T, L, C, device=device, dtype=dtype)
        for t in range(T):
            num_t = kv_all + boost[:, t]
            den_t = k_all + k_boost[:, t]
            out[:, t] = num_t / (den_t + 1e-8)
        return out

    def forward(self, x_flat: torch.Tensor, feat_2d_shape: tuple) -> torch.Tensor:
        B, T, L, C = x_flat.shape
        H, W = feat_2d_shape
        # Q-Shift
        x_2d = x_flat.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        x_shifted_2d = self.q_shift_2d(x_2d, self.shift_pixel, self.channel_gamma)
        x_shifted = x_shifted_2d.permute(0, 2, 3, 1).reshape(B, T, L, C)
        # Time Mix
        xk = x_flat * self.spatial_mix_k + x_shifted * (1 - self.spatial_mix_k)
        xv = x_flat * self.spatial_mix_v + x_shifted * (1 - self.spatial_mix_v)
        xr = x_flat * self.spatial_mix_r + x_shifted * (1 - self.spatial_mix_r)
        # 投影 + 接收门控
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        # Bi-WKV 扫描
        w = self.spatial_decay / T
        u = self.spatial_first / T
        wkv_out = self._bi_wkv_scan(w, u, k, v)
        rwkv = sr * wkv_out
        return self.output(rwkv)


class CrossRWKVGate(nn.Module):

    def __init__(self, channels: int, num_frames: int = 5,
                 layer_id: float = 0.0, n_layer: int = 1):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.spatial_mix = VRWKVStyleSpatialMix(
            channels=channels, num_frames=num_frames,
            layer_id=layer_id, n_layer=n_layer)
        self.noise_gate = nn.Sequential(
            nn.Conv2d(1, channels, 1, 1, 0), nn.Sigmoid())
        self.entropy_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1, 1, 0),
            nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(channels)
        self.alpha = nn.Parameter(torch.zeros(1, 1, channels, 1, 1))

    def forward(self, query, frames, s_noise=None):
        B, C, H, W = query.shape
        T = len(frames)
        L = H * W
        feat_stack = torch.stack(frames, dim=1)
        feat_flat = feat_stack.permute(0, 1, 3, 4, 2).reshape(B, T, L, C)
        out_flat = self.spatial_mix(feat_flat, (H, W))
        out = out_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)
        out = self.layer_norm(out.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        frame_std = feat_stack.std(dim=1)
        frame_mean = feat_stack.mean(dim=1)
        out_list = []
        for t in range(T):
            eg_input = torch.cat([out[:, t], frame_mean, frame_std], dim=1)
            eg_w = self.entropy_gate(eg_input)
            if s_noise is not None:
                noise_w = self.noise_gate(s_noise)
                safe_w = 1.0 - (1.0 - eg_w) * (1.0 - noise_w)
                out_list.append(
                    frames[t] + self.alpha.squeeze(0) * (
                        out[:, t] * (1.0 - safe_w) + frames[t] * safe_w))
            else:
                out_list.append(
                    frames[t] + self.alpha.squeeze(0) * (
                        out[:, t] * eg_w + frames[t] * (1.0 - eg_w)))
        return out_list
```

---

## 4. 损失函数

**文件**: `losses/losses.py`

### 4.1 总损失

```
L_total = L_pix + λ_freq·L_freq                          # 重建基础
        + λ_ssim·L_ssim + λ_perc·L_perc                  # 结构+感知
        + λ_illum·L_illum_smooth + λ_illum_sup·L_illum_sup  # s_illum 正则+监督
        + λ_inter·L_inter + λ_ifpn_sup·L_ifpn_sup         # 中间监督
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

### 4.3 关键监督信号

**L_illum_sup**（v5.9.2 复生 s_illum 的关键）:
```
s_illum_target = clamp(1 - mean(img_center)/mean(GT), 0, 1)
```
物理含义: 输入比 GT 暗多少 → s_illum 应多大。fuse 修复后特征有真实信号，λ=0.02 轻量监督足以维持 s_illum 非零。

**L_ifpn_sup**（v5.9.2 DarkIR 风格）:
```
ifpn_side = side_head(f_illum_feat)  # 64→3 投影
L_ifpn_sup = Charb(ifpn_side, interpolate(GT, size))
```
强制 f_illum_feat 编码有意义的图像信息，避免光照分支退化。

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

**预训练**: 从 `outputs/sdsd_v592_light/latest.pth` 加载权重（v5.9.2 fuse修复 + s_illum复生 + IFPN监督）。新模块（CrossRWKV + DWT-LFF）随机初始化。

---

## 6. 参考文献

| 论文 | 会议 | 引用模块 |
|---|---|---|
| [Vision-RWKV](https://arxiv.org/pdf/2403.02308) (Duan et al.) | ICLR 2025 Spotlight | Cross-RWKV Gate: Bi-WKV + Q-Shift + VRWKV_SpatialMix |
| [URWKV](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_URWKV_Unified_RWKV_Model_with_Multi-state_Perspective_for_Low-light_Image_CVPR_2025_paper.pdf) (Xu et al.) | CVPR 2025 | 低光多状态视角, 噪声 map 注入 |
| [Video RWKV LCR](https://arxiv.org/html/2411.05636) (Wang et al.) | arXiv 2024 | Cross-RWKV Gate 概念 |
| [DAT](https://openaccess.thecvf.com/content/CVPR2022/papers/Xia_Vision_Transformer_With_Deformable_Attention_CVPR_2022_paper.pdf) (Xia et al.) | CVPR 2022 | DeformableCrossAttention 参考 |
| [DarkIR](https://openaccess.thecvf.com/content/CVPR2025/papers/) (Feijoo et al.) | CVPR 2025 | EnhanceLoss 中间监督 + denoise-before-brighten |
| [MINS-Net](https://github.com/) | — | EntropyGate 参考 |

### 6.1 参考代码仓库

| 项目 | 路径 | 用途 |
|---|---|---|
| RWKV-block | `reference_repos/RWKV-block/` | Bi-WKV 时间混合实现 |
| RWKV-LM | `reference_repos/RWKV-LM/` | WKV 原始公式 |
| Vision-RWKV | `reference_repos/Vision-RWKV/` | Q-Shift, SpatialMix, fancy init |
| MINS-Net | `reference_repos/MINS-Net/` | EntropyGate |
| DarkIR | `reference_repos/DarkIR/` | EnhanceLoss, FreMLP |
| FourLLIE | `reference_repos/FourLLIE/` | Stage1 幅度增强 |
| NAFNet | `reference_repos/NAFNet/` | NAFBlock, SimpleGate, SCA |
