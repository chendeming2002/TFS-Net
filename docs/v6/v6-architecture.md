# TFS-Net v6 Delta 模型架构设计文档

> 日期：2026-06-30 (更新: Delta-plan 全量实施)
> 版本：v6 Delta（Charlie-Mark4）
> 训练配置：`configs/v6_bravo.yaml`，batch=1, lr=8e-4, epochs=50, warmup=5
> 参数量：1.638M

---

## 1. 概述

v6 Delta 对 SACE 进行根本性重构，将扫描轴从时间(T)→空间(H×W)，并输出显式时序对应矩阵。

| 模块 | 改动 | 文献 |
|------|------|------|
| **SACE** | MVC-Shift + SpatialWKV2D 四方向空间扫描 + C_omega_list + F_t_aligned | RSRWKV (TCSVT 2025), Vision-RWKV (ICLR 2025) |
| **IFPN** | A_illu 生成（从 IGRF 移入）+ F_t_aligned 光照锚定 | DiTVR (CVPR 2026), FRESCO (2025) |
| **NDPN** | C_omega conf_proj + noise_extract + denoise_strength + gamma | DiTVR, Revisiting Temporal Alignment (CVPR 2022) |
| **MRPN** | C_omega motion_estimator + sigma_proj + comp_gate + motion_refine + gamma | JFFRA (2025), SAT (2026) |
| **IGRF** | 移除 unified_illu，接收 A_illu；移除 s_noise 路径 | — |
| **CrossFusionGate** | deploy 模式重参数化 (DRNet 范式) | DRNet (CVPR 2026) |

### 完整数据流

```
输入 x: (B, T=5, 3, H, W)
  │
  ├─→ Encoder → F_stack (B, T, 64, H, W)
  │
  ├─→ TFSI → s_illum, s_noise
  │     ├─ s_illum → IFPN (唯一路径)
  │     └─ s_noise → NDPN (noise_proj 条件注入)
  │
  ├─→ SACE (Delta) → sace_out, C_omega_list, F_t_aligned, mu_t_clean, sigma_t_clean
  │     ├─ [Downsample H/2]
  │     ├─ [MVC-Shift] 3 dilated DWConv (d=1,2,3) 逐帧
  │     ├─ [SpatialWKV2D] 4方向 Bi-WKV 空间扫描
  │     ├─ [Upsample] → sace_out (B,T,C,H,W)
  │     ├─ [TemporalCorrespondence] Q-K cosine → C_omega_list (T-1)×(B,N²)
  │     └─ [TemporalAggregation] C_omega warp + frame_gate → F_t_aligned (B,C,H,W)
  │
  ├─→ F_aligned_list = [sace_out[:, t] for t] → IFPN/NDPN/MRPN
  │
  ├─→ IFPN(F_aligned_list, s_illum, F_t_aligned) → lit_up_map_raw, f_illum_feat, A_illu
  ├─→ NDPN(F_aligned_list, s_noise, mu, sigma, C_omega_list, F_t_aligned) → f_noise_out
  ├─→ MRPN(F_aligned_list, sigma, C_omega_list, F_t_aligned) → f_motion_out
  │
  ├─→ CrossFusionGate(f_noise_out, f_motion_out) → f_noise_gated, f_motion_gated
  │
  └─→ IGRF(f_illum_feat, f_noise_gated, f_motion_gated, lit_up_map_raw, img_center, A_illu)
        ├─ Stage1: img_s1 = denoise(f_noise_gated, img_center)  ★ s_noise 不在 IGRF
        ├─ Stage2: img_s2 = motion(f_motion_gated, img_s1)
        └─ Stage3: res_t = clamp(img_s2 × lit_up_map × (1+A_illu))
```

### IFPN → IGRF 三路信号

| 信号 | 语义 | IGRF 用法 |
|------|------|----------|
| `lit_up_map_raw` | 像素级增亮增益 ([1, 1+max_bright]) | `img_s2 × lit_up_map` 乘法提亮 |
| `f_illum_feat` | 光照特征（含 s_illum 先验） | `delta_refine(f_illum_feat, img_dark)` 修正 lit_up_map 偏移 |
| `A_illu` | 空间光照注意力 (0~1) | `lit_up_map × (1 + A_illu)` 暗区放大/亮区抑制 |

---

## 2. 核心模块详解

### 2.1 SACE Delta — 空间扫描 + 时序对应

**文件**: `models/modules/pure_rwkv_sace.py`

**输入**: Encoder 原始特征 `(B,T,C,H,W)`（无 DWT-LFF）

**处理流**:
```
feats (B,T,C,H,W)
  │
  ├─ [Downsample] H/2 × W/2
  ├─ [MVC-Shift] 3 dilated DWConv(d=1,2,3) + 1×1 cross-channel → x_shifted
  ├─ [SpatialWKV2D] 4方向扫描 (水平/垂直/主对角/副对角)
  │     ├─ C→4 heads × C/4, each: R/K/V proj → scan → Bi-WKV cumsum
  │     └─ Concat 4 heads → σ(R)⊙wkv → proj_out → LayerNorm
  ├─ [Channel Mix] LN → Conv(C→4C) → GELU → Conv(4C→C) → + x * gamma
  ├─ [Upsample] → sace_out (B,T,C,H,W)
  │
  ├─ [TemporalCorrespondence]
  │     proj_qk (C→D) → normalize → bmm(Q,K^T)/tau → softmax → C_omega_list
  │
  └─ [TemporalAggregation]
        C_omega × neighbor → warp → frame_gate → softmax加权 → upsample → F_t_aligned
```

**输出**:
| 输出 | 形状 | 含义 |
|------|------|------|
| `sace_out` | (B,T,C,H,W) | 空间增强后多帧特征 |
| `C_omega_list` | [(B,N,N)×(T-1)] | 中心帧与邻帧的 temporal correspondence |
| `F_t_aligned` | (B,C,H,W) | 中心帧经 C_omega 加权聚合的增强特征 |
| `mu_t_clean` | (B,C,H,W) | 中心帧参考 |
| `sigma_t_clean` | (B,C,H,W) | 帧间标准差 |

### 2.2 IFPN — 光照三路输出

**文件**: `models/modules/ifpn.py`

```
aligned_feats → coarse_adapter → + s_illum_proj(s_illum) → IllumExtract → feats
  │
  ├─ L_ratio → lit_up_delta → lit_up_map_raw        [像素增亮图]
  ├─ feat_refine → f_illum_feat                      [光照特征, 含 s_illum 先验]
  │     └─ + illu_anchor(F_t_aligned) × gate.tanh()  [时序锚定, 防闪烁]
  └─ illu_conv(f_illum_feat) → A_illu                [空间光照注意力]
```

### 2.3 NDPN — 置信度引导去噪

**文件**: `models/modules/ndpn.py`

```
C_omega diag → conf_proj(Linear) → conf_map            [对应置信度]
f_enc + F_t_aligned → noise_extract → noise_feat       [噪声估计]
noise_feat + conf_map → denoise_strength → strength     [去噪强度]
f_noise = f_enc - noise × strength × gamma + noise_proj(s_noise)
```

### 2.4 MRPN — 运动强度补偿

**文件**: `models/modules/mrpn.py`

```
C_omega diag → reshape → motion_estimator(Conv) → motion_mag   [运动强度]
F_t_aligned → _aggregate_neighbors → f_omega_aligned            [窗口相关]
F_t_aligned + f_omega → motion_refine → motion_delta            [运动修正]
motion_delta × comp_gate(motion_mag) × gamma → compensation     [门控补偿]
```

### 2.5 CrossFusionGate — 结构性重参数化

**文件**: `models/tfs_net.py`

| 模式 | deploy=False (训练) | deploy=True (推理) |
|------|-------------------|-------------------|
| gate_noise | SE(f_motion)→sigmoid | scale_noise 静态参数 |
| gate_motion | SE(f_noise)→sigmoid | scale_motion 静态参数 |

### 2.6 IGRF — 三阶段修正

**文件**: `models/modules/igrf.py`

```
Stage1: img_s1 = clamp(img_center + δ(f_noise_gated))
Stage2: img_s2 = clamp(img_s1 + δ(f_motion_gated))
Stage3: lit_up_map = lit_up_map_raw × (1+tanh(δ)×max_delta) × (1+A_illu)
        res_t = clamp(img_s2 × lit_up_map, 0, 1)
```

---

## 3. 参数分布

| 模块 | 参数量 |
|------|--------|
| Encoder | ~320K |
| TFSI | ~120K |
| SACE (MVCShift + WKV + Corr + Agg) | ~290K |
| IFPN (含 s_illum_proj + illu_conv + illu_anchor) | ~160K |
| NDPN (含 conf_proj + noise_extract + denoise_strength) | ~85K |
| MRPN (含 motion_estimator + comp_gate + motion_refine) | ~80K |
| CrossFusionGate | ~8K |
| IGRF | ~120K |
| 其他 | ~55K |
| **总计** | **~1.64M** |

---

## 4. 版本演进

| 版本 | 关键改动 | 参数量 | 状态 |
|------|---------|--------|------|
| v5.9.2 | s_illum 复生 + IFPN 监督 | 1.14M | 20.39 PSNR |
| v6.5 | PureRWKV 移除 DAT | 1.17M | 20.36 |
| v6 Bravo | 损失重调 + DWT-LFF分裂 + V raw | 1.20M | 20.05 |
| v6 Charlie | 多帧 F.B. + concat fusion + NDPN s_noise + σ→MRPN | 1.25M | — |
| v6 Charlie2 | s_noise→NDPN + encoder feat→IFPN + s_illum gate + VSRELL A_illu | 1.26M | — |
| v6 Charlie3 | s_illum→IFPN单路径 + CrossFusionGate + LearnableScaleFusion | 1.32M | ep29 loss=0.176 |
| **v6 Delta** | **空间扫描 2D-WKV + C_omega + F_t_aligned + A_illu→IFPN + deploy CFG** | **1.64M** | 训练中 |

---

## 5. 设计原则

1. **空间扫描替代时间扫描** (RSRWKV TCSVT 2025): H×W 序列充分利用 RWKV 长序列能力
2. **显式时序对应** (VRT CVPR 2022): C_omega + F_t_aligned 替代隐式 RWKV mix
3. **光照编码端估计** (DiTVR CVPR 2026): A_illu 在 IFPN 端利用多尺度特征生成
4. **置信度引导去噪** (Revisiting Temporal Alignment CVPR 2022): C_omega diagonal → conf/strength
5. **运动强度补偿** (JFFRA/SAT): C_omega diagonal deviation → motion_mag
6. **结构性重参数化** (DRNet CVPR 2026): CrossFusionGate 训练动态/推理静态
7. **s_noise NDPN 专属**: 不进入 IGRF；s_illum IFPN 唯一出口
