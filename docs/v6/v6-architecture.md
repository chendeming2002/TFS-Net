# TFS-Net v6 Delta Mark1 模型架构设计文档

> 日期：2026-07-01 (更新: Mark1 全量实施)
> 版本：v6 Delta Mark1
> 训练配置：`configs/v6_bravo.yaml`，batch=1, lr=8e-4, epochs=50, warmup=5
> 参数量：1.688M

---

## 1. 概述

v6 Delta Mark1 对 SWD 进行子带级分流重构，解决 s_illum/s_noise norm 爆炸导致的训练崩溃。

| 模块 | 缩写 | 改动 | 文献 |
|------|------|------|------|
| **SWD** | Spatial Wavelet Diverter | 子带级 LL/HF 分流 (替代 DWT-LFF inverse DWT) | — |
| **TFDE** | Temporal-Frequency Degradation Estimator | 时频退化估计 (原 TFSI) | — |
| **TCA** | Temporal Correspondence & Alignment | MVC-Shift + 4方向 WKV 空间扫描 (原 SACE) | RSRWKV (TCSVT 2025) |
| **ISPN** | Illumination-Source Processing Network | 光照处理 + A_illu 生成 (原 IFPN) | DiTVR (CVPR 2026) |
| **NDPN** | Noise Degradation Processing Network | C_omega 置信度引导去噪 (不变) | Revisiting Temporal Alignment (CVPR 2022) |
| **MCPN** | Motion Compensation Processing Network | C_omega 运动强度补偿 (原 MRPN) | JFFRA (2025) |
| **CXG** | Cross-eXcitation Gate | deploy 重参数化 (原 CrossFusionGate) | DRNet (CVPR 2026) |
| **SGRF** | Stage-wise Guided Restoration & Fusion | 阶段式去噪→去模糊→提亮 (原 IGRF) | — |

### 命名变更总表

| 原缩写 | 新缩写 | 中文全名 |
|--------|--------|----------|
| DWT-LFF | **SWD** | 空域小波分流器 |
| TFSI | **TFDE** | 时频退化估计器 |
| SACE | **TCA** | 时序对应对齐 |
| IFPN | **ISPN** | 光照源处理网络 |
| NDPN | **NDPN** | 噪声退化处理网络 (不变) |
| MRPN | **MCPN** | 运动补偿处理网络 |
| IGRF | **SGRF** | 阶段式引导修复融合 |
| CrossFusionGate | **CXG** | 交叉激励门 |

### Mark1 核心诊断

| 指标 | 旧 Delta | Mark1 预期 |
|------|---------|-----------|
| ep5 loss | 0.226→0.436 (反弹) | 单调下降 |
| s_illum norm | 63~113 ❌ | ≈1.0 ✅ |
| 退 化分离 | 失效 | 有效 |

### 完整数据流

```
输入 x: (B, T=5, 3, H, W)
  │
  ├─→ Encoder → F_stack (B, T, 64, H, W)
  │
  ├─→ SWD (逐帧) → feat_tfde (B,T,C,H/2), feat_tca (B,T,C,H/2)
  │     ├─ [HaarDWT] → LL, LH, HL, HH
  │     ├─ [LL分流] alpha_net(LL) → α·LL→TFDE, (1-α)·LL+IN→TCA
  │     ├─ [HF分流] noise_gate×HF→TFDE, struct_gate×HF+LN→TCA
  │     └─ [proj+LN] Conv(4C→C)+LayerNorm → feat_tfde/feat_tca
  │
  ├─→ TFDE(feat_tfde) → s_illum, s_noise (upsample→H×W)
  │     ├─ s_illum → ISPN (唯一路径)
  │     └─ s_noise → NDPN (noise_proj 条件注入)
  │
  ├─→ TCA(feat_tca) → tca_out (B,T,C,H), C_omega_list, F_t_aligned, mu, sigma
  │     ├─ [MVC-Shift] 3 dilated DWConv (d=1,2,3)
  │     ├─ [SpatialWKV2D] 4方向 Bi-WKV 空间扫描
  │     ├─ [Channel Mix] → upsample → tca_out
  │     ├─ [TemporalCorrespondence] Q-K cosine → C_omega_list
  │     └─ [TemporalAggregation] C_omega warp + gate → F_t_aligned
  │
  ├─→ F_aligned_list = [tca_out[:, t] for t] → ISPN/NDPN/MCPN
  │
  ├─→ ISPN(aligned_feats, s_illum, F_t_aligned) → lit_up_map_raw, f_illum_feat, A_illu
  ├─→ NDPN(aligned_feats, s_noise, mu, sigma, C_omega, F_t_aligned) → f_noise_out
  ├─→ MCPN(aligned_feats, sigma, C_omega, F_t_aligned) → f_motion_out
  │
  ├─→ CXG(f_noise_out, f_motion_out) → f_noise_gated, f_motion_gated
  │
  └─→ SGRF(f_illum_feat, f_noise_gated, f_motion_gated, lit_up_map, img_center, A_illu)
        ├─ Stage1: img_s1 = denoise(f_noise_gated, img_center)  ★ s_noise 不在 SGRF
        ├─ Stage2: img_s2 = motion(f_motion_gated, img_s1)
        └─ Stage3: res_t = clamp(img_s2 × lit_up_map × (1+A_illu))
```

---

## 2. 核心模块详解

### 2.1 SWD — 空域小波分流器

**文件**: `models/modules/swd.py`

子带级分流（不做 inverse DWT），显式分离"光照+噪声"和"光照无关结构"。

```
Encoder feat → [HaarDWT] → LL, LH, HL, HH
  ├─ alpha_net(LL) → α ∈ (0,1)
  ├─ LL_tfde = α × LL
  ├─ LL_tca = IN((1-α) × LL)
  ├─ HF_energy = (LH²+HL²+HH²).mean(C)
  ├─ noise_gate(HF_energy) → n_gate ∈ (0,1)
  ├─ HF_tfde = n_gate × HF_cat
  ├─ HF_tca = LN((1-n_gate) × HF_cat)
  └─ proj(4C→C)+LN → feat_tfde, feat_tca (each B,T,C,H/2,W/2)
```

### 2.2 TCA — 时序对应对齐 (Bi-WKV)

**文件**: `models/modules/pure_rwkv_sace.py`

**空间扫描 Bi-WKV**:
```
k.clamp(-8,8), v.clamp(-8,8)
w = -F.softplus(spatial_decay)  → ew < 1 恒衰减
chunk-wise cumsum (CHUNK=256)  → 防长序列数值溢出
双向 (fwd+bwd)/2
```

**四方向扫描** (每方向独立 BiWKV + per-head LN):
```
水平(H×W→L), 垂直(W×H→L), 主对角线, 副对角线
Concat 4 heads → σ(R)⊙wkv → proj_out → post_norm
```

**输入**: feat_tca (B,T,C,H/2,W/2) — Mark1 无需内部降采样

### 2.3 数值稳定性

| 组件 | 措施 |
|------|------|
| BiWKV | `k.clamp(-8,8)`, `v.clamp(-8,8)` |
| BiWKV | `w = -F.softplus(spatial_decay)` → ew < 1 数学保证 |
| BiWKV | chunk-wise cumsum (CHUNK=256) |
| SpatialWKV2D | `pre_norm` LayerNorm 在 R/K/V 投影前 |
| SpatialWKV2D | R/K/V RWKV-7 风格小初始化 `±0.05~0.5/√C` |
| SWD | `proj_tfde/proj_tca` 后 LayerNorm → norm≈1 |
| Tau | `F.softplus(tau_raw) + 0.05` → 下界 0.05 |

---

## 3. 参数分布

| 模块 | 参数量 |
|------|--------|
| Encoder | ~320K |
| SWD | ~25K |
| TFDE | ~120K |
| TCA (MVCShift + WKV + Corr + Agg) | ~300K |
| ISPN (含 s_illum_proj + illu_conv + illu_anchor) | ~160K |
| NDPN (含 conf_proj + noise_extract + denoise_strength) | ~85K |
| MCPN (含 motion_estimator + comp_gate + motion_refine) | ~80K |
| CXG | ~8K |
| SGRF | ~120K |
| 其他 | ~70K |
| **总计** | **~1.69M** |

---

## 4. 版本演进

| 版本 | 关键改动 | 参数量 | 状态 |
|------|---------|--------|------|
| v5.9.2 | s_illum 复生 + IFPN 监督 | 1.14M | 20.39 PSNR |
| v6.5 | PureRWKV 移除 DAT | 1.17M | 20.36 |
| v6 Bravo | 损失重调 + DWT-LFF分裂 + V raw | 1.20M | 20.05 |
| v6 Charlie | 多帧 F.B. + concat fusion | 1.25M | — |
| v6 Charlie3 | s_illum→IFPN单路径 + CXG + LearnableScaleFusion | 1.32M | ep29 loss=0.176 |
| v6 Delta | 空间扫描2D-WKV + C_omega + F_t_aligned | 1.64M | 训练崩溃 (loss反弹) |
| **v6 Delta Mark1** | **SWD子带分流 + 命名统一 + 数值稳定** | **1.69M** | 训练中 |

---

## 5. 关键文件

| 文件 | 模块 |
|------|------|
| `models/modules/swd.py` | SWD (SpatialWaveletDiverter, HaarDWT2D) |
| `models/modules/pure_rwkv_sace.py` | TCA, BiWKV, SpatialWKV2D, MVCShift, TemporalCorrespondence, TemporalAggregation |
| `models/modules/tfsi.py` | TFDE (Temporal-Frequency Degradation Estimator) |
| `models/modules/ifpn.py` | ISPN (Illumination-Source Processing Network) |
| `models/modules/ndpn.py` | NDPN (Noise Degradation Processing Network) |
| `models/modules/mrpn.py` | MCPN (Motion Compensation Processing Network) |
| `models/modules/igrf.py` | SGRF (Stage-wise Guided Restoration & Fusion) |
| `models/tfs_net.py` | CXG, TFSNet (主入口, 数据流编排) |
| `configs/v6_bravo.yaml` | 训练配置 |
