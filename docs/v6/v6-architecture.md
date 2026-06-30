# TFS-Net v6 Charlie 模型架构设计文档

> 日期：2026-06-29 (更新: Charlie P0/P1/P2 实施)
> 版本：v6 Charlie（Charlie-plan P0+P1+P2 完整实施）
> 训练配置：`configs/v6_charlie.yaml`，batch=2, lr=2e-4, epochs=50, warmup=5
> 参数量：1.25M

---

## 1. 概述

v6 Charlie 在 v6 Bravo 基础上实施 Charlie Plan 的全部 P0+P1+P2 改动：

| 优先级 | 改动 | 说明 | 状态 |
|--------|------|------|------|
| **P0-1** | **FrequencyBranch 多帧** | temporal_fuse Conv3d 融合邻帧时域信息 | ✅ 已实施 |
| **P0-2** | **s_noise → NDPN** | s_noise 从 IGRF 移除, 注入 NDPN 条件输入 (noise_proj) | ✅ 已实施 |
| **P1-1** | **多尺度 concat 融合** | SACE 用 concat+channel_mix 替代 /3 等权平均 | ✅ 已实施 |
| **P1-2** | **σ→MRPN** | σ_t_clean 输入 MRPN (blur_estimator 运动感知) | ✅ 已实施 |
| **P2-1** | **MRPN blur_mask** | σ_t_clean + 帧间差异 → blur_estimator 门控 | ✅ 已实施 |
| **P2-2** | **噪声图来源调制** | SACE sigma_sace → sigmoid 调制 s_noise 置信度 | ✅ 已实施 |

### 完整数据流 (Charlie)

```
输入 y (B, T, 3, H, W)
  │
  ├─ [Encoder] 3级金字塔 → F_stack (B, T, 64, H, W)
  │     ├─ F_t = F_stack[:, T//2] 中心帧 (用于 SACE V 源 + TFSI 输入)
  │     └─ F_Ω = F_stack[:, ≠T//2] 邻居帧
  │
  ├─ [DWT-LFF 分裂]
  │     ├─ lff_center(F_t) → feat_sace, feat_tfsi   α init=0.6
  │     └─ lff_neighbor(F_Ω per-frame) → feat_sace_Ω  α init=0.4
  │
  ├─ [TFSI] (Charlie P0-1: 多帧 FrequencyBranch)
  │     ├─ SpatialBranch: soft-median → μ_t,σ_t,SNR → F_s
  │     ├─ FrequencyBranch: temporal_fuse(中心+邻帧均值) → F_f
  │     ├─ phase_conf_head(F_f) → phase_conf
  │     └─ IntensityHead(F_s∥F_f∥phase_conf) → s_illum, s_noise
  │           └─ s_noise *= (1 - 0.3·(1-phase_conf))
  │
  ├─ [SACE] PureRWKVSACE
  │     ├─ DWT-LFF: center→lff_center, neighbor→lff_neighbor
  │     ├─ μ_t_clean = lff_stack[center], σ_t_clean = lff_stack.std
  │     ├─ 多尺度双向 RWKV → channel_mix (Charlie P1-1)
  │     ├─ 边缘门控 + s_noise 残差 (V = Encoder 原始中心帧)
  │     └─ 输出: F_aligned_list, μ_t_clean, σ_t_clean
  │
  ├─ [Charlie P2: σ 调制 s_noise]
  │     └─ sigma_sace = σ_t_clean.mean(dim=1)
  │        s_noise *= sigmoid(-sigma_sace × 2)  # 高运动 → 低噪声置信
  │
  ├─ [三源恢复] (SACE 对齐特征 → 三分支)
  │     ├─ IFPN(img_center↓, aligned_feats) → lit_up_map, f_illum_feat
  │     ├─ NDPN(aligned, μ_t_clean, σ_t_clean, s_noise) → f_noise_out
  │     │     └─ Charlie P0-2: noise_proj(s_noise) 条件注入
  │     └─ MRPN(aligned, σ_t_clean) → f_motion_out
  │           └─ Charlie P1+P2: blur_estimator(σ, frame_diff) 运动感知
  │
  └─ [IGRF] 修正去噪
        ├─ Stage1: f_noise + img_center → img_s1   ★ s_noise 已移除
        ├─ Stage2: f_motion + img_s1 → img_s2
        └─ Stage3: img_s2 × lit_up_map + s_illum·corr → res_t
```

### 损失函数 (Bravo2 P0)

```
L_total = 1.0·Charbonnier(res, GT)    # Charlie: 主导 (P0)
        + 0.04·VGG(relu3_3)           # ★ 0.04 (vs Bravo 0.8)
        + 0.2·(1-SSIM)               # ★ 0.2 (vs Bravo 0.5)
        + 0.05·L_freq
        + 0.001·L_illum_smooth + 0.02·L_illum_sup
        + 0.2·L_inter + 0.1·L_ifpn_sup
```

### 训练超参 (Charlie)

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 2 | 24GB 约束 |
| epochs | 50 | |
| lr | 2e-4 | ★ Charlie: 降低学习率 (blur_estimator 零初始化) |
| warmup | 5 | |
| weight_decay | 1e-4 | |
| grad_clip | 0.5 | |
| charlie_mode | true | ★ 激活 Charlie 数据流路径 |

### 参数分布

| 模块 | 参数量 | 新增 (vs Bravo) |
|------|--------|----------------|
| Encoder | ~320K | — |
| TFSI (含 temporal_fuse) | ~120K | +12K (Conv3d) |
| PureRWKVSACE (含 channel_mix) | ~325K | +5K (Linear 3C→C) |
| IFPN | ~150K | — |
| NDPN (含 noise_proj) | ~75K | +5K (Conv2d 1→64) |
| MRPN (含 blur_estimator) | ~60K | +10K (blur_est) |
| IGRF | ~130K | — |
| 其他 | ~70K | — |
| **总计** | **~1.25M** | **+32K vs Bravo 1.20M** |

### 设计原则

1. **中心帧锚定** (VSRELL CVPR 2026): F_t 全局贯穿 Q/V/TFSI
2. **归一化-原始双空间分离** (STCD IJCAI 2025): DWT-LFF 做对齐，Encoder 原始做内容融合
3. **相位不一致建模** (FDN/FourierDiff/FAN): phase_conf 贯穿 s_noise
4. **Charbonnier 主导损失** (Charlie P0): L1 1.0 主导，VGG 0.04 + SSIM 0.2 辅助
5. **纯 RWKV 对齐** (pureRWKV.md): 去 DAT 省 130K
6. **模块化数据流** (Charlie P0): s_noise→NDPN; s_illum→IGRF; σ→MRPN; 各司其职

---

## 2. 核心模块详解

### 2.1 PureRWKVSACE（纯 RWKV 多尺度帧间注意力）

**文件**: `models/modules/pure_rwkv_sace.py`

**DWT-LFF 分裂**:
```
中心帧 → lff_center (α init=0.6, 偏 LL_ref) → 干净锚定特征
邻居帧 → lff_neighbor (α init=0.4, 偏 LL_deg) → 退化诊断特征
```

**多尺度双向 RWKV + Charlie P1-1 fusion**:
```
full:     VRWKVStyleSpatialMix(5帧, H×W)     ×2 (双向)
half:     avg_pool3d(LFF, (1,2,2)) → RWKV ×2 → bilinear 上采样
quarter:  avg_pool3d(LFF, (1,4,4)) → RWKV ×2 → bilinear 上采样
out = channel_mix(concat[full, half, quarter])  ← Charlie: 可学习融合
```

**V 源 = Encoder 原始中心帧**:
```
f_raw_center = feats[:, T//2]
edge_weight = edge_prompt(f_raw_center)
F_aligned[t] = out[t] + (1-edge_weight)·f_raw_center + (1-s_noise)·f_raw_center
```

**输出 σ_t_clean → MRPN (Charlie P1)**:
```
σ_t_clean = lff_stack.std(dim=1, unbiased=False)
→ blur_estimator(σ_t_clean, frame_diff)  # 运动感知门控
```

### 2.2 TFSI (Charlie P0-1: 多帧 FrequencyBranch)

**文件**: `models/modules/tfsi.py`

```
SpatialBranch(feats) → F_s
FrequencyBranch: temporal_fuse(中心+邻帧均值) → F_f  ★ Charlie 多帧
phase_conf = phase_conf_head(F_f)
F_fused = ConcatFusion(F_s, F_f)
s_illum, s_noise = IntensityHead(F_fused ∥ phase_conf)
s_noise *= (1 - 0.3·(1-phase_conf))
```

### 2.3 NDPN (Charlie P0-2: s_noise 条件输入)

**文件**: `models/modules/ndpn.py`

```
SNR = |μ_t_clean| / (σ_t_clean + ε)
α_i = sigmoid(Conv(residual)) × (1 - s_SNR)  # 邻帧权重
α_t = s_SNR                                    # 中心帧权重
F_denoised = Σ w_i × F_i^aligned

★ Charlie P0-2:
noise_cond = noise_proj(s_noise)  # Conv2d(1→64), 零初始化
f_noise_out = F_denoised + noise_cond
```

### 2.4 MRPN (Charlie P1+P2: σ→MRPN + blur_mask)

**文件**: `models/modules/mrpn.py`

```
f_omega_aligned = window_corr(f_t, f_neighbors)

★ Charlie P1+P2:
sigma_1ch = σ_t_clean.mean(dim=1)              # (B,1,H,W)
frame_diff = |f_omega_aligned - f_t_aligned|   # 帧间差异
blur_mask = blur_estimator([sigma_1ch, frame_diff])  # 零初始化

g_t = sigmoid(gate([f_t, f_omega]))
g_t = g_t × (1-blur_mask) + blur_mask × 0.3  # 模糊区→邻帧补偿
f_fuse = g_t×f_t + (1-g_t)×f_omega
f_motion = Refine(f_fuse) + f_t
```

### 2.5 IGRF (Charlie P0: s_noise 已移除)

**文件**: `models/modules/igrf.py`

```
Stage1: img_s1 = clamp(img_center + δ(f_noise))     ★ s_noise 不再注入
Stage2: img_s2 = clamp(img_s1 + δ(f_motion))
Stage3: res_t = clamp(img_s2 × lit_up_map + s_illum × corr_mag)
```

---

## 3. 版本演进

| 版本 | 关键改动 | 参数量 | 最佳 PSNR |
|------|---------|--------|-----------|
| v5.5 | 基线 | 1.12M | 19.23 |
| v5.9.2 | s_illum 复生 + IFPN 监督 | 1.14M | 20.39 |
| v6.5 | PureRWKV 移除 DAT | 1.17M | 20.36 |
| v6 Bravo | 损失重调 + DWT-LFF分裂 + V raw | 1.20M | 20.05 |
| v6 Charlie | 多帧 F.B. + concat fusion + NDPN s_noise + σ→MRPN + blur_mask | 1.25M | 训练中 |
| v6 Charlie2 | D1 s_noise→NDPN only + D2 encoder feat→IFPN + D3 s_illum→IFPN gate + D4 VSRELL A_illu | 1.26M | ep7 loss=0.19（中断） |
| **v6 Charlie3** | **P0 s_illum→IFPN唯一路径 + P1 CrossFusionGate + P2 LearnableScaleFusion** | **1.32M** | ep29 loss=0.176（中断） |
| **v6 Delta** | **SACE 空间扫描 2D-WKV + MVCShift + C_omega_list + F_t_aligned + A_illu→IFPN** | **1.33M** | 训练中 |
