# TFS-Net v6 Bravo 模型架构设计文档

> 日期：2026-06-28
> 版本：v6 Bravo（P0-1 损失重调 + P0-2 TFSI phase_conf + P1-2 DWT-LFF 分裂 + V raw）
> 训练配置：`configs/v6_bravo.yaml`，batch=2, lr=0.001, epochs=50, warmup=5
> 参数量：1.198M

---

## 1. 概述

v6 Bravo 在 v6.5 PureRWKV 基础上引入 Bravo Plan 的 P0+P1 改动：

| 改造 | 动机 | 来源 |
|---|---|---|
| **损失权重重调** | BVI-Lowlight 实证：L1 对错位敏感，VGG/SSIM 容忍度高 | BVI-Lowlight / TCE-Net / LAN |
| **TFSI phase_conf** | 低光相位 = 结构+运动+噪声，不能假设相位一致 | FDN / FourierDiff / FAN |
| **DWT-LFF 分裂** | VSRELL 中心帧锚定 + STCD 视角分解：Q/K 归一化空间对齐，V 原始空间融合 | VSRELL (CVPR 2026) / STCD (IJCAI 2025) |
| **PureRWKV 多尺度** | 移除 DAT 后纯 RWKV 替代 | pureRWKV.md (9 篇论文) |

### 完整数据流

```
输入 y (B, T, 3, H, W)
  │
  ├─ [Encoder] 3级金字塔 → F_stack (B, T, 64, H, W)
  │     ├─ F_t = F_stack[:, T//2] 中心帧 (用于 SACE V 源 + TFSI 输入)
  │     └─ F_Ω = F_stack[:, ≠T//2] 邻居帧
  │
  ├─ [DWT-LFF 分裂] (Bravo P1-2)
  │     ├─ lff_center(F_t) → F_lff_t, feat_tfsi    α init=0.6
  │     └─ lff_neighbor(F_Ω per-frame) → F_lff_Ω     α init=0.4
  │
  ├─ [TFSI] (Bravo P0-2: phase_conf)
  │     ├─ SpatialBranch: soft-中位值 → μ_t,σ_t,SNR → F_s
  │     ├─ FrequencyBranch: lff_center(中心帧) → feat_tfsi → F_f
  │     ├─ phase_conf_head(F_f) → phase_conf (相位置信度)
  │     └─ IntensityHead(F_s∥F_f∥phase_conf) → s_illum, s_noise
  │
  ├─ [SACE] PureRWKVSACE (v6.5)
  │     ├─ DWT-LFF: center→lff_center, neighbor→lff_neighbor
  │     ├─ 中心帧参考: lff_stack[center_idx] → μ_t_clean
  │     ├─ 多尺度双向 RWKV (full/half/quarter ×2)
  │     ├─ 边缘门控 (on encoder raw f_raw_center)
  │     └─ V source = encoder raw center (Bravo P1)
  │
  ├─ [IFPN] → lit_up_map, f_illum_feat, ifpn_side
  ├─ [NDPN] → f_noise_out
  ├─ [MRPN] → f_motion_out
  │
  └─ [IGRF] 逆序修复 → res_t
```

### 损失函数 (Bravo P0-1)

```
L_total = 0.3·Charbonnier(res, GT)    # ↓ 1.0→0.3
        + 0.8·VGG(relu3_3)           # ↑ 0.2→0.8 ★主导
        + 0.5·(1-SSIM)               # ↑ 0.1→0.5
        + 0.05·L_freq                # ↓ 0.1→0.05
        + 0.001·L_illum_smooth + 0.02·L_illum_sup
        + 0.2·L_inter + 0.1·L_ifpn_sup
```

### 训练超参

| 参数 | 值 | 说明 |
|---|---|---|
| batch_size | 2 | 24GB 约束 |
| epochs | 50 | 多尺度 RWKV 需要更长训练 |
| lr | 0.001 | 匹配 v5.9.2 成功值 |
| warmup | 5 | lff_neighbor 全新初始化 |
| weight_decay | 1e-4 | 标准 |
| grad_clip | 0.5 | 标准 |

### 参数分布

| 模块 | 参数量 |
|---|---|
| Encoder | ~320K |
| TFSI (含 phase_conf) | ~108K |
| PureRWKVSACE (3×RWKV+双DWT-LFF+edge_prompt) | ~320K |
| IFPN | ~150K |
| NDPN | ~70K |
| MRPN | ~50K |
| IGRF | ~130K |
| 其他 | ~50K |
| **总计** | **1.198M** |

### 设计原则

1. **中心帧锚定** (VSRELL CVPR 2026)：F_t 全局贯穿 Q/V/TFSI
2. **归一化-原始双空间分离** (STCD IJCAI 2025)：DWT-LFF 做对齐，Encoder 原始做内容融合
3. **相位不一致建模** (FDN/FourierDiff/FAN)：phase_conf 贯穿 s_noise
4. **对齐容忍损失主导** (BVI-Lowlight)：VGG/SSIM 主导，L1 降权
5. **纯 RWKV 对齐** (pureRWKV.md)：去 DAT 省 130K

---

## 2. 核心模块详解

### 2.1 PureRWKVSACE（纯 RWKV 多尺度帧间注意力）

**文件**: `models/modules/pure_rwkv_sace.py`

移除 DeformableCrossAttention，用 3 尺度双向 RWKV 替代。

**DWT-LFF 分裂** (Bravo P1-2):
```
中心帧 → lff_center (α init=0.6, 偏 LL_ref) → 干净锚定特征
邻居帧 → lff_neighbor (α init=0.4, 偏 LL_deg) → 退化诊断特征
```

**多尺度双向 RWKV**:
```
full:     VRWKVStyleSpatialMix(5帧, H×W)     ×2 (双向)
half:     avg_pool3d(LFF, (1,2,2)) → RWKV ×2 → bilinear 上采样
quarter:  avg_pool3d(LFF, (1,4,4)) → RWKV ×2 → bilinear 上采样
out = (full + half + quarter) / 3
```

**V 源 = Encoder 原始中心帧** (Bravo P1):
```
f_raw_center = feats[:, T//2]            # Encoder 原始, 非 DWT-LFF
edge_weight = edge_prompt(f_raw_center)   # 在原始特征上做边缘检测
F_aligned[t] = out[t] + (1-edge_weight)·f_raw_center + (1-s_noise)·f_raw_center
```

### 2.2 TFSI（Bravo P0-2: phase_conf）

**文件**: `models/modules/tfsi.py`

```
SpatialBranch(feats) → F_s
FrequencyBranch(中心帧) → lff_center → feat_tfsi → F_f
phase_conf = phase_conf_head(F_f)     # Conv→GELU→Conv→Sigmoid, (B,1,H,W)
F_fused = ConcatFusion(F_s, F_f)
s_illum, s_noise = IntensityHead(F_fused ∥ phase_conf)
s_noise *= (1 + 0.5·(1-phase_conf))   # 相位不可靠 → 增强去噪
```

---

## 3. 版本演进

| 版本 | 关键改动 | 参数量 | 最佳 PSNR |
|---|---|---|---|
| v5.5 | 基线 | 1.12M | 19.23 |
| v5.9.1 | fuse 修复 | 1.12M | 20.11 |
| v5.9.2 | s_illum 复生 + IFPN 监督 | 1.14M | 20.39 |
| v6.5 | PureRWKV 移除 DAT | 1.17M | 20.36 |
| **v6 Charlie** | **多帧 FrequencyBranch + 多尺度 concat + NDPN s_noise条件 + phase_conf方向修正** | **1.25M** | ✅ 训练中 | loss 0.28→0.25 正常下降, illum=0.08 活跃 |
