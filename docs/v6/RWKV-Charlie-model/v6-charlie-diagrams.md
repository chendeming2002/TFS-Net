# TFS-Net v6 Charlie3 整体架构图 (2026-06-30, 更新: Charlie3 P0-P2 实施)

## 图一：最简架构（三源分离估计-三源处理-修正去噪）

```mermaid
flowchart TD
    subgraph 输入
        IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]
    end

    subgraph 诊断层["三源分离估计"]
        ENC["Encoder<br/>3级金字塔编码"]
        DWT["DWT-LFF<br/>小波光照分离<br/>中心 α=0.6 / 邻居 α=0.4"]
        TFSI["TFSI<br/>时频诊断<br/>多帧 FrequencyBranch + phase_conf"]
    end

    subgraph 对齐层
        SACE["SACE<br/>纯 RWKV 帧间注意力<br/>多尺度双向 Bi-WKV<br/>★ P2: LearnableScaleFusion<br/>V 源 = Encoder 原始中心帧<br/>s_noise 已移除"]
    end

    subgraph 处理层["三源退化并行建模"]
        IFPN["IFPN<br/>光照图估计<br/>★ P0: s_illum 唯一路径 + s_illum_proj 先验注入<br/>★ encoder 特征输入"]
        NDPN["NDPN<br/>SNR自适应去噪<br/>s_noise 条件输入"]
        MRPN["MRPN<br/>运动补偿<br/>σ→MRPN 运动感知"]
        CFG["★ P1: CrossFusionGate<br/>噪声↔运动 交叉门控"]
    end

    subgraph 执行层["修正去噪"]
        IGRF["IGRF<br/>去噪→去模糊→提亮<br/>★ P0: s_illum 已移除 — 光照经 IFPN 唯一出口"]
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC --> DWT
    ENC -- "F_t (中心帧原始)" --> SACE
    ENC -- "encoder 特征" --> IFPN
    DWT --> TFSI
    DWT --> SACE
    TFSI -- "s_illum ★ P0: 唯一路径 →" --> IFPN
    TFSI -- "s_noise" --> NDPN

    SACE -- "F_aligned" --> IFPN
    SACE -- "F_aligned, μ_t_clean, σ_t_clean" --> NDPN
    SACE -- "F_aligned, σ_t_clean" --> MRPN

    IFPN -- "lit_up_map + f_illum_feat" --> IGRF
    NDPN -- "f_noise" --> CFG
    MRPN -- "f_motion" --> CFG
    CFG -- "f_noise_gated" --> IGRF
    CFG -- "f_motion_gated" --> IGRF

    IGRF --> OUT
    IN -- "img_center" --> IGRF
```

### 框架要点

- **三源分离估计**：DWT-LFF + TFSI 联合诊断光照(s_illum)和噪声(s_noise)；SACE 通过帧间方差(σ_t_clean)隐式感知运动
- **s_illum 唯一路径 (P0)**：TFSI → IFPN（s_illum_proj 零初始化先验注入）→ IGRF（f_illum_feat 已蕴含光照信息）
- **CrossFusionGate (P1)**：NDPN/MRPN 各自独立推理后，交叉门控交换互补置信度（运动剧烈去噪低、高噪运动补偿低）
- **LearnableScaleFusion (P2)**：SACE 三尺度 softmax 加权融合，替代 concat+channel_mix（12K→1.5K 参数）

### Charlie3 vs Charlie2 核心差异

| 维度 | Charlie2 | Charlie3 |
|------|----------|----------|
| **s_illum 路径** | TFSI → IFPN + TFSI → IGRF 双路径 | ★ P0: TFSI → IFPN 唯一路径 |
| **IFPN 光照注入** | illumin_modulate × s_illum (对齐特征调制) | ★ P0: s_illum_proj (零初始化, coarse 级加法注入) |
| **IGRF Brighten** | 接收 s_illum 做 A_illu 门控 | ★ P0: 移除 s_illum, f_illum_feat 自含光照 |
| **NDPN↔MRPN 关系** | 无交互，各自直送 IGRF | ★ P1: CrossFusionGate 交叉门控 |
| **SACE 多尺度融合** | concat[3C]→Linear(3C→C) (~12K) | ★ P2: LearnableScaleFusion (~1.5K) |
| **参数量** | 1.26M | ★ 1.32M (+59K) |

---

## 图二：带细节架构

```mermaid
flowchart TD
    IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]

    subgraph Encoder
        ENC["PyramidEncoder<br/>[32,64,96] → 64ch<br/>逐帧编码 → F_stack"]
    end

    subgraph DWT["DWT-LFF 分裂"]
        DWT_C["lff_center<br/>中心帧, α init=0.6<br/>→ F_lff_t + feat_tfsi"]
        DWT_N["lff_neighbor<br/>邻居帧, α init=0.4<br/>→ F_lff_Ω"]
    end

    subgraph TFSI["TFSI 时频诊断"]
        direction TB
        TFSI_SPATIAL["SpatialBranch<br/>soft-median → μ,σ,SNR → Conv → F_s"]
        TFSI_FREQ["FrequencyBranch ★多帧<br/>lff_center(中心) + 邻帧时域均值<br/>→ temporal_fuse → F_f"]
        TFSI_PHASE["phase_conf_head<br/>→ phase_conf (B,1,H,W)"]
        TFSI_HEAD["IntensityHead<br/>Conv(F_s||F_f||phase_conf)<br/>→ s_illum, s_noise<br/>s_noise×(1-0.3(1-phase_conf))"]
        TFSI_SPATIAL --> TFSI_HEAD
        TFSI_FREQ --> TFSI_PHASE
        TFSI_FREQ --> TFSI_HEAD
        TFSI_PHASE --> TFSI_HEAD
    end

    subgraph SACE["SACE 纯 RWKV 帧间注意力"]
        direction TB
        SACE_DWT["DWT-LFF 逐帧<br/>center→lff_center<br/>neighbor→lff_neighbor"]
        SACE_REF["μ_t_clean = lff_stack[center]<br/>σ_t_clean = lff_stack.std<br/>→ NDPN (SNR基线) + MRPN (运动感知)"]

        subgraph SACE_RWKV["多尺度 Bi-RWKV 帧间注意力"]
            RWKV_F["full: VRWKV ×2 (双向)"]
            RWKV_H["half: avg_pool3d(×2) → VRWKV ×2 → bilinear↑"]
            RWKV_Q["quarter: avg_pool3d(×4) → VRWKV ×2 → bilinear↑"]
            RWKV_FUSE["★ P2: LearnableScaleFusion<br/>softmax 权 + 逐尺度 SE 校准<br/>→ 可学习加权融合 (1.5K params)"]
        end

        SACE_GATE["边缘门控 + V raw<br/>edge_weight = edge_prompt(f_raw_center)<br/>F_aligned[t] = out[t] + f_raw_center<br/>s_noise 已移除 — 噪声仅去 NDPN"]
    end

    subgraph 三源处理["三源退化并行建模"]
        direction LR
        subgraph IFPN["IFPN 光照图估计 ★P0"]
            IFPN_ILLUM["★ s_illum_proj (零初始化)<br/>Conv2d(1→64→32)<br/>→ 注入 coarse_adapter 输出"]
            IFPN_COARSE["coarse_adapter → + illum_prior<br/>→ IllumExtract×T → L_ratio<br/>→ lit_up_map_raw [1,5]<br/>→ side_head → f_illum_feat"]
        end
        subgraph NDPN["NDPN 去噪"]
            NDPN_SNR["SNR估计 (|μ|/σ)<br/>→ 双因素聚合权重"]
            NDPN_COND["s_noise 条件输入 (noise_proj)<br/>Conv2d(1→64) → 加法注入"]
            NDPN_SNR --> NDPN_COND
        end
        subgraph MRPN["MRPN 运动"]
            MRPN_CORR["窗口相关 → f_omega_aligned"]
            MRPN_BLUR["blur_estimator<br/>σ_t_clean + 帧间差异 → blur_mask"]
            MRPN_GATE["门控融合<br/>g_t × blur_mask 调制<br/>→ Refine → f_motion"]
            MRPN_CORR --> MRPN_BLUR
            MRPN_BLUR --> MRPN_GATE
        end
        subgraph CFG["★ P1: CrossFusionGate"]
            CFG_NOISE["gate_noise(f_motion)<br/>运动剧烈 → 降低去噪"]
            CFG_MOTION["gate_motion(f_noise)<br/>高噪声 → 降低运动补偿"]
        end
    end

    subgraph IGRF["IGRF 修正去噪 ★P0"]
        IGRF_S1["Stage1: δ=Fuse(f_noise_gated, img)<br/>img_s1=clamp(img+δ)"]
        IGRF_S2["Stage2: δ=Fuse(f_motion_gated, img_s1)<br/>img_s2=clamp(img_s1+δ)"]
        IGRF_S3["Stage3: ★ s_illum 已移除<br/>A_illu = SigConv(f_illum_feat)<br/>res_t=clamp(img_s2×lit_up_map×(1+A_illu))"]
    end

    OUT["输出 res_t"]

    %% ── 数据流 ──
    IN --> ENC
    ENC --> DWT_C
    ENC --> DWT_N
    ENC -- "F_t (中心帧原始)" --> SACE_GATE
    ENC -- "encoder 特征" --> IFPN_COARSE
    DWT_C -- "F_lff_t, feat_tfsi" --> TFSI_FREQ
    DWT_N --> SACE_DWT
    DWT_C --> SACE_DWT
    TFSI_HEAD -- "s_illum ★ P0: 唯一路径 →" --> IFPN_ILLUM
    TFSI_HEAD -- "s_noise" --> NDPN_COND
    SACE_REF -- "σ_t_clean" --> MRPN_BLUR
    SACE_REF -- "μ_t_clean" --> NDPN_SNR
    SACE_GATE -- "F_aligned_list" --> IFPN_COARSE
    SACE_GATE -- "F_aligned_list" --> NDPN_COND
    SACE_GATE -- "F_aligned_list" --> MRPN_CORR
    IFPN_COARSE -- "lit_up_map_raw" --> IGRF_S3
    IFPN_COARSE -- "f_illum_feat" --> IGRF_S3
    NDPN_COND -- "f_noise" --> CFG_NOISE
    MRPN_GATE -- "f_motion" --> CFG_MOTION
    CFG_NOISE -- "f_noise_gated" --> IGRF_S1
    CFG_MOTION -- "f_motion_gated" --> IGRF_S2
    IN -- "image_center" --> IGRF_S1
    IGRF_S1 --> IGRF_S2 --> IGRF_S3 --> OUT
```

### Charlie3 数据流总览

```
TFSI → s_noise ──→ NDPN (noise_proj 条件输入)        — 保留
TFSI → s_illum ──→ IFPN (s_illum_proj ★P0 唯一出口)  — 新增, 替代双路径
TFSI → s_illum ─→ IGRF                              — ★P0: 已移除

SACE → σ_t_clean ─┬─ NDPN SNR 估计 (|μ|/σ)          — 保留
                   └─ blur_estimator → MRPN           — 保留

NDPN → f_noise ─┬─ CrossFusionGate.gate_motion ★P1   — 新增
MRPN → f_motion ─┼─ CrossFusionGate.gate_noise ★P1   — 新增
CFG → f_noise_gated, f_motion_gated → IGRF            — 替代直连

SACE 三尺度 → LearnableScaleFusion ★P2               — 替代 channel_mix
```

### 版本演进 (Charlie 系列)

| 版本 | 关键改动 | 参数量 | 收敛状态 |
|------|---------|--------|----------|
| Charlie | 多帧 FrequencyBranch + concat fusion + NDPN s_noise + σ→MRPN + blur_mask | 1.25M | ep1 loss=0.205 |
| Charlie2 | s_noise→NDPN only + encoder feat→IFPN + s_illum gate + VSRELL A_illu | 1.26M | 微不稳定 (loss 反升) |
| **Charlie3** | **s_illum 单路径 + CrossFusionGate + LearnableScaleFusion** | **1.32M** | 单调下降 (ep16 loss=0.183) |
