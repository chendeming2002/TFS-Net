# TFS-Net v6 Delta 整体架构图

## 图一：最简架构 (三源分离估计→三源退化并行建模→修正去噪)

```mermaid
flowchart TD
    subgraph 输入
        IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]
    end

    subgraph 诊断层["三源分离估计"]
        ENC["Encoder<br/>3级金字塔编码"]
        TFSI["TFSI 时频诊断<br/>多帧 temporal_fuse<br/>→ s_illum + s_noise"]
    end

    subgraph 对齐层["空间扫描 + 时序对应"]
        SACE["SACE Delta<br/>MVC-Shift + SpatialWKV2D<br/>4方向空间扫描<br/>→ sace_out (B,T,C,H,W)<br/>→ C_omega (时序对应矩阵)<br/>→ F_t_aligned (对齐锚)"]
    end

    subgraph 处理层["三源退化并行建模"]
        IFPN["IFPN<br/>光照图估计<br/>+ s_illum_proj<br/>→ A_illu"]
        NDPN["NDPN<br/>SNR去噪<br/>+ C_omega conf<br/>+ s_noise 条件"]
        MRPN["MRPN<br/>运动补偿<br/>+ C_omega motion<br/>+ sigma blur_mask"]
    end

    subgraph 融合层
        CXF["CrossFusionGate<br/>noise-motion 交叉门控<br/>训练: 动态 / 推理: 静态重参数化"]
    end

    subgraph 执行层["修正去噪"]
        IGRF["IGRF<br/>去噪→去模糊→提亮<br/>A_illu (IFPN生成, 替代s_illum注入)"]
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC --> TFSI
    ENC --> SACE
    TFSI -- "s_illum" --> IFPN
    TFSI -- "s_noise" --> NDPN

    SACE -- "sace_out (F_aligned)" --> IFPN
    SACE -- "F_t_aligned, C_omega" --> NDPN
    SACE -- "F_t_aligned, C_omega, sigma" --> MRPN

    IFPN -- "lit_up_map, f_illum, A_illu" --> IGRF
    NDPN -- "f_noise" --> CXF
    MRPN -- "f_motion" --> CXF
    CXF -- "cross-modulated" --> IGRF

    IGRF --> OUT
    IN -- "img_center" --> IGRF
    ENC -- "feat_to_img" --> IFPN
```

### 框架要点

- **三源分离估计**：Encoder → TFSI (s_illum/s_noise)；SACE 统计量 (σ_t_clean)
- **TFSI ↔ SACE 关系**：TFSI 进行时频域诊断 (temporal_fuse + FFT)，SACE 进行空域扫描 (MVC-Shift + 4方向 WKV)，两者平行独立，在 Encoder 输出处分叉。s_noise 仅注入 NDPN，不再传入 IGRF
- **SACE 内部**：MVC-Shift(多尺度空洞DWConv) → SpatialWKV2D(4方向空间扫描) → Channel Mix → TemporalCorrespondence(时序对应矩阵) → TemporalAggregation(时序对齐聚合)
- **C_omega_list**：Delta 核心创新 — 中心帧与邻帧的空间 cosine similarity 矩阵，同时注入 NDPN(conf_map) 和 MRPN(motion_mag) 作为置信度参考
- **A_illu**：s_illum 经 IFPN s_illum_proj (零初始化) → 最终 A_illu 传入 IGRF，替代旧的 s_illum 直接注入

---

## 图二：带细节架构

```mermaid
flowchart TD
    IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]

    subgraph Encoder["Encoder 编码器"]
        ENC["PyramidEncoder<br/>[32,64,96] → 64ch<br/>逐帧编码 → F_stack"]
    end

    subgraph TFSI["TFSI 时频诊断"]
        direction TB
        TFSI_SPATIAL["SpatialBranch<br/>soft-median → mu,sigma,SNR → F_s"]
        TFSI_FREQ["FrequencyBranch 多帧<br/>temporal_fuse(中心+邻帧均值)→F_f"]
        TFSI_PHASE["phase_conf_head → phase_conf"]
        TFSI_HEAD["IntensityHead(F_s,F_f,phase_conf)<br/>→ s_illum, s_noise<br/>s_noise*=(1-0.3(1-phase_conf))"]
        TFSI_SPATIAL --> TFSI_HEAD
        TFSI_FREQ --> TFSI_PHASE
        TFSI_FREQ --> TFSI_HEAD
        TFSI_PHASE --> TFSI_HEAD
    end

    subgraph SACE["SACE Delta: 空间扫描 + 时序对应"]
        direction TB
        SACE_DS["特征降采样 H/2 x W/2"]

        subgraph SACE_SPATIAL["帧内空间处理 (RWKV注意力)"]
            MVC["MVC-Shift<br/>3分支空洞DWConv(d=1,2,3)<br/>多尺度上下文偏移"]
            WKV2D["SpatialWKV2D<br/>4方向扫描: 水平/垂直/主对/副对<br/>Bi-WKV cumsum O(LxC)<br/>recep gate + post_norm"]
            CHMIX["Channel Mix<br/>LN->Conv(1->4C)->GELU->Conv(4C->C)<br/>+ spatial_gamma residual"]
            MVC --> WKV2D --> CHMIX
        end

        SACE_UP["上采样 HxW → sace_out (B,T,C,H,W)"]
        SACE_STATS["mu_t_clean, sigma_t_clean"]

        subgraph SACE_TEMPORAL["时序对应 Delta"]
            TCORR["TemporalCorrespondence<br/>proj_qk → cosine similarity / tau<br/>→ C_omega_list (T-1)x(B,N,N)"]
            TAGG["TemporalAggregation<br/>C_omega x neighbor → warp<br/>frame_gate → softmax加权<br/>up+residual+LN → F_t_aligned"]
            TCORR --> TAGG
        end

        SACE_DS --> SACE_SPATIAL
        SACE_SPATIAL --> SACE_UP
        SACE_UP --> SACE_STATS
        SACE_DOWNSAMPLED["降采样特征 HdxWd"] --> TCORR
        SACE_UP --> TCORR
        TAGG --> SACE_UP
    end

    subgraph 三源处理["三源退化并行建模"]
        direction LR
        subgraph IFPN["IFPN 光照"]
            IFPN_PROJ["s_illum_proj zero-init<br/>Conv(1->64->C_c) → coarse注入"]
            IFPN_A["→ lit_up_map_raw<br/>→ f_illum_feat<br/>→ A_illu (光照先验输出)"]
            IFPN_PROJ --> IFPN_A
        end
        subgraph NDPN["NDPN 去噪"]
            NDPN_SNR["SNR = mu/sigma → s_snr<br/>+ conf_proj(C_omega diag) → conf_map"]
            NDPN_NOISE["★ noise_extract<br/>feat(center) vs F_t_aligned差异<br/>→ noise_feat"]
            NDPN_STR["★ denoise_strength<br/>noise_feat + conf_map → strength<br/>→ f_enc - noise × strength × γ<br/>+ s_noise条件注入"]
            NDPN_SNR --> NDPN_NOISE
            NDPN_NOISE --> NDPN_STR
        end
        subgraph MRPN["MRPN 运动"]
            MRPN_CORR["窗口相关聚合邻帧<br/>+ F_t_aligned 参考"]
            MRPN_MOTION["★ motion_estimator<br/>C_omega diag → Conv → motion_mag"]
            MRPN_BLUR["blur_estimator(sigma,frame_diff)<br/>→ blur_mask"]
            MRPN_COMP["★ comp_gate + motion_refine<br/>motion_delta × comp × γ<br/>+ gate×f_center + (1-gate)×f_omega"]
            MRPN_CORR --> MRPN_MOTION
            MRPN_MOTION --> MRPN_BLUR
            MRPN_BLUR --> MRPN_COMP
        end
    end

    subgraph CXF["CrossFusionGate 交叉门控"]
        CXF_NM["gate_noise(f_motion) → f_noise_mod<br/>gate_motion(f_noise) → f_motion_mod<br/>训练:动态 / 推理:静态重参数化"]
    end

    subgraph IGRF["IGRF 修正去噪"]
        IGRF_S1["Stage1: f_noise + img_center<br/>→ img_s1"]
        IGRF_S2["Stage2: f_motion + img_s1<br/>→ img_s2"]
        IGRF_S3["Stage3: img_s2 x lit_up_map x (1+A_illu)<br/>→ res_t"]
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC --> TFSI_SPATIAL
    ENC --> TFSI_FREQ
    ENC --> SACE_DS
    TFSI_HEAD -- "s_illum" --> IFPN_PROJ
    TFSI_HEAD -- "s_noise" --> NDPN_STR
    SACE_UP -- "aligned_feats" --> IFPN
    SACE_TEMPORAL -- "F_t_aligned, C_omega" --> NDPN_SNR
    SACE_TEMPORAL -- "F_t_aligned, C_omega" --> MRPN_CORR
    SACE_STATS -- "sigma_t_clean" --> MRPN_BLUR
    IFPN_A -- "lit_up_map, f_illum, A_illu" --> IGRF_S3
    NDPN_STR -- "f_noise_out" --> CXF_NM
    MRPN_COMP -- "f_motion_out" --> CXF_NM
    CXF_NM -- "f_noise, f_motion" --> IGRF_S1
    CXF_NM -- "f_motion" --> IGRF_S2
    IN -- "image_center" --> IGRF_S1
    ENC -- "feat_to_img → image_down" --> IFPN
    IGRF_S1 --> IGRF_S2 --> IGRF_S3 --> OUT
```

### SACE 纯 RWKV 空间扫描详解 (Delta)

```
feats (B,T,C,H,W) from Encoder (no DWT-LFF!)
  │
  ├─ [Downsample] H/2 × W/2 → feats_ds (B,T,C,Hd,Wd)
  │
  ├─ [MVC-Shift] 逐帧处理
  │     3分支空洞DWConv(d=1,2,3) + 1x1混频 → x_shifted
  │
  ├─ [SpatialWKV2D] 4方向空间扫描
  │     ├─ Split C→4 heads × C/4
  │     ├─ Head0: horizontal scan (H×W → L sequence)
  │     ├─ Head1: vertical scan (W×H → L sequence)
  │     ├─ Head2: main-diagonal scan
  │     ├─ Head3: anti-diagonal scan
  │     ├─ Per-direction: Bi-WKV cumsum O(L×C)
  │     │     S_t = cumsum(exp(w)^(t-i)·k_i·v_i)
  │     │     y = (u·k·v + S) / (u·k + cumsum_k + eps)
  │     └─ Concat 4 heads → recep_gate(sigmoid(r)*wkv) → post_norm
  │
  ├─ [Channel Mix] LN → Conv(1→4C) → GELU → Conv(4C→C) → + x * gamma
  │
  ├─ [Upsample] H×W → sace_out (B,T,C,H,W)
  │
  ├─ [TemporalCorrespondence] Delta 新增
  │     proj_qk (C→C/4) → cosine similarity / tau_learnable
  │     → C_omega_list: [(B, ds², ds²) × (T-1)]
  │     注入 NDPN (conf_map) + MRPN (motion_mag)
  │
  └─ [TemporalAggregation] Delta 新增
        C_omega × neighbor_ds → warp per frame
        frame_gate → softmax权重 → 加权聚合
        up + residual + LayerNorm → F_t_aligned (B,C,H,W)
        注入 NDPN/MRPN 作为对齐参考锚
```

### Delta vs Charlie3 核心差异

| 维度 | Charlie3 | Delta |
|------|----------|-------|
| **SACE 输入** | DWT-LFF 归一化特征 | Encoder 原始特征 (无 DWT-LFF) |
| **Token Shift** | Q-Shift (固定位移) | MVC-Shift (可学习空洞DWConv) |
| **扫描方式** | Bi-WKV 帧间 (T维) | **SpatialWKV2D** 帧内4方向 (HW维) |
| **时序对齐** | 隐式 (RWKV mix) | **显式 C_omega + F_t_aligned** |
| **光照注入** | s_illum → IGRF 直接 | **s_illum → IFPN → A_illu → IGRF** |
| **交叉门控** | CrossFusionGate | + deploy 模式 (重参数化) |
| **多尺度融合** | concat+channel_mix | 移除 (MVC-Shift 承担多尺度) |
