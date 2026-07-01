# TFS-Net v6 Delta Mark1 整体架构图 (2026-07-01)

## 图一：最简架构

```mermaid
flowchart TD
    subgraph 输入
        IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]
    end

    subgraph 编码分流["编码 + 小波分流"]
        ENC["Encoder<br/>3级金字塔编码 → F_stack"]
        SWD["SWD 空域小波分流器<br/>HaarDWT → alpha(LL) + noise_gate(HF)<br/>→ feat_tfde (光照+噪声)<br/>→ feat_tca (光照无关+结构)"]
    end

    subgraph 诊断["退化估计"]
        TFDE["TFDE 时频退化估计器<br/>SpatialBranch + FrequencyBranch<br/>→ s_illum + s_noise"]
    end

    subgraph 对齐["时序对应对齐"]
        TCA["TCA<br/>MVC-Shift + SpatialWKV2D<br/>4方向空间扫描 Bi-WKV<br/>→ tca_out (B,T,C,H,W)<br/>→ C_omega (时序对应矩阵)<br/>→ F_t_aligned (对齐锚)"]
    end

    subgraph 处理层["三源退化并行建模"]
        ISPN["ISPN 光照源处理<br/>s_illum_proj + F_t_aligned锚定<br/>→ lit_up_map + A_illu"]
        NDPN["NDPN 噪声退化处理<br/>C_omega conf + s_noise条件"]
        MCPN["MCPN 运动补偿处理<br/>C_omega motion + sigma"]
    end

    subgraph 融合层
        CXG["CXG 交叉激励门<br/>训练:动态 / 推理:静态重参数化"]
    end

    subgraph 执行层["阶段式修复融合"]
        SGRF["SGRF<br/>S1:去噪 S2:去模糊 S3:提亮<br/>A_illu (ISPN生成)"]
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC --> SWD
    SWD -- "feat_tfde (H/2)" --> TFDE
    SWD -- "feat_tca (H/2)" --> TCA
    TFDE -- "s_illum ↑H" --> ISPN
    TFDE -- "s_noise ↑H" --> NDPN

    TCA -- "F_aligned_list" --> ISPN
    TCA -- "F_aligned, C_omega, mu, sigma" --> NDPN
    TCA -- "F_aligned, C_omega, sigma" --> MCPN

    ISPN -- "lit_up_map, A_illu" --> SGRF
    NDPN -- "f_noise" --> CXG
    MCPN -- "f_motion" --> CXG
    CXG -- "f_noise, f_motion" --> SGRF

    IN -- "img_center" --> SGRF
    SGRF --> OUT
```

### 框架要点

- **SWD 小波分流**：Mark1 核心 — Encoder 特征经 HaarDWT 在子带级分离：alpha(LL) 分配光照给 TFDE/TCA，noise_gate(HF) 区分噪声/结构。输出 H/2 分辨率 + LayerNorm，TFDE 工作在正常 norm 范围
- **TFDE ← TCA**：TFDE 接收 feat_tfde（含光照+噪声信号），TCA 接收 feat_tca（光照无关结构），两路真正分流
- **TCA 空间扫描**：输入 SWD 的 H/2 特征，无需内部再降采样。MVC-Shift → 4方向 Bi-WKV → Channel Mix → upsample H
- **C_omega_list**：中心帧与邻帧的 cosine similarity 矩阵，注入 NDPN (conf_map) 和 MCPN (motion_mag) 作为置信度参考
- **命名统一**：TFDE/TCA/ISPN/NDPN/MCPN/CXG/SGRF — 各模块唯一缩写，语义清晰

### Mark1 vs Delta 核心差异

| 维度 | Delta | Mark1 |
|------|-------|-------|
| **Encoder→下游** | Encoder 直连 TFSI/SACE | **SWD 子带分流** → TFDE(光照+噪声) / TCA(结构) |
| **TFDE 输入** | 全分辨率 H×W, norm=60~113 | **H/2 子带, LayerNorm, norm≈1** |
| **TCA 输入** | Encoder 原始特征 → 内部 H/2 降采样 | **SWD feat_tca (H/2 已降采样)** |
| **IDWT** | 是 (DWT-LFF inverse 重建) | **否** (子带级直接输出) |
| **模块命名** | TFSI/SACE/IFPN/MRPN/IGRF | **TFDE/TCA/ISPN/MCPN/SGRF/CXG** |

---

## 图二：带细节架构

```mermaid
flowchart TD
    IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]

    subgraph Encoder["Encoder"]
        ENC["PyramidEncoder<br/>[32,64,96] → 64ch<br/>→ F_stack (B,T,64,H,W)"]
    end

    subgraph SWD["SWD 空域小波分流器"]
        direction TB
        DWT["HaarDWT2D<br/>→ LL, LH, HL, HH (H/2×W/2)"]
        ALPHA["alpha_net(LL)<br/>DWConv3x3 → GELU → Conv1x1 → Sigmoid<br/>α ∈ (0,1) — 光照分配"]
        LL_DIV["LL分流: α·LL → TFDE<br/>(1-α)·LL + InstanceNorm → TCA"]
        HF_ENERGY["HF能量: (LH²+HL²+HH²).mean(C) → (1,H/2,W/2)"]
        NOISE_G["noise_gate(HF_energy)<br/>Conv→Sigmoid<br/>高能量=噪声"]
        HF_DIV["HF分流: n_gate×HF_cat → TFDE<br/>(1-n_gate)×HF_cat + LN → TCA"]
        PROJ["proj_tfde/proj_tca: Conv(4C→C)+GELU+LN<br/>→ feat_tfde, feat_tca"]
        DWT --> ALPHA --> LL_DIV
        DWT --> HF_ENERGY --> NOISE_G --> HF_DIV
        LL_DIV --> PROJ
        HF_DIV --> PROJ
    end

    subgraph TFDE["TFDE 时频退化估计器"]
        direction TB
        TFDE_SP["SpatialBranch<br/>soft-median → μ,σ,SNR → F_s"]
        TFDE_FQ["FrequencyBranch<br/>temporal_fuse → F_f"]
        TFDE_PH["phase_conf_head → phase_conf"]
        TFDE_HD["IntensityHead(F_s,F_f,phase_conf)<br/>→ s_illum, s_noise"]
        TFDE_SP --> TFDE_HD
        TFDE_FQ --> TFDE_PH --> TFDE_HD
    end

    subgraph TCA["TCA 时序对应对齐 (Mark1:SWD输入)"]
        direction TB
        TCA_SP["帧内空间处理<br/>MVC-Shift(3 dilated DWConv)<br/>→ SpatialWKV2D(4方向 Bi-WKV cumsum)<br/>→ Channel Mix + residual"]
        TCA_UP["上采样 → tca_out (B,T,C,H,W)"]

        subgraph TCA_TEMP["时序对应"]
            TCORR["TemporalCorrespondence<br/>Q-K cosine / softplus tau<br/>→ C_omega_list (T-1)×(B,N,N)"]
            TAGG["TemporalAggregation<br/>C_omega warp + frame_gate<br/>→ upsample → F_t_aligned"]
            TCORR --> TAGG
        end

        TCA_SP --> TCA_UP
        TCA_SP --> TCORR
    end

    subgraph 三源处理["三源退化并行建模"]
        direction LR
        subgraph ISPN["ISPN 光照源处理"]
            ISPN_P["s_illum_proj(zeroinit) → coarse注入<br/>+ F_t_aligned → illu_anchor.gate<br/>→ lit_up_map_raw + f_illum_feat + A_illu"]
        end
        subgraph NDPN["NDPN 噪声退化处理"]
            NDPN_S["SNR + conf_proj(C_omega diag) → conf_map<br/>noise_extract(enc vs F_t_aligned)<br/>→ denoise_strength → f_noise"]
        end
        subgraph MCPN["MCPN 运动补偿处理"]
            MCPN_M["motion_estimator(C_omega diag)<br/>→ motion_mag<br/>blur_estimator(sigma, diff) + comp_gate<br/>→ f_motion"]
        end
    end

    subgraph CXG["CXG 交叉激励门"]
        CXG_G["gate_noise(f_motion) → f_noise_gated<br/>gate_motion(f_noise) → f_motion_gated<br/>deploy模式: 静态scale参数"]
    end

    subgraph SGRF["SGRF 阶段式修复融合"]
        SGRF_S1["S1: f_noise + img_center → img_s1"]
        SGRF_S2["S2: f_motion + img_s1 → img_s2"]
        SGRF_S3["S3: img_s2 × lit_up_map × (1+A_illu) → res_t"]
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC --> DWT
    PROJ -- "feat_tfde (H/2)" --> TFDE_SP
    PROJ -- "feat_tfde" --> TFDE_FQ
    PROJ -- "feat_tca (H/2)" --> TCA_SP
    TFDE_HD -- "s_illum ↑H" --> ISPN_P
    TFDE_HD -- "s_noise ↑H" --> NDPN_S
    TCA_UP -- "F_aligned" --> ISPN_P
    TCA_UP -- "F_aligned" --> NDPN_S
    TCA_UP -- "F_aligned" --> MCPN_M
    TCA_TEMP -- "F_t_aligned" --> ISPN_P
    TCA_TEMP -- "C_omega, F_t_aligned" --> NDPN_S
    TCA_TEMP -- "C_omega, F_t_aligned" --> MCPN_M
    ISPN_P -- "lit_up_map, A_illu" --> SGRF_S3
    NDPN_S -- "f_noise" --> CXG_G
    MCPN_M -- "f_motion" --> CXG_G
    CXG_G -- "f_noise_gated" --> SGRF_S1
    CXG_G -- "f_motion_gated" --> SGRF_S2
    IN -- "img_center" --> SGRF_S1
    SGRF_S1 --> SGRF_S2 --> SGRF_S3 --> OUT
```

### TCA 空间扫描详解 (Mark1)

```
feat_tca (B,T,C,H/2,W/2) from SWD — 已经是半分辨率, 无需再降采样
  │
  ├─ [MVC-Shift] 3分支空洞DWConv(d=1,2,3) + 1x1混频 → x_shifted
  │
  ├─ [SpatialWKV2D] 4方向 Bi-WKV
  │     ├─ Split C→4 heads × C/4
  │     ├─ pre_norm (LayerNorm) — 稳定 R/K/V 投影输入
  │     ├─ Head0/H1/H2/H3: 水平/垂直/主对角/副对角扫描
  │     ├─ 每方向独立 BiWKV (chunk-wise cumsum, ew<1 强制衰减)
  │     └─ Concat 4 heads → σ(R)⊙wkv → proj_out → post_norm
  │
  ├─ [Channel Mix] LN → Conv(C→4C) → GELU → Conv(4C→C) → + x * gamma
  │
  ├─ [Upsample] H/2 → H×W → tca_out (B,T,C,H,W)
  │
  ├─ [TemporalCorrespondence]
  │     proj_qk (C→C/4) → normalize → bmm(Q,K^T)/tau(softplus) → softmax
  │     → C_omega_list: [(B, ds², ds²) × (T-1)]
  │
  └─ [TemporalAggregation]
        C_omega × neighbor_ds → warp → frame_gate → softmax加权
        → upsample + residual + LayerNorm → F_t_aligned (B,C,H,W)
```

### 数值稳定性保证

| 组件 | 措施 |
|------|------|
| **BiWKV** | `w = -F.softplus(spatial_decay)` → ew < 1 恒衰减 |
| **BiWKV** | chunk-wise cumsum (CHUNK=256), `k.clamp(-8,8)` |
| **SpatialWKV2D** | `pre_norm` LayerNorm 在 R/K/V 投影前 |
| **SpatialWKV2D** | R/K/V RWKV-7 小初始化 `±0.05~0.5/√C` |
| **SWD** | proj_tfde/proj_tca 后 LayerNorm → IntensityHead norm≈1 |
| **SWD** | alpha_net 零初始化 → sigmoid(bias=log(0.6/0.4)) → α初始≈0.6 |
| **Tau** | `F.softplus(tau_raw) + 0.05` → 下界 0.05 防除零 |
