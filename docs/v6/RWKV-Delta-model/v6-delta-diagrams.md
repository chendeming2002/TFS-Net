# TFS-Net v6 Delta Flight10 Mark4 整体架构图 (2026-07-27, current)

## 图一：简化架构 (NDPN 1×1 bypass + MCPN 1×1 bypass + l2 coarse)

```mermaid
flowchart TD
    subgraph 输入
        IN["多帧低光输入 x : (B,T,3,H,W)"]
    end

    subgraph 编码["Encoder → l1/l2/l3"]
        ENC["PyramidEncoder → l1(64,H,W), l2(64,H/2), l3(64,H/4)"]
    end

    subgraph 诊断["DPE (softplus, H/4)"]
        DPE["DPE @ H/4<br/>L_spatial: relu(1.0-std) 权0.03<br/>→ s_illum(softplus) + s_noise(sigmoid)"]
    end

    subgraph 对齐["TCA (HaarDWT anchor)"]
        TCA["TCA @ H/2<br/>HaarDWT: LL(IN→1×1)→anchor + HF(DWConv)→edge<br/>→ fuse → MVC-Shift → 4dirWKV → ChannelMix<br/>→ C_omega + F_t_aligned"]
    end

    subgraph NDPN["NDPN (temporal base + 1×1 bypass + l2 coarse)"]
        NDPN_DETAIL["F_denoised = SNR加权多帧聚合 (主去噪)"]
        NDPN_CORR["correction = (corr_spatial 3×3 + corr_pointwise 1×1)<br/>× γ × (1-detail_map)<br/>+ coarse_proj(l2↑) × γ × 0.5"]
        NDPN_OUT["f_noise = F_denoised + correction + noise_proj(s_noise)"]
        NDPN_DETAIL --> NDPN_CORR --> NDPN_OUT
    end

    subgraph MCPN["MCPN (1×1 bypass)"]
        MCPN_MOT["motion_refine = spatial(3×3) + pointwise(1×1)"]
        MCPN_FUSE["g_t·center + (1-g_t)·omega + motion_delta·comp·γ"]
        MCPN_MOT --> MCPN_FUSE
    end

    subgraph 融合["CXG + SGRF"]
        CXG["CXG 交叉激励门"]
        SGRF["SGRF<br/>Stage A: S1(zero-mean)→S2(zero-mean)→TCC×6→gain→img_lit<br/>Stage B: delta_scale=0.2×residual→res_t"]
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC -- "l3(H/4)" --> DPE
    ENC -- "l2(H/2)" --> TCA
    ENC -- "l1(H)" --> NDPN_DETAIL
    ENC -- "l2(H/2)" -.-> NDPN_CORR
    DPE -- "s_illum" --> ISPN
    TCA -- "C_omega, F_aligned" --> NDPN_DETAIL
    TCA -- "C_omega, F_aligned" --> MCPN_MOT
    NDPN_OUT --> CXG
    MCPN_FUSE --> CXG
    CXG --> SGRF
    IN -- "img_center" --> SGRF
    SGRF --> OUT
```

## 核心变更 (m3 → m4)

| # | 模块 | 变更 | 目的 |
|:--:|------|------|------|
| 1 | NDPN | corr_spatial(3×3) + **corr_pointwise(1×1)** | 1×1 无色散混参，细纹直接透传 |
| 2 | NDPN | + **coarse_proj(l2_lat↑)** | H/2 全局噪声分布，补充局部细节 |
| 3 | MCPN | motion_refine 3×3 + **1×1 bypass** | 运动补偿保留亚像素定位精度 |

## NDPN 数据流 (m4 详细)

```
f_enc (B,64,H,W) ────┬──→ SNR-weighted multi-frame aggregation
l2_lat[:,c] (64,H/2)  │    → F_denoised = Σ(w_i × F_aligned[i])
                      │
image_center (B,3,H,W) → |∇I| → detail_proj → detail_map ∈ [0,1]
                      │
          ┌───────────┘
          ▼
  correction_input = [f_enc, F_denoised, detail_map]
          │
          ├─→ corr_spatial (3×3→GELU→3×3) → 空间去噪 (identify noise+texture patterns)
          ├─→ corr_pointwise (1×1→GELU→1×1) → 像素直通 (pointwise detail pass)
          │
          └─→ correction = (spatial + pointwise) × γ × (1-detail_map)
              + coarse_proj(l2_center↑) × γ × 0.5
              × conf_map (if C_omega available)
          │
          ▼
  f_noise = F_denoised + correction + noise_proj(s_noise)
```

## 版本演进

| 版本 | 关键NDPN设计 | ep40 PSNR |
|------|-------------|:--:|
| F10m1 | f_enc - noise_feat × strength × γ | 17.35 (bug inflate) |
| F10m2 | same + S1/S2 bug fix | 17.60 |
| **F10m3** | **F_denoised + detail-gated correction** | **18.27** |
| **F10m4** | **m3 + 1×1 bypass + l2 coarse + MCPN 1×1** | **训练中** |

## 损失函数 (15项, 不变于 m3)

### Phase 1: 7项
L_pix(UW), L_ssim(UW), L_illum_smooth(UW), L_illum_spatial(0.03), L_illum_tv(0.05), L_gain_sup(0.5), L_brightness_preserve(0.5)

### Phase 1.5: +3项
L_ndpn_aux(0.2), L_mcpn_aux(0.1), L_residual_reg(0.05)

### Phase 2: +5项
L_lit(0.5), L_perc(UW), L_freq(UW), L_inter(UW), L_ssim_s2(0.1)
