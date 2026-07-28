# TFS-Net v6 Delta Flight10 Mark5 整体架构图 (2026-07-28, current)

## 图一：简化架构 (NDPN detail_residual highway + F_denoised prior)

```mermaid
flowchart TD
    subgraph Input
        IN["多帧低光 x : (B,T,3,H,W)"]
    end

    subgraph Enc["Encoder → l1/l2/l3"]
        ENC["PyramidEncoder → l1(64,H,W), l2(64,H/2), l3(64,H/4)"]
    end

    subgraph DPE["DPE (softplus, H/4)"]
        DPE1["DPE @ H/4<br/>L_spatial: relu(1.0-std) 0.03<br/>→ s_illum + s_noise"]
    end

    subgraph TCA["TCA (HaarDWT anchor)"]
        TCA1["HaarDWT: LL(IN)→anchor + HF(DWConv)→edge<br/>→ fuse → MVC-Shift → 4dirWKV → C_omega"]
    end

    subgraph NDPN["NDPN m5: temporal + detail highway + F_denoise prior"]
        N1["F_denoised = SNR 加权多帧聚合 + refine skip"]
        N2["detail_residual = f_enc - F_denoised → 1×1 gate → preserved_detail"]
        N3["corr_spatial(3×3) × γ × (1-detail_map)<br/>coarse(F_denoised avgpool) × γ × 0.5 × (1-detail_map)"]
        N4["f_noise = F_denoised + correction + preserved_detail + s_noise"]
        N1 --> N2 --> N4
        N1 --> N3 --> N4
    end

    subgraph MCPN["MCPN m5: single 3×3 motion_refine"]
        M1["window_corr → f_omega_aligned"]
        M2["motion_refine(3×3) × γ × comp_gate"]
        M3["g_t × center + (1-g_t) × omega + motion_delta"]
        M1 --> M2 --> M3
    end

    subgraph Fusion["CXG + SGRF"]
        CXG["CXG 交叉激励门"]
        SGRF["SGRF: S1→S2(zero-mean)→TCC×6→gain→img_lit<br/>Stage B: delta_scale=0.2→res_t"]
    end

    OUT["res_t"]

    IN --> ENC
    ENC -- "l3" --> DPE1
    ENC -- "l2" --> TCA1
    ENC -- "l1" --> N1
    DPE1 -- "s_illum" --> ISPN["ISPN sigmoid[0.5,2.0]"]
    TCA1 -- "C_omega, F_aligned" --> N1
    TCA1 -- "C_omega, F_aligned" --> M1
    N4 --> CXG
    M3 --> CXG
    CXG --> SGRF
    IN -- "img_center" --> SGRF
    ISPN -- "gain+curve" --> SGRF
    SGRF --> OUT
```

## NDPN m5 详细数据流

```
f_enc (B,64,H,W)
  │
  ├─→ SNR weighted multi-frame aggregation → F_denoised
  │     → + refine(F_denoised) skip connect (P2 fix)
  │
  ├─→ detail_map: |∇F_denoised| → detail_proj → (B,1,H,W) ∈ [0,1]
  │     P1 fix: F_denoised is temporal-filtered, noise-free
  │
  ├─→ detail_residual = f_enc - F_denoised  (what temporal agg removed?)
  │     → detail_gate_1x1(detail_residual) → sigmoid gate
  │     → preserved_detail = detail_residual × gate × detail_map
  │     P0 fix: true highway — only preserve details that differ from smoothed F_denoised
  │
  ├─→ corr_spatial([f_enc, F_denoised, detail_map], 3×3)  (spatial correction)
  │     × γ(max=0.1, P2: relaxed from 0.03) × (1-detail_map)
  │
  ├─→ coarse_prior(F_denoised avgpool→upsample→conv, 3×3)
  │     × γ × 0.5 × (1-detail_map)
  │     P1 fix: F_denoised avgpool replaces l2_lat, properly gated
  │
  └─→ f_noise = F_denoised + corr_spatial + preserved_detail + coarse_prior + s_noise
```

## m4 → m5 修复对照

| # | 问题 (plan) | m4 | **m5** |
|:--:|------|-----|------|
| P0 | NDPN 1×1 与 3×3 同目标冗余 | corr_spatial + corr_pointwise | **detail_residual = f_enc - F_denoised → gate** |
| P0 | MCPN 1×1 对运动无意义 | motion_refine 3×3 + 1×1 | **回退单 3×3** |
| P1 | coarse l2 含噪且无门控 | l2_lat → coarse_proj | **F_denoised avgpool → coarse_prior, +detail_map** |
| P1 | detail_map 被噪声污染 | image_center gradient | **F_denoised gradient** |
| P2 | gamma 0.03 压制三路 | clamp 0.03 | **clamp 0.1** |
| P2 | refine 无 skip | F_denoised = refine(x) | **F_denoised = x + refine(x)** |

## 版本演进

| 版本 | NDPN 核心 | ep40 PSNR |
|------|----------|:--:|
| F10m1 | f_enc - noise_feat × strength (bug: softplus δ) | 17.35 |
| F10m2 | 同上 + bug fixed | 17.60 |
| F10m3 | F_denoised + detail-gated correction | **18.27** |
| F10m4 | m3 + 1×1 bypass + l2 coarse | 设计缺陷 |
| **F10m5** | **m3 + detail_residual highway + F_denoised prior + γ→0.1** | **训练中** |
