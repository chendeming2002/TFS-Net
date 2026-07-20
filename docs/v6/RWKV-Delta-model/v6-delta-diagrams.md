# TFS-Net v6 Delta Flight9 整体架构图 (2026-07-20, current)

## 图一：简化架构 (No WFR, Single-Scale DPE, H/2 TCA)

```mermaid
flowchart TD
    subgraph 输入
        IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]
    end

    subgraph 编码["Encoder (多尺度直连)"]
        ENC["PyramidEncoder → 三尺度 lateral<br/>l1_lat(64,H,W), l2_lat(64,H/2), l3_lat(64,H/4)<br/>+ LRU帧缓存(64帧) + 训练batch编码 b×T"]
    end

    subgraph 诊断["DPE (Flight9: single-scale softplus)"]
        DPE["DPE @ H/4<br/>l3_lat → 时域统计(μ,σ,SNR) + gray/lum<br/>→ Conv→GELU→DWConv→LayerNorm<br/>→ IllumHead(softplus+s_max, base=0.3, max=3.0)<br/>→ NoiseHead(sigmoid, zero-init)<br/>Loss: L_spatial(-log std) + L_tv(edge-aware)"]
    end

    subgraph 对齐["TCA @ H/2 (Flight9: no WFR, no FPN)"]
        TCA["TCA: l2_lat(64,H/2,W/2) 直连<br/>→ MVC-Shift(d=1,2,3) → SpatialWKV2D 4方向<br/>→ ChannelMix + γ*residual<br/>C_omega @ H/2 → TemporalAggregation → F_t_aligned<br/>↑ upsample to H×W for NDPN/MCPN"]
    end

    subgraph 三源["三源并行处理"]
        ISPN["ISPN @ l1<br/>f_enc + s_illum → TCC 6iter + gain[0.5,2.0]"]
        NDPN["NDPN (γ≤0.03)<br/>C_omega conf + noise_extract + s_noise<br/>→ f_noise_out"]
        MCPN["MCPN (γ=0.01)<br/>window_corr + motion_estimator + compensation<br/>→ f_motion_out"]
    end

    subgraph 融合["CXG + SGRF"]
        CXG["CXG 交叉激励门<br/>gate_noise(f_motion)→f_noise_gated<br/>gate_motion(f_noise)→f_motion_gated"]
        SGRF["SGRF 阶段式修复<br/>S1(denoise)→S2(deblur)→TCC×6→gain→img_lit<br/>sg[img_lit]+residual×tanh(β)→res_t<br/>Aux: L_ndpn(SSIM→s1) + L_mcpn(L1→s2)"]
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC -- "l3 (H/4)" --> DPE
    ENC -- "l2 (H/2)" --> TCA
    ENC -- "l1 (H)" --> ISPN
    DPE -- "s_illum" --> ISPN
    DPE -- "s_noise" --> NDPN
    TCA -- "F_aligned↑, C_omega, μ, σ" --> NDPN
    TCA -- "F_aligned↑, C_omega, σ" --> MCPN
    ISPN -- "gain + curve_α" --> SGRF
    NDPN --> CXG
    MCPN --> CXG
    CXG --> SGRF
    IN -- "img_center" --> SGRF
    SGRF --> OUT
```

### Flight9 关键设计点

- **取消 WFR**: Encoder l1/l2/l3 三尺度直连替代小波分流—l3(H/4)天然低频→DPE, l2(H/2)中等分辨率→TCA, l1(H)全分辨率→ISPN/NDPN/MCPN
- **DPE softplus+soft_clamp**: s_illum 不再是 [0,1] 概率而是连续光照强度(base=0.3, asymptotic max=3.0)。softplus 提供全值域非零梯度, soft_clamp x/(1+x/s_max) 物理上界
- **L_illum_spatial**: `-log(std(s_illum) + 1e-6)` — std→0 时损失→∞, 根除常数解(比 sigmoid 的"鼓励极值"梯度方向更合理)
- **TCA H/2**: 128×128 WKV扫描, batch=4 恢复样本多样性(vs Flight8 batch=2), accum=4 保持 eff=16
- **Gamma clamp**: NDPN.gamma = clamp(gamma_raw, max=0.03), 防止 DPE 饱和期间 gn 暴走

### 多阶段训练策略 (epochs=80)

| 阶段 | Epoch | lr | NDPN/MCPN | 损失项 |
|------|-------|-----|-----------|--------|
| Phase 1 Warmup | 0-4 | 8e-6→8e-4 | zero | pix+ssim+illum+gain_sup+L_spatial+L_tv+ifpn |
| Phase 1 Main | 5-10 | 6e-4 | zero (γ=0.01) | 同上 |
| Phase 1.5 | 11-25 | 6e-4→4.4e-4 | 线性 0→100% | +L_ndpn_aux+L_mcpn_aux+L_gamma_reg |
| Phase 2 | 26-80 | 4e-4→1.6e-5 | 100%+CXG | +percep+freq+inter (diag_prior取消) |

---

## 图二：详细架构

```mermaid
flowchart TD
    IN["多帧低光输入"]

    subgraph Encoder["Encoder"]
        ENC["PyramidEncoder [32,64,96]<br/>→ l1(64,H,W), l2(64,H/2), l3(64,H/4)"]
    end

    subgraph DPE["DPE (H/4, softplus)"]
        DPE_STATS["时域统计: GroupNorm→soft-median(μ),var(σ),μ/σ(SNR)"]
        DPE_PRIOR["物理先验: gray=I.mean, lum=I.max→downsample"]
        DPE_FUSE["Concat[3C+2]→proj(C/1)→DWConv+GELU→LayerNorm"]
        DPE_ILLUM["IllumHead: Conv1x1→softplus+base(0.3)→soft_clamp(max=3.0)"]
        DPE_NOISE["NoiseHead: Conv1x1→sigmoid(zero-init)"]
        DPE_STATS --> DPE_FUSE
        DPE_PRIOR --> DPE_FUSE
        DPE_FUSE --> DPE_ILLUM
        DPE_FUSE --> DPE_NOISE
    end

    subgraph TCA["TCA (H/2, l2直连)"]
        TCA_MVC["MVC-Shift: 3 DWConv(d=1,2,3)+1×1"]
        TCA_WKV["SpatialWKV2D: 4方向Bi-WKV(H/V/对角)<br/>pre_norm→R/K/V→BiWKV(cumsum=256,e^w<1)<br/>→σ(R)⊙wkv→proj_out(zero-init)→post_norm"]
        TCA_CM["ChannelMix: LN→Conv(64→256)→GELU→Conv(256→64)"]
        TCA_CORR["CorrGen: proj_qk→cosine_sim→/τ→softmax→C_omega"]
        TCA_AGG["TempAgg: C_omega warp→frame_gate→softmax→F_t_aligned"]
        TCA_MVC --> TCA_WKV --> TCA_CM --> TCA_CORR --> TCA_AGG
    end

    subgraph ISPN["ISPN"]
        ISPN_REF["refine: Conv(f_enc+s_illum)→GELU→Conv→GELU→h"]
        ISPN_CURVE["SpatialCurveBranch: h→Conv→GELU→Conv→Tanh→A(B,6,3,H,W)"]
        ISPN_GAIN["gain_head: Conv→GELU→Conv→softplus, base=0.5→[0.5,2.0]"]
        ISPN_REF --> ISPN_CURVE
        ISPN_REF --> ISPN_GAIN
    end

    subgraph NDPN["NDPN (γ≤0.03)"]
        NDPN_CONF["conf_proj: diag(C_omega)→Linear→GELU→Linear→Sigmoid(zero-init)"]
        NDPN_NOISE["noise_extract: concat(enc,F_align)→Conv→GELU→Conv<br/>denoise_strength: concat(noise,conf)→Conv→GELU→Conv→Sigmoid(zero-init)"]
        NDPN_OUT["f_noise = enc - γ·noise·strength + noise_proj(s_noise)"]
        NDPN_CONF --> NDPN_NOISE --> NDPN_OUT
    end

    subgraph MCPN["MCPN (γ=0.01)"]
        MCPN_MOT["motion_estimator: diag(C_omega)→Conv→GELU→Conv→Sigmoid(zero-init)"]
        MCPN_COMP["comp_gate: concat(center,motion)→Conv→GELU→Conv→Sigmoid(zero-init)"]
        MCPN_FUSE["g_t·center+(1-g_t)·omega+γ·delta·comp→refine→output"]
        MCPN_MOT --> MCPN_COMP --> MCPN_FUSE
    end

    subgraph SGRF["SGRF (Stage A/B)"]
        SGRF_S1["S1 Denoise: f_noise_gated+img→StageBlock(gate=0)→soft_clamp"]
        SGRF_S2["S2 Deblur: f_motion_gated+img_s1→StageBlock(gate=0)→soft_clamp"]
        SGRF_TCC["S2.5 TCC: img_s2+=A_n·img_s2·(1-img_s2)×6iter"]
        SGRF_S3["S3 Brighten: img_curved×gain→refine→res_t"]
        SGRF_S1 --> SGRF_S2 --> SGRF_TCC --> SGRF_S3
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC -- "l3" --> DPE_STATS
    ENC -- "l2" --> TCA_MVC
    ENC -- "l1" --> ISPN_REF
    DPE_ILLUM -- "s_illum" --> ISPN_REF
    DPE_NOISE -- "s_noise" --> NDPN_OUT
    TCA_AGG -- "F_aligned↑" --> NDPN_NOISE
    TCA_CORR -- "C_omega" --> NDPN_CONF
    TCA_CORR -- "C_omega" --> MCPN_MOT
    ISPN_CURVE -- "curve_α" --> SGRF_TCC
    ISPN_GAIN --> SGRF_S3
    NDPN_OUT --> SGRF_S1
    MCPN_FUSE --> SGRF_S2
    IN -- "img_center" --> SGRF_S1
    SGRF_S3 --> OUT
```

### 数值稳定性

| 组件 | 措施 |
|------|------|
| **DPE IllumHead** | softplus (`log(1+e^x)` 梯度>0 全值域) + soft_clamp `x/(1+x/s_max)` asymptotic max |
| **L_illum_spatial** | `-log(std+1e-6)` → 零方差损失∞, 强制空间分布 |
| **L_illum_tv** | `|∇s|·exp(-10|∇I|)` → 边缘处允许突变, 平坦区强制平滑 |
| **BiWKV** | `w = -F.softplus(spatial_decay)` → e^w < 1 恒衰减, chunk-wise cumsum(256) |
| **Tau** | `F.softplus(tau_raw) + 0.05` → 下界 0.05 防除零 |
| **Gamma** | NDPN.gamma = clamp(raw, max=0.03) → 硬上限防暴走 |
| **C_omega** | ds capped ≤ 96 → N≤9216 → C_omega≤340MB |

### Flight9 vs 前代核心差异

| 维度 | Flight7.2 | Flight8 | **Flight9** |
|------|-----------|---------|-----------|
| **WFR** | feat_tfde + feat_tca | feat_tca 残差 | **取消 (Encoder直连)** |
| **DPE激活** | sigmoid | sigmoid | **softplus+soft_clamp** |
| **DPE尺度** | 单尺 multi-dilation | 3-stage cascade | **单尺 H/4** |
| **TCA WKV** | H/2 128×128 | H 256×256 | **H/2 128×128** |
| **TCA输入** | WFR+Enc feats | Internal FPN+WFR | **l2_lat 直连** |
| **batch/accum** | 4/2 | 2/8 | **4/4** |
| **Loss** | UW-only | UW-only | **+L_spatial+L_tv** |
| **Gamma** | 无限制 | 无限制 | **≤0.03** |
| **Epochs** | 85 | 100→ep22终止 | **80** |
