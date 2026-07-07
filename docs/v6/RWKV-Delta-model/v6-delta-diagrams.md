# TFS-Net v6 Delta Mark4 整体架构图 (2026-07-06)

## 图一：最简架构 (含 Mark4 Phase-Dependent + 三部保险)

```mermaid
flowchart TD
    subgraph 输入
        IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]
    end

    subgraph 编码分流["编码 + 小波分流"]
        ENC["Encoder<br/>3级金字塔编码 → F_stack"]
        SWD["WSD 基于小波的源分流器<br/>HaarDWT → alpha(LL) + noise_gate(HF)<br/>→ feat_tfde (光照+噪声) H/2<br/>→ feat_tca (光照无关+结构) H/2"]
    end

    subgraph 诊断["退化估计 (Mark4: 纯空域多尺度)"]
        TFDE["DIE<br/>多尺度空洞卷积 (d=1,2,4)<br/>时域统计 [μ,σ,SNR] → s_illum + s_noise"]
    end

    subgraph 对齐["时序对应对齐"]
        TCA["TCA<br/>MVC-Shift + SpatialWKV2D<br/>4方向空间扫描 Bi-WKV<br/>→ F_out_list + C_omega + F_t_aligned"]
    end

    subgraph 处理层["三源退化并行建模<br/>Phase 1: NDPN/MCPN 截断=0"]
        ISPN["ISPN 光照源处理<br/>f_enc + s_illum → gain + bias<br/>gain=exp(log_gain)∈[1,max], bias=tanh·range"]
        NDPN["NDPN 噪声处理 (Phase 1 = zero)"]
        MCPN["MCPN 运动补偿 (Phase 1 = zero)<br/>Mark4: gamma=0, refine=0, startup_gate=1"]
    end

    subgraph 融合层["CXG 交叉激励门<br/>Phase 1: bypass<br/>Phase 1.5: ratio激活<br/>Phase 2: 正常"]
        CXG["CXG<br/>训练:动态交叉调制<br/>推理:静态重参数化"]
    end

    subgraph 执行层["SGRF 阶段式修复融合 (Mark4: gain/bias)"]
        SGRF["SGRF<br/>S1: zero-gate StageBlock + f_noise → img_s1<br/>S2: zero-gate StageBlock + f_motion → img_s2<br/>S3: img_s2 × gain + bias → res_t"]
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC --> SWD
    SWD -- "feat_tfde" --> TFDE
    SWD -- "feat_tca" --> TCA
    TFDE -- "s_illum" --> ISPN
    TFDE -- "s_noise" --> NDPN

    TCA -- "F_aligned_list" --> ISPN
    TCA -- "F_aligned, C_omega, mu, sigma" --> NDPN
    TCA -- "F_aligned, C_omega, sigma" --> MCPN

    ISPN -- "gain_map + bias_map" --> SGRF
    NDPN -- "f_noise (Zero if Phase1)" --> CXG
    MCPN -- "f_motion (Zero if Phase1)" --> CXG
    CXG -- "f_noise_gated, f_motion_gated" --> SGRF

    IN -- "img_center" --> SGRF
    SGRF --> OUT
```

### 框架要点

- **WSD 子带分流**：HaarDWT 在子带级分离 LL→光照/噪声 (TFDE), HF→结构 (TCA)。alpha_net(LL) 可学习 α ∈ (0,1) 分配 LL 子带, noise_gate(HF_energy) 门控 HF 子带。输出 LayerNorm。
- **DIE (Mark4 简化)**：移除 FrequencyBranch/LFF/phase_conf，用多尺度空洞卷积 (d=1,2,4) 提取时域统计量 [μ,σ,SNR] 的空域特征。梯度路径从 `feats→LFF→DWT→phase→s_noise` 简化为 `feats→统计量→Conv→head`。
- **ISPN (Mark4 简化)**：移除多帧 cosine attention + L_ratio 锚定，直接用 `f_enc + s_illum` 生成 gain_map (乘法提亮) + bias_map (加法修正)。零初始化 → gain≈1, bias≈0 → Phase 1 友好。
- **TCA 空间扫描**：输入 SWD 的 H/2 特征。MVC-Shift(3 dilated DWConv) → 4方向 Bi-WKV → Channel Mix → H 上采样。C_omega 行 softmax 归一化，L_diag_prior 自监督。
- **三部保险 (Mark4)**：Phase 2 启动时 NDPN (gamma=0, noise_proj=0) + MCPN (gamma=0, refine=0, startup_gate=1, out_scale=0) + SGRF StageBlock (gate=0) — 任一层都能独立保证无扰动。

### Mark4 多阶段训练策略

| 阶段 | Epoch | lr | NDPN/MCPN | SGRF gate | 损失项 |
|------|-------|-----|-----------|-----------|--------|
| Phase 1 Warmup | 0-4 | 8e-6→8e-4 | zero | gate=0 | pix+ssim+illum+gain_sup+swd_reg+ifpn(align+diag) |
| Phase 1 Main | 5-19 | 6e-4 | zero | gate=0 | 同上 |
| Phase 1.5 | 20-29 | 6e-4→4e-4 | 线性 0→100% | 渐进学习 | 同上 |
| Phase 2 | 30-49 | 4e-4→1e-4 | 100%+CXG | 已学习 | +perceptual_decoupling(SSIM→S1,VGG→S2)+freq+inter |

---

## 图二：带细节架构

```mermaid
flowchart TD
    IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]

    subgraph Encoder["Encoder"]
        ENC["PyramidEncoder<br/>[32,64,96] → 64ch<br/>→ F_stack (B,T,64,H,W)"]
    end

    subgraph SWD["WSD 基于小波的源分流器"]
        direction TB
        DWT["HaarDWT2D<br/>→ LL, LH, HL, HH (H/2×W/2)"]
        ALPHA["alpha_net(LL)<br/>DWConv3x3 → GELU → Conv1x1 → Sigmoid<br/>α ∈ (0,1) — 光照分配比"]
        LL_DIV["LL分流: α·LL → TFDE (光照+噪声)<br/>(1-α)·LL + InstanceNorm → TCA (去光照结构)"]
        HF_ENERGY["HF能量: (LH²+HL²+HH²).mean(dim=C) → (B,1,H/2,W/2)"]
        NOISE_G["noise_gate(HF_energy)<br/>Conv→Sigmoid → g_n ∈ (0,1)<br/>高能量=高噪声概率"]
        HF_DIV["HF分流: g_n×HF_cat → TFDE (噪声相关)<br/>(1-g_n)×HF_cat + LayerNorm → TCA (结构)"]
        PROJ["proj_tfde/proj_tca: Conv(4C→C)+GELU+LN<br/>→ feat_tfde, feat_tca (B,T,C,H/2,W/2)"]
        DWT --> ALPHA --> LL_DIV
        DWT --> HF_ENERGY --> NOISE_G --> HF_DIV
        LL_DIV --> PROJ
        HF_DIV --> PROJ
    end

    subgraph TFDE["DIE 退化估计器 (Mark4: 纯空域)"]
        direction TB
        TFDE_STATS["时域统计量<br/>GroupNorm → soft-median(μ), var(σ), μ/σ(SNR)<br/>→ concat[μ,σ,SNR] (B, 3C, H, W)"]
        TFDE_MS["MultiScaleSpatialBranch<br/>3×3(d=1): 局部纹理 → mid ch<br/>3×3(d=2): 中尺度光照 → mid ch<br/>3×3(d=4): 大尺度区域 → wide ch<br/>Concat + 1×1 fuse → F_fused"]
        TFDE_HD["Conv2d(F_fused → 2ch) → Sigmoid<br/>s_illum = [:,0:1], s_noise = [:,1:2]"]
        TFDE_STATS --> TFDE_MS --> TFDE_HD
    end

    subgraph TCA["TCA 时序对应对齐"]
        direction TB

        subgraph TCA_SPATIAL["空间扫描 Bi-WKV"]
            TCA_MVC["MVC-Shift<br/>3分支空洞DWConv(d=1,2,3) + 1x1融合"]
            TCA_WKV["SpatialWKV2D<br/>4方向(chunk=4) H/V/主对角/副对角<br/>pre_norm → R/K/V proj → BiWKV cumsum<br/>→ σ(R)⊙wkv → proj_out → post_norm"]
            TCA_CM["Channel Mix<br/>LN → Conv(64→256)→GELU→Conv(256→64) + γ×residual"]
            TCA_UP["上采样 H/2→H → tca_out (B,T,64,H,W)"]
            TCA_MVC --> TCA_WKV --> TCA_CM --> TCA_UP
        end

        subgraph TCA_TEMP["时序对应 + 聚合"]
            TCORR["TemporalCorrespondence<br/>proj_qk(Q,K:64→16) → cosine_sim<br/>÷ softplus(tau)+0.05 → softmax(dim=-1)<br/>→ C_omega_list: (T-1)×(B,N,N) N=ds², ds≤96"]
            TAGG["TemporalAggregation<br/>C_omega warp 邻帧 → frame_gate 加权<br/>→ upsample + LN → F_t_aligned (B,64,H,W)"]
            TCORR --> TAGG
        end

        TCA_SPATIAL --> TCORR
    end

    subgraph ISPN["ISPN 光照源处理 (Mark4: Retinex gain/bias)"]
        direction TB
        ISPN_REFINE["refine: Conv(f_enc + s_illum, 65→64)<br/>→ GELU → Conv(64→64) → GELU → h"]
        ISPN_GAIN["gain_head: Conv(64→16)→GELU→Conv(16→1)<br/>→ exp(log_gain) → clamp[1, max_gain]<br/>零初始化 → gain≈1 (恒等)"]
        ISPN_BIAS["bias_head: Conv(64→16)→GELU→Conv(16→3)<br/>→ tanh × range → bias∈[-0.1, 0.1]<br/>零初始化 → bias≈0 (无偏置)"]
        ISPN_REFINE --> ISPN_GAIN
        ISPN_REFINE --> ISPN_BIAS
    end

    subgraph NDPN["NDPN 噪声退化处理 (Phase 1: zero截断)"]
        NDPN_CONF["conf_proj: C_omega diag → conf_map<br/>noise_extract: |enc - F_t_aligned| → residual"]
        NDPN_STR["f_noise = enc - γ·residual·strength + noise_proj(s_noise)<br/>γ=0 → f_noise≈f_enc (pass-through)"]
        NDPN_CONF --> NDPN_STR
    end

    subgraph MCPN["MCPN 运动补偿处理 (Phase 1: zero截断, Mark4 静默启动)"]
        MCPN_MOT["motion_estimator + window_corr → f_omega_aligned"]
        MCPN_GATE["gate + startup_bias → g_t (初→1.0, 纯中心帧)<br/>f_fuse = g_t·f_center + (1-g_t)·f_omega + γ·δ"]
        MCPN_REF["refine(conv2=0) + f_center·out_scale(0)<br/>→ f_motion≈f_center (pass-through)"]
        MCPN_MOT --> MCPN_GATE --> MCPN_REF
    end

    subgraph CXG["CXG 交叉激励门 (Phase 2 启用)"]
        CXG_G["gate_noise(f_motion) → f_noise_gated<br/>gate_motion(f_noise) → f_motion_gated"]
    end

    subgraph SGRF["SGRF 阶段式修复融合 (Mark4: zero-gate + gain/bias)"]
        direction LR
        SGRF_S1["S1: Denoise<br/>f_noise_gated + img_center<br/>→ StageBlock(gate=0) → img_s1"]
        SGRF_S2["S2: Deblur<br/>f_motion_gated + img_s1<br/>→ StageBlock(gate=0) → img_s2"]
        SGRF_S3["S3: Brighten<br/>img_s2 × gain_map + bias_map<br/>+ refine(0-gate) → clamp → res_t"]
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC --> DWT
    PROJ -- "feat_tfde" --> TFDE_STATS
    PROJ -- "feat_tca" --> TCA_MVC
    TFDE_HD -- "s_illum" --> ISPN_REFINE
    TFDE_HD -- "s_noise" --> NDPN_STR
    TCA_UP -- "F_aligned_list" --> ISPN_REFINE
    TCA_TEMP -- "F_t_aligned" --> ISPN_REFINE
    TCA_TEMP -- "C_omega" --> NDPN_CONF
    TCA_TEMP -- "C_omega" --> MCPN_MOT
    ISPN_GAIN --> SGRF_S3
    ISPN_BIAS --> SGRF_S3
    NDPN_STR -- "f_noise" --> CXG_G
    MCPN_REF -- "f_motion" --> CXG_G
    CXG_G -- "f_noise_gated" --> SGRF_S1
    CXG_G -- "f_motion_gated" --> SGRF_S2
    IN -- "img_center" --> SGRF_S1
    SGRF_S1 --> SGRF_S2 --> SGRF_S3 --> OUT
```

### TCA 空间扫描详解

```
feat_tca (B,T,C,H/2,W/2) from SWD — 已是半分辨率, 无需再降采样
  │
  ├─ [MVC-Shift] 3分支空洞DWConv(d=1,2,3) + 1x1混频 → x_shifted
  │
  ├─ [SpatialWKV2D] 4方向 Bi-WKV
  │     ├─ Split C→4 heads × C/4
  │     ├─ pre_norm (LayerNorm) — 稳定 R/K/V 投影输入
  │     ├─ Head0/H1/H2/H3: 水平/垂直/主对角/副对角扫描
  │     ├─ 每方向独立 BiWKV (chunk-wise cumsum=256, e^w<1 强制衰减)
  │     └─ Concat 4 heads → σ(R)⊙wkv → proj_out → post_norm
  │
  ├─ [Channel Mix] LN → Conv(C→4C) → GELU → Conv(4C→C) → + x * gamma
  │
  ├─ [Upsample] H/2 → H×W → tca_out (B,T,C,H,W)
  │
  ├─ [TemporalCorrespondence]
  │     proj_qk (C→C/4) → cosine_sim → ÷ softplus(tau)+0.05 → softmax(dim=-1)
  │     → C_omega_list: [(B, ds², ds²) × (T-1)], ds=min(H,W)//4 capped≤96
  │
  └─ [TemporalAggregation]
        C_omega × neighbor_ds → warp → frame_gate → softmax加权
        → upsample + residual + LayerNorm → F_t_aligned (B,C,H,W)
```

### Mark4 三部保险 (零扰动启动)

```
Phase 1 → Phase 1.5 → Phase 2 过渡

NDPN:   f_noise  = f_enc - γ·noise·strength + noise_proj(s_noise)
          γ=0 → f_noise ≈ f_enc (pass-through)
          noise_proj 零初始化 → 第二项 = 0

MCPN:   g_t = sigmoid(raw_gate + startup_gate=1) → ≈0.73
          f_fuse = g_t·f_center + (1-g_t)·f_omega + δ·γ(0)
          refine(conv2=0, out_scale=0) → f_motion ≈ f_center

SGRF:   StageBlock gate = 0 → delta = Conv(x)*0 = 0 → img unchanged
                ↓
           三重零保证: 分支零 × gate零 × unlock零
                ↓
           Phase 2 启动时输出 ≡ Phase 1 输出
```

### Mark4 损失函数架构

```
                    Phase 1 Warmup (0-4)         Phase 1 Main (5-19)
                    ─────────────────────        ─────────────────────
res_t    ←→ GT  ─→  pix (PECharbonnier)          pix
                    ssim (1-SSIM)                 ssim
gain_map ← GT/I̅ ─→  gain_sup (L1, 0.02×illum)   gain_sup
s_illum          ─→  illum_smooth (edge-TV)       illum_smooth
SWD α            ─→  swd_reg (0.001固定)          swd_reg
C_omega          ─→                               align_warp (L1 warp一致性)
                                                  diag_prior (-log(diag)自监督)
NDPN/MCPN         →  零截断 (不参与训练)
CXG               →  bypass (不参与训练)
SGRF gate         →  零 (不参与训练, 三重保险)

                    Phase 1.5 (20-29)             Phase 2 (30-49)
                    ─────────────────────        ────────────────────
                    损失同上, NDPN/MCPN           img_s1 ←→ GT → ssim_s1
                    线性unlock_ratio              img_s2 ←→ GT → perc_s2 (VGG)
                    SGRF gate 渐进学习             img_s2×gain ← GT → inter
                    CXG: ratio>0.3启用            res_t ←→ GT → freq (FFT L1)
                            all via Kendall UW (learnable log_vars)
```

### 数值稳定性保证

| 组件 | 措施 |
|------|------|
| **BiWKV** | `w = -F.softplus(spatial_decay)` → e^w < 1 恒衰减 |
| **BiWKV** | chunk-wise cumsum (CHUNK=256), `k.clamp(-8,8)`, `v.clamp(-8,8)` |
| **SpatialWKV2D** | `pre_norm` LayerNorm 在 R/K/V 投影前, R/K/V RWKV-7 小初始化 |
| **SWD** | proj_tfde/proj_tca 后 LayerNorm → IntensityHead norm≈1 |
| **SWD** | alpha_net: sigmoid初始≈0.6 (bias≈0.4) → 偏TFDE但不极端 |
| **Tau** | `F.softplus(tau_raw) + 0.05` → 下界 0.05 防除零 |
| **C_omega** | ds capped ≤ 96 → N≤9216 → C_omega≤340MB, 防OOM |
| **diag_prior** | `C_omega.diag().clamp(min=1e-6)` → -log防数值爆炸 |
| **gain_map** | `exp(log_gain).clamp(1.0, max_gain)` → 物理合理范围 |
| **bias_map** | `tanh(·)·range` → 有界输出 |

### Mark4 vs Mark3/Mark1 核心差异

| 维度 | Mark1 | Mark3 | **Mark4** |
|------|-------|-------|-----------|
| **TFDE** | Spatial+Freq+LFF | 同Mark1 | **纯空域多尺度空洞卷积** |
| **ISPN** | 多帧cosine attention | 同Mark1 | **gain/bias Retinex 头** |
| **SGRF Stage3** | lit_up_map×(1+A_illu) | 同Mark1 | **img×gain+bias** |
| **StageBlock** | 标准ResBlock | 同Mark1 | **zero-gate 零初始化** |
| **MCPN** | 随机初始化 | startup_gate=1 | **gamma=0 + refine=0 + out_scale=0** |
| **Phase 2 启动** | 剧烈扰动 | PSNR 18→8.7 暴跌 | **三重保险 (无扰动)** |
| **Phase 1.5** | 5 epoch (20-25) | 同 | **10 epoch (20-30)** |
| **参数** | 1.69M | 1.69M | **1.45M** |
