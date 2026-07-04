# TFS-Net v6 Delta Mark3 整体架构图 (2026-07-04)

## 图一：最简架构 (含 Mark3 Phase-Dependent)

```mermaid
flowchart TD
    subgraph 输入
        IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]
    end

    subgraph 编码分流["编码 + 小波分流"]
        ENC["Encoder<br/>3级金字塔编码 → F_stack"]
        SWD["SWD 空域小波分流器<br/>HaarDWT → alpha(LL) + noise_gate(HF)<br/>→ feat_tfde (光照+噪声) H/2<br/>→ feat_tca (光照无关+结构) H/2"]
    end

    subgraph 诊断["退化估计"]
        TFDE["TFDE 时频退化估计器<br/>→ s_illum + s_noise<br/>↑Bilinear → H×W"]
    end

    subgraph 对齐["时序对应对齐"]
        TCA["TCA<br/>MVC-Shift + SpatialWKV2D<br/>4方向空间扫描 Bi-WKV<br/>→ F_out_list (B,T,C,H,W)<br/>→ C_omega (时序对应矩阵)<br/>→ F_t_aligned (对齐锚)"]
    end

    subgraph 处理层["三源退化并行建模<br/>Phase 1: NDPN/MCPN 截断=0"]
        ISPN["ISPN 光照源处理<br/>L_ratio锚 + lit_up_delta<br/>→ lit_up_map (空间图,B,3,H,W)<br/>→ f_illum_feat + A_illu"]
        NDPN["NDPN 噪声处理 (Phase 1 = zero)"]
        MCPN["MCPN 运动补偿 (Phase 1 = zero)"]
    end

    subgraph 融合层["CXG 交叉激励门<br/>Phase 1: bypass<br/>Phase 1.5: ratio激活<br/>Phase 2: 正常"]
        CXG["CXG<br/>训练:动态交叉调制<br/>推理:静态重参数化"]
    end

    subgraph 执行层["SGRF 阶段式修复融合"]
        SGRF["SGRF<br/>S1:去噪 S2:去模糊 S3:提亮<br/>∀ phase: S1/S2/S3 始终激活<br/>S1中间出img_s1, S2出img_s2"]
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

    ISPN -- "lit_up_map + f_illum_feat + A_illu" --> SGRF
    NDPN -- "f_noise (Zero if Phase1)" --> CXG
    MCPN -- "f_motion (Zero if Phase1)" --> CXG
    CXG -- "f_noise_gated, f_motion_gated" --> SGRF

    IN -- "img_center" --> SGRF
    SGRF --> OUT
```

### 框架要点

- **SWD 子带分流**：HaarDWT 在子带级分离 LL→光照/噪声 (TFDE), HF→结构 (TCA)。alpha_net(LL) 可学习 α ∈ (0,1) 分配 LL 子带, noise_gate(HF_energy) 门控 HF 子带。输出 LayerNorm, TFDE 工作在正常 norm 范围。
- **TCA 空间扫描**：输入 SWD 的 H/2 特征, 无需内部再降采样。MVC-Shift(3 dilated DWConv) → 4方向 Bi-WKV → Channel Mix → H 上采样。
- **C_omega 时序对应**：中心帧与邻帧 cosine similarity 矩阵, softmax 归一化。L_diag_prior 自监督鼓励对角→1 (静止区域 identity 对应)。
- **Mark3 Phase-Dependent Forward**：Phase 1 (0-19): NDPN/MCPN 输出截断为零, CXG bypass, 仅 ISPN+SGRF 激活。Phase 1.5 (20-24): 线性 unlock_ratio。Phase 2 (25-49): 全网络。

### Mark3 多阶段训练策略

| 阶段 | Epoch | lr | NDPN/MCPN | 损失项 |
|------|-------|-----|-----------|--------|
| Phase 1 Warmup | 0-4 | 8e-6→8e-4 | zero | pix+ssim+illum+lit_up_sup+swd_reg+ifpn(align+diag) |
| Phase 1 Main | 5-19 | 6e-4 | zero | 同上 |
| Phase 1.5 | 20-24 | 6e-4→4e-4 | 线性 0→100% | 同上 |
| Phase 2 | 25-49 | 4e-4→1e-4 | 100%+CXG | +perceptual_decoupling(SSIM→S1,VGG→S2)+freq+inter |

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

    subgraph TFDE["TFDE 时频退化估计器"]
        direction TB
        TFDE_SP["SpatialBranch<br/>soft-median → μ,σ,SNR → F_s"]
        TFDE_FQ["FrequencyBranch<br/>temporal_fuse(5帧频谱平均)→ F_f"]
        TFDE_HD["IntensityHead(F_s, F_f)<br/>→ s_illum (光照退化, [0,1])<br/>→ s_noise (噪声退化, [0,1])<br/>↑Bilinear → H×W"]
        TFDE_SP --> TFDE_HD
        TFDE_FQ --> TFDE_HD
    end

    subgraph TCA["TCA 时序对应对齐 (SWD feat_tca H/2 输入)"]
        direction TB

        subgraph TCA_SPATIAL["空间扫描 Bi-WKV"]
            TCA_MVC["MVC-Shift<br/>3分支空洞DWConv(d=1,2,3) + 1x1融合"]
            TCA_WKV["SpatialWKV2D<br/>4方向(chunk=4) H/V/主对角/副对角<br/>pre_norm → R/K/V proj → BiWKV cumsum<br/>→ σ(R)⊙wkv → proj_out → post_norm"]
            TCA_CM["Channel Mix<br/>LN → Conv(64→256)→GELU→Conv(256→64) + γ×residual"]
            TCA_UP["上采样 H/2→H → tca_out (B,T,64,H,W)"]
            TCA_MVC --> TCA_WKV --> TCA_CM --> TCA_UP
        end

        subgraph TCA_TEMP["时序对应 + 聚合"]
            TCORR["TemporalCorrespondence<br/>proj_qk(Q,K:64→16) → cosine_sim<br/>÷ softplus(tau)+0.05 → softmax(dim=-1)<br/>→ C_omega_list: (T-1)×(B,N,N) N=(H/2)²"]
            TAGG["TemporalAggregation<br/>C_omega warp 邻帧 → frame_gate 加权<br/>→ upsample + LN → F_t_aligned (B,64,H,W)"]
            TCORR --> TAGG
        end

        TCA_SPATIAL --> TCORR
    end

    subgraph ISPN["ISPN 光照源处理网络 (Phase 1/2 始终全部激活)"]
        direction TB
        ISPN_COARSE["coarse_adapter(64→128)→ GELU<br/>+ s_illum_proj(zeroinit) → coarse_feats (B,T,128,H/4,W/4)"]
        ISPN_ILLUM["IllumExtract(coarse, img) → L_t, L_neighbors<br/>cosine_sim/temp → weights → weighted L_ref<br/>L_ratio = L_ref / (L_t+ε) → clamp[0.5,8] → ↑bilinear H×W"]
        ISPN_DELTA["ratio_proj(3→64)→ConvBlock×2→f_illum_feat<br/>+ F_t_aligned锚定(illu_anchor, tanh_gate)<br/>→ lit_up_delta=lit_up_proj(f_illum_feat)"]
        ISPN_HYBRID["lit_up_map = 1 + max_bright·σ(L_ratio + lit_up_delta)<br/>∈ [1.0, 1+max_bright] 空间图 (B,3,H,W)"]
        ISPN_AILLU["A_illu = illu_conv(f_illum_feat)<br/>DWConv(64→64)→Conv(64→1)→Sigmoid<br/>∈ [0,1] 空间光照注意力"]
        ISPN_COARSE --> ISPN_ILLUM --> ISPN_DELTA --> ISPN_HYBRID
        ISPN_DELTA --> ISPN_AILLU
    end

    subgraph NDPN["NDPN 噪声退化处理 (Phase 1: zero截断)"]
        NDPN_CONF["conf_proj: C_omega diag → conf_map<br/>noise_extract: |enc - F_t_aligned| → residual"]
        NDPN_STR["denoise_strength(conf, residual) → strength<br/>f_noise = enc - γ·residual·strength + σ_noise"]
        NDPN_CONF --> NDPN_STR
    end

    subgraph MCPN["MCPN 运动补偿处理 (Phase 1: zero截断)"]
        MCPN_MOT["motion_estimator: C_omega diag偏移 → motion_mag"]
        MCPN_COMP["comp_gate: [F_t_aligned, motion_mag] → compensation<br/>f_motion = gate·F_t_aligned + (1-gate)·F_omega"]
        MCPN_MOT --> MCPN_COMP
    end

    subgraph CXG["CXG 交叉激励门 (Phase 2 启用, Phase 1 bypass)"]
        CXG_G["gate_noise(f_motion) → f_noise_gated<br/>gate_motion(f_noise) → f_motion_gated"]
    end

    subgraph SGRF["SGRF 阶段式修复融合 (所有 Phase 始终激活)"]
        direction LR
        SGRF_S1["S1: Denoise<br/>f_noise_gated + img_center<br/>→ StageBlock(resblock×2)<br/>→ img_s1 (中间监督:SSIM)"]
        SGRF_S2["S2: Deblur<br/>f_motion_gated + img_s1<br/>→ StageBlock(resblock×2)<br/>→ img_s2 (中间监督:VGG感知)"]
        SGRF_S3["S3: Brighten<br/>img_s2 × lit_up_map × (1+A_illu)<br/>→ clamp[0,1] → res_t<br/>(中间监督:Charbonnier)"]
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC --> DWT
    PROJ -- "feat_tfde" --> TFDE_SP
    PROJ -- "feat_tfde" --> TFDE_FQ
    PROJ -- "feat_tca" --> TCA_MVC
    TFDE_HD -- "s_illum" --> ISPN_COARSE
    TFDE_HD -- "s_noise" --> NDPN_STR
    TCA_UP -- "F_aligned_list" --> ISPN_COARSE
    TCA_TEMP -- "F_t_aligned" --> ISPN_DELTA
    TCA_TEMP -- "C_omega" --> NDPN_CONF
    TCA_TEMP -- "C_omega" --> MCPN_MOT
    ISPN_HYBRID --> SGRF_S3
    ISPN_AILLU --> SGRF_S3
    ISPN_DELTA -- "f_illum_feat" --> SGRF_S3
    NDPN_STR -- "f_noise" --> CXG_G
    MCPN_COMP -- "f_motion" --> CXG_G
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
  │     ├─ 每方向独立 BiWKV (chunk-wise cumsum=256, ew<1 强制衰减)
  │     └─ Concat 4 heads → σ(R)⊙wkv → proj_out → post_norm
  │
  ├─ [Channel Mix] LN → Conv(C→4C) → GELU → Conv(4C→C) → + x * gamma
  │
  ├─ [Upsample] H/2 → H×W → tca_out (B,T,C,H,W)
  │
  ├─ [TemporalCorrespondence]
  │     proj_qk (C→C/4) → cosine_sim → ÷ softplus(tau)+0.05 → softmax(dim=-1)
  │     → C_omega_list: [(B, (H/2)², (H/2)²) × (T-1)]
  │
  └─ [TemporalAggregation]
        C_omega × neighbor_ds → warp → frame_gate → softmax加权
        → upsample + residual + LayerNorm → F_t_aligned (B,C,H,W)
```

### Mark3 损失函数架构

```
                    Phase 1 Warmup (0-4)         Phase 1 Main (5-19)
                    ─────────────────────        ─────────────────────
res_t    ←→ GT  ─→  pix (PECharbonnier)          pix
                    ssim (1-SSIM)                 ssim
lit_up_map ← GT/I̅ ─→ lit_up_sup (L1, 0.02×illum) lit_up_sup
s_illum          ─→  illum_smooth (edge-TV)       illum_smooth
SWD α            ─→  swd_reg (0.001固定)          swd_reg
C_omega          ─→                               align_warp (L1 warp一致性)
                                                  diag_prior (-log(diag)自监督)
NDPN/MCPN         →  零截断 (不参与训练)
CXG               →  bypass (不参与训练)

                    Phase 1.5 (20-24)             Phase 2 (25-49)
                    ─────────────────────        ────────────────────
                    损失同上, NDPN/MCPN           img_s1 ←→ GT → ssim_s1
                    线性unlock_ratio              img_s2 ←→ GT → perc_s2 (VGG)
                    CXG: ratio>0.3启用            img_s2·lit_up ← GT → pix_s3 (Charb)
                                                  ifpn_side ← GT↓ → ifpn_sup
                                                  res_t ←→ GT → freq (FFT L1)
                            all via Kendall UW (learnable log_vars)
```

### 数值稳定性保证

| 组件 | 措施 |
|------|------|
| **BiWKV** | `w = -F.softplus(spatial_decay)` → e^w < 1 恒衰减 |
| **BiWKV** | chunk-wise cumsum (CHUNK=256), `k.clamp(-8,8)`, `v.clamp(-8,8)` |
| **SpatialWKV2D** | `pre_norm` LayerNorm 在 R/K/V 投影前 |
| **SpatialWKV2D** | R/K/V RWKV-7 小初始化 `±0.05~0.5/√C` |
| **SWD** | proj_tfde/proj_tca 后 LayerNorm → IntensityHead norm≈1 |
| **SWD** | alpha_net: sigmoid初始≈0.6 (bias≈0.4) → 偏TFDE但不极端 |
| **Tau** | `F.softplus(tau_raw) + 0.05` → 下界 0.05 防除零 |
| **lit_up_map** | max_bright clamp → [1.0, 1+max_bright] ∈ [1.0, 5.0] 物理合理 |
| **diag_prior** | `C_omega.diag().clamp(min=1e-6)` → -log防数值爆炸 |

### Mark3 vs Delta/Mark1 核心差异

| 维度 | Delta | Mark1 | **Mark3** |
|------|-------|-------|-----------|
| **Encoder→下游** | 直连 TFSI/SACE | SWD 子带分流 | 同 Mark1 |
| **训练策略** | 单阶段 | 单阶段 | **4阶段渐进** |
| **NDPN/MCPN** | 始终激活 | 始终激活 | **Phase 1 截断=0** |
| **lit_up_map** | 无 | 空间图但loss池化 | **空间图+空间L1监督** |
| **diag_prior** | 无 | 无 | **C_omega对角-log先验** |
| **感知解耦** | 统一输出 | 统一输出 | **SSIM→S1, VGG→S2, Charb→S3** |
| **Kendall UW** | 无 | Mark2加入 | 同Mark2 (7个log_vars) |
| **命名统一** | 旧(TFSI等) | ✓ 新命名 | ✓ 新命名 |
