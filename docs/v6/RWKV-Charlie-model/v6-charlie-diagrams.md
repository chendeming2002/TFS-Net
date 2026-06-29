# TFS-Net v6 Charlie2 整体架构图 (2026-06-29, 更新: Charlie2 D1-D4 实施)

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
        SACE["SACE<br/>纯 RWKV 帧间注意力<br/>多尺度双向 Bi-WKV<br/>concat+channel_mix 融合<br/>V 源 = Encoder 原始中心帧<br/>★ Charlie2: s_noise 已移除"]
    end

    subgraph 处理层["三源退化并行建模"]
        IFPN["IFPN<br/>光照图估计<br/>★ Charlie2: s_illum 调制 + encoder 特征输入"]
        NDPN["NDPN<br/>SNR自适应去噪<br/>s_noise 条件输入"]
        MRPN["MRPN<br/>运动补偿<br/>σ→MRPN 运动感知"]
    end

    subgraph 执行层["修正去噪"]
        IGRF["IGRF<br/>去噪→去模糊→提亮<br/>★ Charlie2: VSRELL A_illu 单一光照"]
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC --> DWT
    ENC -- "F_t (中心帧原始)" --> SACE
    ENC -- "encoder 特征 ★" --> IFPN
    DWT --> TFSI
    DWT --> SACE
    TFSI -- "s_illum ★" --> IGRF
    TFSI -- "s_illum ★" --> IFPN
    TFSI -- "s_noise ★" --> NDPN

    SACE -- "F_aligned" --> IFPN
    SACE -- "F_aligned, μ_t_clean, σ_t_clean" --> NDPN
    SACE -- "F_aligned, σ_t_clean" --> MRPN

    IFPN -- "f_illum" --> IGRF
    NDPN -- "f_noise" --> IGRF
    MRPN -- "f_motion" --> IGRF

    IGRF --> OUT
    IN -- "img_center" --> IGRF
    IN -- "img_center" --> IFPN
```

### 框架要点

- **三源分离估计**：DWT-LFF + TFSI 联合诊断光照(s_illum)和噪声(s_noise)；SACE 通过帧间方差(σ_t_clean)隐式感知运动
- **TFSI ↔ SACE 关系**：TFSI 的 FrequencyBranch 使用多帧邻居融合（Charlie P0-1）；SACE 使用 DWT-LFF 双实例（中心 α=0.6 / 邻居 α=0.4）做对齐前归一化
- **三源并行处理**：IFPN/NDPN/MRPN 三个分支从 SACE 对齐特征各自估计修复方案
- **SACE 纯 RWKV 注意力**：多尺度双向 Bi-WKV（full/half/quarter），concat+channel_mix 可学习融合（Charlie P1-1）

### Charlie 数据流改动（vs Bravo）

| 改动 | 优先级 | 说明 |
|------|--------|------|
| **s_noise → NDNP (+ noise_proj)** | P0 | s_noise 从 IGRF Stage1 移除, 注入 NDPN 作为条件输入 (Conv2d 1→64, 零初始化) |
| **σ_t_clean → MRPN (+ blur_estimator)** | P1 | σ_t_clean 从 NDPN 独占到 MRPN 共享, 用于 blur_mask 运动感知 |
| **sigma_sace 调制 s_noise** | P2 | 高帧间方差(运动) → sigmoid(-σ×2) 降低 s_noise 置信度 |
| **blur_mask 门控 MRPN** | P2 | σ_t_clean + 帧间差异 → blur_estimator → 模糊区偏向邻帧补偿 |

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
        SACE_REF["μ_t_clean = lff_stack[center]<br/>σ_t_clean = lff_stack.std<br/>→ NDPN (SNR基线) + MRPN (运动感知) ★"]

        subgraph SACE_RWKV["多尺度 Bi-RWKV 帧间注意力"]
            RWKV_F["full: VRWKV ×2 (双向)"]
            RWKV_H["half: avg_pool3d(×2) → VRWKV ×2 → bilinear↑"]
            RWKV_Q["quarter: avg_pool3d(×4) → VRWKV ×2 → bilinear↑"]
            RWKV_FUSE["★ channel_mix (Charlie P1-1)<br/>concat[full,half,quarter](3C)<br/>→ Linear(3C→C) 可学习融合"]
        end

        SACE_GATE["边缘门控 + V raw<br/>edge_weight = edge_prompt(f_raw_center)<br/>F_aligned[t] = out[t] + (1-edge_w)·f_raw<br/>+ (1-s_noise)·f_raw"]
    end

    subgraph 三源处理["三源退化并行建模"]
        direction LR
        subgraph IFPN["IFPN 光照图估计"]
            IFPN_F["coarse_adapter → IllumExtract×T<br/>→ L_ratio → lit_up_map_raw [1,5]<br/>→ side_head → f_illum_feat"]
        end
        subgraph NDPN["NDPN 去噪 ★Charlie P0"]
            NDPN_SNR["SNR估计 (|μ|/σ)<br/>→ 双因素聚合权重"]
            NDPN_COND["★ s_noise 条件输入 (noise_proj)<br/>Conv2d(1→64) → 加法注入<br/>(替代 IGRF 直接注入)"]
            NDPN_SNR --> NDPN_COND
        end
        subgraph MRPN["MRPN 运动 ★Charlie P1+P2"]
            MRPN_CORR["窗口相关 → f_omega_aligned"]
            MRPN_BLUR["★ blur_estimator<br/>σ_t_clean + 帧间差异 → blur_mask<br/>(零初始化, 逐步学习)"]
            MRPN_GATE["门控融合<br/>g_t × blur_mask 调制<br/>→ Refine → f_motion"]
            MRPN_CORR --> MRPN_BLUR
            MRPN_BLUR --> MRPN_GATE
        end
    end

    subgraph IGRF["IGRF 修正去噪"]
        IGRF_S1["Stage1: δ=Fuse(f_noise,img)<br/>★ s_noise 已移除<br/>img_s1=clamp(img+δ)"]
        IGRF_S2["Stage2: δ=Fuse(f_motion, img_s1)<br/>img_s2=clamp(img_s1+δ)"]
        IGRF_S3["Stage3: res_t=clamp(img_s2×lit_up_map<br/>+ s_illum·corr_mag)"]
    end

    OUT["输出 res_t"]

    %% ── 数据流 ──
    IN --> ENC
    ENC --> DWT_C
    ENC --> DWT_N
    ENC -- "F_t (中心帧原始)" --> SACE_GATE
    DWT_C -- "F_lff_t, feat_tfsi" --> TFSI_FREQ
    DWT_N --> SACE_DWT
    DWT_C --> SACE_DWT
    TFSI_HEAD -- "s_illum" --> IGRF_S3
    TFSI_HEAD -- "s_noise + phase_conf" --> SACE_GATE
    TFSI_HEAD -- "s_noise ★ (P0)" --> NDPN_COND
    SACE_REF -- "σ_t_clean ★ (P1)" --> MRPN_BLUR
    SACE_REF -- "μ_t_clean" --> NDPN_SNR
    SACE_GATE -- "F_aligned_list" --> IFPN_F
    SACE_GATE -- "F_aligned_list" --> NDPN_COND
    SACE_GATE -- "F_aligned_list" --> MRPN_CORR
    IFPN_F -- "lit_up_map_raw" --> IGRF_S3
    IFPN_F -- "f_illum_feat" --> IGRF_S3
    NDPN_COND -- "f_noise_out" --> IGRF_S1
    MRPN_GATE -- "f_motion_out" --> IGRF_S2
    IN -- "image_center" --> IGRF_S1
    IN -- "image_down" --> IFPN_F
    IGRF_S1 --> IGRF_S2 --> IGRF_S3 --> OUT
```

### Charlie 数据流总览 (实施后)

```
TFSI → s_illum ──────────────────────────────────→ IGRF Stage3 (保留)
TFSI → s_noise ─┬─ (1-s_noise)·f_raw ──→ SACE V 残差    (保留)
                ├─ noise_proj (★P0) ──→ NDPN 条件输入  (新增)
                └─ 不进入 IGRF                          (P0: 移除)

SACE → σ_t_clean ─┬─ NDPN SNR 估计 (|μ|/σ)              (保留)
                   └─ blur_estimator (★P1) → MRPN        (新增)

SACE → sigma_sace → sigmoid(-σ×2) → s_noise 调制 (★P2)  (新增)
```

### Charlie vs Bravo 核心差异

| 维度 | Bravo | Charlie (实施后) |
|------|-------|-----------------|
| **FrequencyBranch** | 仅中心帧 LFF | ★ 多帧 temporal_fuse |
| **多尺度融合** | /3 等权平均 | ★ concat+channel_mix |
| **σ_t_clean 路由** | → NDPN only | → NDPN + ★MRPN (blur_mask) |
| **s_noise → IGRF** | ✓ Stage1 直接注入 | ★ 移除, 仅 NDPN 接收 |
| **s_noise 调制** | phase_conf only | ★ + sigma_sace 调制 (P2) |
| **MRPN 设计** | 简单 gate+ResBlock | ★ + blur_estimator (σ感知) |
