# TFS-Net v5.7 整体架构图（s_illum 门控提亮，2026-06-22）

## 图一：最简架构（概括框架）

```mermaid
flowchart TD
    subgraph 输入
        IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]
    end

    subgraph 诊断层
        ENC["Encoder<br/>多帧金字塔编码"]
        TFSI["TFSI<br/>时频源指示器"]
        SACE["SACE<br/>跨帧对齐增强"]
    end

    subgraph 方案估计层_三源并行
        IFPN["IFPN<br/>光照图估计"]
        NDPN["NDPN<br/>SNR自适应去噪"]
        MRPN["MRPN<br/>运动补偿"]
    end

    subgraph 执行层
        IGRF["IGRF<br/>逆序级联修复"]
    end

    OUT["输出 res_t : (B, 3, H, W)"]

    IN --> ENC --> TFSI
    ENC --> SACE
    TFSI -- "s_illum, s_noise" --> IGRF
    TFSI -- "s_noise" --> SACE
    SACE -- "F_aligned" --> IFPN
    SACE -- "F_aligned, μ, σ" --> NDPN
    SACE -- "F_aligned" --> MRPN
    IFPN -- "lit_up_map, f_illum" --> IGRF
    NDPN -- "f_noise" --> IGRF
    MRPN -- "f_motion" --> IGRF
    IGRF --> OUT
    IN -- "image_center" --> IGRF
```

### 框架要点

- **三源分离估计**：TFSI 从多帧特征诊断两种退化强度（s_illum 光照、s_noise 噪声），运动强度由 MRPN 隐式感知
- **三源处理**：IFPN/NDPN/MRPN 三个分支并行，各自从 SACE 对齐特征估计修复方案
- **修正去噪**：IGRF 按物理逆序执行修复——去噪→去模糊→提亮，s_illum 门控 lit_up_map（s_illum=0→无提亮→模型被迫让 s_illum>0），s_noise 作加法修正参与去噪
- **TFSI 与 SACE 的关系**：共享 LFF 频域模块；TFSI 用 LFF 诊断退化强度，SACE 用 LFF 做光照归一化后再对齐

---

## 图二：带细节的架构（模块内部数据流）

```mermaid
flowchart TD
    subgraph 输入
        IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]
    end

    subgraph Encoder
        ENC["PyramidEncoder<br/>3级金字塔 [32,64,96]<br/>逐帧编码 → 融合特征<br/>feats : (B, T, 64, H, W)"]
    end

    subgraph TFSI["TFSI 时频源指示器"]
        direction TB
        TFSI_NORM["LayerNorm2d<br/>逐帧归一化"]
        TFSI_SPATIAL["SpatialBranch<br/>soft_median → μ_t, σ_t, SNR<br/>→ Conv → F_s"]
        TFSI_FREQ["FrequencyBranch<br/>LFF(中心帧) → F_f"]
        TFSI_FUSE["ConcatFusion<br/>Conv[F_s ∥ F_f] → F_fused"]
        TFSI_HEAD["IntensityHead<br/>Conv1x1 → Sigmoid<br/>→ s_illum, s_noise"]

        TFSI_NORM --> TFSI_SPATIAL
        TFSI_NORM --> TFSI_FREQ
        TFSI_SPATIAL --> TFSI_FUSE
        TFSI_FREQ --> TFSI_FUSE
        TFSI_FUSE --> TFSI_HEAD
    end

    subgraph SACE["SACE 跨帧对齐增强"]
        direction TB
        SACE_LFF["LFF 逐帧光照归一化<br/>(共享 TFSI 的 LFF)"]
        SACE_REF["soft-median 参考帧<br/>μ_t_clean, σ_t_clean"]
        SACE_DEFORM["DeformableCrossAttention<br/>OffsetMaskHead 生成 offset+mask<br/>→ grid_sample 采样 → 加权聚合"]
        SACE_GATE["噪声感知残差门控<br/>F_aligned = DeformAttn + (1-s_noise)·F_lff"]

        SACE_LFF --> SACE_REF
        SACE_REF --> SACE_DEFORM
        SACE_DEFORM --> SACE_GATE
    end

    subgraph IFPN["IFPN 光照图估计"]
        direction TB
        IFPN_ADAPTER["coarse_adapter<br/>F_aligned → 粗特征"]
        IFPN_ILLUM["IllumExtract ×T帧<br/>img + feat → L_t, L_i"]
        IFPN_RATIO["L_ratio = L_ref / L_t<br/>帧间相似度加权"]
        IFPN_LITUP["lit_up_map_raw = 1 + 4·σ(L_ratio + Δ)<br/>有界 [1, 5]"]
        IFPN_FEAT["f_illum_feat = Refine(ratio_feat)"]

        IFPN_ADAPTER --> IFPN_ILLUM
        IFPN_ILLUM --> IFPN_RATIO
        IFPN_RATIO --> IFPN_LITUP
        IFPN_RATIO --> IFPN_FEAT
    end

    subgraph NDPN["NDPN SNR自适应去噪"]
        direction TB
        NDPN_SNR["SNR估计<br/>s_SNR = σ(|μ_t_clean| / σ_t_clean)"]
        NDPN_ALPHA["双因素权重<br/>α_t = s_SNR (中心帧)<br/>α_i = σ(Conv(残差))·(1-s_SNR) (邻帧)"]
        NDPN_AGG["加权聚合 + Refine<br/>f_noise = Refine(Σ α_i·F_aligned_i)"]

        NDPN_SNR --> NDPN_ALPHA
        NDPN_ALPHA --> NDPN_AGG
    end

    subgraph MRPN["MRPN 运动补偿"]
        direction TB
        MRPN_CORR["窗口 dot-product 相关<br/>corr = softmax(f_t · f_nbrs^T / √C)"]
        MRPN_AGG["相关聚合邻帧<br/>f_omega = corr · f_nbrs"]
        MRPN_GATE["门控融合 + 残差精炼<br/>g = σ(Conv[f_t ∥ f_omega])<br/>f_motion = ResBlock(g·f_t + (1-g)·f_omega) + f_t"]

        MRPN_CORR --> MRPN_AGG
        MRPN_AGG --> MRPN_GATE
    end

    subgraph IGRF["IGRF 逆序级联修复"]
        direction TB
        IGRF_S1["Stage 1 : 暗域降噪<br/>δ_noise = Fuse(f_noise, img) + intensity_corr(s_noise)<br/>img_s1 = clamp(img + δ_noise)"]
        IGRF_S2["Stage 2 : 运动去模糊<br/>δ_motion = Fuse(f_motion, img_s1)<br/>img_s2 = clamp(img_s1 + δ_motion)"]
        IGRF_S3["Stage 3 : s_illum 门控提亮 ⬅ v5.7 新设计<br/>lit_up_map_full = lit_up_map_raw × (1 + tanh(Δ) × 0.5)<br/>lit_up_map = 1 + s_illum × (lit_up_map_full − 1)<br/>res_t = clamp(img_s2 × lit_up_map)"]

        IGRF_S1 --> IGRF_S2
        IGRF_S2 --> IGRF_S3
    end

    OUT["输出 res_t : (B, 3, H, W)"]

    %% 主数据流
    IN --> ENC
    ENC -- "feats (B,T,64,H,W)" --> TFSI_NORM
    ENC -- "feats" --> SACE_LFF

    %% TFSI → SACE/IGRF
    TFSI_HEAD -- "s_illum (门控提亮强度)" --> IGRF_S3
    TFSI_HEAD -- "s_noise" --> SACE_GATE
    TFSI_HEAD -- "s_noise (加法修正)" --> IGRF_S1

    %% SACE → 三分支
    SACE_GATE -- "F_aligned_list" --> IFPN_ADAPTER
    SACE_GATE -- "F_aligned_list" --> NDPN_AGG
    SACE_GATE -- "F_aligned_list" --> MRPN_CORR
    SACE_REF -- "μ_t_clean, σ_t_clean" --> NDPN_SNR

    %% 三分支 → IGRF
    IFPN_LITUP -- "lit_up_map_raw" --> IGRF_S3
    IFPN_FEAT -- "f_illum_feat" --> IGRF_S3
    NDPN_AGG -- "f_noise_out" --> IGRF_S1
    MRPN_GATE -- "f_motion_out" --> IGRF_S2

    %% 原始图像
    IN -- "image_center" --> IGRF_S1
    IN -- "image_center (下采样)" --> IFPN_ILLUM

    %% 输出
    IGRF_S3 --> OUT
```

### v5.7 核心改动：s_illum 门控提亮（Stage 3）

| 版本 | Stage 3 公式 | s_illum 角色 | 问题 |
|------|-------------|-------------|------|
| v5.5 (加法修正) | `res_t = img_s2 × lit_up_map + s_illum × corr_mag` | 加法补正 (可旁路) | s_illum 塌缩到 0, lit_up_map 独自完成提亮 |
| **v5.7 (乘法门控)** | `lit_up_map = 1 + s_illum × (lit_up_map_full − 1)`<br/>`res_t = img_s2 × lit_up_map` | **门控因子 (不可旁路)** | s_illum=0→无提亮→高损失→**模型被迫让 s_illum>0** |

**门控机制的关键性质**：
- `lit_up_map_full` 提供"提亮什么"（内容感知的空间模式，来自 IFPN）
- `s_illum` 提供"提亮多少"（诊断驱动的强度门控，来自 TFSI）
- `s_illum=0` → `lit_up_map=1` → `res_t=img_s2`（无提亮）→ 重建损失高 → **s_illum 不能为 0**
- `s_illum=1` → `lit_up_map=lit_up_map_full`（完全提亮）
- 梯度：`∂res_t/∂s_illum = img_s2 × (lit_up_map_full − 1)`（非零，无衰减）

### 关键关系说明

| 关系 | 说明 |
|------|------|
| **TFSI ↔ SACE 共享 LFF** | LFF 是频域幅度整形模块。TFSI 用它从中心帧提取频域特征 F_f 供诊断；SACE 用它对每帧做光照归一化（低频抑制），使后续对齐不受光照差异干扰 |
| **TFSI → IGRF s_illum 门控** | s_illum 门控 lit_up_map：lit_up_map=1+s_illum×(lit_up_map_full−1)。s_illum=0→无提亮→模型被迫让 s_illum>0，消除功能冗余 |
| **TFSI → IGRF s_noise 加法** | s_noise 直入 Stage 1 作加法修正 intensity_corr(s_noise)，不经中间分支，避免梯度衰减 |
| **TFSI → SACE 门控** | s_noise 传入 SACE 的残差门控：高噪区 (s_noise→1) 抑制残差（噪声不穿透），低噪区保留信息流 |
| **三分支并行** | IFPN/NDPN/MRPN 各自从 SACE 对齐特征估计修复方案，不接收强度先验（纯数据驱动），互不依赖 |
| **IGRF 逆序执行** | 物理逆序：去噪(逆 n_t) → 去模糊(逆 k_t) → 提亮(逆 γ_t)。Stage 1/2 硬 clamp，最终输出硬 clamp |
| **SACE 可变形对齐** | OffsetMaskHead 从 [μ_t_clean ∥ F_lff] 预测 offset 和 mask；DeformableCrossAttention 在 offset 位置 grid_sample 采样，mask softmax 加权聚合。无 QK 点积，mask 即注意力权重 |
