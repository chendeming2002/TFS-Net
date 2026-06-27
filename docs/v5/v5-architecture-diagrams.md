# TFS-Net v6 整体架构图（含 Cross-RWKV Gate 帧间注意力，2026-06-27）

## 图一：最简架构（概括框架）

```mermaid
flowchart TD
    subgraph 输入
        IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]
    end

    subgraph 诊断层
        ENC["Encoder<br/>多帧金字塔编码"]
        TFSI["TFSI<br/>时频源指示器<br/>(共享 LFF)"]
        SACE["SACE<br/>跨帧对齐增强<br/>可变形注意力 + RWKV"]
    end

    subgraph 方案估计层_三源并行
        IFPN["IFPN<br/>光照图估计"]
        NDPN["NDPN<br/>SNR自适应去噪"]
        MRPN["MRPN<br/>运动补偿"]
    end

    subgraph 执行层
        IGRF["IGRF<br/>逆序级联修复<br/>去噪→去模糊→提亮"]
    end

    OUT["输出 res_t : (B, 3, H, W)"]

    IN --> ENC --> TFSI
    ENC --> SACE
    TFSI -- "s_illum" --> IGRF
    TFSI -- "s_noise" --> SACE
    TFSI -- "s_noise" --> IGRF
    TFSI == "共享 LFF" ==> SACE

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
- **TFSI ↔ SACE 共享 LFF**：LFF 是频域幅度整形模块。TFSI 用它从中心帧提取频域诊断特征；SACE 用它逐帧做光照归一化后再对齐
- **SACE 两阶段注意力**：第一阶段 DeformableCrossAttention 做空间精对齐（4组×9采样点），第二阶段 Cross-RWKV Gate 做多帧长程聚合（Vision-RWKV Bi-WKV + spatial_first）
- **三源处理**：IFPN/NDPN/MRPN 三个分支并行，各自从 SACE 对齐特征估计修复方案
- **IGRF 逆序执行**：物理逆序——去噪(n_t) → 去模糊(k_t) → 提亮(γ_t)，s_illum/s_noise 直入指导执行强度

---

## 图二：带细节架构（SACE 内部展开 + 全数据流）

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
        TFSI_FREQ["LFF 频域整形<br/>(与 SACE 共享)<br/>中心帧 → F_f"]
        TFSI_SPATIAL["SpatialBranch<br/>soft_median → μ_t, σ_t, SNR<br/>→ Conv → F_s"]
        TFSI_FUSE["ConcatFusion<br/>[F_s ∥ F_f] → F_fused"]
        TFSI_HEAD["IntensityHead<br/>Conv1x1 → Sigmoid<br/>→ s_illum, s_noise"]
        TFSI_SPATIAL --> TFSI_FUSE
        TFSI_FREQ --> TFSI_FUSE
        TFSI_FUSE --> TFSI_HEAD
    end

    subgraph SACE["SACE 跨帧对齐增强"]
        direction TB
        SACE_LFF["LFF 逐帧光照归一化<br/>(共享 TFSI 的 LFF)"]
        SACE_REF["soft-median 参考帧<br/>μ_t_clean, σ_t_clean"]
        
        subgraph SACE_ATTN["帧间注意力模块"]
            direction TB
            SACE_DEFORM["DeformableCrossAttention<br/>OffsetMaskHead → offset+mask<br/>→ grid_sample 采样<br/>→ mask-softmax 加权聚合"]
            SACE_RWKV["★ Cross-RWKV Gate (v6)<br/>Vision-RWKV SpatialMix<br/>Q-Shift 通道位移<br/>Bi-WKV 含 spatial_first<br/>per-channel decay<br/>time_mix 可学习混合比"]
        end
        
        SACE_GATE["噪声感知残差门控<br/>F_aligned = DeformAttn<br/>+ RWKV-enhanced<br/>+ (1-s_noise)·F_lff"]

        SACE_LFF --> SACE_REF
        SACE_REF --> SACE_DEFORM
        SACE_DEFORM --> SACE_RWKV
        SACE_RWKV --> SACE_GATE
    end

    subgraph IFPN["IFPN 光照图估计"]
        direction TB
        IFPN_ADAPTER["coarse_adapter<br/>F_aligned → 粗特征"]
        IFPN_ILLUM["IllumExtract ×T帧<br/>img + feat → L_t, L_i"]
        IFPN_RATIO["L_ratio = L_ref / L_t<br/>帧间相似度加权"]
        IFPN_LITUP["lit_up_map_raw = 1 + 4·σ(L_ratio + Δ)"]
        IFPN_SIDE["side_head 中间监督<br/>f_illum_feat → 低分辨率图像<br/>→ L_ifpn_sup 监督 GT↓"]
        IFPN_ADAPTER --> IFPN_ILLUM
        IFPN_ILLUM --> IFPN_RATIO
        IFPN_RATIO --> IFPN_LITUP
        IFPN_RATIO --> IFPN_SIDE
    end

    subgraph NDPN["NDPN SNR自适应去噪"]
        direction TB
        NDPN_SNR["SNR估计<br/>s_SNR = σ(|μ_t_clean| / σ_t_clean)"]
        NDPN_ALPHA["双因素权重<br/>α_t = s_SNR (中心帧)<br/>α_i = σ(Conv)·(1-s_SNR) (邻帧)"]
        NDPN_AGG["加权聚合 + Refine<br/>f_noise = Refine(Σ α_i·F_aligned_i)"]
        NDPN_SNR --> NDPN_ALPHA
        NDPN_ALPHA --> NDPN_AGG
    end

    subgraph MRPN["MRPN 运动补偿"]
        direction TB
        MRPN_CORR["窗口 dot-product 相关<br/>corr = softmax(f_t · f_nbrs / √C)"]
        MRPN_GATE["门控融合 + 残差精炼<br/>g = σ(Conv[f_t ∥ f_agg])<br/>f_motion = ResBlock(g·f_t + (1-g)·f_agg) + f_t"]
        MRPN_CORR --> MRPN_GATE
    end

    subgraph IGRF["IGRF 逆序级联修复"]
        direction TB
        IGRF_S1["Stage 1 : 暗域降噪<br/>δ = Fuse(f_noise, img) + intensity_corr(s_noise)<br/>img_s1 = clamp(img + δ)"]
        IGRF_S2["Stage 2 : 运动去模糊<br/>δ = Fuse(f_motion, img_s1)<br/>img_s2 = clamp(img_s1 + δ)"]
        IGRF_S3["Stage 3 : 混合提亮<br/>lit_up_map = Refine(lit_up_map_raw, f_illum)<br/>res_t = clamp(img_s2 × lit_up_map + s_illum × corr_mag)"]
        IGRF_S1 --> IGRF_S2
        IGRF_S2 --> IGRF_S3
    end

    OUT["输出 res_t : (B, 3, H, W)"]

    %% === 主数据流 ===
    IN --> ENC
    ENC -- "feats" --> TFSI_SPATIAL
    ENC -- "feats" --> SACE_LFF
    ENC -- "center frame" --> TFSI_FREQ

    %% TFSI → 下游
    TFSI_HEAD -- "s_illum (光照强度)" --> IGRF_S3
    TFSI_HEAD -- "s_noise" --> SACE_GATE
    TFSI_HEAD -- "s_noise" --> IGRF_S1

    %% SACE → 三分支
    SACE_GATE -- "F_aligned_list" --> IFPN_ADAPTER
    SACE_GATE -- "F_aligned_list" --> NDPN_AGG
    SACE_GATE -- "F_aligned_list" --> MRPN_CORR
    SACE_REF -- "μ_t_clean, σ_t_clean" --> NDPN_SNR

    %% 三分支 → IGRF
    IFPN_LITUP -- "lit_up_map_raw" --> IGRF_S3
    IFPN_SIDE -- "f_illum_feat" --> IGRF_S3
    NDPN_AGG -- "f_noise_out" --> IGRF_S1
    MRPN_GATE -- "f_motion_out" --> IGRF_S2

    %% 原始图像
    IN -- "image_center" --> IGRF_S1
    IN -- "image_center (下采样)" --> IFPN_ILLUM

    %% loss 监督
    IFPN_SIDE -. "L_ifpn_sup" .-> OUT
    TFSI_HEAD -. "L_illum_sup" .-> OUT

    %% 输出
    IGRF_S3 --> OUT
```

### 关键模块说明

| 模块 | 核心机制 | 关键参数 |
|---|---|---|
| **Encoder** | 3 级金字塔编码，无归一化 Conv 主干 | fused=64ch, H×W 全分辨率 |
| **TFSI** | 双分支诊断：空间(中位值/方差/SNR) + 频域(LFF)，独立 Sigmoid 输出 | s_illum, s_noise ∈ [0,1] |
| **SACE DeformAttn** | 4 组 × 3×3 采样点 = 36 个偏移采样位置，offset+mask 直接预测 | n_groups=4, kernel_size=3 |
| **SACE Cross-RWKV** | Vision-RWKV Bi-WKV 含 spatial_first + time_mix 可学习混合比，per-channel decay | T=5 帧, C=64 通道 |
| **IFPN** | Retinex 光照估计 + L_ratio 锚点 + 特征 delta 修正 + side_head 中间监督 | lit_up_map ∈ [1,5] |
| **NDPN** | SNR 自适应双因素聚合权重 + Refine | s_SNR 控制中心帧置信度 |
| **MRPN** | 窗口 dot-product 相关 + 门控融合 + 残差精炼 | window_size=8 |
| **IGRF Stage1** | 暗域降噪 + s_noise 加法修正 | intensity_corr 零初始化 |
| **IGRF Stage2** | 运动去模糊，无强度先验 | — |
| **IGRF Stage3** | 混合提亮：乘法 lit_up_map + s_illum 加法修正 | illum_corr 零初始化 |

### TFSI ↔ SACE 关系

| 关系 | 说明 |
|---|---|
| **共享 LFF** | TFSI FrequencyBranch 和 SACE 使用**同一 LFF 模块实例**。TFSI 用 LFF 对中心帧提取频域诊断特征 F_f；SACE 用 LFF 对每帧做光照归一化后再对齐 |
| **s_noise 门控** | TFSI 输出的 s_noise 传入 SACE 的噪声感知残差门控：高噪区抑制残差（噪声不透穿），低噪区保留信息流 |
| **s_illum 直入 IGRF** | TFSI 输出的 s_illum 直入 IGRF Stage 3 做加法修正，不经 IFPN（避免通道稀释） |

### SACE 帧间注意力（v6）

```
SACE 内部数据流:
  feats → [LFF 归一化] → soft-median(μ_t_clean)
       → [DeformableCrossAttention]
            OffsetMaskHead → offset + mask → grid_sample 采样 → mask-softmax 加权
       → [Cross-RWKV Gate] ★ v6
            已对齐的 F_aligned → Q-Shift 通道位移
            → time_mix(xk = x·mix_k + shifted·(1-mix_k))
            → Bi-WKV: (u·k_t·v_t + Σ ew·k·v) / (u·k_t + Σ ew·k)
            → sigmoid(r) 门控 → output → +残差
       → F_aligned = DeformAttn + RWKV_enhanced + (1-s_noise)·F_lff
```

**Cross-RWKV Gate 关键设计**：
- 继承 Vision-RWKV (ICLR 2025) 的 Bi-WKV + Q-Shift + VRWKV_SpatialMix
- `spatial_first(u)` 给当前帧额外 boost 权重
- `spatial_mix_k/v/r` 可学习混合比（当前 vs 历史特征）
- per-channel decay 从 -5 到 +3 指数分布
- 零初始化输出投影 → 初始恒等 = 完全保留 v5.9.2 预训练质量
