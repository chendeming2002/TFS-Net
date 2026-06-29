# TFS-Net v6 Charlie 整体架构图（2026-06-29）

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
        SACE["SACE<br/>纯 RWKV 帧间注意力<br/>多尺度双向 Bi-WKV<br/>concat+channel_mix 融合<br/>V 源 = Encoder 原始中心帧"]
    end

    subgraph 处理层["三源退化并行建模"]
        IFPN["IFPN<br/>光照图估计"]
        NDPN["NDPN<br/>SNR自适应去噪<br/>+ s_noise 条件输入"]
        MRPN["MRPN<br/>运动补偿<br/>+ σ_t_clean 运动感知"]
    end

    subgraph 执行层["修正去噪"]
        IGRF["IGRF<br/>去噪→去模糊→提亮"]
    end

    OUT["输出 res_t"]

    IN --> ENC
    ENC --> DWT
    ENC -- "F_t (中心帧原始)" --> SACE
    DWT --> TFSI
    DWT --> SACE
    TFSI -- "s_illum" --> IGRF
    TFSI -- "s_noise (+phase_conf)" --> SACE
    TFSI -- "s_noise" --> NDPN
    TFSI -- "s_noise" --> IGRF

    SACE -- "F_aligned" --> IFPN
    SACE -- "F_aligned, σ_t_clean" --> MRPN
    SACE -- "F_aligned, μ_t_clean" --> NDPN

    IFPN -- "lit_up_map, f_illum" --> IGRF
    NDPN -- "f_noise" --> IGRF
    MRPN -- "f_motion" --> IGRF

    IGRF --> OUT
    IN -- "img_center" --> IGRF
    IN -- "img_center" --> IFPN
```

### 框架要点

- **三源分离估计**：DWT-LFF + TFSI 联合诊断光照(s_illum)和噪声(s_noise)；SACE 通过帧间方差(σ_t_clean)隐式感知运动
- **TFSI ↔ SACE 关系**：TFSI 的 FrequencyBranch 使用多帧邻居融合（Charlie P0-1）；SACE 使用 DWT-LFF 双实例（中心 α=0.6 / 邻居 α=0.4）做对齐前归一化；两者共享 DWT-LFF 核心思想但独立实现
- **三源并行处理**：IFPN/NDPN/MRPN 三个分支从 SACE 对齐特征各自估计修复方案；NDPN 接收 s_noise 作为条件输入，MRPN 接收 σ_t_clean 作为运动感知信号
- **SACE 纯 RWKV 注意力**：多尺度双向 Bi-WKV（full/half/quarter），concat+channel_mix 可学习融合（Charlie P1-1），V 源使用 Encoder 原始中心帧保留内容

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
        TFSI_HEAD["IntensityHead<br/>Conv(F_s||F_f||phase_conf)<br/>→ s_illum, s_noise<br/>s_noise×(1+0.5(1-phase_conf))"]
        TFSI_SPATIAL --> TFSI_HEAD
        TFSI_FREQ --> TFSI_PHASE
        TFSI_FREQ --> TFSI_HEAD
        TFSI_PHASE --> TFSI_HEAD
    end

    subgraph SACE["SACE 纯 RWKV 帧间注意力"]
        direction TB
        SACE_DWT["DWT-LFF 逐帧<br/>center→lff_center<br/>neighbor→lff_neighbor"]
        SACE_REF["μ_t_clean = lff_stack[center]<br/>σ_t_clean = lff_stack.std<br/>→ MRPN (运动感知)"]
        
        subgraph SACE_RWKV["多尺度 Bi-RWKV 帧间注意力"]
            RWKV_F["full: VRWKV ×2 (双向)"]
            RWKV_H["half: avg_pool3d(×2) → VRWKV ×2 → bilinear↑"]
            RWKV_Q["quarter: avg_pool3d(×4) → VRWKV ×2 → bilinear↑"]
            RWKV_FUSE["★ channel_mix<br/>concat[full,half,quarter](3C)<br/>→ Linear(3C→C) 可学习融合"]
        end
        
        SACE_GATE["边缘门控 + V raw<br/>edge_weight = edge_prompt(f_raw_center)<br/>F_aligned[t] = out[t] + (1-edge_w)·f_raw<br/>+ (1-s_noise)·f_raw"]
    end

    subgraph 三源处理["三源退化并行建模"]
        direction LR
        subgraph IFPN["IFPN 光照图估计"]
            IFPN_F["coarse_adapter → IllumExtract×T<br/>→ L_ratio → lit_up_map_raw [1,5]<br/>→ side_head → ifpn_side<br/>→ f_illum_feat"]
        end
        subgraph NDPN["NDPN 去噪 ★s_noise条件"]
            NDPN_F["SNR估计 (μ_t_clean/σ_t_clean)<br/>→ 双因素权重 + s_noise 条件<br/>→ 加权聚合 → f_noise"]
        end
        subgraph MRPN["MRPN 运动 ★σ_t_clean感知"]
            MRPN_F["窗口相关 → 门控融合<br/>+ σ_t_clean 运动置信<br/>→ ResBlock → f_motion"]
        end
    end

    subgraph IGRF["IGRF 逆序修复"]
        IGRF_S1["Stage1: δ=Fuse(f_noise,img)+s_noise_corr<br/>img_s1=clamp(img+δ)"]
        IGRF_S2["Stage2: δ=Fuse(f_motion, img_s1)<br/>img_s2=clamp(img_s1+δ)"]
        IGRF_S3["Stage3: res_t=clamp(img_s2×lit_up_map<br/>+ s_illum·corr_mag)"]
    end

    OUT["输出 res_t"]

    %% 数据流
    IN --> ENC
    ENC --> DWT_C
    ENC --> DWT_N
    ENC -- "F_t (中心帧原始)" --> SACE_GATE
    DWT_C -- "F_lff_t, feat_tfsi" --> TFSI_FREQ
    DWT_N --> SACE_DWT
    DWT_C --> SACE_DWT
    TFSI_HEAD -- "s_illum" --> IGRF_S3
    TFSI_HEAD -- "s_noise + phase_conf" --> SACE_GATE
    TFSI_HEAD -- "s_noise" --> NDPN_F
    TFSI_HEAD -- "s_noise" --> IGRF_S1
    SACE_REF -- "σ_t_clean" --> MRPN_F
    SACE_REF -- "μ_t_clean" --> NDPN_F
    SACE_GATE -- "F_aligned_list" --> IFPN_F
    SACE_GATE -- "F_aligned_list" --> NDPN_F
    SACE_GATE -- "F_aligned_list" --> MRPN_F
    IFPN_F -- "lit_up_map_raw" --> IGRF_S3
    IFPN_F -- "f_illum_feat" --> IGRF_S3
    NDPN_F -- "f_noise_out" --> IGRF_S1
    MRPN_F -- "f_motion_out" --> IGRF_S2
    IN -- "image_center" --> IGRF_S1
    IN -- "image_down" --> IFPN_F
    IGRF_S1 --> IGRF_S2 --> IGRF_S3 --> OUT
```

### Charlie vs Bravo 核心差异

| 维度 | Bravo | Charlie |
|---|---|---|
| **FrequencyBranch** | 仅中心帧 LFF | ★ 多帧：中心 LFF + 邻帧时域均值 → temporal_fuse |
| **多尺度融合** | /3 等权平均 | ★ concat+channel_mix (Linear 3C→C) |
| **σ_t_clean 路由** | → NDPN | ★ → MRPN (运动感知)，NDPN 保留 μ_t_clean |
| **s_noise 路由** | → SACE + IGRF | → SACE + **NDPN** (条件输入) + IGRF |

### SACE 纯 RWKV 注意力详解

```
F_stack (B,T,64,H,W)
  ├─ center frame → lff_center (α=0.6) → Q/V reference
  ├─ neighbor frames → lff_neighbor (α=0.4) → K source
  └─ f_raw_center (Encoder 原始) → V source (content)

多尺度 Bi-RWKV (Charlie P1-1):
  full → VRWKV ×2 (fwd+bwd)
  half → avg_pool3d(1,2,2) → VRWKV ×2 → bilinear↑
  quarter → avg_pool3d(1,4,4) → VRWKV ×2 → bilinear↑
  out = channel_mix(concat[full, half, quarter])  ← 可学习融合

边缘门控残差:
  edge_weight = sigmoid(Conv(f_raw_center))
  F_aligned[t] = out[t] + (1-edge_weight)·f_raw_center + (1-s_noise)·f_raw_center
```

### 损失函数

```
L_total = 1.0·Charbonnier(res, GT) + 0.04·VGG(relu3_3) + 0.2·(1-SSIM)
        + 0.05·L_freq + 0.001·L_illum_smooth + 0.02·L_illum_sup
        + 0.2·L_inter + 0.1·L_ifpn_sup
```
