# TFS-Net v6 Bravo 整体架构图（2026-06-28）

## 图一：最简架构

```mermaid
flowchart TD
    subgraph 输入
        IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]
    end

    subgraph 诊断层
        ENC["Encoder<br/>3级金字塔编码"]
        DWT["DWT-LFF<br/>小波光照分离<br/>中心 α=0.6 / 邻居 α=0.4"]
        TFSI["TFSI<br/>时频诊断<br/>+ phase_conf"]
    end

    subgraph 对齐层
        SACE["SACE<br/>纯 RWKV 帧间注意力<br/>多尺度双向 Bi-WKV<br/>V 源 = Encoder 原始中心帧"]
    end

    subgraph 方案估计层
        IFPN["IFPN<br/>光照图估计"]
        NDPN["NDPN<br/>SNR自适应去噪"]
        MRPN["MRPN<br/>运动补偿"]
    end

    subgraph 执行层
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
    TFSI -- "s_noise" --> IGRF

    SACE -- "F_aligned" --> IFPN
    SACE -- "F_aligned, μ,σ" --> NDPN
    SACE -- "F_aligned" --> MRPN

    IFPN -- "lit_up_map, f_illum" --> IGRF
    NDPN -- "f_noise" --> IGRF
    MRPN -- "f_motion" --> IGRF

    IGRF --> OUT
    IN -- "img_center" --> IGRF
    IN -- "img_center" --> IFPN
```

## 图二：带细节架构

```mermaid
flowchart TD
    IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]

    subgraph Encoder
        ENC["PyramidEncoder<br/>[32,64,96] → 64ch<br/>逐帧编码 → F_stack"]
    end

    subgraph DWT["DWT-LFF 分裂 (Bravo P1-2)"]
        DWT_C["lff_center<br/>中心帧, α init=0.6<br/>→ F_lff_t + feat_tfsi"]
        DWT_N["lff_neighbor<br/>邻居帧, α init=0.4<br/>→ F_lff_Ω"]
    end

    subgraph TFSI["TFSI 时频诊断 (Bravo P0-2)"]
        TFSI_SPATIAL["SpatialBranch<br/>soft-median → μ,σ,SNR<br/>→ Conv → F_s"]
        TFSI_FREQ["FrequencyBranch<br/>lff_center → feat_tfsi → F_f"]
        TFSI_PHASE["phase_conf_head<br/>Conv→GELU→Conv→Sigmoid<br/>→ phase_conf (B,1,H,W)"]
        TFSI_HEAD["IntensityHead<br/>Conv(F_s∥F_f∥phase_conf)<br/>→ s_illum, s_noise<br/>s_noise×=(1+0.5·(1-phase_conf))"]
        TFSI_SPATIAL --> TFSI_HEAD
        TFSI_FREQ --> TFSI_PHASE
        TFSI_FREQ --> TFSI_HEAD
        TFSI_PHASE --> TFSI_HEAD
    end

    subgraph SACE["SACE 纯 RWKV 帧间注意力 (v6.5)"]
        direction TB
        SACE_DWT["DWT-LFF 逐帧<br/>center→lff_center<br/>neighbor→lff_neighbor"]
        SACE_REF["μ_t_clean = lff_stack[center]<br/>σ_t_clean = lff_stack.std"]
        
        subgraph SACE_RWKV["多尺度 Bi-RWKV 帧间注意力"]
            RWKV_F["full: VRWKV ×2 (双向)<br/>Bi-WKV scan: (u·k_t·v_t + Σe^w·k·v)/(u·k_t + Σe^w·k)"]
            RWKV_H["half: avg_pool3d(×2) → VRWKV ×2 → ↑"]
            RWKV_Q["quarter: avg_pool3d(×4) → VRWKV ×2 → ↑"]
            RWKV_AGG["out = (full + half + quarter) / 3"]
        end
        
        SACE_GATE["边缘门控 + V raw (Bravo P1)<br/>edge_weight = edge_prompt(f_raw_center)<br/>F_aligned[t] = out[t] + (1-edge_w)·f_raw_center + (1-s_noise)·f_raw_center"]

        SACE_DWT --> SACE_REF
        SACE_DWT --> SACE_RWKV
        SACE_RWKV --> SACE_GATE
    end

    subgraph IFPN["IFPN 光照图估计"]
        IFPN_F["coarse_adapter → IllumExtract×T<br/>→ L_ratio → lit_up_map_raw [1,5]<br/>→ side_head → ifpn_side"]
    end

    subgraph NDPN["NDPN 去噪"]
        NDPN_F["SNR估计 → 双因素权重<br/>→ 加权聚合 → f_noise"]
    end

    subgraph MRPN["MRPN 运动"]
        MRPN_F["窗口相关 → 门控融合<br/>→ ResBlock → f_motion"]
    end

    subgraph IGRF["IGRF 逆序修复"]
        IGRF_S1["Stage1: δ=Fuse(f_noise,img)+intensity_corr(s_noise)<br/>img_s1=clamp(img+δ)"]
        IGRF_S2["Stage2: δ=Fuse(f_motion, img_s1)<br/>img_s2=clamp(img_s1+δ)"]
        IGRF_S3["Stage3: res_t=clamp(img_s2×lit_up_map + s_illum·corr_mag)"]
        IGRF_S1 --> IGRF_S2 --> IGRF_S3
    end

    OUT["输出 res_t"]

    %% 数据流
    IN --> ENC
    ENC --> DWT_C
    ENC --> DWT_N
    ENC -- "F_t (中心帧原始)" --> SACE_GATE

    DWT_C -- "F_lff_t" --> TFSI_HEAD
    DWT_N --> SACE_DWT
    DWT_C --> SACE_DWT

    TFSI_HEAD -- "s_illum" --> IGRF_S3
    TFSI_HEAD -- "s_noise (+phase_conf调制)" --> SACE_GATE
    TFSI_HEAD -- "s_noise" --> IGRF_S1

    SACE_GATE -- "F_aligned_list" --> IFPN_F
    SACE_GATE -- "F_aligned_list" --> NDPN_F
    SACE_GATE -- "F_aligned_list" --> MRPN_F
    SACE_REF -- "μ_t_clean, σ_t_clean" --> NDPN_F

    IFPN_F -- "lit_up_map_raw" --> IGRF_S3
    IFPN_F -- "f_illum_feat" --> IGRF_S3
    NDPN_F -- "f_noise_out" --> IGRF_S1
    MRPN_F -- "f_motion_out" --> IGRF_S2

    IN -- "image_center" --> IGRF_S1
    IN -- "image_down" --> IFPN_F

    IGRF_S3 --> OUT
```

### SACE 帧间注意力详解

```
SACE 内部:
  F_stack (B,T,64,H,W)
    ├─ center frame → lff_center (α=0.6) → Q 源 (归一化锚定)
    ├─ neighbor frames → lff_neighbor (α=0.4) → K 源 (退化诊断)
    └─ f_raw_center (Encoder 原始) → V 源 (内容保留)
  
  多尺度 Bi-RWKV:
    full: VRWKVStyleSpatialMix(5帧, H×W) ×2 (前向+反向)
    half: avg_pool3d → VRWKV ×2 → bilinear ↑
    quarter: avg_pool3d → VRWKV ×2 → bilinear ↑
    out = (full + half + quarter) / 3
  
  残差:
    edge_weight = sigmoid(Conv(f_raw_center))
    F_aligned[t] = out[t] + (1-edge_weight)·f_raw_center + (1-s_noise)·f_raw_center
```

### 关键创新

| 创新 | 描述 | 来源 |
|---|---|---|
| **PureRWKV 替代 DAT** | 3尺度双向 Bi-WKV 替代可变形采样，省130K参数 | pureRWKV.md |
| **DWT-LFF 分裂** | 中心帧α=0.6(干净锚定) vs 邻居α=0.4(退化诊断) | VSRELL + STCD |
| **V 源 = Encoder 原始** | 对齐用归一化空间(Q/K)，内容用原始空间(V) | STCD |
| **phase_conf 调制 s_noise** | 相位不可靠→增强去噪 | FDN + FAN |
| **VGG/SSIM 主导损失** | L1降权0.3，VGG 0.8+SSIM 0.5主导 | BVI-Lowlight |
