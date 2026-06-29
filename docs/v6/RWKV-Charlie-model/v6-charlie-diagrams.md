# TFS-Net v6 Charlie 整体架构图

## 图一：最简架构

```mermaid
flowchart TD
    subgraph 输入
        IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]
    end

    subgraph 编码层
        ENC["Encoder<br/>3级逐帧编码<br/>C32→64→128"]
    end

    subgraph 三源分离估计
        DWT["DWT-LFF<br/>Haar小波光照分离<br/>α 自适应 LL 拆分为<br/>feat_sace (光照/结构) 与<br/>feat_tfsi (噪声/退化)"]
    end

    subgraph 三源退化并行建模
        direction LR
        TFSI["TFSI 路径<br/>时频交互<br/>temporal_fuse + FFT"]
        SACE["SACE 路径<br/>多尺度帧间注意<br/>concat+channel_mix 融合"]
    end

    subgraph 融合修正
        FUSE["特征融合<br/>feat_sace + feat_tfsi"]
        CRWKV["Cross-RWKV<br/>Bi-WKV 跨帧扫描<br/>post_norm + clamp 稳定化"]
        CSA["Cross-Scale<br/>Attention"]
    end

    subgraph 解码层
        DEC["Decoder<br/>PixelShuffle 逐层上采样<br/>+ DWT-LFF 跳跃连接"]
    end

    OUT["输出增强帧<br/>(B×T×3×H×W)"]

    IN --> ENC
    ENC --> DWT
    DWT -- "feat_sace" --> SACE
    DWT -- "feat_tfsi" --> TFSI
    TFSI --> FUSE
    SACE --> FUSE
    FUSE --> CRWKV
    CRWKV --> CSA
    CSA --> DEC
    DEC --> OUT
    ENC -- "f2 skip (DWT-LFF)" --> DEC
    ENC -- "f1 skip" --> DEC
```

## 图二：带细节架构

```mermaid
flowchart TD
    IN["多帧低光输入<br/>x : (B, T, 3, H, W)"]

    subgraph Encoder["Encoder 编码器 (3级金字塔)"]
        ENC_H["head: Conv2d(3→32) + ConvBlock"]
        ENC_D2["conv_down2: ConvBlock(stride2) → 64"]
        ENC_D4["conv_down4: ConvBlock(stride2) → 128"]
        ENC_H --> ENC_D2 --> ENC_D4
    end

    subgraph DWT["DWT-LFF 三源分离估计"]
        DWT_D["HaarDWT2D<br/>→ LL, LH, HL, HH"]
        DWT_A["illum_alpha<br/>Conv→GELU→Conv→Sigmoid<br/>LL_ref = α·LL (光照/结构)<br/>LL_deg = (1-α)·LL (噪声/退化)"]
        DWT_S["feat_sace<br/>IDWT(LL_ref, LH, HL, HH)"]
        DWT_T["feat_tfsi<br/>IDWT(LL_deg, LH, HL, HH)"]
        DWT_D --> DWT_A
        DWT_A --> DWT_S
        DWT_A --> DWT_T
    end

    subgraph TFSI["TFSI 路径: 时频交互 (噪声/退化建模)"]
        direction TB
        TFSI_SFI["SpatialFreqInteraction<br/>FFT → 幅度增强(相位保留)<br/>→ IFFT"]
        subgraph TFSI_FB["FrequencyBranch (Charlie)"]
            TFSI_TF["temporal_fuse<br/>Conv3d 中心+邻帧均值融合"]
            TFSI_FFT["LayerNorm → Conv → FFT<br/>→ 幅度增益 (α·tanh)<br/>→ IFFT"]
            TFSI_TF --> TFSI_FFT
        end
        TFSI_OUT["TFSI 输出: f4_tfsi<br/>(B, T, 128, H/4, W/4)"]
        TFSI_SFI --> TFSI_OUT
        TFSI_FB --> TFSI_OUT
    end

    subgraph SACE["SACE 路径: 多尺度帧间注意 (光照/结构建模)"]
        direction TB
        SACE_IN["sace_in ×3<br/>3组并行投影头<br/>建模三重退化源"]

        subgraph SACE_ATTN["Bi-RWKV 帧间注意力"]
            direction TB
            RWKV_S["Q-Shift → Time Mix<br/>mix_k/v/r 可学习混合"]
            RWKV_WKV["Bi-WKV 跨帧扫描<br/>y = (u·k·v + Σe^w·k·v) / (u·k + Σe^w·k)<br/>post_norm(LayerNorm)<br/>spatial_decay.clamp(-8,8)<br/>spatial_first.clamp(-5,5)"]
            RWKV_S --> RWKV_WKV
        end

        subgraph SACE_3SRC["三源退化并行处理"]
            SRC1["Head 1<br/>光照退化估计"]
            SRC2["Head 2<br/>噪声退化估计"]
            SRC3["Head 3<br/>纹理退化估计"]
        end

        SACE_MULTI["多尺度处理<br/>full: H×W<br/>half: avg_pool ×2<br/>quarter: avg_pool ×4"]

        SACE_FUSE["concatenation + channel_mix (Charlie)<br/>cat(full, half↑, quarter↑) → Conv → 融合"]

        SACE_OUT["sace_out ×3<br/>3组输出头 → stack → mean<br/>→ f4_sace (B, T, 128, H/4, W/4)"]

        SACE_IN --> SACE_3SRC
        SACE_3SRC --> SACE_MULTI
        SACE_MULTI --> SACE_ATTN
        SACE_ATTN --> SACE_FUSE
        SACE_FUSE --> SACE_OUT
    end

    subgraph 融合层["融合 & 修正"]
        FUSE["f4_fused = f4_sace + f4_tfsi"]

        subgraph CRWKV["Cross-RWKV (帧间上下文聚合)"]
            CRWKV_Q["Q-Shift → Time Mix"]
            CRWKV_WKV["Bi-WKV 跨帧扫描<br/>post_norm + clamp"]
            CRWKV_Q --> CRWKV_WKV
        end

        CSA["CrossScaleAttention<br/>MultiheadAttention<br/>跨空间位置自注意"]
    end

    subgraph Decoder["Decoder 解码器"]
        DEC_DWT["DWT-LFF skip (f2)<br/>64ch, 全高频"]
        DEC_U2["conv_up2<br/>128→64→32 (PixelShuffle 2×)"]
        DEC_U4["conv_up4<br/>64→32→16 (PixelShuffle 2×)"]
        DEC_TAIL["tail<br/>ConvBlock(48→3) → Conv2d(3→3)"]
        DEC_DWT --> DEC_U2
        DEC_U2 --> DEC_U4
        DEC_U4 --> DEC_TAIL
    end

    OUT["输出增强帧<br/>(B×T×3×H×W)"]

    %% ── 数据流 ──
    IN --> ENC_H

    ENC_D4 --> DWT_D
    ENC_D2 --> DEC_DWT
    ENC_H -- "f1 skip" --> DEC_TAIL

    DWT_T --> TFSI_SFI
    DWT_T --> TFSI_FB
    DWT_S --> SACE_IN

    TFSI_OUT --> FUSE
    SACE_OUT --> FUSE

    FUSE --> CRWKV_Q
    CRWKV_WKV --> CSA
    CSA --> DEC_U2

    DEC_TAIL --> OUT
```

### SACE 三源退化并行建模 详解

```
SACE 内部 (Charlie 关键改动):

  feat_sace (B, T, 128, H/4, W/4)
    │
    ├─ sace_in ×3 (3组独立投影头)
    │     ├─ Head 1: Conv(dw)→Conv→GELU  → 光照退化估计
    │     ├─ Head 2: Conv(dw)→Conv→GELU  → 噪声退化估计
    │     └─ Head 3: Conv(dw)→Conv→GELU  → 纹理退化估计
    │
    ├─ 多尺度 Bi-RWKV 帧间注意力 (per head)
    │     full:  VRWKVStyleSpatialMix(H×W, 5帧) + post_norm + clamp
    │     half:  avg_pool(x2) → RWKV → bilinear ↑
    │     qtr:   avg_pool(x4) → RWKV → bilinear ↑
    │
    ├─ 多尺度融合 (Charlie P1-1)
    │     cat(full, half↑, qtr↑) → channel_mix(3C → C)    ← 替代 /3 平均
    │
    └─ sace_out ×3 → stack → mean → f4_sace
```

### 整体框架：三源分离估计 → 三源退化并行建模 → 修正去噪

| 阶段 | 模块 | 功能 |
|------|------|------|
| **Phase 1: 三源分离估计** | DWT-LFF (Haar + α门控) | 小波分解 4子带, α 自适应分配 LL 到光照/退化两路; HF (LH/HL/HH) 全共享给两分支 |
| **Phase 2: 三源退化并行建模** | TFSI (时频路径) + SACE (空域路径) | TFSI: temporal_fuse 多帧融合 + FFT 频域增强 建模噪声/退化; SACE: 3头多尺度 RWKV 帧间注意力 建模光照/噪声/纹理三重退化 |
| **Phase 3: 修正去噪** | Fusion + Cross-RWKV + Decoder | 特征相加融合 → Bi-WKV 跨帧修正 → CSA 空间自注意 → PixelShuffle 解码 (含 DWT-LFF skip) |

### Charlie vs Bravo 架构差异

| 模块 | Bravo | Charlie |
|------|-------|---------|
| TFSI F.B. | 单帧 LFF + phase_conf | 多帧 temporal_fuse (Conv3d) |
| SACE 融合 | /3 等权平均 | concat + channel_mix |
| SACE 三源建模 | 隐式 (统一处理) | 显式 3 头并行 |
| DWT-LFF HF | 0.5 共享 | 全高频 (无共享) |
| RWKV 稳定化 | 无 | post_norm + clamp(-8,8)+clamp(-5,5) |
| MRPN | 窗口相关 + 门控 | σ_t 输入 (实验中) |
| 损失权重 | 0.8p/0.3c/0.5s | 0.04p/1.0c/0.2s |
