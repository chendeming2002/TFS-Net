# RWKV-Charlie (Bravo2) 代码快照

> Charlie 是 TFS-Net v6 架构的第三个变体, 在 Bravo 基础上引入:
> 1. **TFSI 多帧 FrequencyBranch**: `temporal_fuse` 融合相邻帧频域特征
> 2. **多尺度 concat 融合**: PureRWKVSACE 用 `concat + channel_mix` 替代 `/3` 平均
> 3. **σ→MRPN**: sigma_t 作为额外输入 (实验中)
> 4. **Bravo2 P1 (RWKV 稳定化)**: `post_norm` LayerNorm, `spatial_decay.clamp(-8,8)`, `spatial_first.clamp(-5,5)`
> 5. **Bravo2 P2 (DWT-LFF 全高频)**: 取消 0.5 HF 共享, 两分支使用完整 LH/HL/HH
> 6. **Bravo2 P0 (Loss 权重调整)**: `lambda_perc=0.04`, `lambda_pix=1.0`, `lambda_ssim=0.2`

## 整体架构概览

```
输入低光视频 → [Encoder] → [DWT-LFF 三源分离] → ┬─ TFSI (频域/噪声路径)
                                                    ├─ SACE (空域/结构路径, 3头并行)
                                                    └→ Fusion → Cross-RWKV → Decoder → 增强输出
```

> 详见: [`v6-charlie-diagrams.md`](v6-charlie-diagrams.md) — 最简架构图 + 带细节架构图 (Mermaid)

## 文件结构

```
docs/v6/RWKV-Charlie-model/
├── configs/
│   ├── v6_charlie.yaml        # 完整超参配置 (YAML)
│   └── loss_weights.yaml      # 损失权重独立配置
├── datasets/
│   ├── __init__.py            # 导出
│   ├── sdsd_dataset.py        # SDSD 数据集加载器
│   └── transforms.py          # 视频增强: Crop, Flip, TTA
├── losses/
│   ├── __init__.py            # 导出
│   └── losses.py              # Charbonnier + Perceptual + SSIM (P0 权重)
├── models/
│   ├── __init__.py            # 导出 TFSNet
│   ├── blocks.py              # ConvBlock, ResBlock, NAFBlock, LayerNorm2d, 窗口函数
│   ├── cross_rwkv.py          # VRWKVStyleSpatialMix (P1: post_norm + clamp)
│   ├── dwt_lff.py             # SpatialDWTLFFAdapter (P2: 全高频, 无 0.5 共享)
│   └── tfs_net.py             # TFSNet 完整模型 (合并所有子模块)
├── train.py                   # 训练+推理脚本 (合并 utils)
└── README.md                  # 本文件
```

## 核心模块

### `models/tfs_net.py` (888 行)

| 组件 | 说明 | Charlie 差异 |
|------|------|------------|
| `TFSNet` | 主模型 | 配置参数 `use_temporal_fusion` |
| `PureRWKVSACE` | SACE 多尺度融合 | concat + channel_mix |
| `FrequencyBranch` | 频域增强分支 | `temporal_fuse` Conv3d |
| `TFBI` | TFS-Block 时频交互 | 原生 |
| `MRPN` | 运动细化金字塔 | sigma_t 输入 |
| `PCDwAlign` | PCD 对齐 | 原生 |
| `CrossScaleAttention` | 跨尺度注意力 | 原生 |
| `SpatialFreqInteraction` | SFI 空域频域交互 | 原生 |

## 与 Bravo 的主要差异

| 特性 | Bravo | Charlie |
|------|-------|---------|
| TFSI FrequencyBranch | 单帧处理 | 多帧 temporal_fuse |
| SACE 多尺度融合 | `/3` 平均 | concat + channel_mix |
| RWKV 稳定化 | 无 post_norm/clamp | post_norm + decay/first clamp |
| DWT-LFF 高频 | 0.5 共享 HF | 全高频 (无共享) |
| MRPN sigma | 未使用 | sigma_t 输入 (实验中) |
| lambda_perc | 0.8 | 0.04 |
| lambda_pix | 0.3 | 1.0 |
| lambda_ssim | 0.5 | 0.2 |
