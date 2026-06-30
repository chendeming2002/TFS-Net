# RWKV-Delta 代码快照

> Delta 是 TFS-Net v6 架构的第四个变体，对 SACE 进行全面重构：
> 1. **SpatialWKV2D**：四方向空间扫描 (水平/垂直/主对/副对)，替代帧间 RWKV
> 2. **MVC-Shift**：多尺度空洞 Depthwise Conv 替代 Q-Shift
> 3. **C_omega_list**：显式时序对应矩阵 (cosine similarity)，注入 NDPN/MRPN
> 4. **F_t_aligned**：时序对齐锚，NDPN/MRPN 的统一参考
> 5. **A_illu**：s_illum 经 IFPN s_illum_proj → 传入 IGRF
> 6. **CrossFusionGate**：deploy 模式支持重参数化

## 文件结构

```
docs/v6/RWKV-Delta-model/
├── configs/
│   ├── v6_charlie.yaml        # 训练超参
│   └── loss_weights.yaml      # 损失权重
├── datasets/
│   ├── __init__.py
│   ├── sdsd_dataset.py        # SDSD 数据集加载
│   └── transforms.py          # 视频增强
├── losses/
│   ├── __init__.py
│   └── losses.py              # Charbonnier + Perceptual + SSIM
├── models/
│   ├── __init__.py            # 导出 TFSNet
│   ├── blocks.py              # ConvBlock, ResBlock, NAFBlock, 窗口函数 (175行)
│   ├── dwt_lff.py             # SpatialDWTLFFAdapter + HaarDWT2D (84行)
│   └── tfs_net.py             # TFSNet 完整模型 (1919行, 合并23个类)
├── train.py                   # 训练+推理 (259行)
├── README.md                  # 本文件
└── v6-delta-diagrams.md       # 架构图 (Mermaid)
```

## 整体架构概览

```
输入 → [Encoder] → ┬─ [TFSI] → s_illum, s_noise
                    └─ [SACE] → sace_out, C_omega, F_t_aligned
                         └→ [IFPN/NDPN/MRPN] (三源并行)
                            → [CrossFusionGate] (交叉门控)
                            → [IGRF] (修正去噪) → 输出
```

> 详见: [`v6-delta-diagrams.md`](v6-delta-diagrams.md)

## 核心模块 (models/tfs_net.py, 1864行, 23个类)

| 类 | 所属模块 | Delta 创新 |
|------|------|-----------|
| `PyramidEncoder` | Encoder | 3级金字塔编码 |
| `TFSI` | 时频诊断 | temporal_fuse 多帧 FrequencyBranch |
| `MVCShift` | SACE | 3分支空洞DWConv 替代 Q-Shift |
| `SpatialWKV2D` | SACE | 4方向空间 Bi-WKV 扫描 |
| `TemporalCorrespondence` | SACE | cosine similarity → C_omega_list |
| `TemporalAggregation` | SACE | C_omega-warp + frame_gate → F_t_aligned |
| `PureRWKVSACE` | SACE | 空间扫描+时序对应, 移除DWT-LFF |
| `IFPN` | 光照 | s_illum_proj → A_illu 输出 |
| `NDPN` | 去噪 | conf_proj + noise_extract + denoise_strength + s_noise |
| `MRPN` | 运动 | motion_estimator + sigma_proj + comp_gate + motion_refine + γ |
| `CrossFusionGate` | 交叉 | deploy 重参数化 |
| `IGRF` | 合成 | A_illu 替代 s_illum 直接注入 |

## 数据流

```
TFSI → s_illum → IFPN s_illum_proj → A_illu → IGRF Stage3
TFSI → s_noise → NDPN noise_proj (条件) + IGRF Stage1 (加法)
SACE → C_omega_list → NDPN conf_map + MRPN motion_mag
SACE → F_t_aligned → NDPN/MRPN 对齐参考
CrossFusionGate → f_noise↔f_motion 交叉调制 → IGRF
```

## 已移除 (vs 生产代码)

| 移除组件 | 原因 |
|----------|------|
| DeformableCrossAttention | PureRWKVSACE 替代 |
| VRWKVStyleSpatialMix | SpatialWKV2D 替代 |
| AmpEnhance | 实验性, 未使用 |
| LFFFeatureAdapter | DWT-LFF 替代 |
| DWT-LFF in SACE | Delta 直接使用 Encoder 特征 |
| edge_prompt | Delta 移除边缘门控 |
