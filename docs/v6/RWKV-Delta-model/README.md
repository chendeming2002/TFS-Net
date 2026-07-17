# RWKV-Delta 代码快照 — Flight8

> Flight8 是 TFS-Net v6 的最新一代架构，对编码器和模块路由进行全面重构：
> 1. **Multi-Scale Encoder**: 输出 l1/l2/l3 三尺度特征，取消 FPN 融合
> 2. **DPE 3-Stage Scan**: 粗→细渐进扫描 (l3@H/4→l2@H/2→l1@H) + gray/lum 物理先验
> 3. **TCA Internal FPN**: 自底向上聚合三尺度 → 全分辨率 (256×256) WKV 扫描
> 4. **LRU Frame Cache**: 推理滑动窗口复用 80% 编码特征，64帧上限
> 5. **Sigmoid-Terminal Zero-Init**: NDPN/MCPN 4层 sigmoid 末端全部零初始化
> 6. **训练** batch=2, accum=8 (eff=16), epochs=100, 参数量 1.85M

## 文件结构

```
docs/v6/RWKV-Delta-model/
├── configs/
│   ├── delta_flight8.yaml     # 当前训练配置 (batch=2, accum=8, epochs=100)
│   ├── delta_flight3.yaml     # Flight3 旧配置
│   └── loss_weights.yaml      # 损失权重
├── datasets/
│   ├── __init__.py
│   ├── sdsd_dataset.py        # SDSD 数据集加载
│   └── transforms.py          # 视频增强
├── losses/
│   ├── __init__.py
│   └── losses.py              # TFSNetLoss (Kendall UW + phase schedule)
├── models/
│   ├── __init__.py            # 导出 TFSNet
│   ├── tfs_net.py             # TFSNet + CXG + LRU frame_cache (348行, Flight8)
│   ├── swd.py                 # WFR (HaarDWT子带分流, Flight3取消噪声门控)
│   ├── pure_rwkv_sace.py      # TCA (Internal FPN + full-res WKV + C_omega)
│   ├── ndpn.py                # NDPN (C_omega置信度引导 + zero-init sigmoid)
│   ├── mrpn.py                # MCPN (motion_estimator + zero-init sigmoid)
│   ├── igrf.py                # SGRF (Stage A/B + auxiliary losses)
│   └── modules/
│       ├── blocks.py          # ConvBlock, ResBlock, NAFBlock, LayerNorm
│       └── encoder.py         # PyramidEncoder (3-stage + lateral convs)
├── utils/
│   ├── inference.py           # tiled_forward (with cache support)
│   └── ...
├── train.py                   # 训练脚本 (phase schedule for 100 epochs)
├── README.md                  # 本文件
└── v6-delta-diagrams.md       # 架构图 (Mermaid, Flight8 updated)
```

## 整体架构概览

```
输入 → [Encoder(l1/l2/l3)] → ┬─ [WFR(l1)] → feat_tca (H/2) → TCA 残差
                              ├─ [DPE(l1/l2/l3)] → s_illum, s_noise
                              └─ [TCA(l1/l2/l3 + WFR)] → tca_out, C_omega, F_t_aligned
                                   └→ [ISPN/NDPN/MCPN] (三源并行)
                                      → [CXG] (交叉门控)
                                      → [SGRF] (阶段式修复) → 输出
```

### Flight8 核心模块

| 类 | 模块 | Flight8 创新 |
|------|------|-----------|
| `PyramidEncoder` | Encoder | 3级金字塔 + lateral convs → l1/l2/l3 三尺度 |
| `WFR` | 小波分流 | HaarDWT LL/HF 子带级分流 (Flight3取消噪声门控) |
| `DPE` | 退化估计 | 3-stage coarse-to-fine scan + gray/lum priors + cls_token |
| `TCA` | 时序对齐 | Internal FPN(l3→l2→l1) → full-res 256×256 WKV + WFR残差 |
| `MVCShift` | TCA | 3分支空洞DWConv (RSRWKV) |
| `SpatialWKV2D` | TCA | 4方向Bi-WKV 空间扫描 |
| `TemporalCorrespondence` | TCA | cosine similarity → C_omega_list @ H/2 |
| `TemporalAggregation` | TCA | C_omega-warp + frame_gate → F_t_aligned |
| `ISPN` | 光照 | TCC曲线 6iter 4×↓ + pixel-wise gain [0.5,2.0] |
| `NDPN` | 去噪 | conf_proj + noise_extract + denoise_strength (zero-init sigmoid) |
| `MCPN` | 运动 | motion_estimator + comp_gate (zero-init sigmoid) + window_corr |
| `CXG` | 交叉 | cross-excitation gate, deploy重参数化 |
| `SGRF` | 合成 | Stage A/B + sg[img_lit] + residual_head |

## 数据流

```
Encoder → l1, l2, l3 (多尺度 lateral)
  ├─ WFR(l1) → feat_tca (H/2) → TCA 残差路径
  ├─ DPE(l1,l2,l3) + gray/lum → s_illum → ISPN
  │                           → s_noise → NDPN
  └─ TCA(l1,l2,l3 + WFR residual) → tca_out → ISPN
                                    → C_omega → NDPN conf_map + MCPN motion_mag
                                    → F_t_aligned → NDPN/MCPN 对齐参考
CXG → f_noise↔f_motion 交叉调制 → SGRF
```

## 已移除 / 变更 (vs 生产代码)

| 组件 | 原因 |
|------|------|
| FPN fusion in Encoder | Flight8: 取消融合, 多尺度 lateral 直接路由各模块 |
| feat_tfde from WFR | Flight8: DPE 改由 encoder l1/l2/l3 输入 → WFR 仅输出 feat_tca |
| H/2 WKV in TCA | Flight8: 升级为 Internal FPN → full-res 256×256 WKV |
| DeformableCrossAttention | PureRWKVSACE 替代 |
| AmpEnhance | 实验性, 未启用 |
| LFFFeatureAdapter | DWT-LFF / WFR 替代 |
| edge_prompt | Delta 移除边缘门控 |
