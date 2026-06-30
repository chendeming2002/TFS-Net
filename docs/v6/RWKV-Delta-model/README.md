# RWKV-Delta 代码快照

> Delta 是 TFS-Net v6 架构的第四个变体，对 SACE 进行全面重构：
> 1. **SpatialWKV2D**：四方向空间扫描 (水平/垂直/主对/副对)，替代帧间 RWKV
> 2. **MVC-Shift**：多尺度空洞 Depthwise Conv 替代 Q-Shift
> 3. **C_omega_list**：显式时序对应矩阵 (cosine similarity)，注入 NDPN/MRPN
> 4. **F_t_aligned**：时序对齐锚，NDPN/MRPN 的统一参考
> 5. **A_illu**：s_illum 经 IFPN s_illum_proj → 传入 IGRF
> 6. **CrossFusionGate**：deploy 模式支持重参数化

## 整体架构概览

```
输入 → [Encoder] → ┬─ [TFSI] → s_illum, s_noise
                    └─ [SACE] → sace_out, C_omega, F_t_aligned
                         └→ [IFPN/NDPN/MRPN] (三源并行)
                            → [CrossFusionGate] (交叉门控)
                            → [IGRF] (修正去噪) → 输出
```

> 详见: [`v6-delta-diagrams.md`](v6-delta-diagrams.md)

## 核心模块

| 组件 | 说明 | Delta 创新 |
|------|------|-----------|
| `PureRWKVSACE` | SACE 空间扫描+时序对应 | MVC-Shift + SpatialWKV2D + C_omega + F_t_aligned |
| `MVCShift` | Token Shift | 3分支空洞DWConv(d=1,2,3) |
| `SpatialWKV2D` | 4方向空间扫描 | 水平/垂直/主对角/副对角 Bi-WKV |
| `TemporalCorrespondence` | 时序对应矩阵 | cosine similarity → C_omega_list |
| `TemporalAggregation` | 时序对齐聚合 | C_omega-warp + frame_gate → F_t_aligned |
| `CrossFusionGate` | NDPN↔MRPN 交叉门控 | deploy 模式 (重参数化) |
| `IFPN` | 光照估计 | s_illum_proj → A_illu 输出 |
| `NDPN` | 去噪 | C_omega conf_map 调制 |
| `MRPN` | 运动补偿 | C_omega motion_mag 调制 |

## 数据流

```
TFSI → s_illum → IFPN s_illum_proj → A_illu → IGRF Stage3
TFSI → s_noise → NDPN noise_proj (条件) + IGRF Stage1 (加法)
SACE → C_omega_list → NDPN conf_map + MRPN motion_mag
SACE → F_t_aligned → NDPN/MRPN 对齐参考
CrossFusionGate → f_noise↔f_motion 交叉调制 → IGRF
```
