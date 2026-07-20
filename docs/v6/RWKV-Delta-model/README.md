# RWKV-Delta 代码快照 — Flight9

> Flight9 是 TFS-Net v6 的最新架构，对 DPE/Encoder-TCA 路由/WFR 进行全面重构：
> 1. **取消 WFR**: Encoder l1/l2/l3 三尺度直连各模块, 天然频率分离
> 2. **DPE softplus+soft_clamp**: sigmoid→softplus 根除饱和, 单尺度 H/4
> 3. **L_illum_spatial + L_illum_tv**: 反零方差 + 边缘感知 TV 双重防饱和
> 4. **TCA H/2**: 128×128 WKV, l2_lat 直连, 无 internal FPN
> 5. **Gamma clamp**: NDPN gamma max=0.03 防暴走
> 6. **训练** batch=4, accum=4 (eff=16), epochs=80

## 整体架构概览

```
输入 → Encoder[l1/l2/l3] → ┬ DPE(l3) → s_illum(softplus) + s_noise(sigmoid)
                           ├ TCA(l2) → tca_out + C_omega + F_t_aligned
                           └ ISPN(l1) → TCC + gain
                              └→ [NDPN/MCPN] (三源并行)
                                 → [CXG] → [SGRF] → 输出
```

## 文件结构

```
docs/v6/RWKV-Delta-model/
├── configs/
│   └── delta_flight9.yaml     # 当前训练配置 (batch=4, accum=4, epochs=80)
├── models/
│   ├── tfs_net.py             # TFSNet + CXG + LRU frame_cache (Flight9, 无WFR)
│   ├── tfsi_v2.py             # DPE (softplus+soft_clamp, 单尺度 H/4)
│   ├── pure_rwkv_sace.py      # TCA (H/2 WKV, l2_lat 直连)
│   ├── ndpn.py                # NDPN (gamma clamp ≤0.03, zero-init sigmoid)
│   ├── mrpn.py                # MCPN (zero-init sigmoid)
│   ├── igrf.py                # SGRF (Stage A/B + auxiliary losses)
│   └── modules/               # boxes, encoder
├── losses/
│   └── losses.py              # TFSNetLoss (+L_illum_spatial+L_illum_tv)
├── train.py                   # phase schedule for 80 epochs
└── v6-delta-diagrams.md       # 架构图 (Flight9 updated)
```

## Flight9 核心模块

| 类 | 模块 | Flight9 创新 |
|------|------|-----------|
| `PyramidEncoder` | Encoder | l1/l2/l3 多尺度直连 (无 FPN 融合, 无 WFR 中介) |
| `DPE` | 退化估计 | softplus+soft_clamp IllumHead, 单尺度 H/4, L_spatial+L_tv |
| `IllumHead` | DPE | Conv1x1→softplus+base(0.3)→soft_clamp(max=3.0) |
| `TCA` | 时序对齐 | H/2 WKV, l2_lat 直连, 无 internal FPN |
| `NDPN` | 去噪 | gamma property clamp ≤0.03, zero-init conf_proj+denoise_strength |
| `MCPN` | 运动 | zero-init motion_estimator+comp_gate |

## 数据流

```
Encoder → l1, l2, l3 (多尺度)
  ├─ DPE(l3) + gray/lum → s_illum(softplus) → ISPN
  │                     → s_noise(sigmoid) → NDPN
  └─ TCA(l2) → tca_out(H/2) ↑upsample → ISPN/NDPN/MCPN
              → C_omega → NDPN conf_map + MCPN motion_mag
              → F_t_aligned → NDPN/MCPN 对齐参考
CXG → f_noise↔f_motion → SGRF
```

## 已取消 / 变更 (vs Flight8)

| 组件 | 原因 |
|------|------|
| WFR (SWD) | 多尺度 Encoder 天然实现频域分离, 取消 -25K 参数 |
| feat_tfde | DPE 重新设计为 l3 直连, 不再需要 WFR 输出 |
| TCA internal FPN | 回退, TCA 直接从 l2_lat 输入 H/2 |
| TCA wfr_lambda | 取消 WFR 残差注入参数 |
| 3-stage DPE | 级联放大饱和, 单尺度 H/4 足够 (光照是低频) |
| sigmoid s_illum head | 替换为 softplus+soft_clamp |
| AmpEnhance | 从未启用 |
