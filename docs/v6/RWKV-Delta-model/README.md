# RWKV-Delta 代码快照 — Flight10 Mark5

> Flight10m5 基于 m4 设计审查逐项修复:
> 1. **detail_residual highway**: f_enc - F_denoised → 1×1 gate → preserved (真信息高速路)
> 2. **coarse F_denoised prior**: F_denoised avgpool 替代 l2_lat, 加 detail_map 门控
> 3. **detail_map: F_denoised grad**: 时序滤波后无噪声, 替代 image_center
> 4. **gamma→0.1 + refine skip**: 释放校正路径 + 保底 identity
> 5. **MCPN 回退单 3×3**: 运动是空间位移, 1×1 无意义
> 6. **训练** batch=2, accum=8 (eff=16), epochs=80

## 整体架构

```
Enc[l1/l2/l3] → DPE(l3)→s_illum/s_noise
              → TCA(l2)+HaarDWT→WKV→C_omega
              → NDPN(l1): F_denoised + detail_residual(1×1 gate) + corr_spatial(3×3)
                  + coarse(F_denoised pool) × γ×0.5×(1-d) → f_noise
              → MCPN(l1): window_corr + motion_refine(3×3) → f_motion
              → CXG → SGRF → res_t
```

## m5 修复项

| 级别 | 变更 | 目的 |
|:--:|------|------|
| P0 | NDPN: corr_pointwise→detail_residual gate | 真 highway, 非冗余矫正 |
| P0 | MCPN: 移除 1×1 motion_refine | 运动非 per-pixel |
| P1 | coarse: l2_lat→F_denoised avgpool | 干净时序先验 |
| P1 | detail_map: image→F_denoised gradient | 无噪声纹理检测 |
| P2 | gamma: 0.03→0.1 | 释放三路校正 |
| P2 | refine: +skip connection | 保底 identity |
