# RWKV-Delta 代码快照 — Flight10 Mark4

> Flight10m4 在 m3 temporal-base NDPN 基础上新增 1×1 bypass + l2 粗尺度:
> 1. **NDPN 1×1 bypass**: 3×3 spatial + 1×1 pointwise 并行, 细纹无色散直通
> 2. **NDPN l2 coarse**: l2_lat(H/2)→coarse_proj 全局噪声引导
> 3. **MCPN 1×1 bypass**: motion_refine 拆 spatioal + pointwise
> 4. **保留**: m3 detail_map门控 + temporal base, m2 sigmoid gain + HaarDWT锚点 + 15项损失
> 5. **训练** batch=4, accum=4, epochs=80

## 整体架构

```
Enc[l1/l2/l3] → DPE(l3)→s_illum/s_noise
              → TCA(l2)+HaarDWT→WKV→C_omega
              → NDPN(l1+l2): F_denoised + (3×3+1×1)correction + l2_coarse
              → MCPN(l1): window_corr + (3×3+1×1)motion_refine
              → CXG → SGRF → res_t
```

## m4 新增 (vs m3)

| # | 模块 | 新增 | 文件 |
|:--:|------|------|------|
| 1 | NDPN | corr_pointwise (1×1→GELU→1×1) parallel bypass | `ndpn.py` |
| 2 | NDPN | coarse_proj: l2_lat(H/2) → interpolate → +correction×γ×0.5 | `ndpn.py` |
| 3 | MCPN | motion_refine split: spatial(3×3) + pointwise(1×1) | `mrpn.py` |

## 已删除组件

| 组件 | 原因 |
|------|------|
| WFR (SWD) | Flight9 取消 |
| AmpEnhance | 从未启用 |
| L_wfr_reg / L_gamma_reg / L_dpe_prior | 死代码 |
| L_gain_range / L_align_warp | 冲突/无效梯度 |
| NDPN noise_extract + denoise_strength | m3 temporal-base 替换 |
| perceptual_decoupling | SSIM domain mismatch |
