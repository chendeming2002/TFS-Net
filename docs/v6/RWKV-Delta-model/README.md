# RWKV-Delta 代码快照 — Flight10 Mark1

> Flight10m1 在 F10 基础上精简收敛：回退 gain + 感知解耦失败设计，新增 TCA HaarDWT 锚点。
> 1. **gain 回退**: softplus×scale → sigmoid[0.5,2.0] (修复坍缩)
> 2. **感知解耦关闭**: SSIM/VGG → res_t (避免暗态亮度域偏移)
> 3. **删除**: L_gain_range + L_align_warp (冲突梯度 + 无效正则)
> 4. **TCA HaarDWT 锚点**: LL(IN去光照) + HF(DWConv边缘) → +15K
> 5. **保留**: softplus DPE, L_spatial单边, Stage B delta_scale, S1/S2下界, L_brightness_preserve
> 6. **训练** batch=4, accum=4, epochs=80

## 整体架构

```
输入 → Encoder[l1/l2/l3] → ┬ DPE(l3) → s_illum(softplus) + s_noise(sigmoid)
                           ├ TCA(l2) → HaarDWT[LL→IN→anchor, HF→DWConv→edge]
                           │            → fuse → WKV → C_omega → F_t_aligned
                           └ ISPN(l1) → TCC + gain[0.5,2.0] (sigmoid)
                              └→ [NDPN/MCPN] → [CXG] → [SGRF] → res_t
```

## F10m1 关键变更

| # | 模块 | 变更 | 文件 |
|:--:|------|------|------|
| 1 | ISPN | gain: sigmoid[0.5,2.0] (回退F9) | `ispn_v2.py` |
| 2 | Loss | perceptual_decoupling→False, 删L_gain_range+L_align_warp | `losses.py` |
| 3 | TCA | +HaarDWT: LL(IN→1×1 anchor) + HF(DWConv groups=4→edge) | `pure_rwkv_sace.py` |
| 4 | Train | 移除max_gain调度, perceptual_decoupling关闭 | `train.py` |
