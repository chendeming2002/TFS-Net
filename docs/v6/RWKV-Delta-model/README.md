# RWKV-Delta 代码快照 — Flight10

> Flight10 是 TFS-Net v6 的最新架构，聚焦损失函数清理 + Stage B/ISPN 管线修复：
> 1. **损失清理**: 删除 L_wfr_reg/L_gamma_reg/L_dpe_prior 死代码
> 2. **Stage B delta_scale=0.2**: 替代 tanh(β=0), 消除梯度死区
> 3. **ISPN softplus gain**: gain[1,20] via softplus×learnable_scale(2.0)
> 4. **S1/S2亮度约束**: softplus软下界, 防止去噪/去模糊暗化
> 5. **感知解耦启用**: SSIM→img_s1+S2, VGG→res_t, +L_ssim_s2
> 6. **L_spatial单边**: relu(1.0-std), 不推动高方差
> 7. **训练** batch=4, accum=4 (eff=16), epochs=80

## 整体架构

```
输入 → Encoder[l1/l2/l3] → ┬ DPE(l3) → s_illum(softplus) + s_noise(sigmoid)
                           ├ TCA(l2) → tca_out + C_omega + F_t_aligned
                           └ ISPN(l1) → TCC + gain[1,20]
                              └→ [NDPN/MCPN] → [CXG] → [SGRF] → res_t
```

## Flight10 核心变更

| # | 模块 | 变更 | 文件 |
|:--:|------|------|------|
| 1 | SGRF | Stage B tanh(β)→delta_scale=0.2, S1/S2 softplus下界 | `igrf.py` |
| 2 | ISPN | gain: sigmoid[0.5,2.0]→softplus×scale(2)+clamp[1,20] | `ispn_v2.py` |
| 3 | Loss | 删L_wfr_reg/L_gamma_reg/L_dpe_prior, L_spatial单边relu | `losses.py` |
| 4 | Loss | +L_brightness_preserve(0.5)+L_gain_range(0.3)+L_ssim_s2(0.1) | `losses.py` |
| 5 | Loss | perceptual_decoupling=True, L_align_warp→0.005 | `losses.py` |
| 6 | Train | 移除 max_gain 动态调度 | `train.py` |

## 已删除组件

| 组件 | 原因 |
|------|------|
| WFR (SWD) | Flight9取消 |
| AmpEnhance | 从未启用 |
| L_wfr_reg | WFR不存在, 死代码 |
| L_gamma_reg | gn=0.034>>0.005, relu恒零 |
| L_dpe_prior | 与softplus DPE+L_spatial冲突 |
| max_gain scheduling | ISPN改用learnable scale替代 |
