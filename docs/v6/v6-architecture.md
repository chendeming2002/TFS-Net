# TFS-Net v6 Delta Mark1 模型架构设计文档

> 日期：2026-07-01 (更新: Mark1 全量实施)
> 版本：v6 Delta Mark1
> 训练配置：`configs/v6_bravo.yaml`，batch=1, lr=8e-4, epochs=50, warmup=5
> 参数量：1.688M

---

## 1. 概述

v6 Delta Mark1 对 SWD 进行子带级分流重构，解决 s_illum/s_noise norm 爆炸导致的训练崩溃。

| 模块 | 缩写 | 改动 | 文献 |
|------|------|------|------|
| **SWD** | Spatial Wavelet Diverter | 子带级 LL/HF 分流 (替代 DWT-LFF inverse DWT) | — |
| **TFDE** | Temporal-Frequency Degradation Estimator | 时频退化估计 (原 TFSI) | — |
| **TCA** | Temporal Correspondence & Alignment | MVC-Shift + 4方向 WKV 空间扫描 (原 SACE) | RSRWKV (TCSVT 2025) |
| **ISPN** | Illumination-Source Processing Network | 光照处理 + A_illu 生成 (原 IFPN) | DiTVR (CVPR 2026) |
| **NDPN** | Noise Degradation Processing Network | C_omega 置信度引导去噪 (不变) | Revisiting Temporal Alignment (CVPR 2022) |
| **MCPN** | Motion Compensation Processing Network | C_omega 运动强度补偿 (原 MRPN) | JFFRA (2025) |
| **CXG** | Cross-eXcitation Gate | deploy 重参数化 (原 CrossFusionGate) | DRNet (CVPR 2026) |
| **SGRF** | Stage-wise Guided Restoration & Fusion | 阶段式去噪→去模糊→提亮 (原 IGRF) | — |

### 命名变更总表

| 原缩写 | 新缩写 | 中文全名 |
|--------|--------|----------|
| DWT-LFF | **SWD** | 空域小波分流器 |
| TFSI | **TFDE** | 时频退化估计器 |
| SACE | **TCA** | 时序对应对齐 |
| IFPN | **ISPN** | 光照源处理网络 |
| NDPN | **NDPN** | 噪声退化处理网络 (不变) |
| MRPN | **MCPN** | 运动补偿处理网络 |
| IGRF | **SGRF** | 阶段式引导修复融合 |
| CrossFusionGate | **CXG** | 交叉激励门 |

### Mark1 核心诊断

| 指标 | 旧 Delta | Mark1 预期 |
|------|---------|-----------|
| ep5 loss | 0.226→0.436 (反弹) | 单调下降 |
| s_illum norm | 63~113 ❌ | ≈1.0 ✅ |
| 退 化分离 | 失效 | 有效 |

### 完整数据流

```
输入 x: (B, T=5, 3, H, W)
  │
  ├─→ Encoder → F_stack (B, T, 64, H, W)
  │
  ├─→ SWD (逐帧) → feat_tfde (B,T,C,H/2), feat_tca (B,T,C,H/2)
  │     ├─ [HaarDWT] → LL, LH, HL, HH
  │     ├─ [LL分流] alpha_net(LL) → α·LL→TFDE, (1-α)·LL+IN→TCA
  │     ├─ [HF分流] noise_gate×HF→TFDE, struct_gate×HF+LN→TCA
  │     └─ [proj+LN] Conv(4C→C)+LayerNorm → feat_tfde/feat_tca
  │
  ├─→ TFDE(feat_tfde) → s_illum, s_noise (upsample→H×W)
  │     ├─ s_illum → ISPN (唯一路径)
  │     └─ s_noise → NDPN (noise_proj 条件注入)
  │
  ├─→ TCA(feat_tca) → tca_out (B,T,C,H), C_omega_list, F_t_aligned, mu, sigma
  │     ├─ [MVC-Shift] 3 dilated DWConv (d=1,2,3)
  │     ├─ [SpatialWKV2D] 4方向 Bi-WKV 空间扫描
  │     ├─ [Channel Mix] → upsample → tca_out
  │     ├─ [TemporalCorrespondence] Q-K cosine → C_omega_list
  │     └─ [TemporalAggregation] C_omega warp + gate → F_t_aligned
  │
  ├─→ F_aligned_list = [tca_out[:, t] for t] → ISPN/NDPN/MCPN
  │
  ├─→ ISPN(aligned_feats, s_illum, F_t_aligned) → lit_up_map_raw, f_illum_feat, A_illu
  ├─→ NDPN(aligned_feats, s_noise, mu, sigma, C_omega, F_t_aligned) → f_noise_out
  ├─→ MCPN(aligned_feats, sigma, C_omega, F_t_aligned) → f_motion_out
  │
  ├─→ CXG(f_noise_out, f_motion_out) → f_noise_gated, f_motion_gated
  │
  └─→ SGRF(f_illum_feat, f_noise_gated, f_motion_gated, lit_up_map, img_center, A_illu)
        ├─ Stage1: img_s1 = denoise(f_noise_gated, img_center)  ★ s_noise 不在 SGRF
        ├─ Stage2: img_s2 = motion(f_motion_gated, img_s1)
        └─ Stage3: res_t = clamp(img_s2 × lit_up_map × (1+A_illu))
```

---

## 2. 核心模块详解

### 2.1 SWD — 空域小波分流器

**文件**: `models/modules/swd.py`

子带级分流（不做 inverse DWT），显式分离"光照+噪声"和"光照无关结构"。

```
Encoder feat → [HaarDWT] → LL, LH, HL, HH
  ├─ alpha_net(LL) → α ∈ (0,1)
  ├─ LL_tfde = α × LL
  ├─ LL_tca = IN((1-α) × LL)
  ├─ HF_energy = (LH²+HL²+HH²).mean(C)
  ├─ noise_gate(HF_energy) → n_gate ∈ (0,1)
  ├─ HF_tfde = n_gate × HF_cat
  ├─ HF_tca = LN((1-n_gate) × HF_cat)
  └─ proj(4C→C)+LN → feat_tfde, feat_tca (each B,T,C,H/2,W/2)
```

### 2.2 TCA — 时序对应对齐 (Bi-WKV)

**文件**: `models/modules/pure_rwkv_sace.py`

**空间扫描 Bi-WKV**:
```
k.clamp(-8,8), v.clamp(-8,8)
w = -F.softplus(spatial_decay)  → ew < 1 恒衰减
chunk-wise cumsum (CHUNK=256)  → 防长序列数值溢出
双向 (fwd+bwd)/2
```

**四方向扫描** (每方向独立 BiWKV + per-head LN):
```
水平(H×W→L), 垂直(W×H→L), 主对角线, 副对角线
Concat 4 heads → σ(R)⊙wkv → proj_out → post_norm
```

**输入**: feat_tca (B,T,C,H/2,W/2) — Mark1 无需内部降采样

### 2.3 数值稳定性

| 组件 | 措施 |
|------|------|
| BiWKV | `k.clamp(-8,8)`, `v.clamp(-8,8)` |
| BiWKV | `w = -F.softplus(spatial_decay)` → ew < 1 数学保证 |
| BiWKV | chunk-wise cumsum (CHUNK=256) |
| SpatialWKV2D | `pre_norm` LayerNorm 在 R/K/V 投影前 |
| SpatialWKV2D | R/K/V RWKV-7 风格小初始化 `±0.05~0.5/√C` |
| SWD | `proj_tfde/proj_tca` 后 LayerNorm → norm≈1 |
| Tau | `F.softplus(tau_raw) + 0.05` → 下界 0.05 |

---

## 3. 参数分布

| 模块 | 参数量 |
|------|--------|
| Encoder | ~320K |
| SWD | ~25K |
| TFDE | ~120K |
| TCA (MVCShift + WKV + Corr + Agg) | ~300K |
| ISPN (含 s_illum_proj + illu_conv + illu_anchor) | ~160K |
| NDPN (含 conf_proj + noise_extract + denoise_strength) | ~85K |
| MCPN (含 motion_estimator + comp_gate + motion_refine) | ~80K |
| CXG | ~8K |
| SGRF | ~120K |
| 其他 | ~70K |
| **总计** | **~1.69M** |

---

## 4. 版本演进

| 版本 | 关键改动 | 参数量 | 状态 |
|------|---------|--------|------|
| v5.9.2 | s_illum 复生 + IFPN 监督 | 1.14M | 20.39 PSNR |
| v6.5 | PureRWKV 移除 DAT | 1.17M | 20.36 |
| v6 Bravo | 损失重调 + DWT-LFF分裂 + V raw | 1.20M | 20.05 |
| v6 Charlie | 多帧 F.B. + concat fusion | 1.25M | — |
| v6 Charlie3 | s_illum→IFPN单路径 + CXG + LearnableScaleFusion | 1.32M | ep29 loss=0.176 |
| v6 Delta | 空间扫描2D-WKV + C_omega + F_t_aligned | 1.64M | 训练崩溃 (loss反弹) |
| **v6 Delta Mark1** | **SWD子带分流 + 命名统一 + 数值稳定** | **1.69M** | 训练中 |

---

## 5. 关键文件

| 文件 | 模块 |
|------|------|
| `models/modules/swd.py` | SWD (SpatialWaveletDiverter, HaarDWT2D) |
| `models/modules/pure_rwkv_sace.py` | TCA, BiWKV, SpatialWKV2D, MVCShift, TemporalCorrespondence, TemporalAggregation |
| `models/modules/tfsi.py` | TFDE (Temporal-Frequency Degradation Estimator) |
| `models/modules/ifpn.py` | ISPN (Illumination-Source Processing Network) |
| `models/modules/ndpn.py` | NDPN (Noise Degradation Processing Network) |
| `models/modules/mrpn.py` | MCPN (Motion Compensation Processing Network) |
| `models/modules/igrf.py` | SGRF (Stage-wise Guided Restoration & Fusion) |
| `models/tfs_net.py` | CXG, TFSNet (主入口, 数据流编排) |
| `configs/v6_bravo.yaml` | 训练配置 |

---

## 6. 损失函数设计

**文件**: `losses/losses.py` — `TFSNetLoss`

### 6.1 总损失公式

```
L_total = λ_pix · L_pix
        + λ_freq · L_freq
        + λ_ssim · L_ssim
        + λ_perc · L_perc
        + λ_illum · L_illum_smooth
        + λ_illum_sup · L_illum_sup
        + λ_noise_sup · L_noise_sup
        + λ_inter · L_inter
        + λ_ifpn_sup · L_ifpn_sup
```

### 6.2 各损失项

| 项 | 公式 | 语义 | 权重 | 监督对象 |
|----|------|------|------|---------|
| **L_pix** | `Charbonnier(res_t, GT)` | 像素级保真（smooth L1） | **1.0** | 最终输出 |
| **L_freq** | `L1(\|FFT(res)\|, \|FFT(GT)\|)` | 频域保真（纹 理/边缘） | 0.05 | 最终输出 |
| **L_ssim** | `1 - SSIM(res_t, GT)` | 结构相似度 | 0.2 | 最终输出 |
| **L_perc** | `L1(VGG_relu3_3(res), VGG_relu3_3(GT))` | 感知一致性（VGG 浅层特征） | 0.04 | 最终输出 |
| **L_illum_smooth** | `TV(s_illum) × edge_weight` | 光照图空间平滑度（边缘感知） | 0.001 | s_illum |
| **L_illum_sup** | `L1(s_illum, GT_illum)` where `GT_illum = clamp(1-L_t/L_ref, 0, 1)` | s_illum 显式监督（目标：暗区高、亮区低） | 0.02 | s_illum |
| **L_noise_sup** | `L1(s_noise, GT_noise)` where `GT_noise = clamp(1-SNR/5, 0, 1)` | s_noise 显式监督（目标：低SNR区高） | **0.0** (关闭) | s_noise |
| **L_inter** | `Charbonnier(clamp(img_s2×lit_up_map,0,1), GT)` | 中间乘法路径监督 | 0.2 | img_s2 × lit_up_map |
| **L_ifpn_sup** | Perceptual(ifpn_side, F↓(GT)) | ISPN 中间输出感知监督（DarkIR 风格） | 0.1 | ifpn_side |

### 6.3 设计要点

- **Charbonnier 主导**：λ_pix=1.0 为最大权重，消融验证 Charbonnier 比 L1/L2 更稳定（smooth gradient near 0）
- **VGG 极低权重**：λ_perc=0.04 避免早期训练的高感知损失主导梯度（旧版 0.8 时 ep10 崩溃 -8.5dB）
- **SSIM 辅助**：λ_ssim=0.2 提供结构先验，补充像素损失的纹理盲区
- **n_sup=0.0**：s_noise 通过 `noise_proj` 端到端注入 NDPN 学习，不做显式 SNR 目标监督（避免与 phase_conf 调制冲突）
- **ifpn_sup**：ISPN 的 `side_head` 将光照特征投影为低分辨率图像，用感知损失监督，强制其学习有意义的光照表示

---

## 7. 训练策略

### 7.1 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | **1** | 24GB VRAM 极限（TCA 空间扫描 + C_omega 内存需求大） |
| epochs | 50 | — |
| lr | 8e-4 | Adam 初始学习率 |
| warmup_epochs | 5 | 线性 warmup (0.01→1.0×lr) |
| weight_decay | 1e-4 | L2 正则 |
| grad_clip | 0.5 | 梯度裁剪阈值 |

### 7.2 优化器

Adam (β₁=0.9, β₂=0.999, ε=1e-8)，所有参数统一学习率。

### 7.3 学习率调度

```
Epoch 0-4:   lr = (0.01 + 0.99 × t/warmup_steps) × base_lr   (线性 warmup)
Epoch 5-24:  lr = base_lr                                       (平稳期)
Epoch 25-49: lr = base_lr × (1 - (t-25)/25)                    (线性 decay → 0)
```

### 7.4 数据与评估

| 项目 | 配置 |
|------|------|
| 训练集 | SDSD indoor (2064 samples), crop=256×256, window=5 |
| 验证集 | SDSD test (1120 samples), 每 5 epoch 评估一次 |
| 评估指标 | PSNR, SSIM, LPIPS |
| 推理 | `infer.py`, tile_size=256, tile_overlap=32, no AMP |

### 7.5 已知稳定性问题与对策

| 问题 | 现象 | 对策 |
|------|------|------|
| **exp(k) 溢出** | NaN loss, non-finite grad | `k.clamp(-8,8)`, `v.clamp(-8,8)`, `pre_norm` before R/K/V proj |
| **ew_pow 下溢** | cumsum 精度丢失 | `w = -F.softplus(spatial_decay)` 强制 w<0, chunk-wise (256 steps) |
| **s_illum norm 爆炸** | IntensityHead 输入 norm=60~113 → loss 反弹 | SWD 子带分流 + proj 后 LayerNorm → norm≈1 |
| **NDPN 减法溢出** | γ 零初始化 + `Sigmoid(strength)` → 安全渐进 |
| **SGRF 梯度消失** | 暗区 `img_s2≈0` → ∂loss/∂lit_up_map≈0 | `img_s2 + 0.01` bias 保证最小梯度 |

### 7.6 训练监控

```bash
# 实时日志
tail -f outputs/sdsd_delta/nohup.out

# 关键指标趋势
grep "Train stats" outputs/sdsd_delta/nohup.out | tail -20

# NaN/异常检测
grep "non-finite\|NaN\|WARNING" outputs/sdsd_delta/nohup.out
```

正常训练中每 50 step 输出的均值应持续下降，无 `non-finite` 警告，pix/ssim/perc 三项逐步收敛。
