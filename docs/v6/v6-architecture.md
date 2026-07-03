# TFS-Net v6 Delta Mark2 模型架构设计文档

> 日期：2026-07-03 (更新: Mark2 损失调度实施)
> 版本：v6 Delta Mark2
> 训练配置：`configs/v6_bravo.yaml`，batch=1 (grad_accum=3), lr=8e-4, epochs=50, warmup=5
> 参数量：1.688M

---

## 1. 概述

TFS-Net v6 是一个端到端多帧低光视频增强网络。核心思想是**信号分流 → 退化估计 → 并行重建 → 阶段式融合**。

| 模块 | 缩写 | 功能 |
|------|------|------|
| **SWD** | Spatial Wavelet Diverter | Haar DWT 子带级分流 (LL→光照/噪声, HF→结构) |
| **TFDE** | Temporal-Frequency Degradation Estimator | 时频退化估计 → s_illum, s_noise |
| **TCA** | Temporal Correspondence & Alignment | 4方向空间 WKV 扫描 + C_omega 时序矩阵 |
| **ISPN** | Illumination-Source Processing Network | 光照图估计 + A_illu 生成 |
| **NDPN** | Noise Degradation Processing Network | C_omega 置信度引导去噪 |
| **MCPN** | Motion Compensation Processing Network | C_omega 运动强度补偿 |
| **CXG** | Cross-eXcitation Gate | 去噪↔运动 交叉激励门 (deploy 重参数化) |
| **SGRF** | Stage-wise Guided Restoration & Fusion | S1:去噪 → S2:去模糊 → S3:提亮 |

### 命名变更总表

| 原缩写 | 新缩写 | 中文全名 |
|--------|--------|----------|
| DWT-LFF | **SWD** | 空域小波分流器 |
| TFSI | **TFDE** | 时频退化估计器 |
| SACE | **TCA** | 时序对应对齐 |
| IFPN | **ISPN** | 光照源处理网络 |
| MRPN | **MCPN** | 运动补偿处理网络 |
| IGRF | **SGRF** | 阶段式引导修复融合 |
| CrossFusionGate | **CXG** | 交叉激励门 |

### 完整数据流

```
输入 x: (B, T=5, 3, H, W)
  │
  ├─→ Encoder → F_stack (B, T, 64, H, W)
  │
  ├─→ SWD (逐帧 HaarDWT) → feat_tfde (B,T,C,H/2), feat_tca (B,T,C,H/2)
  │
  ├─→ TFDE(feat_tfde) → s_illum, s_noise (↑H×W)
  │     s_illum → ISPN   s_noise → NDPN
  │
  ├─→ TCA(feat_tca) → tca_out (B,T,C,H), C_omega_list, F_t_aligned, mu, sigma
  │
  ├─→ ISPN(aligned, s_illum, F_t_aligned) → lit_up_map_raw, f_illum_feat, A_illu
  ├─→ NDPN(aligned, s_noise, mu, sigma, C_omega, F_t_aligned) → f_noise_out
  ├─→ MCPN(aligned, sigma, C_omega, F_t_aligned) → f_motion_out
  │
  ├─→ CXG(f_noise_out, f_motion_out) → f_noise_gated, f_motion_gated
  │
  └─→ SGRF(f_illum, f_noise_gated, f_motion_gated, lit_up_map, img_center, A_illu)
        S1: img_s1 = denoise(f_noise_gated, img_center)
        S2: img_s2 = motion(f_motion_gated, img_s1)
        S3: res_t = clamp(img_s2 × lit_up_map × (1+A_illu))
```

---

## 2. 理论动机与公式

### 2.1 问题建模

给定多帧低光输入 $\{y_t\}_{t=0}^{T-1}$，目标是恢复干净图像 $\hat{x}$。结构化假设：

$$y_t = x \odot \ell_t + n_t, \quad t=0,\dots,T-1$$

其中 $\ell_t$ 为光照退化因子，$n_t$ 为噪声。将问题分解为三步：

1. **退化分解** (SWD): 分离 $\ell_t$ 和 $n_t$ 的估计信号
2. **时序对应** (TCA): 建立帧间几何映射 $C_\omega$
3. **并行重建** (ISPN/NDPN/MCPN): 独立修复三种退化
4. **阶段融合** (SGRF): 顺序施加去噪→去模糊→提亮

### 2.2 SWD — 空域小波分流

**动机**: 旧 DWT-LFF 通过 IDWT 重建全分辨率，两路特征几乎相同，导致 TFDE 的 IntensityHead 输入 norm 爆炸 (60~113)。

**方法**: 在 Haar DWT 子带级别直接分流，不做 IDWT 重建。

Haar DWT 分解：
$$F \xrightarrow{\text{DWT}} \{LL, LH, HL, HH\}, \quad LL \in \mathbb{R}^{B\times C\times H/2\times W/2}$$

**低频分流** (LL 编码全局光照强度)：
$$\text{LL}_{\text{tfde}} = \alpha(LL) \odot LL, \quad \text{LL}_{\text{tca}} = \text{IN}((1-\alpha(LL)) \odot LL)$$

其中 $\alpha(\cdot) = \sigma(\text{DWConv}_{3\times3} \to \text{GELU} \to \text{Conv}_{1\times1})$。

**高频分流** (LH/HL/HH 编码纹理/噪声)：
$$E_{\text{HF}} = \frac{1}{C}\sum_{c}(LH_c^2+HL_c^2+HH_c^2), \quad g_n = \sigma(\text{Conv}(E_{\text{HF}}))$$

$$\text{HF}_{\text{tfde}} = g_n \odot [LH,HL,HH], \quad \text{HF}_{\text{tca}} = \text{LN}((1-g_n) \odot [LH,HL,HH])$$

**输出投影**：
$$\text{feat}_{\text{tfde}} = \text{LN}(\text{Conv}_{1\times1}([\text{LL}_{\text{tfde}}, \text{HF}_{\text{tfde}}]))$$
$$\text{feat}_{\text{tca}} = \text{LN}(\text{Conv}_{1\times1}([\text{LL}_{\text{tca}}, \text{HF}_{\text{tca}}]))$$

### 2.3 TCA — 时序对应对齐 (RWKV)

**动机**: 扫描轴从时间 (5 token) 改为空间 (H×W token)，利用 RWKV 的长序列建模能力捕捉帧内结构，帧间用显式 correspondence 替代隐式 WKV mix。

**帧内空间扫描 (Bi-WKV)**：
输入 $k_t, v_t \in \mathbb{R}^{C}$，沿空间序列 $L=H\times W$ 做带衰减的递归：

$$S_t = e^{w} \cdot S_{t-1} + e^{k_t} \odot v_t, \quad D_t = e^{w} \cdot D_{t-1} + e^{k_t}$$

$$y_t = \frac{e^{u}\cdot e^{k_t}\odot v_t + S_t}{e^{u}\cdot e^{k_t} + D_t}$$

其中 $w = -\text{softplus}(\theta_w) < 0$ 保证 $e^w < 1$（恒衰减），$u$ 为当前 token 的 bonus。

**四方向扫描** (RSRWKV)。通道分为 4 组 $(C/4)$，每组沿一个方向展开：

| 方向 | 展开方式 |
|------|---------|
| 水平 (→) | $(H\times W) \to L$ 行优先 |
| 垂直 (↓) | $(W\times H) \to L$ 列优先 |
| 主对角线 (↘) | 按 $i+j$ 排序 |
| 副对角线 (↗) | 按 $i-j+W-1$ 排序 |

**TCA 完整计算**：

$$\mathbf{x}_{\text{shifted}} = \text{MVC-Shift}(\mathbf{F}_{\text{tca}}) \quad\text{(3 dilated DWConv, }d=1,2,3\text{)}$$
$$\mathbf{k}, \mathbf{v} = \text{proj}_k(\mathbf{x}), \text{proj}_v(\mathbf{x})$$
$$\text{head}_i = \text{BiWKV}(\text{scan}_i(\mathbf{k}_i), \text{scan}_i(\mathbf{v}_i)), \quad i=0,1,2,3$$
$$\text{WKV} = \sigma(\text{proj}_r(\mathbf{x})) \odot \text{Concat}(\text{head}_0,\dots,\text{head}_3)$$
$$\mathbf{F}_{\text{out}} = \mathbf{F}_{\text{tca}} + \gamma \cdot \text{ChannelMix}(\text{WKV})$$

**时序对应** (C_omega)：
$$\mathbf{q} = \text{proj}_q(\mathbf{F}_{\text{center}}), \quad \mathbf{k}_t = \text{proj}_k(\mathbf{F}_t), \quad t \neq \text{center}$$
$$C_{\omega}^{(t)} = \text{softmax}\left(\frac{\mathbf{q} \cdot \mathbf{k}_t^T}{\tau}\right), \quad \tau = \text{softplus}(\tau_{\text{raw}}) + 0.05$$

**时序聚合** (F_t_aligned)：
$$\mathbf{F}_{\text{aligned}} = \sum_{t \neq c} w_t \cdot \left(C_{\omega}^{(t)} \times \mathbf{F}_t\right), \quad w_t = \text{frame\_gate}([\mathbf{F}_c, \mathbf{F}_{\text{warped}}])$$

### 2.4 ISPN — 光照源处理

**三路输出分工**：

$$\text{lit\_up\_map} = 1 + b_{\text{max}} \cdot \sigma\left(\text{L\_ratio} + \delta(\text{f\_illum\_feat})\right) \quad\text{(像素增亮增益, [1,1+4])}$$
$$\text{f\_illum\_feat} = \text{refine}(s_{\text{illum}}\_ \text{proj}(s_{\text{illum}}) + \text{CoarseAdapter}(\mathbf{F}_{\text{aligned}}))$$
$$A_{\text{illu}} = \sigma\left(\text{DWConv}_{3\times3} \to \text{Conv}_{1\times1} \to \sigma\right)(\text{f\_illum\_feat})$$

其中 $A_{\text{illu}} \in [0,1]$ 为空间光照注意力：暗区 → $A_{\text{illu}} \uparrow$ → 放大提亮；亮区 → $A_{\text{illu}} \downarrow$ → 抑制过曝。

F_t_aligned 锚定（防帧间闪烁）：
$$\text{f\_illum\_feat} = \text{f\_illum\_feat} + \tanh(g) \cdot \text{illu\_anchor}([\text{f\_illum\_feat}, \mathbf{F}_{\text{aligned}}])$$

### 2.5 NDPN — 噪声退化处理

**置信度引导去噪**：

1. 从 C_omega 对角线提取 correspondence confidence：
$$c = \text{conf\_proj}(\text{diag}(C_\omega^{(1)}),\dots,\text{diag}(C_\omega^{(T-1)})) \quad\text{(diag → confidence)}$$

2. 噪声特征提取 (encoder vs aligned)：
$$\mathbf{n} = \text{noise\_extract}([\mathbf{F}_{\text{enc}}, \mathbf{F}_{\text{aligned}}])$$

3. 置信度门控去噪强度：
$$\text{strength} = \sigma(\text{denoise\_strength}([\mathbf{n}, c])), \quad \gamma_n = 0\text{-init}$$
$$\mathbf{f}_{\text{noise}} = \mathbf{F}_{\text{enc}} - \gamma_n \cdot \mathbf{n} \odot \text{strength} + \text{noise\_proj}(s_{\text{noise}})$$

**直觉**: $c \uparrow$ (高对应) → 时域去噪充分 → strength $\downarrow$ (保留细节)；$c \downarrow$ → 空域去噪增强。

### 2.6 MCPN — 运动补偿处理

**运动强度补偿**：

1. 从 C_omega 对角线偏离估计运动强度：
$$m = \text{motion\_estimator}(\text{diag}(C_\omega^{(1)}),\dots,\text{diag}(C_\omega^{(T-1)}))$$

2. 对齐邻帧 (窗口相关) + 运动修正：
$$\mathbf{F}_{\text{omega}} = \text{window\_corr}(\mathbf{F}_{\text{aligned}}, \mathbf{F}_{\text{neighbors}})$$
$$\mathbf{\Delta}_m = \text{motion\_refine}([\mathbf{F}_{\text{aligned}}, \mathbf{F}_{\text{omega}}])$$

3. 运动强度门控：
$$\text{comp} = \sigma(\text{comp\_gate}([\mathbf{F}_{\text{aligned}}, m])), \quad \gamma_m = 0.1$$
$$\mathbf{f}_{\text{motion}} = g_t \odot \mathbf{F}_{\text{aligned}} + (1-g_t) \odot \mathbf{F}_{\text{omega}} + \gamma_m \cdot \mathbf{\Delta}_m \odot \text{comp}$$

### 2.7 CXG — 交叉激励门

**动机**: 训练时协调 NDPN/MCPN 梯度冲突（去噪 vs 运动补偿方向可能不一致），推理时融合为静态缩放（DRNet 重参数化范式）。

$$\mathbf{f}_{\text{noise}}^{\text{out}} = \mathbf{f}_{\text{noise}} \odot \begin{cases} \sigma(\text{SE}(\mathbf{f}_{\text{motion}})) & \text{train} \\ \bar{g}_n & \text{infer (deploy)} \end{cases}$$
$$\mathbf{f}_{\text{motion}}^{\text{out}} = \mathbf{f}_{\text{motion}} \odot \begin{cases} \sigma(\text{SE}(\mathbf{f}_{\text{noise}})) & \text{train} \\ \bar{g}_m & \text{infer (deploy)} \end{cases}$$

### 2.8 SGRF — 阶段式修复融合

**三阶段 Retinex 修正** (Denoise → Deblur → Brighten)：

$$\text{S1 (去噪): } \quad \mathbf{I}_1 = \text{clamp}\left(\mathbf{I}_{\text{center}} + \delta_1(\mathbf{f}_{\text{noise}}^{\text{out}}), 0, 1\right)$$
$$\text{S2 (去模糊): } \quad \mathbf{I}_2 = \text{clamp}\left(\mathbf{I}_1 + \delta_2(\mathbf{f}_{\text{motion}}^{\text{out}}), 0, 1\right)$$
$$\text{S3 (提亮): } \quad \mathbf{I}_3 = \text{clamp}\left((\mathbf{I}_2 + 0.01) \odot \text{lit\_up\_map} \odot (1+A_{\text{illu}}), 0, 1\right)$$

其中 $\delta_1, \delta_2$ 为 ResBlock 残差。+0.01 bias 防止暗区梯度消失。

---

## 3. 核心模块详解

### 3.1 SWD — 空域小波分流器

**文件**: `models/modules/swd.py`

子带级分流（不做 inverse DWT），显式分离"光照+噪声"和"光照无关结构"。

```
Encoder feat → [HaarDWT] → LL, LH, HL, HH
  ├─ alpha_net(LL) → α ∈ (0,1)
  ├─ LL_tfde = α × LL          (光照估 计信号)
  ├─ LL_tca = IN((1-α) × LL)   (去光照, 保结构)
  ├─ noise_gate(HF_energy) → n_gate ∈ (0,1)
  ├─ HF_tfde = n_gate × HF_cat  (噪声相关高频)
  ├─ HF_tca = LN((1-n_gate) × HF_cat)  (结构高频, 归一化)
  └─ proj(4C→C)+LN → feat_tfde, feat_tca
```

### 3.2 TCA — 时序对应对齐

**文件**: `models/modules/pure_rwkv_sace.py`

**空间扫描 Bi-WKV**:
```
k.clamp(-8,8), v.clamp(-8,8)
w = -F.softplus(spatial_decay)  → ew < 1 数学保证
chunk-wise cumsum (CHUNK=256)  → 防长序列数值溢出
双向 (fwd+bwd)/2
per-direction independent BiWKV + per-head LayerNorm
R/K/V RWKV-7 风格小初始化 ±0.05~0.5/√C
pre_norm LayerNorm 在 R/K/V 投影前
```

**四方向扫描**: 水平 / 垂直 / 主对角线 / 副对角线

### 3.3 数值稳定性

| 组件 | 措施 |
|------|------|
| BiWKV | `k.clamp(-8,8)`, `v.clamp(-8,8)` |
| BiWKV | `w = -F.softplus(spatial_decay)` → ew < 1 |
| BiWKV | chunk-wise cumsum (CHUNK=256) |
| SpatialWKV2D | `pre_norm` LayerNorm 在 R/K/V 投影前 |
| SpatialWKV2D | R/K/V RWKV-7 小初始化 |
| SWD | `proj_tfde/proj_tca` 后 LayerNorm → norm≈1 |
| Tau | `F.softplus(tau_raw) + 0.05` → 下界 0.05 |

---

## 4. 参数分布

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

## 5. 版本演进

| 版本 | 关键改动 | 参数量 | 状态 |
|------|---------|--------|------|
| v5.9.2 | s_illum 复生 + IFPN 监督 | 1.14M | 20.39 PSNR |
| v6.5 | PureRWKV 移除 DAT | 1.17M | 20.36 |
| v6 Delta | 空间扫描2D-WKV + C_omega + F_t_aligned | 1.64M | 训练崩溃 |
| v6 Delta Mark1 | SWD子带分流 + 命名统一 + 数值稳定 | 1.69M | ep15 loss收敛 |
| **v6 Delta Mark2** | **Kendall不确定性加权 + PE Loss + 感知解耦 + 损失调度** | **1.69M** | 训练中(ep16) |

---

## 6. 损失函数设计 (Mark2)

**文件**: `losses/losses.py` — `TFSNetLoss`

### 6.1 Kendall 不确定性加权

$$L_{\text{total}} = \sum_{i} \frac{1}{2\exp(s_i)} L_i + \frac{1}{2}s_i$$

其中 $s_i = \log\sigma_i^2$ 为可学习 log-variance。损失大的任务自动降权，损失小的任务自动提权。

### 6.2 各损失项

| 项 | 公式 | 监督对象 |
|----|------|---------|
| **L_pix** | PECharbonnier(res_t, GT) | 最终输出 |
| **L_freq** | L1(\|FFT(res)\|, \|FFT(GT)\|) | 频域纹理 |
| **L_ssim** | 1 - SSIM(res_t, GT) | 结构相似度 |
| **L_perc** | L1(VGG_relu3_3, VGG_relu3_3(GT)) | 感知一致性 |
| **L_illum** | TV(s_illum)·e^{-\|\nabla I\|} (+ 0.02·L1监督) | s_illum |
| **L_inter** | Charbonnier(img_s2×lit_up_map, GT) | 中间乘法路径 |
| **L_ifpn_sup** | Charbonnier(ifpn_side, GT↓) | ISPN 侧输出 |

### 6.3 损失调度 (Phase Schedule)

| 阶段 | Epoch | 行为 |
|------|-------|------|
| **Warmup** | 0-4 | 仅 L_pix + L_ssim (屏蔽 L_perc, L_freq, 中间监督) |
| **Phase 1** | 5-14 | 全损失，perc log_var=2.0 (极低权重≈0.07) |
| **Phase 2** | 15-34 | perc log_var reset→-1.0 (正常权重≈1.35), freq 解锁 |
| **Phase 3** | 35-49 | lr ×0.3，所有损失自由竞争 |

---

## 7. 训练策略

### 7.1 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 1 | 物理 batch |
| grad_accum_steps | **3** | 等效 batch=3 |
| epochs | 50 | — |
| lr | 8e-4 (Phase 1-2), **2.4e-4** (Phase 3) | Adam |
| warmup_epochs | 5 | 线性 (0.01→1.0×lr) |
| weight_decay | 1e-4 | L2 正则 |
| grad_clip | 0.5 | 梯度裁剪 |

### 7.2 数据与评估

| 项目 | 配置 |
|------|------|
| 训练集 | SDSD indoor (2064 samples), crop=256×256, T=5 |
| 验证集 | SDSD test **前 5 段** (~620 frames) |
| 评估频次 | 每 **10 epoch** |
| 评估耗时 | ~40 min/eval |
| 推理 | `infer.py`, tile_size=256, no AMP |

### 7.3 训练监控

```bash
tail -f outputs/sdsd_delta/nohup.out
grep "Train stats" outputs/sdsd_delta/nohup.out | tail -20
grep "non-finite\|NaN" outputs/sdsd_delta/nohup.out
```

---

## 8. 关键文件

| 文件 | 模块 |
|------|------|
| `models/modules/swd.py` | SWD (SpatialWaveletDiverter, HaarDWT2D) |
| `models/modules/pure_rwkv_sace.py` | TCA, BiWKV, SpatialWKV2D, MVCShift, TemporalCorrespondence, TemporalAggregation |
| `models/modules/tfsi.py` | TFDE |
| `models/modules/ifpn.py` | ISPN |
| `models/modules/ndpn.py` | NDPN |
| `models/modules/mrpn.py` | MCPN |
| `models/modules/igrf.py` | SGRF |
| `models/tfs_net.py` | CXG, TFSNet (数据流编排) |
| `losses/losses.py` | TFSNetLoss (Kendall UW + PE Loss + Phase Schedule) |
| `train.py` | 训练循环 (grad accum + loss schedule + phase lr) |
| `configs/v6_bravo.yaml` | 训练配置 |
