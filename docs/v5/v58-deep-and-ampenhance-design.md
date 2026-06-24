# v5.8 增深实验结论与 AmpEnhance 设计思路

> 日期：2026-06-23
> 关联文档：v5-experiment-comparison.md（实验比较报告），v5-design.md §10（改进方案）

---

## 1. v5.8 增深实验结论

### 1.1 实验配置

| 项目 | v5.5（基线） | v5.8 |
|------|-------------|------|
| 参数量 | 1.119M | 2.079M (+86%) |
| Encoder 瓶颈块 | 0 | 4 ResBlock (96ch, H/4) |
| IGRF 每阶段 ResBlock | 2 | 4 |
| 其他架构 | v5.5 不变 | v5.5 不变 |
| 训练参数 | batch=3, lr=0.001, warmup=10 | 同 v5.5, 50 epoch |
| 数据集 | SDSD indoor (2064 样本) | 同 |

### 1.2 SDSD 验证结果

| Epoch | v5.8 PSNR | v5.8 SSIM | v5.5 PSNR | v5.5 SSIM | 差值 |
|-------|-----------|-----------|-----------|-----------|------|
| 5 | 18.616 | 0.7493 | 19.094 | 0.7577 | **-0.48** |
| 10 | 18.654 | 0.7498 | 19.026 | 0.7590 | **-0.37** |

v5.8 增深 86% 参数后，SDSD 验证 PSNR 仍比 v5.5 差 0.4 dB。s_illum 仍塌缩（`illum=0` 从 ep3 起）。

### 1.3 DID 跨数据集评测

DID 数据集（10 个测试序列，各 100 帧，1920×1080 RGB）作为跨域泛化测试。在 SDSD 训练的模型直接在 DID test 评测，取 3 个视频各 20 帧：

| 模型 | 参数量 | SDSD ep5 PSNR | DID PSNR (3视频×20帧) | SDSD→DID 差值 |
|------|--------|--------------|---------------------|--------------|
| **v5.5** | 1.119M | 19.094 | **23.119** | +4.03 dB |
| **v5.8** | 2.079M | 18.616 | **19.737** | +1.12 dB |
| v5.5 vs v5.8 on DID | | | **+3.38 dB** | |

**关键发现**：v5.5 在 DID 上达到 23.12 dB（比 SDSD 高 4 dB，DID 退化较轻），而 v5.8 仅 19.74 dB——**增深导致跨域泛化能力大幅下降 3.38 dB**。

### 1.4 结论

**增加模型深度是错误方向**。v5.5 的 1.1M 参数在 SDSD 和 DID 上均优于 v5.8 的 2.1M。参数翻倍后在训练集（SDSD）上仅差 0.4 dB，但在跨域（DID）上差 3.38 dB——典型的过拟合：增加的容量用于记忆 SDSD 的特定分布，而非学习可泛化的低光增强能力。

**根因**：SDSD indoor 仅 2064 样本，模型容量超过数据复杂度后，额外参数用于过拟合训练分布。突破天花板的正确方向是**更好的先验注入**（如频域物理先验），而非堆参数。

---

## 2. AmpEnhance 设计思路：FourLLIE 式图像级幅度增强

### 2.1 动机

所有架构改进（LN、soft_clamp、SACE init、门控）和损失改进（s_illum 监督、多层 VGG、频域相位）均无法突破 v5.5 的 19.09 dB。增深实验证明堆参数导致过拟合。根本瓶颈是**低光输入的幅度压缩**——当前框架在特征层（Encoder 之后）处理光照，但输入图像本身已被 γ_t 压缩到极暗（均值≈0.05），后续模块在如此暗的输入上操作，特征表示能力受限。

FourLLIE (CVPR 2024) 提出在**输入图像层**做频域幅度增强：保留相位、幅度除以可学习的 curve_amps 图，从图像层恢复被压缩的幅度。这不增加模型深度，而是在正确的层级（图像层）注入频域物理先验。

### 2.2 AmpEnhance 模块设计

```
AmpEnhance(y_t):
  1. curve_amps = AmpNet(y_t)                      # 从原图估计幅度变换图
  2. curve_amps = clamp(sigmoid(curve_amps), 0.1, 1.0)  # 下界 0.1 限制最大放大 10×
  3. F = fft2(y_t, norm='ortho')                   # 全 FFT (ortho 能量守恒)
  4. mag, pha = |F|, ∠F
  5. mag_new = mag / curve_amps                    # 幅度除法提亮
  6. ỹ_t = ifft2(polar(mag_new, pha), norm='ortho').real  # 保留相位
  7. return ỹ_t
```

**AmpNet 设计**（轻量，参照 FourLLIE 但适配 TFS-Net）：
- 结构：3→16→16→3 通道，3×3 conv + GELU + 3×3 conv + GELU + 1×1 conv + sigmoid
- 从原图估计 curve_amps（空间域估计 → 频域作用）
- 全帧共用 center frame 的 curve_amps（保证 SACE 对齐一致性）
- 参数量：约 5K（远小于 Encoder/IFPN 的几十万参数）

**集成位置**：Encoder.forward_single 之前，对每帧做 AmpEnhance。

```
旧: y_t → Encoder → feats → TFSI/SACE/三分支/IGRF
新: y_t → AmpEnhance → ỹ_t → Encoder → feats → TFSI/SACE/三分支/IGRF
```

### 2.3 物理推导：幅度增强不改变 SNR

#### 2.3.1 退化模型

当前退化模型（v5-design.md §4.11）：

$$y_t = \gamma_t \cdot (x_t * k_t + n_t)$$

其中：
- $x_t$：真实场景辐照度（GT）
- $k_t$：运动模糊核（确定性，空间卷积）
- $n_t$：传感器噪声（加性零均值，方差与信号相关）
- $\gamma_t$：低光非线性亮度衰减（逐像素乘性标量）
- $y_t$：观测低光帧

#### 2.3.2 频域分解

对 $y_t$ 做 2D FFT。假设 $\gamma_t$ 空间缓变（在 crop 内近似常数），则乘性标量 $\gamma_t$ 在频域表现为对幅度谱的整体缩放，不影响相位谱：

$$\mathcal{F}\{y_t\}(u,v) = A_y(u,v) \cdot e^{j\phi_y(u,v)}$$

其中：

$$A_y(u,v) \approx \gamma_t \cdot A_{x*k+n}(u,v)$$

$$\phi_y(u,v) = \phi_{x*k+n}(u,v)$$

（乘性标量不改相位，与 v5-design.md §1.2 "γ_t 低频整体缩放、无影响相位"一致）

#### 2.3.3 幅度增强操作

AmpEnhance 估计 curve_amps ≈ γ_t，执行幅度除法：

$$A_{\tilde{y}}(u,v) = \frac{A_y(u,v)}{\text{curve\_amps}} \approx \frac{\gamma_t \cdot A_{x*k+n}(u,v)}{\gamma_t} = A_{x*k+n}(u,v)$$

$$\phi_{\tilde{y}}(u,v) = \phi_y(u,v) = \phi_{x*k+n}(u,v)$$

因此：

$$\tilde{y}_t \approx x_t * k_t + n_t$$

即幅度增强**逆转了 γ_t 的幅度压缩**，恢复到退化前（光照衰减前）的亮度水平，同时保留全部结构/噪声信息（相位不变）。

#### 2.3.4 SNR 不变的严格证明

定义信噪比为信号功率与噪声功率之比。在频域：

**增强前** ($y_t$)：

$$\text{SNR}_{y} = \frac{|\gamma_t|^2 \cdot |X_t \cdot K_t|^2}{|\gamma_t|^2 \cdot |N_t|^2} = \frac{|X_t \cdot K_t|^2}{|N_t|^2}$$

**增强后** ($\tilde{y}_t$)：

$$\text{SNR}_{\tilde{y}} = \frac{|X_t \cdot K_t|^2}{|N_t|^2}$$

因此：

$$\boxed{\text{SNR}_{\tilde{y}} = \text{SNR}_{y}}$$

**SNR 严格不变**——因为 γ_t 是乘性标量，对信号和噪声等比例缩放，比值不变。AmpEnhance 逆转 γ_t 后，信号和噪声同时被等比放大，SNR 保持恒定。

#### 2.3.5 噪声绝对值的变化

虽然 SNR 不变，噪声的**绝对幅度**发生变化：

| 量 | 增强前 ($y_t$) | 增强后 ($\tilde{y}_t$) |
|---|---|---|
| 信号幅度 | $\gamma_t \cdot \|X_t \cdot K_t\|$ | $\|X_t \cdot K_t\|$ |
| 噪声幅度 | $\gamma_t \cdot \|N_t\|$ | $\|N_t\|$ |
| **SNR** | $\|X \cdot K\| / \|N\|$ | $\|X \cdot K\| / \|N\|$ ← **不变** |
| 噪声绝对值 | 小（暗域，γ_t≪1） | **大**（恢复到退化前水平） |

噪声从"暗域小绝对值"变为"亮域大绝对值"，但 SNR 恒定——信号和噪声等比例放大。

### 2.4 AmpEnhance 对各模块的影响

#### 2.4.1 对噪声处理模块的影响

**TFSI（诊断层）**：
- SpatialBranch 计算 σ_t（时域方差）→ 增强后 σ_t 增大（噪声绝对值变大），但 μ_t 也等比增大 → **SNR 估计不变**
- s_noise 诊断的是"噪声退化强度"→ 物理上 n_t 本身没变（只是被 γ_t⁻¹ 放大）→ **s_noise 的语义不变，但数值范围变化**
- LFF 频域分支：输入特征来自增强后的图像 → 低频分量已部分恢复 → **LFF 的光照归一化负担减轻**

**SACE（对齐层）**：
- LFF 逐帧光照归一化 → 增强后帧间光照差异已减小 → **LFF 更容易对齐**
- σ_t_clean → 增大但与 μ_t_clean 等比 → **NDPN 的尺度一致化不受影响**

**NDPN（去噪分支）**：
- SNR = |μ_t_clean| / σ_t_clean → **不变**（§2.3.4 已证明）
- 聚合权重 α_i → 基于 SNR 和帧间残差 → **逻辑不变**

**IGRF Stage 1（降噪）**：
- 当前设计为"暗域降噪"——在 img≈0.05 的暗域，噪声绝对值小
- 增强后 img≈0.3-0.5，噪声绝对值增大 → Stage 1 需要更大的 delta 修正量
- DarkIR (CVPR 2025) 在引言中提及"低光环境下的噪声（散粒噪声、读出噪声）远高于白天"这一客观成像特性，但其双瓶颈设计（EBlock 提亮在前、DBlock 去噪在后）基于架构分工，**未延伸到亮域/暗域降噪效果优劣的论证**。因此 AmpEnhance 将 IGRF Stage 1 从暗域转为亮域降噪后，**效果是开放问题，需实验验证**，不预设为正面或负面

#### 2.4.2 对 IFPN 职能的影响

当前 IFPN 承担**全量光照恢复**——从 y_t（极暗，均值≈0.05）到 res_t（正常亮度）的全部提亮由 lit_up_map 完成。

加 AmpEnhance 后：

| 量 | 无 AmpEnhance | 有 AmpEnhance |
|---|---|---|
| IFPN 输入图像 | y_t ≈ 0.05 均值 | ỹ_t ≈ 0.3-0.5 均值 |
| L_t（中心帧光照） | 极小值 | 增大（接近 L_ref） |
| L_ratio = L_ref/L_t | 大（5-8×） | 接近 1（1.0-1.5×） |
| lit_up_map_raw | [1, 5]（大提亮） | 接近 [0.9, 1.3]（微调） |

**IFPN 职能从"全量光照恢复"转为"残差光照精修"**：
- AmpEnhance 做频域全局提亮（逆转 γ_t）
- IFPN 做空域局部精修（处理 AmpEnhance 的残余误差 + 内容感知的局部亮度调整）
- 分工明确：AmpEnhance=频域全局，IFPN=空域局部

**度量改进后 IFPN 职能的指标**：
1. `lit_up_map_range = max(lit_up_map) - min(lit_up_map)`：越小说明 IFPN 修正越精细
2. `L_ratio.std()`：越小说明 AmpEnhance 已处理大部分光照差异
3. 消融实验：`ΔPSNR = PSNR(with IFPN) - PSNR(IFPN disabled)`，越小说明 IFPN 职能被 AmpEnhance 分担
4. `corr(s_illum.mean(), lit_up_map.mean())`：正相关说明两者分工协调

### 2.5 框架适配性总评

| 模块 | 兼容性 | 理由 |
|------|:---:|------|
| Encoder | ✅ | 输入更亮，特征更丰富 |
| TFSI | ✅ | SNR 不变，诊断逻辑不变，LFF 负担减轻 |
| SACE | ✅ | 光照差异减小，对齐更容易 |
| IFPN | ✅ | 从全量提亮转为残差精修，职能更清晰 |
| NDPN | ✅ | SNR 不变（§2.3.4），聚合逻辑不变 |
| MRPN | ✅ | 运动信息在相位，相位保留，不受影响 |
| IGRF Stage1 | ⚠️ | 从暗域降噪转为亮域降噪，效果需实验验证 |
| IGRF Stage3 | ✅ | 提亮量减小，lit_up_map 更接近 1，更易控制 |

### 2.6 风险与缓解

**风险 1：curve_amps 估计不准导致局部噪声过度放大**
- curve_amps 空间变化，暗区 curve_amps 小→放大倍数大
- 若估计不准，暗区噪声可能被过度放大
- **缓解**：clamp(curve_amps, 0.1, 1.0)，限制最大放大 10×

**风险 2：亮域降噪效果未经验证**
- AmpEnhance 将 IGRF Stage 1 转为亮域降噪，但亮/暗域降噪优劣尚无定论
- **缓解**：先做 ablation（有/无 AmpEnhance 对比），观察 Stage 1 降噪质量

**风险 3：多帧 curve_amps 一致性**
- 如果每帧独立估计 curve_amps，帧间可能不一致→ SACE 对齐困难
- **缓解**：全帧共用 center frame 的 curve_amps，保证对齐一致性

---

## 3. 下一步实施计划

1. 实现 AmpEnhance 模块（`models/modules/amp_enhance.py`）
2. 集成到 Encoder 之前（`models/tfs_net.py` forward）
3. 创建配置 `configs/v59_ampenhance.yaml`（v5.5 基础 + AmpEnhance，不增深）
4. 训练 30 epoch + DID 跨域评测
5. Ablation：有/无 AmpEnhance 对比，观察 IFPN 职能变化和 IGRF Stage 1 降噪质量
