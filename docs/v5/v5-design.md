# TFS-Net v5 Design Document

## 0. v4.3 训练测试指标（基线记录）

### 0.1 训练配置

| 参数 | 值 |
|------|-----|
| 数据集 | SDSD indoor（70序列，2064样本） |
| 输入 | window_size=5, crop_size=256 |
| 编码器 | 4级金字塔 [32,64,96,128], fused=64 |
| Batch Size | 3 |
| 学习率 | 1e-3（CosineAnnealing + LinearWarmup 5 epoch） |
| 优化器 | AdamW, weight_decay=1e-4 |
| AMP | 关闭 |
| 梯度裁剪 | max_norm=0.5 |
| 总 Epochs | 100 |
| 验证间隔 | 每 5 epoch |

**损失权重配置（sdsd_stage1.yaml）：**

| λ_pix | λ_freq | λ_ssim | λ_perc | λ_illum | λ_inter |
|--------|--------|--------|--------|---------|---------|
| 1.0 | 0.1 | 0.1 | 0.1 | 0.1 | 0.3 |

### 0.2 验证集指标（每5 epoch）

| Epoch | PSNR (dB) | SSIM | val_l1 | 备注 |
|-------|-----------|------|--------|------|
| 5 | 18.567 | 0.7484 | 0.0860 | |
| 10 | 19.135 | 0.7540 | 0.0829 | |
| 15 | **18.229** | 0.7479 | 0.0980 | 谷值 |
| 20 | 19.184 | 0.7550 | 0.0784 | |
| 25 | **18.297** | 0.7494 | 0.0928 | 谷值 |
| 30 | 18.960 | 0.7566 | 0.0844 | |
| 35 | 18.509 | 0.7430 | 0.0864 | |
| 40 | 19.295 | 0.7586 | 0.0791 | |
| 45 | 19.343 | 0.7605 | 0.0787 | |
| 50 | 19.198 | 0.7602 | 0.0796 | |
| 55 | 19.324 | 0.7599 | 0.0792 | |
| 60 | 19.300 | 0.7588 | 0.0786 | |
| 65 | 19.744 | 0.7620 | 0.0742 | |
| 70 | 19.762 | 0.7623 | 0.0737 | |
| 75 | 19.689 | 0.7639 | 0.0746 | |
| **80** | **19.905** | **0.7653** | **0.0726** | **best checkpoint** |
| 85 | 19.579 | 0.7616 | 0.0752 | |
| 90 | 19.766 | 0.7643 | 0.0737 | |
| 95 | 19.856 | 0.7651 | 0.0727 | |
| 100 | 19.824 | 0.7648 | 0.0729 | 最终 |

### 0.3 训练损失演进

| Epoch | loss_total | loss_pix | loss_freq | loss_ssim | loss_perc | loss_illum | loss_inter |
|-------|------------|----------|-----------|-----------|-----------|------------|------------|
| 1 | 0.2473 | 0.1222 | 0.0128 | 0.3158 | 0.5549 | 0.0002 | 0.1222 |
| 10 | 0.2225 | 0.1098 | 0.0121 | 0.2777 | 0.5078 | 0.0000 | 0.1098 |
| 50 | 0.1980 | 0.0934 | 0.0117 | 0.2650 | 0.4821 | 0.0000 | 0.0934 |
| 80 | 0.1941 | 0.0915 | 0.0116 | 0.2622 | 0.4785 | 0.0000 | 0.0915 |
| 100 | 0.1933 | 0.0912 | 0.0116 | 0.2613 | 0.4745 | 0.0000 | 0.0912 |

### 0.4 关键观察

| 观察项 | 数值 | 分析 |
|--------|------|------|
| Best PSNR | **19.905 dB** (epoch 80) | 距 SOTA (~25 dB) 仍有 ~5 dB 差距 |
| Best SSIM | **0.7653** (epoch 80) | 结构恢复不足 |
| PSNR 振荡 | epoch 5~60 在 18.2~19.3 dB 间波动 | 训练不稳定 |
| 后期退化 | epoch 80→85 PSNR 19.9→19.6 dB | 过拟合或 lr 衰减过慢 |
| L_perc 居高不下 | 0.475（最终） | VGG 特征空间恢复差 |
| L_inter ≈ L_pix | 始终同步 | 中间监督未提供额外学习信号 |
| L_illum → 0 | epoch 10 后趋零 | 光照平滑约束提前饱和 |
| 100 epoch 总降幅 | loss 0.247→0.193 (↓22%) | 收敛速度慢 |

--

## 1. v4.3 理论基础与符号定义

> 本章基于实际代码实现整理，所有公式均可在对应源文件中验证。

### 1.1 物理退化模型

低光视频的成像过程可建模为复合退化函数：

$$y_t = \gamma_t \cdot (x_t * k_t + n_t)$$

其中：

| 符号 | 含义 | 维度 |
|------|------|------|
| $x_t$ | 真实场景辐照度（GT） | $\mathbb{R}^{3 \times H \times W}$ |
| $k_t$ | 运动模糊核（帧间运动 + 相机抖动） | 空间卷积核 |
| $n_t$ | 传感器噪声（高斯噪声 + 散粒噪声） | $\mathbb{R}^{3 \times H \times W}$ |
| $\gamma_t$ | 低光非线性亮度衰减（类 gamma 响应） | 逐像素标量因子 |
| $y_t$ | 观测到的低光视频帧 | $\mathbb{R}^{3 \times H \times W}$ |

退化顺序：先运动模糊（线性卷积），再加噪声（加性），最后低光衰减（乘性非线性）。

修复目标为学习逆映射 $\hat{x}_t = f^{-1}(y_t)$，物理上的合理修复**逆序**为：先去噪（逆 $n_t$），再去模糊（逆 $k_t$），最后恢复光照（逆 $\gamma_t$）。

### 1.2 傅里叶域退化分离

对退化帧做二维傅里叶变换：

$$\mathcal{F}\{y_t\}(u,v) = A(u,v) \cdot \exp(j \cdot \phi(u,v))$$

其中 $A(u,v) = |\mathcal{F}\{y_t\}|$ 为幅度谱，$\phi(u,v) = \angle\mathcal{F}\{y_t\}$ 为相位谱。

各退化因素在频域的结构性差异：

| 退化类型 | 幅度谱 $A(u,v)$ 影响 | 相位谱 $\phi(u,v)$ 影响 |
|----------|----------------------|------------------------|
| 光照衰减 $\gamma_t$ | 低频幅度整体缩放 | 无影响（乘性标量不改变相位） |
| 传感器噪声 $n_t$ | 全频段能量提升 | 全频段随机扰动 |
| 运动模糊 $k_t$ | 沿运动方向的中高频带状衰减 | 线性相移 |

此分离特性是 TFSI 频域分支和 IFPN 光照估计的物理基础。

### 1.3 符号约定

| 符号 | 含义 | 代码位置 |
|------|------|----------|
| $B$ | batch size | — |
| $T$ | 输入帧数（window_size） | `configs/sdsd_stage1.yaml` |
| $C$ | 编码器 fused 通道数（64） | `fused_channels` |
| $C_L$ | 编码器最粗层通道数（128） | `level_channels[-1]` |
| $I_t$ | 中心帧原始图像 | `image_center = x[:, T//2]` |
| $I_t^{GT}$ | 中心帧 ground truth | `target` |
| $F_i$ | 第 $i$ 帧编码器特征 | `feats[:, i]` |
| $F_i^{(L)}$ | 第 $i$ 帧最粗尺度特征 | `coarse_feats[:, i]` |
| $s_{illum}, s_{noise}$ | 双源退化强度图 | TFSI 输出，$\in [0,1]^{1 \times H \times W}$（v5: s_motion 已废弃） |

### 1.4 五阶段管线公式

#### Stage 0：PyramidEncoder — 多尺度金字塔编码

对每帧 $I_i$ 独立执行 4 级下采样 + FPN 上采样融合：

$$l_1 = \text{Stage}_1(I_i), \quad l_2 = \text{Stage}_2(l_1), \quad l_3 = \text{Stage}_3(l_2), \quad l_4 = \text{Stage}_4(l_3)$$

其中 $\text{Stage}_k$ 为两层 Conv3×3+GELU，stride=2 实现下采样。
分辨率：$l_1 \in \mathbb{R}^{32 \times H \times W}$，$l_2 \in \mathbb{R}^{64 \times H/2}$，$l_3 \in \mathbb{R}^{96 \times H/4}$，$l_4 \in \mathbb{R}^{128 \times H/8}$。

FPN 自顶向下融合：

$$p_4 = W_4 l_4, \quad p_3 = W_3 l_3 + \text{Up}(p_4), \quad p_2 = W_2 l_2 + \text{Up}(p_3), \quad p_1 = W_1 l_1 + \text{Up}(p_2)$$

$$F_i = \text{Fuse}(p_1) \in \mathbb{R}^{C \times H \times W}$$

其中 $W_k$ 为 1×1 卷积（lateral 投影到 $C$ 通道），$\text{Up}$ 为双线性上采样，$\text{Fuse}$ 为两层 Conv3×3+GELU。

输出：
- 全分辨率特征：$\{F_i\}_{i=1}^T$，shape $(B, T, C, H, W)$
- 最粗尺度特征：$\{F_i^{(L)}\}_{i=1}^T$（即 $l_4$），shape $(B, T, C_L, H/8, W/8)$

#### Stage 1：TFSI — 时频源指示器

**输入**：多帧特征 $\{F_i\}_{i=1}^T \in \mathbb{R}^{B \times T \times C \times H \times W}$

**前置处理**：逐帧 LayerNorm

$$\bar{F}_i = \text{LayerNorm2d}(F_i)$$

**空间分支**（时域统计量）：

$$\mu_t = \text{Median}_{i=1}^T \{\bar{F}_i\}, \quad \sigma_t = \sqrt{\text{Var}_{i=1}^T \{\bar{F}_i\} + \epsilon}$$

$$\text{SNR} = \frac{\mu_t}{\sigma_t + \epsilon}$$

$$F_s = \text{Conv}_{3\times3}(\text{GELU}(\text{Conv}_{3\times3}([\mu_t \| \sigma_t \| \text{SNR}]))) \in \mathbb{R}^{C \times H \times W}$$

其中 $\|$ 表示通道维拼接，输入为 $3C$ 通道。

**频域分支**（可学习频率滤波 LFF）：

$$\mathcal{F} = \text{FFT2}(\bar{F}_{center}), \quad A = |\mathcal{F}|, \quad \phi = \angle\mathcal{F}$$

径向基滤波器（$K$ 个基函数，均匀分布在归一化频率轴 $\hat{r} \in [0,1]$ 上）：

$$\psi_k(\hat{r}) = \exp\left(-\frac{(\hat{r} - \mu_k)^2}{2b^2}\right) \cdot (1 + 0.1 \cos(n_{ang} \cdot \theta))$$

其中 $\mu_k = k/(K-1)$（固定），$b$（带宽，可学习），$\theta = \text{atan2}(f_y, f_x)$。

幅度/相位独立整形（残差形式）：

$$\Delta A = \sum_{k=1}^K g_k^{(A)} c_k^{(A)} \psi_k, \quad \Delta\phi = \sum_{k=1}^K g_k^{(\phi)} c_k^{(\phi)} \psi_k$$

$$A' = A \odot (1 + \Delta A), \quad \phi' = \phi + \Delta\phi$$

$$F_f = W_{post} \cdot \text{IFFT2}(A' \cdot e^{j\phi'}) \in \mathbb{R}^{C \times H \times W}$$

其中 $g_k = \text{sigmoid}(\text{raw\_gate}_k)$，$c_k$ 为可学习系数，$W_{post}$ 为 1×1 卷积（初始化为恒等映射）。

**门控融合**：

$$g = \sigma(\text{Conv}_{1\times1}([F_s \| F_f]))$$

$$F_{fused} = g \odot F_s + (1 - g) \odot F_f \in \mathbb{R}^{C \times H \times W}$$

**双源强度输出**（v5: 移除 s_motion，独立 Sigmoid，允许多源叠加）：

$$[s_{illum}, s_{noise}] = \sigma(\text{Conv}_{1\times1}(F_{fused})) \in [0,1]^{2 \times H \times W}$$

#### Stage 2：SACE — 空间自适应跨帧增强

**输入**：多帧特征 $\{F_i\}$ + TFSI 输出

**Step 1**：逐帧 LFF 频域整形（复用 TFSI 频域分支的 LFF 模块，共享权重）

$$\tilde{F}_i = \text{LFF}(F_i), \quad i = 1, \ldots, T$$

**Step 2**：梯度友好的软中位值（soft-median）

$$m = \text{Median}_{i} \{\tilde{F}_i\} \quad \text{(no-grad)}$$

$$w_i = \text{Softmax}_i\left(-\frac{|\tilde{F}_i - m|}{\tau}\right), \quad \tau = 0.1$$

$$\mu_t^{clean} = \sum_{i=1}^T w_i \cdot \tilde{F}_i$$

所有帧都获得梯度回传（距中位值近的帧权重更大）。

**Step 3**：逐帧可变形跨帧注意力对齐

对每帧 $i$，以 $\mu_t^{clean}$ 为 query，$\tilde{F}_i$ 为 key/value：

$$(\Delta p_i, m_i) = \text{OffsetMaskHead}([\mu_t^{clean} \| \tilde{F}_i])$$

$$F_i^{aligned} = \text{DeformAttn}(\mu_t^{clean}, \tilde{F}_i, \Delta p_i, m_i) + \tilde{F}_i \quad \text{(残差连接)}$$

其中 $\Delta p_i$ 为可学习空间偏移，$m_i$ 为采样点注意力权重（Softmax 归一化），$G$ 组分组并行。

输出：$\{F_i^{aligned}\}_{i=1}^T$，$\mu_t^{clean}$

#### Stage 3：三源恢复分支（并行）

**IFPN（光照恢复）**：

光照图提取（双流输入：低分辨率原图 + 最粗尺度特征）：

$$L_t = \text{IllumExtract}([I_t^{down} \| \text{mean}(I_t^{down}) \| W_{fp} F_t^{(L)}])$$

其中 $\text{IllumExtract}$ 为 5×5 depthwise 分组卷积（groups=4），$I_t^{down}$ 为 $H/8$ 分辨率。

邻帧参考光照（余弦相似度加权）：

$$\text{sim}_i = \frac{\langle F_i^{(L)}, F_t^{(L)} \rangle}{\|F_i^{(L)}\| \cdot \|F_t^{(L)}\|}, \quad w_i = \text{Softmax}(\text{sim}_i / \tau_{sim})$$

$$L_{ref} = \sum_{i \neq t} w_i \cdot L_i$$

光照比率估计（clamp 稳定化）：

$$L_{ratio} = \text{clamp}\left(\frac{L_{ref}}{|L_t| + \epsilon},\; 0.5,\; 8.0\right)$$

Hybrid 提亮图估计（v4.3 核心：物理锚定 + 特征修正 + 有界输出）：

$$r_{feat} = W_{ratio} L_{ratio} + [s_{illum}] \xrightarrow{W_{cond}} \text{ratio\_feat} \in \mathbb{R}^{C \times H \times W}$$

$$f_{illum} = \text{Refine}(\text{ratio\_feat}) \in \mathbb{R}^{C \times H \times W}$$

$$\Delta_{lit} = W_{lit} f_{illum} \in \mathbb{R}^{3 \times H \times W} \quad \text{(64ch → 3ch)}$$

$$\text{lit\_up\_map\_raw} = 1 + M_{bright} \cdot \sigma(L_{ratio} + \Delta_{lit}) \in [1, 1+M_{bright}]^{3 \times H \times W}$$

其中 $M_{bright}=4$（最大提亮倍数），$\sigma$ 为 sigmoid 函数。

输出：$\text{lit\_up\_map\_raw}$，$f_{illum} \in \mathbb{R}^{C \times H \times W}$

**NDPN（噪声去除）**：

SNR 估计（可学习归一化参数）：

$$\widehat{\text{SNR}} = \frac{\text{mean}_c(|\mu_t^{clean}|)}{\text{mean}_c(\sigma_t) + \epsilon}, \quad \tilde{s}_{SNR} = \sigma\left(\frac{\widehat{\text{SNR}} - \tau_{mid}}{\exp(\log\tau_{scale})}\right)$$

其中 $\tau_{mid}, \log\tau_{scale}$ 为可学习标量。

双因素动态聚合权重：

$$\alpha_i = \begin{cases} \tilde{s}_{SNR} & i = t \\ \sigma(\text{Conv}(|F_i^{aligned} - F_t|)) \cdot (1 - \tilde{s}_{SNR}) & i \neq t \end{cases}$$

$$f_{noise} = \text{Refine}\left(\sum_{i=1}^T \frac{\alpha_i}{\sum_j \alpha_j} F_i^{aligned}\right) \in \mathbb{R}^{C \times H \times W}$$

**MRPN（运动补偿）**：

残差驱动的隐式遮挡处理：

$$w_i = \begin{cases} 1 & i = t \\ \sigma(-\text{Conv}(|F_i^{aligned} - F_t|)) & i \neq t \end{cases}$$

$$f_{motion} = \text{Refine}\left(\sum_{i=1}^T \frac{w_i}{\sum_j w_j} F_i^{aligned}\right) \in \mathbb{R}^{C \times H \times W}$$

#### Stage 4：IGRF v4.3 — 强度引导残差融合（顺序级联）

**Stage 4a（去噪）**：加法残差

$$\delta_{s1} = \text{Fuse}_{noise}([f_{noise} \| \text{Conv}_{3\times3}(I_t)]) \in \mathbb{R}^{3 \times H \times W}$$

$$I_{s1} = \text{clamp}(I_t + \delta_{s1},\; 0,\; 1)$$

其中 $\text{Fuse}$ 结构为：$\text{Conv}_{2C \to C} \to \text{GELU} \to \text{ResBlock} \times 2 \to \text{Conv}_{C \to 3}$。

**Stage 4b（去模糊）**：加法残差

$$\delta_{s2} = \text{Fuse}_{motion}([f_{motion} \| \text{Conv}_{3\times3}(I_{s1})])$$

$$I_{s2} = \text{clamp}(I_{s1} + \delta_{s2},\; 0,\; 1)$$

**Stage 4c（提亮）**：有界乘法（v4.3 核心）

$$\delta = \text{DeltaRefine}([\text{Conv}_{3\times3}(f_{illum}) \| \text{Conv}_{3\times3}(I_{s2})])$$

$$\text{lit\_up\_map} = \text{lit\_up\_map\_raw} \cdot (1 + \tanh(\delta) \cdot \delta_{max})$$

$$\text{lit\_up\_map} = \text{clamp}(\text{lit\_up\_map},\; \min=0.5)$$

$$\hat{I}_t = \text{clamp}(I_{s2} \odot \text{lit\_up\_map},\; 0,\; 1)$$

其中 $\delta_{max}=0.5$（提亮调整范围 $\pm 50\%$），$\tanh$ 限制调整幅度，无 `.detach()`（$L_{recon}$ 梯度可流回 NDPN/MRPN）。

### 1.5 损失函数

**总损失**：

$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{ssim} \mathcal{L}_{ssim} + \lambda_{perc} \mathcal{L}_{perc} + \lambda_{illum} \mathcal{L}_{illum} + \lambda_{inter} \mathcal{L}_{inter}$$

各项定义：

**(1) 重建损失**（Charbonnier + 频域）

$$\mathcal{L}_{pix} = \mathbb{E}\left[\sqrt{(\hat{I}_t - I_t^{GT})^2 + \epsilon}\right], \quad \epsilon = 10^{-6}$$

$$\mathcal{L}_{freq} = \|\,|\text{FFT}(\hat{I}_t)| - |\text{FFT}(I_t^{GT})|\,\|_1$$

$$\mathcal{L}_{recon} = \mathcal{L}_{pix} + \lambda_{freq} \mathcal{L}_{freq}$$

**(2) SSIM 损失**

$$\mathcal{L}_{ssim} = 1 - \text{SSIM}(\hat{I}_t, I_t^{GT})$$

**(3) 感知损失**（VGG-16 relu3_3 特征距离）

$$\mathcal{L}_{perc} = \|\phi_{VGG}(\hat{I}_t) - \phi_{VGG}(I_t^{GT})\|_1$$

**(4) 光照场边缘感知平滑**

$$\mathcal{L}_{illum} = \mathbb{E}\left[|
abla_h s_{illum}| \cdot \exp(-|
abla_h I_t^{GT}|)\right] + \mathbb{E}\left[|
abla_v s_{illum}| \cdot \exp(-|
abla_v I_t^{GT}|)\right]$$

在 GT 边缘处允许 $s_{illum}$ 不连续，平坦区域强制平滑。

**(5) 中间监督损失**（v4.3：仅 $I_{s2}$ 提亮后监督）

$$\mathcal{L}_{inter} = \mathbb{E}\left[\sqrt{(I_{s2} \odot \text{lit\_up\_map} - I_t^{GT})^2 + \epsilon}\right]$$

**超参数配置**（`sdsd_stage1.yaml`）：

| $\lambda_{freq}$ | $\lambda_{ssim}$ | $\lambda_{perc}$ | $\lambda_{illum}$ | $\lambda_{inter}$ |
|:-:|:-:|:-:|:-:|:-:|
| 0.1 | 0.1 | 0.1 | 0.1 | 0.3 |

### 1.6 梯度流路径

**主梯度路径**（$L_{recon} \to$ 各模块）：

$$\frac{\partial \mathcal{L}_{recon}}{\partial \hat{I}_t} \xrightarrow{\text{BrightenStage}} \frac{\partial}{\partial I_{s2}} = \text{lit\_up\_map} \cdot \frac{\partial \mathcal{L}_{recon}}{\partial \hat{I}_t}$$

$$\xrightarrow{\text{Stage}_{motion}} \frac{\partial}{\partial I_{s1}} \xrightarrow{\text{Stage}_{noise}} \frac{\partial}{\partial I_t}$$

无 `.detach()` 时，梯度从 $\hat{I}_t$ 经 $I_{s2} \to I_{s1} \to I_t$ 完整回传到 NDPN/MRPN。

**提亮阶段梯度**：

$$\frac{\partial \hat{I}_t}{\partial (\tanh \delta)} = I_{s2} \odot \text{lit\_up\_map\_raw} \cdot \delta_{max} \cdot \text{sech}^2(\delta)$$

$\text{sech}^2(\delta) \in (0, 1]$，$\delta=0$ 时梯度最大。梯度幅值与提亮后值 $I_{s2} \odot \text{lit\_up\_map\_raw}$ 成正比。

**IFPN 梯度路径**：

$$\frac{\partial \text{lit\_up\_map\_raw}}{\partial f_{illum}} = M_{bright} \cdot \sigma'(L_{ratio} + \Delta_{lit}) \cdot W_{lit}$$

sigmoid 导数 $\sigma' \in (0, 0.25]$，确保梯度有界且非零。

---

## 2. v4.3 瓶颈诊断

### 2.1 L_inter 与 L_pix 同源退化

v4.3 的 L_inter 计算：

```python
img_s2_lit = clamp(img_s2 * lit_up_map, 0, 1)
L_inter = Charbonnier(img_s2_lit, GT)
```

**问题**：`img_s2_lit` 本质上就是 `res_t` 的近似（两者都用 `lit_up_map` 提亮），
L_inter 和 L_recon 的 L_pix 高度相关。从训练日志看，L_inter 始终 ≈ L_pix（差距 < 0.001），
说明中间监督未产生独立于最终重建的额外梯度信号。

**梯度路径分析**：
```
L_recon → res_t = img_s2 * lit_up_map → ∂res_t/∂img_s2 = lit_up_map
L_inter → img_s2_lit = img_s2 * lit_up_map → ∂img_s2_lit/∂img_s2 = lit_up_map
```

两条路径对 img_s2 的梯度方向一致（都是 lit_up_map × (pred - GT)），L_inter 实质是 L_recon 的冗余副本。

### 2.2 顺序级联的监督困境

v4.3 级联顺序：Denoise → Motion → Brighten

| Stage | 输入 | 操作 | 输出 | 物理含义 |
|-------|------|------|------|---------|
| 1 (denoise) | image_center (暗+噪+糊) | +delta | img_s1 (暗+糊, 降噪) | 在暗域降噪 |
| 2 (motion) | img_s1 (暗+糊) | +delta | img_s2 (暗, 降噪+去糊) | 在暗域去糊 |
| 3 (brighten) | img_s2 (暗) | ×lit_up_map | res_t (亮, 全修复) | 乘法提亮 |

**核心矛盾**：Stage 1 和 Stage 2 在**暗域**操作，但 L_inter 用 **GT（亮域）** 监督。

- `img_s2` 是去噪+去糊后的**暗图像**，物理上应该是一张低亮度的清晰图像
- `img_s2_lit = img_s2 × lit_up_map` 试图用同一个 `lit_up_map` 把 img_s2 提到 GT 亮度
- 但 `lit_up_map` 是为 `image_center`（含噪+含糊）设计的，应用到已修复的 `img_s2` 上不一定最优

这意味着 L_inter 在迫使中间表示学习一个物理上不一致的映射。

### 2.3 三分支完全并行 — 信息孤岛

当前 IFPN/NDPN/MRPN 完全独立运行：

```
IFPN: feats, coarse_feats, s_illum → f_illum_feat, lit_up_map_raw
NDPN: feats, F_aligned_list, μ_t_clean, σ_t, s_noise → f_noise_out
MRPN: feats, F_aligned_list, s_motion → f_motion_out
```

**问题**：
- 光照估计不知道噪声分布 → 暗区光照估计受噪声干扰
- 去噪不知道光照水平 → 无法区分暗区噪声和正常纹理
- 去模糊不知道噪声水平 → 去糊时对噪声敏感

物理退化模型 `y = γ(x*k + n)` 中三个退化因素是耦合的，独立处理必然次优。

### 2.4 损失权重设计问题

当前配置 vs v4.1 设计文档推荐值：

| 参数 | 当前 (yaml) | v4.1 设计文档 | 差异 |
|------|-------------|-------------|------|
| λ_ssim | 0.1 | 0.2 | ↓50% |
| λ_perc | 0.1 | 0.2 | ↓50% |
| λ_illum | 0.1 | 0.01 | ↑10x |

- λ_perc=0.1 导致感知损失贡献仅 0.047（总 loss 的 24%），对高频细节恢复约束不足
- λ_illum=0.1 过高，但 L_illum 在 epoch 10 后趋零，说明权重高但约束无效
- λ_ssim=0.1 过低，SSIM loss 贡献仅 0.026（13%），结构约束弱

### 2.5 训练振荡与后期退化

PSNR 在 epoch 5~60 间明显振荡（18.2~19.3 dB），epoch 80 达到峰值后 epoch 85 回落。

可能原因：
1. **Warmup 过短**：5 epoch warmup 后直接进入 CosineAnnealing，早期梯度不稳定
2. **lr 过高**：初始 lr=1e-3 对 batch_size=3 可能过大（有效 batch 仅 3×256² 像素）
3. **缺少正则化**：无 dropout、无 label smoothing、无数据增强的强正则化

---

## 3. v5 设计方案（待讨论）

> 以下为基于瓶颈诊断的改进方向提案，具体方案待讨论确定。

### 3.1 候选改进方向

**方向 A：跨分支特征交互（Cross-Branch Feature Interaction）**

在 IFPN/NDPN/MRPN 之间引入轻量级特征交互，打破信息孤岛：

```
IFPN (光照) ←→ NDPN (噪声): 噪声感知的光照估计
NDPN (噪声) ←→ MRPN (运动): 运动感知的去噪
```

实现方式：1×1 卷积 cross-attention 或简单的特征 concat+project

**方向 B：特征域级联替代图像域级联（Feature-Domain Cascade）**

将 IGRF 的图像域级联改为特征域级联，避免 clamp 和物理不一致的中间监督：

```python
# v4.3: 图像域级联（有 clamp，中间监督在亮域但中间表示在暗域）
img_s1 = clamp(image_center + delta_noise, 0, 1)
img_s2 = clamp(img_s1 + delta_motion, 0, 1)
res_t  = clamp(img_s2 * lit_up_map, 0, 1)

# v5 候选: 特征域级联（无 clamp，中间表示无物理约束）
f_s1 = f_base + fuse_noise(f_noise, f_base)    # 特征域去噪
f_s2 = f_s1 + fuse_motion(f_motion, f_s1)      # 特征域去糊
f_s3 = f_s2 + fuse_illum(f_illum, f_s2)        # 特征域提亮
res_t = clamp(decode(f_s3), 0, 1)              # 仅最终输出 clamp
```

**方向 C：解耦中间监督（Decoupled Intermediate Supervision）**

用"退化模拟"替代直接 GT 监督中间阶段：

```python
# 不直接监督 img_s1/img_s2 与 GT 的距离
# 而是验证 "经过后续退化后能否回到输入"
img_s1_reblurred = blur(img_s1)   # 模拟运动模糊
L_cycle_1 = Charbonnier(img_s1_reblurred, image_center_denoised)  # 循环一致性
```

**方向 D：训练策略优化**

- 学习率调度：延长 warmup（10 epoch）+ CosineAnnealing with warm restarts
- 损失权重调优：恢复 λ_ssim=0.2, λ_perc=0.2, λ_illum=0.01
- 渐进式训练：先 128×128 crop 粗训练，再 256×256 微调
- EMA（Exponential Moving Average）：平滑权重，减少振荡

**方向 E：提亮机制重新设计**

v4.3 的有界乘法提亮虽然解决了梯度死区，但仍有 clamp 截断问题。

候选方案：
- **加法+sigmoid 混合**：`res = img + sigmoid(delta) * max_add`，保证输出有界且无 clamp
- **Learned tone curve**：`res = tone_curve(img, params)`，用可学习的 S 曲线替代乘法

### 3.2 优先级评估

| 方向 | 预期收益 | 实现复杂度 | 风险 |
|------|---------|-----------|------|
| A. 跨分支交互 | 高 | 中 | 低 |
| B. 特征域级联 | 高 | 中 | 中（需重设计中间监督） |
| C. 解耦中间监督 | 中 | 高 | 中 |
| D. 训练策略优化 | 中 | 低 | 低 |
 E. 提亮机制重设计 | 低~中 | 低 | 低 |

### 3.3 MRPN v5 改造方案（基于 MSPN 结构） ✅ 已实现

> 参考 MINS-Net 的 MSPN 模块，将 MRPN 从"逐帧残差权重聚合"改造为"窗口相关 + 门控融合 + 残差精炼"。
> 同步变更：TFSI IntensityHead 从 3 通道缩减为 2 通道，取消 s_motion 生成。
>
> **v5.1 更新**：`f_t` 改用 SACE 对齐后的中心帧特征 `F_aligned_list[center_idx]`（而非原始 Encoder 特征），
> 聚合仅使用对齐后的相邻帧（排除中心帧），确保全链路同域交互。

#### 3.3.1 设计动机

v4.3 的 MRPN 存在三个结构问题：

| 问题 | 表现 | 影响 |
|------|------|------|
| 无残差连接 | `f_motion_out = Refine(agg)`，无跳跃 | 梯度需穿越整个 refine 才能到达上游，信号衰减 |
| s_motion 从未使用 | 传入后在模块内无任何引用 | 浪费 TFSI 计算资源 |
| 逐帧全局权重 | 每帧一个 (B,1,H,W) 权重图 | 无法建模局部遮挡差异（同一帧在不同区域可信度不同） |
| **跨域比较** | f_t 为原始 Encoder 特征，f_omega 为 SACE 对齐特征 | 两特征域不同，窗口相关性不可靠 |

MSPN 的设计优势：
1. **窗口局部相关性**：在 `window_size` 窗口内计算 dot-product correlation，建模局部运动一致性
2. **门控融合**：sigmoid 门控混合中心帧与聚合邻帧，比全局 Softmax 归一化更灵活
3. **恒等跳跃连接**：`Refine(·) + f_t`，梯度直接回传，训练更稳定

**v5.1 同域设计动机**：

原始 Encoder 特征处于"编码域"，而 `F_aligned_list` 中所有帧都经过 LFF 频域整形 + 可变形跨帧对齐，处于"对齐域"。在窗口相关计算中跨域比较会导致匹配不准确。改用 `F_aligned_list[center_idx]` 作为中心帧后，所有交互（聚合、门控、跳跃）均在同一对齐域内完成，语义更一致。

#### 3.3.2 MRPN v5 架构

**输入**：

| 参数 | 类型 | 形状 | 来源 | 是否使用 |
|------|------|------|------|----------|
| `F_aligned_list` | List[Tensor] | $T \times (B, C, H, W)$ | SACE 输出 | ✅ 中心帧 + 邻帧 |
| `center_idx` | int | — | `T // 2` | ✅ |
| ~~`F_center`~~ (原始 Encoder) | Tensor | $(B, C, H, W)$ | ~~`feats[:, center_idx]`~~ | ❌ **废弃**（改用对齐后的中心帧） |
| ~~`s_motion`~~ | Tensor | $(B, 1, H, W)$ | ~~TFSI 输出~~ v5 已移除生成 | ❌ **废弃** |
| ~~`feats`~~ | Tensor | $(B, T, C, H, W)$ | 编码器输出 | ❌ **废弃**（MRPN 不再直接使用 Encoder 特征） |

**参数**：

| 组件 | 结构 | 参数量 |
|------|------|--------|
| `gate` | $\text{Conv}_{1\times1}(2C \to C)$ | $2C^2$ |
| `refine` | ResBlock ($\text{Conv}_{3\times3} \to \text{GELU} \to \text{Conv}_{3\times3}$) | $\approx 18C^2$ |
| **总计** | | $\approx 20C^2$ |

**Step 1：窗口相关性聚合（Window Correlation Aggregation）**

从 `F_aligned_list` 提取对齐中心帧，并排除中心帧构建邻帧集合：

$$F_t^{aligned} = F_{center\_idx}^{aligned} \in \mathbb{R}^{B \times C \times H \times W}$$

$$F_{neighbors} = [F_0^{aligned}, \ldots, F_{center\_idx-1}^{aligned}, F_{center\_idx+1}^{aligned}, \ldots] \in \mathbb{R}^{B \times (T{-}1) \times C \times H \times W}$$

对邻帧做窗口填充 + 分块：

$$\bar{F}_t = \text{PadToWindow}(F_t^{aligned}), \quad \bar{F}_i = \text{PadToWindow}(F_i^{aligned})$$

$$\tilde{F}_t = \text{WindowPartition2d}(\bar{F}_t) \in \mathbb{R}^{B \times N_w \times w_s^2 \times C}$$

$$\tilde{F}_{nbrs} = \text{WindowPartitionVideo}(\bar{F}_{neighbors}) \in \mathbb{R}^{B \times N_w \times (T{-}1) \cdot w_s^2 \times C}$$

其中 $N_w = (H_p / w_s) \times (W_p / w_s)$ 为窗口数，$w_s$ 为窗口大小（默认 8），$H_p, W_p$ 为填充后尺寸。

窗口内 dot-product 相关性：

$$\text{corr} = \text{Softmax}_{-1}\left(\frac{\tilde{F}_t \cdot \tilde{F}_{nbrs}^\top}{\sqrt{C}}\right) \in \mathbb{R}^{B \times N_w \times w_s^2 \times (T{-}1) \cdot w_s^2}$$

加权聚合邻帧特征：

$$\tilde{F}_{agg} = \text{corr} \cdot \tilde{F}_{nbrs} \in \mathbb{R}^{B \times N_w \times w_s^2 \times C}$$

$$F_{omega}^{aligned} = \text{Unpad}(\text{WindowReverse2d}(\tilde{F}_{agg})) \in \mathbb{R}^{B \times C \times H \times W}$$

**语义**：每个对齐中心帧窗口位置，在**对齐邻帧**的窗口内找到最相关的特征位置并聚合。相比 v4.3 的逐帧全局权重，此方法可在同一帧的不同空间区域产生不同的聚合权重（局部遮挡感知）。聚合仅使用邻帧（排除中心帧），避免中心帧自身参与聚合导致"自增强"偏差。

**Step 2：门控融合（Gated Fusion）**

$$z = [F_t^{aligned} \| F_{omega}^{aligned}] \in \mathbb{R}^{B \times 2C \times H \times W}$$

$$g = \sigma(\text{Conv}_{1\times1}(z)) \in \mathbb{R}^{B \times C \times H \times W}$$

$$F_{fuse} = g \odot F_t^{aligned} + (1 - g) \odot F_{omega}^{aligned}$$

**语义**：sigmoid 门控逐通道、逐像素决定"信任对齐中心帧自身"还是"信任聚合后的对齐邻帧"。相比 v4.3 的归一化加权和（所有帧共享一个标量权重），门控融合可产生更细粒度的特征选择。

**Step 3：残差精炼（Residual Refinement）**

$$\hat{F}_t = \text{ResBlock}(F_{fuse}) + F_t^{aligned}$$

其中 $\text{ResBlock}(x) = x + \text{Conv}_{3\times3}(\text{GELU}(\text{Conv}_{3\times3}(x)))$，即残差学习 $\text{ResBlock}(x) = x + \Delta(x)$。

最终输出：

$$f_{motion\_out} = F_t^{aligned} + \Delta(F_{fuse}) + F_{fuse}^{conv} \quad \text{（双重残差：跳跃 + ResBlock 内部）}$$

**语义**：恒等跳跃 $F_t^{aligned}$ 确保梯度直接回传到 SACE 对齐特征，即使 refine 层未学好也不会阻断学习信号。

#### 3.3.3 伪代码

```python
class MRPN_v5(nn.Module):
    def __init__(self, channels=64, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.gate = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.refine = ResBlock(channels)

    def _aggregate_neighbors(self, f_t, f_omega):
        """窗口 dot-product 相关聚合相邻帧（不含中心帧）。"""
        b, t, c, h, w = f_omega.shape
        feat = f_omega.view(b * t, c, h, w)
        feat, pad_hw = pad_to_window(feat, self.window_size)
        hp, wp = feat.shape[-2:]
        feat = feat.view(b, t, c, hp, wp)

        f_t_padded, _ = pad_to_window(f_t, self.window_size)

        center_windows = window_partition_2d(f_t_padded, self.window_size)
        # (b, n_w, ws², c)
        feat_windows = window_partition_video(feat, self.window_size)
        # (b, n_w, t*ws², c)

        corr = torch.matmul(center_windows, feat_windows.transpose(-1, -2)) / math.sqrt(c)
        corr = torch.softmax(corr, dim=-1)

        aligned_windows = torch.matmul(corr, feat_windows)
        aligned = window_reverse_2d(aligned_windows, self.window_size, hp, wp)
        aligned = unpad_from_window(aligned, pad_hw)
        return aligned

    def forward(self, F_aligned_list, center_idx):
        """
        Args:
            F_aligned_list: List[Tensor], SACE 对齐后的全帧特征, 每项 (B, C, H, W)
            center_idx: int, 中心帧索引
        """
        f_t_aligned = F_aligned_list[center_idx]  # (B, C, H, W)

        # 仅保留相邻帧（排除中心帧），避免中心帧自聚合
        f_neighbors = torch.stack(
            [F_aligned_list[i] for i in range(len(F_aligned_list)) if i != center_idx],
            dim=1,
        )  # (B, T-1, C, H, W)

        f_omega_aligned = self._aggregate_neighbors(f_t_aligned, f_neighbors)

        # 门控融合: gate 决定信任对齐中心帧 vs 聚合邻帧
        z_t = torch.cat([f_t_aligned, f_omega_aligned], dim=1)
        g_t = torch.sigmoid(self.gate(z_t))
        f_t_fuse = g_t * f_t_aligned + (1.0 - g_t) * f_omega_aligned

        # 残差精炼 (ResBlock 内部有残差, +f_t_aligned 为第二层恒等跳跃)
        hat_f_t = self.refine(f_t_fuse) + f_t_aligned

        return {
            "f_omega_aligned": f_omega_aligned,
            "z_t": z_t,
            "G_t": g_t,
            "f_t_fuse": f_t_fuse,
            "f_motion_out": hat_f_t,
        }
```

#### 3.3.4 废弃的输入/输出流

| 项目 | v4.3 状态 | v5 状态 | 原因 |
|------|----------|---------|------|
| **输入 `s_motion`** | v4.3 传入但未使用 | ❌ **移除（TFSI 不再生成）** | v5 IntensityHead 从 3ch 缩减为 2ch |
| **输入 `F_center`** (原始 Encoder 特征) | `feats[:, center_idx]` | ❌ **废弃**（改用 `F_aligned_list[center_idx]`） | 同域设计：消除编码域与对齐域的跨域比较 |
| **输入 `feats`** (完整多帧) | 仅用 `feats[:, center_idx]` | ❌ **废弃**（MRPN 不再直接使用 Encoder 特征） | 同上 |
| **输出 `motion_weights`** | (B, T-1, H, W) 邻帧归一化权重 | ❌ **移除** | v4.3 的全局权重机制已替换为窗口相关 + 门控 |
| **输出 `G_t`** (新增) | 无 | ✅ **新增** | 门控图，可用于可视化和中间监督 |

**管线修改（`tfs_net.py`）**：

```python
# v4.3 调用
mrpn_out = self.mrpn(feats=feats, F_aligned_list=F_aligned_list,
                      s_motion=s_motion, center_idx=center_idx)

# v5 初始调用（使用原始 Encoder 中心帧）
mrpn_out = self.mrpn(F_center=feats[:, center_idx],
                      F_aligned_list=F_aligned_list,
                      center_idx=center_idx)

# v5.1 调用（改用 SACE 对齐后的中心帧，聚合仅邻帧）
mrpn_out = self.mrpn(F_aligned_list=F_aligned_list,
                      center_idx=center_idx)
```

返回字典中 `motion_weights` 替换为 `G_t_m`（门控图）。

#### 3.3.5 v4.3 vs v5 MRPN 对比

| 维度 | **MRPN v4.3** | **MRPN v5 (含 v5.1)** |
|------|--------------|-------------|
| **输入** | `feats` (B,T,C,H,W), `F_aligned_list`, ~~`s_motion`~~, `center_idx` | `F_aligned_list`, `center_idx` |
| **中心帧来源** | `feats[:, center_idx]`（原始 Encoder 编码域） | `F_aligned_list[center_idx]`（SACE 对齐域） |
| **聚合源** | 全部 T 帧对齐特征（含中心帧） | 仅邻帧（排除中心帧）|
| **对齐来源** | SACE 可变形注意力（预对齐） | SACE 可变形注意力（预对齐） |
| **相关性度量** | 逐帧全局残差：$R_i = \|F_i^{aligned} - F_t\|$ | 窗口内 dot-product：$\text{corr} = \text{Softmax}(\tilde{F}_t \cdot \tilde{F}_{nbrs}^\top / \sqrt{C})$ |
| **权重粒度** | 每帧一张全局权重图 (B,1,H,W) | 每窗口每位置独立权重 (B,N_w,ws²,(T{-}1)·ws²) |
| **遮挡建模** | 隐式（大残差→低权重） | 显式局部（窗口内 softmax 竞争） |
| **融合方式** | 归一化加权求和：$\sum \frac{w_i}{\sum w} F_i^{aligned}$ | 门控融合：$g \odot F_t^{aligned} + (1-g) \odot F_\omega^{aligned}$ |
| **融合粒度** | T 帧统一参与（含中心帧 w_t=1） | 二分：对齐中心帧 vs 聚合对齐邻帧 |
| **后处理** | `refine` = 2层 ConvBlock，**无跳跃连接** | `refine` = ResBlock，**有恒等跳跃** $+F_t^{aligned}$ |
| **残差连接** | ❌ 无 | ✅ $\hat{F}_t = \text{Refine}(F_{fuse}) + F_t^{aligned}$ |
| **窗口操作** | 无 | ✅ pad/partition/reverse（$w_s=8$） |
| **参数量** | weight_conv ≈ 2K + refine ≈ 74K ≈ **76K** (C=64) | gate ≈ 8K + refine ≈ 74K ≈ **82K** (C=64) |
| **输出** | `f_motion_out`, `motion_weights` | `f_motion_out`, `G_t` |
| **梯度路径** | 梯度必须穿越 refine 全部层 | 恒等跳跃保证梯度直达 $F_t^{aligned}$ |
| **特征域一致性** | ❌ 跨域（编码域 vs 对齐域） | ✅ 同域（全对齐域） |

#### 3.3.6 预期改进分析

| 改进点 | v4.3 问题 | v5 预期效果 |
|--------|----------|------------|
| 恒等跳跃连接 | 梯度经 refine 两层 Conv 后衰减 | 梯度直接回传 SACE 对齐特征，训练更稳定 |
| 窗口局部相关性 | 全局权重无法区分同一帧的遮挡/非遮挡区域 | 局部窗口内 softmax 竞争，自动抑制遮挡区域 |
| 门控融合 | 归一化权重 $w_t=1$ 固定，中心帧永远高权重 | sigmoid 门控可学习何时信任自身/邻帧 |
| 移除 s_motion | 传入未使用，浪费计算 | 简化接口，减少无用参数传递 |
| **同域交互** (v5.1) | f_t 为编码域，f_omega 为对齐域，跨域比较 | 全链路同域，窗口相关性更可靠 |

---

## 4. 版本演进总结

| 版本 | 核心架构 | 关键指标 |
|------|---------|---------|
| v3 | 归一化加权融合 + 共享 aux_proj | PSNR=4.83 dB, 训练停滞 |
| v3.2 | soft_median + L_ratio clamp | PSNR 无改善，梯度改善 1.5-2.2x |
| v4 | concat 融合 + 独立 BranchReconHead | 梯度改善 3.9-17.9x |
| v4.1 | 顺序级联（光照→噪声→运动）+ 中间监督 | IFPN 梯度改善 3x |
| v4.2 | 去噪→运动→提亮 + 乘法提亮 | **训练失败**（loss 不下降） |
| v4.3 | 有界乘法提亮 + hybrid lit_up_map + 移除 .detach() | **PSNR=19.9 dB, SSIM=0.765** |
| **v5** | **MRPN 窗口相关+门控融合+残差精炼 / TFSI 移除 s_motion / v5.1 同域设计** | — |

---

## 5. 文件对应关系

| 文件 | 内容 |
|------|------|
| `models/tfs_net.py` | 主网络（5 stage pipeline） |
| `models/modules/igrf.py` | IGRF v4.3（StageBlock + BrightenStage） |
| `models/modules/ifpn.py` | IFPN v4.3（hybrid lit_up_map） |
| `models/modules/ndpn.py` | NDPN 去噪分支 |
| `models/modules/mrpn.py` | MRPN 运动补偿分支 |
| `models/modules/tfsi.py` | TFSI 时频源指示器 |
| `models/modules/sace.py` | SACE 可变形跨帧对齐 |
| `models/modules/encoder.py` | PyramidEncoder 4级金字塔 |
| `losses/losses.py` | TFSNetLoss v4.3 |
| `configs/sdsd_stage1.yaml` | 训练超参数 |

---

## 6. 参考文献

1. Tu et al., "Fourier-Based Decoupling Network for Joint Low-Light Image Enhancement and Deblurring," IEEE TIP, 2025
2. Zamir et al., "Multi-Stage Progressive Image Restoration (MPRNet)," CVPR 2021
3. Feijoo et al., "DarkIR: Robust Low-Light Image Restoration," CVPR 2025
4. Chen et al., "Simple Baselines for Image Restoration (NAFNet)," ECCV 2022
5. Cai et al., "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement," ICCV 2023
