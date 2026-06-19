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
| $F_i^{(L)}$ | 第 $i$ 帧粗尺度特征 | v5.3: 由 `coarse_adapter`$(F_i^{aligned})$ 生成（原为 `coarse_feats[:, i]`） |
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

**拼接融合**（v5.2: 替代 sigmoid 门控）：

$$F_{fused} = \text{Conv}_{3\times3}(\text{GELU}(\text{Conv}_{3\times3}([F_s \| F_f]))) \in \mathbb{R}^{C \times H \times W}$$

两分支拼接后经两层 3×3 Conv 学习跨通道交互，比 sigmoid 门控（$g + (1-g) = 1$ 互补约束）表达更灵活。

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

**IFPN（光照恢复）** v5.3: 使用 SACE 对齐特征替代 Encoder 粗特征：

粗特征适配器（通道投影 + 空间下采样）：

$$F_i^{(L)} = \text{AvgPool}(\text{GELU}(W_{adapt} \cdot F_i^{aligned})) \in \mathbb{R}^{C_L \times H/4 \times W/4}$$

其中 $W_{adapt}: C_f \to C_L$，$C_L=128$。

光照图提取（双流输入：低分辨率原图 + 适配后的粗特征）：

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
IFPN: feats, aligned_feats (SACE), s_illum → f_illum_feat, lit_up_map_raw
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

## 4. LFF 机制修改（M1-M4）：相位保留、SNR 一致化、噪声感知残差门控

> 本节记录针对 TFSI/SACE 共享 LFF 机制的原理分析、四项修改方案（M1-M4）及消融实证。
> 动机：v4.3 训练中 `L_illum` 在 epoch 10 后趋零、`L_perc` 居高不下、PSNR 距 SOTA ~5 dB，
> 怀疑共享 LFF 的任务冲突与噪声分离实现存在结构性缺陷。

### 4.1 问题建模与 LFF 设计目的回顾

**物理退化模型**（§1.1）：
$$y_t = \gamma_t \cdot (x_t * k_t + n_t)$$

**傅里叶域退化签名**（§1.2，源自 FRBNet + ExpoMamba WACV 2026）：

| 退化类型 | 幅度谱影响 | 相位谱影响 | 频段 |
|----------|-----------|-----------|------|
| 光照衰减 $\gamma_t$ | 低频整体缩放 | **无影响**（乘性标量不改相位） | 低频 |
| 传感器噪声 $n_t$ | 全频段能量提升 | 全频段随机扰动 | **宽带** |
| 运动模糊 $k_t$ | 运动方向中高频带状衰减 | 线性相移 | 中高频 |

**关键物理事实**：光照是低频幅度现象，噪声是宽带幅度+相位现象——二者在频域并非完全正交（低频段同时承载光照与低频噪声）。

**LFF 的本职工作**（源自 FRBNet NeurIPS 2025）：从 Phong 光照模型扩展，理论证明频域通道比可提取"光照不变特征"。LFF 通过零DC 窗 + 可学习径向基滤波器抑制低频光照成分——**LFF 的设计目标是光照归一化，而非直接去噪**。

### 4.2 TFSI 与 SACE 各自如何使用 LFF

**TFSI 中的 LFF**（三源分离估计，`tfsi.py` FrequencyBranch）：
```
feats[:, center_idx] → LFF → F_f  (仅中心帧, 1 次调用)
F_f ‖ F_s(μ_t,σ_t,SNR) → ConcatFusion → F_fused → IntensityHead → s_illum, s_noise
```
- 角色：为强度估计提供"光照不变的结构特征" $F_f$，与保留光照信息的空间分支 $F_s$ 互补
- 目的：估计退化强度图 $s_{illum}$ / $s_{noise}$

**SACE 中的 LFF**（跨帧对齐，`sace.py` Step 1-3）：
```
for t in T: feats[:,t] → LFF → F̄_t  (全部 T 帧, T 次调用)
soft-median(F̄_1..F̄_T) → μ_t_clean  (参考帧)
DeformAttn(query=μ_t_clean, kv=F̄_t) + F̄_t → F_aligned_t
```
- 角色：对每帧做光照归一化，**在时域聚合之前**消除光照闪烁
- 目的：v3 关键"顺序修正"——若光照闪烁未消除，时域中位值会偏向某一光照水平的帧，扭曲结构信息

**共享实现**（`tfs_net.py`）：
```python
shared_lff = self.tfsi.freq_branch.lff if share_lff else None
self.sace = SACE(..., lff_module=shared_lff)
```
单一 `LFFFeatureAdapter` 实例（~2.4K 参数，全通道共享 RBF），权重绑定。

设计文档给出的三点共享理由（`v3answer.md §2.2`）：
1. **参数效率**：LFF 仅 ~20 个可学习标量，共享不影响表达力
2. **物理一致性**：两者都"提取光照不变特征"，同一滤波器概念合理
3. **训练稳定性**：避免两个 LFF 学出不同的"光照不变"定义

### 4.3 原理与噪声分离的合理性分析

#### 4.3.1 合理的部分

**(1) "光照不变"目标的一致性 — 成立**
TFSI 频域分支和 SACE Step 1 都在追求"光照不变特征"。FRBNet 的理论保证（频域通道比 → 光照不变）为二者共享同一组 RBF 系数提供概念基础。

**(2) 参数效率论证 — 成立**
LFF 仅 ~2.4K 参数（占 TFSNet ~1.26M 的 0.2%），共享 vs 独立的参数差异可忽略。

**(3) 顺序修正的物理正确性 — 成立**
SACE 先 LFF 后 median 是 v3 相对 v2 的关键修正。在光照归一化后的特征上做时域中位值，各帧光照水平一致，中位值才真实反映"干净结构"。

**(4) 中心帧 LFF 复用的工程合理性 — 成立**
TFSI 对中心帧做的 LFF 与 SACE 对中心帧做的 LFF 是同一操作，共享避免重复计算（推理时通过 `frame_cache["lff"]` 复用）。

#### 4.3.2 存在的原理性张力

**(1) TFSI 与 SACE 对 LFF 的下游目标存在隐性冲突**

| 模块 | LFF 输出用途 | 期望 LFF 行为 |
|------|------------|--------------|
| TFSI | → $F_f$ → 估计 $s_{illum}$（光照强度） | 应让光照差异**可辨** |
| SACE | → $\bar{F}_i$ → 对齐匹配 | 应让光照差异**消除** |

TFSI 的强度头操作在 $F_{fused} = \text{concat}(F_s, F_f)$ 上，其中 $F_s$（空间分支的 $\mu_t/\sigma_t/\text{SNR}$）保留了光照信息。所以 $s_{illum}$ 主要靠 $F_s$ 贡献，$F_f$ 提供"光照不变结构"作互补——这缓解了冲突，但并未消除：**共享 LFF 被 SACE 的对齐目标主导时，$F_f$ 对光照的抑制可能过强，削弱 $s_{illum}$ 估计所需的低频光照线索**。

经验佐证（§0.4）：`L_illum → 0`（epoch 10 后趋零），与上述冲突吻合。

**(2) 中心帧 vs 全帧的梯度不对称**

- TFSI：LFF 仅作用于**中心帧**（1 次）→ 梯度来自强度估计路径
- SACE：LFF 作用于**全部 T 帧**（T 次）→ 梯度来自对齐路径

共享权重下，SACE 的梯度贡献是 TFSI 的 ~T 倍。**LFF 实际被优化为"SACE 的对齐用光照归一化器"，而非"TFSI 的强度估计用频域特征提取器"**。

**(3) 梯度饥饿严重削弱 LFF 的实际学习**

`v3-design-analysis.md §2.1` 实测：SACE 平均梯度范数仅为 IGRF 的 **1.1%**，TFSI 为 52%。LFF 的梯度来自这两条路径，即便共享汇集两路梯度，仍处于"有效学习下限以下"。`v3-design-analysis.md §2.2 原因3` 进一步指出：median 操作使 T=5 时每个位置仅 1 帧获梯度（80% 帧梯度为 0），LFF（2.4K 参数）被稀疏梯度进一步削弱。

#### 4.3.3 噪声分离的实现问题

**(1) LFF 对噪声分离的作用是间接的，非直接的**
根据问题建模：光照=低频幅度 → LFF **直接处理**；噪声=宽带幅度+相位 → LFF **不直接针对**。LFF 的 RBF 可以学到衰减高频，但这会同时损伤高频结构细节——这不是 LFF 的设计意图。噪声分离由以下机制承担：
- TFSI 空间分支：$\sigma_t$（时域方差）直接度量噪声水平，**未经 LFF**
- SACE soft-median：对 LFF 后的特征做时域中位值，零均值噪声被抑制（LFF + median 协同去噪）
- NDPN：SNR 自适应加权多帧聚合，才是真正的去噪执行者

**(2) SNR 估计的尺度不一致（最实质的问题）**
NDPN 原实现（`ndpn.py`）：
```python
signal = mu_t_clean.abs().mean(dim=1, keepdim=True)   # 来自 SACE: LFF'd + soft-median
noise  = sigma_t.mean(dim=1, keepdim=True)             # 来自 TFSI: 原始 feats 的时域方差
snr_hat = signal / (noise + eps)
```
- 分子 $\mu_t^{clean}$：经过 LFF（抑制低频光照）+ soft-median（抑制噪声）→ 信号尺度已被改变
- 分母 $\sigma_t$：原始 LayerNorm 后编码器特征的时域标准差 → 未经 LFF

二者的**幅度尺度不在同一域**。这不是物理意义上的信噪比，而是混合了"清洁信号"与"原始噪声"的非一致量。$\tau_{mid}/\tau_{scale}$ 可学习参数可吸收绝对尺度差异，但 $s_{snr}$ 的**相对动态范围**和**物理可解释性**仍受损。

**(3) SACE 残差把噪声又加回来**
SACE 原实现（`sace.py`）：
```python
f_aligned = self.deform_attn(q_norm, kv_norm, offset, mask) + kv  # 残差 = +F̄_i
```
DeformAttn 输出（已被 query=$\mu_t^{clean}$ 清洁化）加上残差 $\bar{F}_i$（仍含噪声）。**$F_{aligned}$ 并非去噪特征**——噪声通过残差回流到 NDPN。这削弱了"三源分离"的干净性：SACE 的 LFF 没有阻断噪声向 NDPN 的传递。

**(4) σ_t 未经 LFF——这反而是对的**
TFSI 的 $\sigma_t$（噪声估计源）没有经过 LFF。如果 $\sigma_t$ 也经过 LFF，LFF 的低频抑制会改变噪声的频谱分布，使 $\sigma_t$ 不再反映真实噪声水平。当前设计让 $\sigma_t$ 绕过 LFF，对噪声估计合理——但也正是这种"绕过"造成了 (2) 的 SNR 尺度不一致。这是设计上的两难。

### 4.4 修改方案

针对上述分析，设计四项修改（M1-M4）。核心思路：**相位保留（安全化）+ SNR 尺度一致化（修真实缺陷）+ 噪声感知残差门控（堵噪声穿透）+ 可选的共享解耦（根治任务冲突）**。

#### M1：相位保留 LFF（核心，低风险）

**改动**：`LFFFeatureAdapter` 增加 `phase_preserving` 标志（默认 True）。`phase_preserving=True` 时 `phase_new = phase`（相位恒等），`False` 时保留原行为 `phase_new = phase + Δφ`。`RadialBasisFilter` 仍保留相位参数（`coeff_phase`/`raw_gate_phase`）以支持 M4 解耦模式下 SACE 独立启用相位整形。

**理由**：
1. 光照归一化只需幅度低频抑制，相位保留无损于光照处理（ExpoMamba：光照→幅度，结构→相位）
2. 保留相位=保留结构信息，SACE 对齐匹配更可靠
3. 保留相位=保留噪声指纹载体，让 TFSI 的 Conv 能从相位特征学到噪声模式，**有利于 $s_{noise}$ 估计**
4. 减少约一半 LFF 可学参数的更新（`coeff_phase` 冻结），降低过拟合风险
5. 恒等初始化+梯度饥饿下 $Δ\phi$ 本就≈0，显式移除只是把"de facto 行为"变成"by design"

**相位保留能/不能解决的问题**：

| 问题 | 相位保留是否解决 | 原因 |
|------|:---:|------|
| 相位破坏结构/噪声指纹风险 | ✅ 根治 | 相位恒等，不再整形 |
| 任务冲突（SACE 梯度主导 LFF） | ❌ | 冲突主要在**幅度**域，与相位无关 |
| SNR 尺度不一致 | ❌ | $\mu_t^{clean}$ 被**幅度**整形改变尺度 |
| SACE 残差噪声穿透 | ❌ | 残差 `+kv` 与相位整形无关 |
| 梯度饥饿 | ❌ | 根本性梯度衰减，与 LFF 相位无关 |

**结论**：相位保留是**必要的安全修正**，但必须配合 M2/M3 才能实质解决噪声分离问题。

#### M2：SNR 尺度一致化（修真实实现缺陷，中风险）

**改动**：SACE 输出 `sigma_t_clean = lff_stack.std(dim=1)`（LFF 后特征的时域标准差），NDPN 改用 `(μ_t_clean, σ_t_clean)`——二者同处 LFF 域，尺度一致。

**理由**：
1. $\mu_t^{clean}$ 是 soft-median（去噪信号），$\sigma_t^{clean}$ 是 std（实际噪声），二者都在 LFF 域——这才是物理一致的 SNR
2. $\tau_{mid}/\tau_{scale}$ 可学习参数吸收绝对尺度，但**相对动态范围**和**逐像素分布**需要一致才有意义
3. LFF 对低频噪声的抑制会同时影响 $\mu_t^{clean}$ 和 $\sigma_t^{clean}$ 的低频分量，二者同步变化，SNR 比值更稳定
4. 不增加参数，只增加一次 std 计算

**代价**：$\sigma_t^{clean}$ 反映的是 LFF 后的噪声（低频部分被抑制），可能低估宽带噪声的低频分量。但这对 SNR 的**相对**判断影响有限，且 $\tau_{mid}$ 可学习补偿。

#### M3：噪声感知残差门控（堵噪声穿透，中风险）

**改动**：SACE 残差从 `f_aligned = deform_attn(...) + kv` 改为 `f_aligned = deform_attn(...) + (1 - s_noise) * kv`，其中 $s_{noise}$ 来自 TFSI（已在 `tfsi_out` 中传入 SACE）。`tfsi_out=None` 时回退到 `+kv`（向后兼容）。

**理由**：
1. 高噪声区：$s_{noise}→1$，残差→0，$F_{aligned}$ 更干净（来自 query=$\mu_t^{clean}$ 的清洁采样），NDPN 负担减轻
2. 低噪声区：$s_{noise}→0$，残差≈kv，保留原始信息流和梯度路径
3. 噪声感知耦合让 SACE 和 NDPN 的分工更清晰：SACE 在高噪区输出"准对齐+干净"，NDPN 再做精细多帧聚合
4. 不破坏梯度流（低噪区残差仍在），符合 v3.2 soft-median 的"所有帧获梯度"理念

**代价**：高噪区信息略损失，但 deform_attn 输出本身已携带 query($\mu_t^{clean}$) 的清洁结构，信息冗余足够。

#### M4：解耦共享 LFF——任务特化（可选，较激进）

**改动**：`share_lff=False` 时 SACE 内部创建独立 LFF（`phase_preserving=False`，启用相位整形辅助对齐），TFSI 保留相位保留 LFF。不共享，各自独立学习。

**理由**：根治梯度不对称和任务冲突，让 TFSI 的 LFF 不被 SACE 的对齐目标带偏。
**代价**：参数翻倍（~2.4K→~4.8K，仍可忽略），失去"统一光照不变定义"的一致性。**需消融验证**。

### 4.5 各修改对原问题的覆盖矩阵

| 原问题 | M1 相位保留 | M2 SNR一致 | M3 残差门控 | M4 解耦共享 |
|--------|:---:|:---:|:---:|:---:|
| 相位破坏结构/噪声指纹风险 | ✅ 根治 | — | — | ✅ |
| 任务冲突（梯度主导） | ⚠️ 缓解 | — | — | ✅ 根治 |
| SNR 尺度不一致 | — | ✅ 根治 | — | — |
| SACE 残差噪声穿透 | — | — | ✅ 根治 | — |
| 间接噪声分离（经对齐） | ✅ 增强 | ✅ 增强 | ✅ 增强 | ✅ 增强 |
| 梯度饥饿 | — | — | — | ⚠️ 轻微 |

**注**：梯度饥饿是架构性问题（归一化加权+median稀疏），需 v3.2 提出的辅助损失等独立措施，不在 LFF 修改范围。

### 4.6 实施变更

| 文件 | 改动 | 对应方案 |
|------|------|---------|
| `models/modules/lff.py` | `LFFFeatureAdapter` 增加 `phase_preserving` 标志（默认 True）；forward 中相位恒等分支 | M1 |
| `models/modules/tfsi.py` | `FrequencyBranch` 透传 `phase_preserving=True`；TFSI 实例化时显式设 True | M1 |
| `models/modules/sace.py` | SACE 增加 `phase_preserving` 参数（独立模式支持 M4）；计算 `sigma_t_clean` 并输出；残差改为 `(1-s_noise)*kv` 噪声感知门控 | M1/M2/M3 |
| `models/modules/ndpn.py` | `sigma_t` 重命名为 `sigma_t_clean`；docstring 更新 | M2 |
| `models/tfs_net.py` | NDPN 调用改用 `sace_out["sigma_t_clean"]`；移除冗余 `sigma_t` 解包；新增 `sace_phase_preserving` 参数透传 | M2/M4接口 |
| `test_lff_phase.py` | 新建：7 项验证测试 | 验证 |
| `datasets/sdsd_dataset.py` | 新增 `max_seqs` 参数（按序列限制，消融用） | 训练支持 |
| `train.py` | `build_model` 读取消融开关；`build_dataloaders` 支持 `max_val_samples`；`validate` 支持 `val_crop_size` | 训练支持 |
| `configs/ablation_baseline.yaml` | M1-M3 baseline 配置（共享+相位保留） | 消融 |
| `configs/ablation_m4_decoupled.yaml` | M4 消融配置（解耦+SACE相位整形） | 消融 |

### 4.7 验证：单元测试 + 梯度检查

`test_lff_phase.py` 7 项测试全部通过（`python test_lff_phase.py`）：

| 测试 | 验证内容 | 结果 |
|------|---------|------|
| LFF phase_preserving 行为 | True 时相位恒等，False 时相位整形，两者输出可区分 | ✅ diff=1.18e-2 |
| LFF 梯度隔离 | True 时 `coeff_phase` 梯度为零（冻结），`coeff_mag` 正常学习；False 时两者均学习 | ✅ |
| SACE sigma_t_clean 输出 | shape 正确，与 `lff_stack.std` 完全一致（diff=0） | ✅ |
| NDPN SNR 一致性 | 接受 `sigma_t_clean`，SNR 单调（低噪 s_snr=1.0 > 高噪 0.289） | ✅ |
| SACE 残差门控 | s_noise=0/0.5/1 时 norm 单调递减（2.47→1.32→0.54）；`tfsi_out=None` 回退兼容 | ✅ |
| 端到端梯度 | LFF 共享保持；`coeff_mag` 收到梯度；三分支梯度非零；18 输出键 shape 正确 | ✅ |
| Shape 一致性 | 全模块 I/O shape 与改动前一致 | ✅ |

`test_smoke.py` 5/5 通过（无回归）。

### 4.8 消融实证：M4 解耦共享 vs Baseline（M1-M3）

**设置**：SDSD indoor，2 序列训练 / 1 序列 5 帧 validation，crop 64，batch 2，4 epoch，CPU，seed=42。此为**简短训练**，仅验证方向性趋势，非最终性能结论。

#### 训练损失演进

| Epoch | Baseline loss_total | M4 loss_total | Baseline loss_pix | M4 loss_pix | Baseline loss_perc | M4 loss_perc | Baseline loss_illum | M4 loss_illum |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.2516 | 0.2221 | 0.1556 | 0.1362 | 0.0331 | 0.0313 | 0.00335 | 0.00284 |
| 2 | 0.3521 | 0.2585 | 0.2317 | 0.1620 | 0.0468 | 0.0366 | **1.7e-6** | **4.5e-7** |
| 3 | 0.2677 | 0.2114 | 0.1710 | 0.1326 | 0.0376 | 0.0311 | **9.5e-9** | **0.0** |
| 4 | 0.2062 | 0.1912 | 0.1306 | 0.1194 | 0.0276 | 0.0263 | **0.0** | **0.0** |

#### Validation 指标

| 指标 | Baseline (共享+相位保留) | M4 (解耦+SACE相位整形) | 差异 |
|------|:---:|:---:|:---:|
| **PSNR (dB)** | **17.44** | 17.16 | M4 ↓0.28 |
| **SSIM** | **0.607** | 0.588 | M4 ↓0.019 |
| Val L1 | 0.1082 | 0.1077 | ≈持平 |
| 参数量 | 1,392,993 | 1,397,194 | M4 +4,201 |

#### 关键发现

**1. 两者 `loss_illum` 均快速趋零（epoch 2-3 即降至 ~0）**
这是最核心的发现——M4 解耦共享**未能解决 $s_{illum}$ 失效问题**。两者趋零速度几乎一致（baseline ep2=1.7e-6 vs M4 ep2=4.5e-7）。这证实了 §4.3.2 的分析：**$s_{illum}$ 失效是架构性的梯度饥饿问题**（归一化加权导致分支梯度衰减至 IGRF 的 1-8%），而非 LFF 共享/解耦所致。

**2. M4 训练损失更低但验证指标略差（PSNR -0.28dB, SSIM -0.019）**
这是典型的**过拟合信号**——M4 解耦后 SACE 独立 LFF（+4201 参数，含相位整形系数 `coeff_phase`/`raw_gate_phase`）增加了表达容量，在 2 序列的小数据上更易拟合训练集，但未泛化到验证集。

**3. M4 的 `loss_perc` 略低（0.0263 vs 0.0276）**
SACE 相位整形可能在特征层面提供了轻微的恢复辅助，但这一微小优势未转化为验证增益，反而被过拟合抵消。

**4. 差异幅度在噪声水平内**
PSNR 0.28dB 的差异在 4 epoch / 2 序列的简短训练下不具统计显著性。但方向一致（训练↓、验证↓），支持过拟合判断。

#### 结论与决策

| 决策 | 理由 |
|------|------|
| **保持 M1-M3 baseline 为默认** | 参数更少（-4201）、验证略优（PSNR +0.28dB）、相位保留物理正确 |
| **不启用 M4 解耦** | 无验证收益、增加过拟合风险、失去"统一光照不变定义"一致性 |
| **$s_{illum}$ 失效需独立解决** | 两者 `loss_illum` 均快速趋零 → 根因是梯度饥饿而非 LFF 策略 → 需辅助损失/分支重建头等 v3.2/v5 提案 |

#### 对原分析问题的最终回应

- **任务冲突（SACE 梯度主导 LFF）**：M4 试图通过解耦根治，但消融显示解耦后 $s_{illum}$ 仍失效（`loss_illum` 同样趋零）——说明任务冲突不是 $s_{illum}$ 失效的主因，梯度饥饿才是
- **共享 LFF 的合理性**：消融支持"保留共享"——共享版本验证更优且参数更省，M4 解耦的额外容量在小数据上有害无益
- **相位保留（M1）的价值**：作为安全基线，避免了 M4 相位整形带来的过拟合风险，是正确的默认选择

### 4.9 未解决与后续方向

本次 LFF 修改（M1-M3）解决了相位破坏风险、SNR 尺度不一致、SACE 残差噪声穿透三个实现缺陷，但**未解决以下架构性问题**（明确边界）：

| 问题 | 根因 | 需要的独立措施 |
|------|------|--------------|
| 梯度饥饿 | 归一化加权导致分支梯度衰减至 IGRF 的 1-8% | v3.2 辅助分支损失（λ_aux=0.2） |
| median 梯度稀疏 | T=5 时每个位置仅 1 帧获梯度 | v3.2 soft-median（已实施） |
| $L_{inter}$ 与 $L_{pix}$ 同源 | 中间监督冗余（§2.1） | v5 候选方向 B/C（特征域级联/解耦中间监督） |
| 三分支信息孤岛 | IFPN/NDPN/MRPN 完全并行无交互 | v5 候选方向 A（跨分支特征交互） |

**保留的消融资产**：

| 文件 | 用途 |
|------|------|
| `configs/ablation_baseline.yaml` | M1-M3 baseline 配置（可复现） |
| `configs/ablation_m4_decoupled.yaml` | M4 消融配置（供后续更大规模消融） |
| `outputs/ablation_baseline_train.log` | Baseline 训练日志 |
| `outputs/ablation_m4_train.log` | M4 训练日志 |
| `outputs/ablation_baseline/best.pth` | Baseline 最佳 checkpoint |
| `outputs/ablation_m4_decoupled/best.pth` | M4 最佳 checkpoint |

### 4.10 梯度饥饿与 s_illum 失效的根因诊断（实证）

> 本节基于 v5.4 实测梯度探针（`retain_grad()` 追踪中间 tensor 梯度），从当前架构精确定位 s_illum 失效与三分支梯度饥饿的成因。

#### 4.10.1 s_illum 失效：两条梯度路径的 14000:1 悬殊比

实测确认 s_illum 有两条梯度来源：

| 梯度来源 | 路径 | 实测梯度范数 |
|---------|------|-----------|
| **L_illum**（直接） | `L_illum → ∂L/∂s_illum`（边缘感知平滑，1 步） | **0.0439** |
| **L_pix**（间接） | `L_pix → res_t → lit_up_map → f_illum_feat → ratio_feat → illum_cond_proj → s_illum`（~8 步链式） | **0.0000031** |

**比值 = 0.0439 / 0.0000031 ≈ 14,000 : 1**。即使 $\lambda_{illum}=0.1$，有效比仍是 **1,400 : 1**。

##### 间接路径（L_pix）的三重衰减

1. **乘性提亮的暗域衰减（~100×）**：$\partial res_t / \partial lit\_up\_map = img_{s2}$。实测 $img_{s2}$ 均值 = **0.036**（暗域），而 $lit\_up\_map$ 均值 = **3.63**。IFPN 路径的梯度乘数是 0.036，NDPN/MRPN 路径的梯度乘数是 3.63——**IFPN 天然获得 100× 更少的梯度**。这是 $res_t = img_{s2} \times lit\_up\_map$ 乘法结构的直接后果。

2. **通道拼接稀释（~65×）**：s_illum 是 1 通道，与 ratio_feat 的 64 通道拼接成 65 通道，再经 `illum_cond_proj`（1×1 Conv 65→64）投影。s_illum 只占输入的 1/65，梯度被均摊。

3. **多层卷积链式衰减**：`lit_up_proj(64→3) → feat_refine(2层Conv) → illum_cond_proj(1×1) → ratio_proj(1×1) → IllumExtract(3层Conv)`，约 7-8 层卷积深度。

**三重衰减累计**：100 × 65 × (卷积链衰减) ≈ 14,000× — 与实测吻合。

##### L_illum 的"平滑陷阱"

$L_{illum} = |\nabla s_{illum}| \cdot \exp(-|\nabla I^{GT}|)$ 的**全局最小值是 $s_{illum}$ = 常数**（任意常数，梯度为零即 loss 为零）。由于 L_illum 梯度比 L_pix 间接梯度强 1400-14000 倍，优化器**完全被 L_illum 主导**：s_illum 在 1-2 个 epoch 内被驱动到空间常数 → $loss_{illum} \to 0$ → s_illum 失效。

#### 4.10.2 梯度饥饿：三级级联的乘性不对称

实测各模块梯度范数（仅 L_pix，排除 L_illum 干扰）：

| 模块 | 梯度范数 | vs IGRF | 衰减倍数 |
|------|---------|---------|---------|
| IGRF | 2.570 | 1.000 | — |
| encoder | 3.575 | 1.391 | 无衰减（主路径） |
| TFSI（整体） | 4.149 | 1.614 | 无衰减（LFF梯度来自SACE） |
| tfsi.intensity_head | 0.0007 | 0.0003 | **也处于饥饿中** |
| **MRPN** | 0.260 | 0.101 | **10×** |
| **NDPN** | 0.077 | 0.030 | **33×** |
| **IFPN** | 0.0075 | 0.0029 | **346×** |
| SACE | 4.289 | 1.669 | 无衰减 |

##### 成因 1：乘性提亮对 IFPN 的不对称衰减（100×）

IGRF 三级级联中，Stage 1/2 是加法（偏导=1），Stage 3 是乘法：
$$res_t = img_{s2} \times lit\_up\_map$$

- 对 $img_{s2}$（NDPN/MRPN 路径）：$\partial res_t / \partial img_{s2} = lit\_up\_map \approx 3.63$ → **放大**
- 对 $lit\_up\_map$（IFPN 路径）：$\partial res_t / \partial lit\_up\_map = img_{s2} \approx 0.036$ → **衰减 100×**

##### 成因 2：StageBlock 深度衰减（~10×）

每个 StageBlock 约 6 层卷积深度，随机初始化权重对梯度产生 ~10× 衰减。

##### 成因 3：encoder 主路径的梯度优势

encoder 有直接到 IGRF 的路径（`image_center → img_s1 → img_s2 → res_t`），每级偏导为 1 或 lit_up_map（放大），梯度 = 1.39× IGRF——**比所有分支都高**。

### 4.11 退化建模与三分支职能重梳理

#### 4.11.1 三种退化因素的物理本质

$$y_t = \gamma_t \cdot (x_t * k_t + n_t)$$

| 退化因素 | 物理本质 | 统计性质 | 频域签名 | 是否"噪声" |
|---------|---------|---------|---------|:---:|
| **$\gamma_t$（光照衰减）** | 像素级乘性缩放 | **确定性**，空间平滑 | 低频幅度缩放，相位无影响 | ❌ 确定性退化 |
| **$n_t$（传感器噪声）** | 加性随机扰动 | **随机性**，零均值 | 全频段幅度+相位扰动 | ✅ 真正的噪声 |
| **$k_t$（运动模糊）** | 空间卷积核 | **确定性**，方向性 | 中高频带状衰减+线性相移 | ❌ 确定性退化 |

关键区分：三个退化源中只有 $n_t$ 是统计意义上的"噪声"。TFS-Net 的"三源"指三种**退化源**，不是三种噪声。

#### 4.11.2 TFSI 诊断两种强度的依据

| 强度图 | 诊断 | 估计依据 | 为什么需要 |
|--------|------|---------|-----------|
| $s_{illum}$ | $\gamma_t$ | $\mu_t/\sigma_t$ + LFF 频域 | $\gamma_t$ 是乘性的，需显式估计"暗到什么程度" |
| $s_{noise}$ | $n_t$ | $\sigma_t$ 时域方差 + SNR | $n_t$ 是随机的，需估计"噪声有多强" |
| ~~$s_{motion}$~~ | ~~$k_t$~~ | — | v5 移除：MRPN 窗口相关性**隐式从对齐残差感知运动** |

#### 4.11.3 修订后的三分支职能

| 模块 | 建模的退化 | 输入来源 | 核心输出 | 角色 |
|------|-----------|---------|---------|------|
| TFSI | 诊断 $\gamma_t$、$n_t$ 强度 | Encoder 特征 | $s_{illum}$, $s_{noise}$ | **诊断**：退化有多严重 |
| IFPN | $\gamma_t$（光照衰减） | SACE 对齐特征 + 图像 | $lit\_up\_map$, $f_{illum\_feat}$ | **方案估计**：每像素提亮多少 |
| NDPN | $n_t$（传感器噪声） | SACE 对齐特征 + $\sigma_t^{clean}$ | $f_{noise\_out}$ | **方案估计**：SNR 加权去噪 |
| MRPN | $k_t$（运动模糊） | SACE 对齐特征 | $f_{motion\_out}$ | **方案估计**：窗口相关运动补偿 |
| IGRF | 执行逆序修复 | 分支特征 + $s_{illum}$ + $s_{noise}$ | $res_t$ | **执行**：按诊断+方案做修复 |

**核心设计原则**：TFSI = 诊断（what's wrong），IFPN/NDPN/MRPN = 方案规划（how to fix），IGRF = 执行（apply fix）。$s_{illum}/s_{noise}$ 作为诊断信号**直接指导 IGRF 的执行强度**，不再被稀释进方案估计模块。

**为什么 IFPN 不需要 $s_{illum}$**：IFPN 从多帧数据估计光照图（Retinex），这是内容感知的——区分"暗物体"（反射率低，不提亮）和"暗光照"（反射率高但光照低，提亮）。$s_{illum}$ 的"诊断"职能（这里光照退化有多严重）应移交 IGRF 执行层，而非在方案估计中作为 1/65 通道的弱先验。

**为什么 MRPN 不需要 $s_{motion}$**：运动模糊的"强度"等价于帧间对齐残差的大小——MRPN 的窗口 dot-product 相关性直接从 $F_{aligned\_list}$ 残差中隐式估计。

### 4.12 根治方案：s_illum/s_noise 直入 IGRF + 混合提亮 + Encoder 3 级

#### 4.12.1 设计总览

修订后 IGRF 三级级联：

```
Stage 1 (去噪):   img_s1 = clamp(img_center + Δ_noise(f_noise, img, s_noise), 0, 1)
                  s_noise 作为 additive correction 直接参与 delta 计算
                  ∂img_s1/∂s_noise = intensity_corr'  (无衰减)

Stage 2 (去模糊): img_s2 = clamp(img_s1 + Δ_motion(f_motion, img_s1), 0, 1)
                  无强度先验（s_motion 已移除）

Stage 3 (提亮):   brighten_base = img_s2 × lit_up_map               (Retinex 乘法基座)
                  illum_residual = s_illum × corr_mag(f_illum_feat)  (加法修正)
                  res_t = clamp(brighten_base + illum_residual, 0, 1)
                  ∂res_t/∂s_illum = corr_mag  (不经 img_s2，无 100× 衰减)
```

#### 4.12.2 方案 B 混合提亮的详细数据流

```
输入（全部保留，新增 s_illum）:
  lit_up_map_raw  (B,3,H,W)   ← 来自 IFPN，不变
  f_illum_feat    (B,64,H,W)  ← 来自 IFPN，不变
  img_dark        (B,3,H,W)   ← img_s2，不变
  s_illum         (B,1,H,W)   ← 新增：来自 TFSI，直入 IGRF

─── 乘法基座路径（完全保留，不动）───
  feat_cond = Conv3x3(f_illum_feat)     # 64→3
  img_cond  = Conv3x3(img_dark)          # 3→3
  delta = Fuse(concat[feat_cond, img_cond])  # 6→3，经 GELU+ResBlock×2+Conv
  lit_up_map = lit_up_map_raw × (1 + tanh(delta) × 0.5)
  brighten_base = img_dark × lit_up_map   ← Retinex 乘法提亮

─── 加法修正路径（新增，零初始化）───
  corr_mag = illum_corr(f_illum_feat)    # 64→3，新 Conv1x1，零初始化
  illum_residual = s_illum × corr_mag    # 1ch × 3ch → 3ch

─── 合并输出 ───
  res_t = clamp(brighten_base + illum_residual, 0, 1)
```

**物理含义**：$lit\_up\_map$ = "内容感知的基座提亮"（区分暗物体/暗光照），$s_{illum}$ = "诊断驱动的修正提亮"（全局光照退化强度的加法补正）。

#### 4.12.3 对称设计：s_noise 直入 Stage 1

为框架对称，$s_{noise}$ 也直入 IGRF Stage 1：
```
delta = Fuse(concat[f_branch, img_proj(img)]) + intensity_corr(s_noise)
img_s1 = clamp(img + delta, 0, 1)
```
$s_{noise}$ 作为 StageBlock 的加法修正项，直接参与 delta 计算。

完整对称结构：
- Stage 1：$f_{noise\_out}$ + **$s_{noise}$** + img → delta → $img_{s1}$
- Stage 2：$f_{motion\_out}$ + $img_{s1}$ → delta → $img_{s2}$（无强度先验）
- Stage 3：$lit\_up\_map$ + $f_{illum\_feat}$ + **$s_{illum}$** + $img_{s2}$ → $res_t$

#### 4.12.4 零初始化策略

所有新增路径零初始化，确保修改后**初始行为 = 修改前行为**：

| 新增组件 | 初始化 | 初始效果 |
|---------|--------|---------|
| `BrightenStage.illum_corr` | weight=0, bias=0 | $illum\_residual=0$ → 纯乘法提亮（与当前一致） |
| `StageBlock.intensity_corr` | weight=0, bias=0 | delta 修正=0 → 纯特征驱动 delta（与当前一致） |

#### 4.12.5 预期梯度改善

| 指标 | 修改前 | 修改后（预期） |
|------|--------|--------------|
| $s_{illum}$ 从 $L_{pix}$ 梯度 | 0.000003 | ~0.001-0.01（无乘法衰减、无通道稀释） |
| $L_{illum}$ 有效梯度 ($\lambda=0.01$) | 0.044×0.1=0.0044 | 0.044×0.01=0.00044 |
| $L_{pix}$ / $L_{illum}$ 梯度比 | 1:1400 | **~2:1 ~ 22:1**（$L_{pix}$ 主导） |

$s_{illum}$ 将**首次被重建损失有效监督**。

#### 4.12.6 Encoder 3 级简化

当前 4 级 `[32, 64, 96, 128]` → 改为 3 级 `[32, 64, 96]`：
- 移除 `stage4`（stride=2, 96→128）和 `lateral4`（128→64 投影）
- `coarse_channels` 从 128 变为 96
- 最粗尺度从 H/8 变为 H/4

收益：编码器参数减少 ~150K，降低"主路径"梯度优势；IFPN 粗特征分辨率从 H/8 提升到 H/4，更多信息。

#### 4.12.7 实施变更清单

| 文件 | 改动 | 解决的问题 |
|------|------|-----------|
| `models/modules/igrf.py` | StageBlock 加 `intensity_corr`（s_noise 修正）；BrightenStage 加 `illum_corr`（s_illum 混合提亮）；IGRF.forward 加 s_illum/s_noise 参数 | 乘法 100× 衰减 + 通道 65× 稀释 |
| `models/modules/ifpn.py` | 移除 s_illum 参数和 `illum_cond_proj`（65→64 改为不需要） | 简化 IFPN，消除稀释 |
| `models/modules/encoder.py` | 移除 stage4/lateral4，3 级 `[32,64,96]` | 降低主路径优势 |
| `models/tfs_net.py` | IFPN 调用去 s_illum；IGRF 调用加 s_illum/s_noise；level_channels 默认改 3 级 | 接线 |
| `configs/sdsd_stage1.yaml` | `level_channels: [32,64,96]`；`lambda_illum: 0.01` | 超参 |
| `test_grad_fix.py` | 新建：梯度改善验证 + shape + 零初始化 | 验证 |

#### 4.12.8 参数量变化

| 组件 | 变化 | 参数量 |
|------|------|--------|
| Encoder stage4 + lateral4 | 移除 | **-~150K** |
| IFPN illum_cond_proj | 移除 | -4,224 |
| BrightenStage illum_corr | 新增 Conv2d(64→3) | +195 |
| StageBlock intensity_corr | 新增 Conv2d(1→3, 3×3) | +30 |
| **净变化** | | **约 -154K** |

总参数从 ~1.39M 降至 ~1.24M。

---

## 5. 版本演进总结

| 版本 | 核心架构 | 关键指标 |
|------|---------|---------|
| v3 | 归一化加权融合 + 共享 aux_proj | PSNR=4.83 dB, 训练停滞 |
| v3.2 | soft_median + L_ratio clamp | PSNR 无改善，梯度改善 1.5-2.2x |
| v4 | concat 融合 + 独立 BranchReconHead | 梯度改善 3.9-17.9x |
| v4.1 | 顺序级联（光照→噪声→运动）+ 中间监督 | IFPN 梯度改善 3x |
| v4.2 | 去噪→运动→提亮 + 乘法提亮 | **训练失败**（loss 不下降） |
| v4.3 | 有界乘法提亮 + hybrid lit_up_map + 移除 .detach() | **PSNR=19.9 dB, SSIM=0.765** |
| **v5** | **MRPN 窗口相关+门控融合+残差精炼 / TFSI 移除 s_motion / v5.1 MRPN同域设计 / v5.2 TFSI concat融合 / v5.3 IFPN同域设计 / v5.4 LFF 相位保留+SNR一致化+噪声感知残差门控 (M1-M3) / v5.5 s_illum/s_noise直入IGRF+混合提亮+Encoder 3级** | — |

---

## 6. 文件对应关系

| 文件 | 内容 |
|------|------|
| `models/tfs_net.py` | 主网络（5 stage pipeline，v5.4 新增 `share_lff`/`sace_phase_preserving` 消融开关） |
| `models/modules/igrf.py` | IGRF v4.3（StageBlock + BrightenStage） |
| `models/modules/ifpn.py` | IFPN v5.3（hybrid lit_up_map + coarse_adapter） |
| `models/modules/ndpn.py` | NDPN 去噪分支（v5.4: `sigma_t` → `sigma_t_clean` 尺度一致化） |
| `models/modules/mrpn.py` | MRPN 运动补偿分支 |
| `models/modules/tfsi.py` | TFSI 时频源指示器（v5.4: FrequencyBranch 相位保留） |
| `models/modules/sace.py` | SACE 可变形跨帧对齐（v5.4: `sigma_t_clean` 输出 + 噪声感知残差门控 + `phase_preserving` 参数） |
| `models/modules/lff.py` | LFF 可学习频率滤波器（v5.4: `phase_preserving` 标志） |
| `models/modules/encoder.py` | PyramidEncoder 4级金字塔 |
| `losses/losses.py` | TFSNetLoss v4.3 |
| `configs/sdsd_stage1.yaml` | 训练超参数 |
| `configs/ablation_baseline.yaml` | v5.4 M1-M3 baseline 消融配置 |
| `configs/ablation_m4_decoupled.yaml` | v5.4 M4 解耦消融配置 |
| `test_lff_phase.py` | v5.4 LFF 机制修改验证测试（7 项） |
| `datasets/sdsd_dataset.py` | SDSD 数据集（v5.4: `max_seqs` 参数支持消融） |

---

## 7. 参考文献

> §4 LFF 机制修改的相关文献见下方 §7.1；§1-§3 原始设计文献见 §7.2。

### 7.1 LFF 机制修改相关（§4）

6. **[Sing et al., NeurIPS 2025]** FRBNet: Revisiting Low-Light Vision through Frequency-Domain Radial Basis Network — LFF 光照不变特征理论依据（[GitHub](https://github.com/Sing-Forevet/FRBNet) / [OpenReview](https://openreview.net/pdf?id=FWflRgqt8X)）
7. **[Adhikarla et al., WACV 2026]** ExpoMamba: From Darkness to Detail — 幅度↔光照、相位↔结构解耦建模（M1 相位保留的物理依据）

### 7.2 原始设计相关（§1-§3）

1. Tu et al., "Fourier-Based Decoupling Network for Joint Low-Light Image Enhancement and Deblurring," IEEE TIP, 2025
2. Zamir et al., "Multi-Stage Progressive Image Restoration (MPRNet)," CVPR 2021
3. Feijoo et al., "DarkIR: Robust Low-Light Image Restoration," CVPR 2025
4. Chen et al., "Simple Baselines for Image Restoration (NAFNet)," ECCV 2022
5. Cai et al., "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement," ICCV 2023
