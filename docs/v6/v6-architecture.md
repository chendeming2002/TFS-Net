# TFS-Net v6 Delta Flight9 模型架构设计文档

> 日期：2026-07-20
> 版本：v6 Delta Flight9 (current)
> 训练配置：`configs/delta_flight9.yaml`，batch=4 (accum=4→eff=16), epochs=80
> 参数量：~1.5M (取消 WFR -25K)
> 核心变更：取消 WFR, DPE sigmoid→softplus, TCA H/2, L_illum_spatial+tv, gamma clamp

---

## 1. 概述

TFS-Net (Tri-Source Fusion & Synthesis Network) 是一个端到端多帧低光视频增强网络。Flight9 核心变更:
1. **取消 WFR** — Encoder l1/l2/l3 直连各模块,天然频率分离
2. **DPE softplus** — sigmoid→softplus+soft_clamp,单尺度 H/4,根除饱和
3. **TCA H/2** — 128×128 WKV, batch=4 恢复样本多样性
4. **L_illum_spatial + L_illum_tv** — 反零方差 + 边缘感知 TV
5. **Gamma clamp** — NDPN gamma max=0.03

| 模块 | 缩写 | Flight9 功能 |
|------|------|------|
| **Encoder** | PyramidEncoder | 3级金字塔 → l1(64,H,W), l2(64,H/2), l3(64,H/4) 多尺度直连 |
| **DPE** | Degradation Prior Estimator | 单尺度 H/4, softplus+s_max_clamp, L_spatial+L_tv 反饱和 |
| **TCA** | Temporal Correspondence & Alignment | H/2 WKV 扫描 (l2_lat 直连) + C_omega 时序矩阵 |
| **ISPN** | Illumination-Source Processing Network | TCC曲线(3ch×4×↓) + pixel-wise gain([0.5,2.0]) |
| **NDPN** | Noise Degradation Processing Network | C_omega 置信度引导去噪 (γ clamp≤0.03) |
| **MCPN** | Motion Compensation Processing Network | C_omega 运动补偿 (gamma=0.01, startup→pass-through) |
| **CXG** | Cross-eXcitation Gate | 去噪↔运动 交叉激励门 |
| **SGRF** | Stage-wise Guided Restoration & Fusion | S1:去噪→S2:去模糊→TCC×6→gain→res_t |

### 命名变更总表

| 原缩写 | 新缩写 | 中文全名 |
|--------|--------|----------|
| DWT-LFF | **Encoder直连** | 多尺度 Encoder 替代小波分流 |
| TFSI | **DPE** | 退化先验估计器 |
| SACE | **TCA** | 时序对应对齐 |
| IFPN | **ISPN** | 光照源处理网络 |
| MRPN | **MCPN** | 运动补偿处理网络 |
| IGRF | **SGRF** | 阶段式引导修复融合 |
| CrossFusionGate | **CXG** | 交叉激励门 |

### 完整数据流 (Flight9)

```
输入: I_{t-2}, I_{t-1}, I_t, I_{t+1}, I_{t+2}  (T=5 窗口)
  │
  ├─→ Encoder → l1_lat(64,H,W), l2_lat(64,H/2), l3_lat(64,H/4)  多尺度直连
  │
  ├─→ DPE(l3_lat, I_t) → s_illum(softplus), s_noise(sigmoid)
  │     Flight9: 单尺度 H/4, softplus+soft_clamp(max=3.0), L_spatial 反零方差
  │     s_illum → ISPN   s_noise → NDPN
  │
  ├─→ TCA(l2_lat) → tca_out(B,T,C,H/2,W/2), C_{t,Ω}, \hat{F}_t, μ, σ
  │     Flight9: H/2 WKV 直连, 无 internal FPN, 无 WFR 残差
  │     ↑ upsample to H×W for NDPN/MCPN/SGRF
  │
  ├─→ ISPN(l1_center, s_illum) → curve_α, gain_map
  ├─→ NDPN({F_out}, s_noise, μ, σ, C_{t,Ω}, \hat{F}_t) → f_noise_out  (γ≤0.03)
  ├─→ MCPN({F_out}, σ, C_{t,Ω}, \hat{F}_t) → f_motion_out
  │
  ├─→ CXG(f_noise_out, f_motion_out) → f_noise_gated, f_motion_gated
  │
  └─→ SGRF(gain, f_noise_gated, f_motion_gated, I_t, curve_A)
        Stage A: S1→S2→TCC×6→gain→img_lit,  Stage B: sg[img_lit]+residual→res_t
```

---

## 2. TSDR 理论框架与公式

### 记号约定

| 符号 | 含义 |
|------|------|
| $I_{t}$ | 视频中第 $t$ 帧低光输入 |
| $F_{t} = \mathcal{E}(I_{t})$ | Encoder 输出的第 $t$ 帧特征 |
| $\{F_{t-k},\dots,F_{t+k}\}$ | 以 $t$ 为中心、窗口 $T=2k+1$ 的帧集合 |
| $F_{\Omega} = \{F_{t}\mid t \in \Omega\}$ | 窗口内除中心帧外的邻帧特征集，$\Omega=\{t-k,\dots,t-1,t+1,\dots,t+k\}$ |
| $C_{t,\Omega}$ | TCA 输出的中心帧 $t$ 与邻帧集 $\Omega$ 的帧间注意力矩阵 |
| $\hat{F}_t$ | TCA 输出对齐后增强的中心帧特征 ($F_{t,\text{aligned}}$) |
| $s_{\text{illu}}, s_{\text{noise}}$ | TFDE 估计的光照退化强度和噪声退化强度 (均为 $[0,1]$) |
| $A_{\text{illu}}$ | ISPN 输出的空间光照注意力图 |

### 2.1 问题建模：三源解耦修复 (Tri-Source Decoupled Restoration)

**框架名**：TSDR (Tri-Source Decoupled Restoration Framework)，网络名：**TSD-Net**。

**主张。** 我们将低光视频增强重新表述为一个 **三源解耦修复问题**。核心洞察是：**光照退化、成像噪声与运动扰动在时域上呈现完全不同的统计行为，因此不应在早期共享同一处理路径**。

**单帧退化模型**。沿用低光成像的物理模型（[Foi et al., TIP 2008](https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf)；[Wang et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Enhancing_Low_Light_Videos_by_Exploring_High_Sensitivity_Camera_Noise_ICCV_2019_paper.pdf)），观测帧 $I_t$ 与潜在干净信号 $X_t$ 之间满足：

$$I_t = X_t \cdot \ell_t + n_t, \quad n_t \sim \mathcal{P}(\alpha \cdot X_t\ell_t) + \mathcal{N}(0, \sigma_r^2) \tag{1}$$

其中 $\ell_t \in (0,1)$ 为逐像素光照衰减系数，$n_t$ 服从泊松-高斯混合分布——泊松部分来自光子散粒噪声（signal-dependent），高斯部分来自读出噪声（signal-independent）。

**时域差分揭示的三源结构**。视频任务的关键差异在于时域。引入时域差分算子 $\Delta_t I_t = I_t - I_{t-1}$：

$$\Delta_t I_t = \underbrace{X_t\cdot(\ell_t - \ell_{t-1})}_{\Delta^{\text{illu}}_t\ \text{(光照源)}} + \underbrace{(n_t - n_{t-1})}_{\Delta^{\text{img}}_t\ \text{(成像源)}} + \underbrace{(X_t - X_{t-1})\cdot \ell_t}_{\Delta^{\text{dyn}}_t\ \text{(动态源)}} \tag{2}$$

三个差分分量在**时空统计特性**上有本质区别，见 Table 1。

**Table 1. 三源扰动的时空统计特性对比**

| 分量 | 空间分布 | 时域特性 | 频域主导 | 物理来源 | 对应模块 |
|------|---------|---------|---------|---------|---------|
| $\Delta^{\text{illu}}_t$ | 全局或大块区域 | 慢变，均值非零 | 低频 (LL 子带) | 自动曝光切换、光源闪烁 [Land & McCann 1971](https://doi.org/10.1364/JOSA.61.000001) | **ISPN** |
| $\Delta^{\text{img}}_t$ | 逐像素独立 | 帧间独立，零均值 | 全频段，HF 显著 | 光子散粒 + 读出噪声 [Foi et al. 2008](https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf) | **NDPN** |
| $\Delta^{\text{dyn}}_t$ | 空间稀疏，运动边界集中 | 非零均值，方向性 | 中高频 | 物体运动、相机抖动 [Baker et al. 2011](https://doi.org/10.1007/s11263-010-0390-2) | **MCPN** |

**核心可分性假设 (工程层面)**。在像素域，$\Delta^{\text{illu}}$、$\Delta^{\text{img}}$、$\Delta^{\text{dyn}}$ 三者**不可解析分离**——它们通过 $X_t$ 和 $\ell_t$ 相互耦合。我们主张：**通过合适的表征变换 $\mathcal{T}$，可以构造出三路先验信号，使得每路先验以某一扰动为主导**：

$$\mathcal{T}: \{I_{t-k},\ldots,I_{t+k}\} \mapsto \{p_{\text{illu}}, p_{\text{img}}, p_{\text{dyn}}\} \tag{3}$$

**TSDR 框架的具体实现**。我们将 $\mathcal{T}$ 分解为两个协同变换：

- **SWD (空域小波分流)**：利用 Haar DWT 频域分离，LL 子带天然对应 $\Delta^{\text{illu}}$ 的低频主导性，HF 子带对应 $\Delta^{\text{img}}$ 与结构成分的高频分布。与 [AFD-LLIE (2024)](https://doi.org/10.48550/arxiv.2409.01641) 的 Laplace 金字塔二源解耦一致，但我们扩展到三源。

- **TCA (时序对应矩阵 $C_{t,\Omega}$)**：构造帧间空间对应关系，$C_{t,t'}$ 的对角线占优程度直接指示 $\Delta^{\text{dyn}}_t$ 的强度——对角线扩散越严重，运动越剧烈。与 [IJCAI 2025 LLVE](https://www.cse.cuhk.edu.hk/~byu/papers/C275-IJCAI2025-LLVE.pdf) 的 cross-frame correspondence 约束同源。

由此得到三路先验：

$$p_{\text{illu}} = \text{SWD}_{\text{LL}}(F_t), \quad p_{\text{img}} = \text{diag}(C_{t,\Omega}) \oplus \text{SWD}_{\text{HF}}(F_t), \quad p_{\text{dyn}} = \text{off-diag}(C_{t,\Omega}) \tag{4}$$

三路先验分别输入 ISPN、NDPN、MCPN，每个专用网络在**接近单一主导扰动**的信号上进行处理，避免了梯度冲突（与 [PDHAT (TMM 2024)](https://doi.org/10.1109/tmm.2024.3355634) 的感知解耦思想一致——不同退化属性分配到异构分支，独立损失独立回传）。

**为什么 SGRF 采用 "去噪 → 去模糊 → 提亮" 顺序**。由式 (1)，若先施加提亮变换 $I_t / \ell_t$，则噪声 $n_t / \ell_t$ 会被同比放大——在暗区 $\ell_t \to 0$ 时噪声爆炸。因此正确顺序必须是：先在原始亮度域抑制 $\Delta^{\text{img}}$（NDPN → S1），再修正帧间 $\Delta^{\text{dyn}}$ 引起的对齐偏差（MCPN → S2），最后施加光照补偿 $\ell_t^{-1}$（ISPN → S3）。与 [DP3DF (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/view/25409/25181) 的处理顺序一致，但我们通过显式的三阶段 SGRF 使这一物理约束成为**架构级归纳偏置**。

### 2.2 TCA — 时序对应对齐 (RWKV 空间扫描)

**动机**。RWKV 的 WKV 扫描需要足够长的序列来建立有意义的依赖关系。旧设计沿时间轴 $T=5$ 扫描，序列过短 ($L=5$)，WKV 无法学习帧内空间结构。新设计改为沿空间轴 $H\times W$ 扫描 ($L \gg 1000$)，帧内结构由 WKV 显式建模，帧间对应由显式 $C_{t,\Omega}$ 计算。

#### 2.2.1 RWKV 架构背景

TCA 遵循 RWKV 系列的核心架构模式 [Peng et al., 2023]：**Spatial Mixing（WKV 注意力）+ Channel Mixing（Conv FFN）分离**，两者通过残差连接组合。

**标准 RWKV 时间混合块**（语言模型版本）：

$$y_t = \frac{\sum_{i=1}^{t} e^{w_{t-i} + k_i} v_i}{\sum_{i=1}^{t} e^{w_{t-i} + k_i}} + u \cdot e^{k_t} v_t \tag{WKV-4}$$

其中 $w_{t-i} = -(t-i) \cdot \text{softplus}(w) < 0$ 保证指数衰减（receptance-weighted decay），$u$ 为当前 token bonus。通道混合块使用 Squared ReLU FFN：$W_v \cdot \text{ReLU}^2(W_k x)$。

**Vision-RWKV 的空间适配** [Duan et al., 2024]：将时间维度的一维递归展开改为二维空间展开，核心创新是 **Bi-WKV（双向 WKV）**——沿空间序列的正反双向扫描后取平均，赋予每个像素对上下文的双向感知：

$$y_\tau = \frac{\text{fwd}_\tau + \text{bwd}_\tau}{2}$$

其中 $\text{fwd}_\tau$ 和 $\text{bwd}_\tau$ 分别是正向和反向的 WKV 递归结果。

**RSRWKV 的四方向扫描** [He et al., 2025]：在 Vision-RWKV 基础上将通道均分为 4 组 $C/4$，每组沿一个方向展开为序列，实现了 2D 空间的全方向感受野覆盖，被 TCSVT 2025 接收。

#### 2.2.2 TCA 的 RWKV 实现

TCA 完整遵循 RWKV 的三步范式 [Peng et al., 2023]：**Token Shift → Spatial Mixing (WKV) → Channel Mixing (FFN)**，具体实现如下：

**Step 1 — MVC-Shift (Token Shift)**（`pure_rwkv_sace.py:28-45`）：

标准 RWKV 的时间域 Token Shift（Q-Shift）在 2D 空间上替换为 **MVC-Shift（Multi-View Context）** [He et al., 2025]——3 支空洞深度可分离卷积（d=1, 2, 3）捕捉不同空间感受野的上下文：

$$\tilde{F}_t = F_t + \sum_{d \in \{1,2,3\}} \text{PWConv}(\text{DWConv}_{3\times3}^d(F_t))$$

其中 $\text{DWConv}_{3\times3}^d$ 为 dilation=d 的 depthwise 卷积（groups=channels），$\text{PWConv}$ 为 1×1 pointwise 混合。MVC-Shift 相比标准 RWKV 的逐 token 偏移，更适合 2D 空间的局部邻域建模。

**Step 2 — SpatialWKV2D (Spatial Mixing)**（`pure_rwkv_sace.py:107-206`）：

这是 TCA 的 WKV 注意力核心，对应 RWKV 的"时间混合"在 2D 空间的投射。流程为：

```
输入 x(B,C,H,W) → pre_norm → R/K/V Linear投影 → 4方向 Bi-WKV → 融合 → post_norm → 输出
```

**pre_norm + R/K/V 投影**（RWKV-7 风格小初始化 [Peng et al., 2025]）：

$$R = x \cdot W_r, \quad K = x \cdot W_k, \quad V = x \cdot W_v$$

$W_k \sim \mathcal{U}(-\frac{0.05}{\sqrt{C}}, \frac{0.05}{\sqrt{C}})$, $W_r, W_v \sim \mathcal{U}(-\frac{0.5}{\sqrt{C}}, \frac{0.5}{\sqrt{C}})$。小初始化防止训练早期 WKV 输出幅值过大，配合 $W_{\text{out}}=0$（零初始化输出投影）保证残差路径的初始 identity 性质。

**四方向独立 Bi-WKV**：通道均分为 4 组 $C/4$，每组沿一个方向展开 $(H \times W) \to L$ 序列：

| 方向 | 扫描方式 | 物理语义 |
|------|---------|---------|
| 水平 $h$ | 行优先（raster） | 水平邻域信息传播 |
| 垂直 $v$ | 列优先 | 垂直方向上下文 |
| 主对角 $d$ | 按 $i+j$ 排序 | 对角线方向纹理 |
| 副对角 $a$ | 按 $i-j+W-1$ 排序 | 反对角线方向纹理 |

每方向独立进行 BiWKV 递归，采用 **chunk-wise cumsum**（CHUNK=256）防止长序列数值溢出：

$$S_\tau = e^{w} \cdot S_{\tau-1} + e^{k_\tau} \odot v_\tau, \quad D_\tau = e^{w} \cdot D_{\tau-1} + e^{k_\tau}$$

$$y_\tau^{(h)} = \frac{e^{u}\cdot e^{k_\tau}\odot v_\tau + S_\tau}{e^{u}\cdot e^{k_\tau} + D_\tau}$$

其中 $w = -\text{softplus}(\theta_w) < 0$ 数学保证 $e^w < 1$ 恒衰减 [Peng et al., 2023]；$u$ 为当前 token bonus（来自 `spatial_first` 参数）。每方向独立 Per-head LayerNorm 后，通过逆扫描恢复原始空间排列。

**四方向融合**：

$$\mathbf{F}_t^{\text{wkv}} = \text{proj}_{\text{out}}\left(\sigma(R) \odot [y^{(h)}; y^{(v)}; y^{(d)}; y^{(a)}]\right)$$

其中 $\sigma$ 为 Sigmoid 激活的 receptance gate，控制 WKV 输出流入残差路径的比例。$W_{\text{out}}$ 零初始化使初始 WKV 输出为零。

**Step 3 — Channel Mix (Conv FFN)**（`pure_rwkv_sace.py:326-331`）：

对应 RWKV 的"通道混合"——空间上已通过 WKV 完成信息传播，通道间通过 1×1 Conv 做非线性变换：

```python
LayerNorm2d → Conv2d(C → 4C, 1×1) → GELU → Conv2d(4C → C, 1×1)
```

与标准 RWKV 的差异：使用 GELU 替代 Squared ReLU（$\text{ReLU}^2$），1×1 Conv2d 在空间上与 per-pixel Linear 等价。

**Residual 连接**（RWKV 标准范式 [Peng et al., 2023]）：

$$\mathbf{F}_t^{\text{out}} = \mathbf{F}_t + \gamma \cdot \mathbf{F}_t^{\text{channel}}$$

其中 $\gamma \in \mathbb{R}^{1 \times C \times 1 \times 1}$ 为可学习的 residual scale（初始化为 0，等价于 LoRA B=0）。

**上采样**：TCA 在 WFR 输出分辨率（$H/2 \times W/2$）上运行全部 Spatial/Channel Mix，然后 bilinear 上采样回全分辨率 $H \times W$。这节省约 4× 的 WKV 计算量（序列长度从 $HW$ 降到 $HW/4$）。

#### 2.2.3 时序对应与聚合

**时序对应矩阵 $C_{t,\Omega}$**。中心帧与邻帧的空间 cosine similarity——在前述 Spatial Mix → Channel Mix → Upsample 全分辨率处理后计算：

$$\mathbf{q}_t = \text{proj}_q(F_t^{\text{out}}), \quad \mathbf{k}_{t'} = \text{proj}_k(F_{t'}^{\text{out}}), \quad t' \in \Omega$$

$$C_{t,t'} = \text{softmax}\left(\frac{\mathbf{q}_t \cdot \mathbf{k}_{t'}^T}{\tau}\right), \quad \tau = \text{softplus}(\tau_{\text{raw}}) + 0.05$$

$C_{t,t'}[i,j]$ 表示中心帧位置 $i$ 与邻帧位置 $j$ 的对齐置信度。对角线 $C_{t,t'}[i,i]$ 为静止对应（高值→静止，低值→运动）——这一性质被 NDPN（利用 diag 做去噪置信度）和 MCPN（利用 off-diag 做运动检测）共同使用。

**时序聚合 $\hat{F}_t$**。用 $C_{t,\Omega}$ 将邻帧 warp 到中心帧坐标系，加权聚合：

$$\hat{F}_t = \text{LN}\left(F_t^{\text{out}} + \sum_{t' \in \Omega} w_{t'} \cdot \left(C_{t,t'} \times F_{t'}^{\text{out}}\right)\right)$$

其中 $w_{t'} = \text{frame\_gate}([F_t^{\text{out}}, C_{t,t'} \times F_{t'}^{\text{out}}])$ 为数据驱动的帧级可靠性权重。

#### 2.2.4 RVK 架构对照表

| 组件 | 标准 RWKV-4/7 | TCA 实现 | 对应论文 |
|------|:-----------:|:--------:|:--------:|
| Token Shift | Q-Shift（逐 token 偏移） | MVC-Shift（3 dilated DWConv） | [RSRWKV, He et al. 2025] |
| Spatial Mixing | Time-Mixing（一维 WKV 递归） | SpatialWKV2D（四方向 Bi-WKV） | [Vision-RWKV, Duan et al. 2024] |
| Channel Mixing | $W_v\cdot\text{ReLU}^2(W_k x)$ | 1×1 Conv(C→4C→C) + GELU | [RWKV-4, Peng et al. 2023] |
| 衰减控制 | $w = -\text{softplus}(\theta)$ | 同，chunk-wise cumsum (256) | 同上 |
| 双向扫描 | 无（自回归单向） | Bi-WKV=(fwd+bwd)/2 | [Vision-RWKV] |
| R/K/V 初始化 | 标准 Xavier/Uniform | $\pm0.05\text{--}0.5/\sqrt{C}$ 小初始化 | [RWKV-7, Peng et al. 2025] |
| pre/post norm | LayerNorm | LayerNorm | 同上 |
| 多方向扫描 | 无 | 4 heads×C/4 (H/V/主对角/副对角) | [RSRWKV] |
| Residual | $+x$ | $+\gamma \cdot \text{ChannelMix}$ | [RWKV-4] |

**核心设计原则总结**：TCA 不是简单地将变压器注意力换成 WKV——它是 RWKV 架构模式的完整 Vision 实例化。通过将"时间混合+通道混合"的范式从 1D 语言序列适配到 2D 空间+时序对应，TCA 在低光视频中为后续的 NDPN（去噪）和 MCPN（运动补偿）提供了既有时序对齐保真度、又有 RWKV 线性复杂度优势的特征表示。

### 2.3 DPE — 退化先验估计器 (Flight9: softplus + L_spatial 反饱和)

**动机**。旧 DWT-LFF 通过 IDWT 重建全分辨率特征，两条分支接收几乎相同的信息（仅 LL 略有差异），退化分离失效。

**方法**。在 Haar DWT 子带级别直接分流。

**LL 低频分流 (物理动机强，保留)**：

$$F \xrightarrow{\text{HaarDWT}} \{LL, LH, HL, HH\}, \quad LL \in \mathbb{R}^{B\times C\times H/2\times W/2}$$

$$\alpha = \sigma\left(\text{DWConv}_{3\times3} \to \text{GELU} \to \text{Conv}_{1\times1}\right)(LL)$$

$$LL_{\text{DPE}} = \alpha \odot LL, \quad LL_{\text{TCA}} = \text{IN}\left((1-\alpha) \odot LL\right)$$

**HF 高频处理 (Flight3: 取消噪声门控)**。旧设计的 `noise_gate(HF_energy)` 基于"高能量=噪声"的物理假设，但在纹理丰富的正常区域会误判。Flight3 取消噪声门控——DPE 和 TCA 共享完整 HF，仅通过各自独立的 `proj_tfde/proj_tca` 层 (Conv(4C→C)+GELU+LN) 隐式学习不同的 HF 关注模式：

$$F_{\text{DPE}} = \text{LN}\left(\text{Conv}_{1\times1}([LL_{\text{DPE}}, \text{cat}(LH,HL,HH)])\right)$$

$$F_{\text{TCA}} = \text{LN}\left(\text{Conv}_{1\times1}([LL_{\text{TCA}}, \text{cat}(LH,HL,HH)])\right)$$

删除 `noise_gate` 和 `hf_tca_norm` 两个子模块，简化 WFR 并消除误分流的归纳偏置。

### 2.4 ISPN — 光照源处理 (Flight5: Target-Convergent Curve + gain)

ISPN 输出 pixel-wise TCC 曲线参数 $A(x,y)$ 和空间增益图。Flight5 将 ZeroDCE LE-curve 替换为 Target-Convergent Curve——曲线自动收敛到可学习目标 $\alpha_{\text{target}}$，永不饱和。

**Flight5 TargetConvergentCurve**：
$$A = 4\cdot\text{Tanh}\left(\text{Conv2d}_{32\to 18}\left(\text{GELU}\left(\text{Conv2d}_{64\to 32}(h)\right)\right)\right), \quad A \in \mathbb{R}^{B\times 6\times 3\times H\times W}$$

**TCC 迭代公式**：
$$\text{TCC}_n = \text{TCC}_{n-1} + A_n \odot \text{TCC}_{n-1} \odot (1 - \text{TCC}_{n-1}) \odot (\alpha_{\text{target}} - \text{TCC}_{n-1})$$

不动点：$\text{TCC}^* = 0$, $\text{TCC}^* = 1$, 或 $\text{TCC}^* = \alpha_{\text{target}}$。从暗图出发，曲线自动停在 $\alpha_{\text{target}}$。
$\alpha_{\text{target}} = \sigma(\alpha_{\text{raw}})$，可学习参数，初始=0.5。

**关键性质**：曲线永不饱和到 1.0，gain 不需充当刹车。

**空间 gain**：
$$G = 0.5 + \frac{\text{softplus}(\text{raw})}{\text{softplus}(4)} \cdot (G_{\text{max}} - 0.5)$$

gain_head bias=0.0 → 初始 G≈1.10（近似 identity），配合 TCC 输出≈0.5 → 最终输出≈0.55，合理起点。

### 2.5 NDPN — 噪声退化处理 (对应 $\sigma^2_{\text{img}}$)

**直觉**：$C_{t,\Omega}$ 对角线 $C_{t,t'}[i,i]$ 表示位置 $i$ 在帧 $t$ 和 $t'$ 间的对应一致性。高对应 → 时序去噪可靠 → 弱空域去噪（保留细节）；低对应 → 运动遮挡 → 强空域去噪。

**置信度引导去噪**：

$$c = \text{conf\_proj}\left(\text{diag}(C_{t,\Omega}^{(1)}),\dots,\text{diag}(C_{t,\Omega}^{(T-1)})\right)$$

$$\mathbf{n} = \text{noise\_extract}\left([F_t^{\text{enc}}, \hat{F}_t]\right) \quad\text{(encoder vs aligned 差异)}$$

$$\text{strength} = \sigma\left(\text{denoise\_strength}([\mathbf{n}, c])\right), \quad \gamma_n = 0.01 \quad\text{(Mod3: 10× stronger for visible noise subtraction)}$$

$$\mathbf{f}_{\text{noise}} = F_t^{\text{enc}} - \gamma_n \cdot \mathbf{n} \odot \text{strength} + \text{noise\_proj}(s_{\text{noise}})$$

### 2.6 MCPN — 运动补偿处理 (对应 $\sigma^2_{\text{dyn}}$)

**直觉**：$C_{t,\Omega}$ 对角线偏离度是运动强度的逆向指标——对角线占优 → 静止 → 少补偿；对角线扩散 → 运动 → 强补偿。

**运动强度补偿**：

$$m = \text{motion\_estimator}\left(\text{diag}(C_{t,\Omega}^{(1)}),\dots,\text{diag}(C_{t,\Omega}^{(T-1)})\right)$$

$$F_{\text{omega}} = \text{window\_corr}\left(\hat{F}_t, \{F_{t'}^{\text{out}}\}_{t' \in \Omega}\right)$$

$$\Delta_m = \text{motion\_refine}([\hat{F}_t, F_{\text{omega}}]), \quad \text{comp} = \sigma(\text{comp\_gate}([\hat{F}_t, m]))$$

$$\mathbf{f}_{\text{motion}} = g_t \odot \hat{F}_t + (1-g_t) \odot F_{\text{omega}} + \gamma_m \cdot \Delta_m \odot \text{comp}$$

其中 $\gamma_m = 0.01$ (Mod3: 10× stronger for visible motion compensation)。$g_t = \sigma(\text{gate}([\hat{F}_t, F_{\text{omega}}]))$ 控制信任对齐中心 vs 聚合邻帧。

### 2.7 CXG — 交叉激励门

**动机**。去噪和运动补偿在重叠区域（既有噪声又有运动）梯度方向可能冲突。CXG 在训练时做动态交叉调制，推理时融合为静态缩放（DRNet, CVPR 2026 重参数化范式）。

$$\mathbf{f}_{\text{noise}}^{\text{out}} = \mathbf{f}_{\text{noise}} \odot \begin{cases} \sigma(\text{SE}(\mathbf{f}_{\text{motion}})) & \text{train} \\ \bar{g}_n & \text{infer (deploy)} \end{cases}$$

$$\mathbf{f}_{\text{motion}}^{\text{out}} = \mathbf{f}_{\text{motion}} \odot \begin{cases} \sigma(\text{SE}(\mathbf{f}_{\text{noise}})) & \text{train} \\ \bar{g}_m & \text{infer (deploy)} \end{cases}$$

### 2.8 SGRF — 阶段式修复融合 (Flight7.2: Stage A/B + auxiliary losses)

SGRF 拆分为两阶段：**Stage A**（提亮）和 **Stage B**（残差精修）。Flight7 引入 `sg[img_lit]` 隔离梯度。Flight7.2 恢复 NDPN/MCPN 进入 S1/S2 的梯度通路，并通过辅助损失为 NDPN/MCPN 提供独立优化目标。

**Stage A — 提亮**（`igrf.py:162-174`）：
$$\text{S1: } I_1 = \text{soft\_clamp}(I_t + \delta_1(\mathbf{f}_{\text{noise}}^{\text{out}})), \quad \text{S2: } I_2 = \text{soft\_clamp}(I_1 + \delta_2(\mathbf{f}_{\text{motion}}^{\text{out}}))$$
$$\text{TCC: } I_{\text{curved}} = \text{TCC}_6(I_2, A, \alpha_{\text{target}}), \quad \mathbf{I}_{\text{lit}} = \text{soft\_clamp}(I_{\text{curved}} \odot \mathbf{G})$$

**Stage B — 残差精修**（`igrf.py:176-181`）：
$$\boldsymbol{\delta} = h_\theta(\mathbf{f}_{\text{noise}}^{\text{out}} + \mathbf{f}_{\text{motion}}^{\text{out}}) \cdot \tanh(\beta), \quad \hat{\mathbf{J}} = \text{sg}[\mathbf{I}_{\text{lit}}] + \boldsymbol{\delta}$$

**Flight7.2 辅助监督**：
$$\mathcal{L}_{\text{ndpn}} = 1 - \text{SSIM}(I_1, \text{GT}), \quad \mathcal{L}_{\text{mcpn}} = \|I_2 - \text{GT}\|_1$$
这两个损失在 Phase 1.5+ 激活，为 NDPN（通过 S1 delta）和 MCPN（通过 S2 delta）提供不依赖 Stage B 的独立梯度通路。

Flight4 用 soft_clamp 替代 hard clamp 消除了梯度截断。Flight5 将曲线升级为 TCC（自动收敛），并新增 Flight4 soft_clamp 保留——所有 StageBlock 和 BrightenStage 统一使用 Tanh 基 soft_clamp。

**S2.5 TCC 曲线**：
$$\text{TCC}_n = \text{TCC}_{n-1} + A_n \odot \text{TCC}_{n-1} \odot (1 - \text{TCC}_{n-1}) \odot (\alpha_{\text{target}} - \text{TCC}_{n-1})$$

**S3 提亮**：
$$\hat{X}_t = \text{soft\_clamp}\left(\text{TCC}(I_2) \odot G\right)$$

SGRF 的三阶段顺序由 §2.1 的物理推导确定。Mod5 对 StageBlock delta 施加零均值约束。Mod6 将曲线升级为 pixel-wise 8 iter，并移除 bias_map——ZeroDCE 曲线 $I+\alpha\cdot I\cdot(1-I)$ 提供有界加性修正，bias 无独立物理角色。

**Zero-gate StageBlock (Mod5: zero-mean)**：
$$\delta = \text{ConvBlock}([f_{\text{branch}}, \text{proj}(I_{\text{current}})]) \cdot \gamma_{\text{gate}}$$
$$\delta = \delta - \bar{\delta} \quad\text{(强制 per-channel 空间均值为零)}$$

**四阶段公式 (Denoise → Deblur → Curve → Brighten)**：

$$\text{S1 (去噪): } \quad I_1 = \text{clamp}\left(I_t + \delta_1(\mathbf{f}_{\text{noise}}^{\text{out}}) \cdot \gamma_1, 0, 1\right)$$

$$\text{S2 (去模糊): } \quad I_2 = \text{clamp}\left(I_1 + \delta_2(\mathbf{f}_{\text{motion}}^{\text{out}}) \cdot \gamma_2, 0, 1\right)$$

$$\text{S2.5 (曲线, Mod6): } \quad I_2^{\text{curve}} = \text{ZeroDCE}_{pixel-wise}(I_2, A(x,y)),\quad A\in\mathbb{R}^{B\times 8\times 3\times H\times W}$$

$$\text{S3 (提亮, Mod6): } \quad \hat{X}_t = \text{clamp}\left(I_2^{\text{curve}} \odot G, 0, 1\right)$$

---

## 3. 核心模块详解

### 3.1 WFR — 空域小波分流器 (Flight3)

**文件**: `models/modules/swd.py`

LL 低频分流 (保留) + HF 高频共享 (取消噪声门控)：

```
Encoder feat → [HaarDWT] → LL, LH, HL, HH
  ├─ alpha_net(LL) → α ∈ (0,1)
  ├─ LL_dpe = α × LL                (光照信号)
  ├─ LL_tca = IN((1-α) × LL)        (去光照, 保结构)
  ├─ HF_cat = cat(LH, HL, HH)        (完整高频, Flight3: 两路共享)
  └─ proj_dpe(LL_dpe + HF_cat) → feat_tfde    (Conv(4C→C)+GELU+LN)
  └─ proj_tca(LL_tca + HF_cat) → feat_tca     (独立权重, 隐式分化)
```

**Flight3 简化**: 已删除 `noise_gate` (Conv→Sigmoid) 和 `hf_tca_norm` (LayerNorm)——消除 "高能量=噪声" 的归纳偏置误判。

### 3.2 DPE — 退化先验估计器 (Flight8: 3-stage progressive scan)

**文件**: `models/modules/tfsi_v2.py`

**Flight8 3-stage coarse-to-fine扫描**:
```
l3_lat(B,C,H/4,W/4) from Encoder — 最粗尺度
  ├─ Stage1: refine(l3) + gray(I_t↓H/4) + lum(I_t↓H/4) → s_c3
  │
l2_lat(B,C,H/2,W/2)
  └─ Stage2: refine(l2) + upsample(s_c3) + gray(I_t↓H/2) + lum(I_t↓H/2) → s_c2
  │
l1_lat(B,C,H,W)
  └─ Stage3: refine(l1) + upsample(s_c2) + gray(I_t) + lum(I_t) + cls_token → head → s_illum, s_noise
```

**gray/lum physics priors**: `gray = I_t.mean(dim=1, keepdim=True)` (亮度), `lum = I_t.max(dim=1, keepdim=True).values` (最大值) — 提供原始光照信息，补偿 WFR 子带分流中可能丢失的全局光照上下文。

**cls_token**: 可学习参数 (1,1,C,1,1)，提供全局统计偏差，防止 DPE 在所有示例上都输出 s_illum≈0.5。初始化为零 → Phase 1 warmup 输出 0.5，后续学习。

**Mark4 问题**: Sigmoid 前无归一化 → F_fused 幅值失控 → s_illum=1.0 饱和。
**Flight3 修复**: LayerNorm + head 零初始化 → 初始输出 0.5，训练中可自由学习。

### 3.3 TCA — 时序对应对齐 (Flight9: H/2 WKV 直连, 无 internal FPN)

**文件**: `models/modules/pure_rwkv_sace.py`

**Flight8 架构升级**:
- **Internal FPN**: l3_lat→l2_lat→l1_lat 自底向上聚合，产生全分辨率 (H×W) WKV 输入
- **Full-Res WKV**: 从 Flight7.2 的 H/2(WFR) 升级到 H×H(256×256)，通过 batch_size 2 + grad_accum 8 补偿显存
- **WFR Residual**: feat_tca 注入路径：`sace_out += wfr_feat_tca × wfr_lambda` (λ 零初始化)
- **C_omega at H/2**: 时序对应矩阵仍在 WFR feat_tca 半分辨率计算 (H/2 对对齐足够)

**文件**: `models/modules/pure_rwkv_sace.py`

**架构总览**：

```
feat_tca (B,T,C,H/2,W/2) from WFR — 半分辨率, 节省 4× WKV 计算量
  │
  ├─ [MVC-Shift] 3支空洞DWConv(d=1,2,3) + 1×1混频 → x_shifted
  │     RWKV Token Shift → 2D 空间局部上下文
  │
  ├─ [SpatialWKV2D] 四方向 Bi-WKV (Spatial Mixing)
  │     ├─ pre_norm (LayerNorm)
  │     ├─ Split C→4 heads × C/4
  │     ├─ Head0/H1/H2/H3: 水平/垂直/主对角/副对角扫描
  │     ├─ 每方向独立 BiWKV (chunk-wise cumsum=256, e^w<1)
  │     ├─ Per-direction LayerNorm
  │     └─ σ(R)⊙wkv → proj_out → post_norm
  │
  ├─ [Channel Mix] LN → Conv(C→4C,1×1)→GELU→Conv(4C→C,1×1) + γ×residual
  │     └─ RWKV Channel Mixing 的 Conv/FFN 等效
  │
  ├─ [Upsample] H/2→H → tca_out (B,T,C,H,W)
  │
  ├─ [TemporalCorrespondence] proj_qk(C→C/4) → cosine_sim
  │     ÷ softplus(τ)+0.05 → softmax(dim=-1)
  │     → C_omega_list: (T-1)×(B,ds²,ds²), ds≤96
  │
  └─ [TemporalAggregation] C_omega warp 邻帧 → frame_gate 加权
        → upsample + residual + LN → F_t_aligned (B,C,H,W)
```

**RWKV 架构特点**：

| 特点 | 值 | 说明 |
|------|-----|------|
| Norm 策略 | pre_norm (Spatial Mix前) + post_norm (融合后) | RWKV-7 风格 |
| R/K/V 初始化 | $\pm0.05\text{--}0.5/\sqrt{C}$ 均匀分布 | 小初始化防溢出 |
| 输出投影 | 零初始化 | 初始残差=identity |
| 衰减保证 | $w = -\text{softplus}(\theta)$ → $e^w < 1$ | 数学强制 |
| 长序列 | chunk-wise cumsum (CHUNK=256) | 防数值溢出 |
| Channel Mix | 1×1 Conv(C→4C→C) + GELU | 等价 per-pixel FFN |
| Token Shift | MVC-Shift(3 dilated DWConv) | 2D 空间上下文 |
| 双向 | (fwd+bwd)/2 | Vision-RWKV Bi-WKV |
| 多方向 | 4 heads (H/V/对角) × C/4 | RSRWKV |

### 3.4 数值稳定性

| 组件 | 措施 | 文献 |
|------|------|------|
| BiWKV | `w = -F.softplus(spatial_decay)` → $e^w < 1$ 恒衰减 | [RWKV-4, Peng et al. 2023] |
| BiWKV | chunk-wise cumsum (CHUNK=256) | 同上 |
| BiWKV | `k.clamp(-8,8)`, `v.clamp(-8,8)` | 同上 |
| BiWKV | 双向 (fwd+bwd)/2 | [Vision-RWKV, Duan et al. 2024] |
| SpatialWKV2D | Per-head LayerNorm + pre_norm | [RSRWKV, He et al. 2025] |
| SpatialWKV2D | R/K/V RWKV-7 风格小初始化 $\pm0.05\text{--}0.5/\sqrt{C}$ | [RWKV-7, Peng et al. 2025] |
| SpatialWKV2D | $W_{\text{out}}$ 零初始化 → 初始残差=identity | 同上 |
| WFR | `proj_tfde/proj_tca` 后 LayerNorm → norm≈1 | — |
| Tau | `F.softplus(tau_raw) + 0.05` → 下界 0.05 | — |
| DPE head | LayerNorm2d + 零初始化 → s_illum≈0.5 (居中, 防饱和) | — |
| ISPN gain | sigmoid → 输出 $\in[0.5,2.0]$, 梯度不消失 | [Zero-DCE++, Li et al. 2022] |
| CurveBranch | Tanh → $A \in [-4,4]$, TCC 有界收敛 | [Zero-DiDCE, 2024] |
| TCC safety | soft_clamp(0.5+0.5·tanh(4·(le-0.5))) | — |

### 3.5 参考文献

| 文献 | 核心贡献 | 本模型使用位置 |
|------|---------|:------------:|
| **RWKV-4** [Peng et al., EMNLP 2023] "RWKV: Reinventing RNNs for the Transformer Era" | WKV 注意力 + Time/Channel Mix 分离 + $w=-\text{softplus}(\theta)$ 衰减保证 | TCA Spatial Mix + Channel Mix 架构范式 |
| **Vision-RWKV** [Duan et al., 2024] "Efficient Visual Perception with RWKV-Like Architectures" | Bi-WKV 双向空间扫描 + quad-directional spatial mix | TCA Bi-WKV (fwd+bwd)/2 |
| **RSRWKV** [He et al., TCSVT 2025] "2D-WKV for Vision" | 四方向并行空间扫描 + MVC-Shift | TCA SpatialWKV2D + MVC-Shift |
| **RWKV-7** [Peng et al., 2025] | 小初始化 ($\pm0.05/\sqrt{C}$) + pre_norm + 零初始化输出投影 | TCA R/K/V 投影初始化 |
| **Zero-DCE** [Guo et al., CVPR 2020] | LE-curve pixel-wise α + 迭代增强 | TCC 公式原型 (§2.4) |
| **Zero-DCE++** [Li et al., TPAMI 2022] | 下采样鲁棒性 + 参数共享 + 轻量化 | TCC 下采样(8×) + A_map 复用 (§2.4) |
| **Zero-DiDCE** [2024] | ALE-curve 收敛不动点 + 动态迭代 | TCC $(\alpha-LE)$ 收敛因子 (§2.4) |
| **Physen-Noise2Noise** [2025] | 多帧有偏噪声联合优化 | NDPN 多帧去噪原理 (§2.5) |
| **GR-VEF** [IJARCCE 2026] | 运动分类替代密集光流 | MCPN 运动补偿设计 (§2.6) |

---

## 4. 参数分布

| 模块 | 参数量 |
|------|--------|
| Encoder | ~320K |
| WFR | ~25K |
| DPE | ~50K |
| TCA (MVCShift + WKV + Corr + Agg) | ~300K |
| ISPN (gain_head + SpatialCurveBranch, Mod6) | ~75K || NDPN (含 conf_proj + noise_extract + denoise_strength) | ~85K |
| MCPN (含 motion_estimator + comp_gate + motion_refine) | ~80K |
| CXG | ~8K |
| SGRF | ~120K |
| 其他 | ~10K |
| **总计** | **~1.55M** |

---

## 5. 版本演进

| 版本 | 关键改动 | 参数量 | 状态 |
|------|---------|--------|------|
| v5.9.2 | s_illum 复生 + IFPN 监督 | 1.14M | 20.39 PSNR |
| v6.5 | PureRWKV 移除 DAT | 1.17M | 20.36 |
| v6 Delta | 空间扫描2D-WKV + C_omega + F_t_aligned | 1.64M | 训练崩溃 |
| v6 Delta Mark1 | WFR子带分流 + 命名统一 + 数值稳定 | 1.69M | ep15 loss收敛 |
| **v6 Delta Mark2** | **Kendall不确定性加权 + PE Loss + 感知解耦 + 损失调度** | **1.69M** | 收敛停滞 |
| **v6 Delta Mark3** | **两阶段渐进训练 + ISPN 隔离 + SGRF 中间监督 + 相位调度** | **1.69M** | PSNR 18→8.7 |
| **v6 Delta Flight7.2+WFR** | **Encoder→TCA主输入+WFR残差, DPE concat Enc center, aux监督, TCC 4×↓, Stage A/B** | **1.55M** | PSNR 17.10@ep80 |
| **v6 Delta Flight8** | Multi-scale encoder, DPE 3-stage+gray/lum, TCA internal FPN+full-res WKV, batch=2 accum=8 | 1.85M | 提前终止 (dpe_si=0.92/0.00) |
| **v6 Delta Flight9** | **取消WFR, DPE softplus单尺度H/4, TCA H/2, L_illum_spatial+tv, gamma clamp, batch=4** | **~1.5M** | **训练中** |

---

## 6. 损失函数设计 (Mark4)

**文件**: `losses/losses.py` — `TFSNetLoss`

### 6.1 Kendall 不确定性加权

$$L_{\text{total}} = \sum_{i} \frac{1}{2\exp(s_i)} L_i + \frac{1}{2}s_i$$

其中 $s_i = \log\sigma_i^2$ 为可学习 log-variance，损失大的任务自动降权，损失小的任务提权。

| Key | 初始 log_var | 物理含义 |
|-----|-------------|---------|
| `pix` | 0.0 | 像素重建 |
| `ssim` | -1.0 | 结构相似度 |
| `illum` | 0.0 | 光照图边缘平滑 |
| `ifpn` | 0.0 | TCA 对齐质量 (align_warp + diag_prior) |
| `perc` | 2.0 | VGG 感知 (Phase 1 屏蔽, Phase 2→-1.0) |
| `freq` | 0.0 | 频域纹理 |
| `inter` | 0.0 | 中间乘法路径监督 |

### 6.2 各损失项

#### 主路径损失 (所有阶段)

| 项 | 公式 | 监督对象 | 路由 |
|----|------|---------|------|
| **L_pix** | PECharbonnier(res_t, GT) | 最终输出 | SGRF-S3 |
| **L_ssim** | 1 - SSIM(res_t, GT) | 结构相似度 | 最终输出 |
| **L_gain_sup** | L1(gain_map, GT/Ī_s2) (Mod5: 曲线感知, target=GT/img_s2) | 光照图空间分布 | ISPN→gain_map |

#### TCA 对齐 & 正则 (Phase 1/2 共享)

| 项 | 公式 | 物理含义 |
|----|------|---------|
| **L_align_warp** | L1(∑C·F_neighbor, F_center) | 时序 warp 一致性：用 C_omega 对齐邻帧特征，应与中心帧一致 |
| **L_diag_prior** | −mean(log(diag(C_Ω)+ε)) | Phase 1 仅用于稳定 C_ω 学习；Phase 2 取消，释放运动检测 |
| **L_illum_smooth** | TV(s_illum)·e^{−‖∇I‖} | 光照图边缘感知平滑 |
| **L_wfr_reg** | (ᾱ − 0.7)² | WFR 分流平衡正则：允许 DPE 获取更多 LL，仅防极端 α→1.0 |
| **L_gamma_reg** | relu(0.005 − |γ_ndpn|) + relu(0.005 − |γ_mcpn|) | 防止γ坍缩到零, 保持三源分支活性 | 权重0.1 |

#### Phase 2 专属 — 感知解耦 & 中间监督

| 项 | 公式 | 监督对象 | 设计要点 |
|----|------|---------|---------|
| **L_ssim_s1** | 1 − SSIM(img_s1, GT) | SGRF-S1 (去噪) | 感知解耦：SSIM→S1 |
| **L_perc_s2** | ‖VGG(img_s2) − VGG(GT)‖ | SGRF-S2 (去模糊) | 感知解耦：VGG→S2 |
| **L_pix_s3** | Charbonnier(img_s2·lit_up_map, GT) | SGRF-S3 提亮路径 | 中间乘法监督 |
| **L_freq** | L1(|FFT(res)|, |FFT(GT)|) | 频域纹理 | 目标纹理细节 |

#### Kendall UW 聚合规则

| 阶段 | 聚合公式 |
|------|---------|
| Phase 1 Warmup | UW(pix) + UW(ssim) + UW(illum_smooth, 'illum') + UW(align+diag, 'ifpn') + 0.5·L_gain_sup + 0.001·L_wfr_reg |
| Phase 1 Main | 同上 |
| Phase 2 | ← + UW(perc, 'perc') + UW(freq, 'freq') + UW(inter, 'inter') + UW(align_warp only, 'ifpn') + 0.1·L_gamma_reg (Mod3) (diag_prior 取消) |

### 6.3 相位调度 (Mark4)

**核心理念**：先收敛无冲突的 ISPN 路径，再逐级解锁 NDPN/MCPN 分支，避免 Mark2 的梯度冲突导致收敛停滞。

| 阶段 | Epoch | 损失 | NDPN/MCPN | CXG | lr |
|------|-------|------|-----------|-----|-----|
| **Phase 1 Warmup** | 0–4 | pix + ssim + illum_smooth + gain_sup + align+diag + wfr_reg | **zero** | bypass | 8e-6→8e-4 |
| **Phase 1 Main** | 5–10 | 同上 | **zero** (γ=0.01流过) | bypass | 6e-4 |
| **Phase 1.5** | 11–30 | ← + L_ndpn_aux + L_mcpn_aux + L_gamma_reg | 线性解锁 0→100% | ratio>0.3启用 | 6e-4→4e-4 |
| **Phase 2** | 31–85 | 全损失 (感知解耦+freq+inter+L_gamma_reg+L_ndpn+L_mcpn; diag_prior取消) | 100% | 启用 | 4e-4→1.6e-5 |

**关键设计决策**：

1. **Mod3 γ=0.01**：Mark4 的 γ=0 使分支无梯度。Flight3 提升至 0.001，但 CXG 输出未被 SGRF 使用导致 gamma 坍缩。Mod3 提升至 0.01 并修复 CXG 路由——SGRF 现在接收 CXG 门控特征 (f_noise_gated/f_motion_gated)，gamma 直接影响输出，赋予 NDPN/MCPN 实际学习动力。

2. **Phase 1 缩短到 10 epoch (0→10), Phase 1.5 20 epoch (10→30), Phase 2 60 epoch (30→90)**：Mark4 的 20 epoch ISPN 独占使特征空间固化于纯光照。V9 实验证实 s_illum=100% 主导退化，Phase 1 仅需 10 epoch。提前解锁使三源分支共享基础特征。

3. **感知解耦** (Perceptual Decoupling)：Mark2 设计 B1 方案——SSIM 损失监督 SGRF-S1（去噪），VGG 感知损失监督 SGRF-S2（去模糊），Charbonnier 监督 S3 乘法路径。消除"同一输出接收 SSIM+感知 冲突梯度"问题。

4. **L_diag_prior Phase 2 取消**：该损失鼓励 C_ω 对角线恒等，Phase 2 中与 MCPN 运动检测冲突。Phase 1 保留用于稳定 C_ω 初始学习，Phase 2 完全移除以释放运动检测能力。

5. **L_gain_sup 直接监督**：权重 0.5（固定，不受 Kendall UW），将 gain_map 与 GT/I̅ 逐像素 L1 对比。解决 ISPN gain_head 训练路径过长（pix loss → res_t → gain_map 的间接梯度）导致的 gain 学习不足。L_wfr_reg 弱正则（0.001）防止 α_net 极端分流失衡。

6. **Gamma anti-collapse 正则化 (Mod3)**：L_gamma_reg = relu(0.005 - |γ_ndpn|) + relu(0.005 - |γ_mcpn|)，权重 0.1。当 gamma 均值降至初始值一半以下时产生惩罚，防止三源退化为单源。

7. **ZeroDCE 曲线增强 (Mod4→Mod6 渐进升级)**：ISPN 曲线分支从 per-image MLP(64→16→9, 3 iter) → MLP(64→16→15, 5 iter) → SpatialCurveBranch (Conv(64→32) + Conv(32→24), 8 iter pixel-wise)。Mod6 最终形式：从 refine 特征 h(64ch) 用 2 层 Conv 预测 per-pixel α ∈ ℝ^{B×8×3×H×W}，在 SGRF S2→S3 之间施加 ZeroDCE 曲线。零初始化 → Phase 1 为 identity。pixel-wise α 可逐像素自适应提亮强度（暗区更强、亮区保持），不再需要 bias_map 的空间补偿。

11. **WFR 路由重构 (Flight7.2+WFR)**：WFR 从"主路径滤器"降级为"残差增强信号"。TCA 主输入改为 Encoder 原始特征（完整信息用于帧间对齐 + DWT 下采样到 H/2），WFR feat_tca 通过可学习的 `wfr_lambda`（初始=0）注入残差 `sace_out += wfr_feat_tca × wfr_lambda`。DPE 同时接收 WFR feat_tfde（时域统计）和 Encoder 中心帧特征（提供原始纹理上下文，concat 到统计量后进入 ms_branch）。设计原则：光源分离（WFR）不再以牺牲信息量为代价——Decoder 级模块各自有权访问完整特征，WFR 提供增量而非替代。

---

## 7. 训练策略 (Mark4)

### 7.1 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 4 (accum=4) | eff batch=16 |
| grad_accum_steps | **4** | 等效 batch=16 |
| epochs | 80 | P1W(0-4)+P1(5-10)+P1.5(11-25)+P2(26-80) |
| lr (Phase 1 Warmup) | 8e-6 → 8e-4 | 线性 warmup |
| lr (Phase 1 Main) | 6e-4 | 固定 |
| lr (Phase 1.5) | 6e-4 → 4.4e-4 | 线性过渡 (15 epoch) |
| lr (Phase 2) | 4e-4 → 1.6e-5 | 阶梯衰减 (55 epoch) |
| optimizer | AdamW | — |
| weight_decay | 1e-4 | L2 正则 |
| grad_clip | 0.5 | 梯度裁剪 |
| L_illum_spatial | 0.1 | 反零方差 (DPE 防饱和) |
| L_illum_tv | 0.05 | 边缘感知 TV |
| gamma clamp (NDPN) | max=0.03 | 防暴走 |

### 7.2 相位判定函数 (epochs=80)

```python
def get_phase(epoch):
    if epoch < 5:   return 'phase1_warmup'
    elif epoch < 11: return 'phase1'
    elif epoch < 26: return 'phase1_5'
    else:            return 'phase2'

def get_unlock_ratio(epoch):
    if epoch < 11: return 0.0
    if epoch >= 26: return 1.0
    return (epoch - 11) / 15.0

def get_lr(epoch, base_lr=8e-4):
    if epoch < 5:   return base_lr * (0.01 + 0.99 * epoch / 5)
    elif epoch < 11: return 0.75 * base_lr
    elif epoch < 26: return base_lr * 0.75 * (1 - (epoch - 11) / 15 * 0.33)
    elif epoch < 51: return base_lr * 0.5
    elif epoch < 65: return base_lr * 0.125
    elif epoch < 73: return base_lr * 0.05
    else:            return base_lr * 0.02
```

### 7.3 Phase-Dependent Forward

**文件**: `models/tfs_net.py`

```python
if phase in ('phase1', 'phase1_warmup'):
    f_noise_out = torch.zeros_like(F_t_aligned)      # NDPN 截断
    f_motion_out = torch.zeros_like(F_t_aligned)      # MCPN 截断
elif phase == 'phase1_5':
    ratio = get_unlock_ratio(epoch)
    f_noise_out = ndpn(...) * ratio                   # 线性解锁
    f_motion_out = mcpn(...) * ratio
elif phase == 'phase2':
    f_noise_out = ndpn(...)                           # 完整版本
    f_motion_out = mcpn(...)
    f_noise_gated, f_motion_gated = cxg(...)
```

### 7.4 数据与评估

| 项目 | 配置 |
|------|------|
| 训练集 | SDSD indoor (2064 samples), crop=256×256, T=5 |
| 验证集 | SDSD test **前 5 段** (~620 frames) |
| 评估频次 | 每 **10 epoch** |
| 评估耗时 | ~40 min/eval |
| 推理 | `infer.py`, tile_size=256, no AMP |

### 7.5 训练监控

```bash
tail -f outputs/sdsd_f9/train.log
grep "diag:" outputs/sdsd_f9/train.log | tail -10
grep "Val stats" outputs/sdsd_f9/train.log | tail -10
```

### Flight9 关键设计决策

| 决策 | 动机 | 文献支持 |
|------|------|---------|
| DPE sigmoid→softplus | 光照是线性/连续的(sigmoid S形梯度→极值吸引子) | IllumFlow (2025): illumination≈linear parametric |
| 取消 3-stage cascade | 级联放大饱和(每级 sigmoid 推高前级输出) | NID-LLIE (2026): stabilize intermediate representations |
| 取消 WFR | 多尺度 Encoder 已是天然频域分离器(l3=低频,l1=高频) | EvLIR (2025): 无小波分流, 达 25.63dB SDSD |
| TCA 回退 H/2 | full-res 256 需 batch=2→样本多样性↓50% | EvLIR: 局部时序建模足够, 全局依赖过度设计 |
| L_illum_spatial | -log(std) 无限惩罚零方差 | QRetinex-Net (2025): freq-aware regularization |
| Gamma clamp 0.03 | DPE 饱和→NDPN gamma 暴走 5× 补偿 | Dynamic Nonlinear Net (2026): explicit noise-illum coupling |

---

## 8. 关键文件

| 文件 | 模块 |
|------|------|
| `models/modules/tfsi_v2.py` | DPE (softplus IllumHead, single-scale H/4) |
| `models/modules/pure_rwkv_sace.py` | TCA (H/2 WKV, l2_lat 直连) |
| `models/modules/ispn_v2.py` | ISPN (Mod6: SpatialCurveBranch + gain_head, bias removed) |
| `models/modules/ifpn.py` | ISPN (legacy, Mark3) |
| `models/modules/ndpn.py` | NDPN |
| `models/modules/mrpn.py` | MCPN |
| `models/modules/igrf.py` | SGRF (zero-gate StageBlock, BrightenStage) |
| `models/tfs_net.py` | CXG, TFSNet (数据流编排) |
| `losses/losses.py` | TFSNetLoss (Kendall UW + gain_sup + Phase Schedule) |
| `train.py` | 训练循环 (grad accum + phase lr + metric logging + frame_cache管理) |
| `configs/delta_flight9.yaml` | 训练配置 (batch=4, accum=4, epochs=80) |
