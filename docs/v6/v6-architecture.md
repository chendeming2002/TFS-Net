# TFS-Net v6 Delta Mark4 模型架构设计文档

> 日期：2026-07-06 (更新: Mark4 三部保险 + TFDE/ISPN 简化)
> 版本：v6 Delta Mark4
> 训练配置：`configs/v6_bravo.yaml`，batch=4, lr=8e-4, epochs=50, warmup=5
> 参数量：1.45M

---

## 1. 概述

TSD-Net (Tri-Source Decoupled Network) 是一个端到端多帧低光视频增强网络，基于 **TSDR (Tri-Source Decoupled Restoration)** 框架。核心思想是**时域差分引导的三源解耦 → 并行重建 → 阶段式融合**。

| 模块 | 缩写 | 功能 |
|------|------|------|
| **WSD** | Wavelet-based Source Diverter | Haar DWT 子带级分流 (LL→光照/噪声, HF→结构) |
| **DIE** | Degradation Intensity Estimator | 时域统计 + 多尺度空洞卷积 → s_illum, s_noise (Mark4 简化) |
| **TCA** | Temporal Correspondence & Alignment | 4方向空间 WKV 扫描 + C_omega 时序矩阵 |
| **ISPN** | Illumination-Source Processing Network | Retinex gain/bias 头 → gain_map, bias_map (Mark4 简化) |
| **NDPN** | Noise Degradation Processing Network | C_omega 置信度引导去噪 (γ=0, noise_proj=0 → pass-through init) |
| **MCPN** | Motion Compensation Processing Network | C_omega 运动强度补偿 (Mark4: gamma=0, refine=0, startup_gate=1 → pass-through init) |
| **CXG** | Cross-eXcitation Gate | 去噪↔运动 交叉激励门 (deploy 重参数化) |
| **SGRF** | Stage-wise Guided Restoration & Fusion | S1:去噪 → S2:去模糊 → S3:gain×img+bias (Mark4: zero-gate StageBlock) |

### 命名变更总表

| 原缩写 | 新缩写 | 中文全名 |
|--------|--------|----------|
| DWT-LFF | **WSD** | 基于小波的源分流器 |
| TFSI | **DIE** | 退化强度估计器 |
| SACE | **TCA** | 时序对应对齐 |
| IFPN | **ISPN** | 光照源处理网络 |
| MRPN | **MCPN** | 运动补偿处理网络 |
| IGRF | **SGRF** | 阶段式引导修复融合 |
| CrossFusionGate | **CXG** | 交叉激励门 |

### 完整数据流 (Mark4)

```
输入: I_{t-2}, I_{t-1}, I_t, I_{t+1}, I_{t+2}  (T=5 窗口)
  │
  ├─→ Encoder → F_{t-2}...F_{t+2}  (B, T, 64, H, W)
  │
  ├─→ WSD (逐帧 HaarDWT) → F_tfde (B,T,C,H/2), F_tca (B,T,C,H/2)
  │
  ├─→ DIE(F_tfde) → s_illum, s_noise
  │     s_illum → ISPN   s_noise → NDPN
  │
  ├─→ TCA(F_tca) → {F_{t'}^{\text{out}}}_{t'=0}^{T-1}, C_{t,Ω}, \hat{F}_t, μ, σ
  │
  ├─→ ISPN(f_enc, s_illum) → gain_map, bias_map
  ├─→ NDPN({F^{\text{out}}}, s_noise, μ, σ, C_{t,Ω}, \hat{F}_t) → f_noise_out
  ├─→ MCPN({F^{\text{out}}}, σ, C_{t,Ω}, \hat{F}_t) → f_motion_out
  │
  ├─→ CXG(f_noise_out, f_motion_out) → f_noise_gated, f_motion_gated
  │
  └─→ SGRF(gain, bias, f_noise_gated, f_motion_gated, I_t)
        S1: I_1 = I_t + StageBlock_1(f_noise_gated, gate=0)
        S2: I_2 = I_1 + StageBlock_2(f_motion_gated, gate=0)
        S3: \hat{X}_t = clamp(I_2 × gain + bias, 0, 1)
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

**帧内空间扫描 (Bi-WKV)**。对每帧特征 $F_t$ 沿空间序列 $L=H\times W$ 做带衰减的递归：

$$S_\tau = e^{w} \cdot S_{\tau-1} + e^{k_\tau} \odot v_\tau, \quad D_\tau = e^{w} \cdot D_{\tau-1} + e^{k_\tau}$$

$$y_\tau^{(t)} = \frac{e^{u}\cdot e^{k_\tau}\odot v_\tau + S_\tau}{e^{u}\cdot e^{k_\tau} + D_\tau}, \quad \tau = 0,\dots,L-1$$

其中 $k_\tau = \text{proj}_k(\tilde{F}_t[\tau])$，$v_\tau = \text{proj}_v(\tilde{F}_t[\tau])$，$\tilde{F}_t = \text{MVC-Shift}(F_t)$（3 支空洞 DWConv, $d=1,2,3$）。$w = -\text{softplus}(\theta_w) < 0$ 保证 $e^w < 1$ 恒衰减，$u$ 为当前 token bonus。

**四方向扫描** (RSRWKV, TCSVT 2025)。通道均分为 4 组 $C/4$，每组沿一个方向展开 $(H\times W)\to L$ 序列：

| 方向 | 排列方式 |
|------|---------|
| 水平 $h$ | 行优先 $(0,W),(0,W+1),\dots$ |
| 垂直 $v$ | 列优先 $(0,W),(1,W),\dots$ |
| 主对角 $d$ | 按 $i+j$ 排序 |
| 副对角 $a$ | 按 $i-j+W-1$ 排序 |

多方向融合：

$$\mathbf{F}_t^{\text{out}} = \mathbf{F}_t + \gamma \cdot \text{ChannelMix}\left(\sigma(R) \odot [y^{(t)}_h; y^{(t)}_v; y^{(t)}_d; y^{(t)}_a]\right)$$

**时序对应矩阵 $C_{t,\Omega}$**。中心帧与邻帧的空间 cosine similarity：

$$\mathbf{q}_t = \text{proj}_q(F_t^{\text{out}}), \quad \mathbf{k}_{t'} = \text{proj}_k(F_{t'}^{\text{out}}), \quad t' \in \Omega$$

$$C_{t,t'} = \text{softmax}\left(\frac{\mathbf{q}_t \cdot \mathbf{k}_{t'}^T}{\tau}\right), \quad \tau = \text{softplus}(\tau_{\text{raw}}) + 0.05$$

$C_{t,t'}[i,j]$ 表示中心帧位置 $i$ 与邻帧位置 $j$ 的对齐置信度。对角线 $C_{t,t'}[i,i]$ 为静止对应（高值→静止，低值→运动）。

**时序聚合 $\hat{F}_t$**。用 $C_{t,\Omega}$ 将邻帧 warp 到中心帧坐标系，加权聚合：

$$\hat{F}_t = \text{LN}\left(F_t^{\text{out}} + \sum_{t' \in \Omega} w_{t'} \cdot \left(C_{t,t'} \times F_{t'}^{\text{out}}\right)\right)$$

其中 $w_{t'} = \text{frame\_gate}([F_t^{\text{out}}, C_{t,t'} \times F_{t'}^{\text{out}}])$ 为数据驱动的帧级可靠性权重。

### 2.3 WSD — 空域小波分流

**动机**。旧 DWT-LFF 通过 IDWT 重建全分辨率特征，两条分支接收几乎相同的信息（仅 LL 略有差异），TFDE 的 IntensityHead 被全 HF 噪声压制（输入 norm 达 60~113），退化分离失效。

**方法**。在 Haar DWT 子带级别直接分流，不做 IDWT 重建：

$$F \xrightarrow{\text{HaarDWT}} \{LL, LH, HL, HH\}, \quad LL \in \mathbb{R}^{B\times C\times H/2\times W/2}$$

**低频分流**。$LL$ 编码全局光照强度，由可学习 $\alpha$ 分配：

$$\alpha = \sigma\left(\text{DWConv}_{3\times3} \to \text{GELU} \to \text{Conv}_{1\times1}\right)(LL)$$

$$LL_{\text{tfde}} = \alpha \odot LL, \quad LL_{\text{tca}} = \text{IN}\left((1-\alpha) \odot LL\right)$$

其中 $\text{IN}$ 为 InstanceNorm——去除通道级光照偏置，保留空间结构。

**高频分流**。$LH,HL,HH$ 编码纹理和噪声，由能量门控 $g_n$ 区分：

$$E_{\text{HF}} = \frac{1}{C}\sum_{c=1}^{C}\left(LH_c^2 + HL_c^2 + HH_c^2\right), \quad g_n = \sigma\left(\text{Conv}_{3\times3} \to \text{GELU} \to \text{Conv}_{1\times1}\right)(E_{\text{HF}})$$

$$HF_{\text{tfde}} = g_n \odot [LH, HL, HH], \quad HF_{\text{tca}} = \text{LN}\left((1-g_n) \odot [LH, HL, HH]\right)$$

高 HF 能量 → $g_n \uparrow$ → 噪声概率高 → 导向 TFDE。低 HF 能量 → $(1-g_n) \uparrow$ → 结构概率高 → 导向 TCA。

**输出投影**：

$$F_{\text{tfde}} = \text{LN}\left(\text{Conv}_{1\times1}([LL_{\text{tfde}}, HF_{\text{tfde}}])\right) \in \mathbb{R}^{B\times C\times H/2\times W/2}$$
$$F_{\text{tca}} = \text{LN}\left(\text{Conv}_{1\times1}([LL_{\text{tca}}, HF_{\text{tca}}])\right) \in \mathbb{R}^{B\times C\times H/2\times W/2}$$

### 2.4 ISPN — 光照源处理 (Mark4 简化)

ISPN 接收编码器中心帧特征 $F_t^{\text{enc}}$ 和 TFDE 的光照先验 $s_{\text{illum}}$，输出 Retinex 风格的增益-偏置对。移除 Mark3 的多帧 cosine attention 和 L_ratio 锚定逻辑。

**Retinex 物理模型**：
$$\hat{X}_t = I_t \cdot G + B$$

其中：
- $G$ (gain_map): 乘法提亮，≈ $1/L_{\text{estimated}}$, 补偿光照不足
- $B$ (bias_map): 加法修正，补偿暗电流、色偏等系统偏差

**网络结构**：
$$h = \text{Conv}_{3\times3} \to \text{GELU} \to \text{Conv}_{3\times3}([F_t^{\text{enc}}, s_{\text{illum}}])$$

$$\log G = \text{Conv}_{1\times1} \to \text{GELU} \to \text{Conv}_{1\times1}(h), \quad G = \exp(\log G) \cdot \text{clamp}(1, G_{\text{max}})$$

$$B = \tanh\left(\text{Conv}_{1\times1} \to \text{GELU} \to \text{Conv}_{1\times1}(h)\right) \cdot B_{\text{range}}$$

**零初始化策略**：
- `gain_head[-1]` (weight, bias) = 0 → $\log G = 0$ → $G = \exp(0) = 1$ (恒等)
- `bias_head[-1]` (weight, bias) = 0 → $B = 0$ (无偏置)
- Phase 1 初始行为：增强图像 = 原图 (不改变亮度)，渐进学习提亮

### 2.5 NDPN — 噪声退化处理 (对应 $\sigma^2_{\text{img}}$)

**直觉**：$C_{t,\Omega}$ 对角线 $C_{t,t'}[i,i]$ 表示位置 $i$ 在帧 $t$ 和 $t'$ 间的对应一致性。高对应 → 时序去噪可靠 → 弱空域去噪（保留细节）；低对应 → 运动遮挡 → 强空域去噪。

**置信度引导去噪**：

$$c = \text{conf\_proj}\left(\text{diag}(C_{t,\Omega}^{(1)}),\dots,\text{diag}(C_{t,\Omega}^{(T-1)})\right)$$

$$\mathbf{n} = \text{noise\_extract}\left([F_t^{\text{enc}}, \hat{F}_t]\right) \quad\text{(encoder vs aligned 差异)}$$

$$\text{strength} = \sigma\left(\text{denoise\_strength}([\mathbf{n}, c])\right), \quad \gamma_n = 0$$

$$\mathbf{f}_{\text{noise}} = F_t^{\text{enc}} - \gamma_n \cdot \mathbf{n} \odot \text{strength} + \text{noise\_proj}(s_{\text{noise}})$$

### 2.6 MCPN — 运动补偿处理 (对应 $\sigma^2_{\text{dyn}}$)

**直觉**：$C_{t,\Omega}$ 对角线偏离度是运动强度的逆向指标——对角线占优 → 静止 → 少补偿；对角线扩散 → 运动 → 强补偿。

**运动强度补偿**：

$$m = \text{motion\_estimator}\left(\text{diag}(C_{t,\Omega}^{(1)}),\dots,\text{diag}(C_{t,\Omega}^{(T-1)})\right)$$

$$F_{\text{omega}} = \text{window\_corr}\left(\hat{F}_t, \{F_{t'}^{\text{out}}\}_{t' \in \Omega}\right)$$

$$\Delta_m = \text{motion\_refine}([\hat{F}_t, F_{\text{omega}}]), \quad \text{comp} = \sigma(\text{comp\_gate}([\hat{F}_t, m]))$$

$$\mathbf{f}_{\text{motion}} = g_t \odot \hat{F}_t + (1-g_t) \odot F_{\text{omega}} + \gamma_m \cdot \Delta_m \odot \text{comp}$$

其中 $\gamma_m = 0.1$。$g_t = \sigma(\text{gate}([\hat{F}_t, F_{\text{omega}}]))$ 控制信任对齐中心 vs 聚合邻帧。

### 2.7 CXG — 交叉激励门

**动机**。去噪和运动补偿在重叠区域（既有噪声又有运动）梯度方向可能冲突。CXG 在训练时做动态交叉调制，推理时融合为静态缩放（DRNet, CVPR 2026 重参数化范式）。

$$\mathbf{f}_{\text{noise}}^{\text{out}} = \mathbf{f}_{\text{noise}} \odot \begin{cases} \sigma(\text{SE}(\mathbf{f}_{\text{motion}})) & \text{train} \\ \bar{g}_n & \text{infer (deploy)} \end{cases}$$

$$\mathbf{f}_{\text{motion}}^{\text{out}} = \mathbf{f}_{\text{motion}} \odot \begin{cases} \sigma(\text{SE}(\mathbf{f}_{\text{noise}})) & \text{train} \\ \bar{g}_m & \text{infer (deploy)} \end{cases}$$

### 2.8 SGRF — 阶段式修复融合 (Mark4)

SGRF 的三阶段顺序由 §2.1 的物理推导确定。Mark4 将 Stage3 替换为 gain/bias 公式，StageBlock 增加 zero-gate 以保证 Phase 2 无扰动启动。

**Zero-gate StageBlock**：
$$\delta = \text{ConvBlock}([f_{\text{branch}}, \text{proj}(I_{\text{current}})]) \cdot \gamma_{\text{gate}}, \quad \gamma_{\text{gate}} \xrightarrow{\text{init}} 0$$

gate=0 → $\delta = 0$ → 主路径不受扰动（Phase 1 安全）。gate 可学习上升，等价于 LoRA B=0 初始化策略。

**三阶段公式 (Denoise → Deblur → Brighten)**：

$$\text{S1 (去噪): } \quad I_1 = \text{clamp}\left(I_t + \delta_1(\mathbf{f}_{\text{noise}}^{\text{out}}) \cdot \gamma_1, 0, 1\right)$$

$$\text{S2 (去模糊): } \quad I_2 = \text{clamp}\left(I_1 + \delta_2(\mathbf{f}_{\text{motion}}^{\text{out}}) \cdot \gamma_2, 0, 1\right)$$

$$\text{S3 (提亮): } \quad \hat{X}_t = \text{clamp}\left(I_2 \odot G + B + \delta_3(I_2G+B) \cdot \gamma_3, 0, 1\right)$$

其中 $\gamma_1 = \gamma_2 = \gamma_3 = 0$ (初始) → Phase 1 完全等效于纯 ISPN 提亮。$\gamma_i$ 在 Phase 1.5/2 渐进释放。

---

## 3. 核心模块详解

### 3.1 WSD — 空域小波分流器

**文件**: `models/modules/swd.py (legacy)`

子带级分流（不做 inverse DWT），显式分离"光照+噪声"和"光照无关结构"。

```
Encoder feat → [HaarDWT] → LL, LH, HL, HH
  ├─ alpha_net(LL) → α ∈ (0,1)
  ├─ LL_tfde = α × LL          (光照估计信号)
  ├─ LL_tca = IN((1-α) × LL)   (去光照, 保结构)
  ├─ noise_gate(HF_energy) → n_gate ∈ (0,1)
  ├─ HF_tfde = n_gate × HF_cat  (噪声相关高频)
  ├─ HF_tca = LN((1-n_gate) × HF_cat)  (结构高频, 归一化)
  └─ proj(4C→C)+LN → feat_tfde, feat_tca
```

### 3.2 DIE — 时域退化估计器 (Mark4)

**文件**: `models/modules/tfsi_v2.py`

Mark4 移除 FrequencyBranch/LFF/phase_conf，用多尺度空洞卷积替代频域分析。

```
feat_tfde (B,T,C,H/2,W/2) from SWD
  │
  ├─ GroupNorm (逐帧)
  ├─ 时域统计: soft-median(μ), var(σ), μ/σ(SNR) 沿 T 维
  ├─ Concat [μ, σ, SNR] → (B, 3C, H/2, W/2)
  │
  ├─ MultiScaleSpatialBranch
  │     ├─ 3×3 (d=1): 局部纹理 → mid ch
  │     ├─ 3×3 (d=2): 中尺度光照 → mid ch
  │     └─ 3×3 (d=4): 大尺度区域 → wide ch
  │
  └─ Conv(→2ch) → Sigmoid → s_illum[:,0:1], s_noise[:,1:2]
```

**简化收益**: 移除 LFFFeatureAdapter/SpatialDWTLFFAdapter/phase_conf_head 等模块；梯度路径缩短；参数量减少。

**文件**: `models/modules/swd.py (legacy)`

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
| WSD | `proj_tfde/proj_tca` 后 LayerNorm → norm≈1 |
| Tau | `F.softplus(tau_raw) + 0.05` → 下界 0.05 |

---

## 4. 参数分布

| 模块 | 参数量 |
|------|--------|
| Encoder | ~320K |
| WSD | ~25K |
| DIE | ~50K |
| TCA (MVCShift + WKV + Corr + Agg) | ~300K |
| ISPN | ~50K |
| NDPN (含 conf_proj + noise_extract + denoise_strength) | ~85K |
| MCPN (含 motion_estimator + comp_gate + motion_refine) | ~80K |
| CXG | ~8K |
| SGRF | ~120K |
| 其他 | ~10K |
| **总计** | **~1.45M** |

---

## 5. 版本演进

| 版本 | 关键改动 | 参数量 | 状态 |
|------|---------|--------|------|
| v5.9.2 | s_illum 复生 + IFPN 监督 | 1.14M | 20.39 PSNR |
| v6.5 | PureRWKV 移除 DAT | 1.17M | 20.36 |
| v6 Delta | 空间扫描2D-WKV + C_omega + F_t_aligned | 1.64M | 训练崩溃 |
| v6 Delta Mark1 | SWD子带分流 + 命名统一 + 数值稳定 | 1.69M | ep15 loss收敛 |
| **v6 Delta Mark2** | **Kendall不确定性加权 + PE Loss + 感知解耦 + 损失调度** | **1.69M** | 收敛停滞 |
| **v6 Delta Mark3** | **两阶段渐进训练 + ISPN 隔离 + SGRF 中间监督 + 相位调度** | **1.69M** | PSNR 18→8.7 |
| **v6 Delta Mark4** | **三部保险 (zero-gate) + TFDE/ISPN 简化 + 延长的 Phase 1.5** | **1.45M** | **训练中(ep1)** |

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
| `illum` | 0.0 | 光照平滑 + 光照图监督 |
| `ifpn` | 0.0 | TCA 对齐质量 (align_warp + diag_prior) |
| `perc` | 2.0 | VGG 感知 (Phase 1 屏蔽, Phase 2→-1.0) |
| `freq` | 0.0 | 频域纹理 |
| `inter` | 0.0 | 中间路径监督 |

### 6.2 各损失项

#### 主路径损失 (所有阶段)

| 项 | 公式 | 监督对象 | 路由 |
|----|------|---------|------|
| **L_pix** | PECharbonnier(res_t, GT) | 最终输出 | SGRF-S3 |
| **L_ssim** | 1 - SSIM(res_t, GT) | 结构相似度 | 最终输出 |
| **L_illum_sup** | L1(lit_up_map, GT/I̅_t) | 光照图空间分布 | ISPN→lit_up_map |

#### Phase 1 专属 — TCA 对齐 & ISPN 约束

| 项 | 公式 | 物理含义 |
|----|------|---------|
| **L_align_warp** | L1(∑C·F_neighbor, F_center) | 时序 warp 一致性：用 C_omega 对齐邻帧特征，应与中心帧一致 |
| **L_diag_prior** | −mean(log(diag(C_Ω)+ε)) | C_Ω 对角先验：鼓励静止区域对应 identity（无 GT 自监督） |
| **L_illum_smooth** | TV(s_illum)·e^{−‖∇I‖} | 光照图边缘感知平滑 |
| **L_swd_reg** | (ᾱ − 0.5)² | WSD 分流平衡正则：防 α 偏离中心 |

#### Phase 2 专属 — 感知解耦 & 中间监督

| 项 | 公式 | 监督对象 | 设计要点 |
|----|------|---------|---------|
| **L_ssim_s1** | 1 − SSIM(img_s1, GT) | SGRF-S1 (去噪) | 感知解耦：SSIM→S1 |
| **L_perc_s2** | ‖VGG(img_s2) − VGG(GT)‖ | SGRF-S2 (去模糊) | 感知解耦：VGG→S2 |
| **L_pix_s3** | Charbonnier(img_s2·lit_up_map, GT) | SGRF-S3 提亮路径 | 中间乘法监督 |
| **L_ifpn_side** | Charbonnier(ifpn_side, GT↓) | ISPN 侧输出 | 暗图监督 |
| **L_freq** | L1(|FFT(res)|, |FFT(GT)|) | 频域纹理 | 目标纹理细节 |

#### Kendall UW 聚合规则

| 阶段 | 聚合公式 |
|------|---------|
| Phase 1 Warmup | UW(pix) + UW(ssim) + UW(illum_smooth + 0.02·L_illum_sup) + 0.001·L_swd_reg |
| Phase 1 Main | Phase1_Warmup + UW(align_warp + diag_prior, 'ifpn') |
| Phase 2 | 完整损失 (perceptual_decoupling: SSIM→S1, VGG→S2, Charb→S3) |

### 6.3 相位调度 (Mark4)

**核心理念**：先收敛无冲突的 ISPN 路径，再逐级解锁 NDPN/MCPN 分支，避免 Mark2 的梯度冲突导致收敛停滞。

| 阶段 | Epoch | 损失 | NDPN/MCPN | CXG | lr |
|------|-------|------|-----------|-----|-----|
| **Phase 1 Warmup** | 0–4 | pix + ssim + illum_smooth + lit_up_sup + swd_reg | **zero** (输出置零) | bypass | 8e-6→8e-4 |
| **Phase 1 Main** | 5–19 | + align_warp + diag_prior | **zero** | bypass | 6e-4 |
| **Phase 1.5** | 20–29 | = Phase 1 Main | 线性解锁 0→100% | ratio>0.3 时启用 | 6e-4→4e-4 |
| **Phase 2** | 30–49 | 全损失 (感知解耦 + freq + 中间监督) | 100% | 启用 | 4e-4→1e-4 |

**关键设计决策**：

1. **Phase 1 NDPN/MCPN/SGRF 截断 (Mark4 三部保险)**：Mark3 仅截断 NDPN/MCPN，但 SGRF StageBlock 在 Phase 1 被训练为处理零输入。Phase 2 激活时，StageBlock 收到非零特征后产生任意输出 → PSNR 18→8.7。Mark4 增加三重零保证：NDPN (gamma=0, noise_proj=0) + MCPN (gamma=0, refine=0, startup_gate=1, out_scale=0) + SGRF StageBlock (gate=0)。任一层独立保底。

2. **Phase 1.5 延长到 10 epoch (20→30)**：Mark3 的 5 epoch 斜坡不足以让 SGRF StageBlock 适应非零输入。延长到 10 epoch，配合 zero-gate 渐进释放，确保 Phase 2 启动平稳。

3. **感知解耦** (Perceptual Decoupling)：Mark2 设计 B1 方案——SSIM 损失监督 SGRF-S1（去噪），VGG 感知损失监督 SGRF-S2（去模糊），Charbonnier 监督 S3 乘法路径。消除"同一输出接收 SSIM+感知 冲突梯度"问题。

4. **L_diag_prior 自监督**：dataloader 只提供中心帧 GT，无法获取邻帧 GT。使用 C_Ω 对角线的 −log(·) 先验，无条件鼓励 identity 对应。Phase 1 无运动分支，副作用可控。Phase 2 可扩展为邻帧 GT 对应监督。

5. **L_swd_reg 弱正则**：固定权重 0.001（不受 Kendall UW 控制），防止 α_net 极端分流失衡（α→0 或 α→1）。

---

## 7. 训练策略 (Mark4)

### 7.1 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 4 | 物理 batch |
| grad_accum_steps | **1** | 等效 batch=4 |
| epochs | 50 | — |
| lr (Phase 1 Warmup) | 8e-6 → 8e-4 | 线性 warmup |
| lr (Phase 1 Main) | 6e-4 | 固定 |
| lr (Phase 1.5) | 6e-4 → 4e-4 | 线性过渡 |
| lr (Phase 2) | 4e-4 → 1e-4 | Cosine annealing |
| optimizer | AdamW | — |
| weight_decay | 1e-4 | L2 正则 |
| grad_clip | 0.5 | 梯度裁剪 |

### 7.2 相位判定函数

**文件**: `train.py`

```python
def get_phase(epoch):
    if epoch < 5:   return 'phase1_warmup'
    elif epoch < 20: return 'phase1'
    elif epoch < 30: return 'phase1_5'   # Mark4: 10-epoch unlock ramp
    else:            return 'phase2'

def get_unlock_ratio(epoch):
    if epoch < 20: return 0.0
    if epoch >= 30: return 1.0
    return (epoch - 20) / 10.0  # 线性 0→1

def get_lr(epoch, base_lr=8e-4):
    if epoch < 5:
        return base_lr * (0.01 + 0.99 * epoch / 5)
    elif epoch < 20:
        return 0.75 * base_lr
    elif epoch < 30:
        ratio = (epoch - 20) / 10.0
        return (0.75 * (1 - ratio) + 0.5 * ratio) * base_lr
    else:
        progress = (epoch - 30) / 20.0
        return 0.50 * base_lr * (0.5 + 0.5 * math.cos(math.pi * progress))
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
tail -f outputs/sdsd_delta/nohup.out
grep "Train stats" outputs/sdsd_delta/nohup.out | tail -20
grep "non-finite\|NaN" outputs/sdsd_delta/nohup.out
```

### 7.6 Epoch 5 过渡检查清单

| 检查项 | 健康阈值 | 不健康信号 |
|-------|---------|-----------|
| `pix` 趋势 | 持续下降到 <0.12 | 平台期或上升 |
| `ssim` 趋势 | 持续下降到 <0.40（即 SSIM>0.60） | 停滞在 >0.45 |
| `lit_up_map` 值域 | 均值在 [2.0, 6.0] | <1.0 或 >10.0 |
| `i_sup` | 下降到 <2.0 | >5.0 (lit_up 未激活) |
| WSD α 均值 | 在 [0.3, 0.7] | <0.1 或 >0.9 |
| `diag_prior` | 从 ~4.0 下降到 <2.0 | 上升（C_Ω 退化） |
| 梯度 norm | Encoder/ISPN/SGRF 同数量级 | 某模块爆炸或消失 |

---

## 8. 关键文件

| 文件 | 模块 |
|------|------|
| `models/modules/swd.py (legacy)` | SWD (SpatialWaveletDiverter, HaarDWT2D) |
| `models/modules/pure_rwkv_sace.py` | TCA, BiWKV, SpatialWKV2D, MVCShift, TemporalCorrespondence, TemporalAggregation |
| `models/modules/tfsi_v2.py` | DIE (MultiScaleSpatialBranch) |
| `models/modules/ispn_v2.py` | ISPN |
| `models/modules/ifpn.py` | ISPN (legacy, Mark3) |
| `models/modules/ndpn.py` | NDPN |
| `models/modules/mrpn.py` | MCPN |
| `models/modules/igrf.py` | SGRF |
| `models/tfs_net.py` | CXG, TFSNet (数据流编排) |
| `losses/losses.py` | TFSNetLoss (Kendall UW + PE Loss + Phase Schedule) |
| `train.py` | 训练循环 (grad accum + loss schedule + phase lr) |
| `configs/v6_bravo.yaml` | 训练配置 |
