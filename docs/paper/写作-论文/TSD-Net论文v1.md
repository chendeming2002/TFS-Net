# TSD\-Net: Triple Source Decoupled Restoration for Low\-Light Video Enhancement

Anonymous Submission

Anonymous affiliation
## 1 Introduction
Low\-light video enhancement \(LLVE\) aims to recover visibility, suppress noise, and maintain temporal coherence for videos captured under insufficient illumination\. LLVE is a long\-standing yet intricate problem, challenge not only human visual perception but also other vision tasks like nighttime object detection\. Under low\-light conditions, the extremely limited photon count leads to an inherently low signal\-to\-noise ratio \(SNR\), and amplifying illumination to improve visibility further magnifies the embedded noise\. Unique to video, independently realized per\-frame noise introduces flickering artifacts that severely undermine the enhancement quality\.
To tackle these challenges, numerous deep\-learning methods have been proposed, which can be broadly categorized into three paradigms: sequential, implicit, and parallel structure\.
**Sequential methods** first adjust illumination and then explicitly denoise the amplified result\. While conceptually clean, the sequential paradigm suffers from error propagation\ [1].
**Implicit methods** embed noise suppression within the model architecture without a dedicated denoising stage\. Most of them leverage Transformer to capture noise patterns, and often warp multi\-frame information to suppress stochastic noise and improve SNR\. Nevertheless, the high computational complexity, compounded by multi\-frame attention in video enhancement models, imposes prohibitive hardware demands during both training and inference, severely limiting their practical applicability\.
**Parallel methods** decompose the degradation into multiple components and assign separate branches to each\. Evidently, whether the noise is modeled suitablally determines the ability of such methods to effectively decompose and suppress the complex, coupled noise inherent in low\-light video enhancement\.
Existing works exploring divide\-and\-conquer denoising strategies primarily fall into three categories: decomposition tailored to specific datasets, partitioning based on mathematical or frequency\-domain statistical characteristics, and division in diffusion\-based denoising methods that hinges on the progressive timesteps\. However, these paradigms face insurmountable challenges in low\-light video denoising\. Source\-driven decomposition relies on inaccessible intermediate imaging pipeline representations in most real\-world LLVE scenarios, where only sRGB frames are available; spectral/statistical decomposition suffers from severe signal\-noise spectral overlap and violated stationarity assumptions due to low signal energy and spatiotemporally varying noise in low\-light conditions; diffusion\-based methods lack interpretable temporal\-aware design while noise is frame\-independently generated\. These limitations highlighting the urgent need for a more adaptive and video\-specific denoising decomposition strategy\.
To cope with the above challenges, we propose **Triple Source Decoupled Network\(TSD\-Net\)**, for low\-light video enhancement\. The framework decomposes the entangled degradation into three sources—illumination, imaging noise, and motion—based on their distinct temporal\-spatial statistics, processes each through a dedicated lightweight branch, and fuses the results in a physically motivated stage\-wise order\.
For the first challenge—the error propagation inherent in sequential methods—we adopt a parallel encoding with task\-specific branching strategy\. TSD\-Net routes the shared shallow features extracted by a lightweight encoder simultaneously into three dedicated branches \(ISPN, NDPN, MCPN\), each specializing in one degradation source\. This parallel topology ensures that no single branch's estimation error propagates into another's input, breaking the causal error chain that plagues sequential designs\. The three branch outputs are then re\-integrated by the Stage\-wise Guided Restoration and Fusion \(SGRF\) module in a physically motivated order\. Each fusion stage receives explicit, interpretable guidance from its corresponding upstream branch, combining the error\-isolation benefit of parallel decomposition with the physical correctness of stage\-wise fusion\.
For the second challenge—the prohibitive computational cost and limited controllability of implicit methods—we replace the expensive multi\-frame spatio\-temporal attention with a lightweight, complementary mechanisms\. The **Temporal Correspondence and Alignment \(TCA\)** module employs RWKV\-style linear\-complexity spatial scanning followed by an explicit temporal correspondence matrix $C_{t,\Omega}$ that encodes frame\-to\-frame affinity without computing optical flow or deformable offsets, reducing the temporal modeling cost from quadratic to linear\.
Central to TSD\-Net is the **TSDR framework**, which provides a principled basis for decomposing the complex, coupled degradation in low\-light video into three independently manageable sources\. Our formulation is grounded in two orthogonal dimensions of noise characterization—**temporal independence** \(whether the degradation is statistically independent across frames\) and **spatial selectivity** \(whether the degradation is globally distributed or concentrated at specific locations\)—yielding four canonical degradation types:
\(I\) globally distributed, frame\-independent noise such as read noise;
\(II\) spatially selective, frame\-independent noise such as photon shot noise concentrated in dark regions;
\(III\) globally distributed, temporally correlated degradation, i\.e\., illumination fluctuation;
\(IV\) spatially selective, temporally structured degradation, i\.e\., motion artifacts localized at object and camera motion boundaries\.
We show that these four types naturally reduce to three operational sources\. Types I and II, despite differing in spatial distribution, share the defining property of temporal independence—their realizations are i\.i\.d\. across frames—making them amenable to the same temporal\-averaging\-based denoising strategy; moreover, they co\-occur at every pixel as the Poisson\-Gaussian mixture inherent to sensor noise and cannot be analytically separated in a single frame\. They thus form a unified **imaging noise source**\. Type III constitutes the **illumination source**, a slowly varying global bias correctable through spatial smoothness priors and temporal anchoring\. Type IV constitutes the **dynamic source**, a structured temporal displacement that requires explicit alignment or compensation rather than averaging\. Crucially, no further merging is admissible: the illumination source and the dynamic source, though both temporally correlated, demand fundamentally incompatible processing—temporal smoothing for the former versus spatial compensation for the latter, with naive averaging of the dynamic source producing ghosting artifacts\. Likewise, the imaging noise source cannot be merged with either temporal source, as averaging eliminates i\.i\.d\. variance but not systematic bias or structured displacement\. This irreducibility guarantees that three is both sufficient and necessary for a complete, non\-redundant decomposition\. Guided by this framework, TSD\-Net instantiates the tri\-source split through a Wavelet Feature Router \(WFR\) for frequency\-level routing, TCA for temporal correspondence\-level separation, and a Degradation Prior Estimator \(DPE\) for source prior estimation, feeding three parallel specialized branches \(ISPN, NDPN, MCPN\) whose outputs are re\-integrated by SGRF in the physically motivated order\.
Our main contributions are as follows:
- We reformulate low\-light video enhancement as a divide\-and\-conquer denoising problem under low\-light scenarios and propose the TSDR framework\. This framework decomposition extends beyond existing two\-source paradigms\.
- We design **TSD\-Net**, an end\-to\-end architecture that instantiates the TSDR principle through wavelet\-based subband routing, RWKV\-style linear\-complexity temporal correspondence, and three dedicated source\-specific branches \(ISPN, NDPN, MCPN\) connected by a physically motivated stage\-wise fusion \(SGRF\) that prevents noise amplification at its root\.
- We introduce a **two\-phase progressive training schedule with perceptual decoupling** to resolve gradient conflicts among competing restoration objectives\.
## 2 Related Work

**Low-Light Video Enhancement.** Early learning-based methods extend single-image low-light enhancement to the temporal domain by independently processing each frame with Retinex-inspired decomposition or curve estimation, then applying post-hoc temporal filtering to reduce flicker. MBLLEN [31] and KinD [54] exemplify this single-frame paradigm. Subsequent works recognize the importance of multi-frame aggregation: SMOID [5] employs recursive hidden states for temporal propagation; StableLLVE [53] introduces a light transport matrix to stabilize inter-frame illumination; and SDSD [41] jointly learns spatial enhancement and temporal denoising from paired static/dynamic datasets. More recently, Transformer-based architectures have been adopted for their capacity to model long-range spatio-temporal dependencies. DNFormer [27] and LD-Net [42] apply multi-head self-attention across frames to capture global temporal context. However, these implicit methods embed noise suppression within the attention mechanism without explicitly modeling distinct degradation sources, leading to entangled gradient flows and prohibitive quadratic complexity when processing multi-frame tokens.

**Noise Decomposition and Divide-and-Conquer Denoising.** A parallel line of work attempts to decompose noise into manageable components. Physics-driven methods such as ELD [43] and PMN [11] model the imaging pipeline to separate Poisson shot noise from Gaussian read noise, but require access to raw sensor data unavailable in most real-world LLVE scenarios. Frequency-domain approaches partition noise by spectral characteristics: Noise2Same [2] exploits spectral stationarity, while WaveletSRNet [17] leverages wavelet sub-bands for signal-noise separation. These methods assume stationary noise statistics that are severely violated under low-light conditions, where spatially varying photon counts produce heteroscedastic noise. Diffusion-based denoisers such as DvSR [45] achieve strong results by leveraging progressive denoising timesteps, but their iterative inference is computationally expensive and their temporal modeling lacks explicit frame-awareness—noise is treated as frame-independent yet the denoising process ignores inter-frame structure. Critically, no existing decomposition framework jointly addresses the three degradation sources—imaging noise, illumination fluctuation, and motion artifacts—that co-occur and interact in low-light video. Our TSDR framework fills this gap by establishing a principled two-dimensional taxonomy (temporal independence × spatial selectivity) that yields three irreducible, non-redundant source categories.

**Temporal Alignment for Video Restoration.** Accurate temporal alignment is essential for multi-frame aggregation. Optical-flow-based methods (BasicVSR [4], BasicVSR++ [3]) achieve precise pixel-level alignment but suffer from flow estimation errors amplified by low SNR. Deformable convolutions (EDVR [40], TDAN [38]) learn implicit offsets but increase memory footprint. Recent works explore attention-based alignment: RVRT [28] employs recurrent attention for long-range propagation, while FGST [26] introduces flow-guided sparse Transformer to reduce complexity. However, these methods assume brightness constancy across frames—an assumption frequently violated in low-light videos due to exposure variation and flicker. Our TCA module addresses this by constructing illumination-invariant temporal correspondence through Retinex-motivated query-key generation and linear-complexity RWKV-style spatial scanning, reducing temporal modeling cost from quadratic to linear while remaining robust to inter-frame brightness changes.

## 3 Methodology
We build TSD-Net upon a Tri-Source Decoupled Restoration (TSDR) framework for low-light video enhancement. Instead of directly mapping a degraded low-light video clip to an enhanced frame, the proposed framework first represents the coupled degradation in the feature space and then separates it into three physically grounded source-aware components—illumination attenuation, imaging noise, and motion artifacts—for branch-specific restoration.

The framework consists of five major modules: Wavelet Feature Router (WFR), Degradation Prior Estimator (DPE), Temporal Correspondence and Alignment (TCA), three parallel source-processing networks (ISPN, NDPN, MCPN), and Stage-wise Guided Restoration and Fusion (SGRF). WFR routes encoder features into illumination/noise-dominant and structure-dominant streams via frequency-domain separation. DPE estimates per-pixel degradation intensity priors for illumination and noise from temporal statistics. TCA establishes illumination-invariant temporal correspondence across frames, producing a correspondence matrix that serves both denoising and motion compensation. ISPN corrects illumination attenuation and color bias, NDPN performs confidence-guided noise suppression, and MCPN compensates motion-induced degradation—each operating on its designated source to avoid gradient conflicts. Finally, SGRF fuses the three branch outputs in a physically motivated sequential order—denoise, deblur, then brighten—to reconstruct the enhanced center frame.

### 3.1 Tri-Source Decoupled Restoration Framework

We formalize low-light video degradation as a superposition of three sources. Let $X_t$ denote the clean frame at time $t$ and $I_t$ the observed low-light frame. The degradation model is:

$$I_t = \ell_t \odot X_t + \Delta_t^{\text{img}} + \Delta_t^{\text{dyn}} \tag{1}$$

where $\ell_t \in (0,1]^{H \times W}$ is the spatially varying illumination attenuation map, $\Delta_t^{\text{img}}$ denotes imaging noise (Poisson-Gaussian mixture from photon shot noise and read noise), and $\Delta_t^{\text{dyn}}$ captures motion-induced degradation (blur and misalignment at object/camera motion boundaries).

We characterize degradation along two orthogonal dimensions—**temporal independence** (whether realizations are i.i.d. across frames) and **spatial selectivity** (whether the degradation is globally distributed or spatially localized)—yielding four canonical types:

| | Globally distributed | Spatially selective |
|---|---|---|
| **Frame-independent** | (I) Read noise | (II) Shot noise (dark regions) |
| **Temporally correlated** | (III) Illumination fluctuation | (IV) Motion artifacts |

Types I and II share temporal independence and co-occur as the Poisson-Gaussian sensor noise mixture, forming a unified **imaging noise source** amenable to temporal averaging. Type III constitutes the **illumination source**, correctable through spatial smoothness priors. Type IV constitutes the **dynamic source**, requiring explicit alignment. No further merging is admissible: temporal smoothing eliminates i.i.d. noise but produces ghosting on structured motion; averaging cannot remove systematic illumination bias. Three sources are therefore both sufficient and necessary.

The overall pipeline proceeds as follows. A shared encoder $E$ maps the input clip $\{I_{t-N}, \dots, I_{t+N}\}$ to feature representations:

$$F_i = E(I_i), \quad i \in \{t\} \cup \Omega, \quad \Omega = \{t-N, \dots, t-1, t+1, \dots, t+N\} \tag{2}$$

A WFR performs sub-band-level routing of $F_i$ into illumination/noise-dominant features $F_i^{\text{DPE}}$ and illumination-invariant structural features $F_i^{\text{tca}}$. These feed into the Degradation Prior Estimator (DPE) for source prior estimation and the Temporal Correspondence and Alignment (TCA) module for temporal modeling. Three parallel branches—ISPN, NDPN, MCPN—process the decoupled sources, whose outputs are re-integrated by SGRF in physically motivated order:

$$\hat{X}_t = \text{SGRF}\!\Big(\underbrace{\text{NDPN}(F_t, C_{t,\Omega}, s_{\text{noise}})}_{\text{denoise}},\; \underbrace{\text{MCPN}(F_\Omega, C_{t,\Omega})}_{\text{deblur}},\; \underbrace{\text{ISPN}(F_t, s_{\text{illum}})}_{\text{brighten}}\Big) \tag{3}$$

The three-stage fusion order—denoise, deblur, brighten—is dictated by Eq. (1): applying illumination compensation $\ell_t^{-1}$ before denoising would amplify $\Delta_t^{\text{img}} / \ell_t$, causing noise explosion in dark regions where $\ell_t \to 0$.

### 3.2 Wavelet Feature Router (WFR)

WFR performs sub-band-level routing via Haar discrete wavelet transform without inverse reconstruction. Given encoder feature $F_i \in \mathbb{R}^{C \times H \times W}$, Haar DWT decomposes it into four sub-bands at half spatial resolution:

$$\text{LL}, \text{LH}, \text{HL}, \text{HH} = \text{HaarDWT}(F_i) \in \mathbb{R}^{C \times \frac{H}{2} \times \frac{W}{2}} \tag{4}$$

The LL sub-band, concentrating low-frequency illumination and noise energy, is split by a learnable gate:

$$\alpha = \sigma\!\big(\text{DWConv}_{3\times3}(\text{LL})\big) \in (0,1) \tag{5}$$

$$\text{LL}_{\text{DPE}} = \alpha \cdot \text{LL}, \quad \text{LL}_{\text{tca}} = \text{IN}\!\big((1-\alpha) \cdot \text{LL}\big) \tag{6}$$

where $\text{IN}(\cdot)$ denotes instance normalization that strips illumination statistics while preserving structural content. For high-frequency sub-bands $\text{HF} = [\text{LH}, \text{HL}, \text{HH}]$, a noise gate based on local HF energy routes noise-dominated components to DPE and structural edges to TCA:

$$g_n = \sigma\!\big(\text{Conv}(\text{HF}_{\text{energy}})\big), \quad \text{HF}_{\text{energy}} = \frac{1}{C}\sum_c (\text{LH}_c^2 + \text{HL}_c^2 + \text{HH}_c^2) \tag{7}$$

$$\text{HF}_{\text{DPE}} = g_n \cdot \text{HF}_{\text{cat}}, \quad \text{HF}_{\text{tca}} = \text{LN}\!\big((1-g_n) \cdot \text{HF}_{\text{cat}}\big) \tag{8}$$

The routed sub-bands are projected back to $C$ channels:

$$F^{\text{DPE}} = \text{LN}\!\big(\text{Conv}_{1\times1}([\text{LL}_{\text{DPE}}, \text{HF}_{\text{DPE}}])\big), \quad F^{\text{tca}} = \text{LN}\!\big(\text{Conv}_{1\times1}([\text{LL}_{\text{tca}}, \text{HF}_{\text{tca}}])\big) \tag{9}$$

A weak entropy regularizer $\mathcal{L}_{\text{wfr}} = -\mathbb{E}[\alpha \log \alpha + (1-\alpha)\log(1-\alpha)]$ prevents degenerate routing ($\alpha \to 0$ or $1$).

### 3.3 Degradation Prior Estimator (DPE)

DPE estimates per-pixel degradation priors $s_{\text{illum}}$ and $s_{\text{noise}}$ from temporal statistics of $\{F_i^{\text{DPE}}\}_{i=1}^{T}$. Exploiting the temporal independence of imaging noise (Types I–II), we compute robust temporal statistics:

$$\mu_t = \text{SoftMedian}(F^{\text{DPE}}_{1:T}), \quad \sigma_t = \sqrt{\text{Var}(F^{\text{DPE}}_{1:T}) + \epsilon}, \quad \text{SNR} = \mu_t / (\sigma_t + \epsilon) \tag{10}$$

where SoftMedian uses differentiable distance-weighted averaging centered at the hard median. The concatenated statistics $[\mu_t, \sigma_t, \text{SNR}] \in \mathbb{R}^{3C \times \frac{H}{2} \times \frac{W}{2}}$ are processed by a multi-scale dilated convolution branch:

$$F_{\text{fused}} = \text{Conv}_{1\times1}\!\Big(\big[\underbrace{\text{Conv}^{d=1}_{3\times3}}_{\text{local texture}},\; \underbrace{\text{Conv}^{d=2}_{3\times3}}_{\text{mid-scale}},\; \underbrace{\text{Conv}^{d=4}_{3\times3}}_{\text{large-scale}}\big]([\mu_t, \sigma_t, \text{SNR}])\Big) \tag{11}$$

The source intensity maps are obtained via:

$$[s_{\text{illum}}, s_{\text{noise}}] = \sigma\!\big(\text{Conv}_{1\times1}(F_{\text{fused}})\big) \in [0,1]^{2 \times \frac{H}{2} \times \frac{W}{2}} \tag{12}$$

The physical intuition is: $\sigma_t$ reflects noise magnitude (high temporal variance indicates strong frame-independent noise), while $\mu_t$ and SNR capture illumination level. The multi-scale receptive fields capture local noise granularity (dilation 1), mid-scale illumination gradients (dilation 2), and large-scale illumination regions (dilation 4).

### 3.4 Temporal Correspondence and Alignment (TCA)

TCA constructs illumination-invariant temporal correspondence and performs linear-complexity spatial modeling via RWKV-style scanning.

**Illumination-Invariant Spatial Scanning.** The input $F^{\text{tca}}$ has already been stripped of illumination statistics by WFR (Eq. 6, 8). A multi-scale shift module with three dilated depth-wise convolutions ($d \in \{1, 2, 3\}$) enriches local context. The shifted features are then processed by a four-directional Bi-WKV spatial scanner:

$$\mathbf{r}, \mathbf{k}, \mathbf{v} = W_r F, W_k F, W_v F \tag{13}$$

$$\text{wkv}_n = \frac{\sum_{m=1}^{n} e^{w_{n-m}} \cdot e^{k_m} v_m}{\sum_{m=1}^{n} e^{w_{n-m}} \cdot e^{k_m}}, \quad w = -\text{softplus}(\mathbf{w}_{\text{raw}}) < 0 \tag{14}$$

$$o_n = \sigma(r_n) \odot \text{wkv}_n \tag{15}$$

where $w < 0$ ensures exponential decay ($e^w < 1$), providing a mathematically guaranteed stable recurrence. Four directions (horizontal, vertical, main-diagonal, anti-diagonal) are assigned to four attention heads, concatenated and projected. Chunk-wise cumulative summation (chunk size 256) with clamped keys $k \in [-8, 8]$ prevents numerical overflow in long sequences. The overall spatial complexity is $\mathcal{O}(HW)$ per frame, compared to $\mathcal{O}(H^2W^2)$ for standard self-attention.

**Temporal Correspondence Matrix.** After spatial scanning, we construct the frame-to-frame correspondence matrix $C_{t,\Omega}$. Query and key tokens are projected to a compact subspace:

$$Q_t = W_q F_t^{\text{out}}, \quad K_i = W_k F_i^{\text{out}}, \quad Q_t, K_i \in \mathbb{R}^{\frac{C}{4} \times N} \tag{16}$$

$$C_{t,i} = \text{softmax}\!\left(\frac{Q_t^\top K_i}{\|Q_t\| \|K_i\| \cdot (\text{softplus}(\tau) + 0.05)}\right) \in \mathbb{R}^{N \times N} \tag{17}$$

where $N = d_s^2$ is the number of spatial tokens at downsampled resolution ($d_s \leq 96$ to bound memory), and cosine similarity with learnable temperature $\tau$ produces a row-normalized correspondence matrix. A self-supervised diagonal prior loss encourages identity correspondence in static regions:

$$\mathcal{L}_{\text{diag}} = -\frac{1}{|\Omega|} \sum_{i \in \Omega} \frac{1}{N} \sum_{j=1}^{N} \log\!\big(C_{t,i}[j,j] + \epsilon\big) \tag{18}$$

**Temporal Aggregation.** Neighboring features are warped into the center-frame coordinate space via $C_{t,i}$ and aggregated with learnable frame gates:

$$\hat{F}_i = C_{t,i} \cdot F_i^{\text{ds}}, \quad F_t^{\text{agg}} = \sum_{i \in \Omega} g_i \cdot \hat{F}_i, \quad g_i = \text{softmax}(\mathbf{g})_i \tag{19}$$

The aggregated feature is upsampled back to full resolution and combined with the center-frame feature via a residual connection and layer normalization.

### 3.5 Illumination-Source Processing Network (ISPN)

ISPN corrects illumination degradation through a Retinex-motivated formulation. Given the center-frame encoder feature $F_t^{\text{enc}}$ and illumination prior $s_{\text{illum}}$ from DPE, ISPN predicts a pixel-wise gain map and bias map:

$$h = \text{GELU}\!\big(\text{Conv}([F_t^{\text{enc}}, s_{\text{illum}}])\big) \tag{20}$$

$$G = \exp(\text{Conv}_{\text{gain}}(h)) \in [1, G_{\max}], \quad B = \tanh(\text{Conv}_{\text{bias}}(h)) \cdot \beta \in [-\beta, \beta] \tag{21}$$

where $G$ is the multiplicative illumination gain corresponding to the inverse illumination map $\ell_t^{-1}$ in Retinex theory, and $B$ is an additive correction for dark current and color bias. Both output heads are zero-initialized: $\text{Conv}_{\text{gain}}$ and $\text{Conv}_{\text{bias}}$ have weights and biases initialized to zero, ensuring $G = \exp(0) = 1$ and $B = 0$ at initialization—the network starts as an identity mapping and progressively learns illumination compensation.

### 3.6 Noise Degradation Processing Network (NDPN)

NDPN performs confidence-guided denoising by exploiting the diagonal structure of $C_{t,\Omega}$. The diagonal entry $C_{t,i}[j,j]$ measures the reliability of temporal correspondence at position $j$ between frames $t$ and $i$. High confidence (strong diagonal) indicates reliable multi-frame denoising; low confidence (diffused diagonal) signals motion or occlusion requiring stronger spatial denoising.

$$c = \text{Conv}\!\big([\text{diag}(C_{t,i})]_{i \in \Omega}\big), \quad \mathbf{n} = \text{Conv}\!\big([F_t^{\text{enc}}, F_t^{\text{agg}}]\big) \tag{22}$$

$$\text{strength} = \sigma\!\big(\text{Conv}([\mathbf{n}, c])\big), \quad f_{\text{noise}} = F_t^{\text{enc}} - \gamma_n \cdot \mathbf{n} \odot \text{strength} + \text{Proj}(s_{\text{noise}}) \tag{23}$$

The scalar parameter $\gamma_n$ and the projection layer $\text{Proj}$ are both zero-initialized, so that $f_{\text{noise}} \approx F_t^{\text{enc}}$ at the start of training. This pass-through initialization ensures that activating NDPN does not perturb the already-converged illumination path.

### 3.7 Motion Compensation Processing Network (MCPN)

MCPN compensates motion-induced degradation by reusing $C_{t,\Omega}$ for explicit temporal alignment. Unlike flow-based methods, MCPN operates on the correspondence matrix directly:

$$f_i^{\text{align}} = C_{t,i} \cdot F_i^{\text{ds}}, \quad f_{\text{agg}} = \text{Conv}\!\big([f_i^{\text{align}}]_{i \in \Omega}\big) \tag{24}$$

A startup gate ensures initial pass-through behavior:

$$g_t = \sigma(w_g + b_{\text{start}}), \quad b_{\text{start}} \xrightarrow{\text{init}} 5.0 \implies g_t \approx 1 \tag{25}$$

$$f_{\text{fused}} = g_t \cdot F_t^{\text{enc}} + (1 - g_t) \cdot f_{\text{agg}} + \gamma_m \cdot \text{Refine}(f_{\text{fused}}) \tag{26}$$

where $\gamma_m = 0$ and the final convolution of Refine is zero-initialized, guaranteeing $f_{\text{motion}} \approx F_t^{\text{enc}}$ at initialization. As training progresses, $b_{\text{start}}$ decreases and $\gamma_m$ grows, allowing MCPN to gradually incorporate multi-frame motion compensation.

### 3.8 Cross-Excitation Gate (CXG)

Since denoising and motion compensation may produce conflicting gradients in overlapping regions (e.g., noisy moving edges), CXG performs cross-branch modulation:

$$f_{\text{noise}}^{\text{out}} = f_{\text{noise}} \odot \sigma\!\big(\text{SE}(f_{\text{motion}})\big), \quad f_{\text{motion}}^{\text{out}} = f_{\text{motion}} \odot \sigma\!\big(\text{SE}(f_{\text{noise}})\big) \tag{27}$$

where $\text{SE}(\cdot)$ is a squeeze-and-excitation block. At inference, the sigmoid gates are replaced by their running-mean statistics $\bar{g}_n, \bar{g}_m$ for zero-overhead re-parameterization.

### 3.9 Stage-wise Guided Restoration and Fusion (SGRF)

SGRF enforces the physically motivated restoration order through three cascaded stages, each equipped with a zero-initialized gate to ensure non-disruptive activation.

**Zero-Gate StageBlock.** Each stage applies a gated residual transformation:

$$\delta = \text{Conv}\!\big(\text{GELU}(\text{Conv}(f_{\text{branch}}))\big) \cdot \gamma_{\text{gate}}, \quad \gamma_{\text{gate}} \xrightarrow{\text{init}} 0 \tag{28}$$

When $\gamma_{\text{gate}} = 0$, the stage output equals its input regardless of $f_{\text{branch}}$, providing an architectural guarantee that newly activated branches cannot corrupt previously learned representations.

**Stage 1 (Denoise):**
$$I_1 = \text{clamp}\!\big(I_t + \text{StageBlock}_1(f_{\text{noise}}^{\text{out}}) \cdot \gamma_1,\; 0,\; 1\big) \tag{29}$$

**Stage 2 (Deblur):**
$$I_2 = \text{clamp}\!\big(I_1 + \text{StageBlock}_2(f_{\text{motion}}^{\text{out}}) \cdot \gamma_2,\; 0,\; 1\big) \tag{30}$$

**Stage 3 (Brighten):**
$$\hat{X}_t = \text{clamp}\!\big(I_2 \odot G + B + \text{Refine}(I_2 \odot G + B) \cdot \gamma_3,\; 0,\; 1\big) \tag{31}$$

All gates $\gamma_1 = \gamma_2 = \gamma_3 = 0$ at initialization, so the network initially reduces to $\hat{X}_t = I_t \cdot G + B$, equivalent to pure ISPN illumination enhancement. The gates are learnable scalars that progressively grow during training, smoothly transitioning from illumination-only to full tri-source restoration.

### 3.10 Progressive Training with Perceptual Decoupling

Training proceeds in two phases to prevent gradient conflicts among the three restoration objectives.

**Phase 1 (Illumination Warmup, epochs 0–19).** Only the encoder, WFR, DPE, and ISPN are trained. NDPN and MCPN outputs are forced to zero; SGRF gates remain at zero. The loss comprises pixel reconstruction ($\mathcal{L}_{\text{pix}}$), structural similarity ($\mathcal{L}_{\text{ssim}}$), illumination smoothness ($\mathcal{L}_{\text{smooth}}$), and gain supervision ($\mathcal{L}_{\text{gain}}$). TCA is trained with $\mathcal{L}_{\text{diag}}$ and a warp consistency loss but its outputs do not flow into restoration branches.

**Phase 1.5 (Gradual Activation, epochs 20–29).** NDPN and MCPN are linearly unlocked over 10 epochs via a ratio $r \in [0, 1]$ that scales their outputs. Combined with the zero-initialized $\gamma_n$, $\gamma_m$, and SGRF gates $\gamma_1$, $\gamma_2$, a triple safety mechanism ensures that the transition introduces no abrupt perturbation: branch zero-init $\times$ unlock ratio $\times$ stage gate = 0 at the boundary.

**Phase 2 (Full Training, epochs 30–49).** All modules are active. Perceptual decoupling assigns stage-specific losses to avoid conflicting gradients on the same output: SSIM loss supervises $I_1$ (encouraging structural denoising), VGG perceptual loss supervises $I_2$ (encouraging sharp deblurring), and Charbonnier loss supervises $\hat{X}_t$ (encouraging accurate brightness). All losses are balanced by learnable Kendall uncertainty weights.

# Reference
[1] W. Yang, W. Wang, H. Huang, S. Wang and J. Liu, "Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement," in IEEE Transactions on Image Processing, vol. 30, pp. 2072-2086, 2021, doi: 10.1109/TIP.2021.3050850.

