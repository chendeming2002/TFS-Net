# TSD-Net: Triple Source Decoupled Restoration for Low-Light Video Enhancement

Anonymous Submission

Anonymous affiliation

## 1 Introduction

Low-light video enhancement (LLVE) is a fundamental task in computer vision, aiming to recover visibility, suppress noise, and maintain temporal coherence for videos captured under insufficient illumination. Limited photon counts yield a low signal-to-noise ratio, and brightening the frames amplifies embedded noise in proportion [1]. Since sensor noise is realized independently in each frame, per-frame enhancement also introduces temporal flicker [2]. The resulting degradation—low visibility, noise, color distortion, and temporal instability—impairs downstream tasks including nighttime object detection [3], face recognition [4], and autonomous driving perception [5].

Existing LLVE methods fall into three design families. **Sequential methods** [6, 7] first estimate and correct illumination, then apply explicit denoising to the amplified result; their cascaded structure inherits and amplifies errors from the illumination stage. **Implicit methods** [8, 9], most notably Transformer-based video restoration architectures, embed restoration entirely within multi-frame spatio-temporal attention, treating every inter-frame variation as motion to be aligned; this conflates brightness drift, sensor noise, and object displacement, which require smoothing, temporal averaging, and explicit compensation respectively. **Parallel methods** [10, 11, 12] decompose degradation into components and assign each to a dedicated branch, offering the most natural match to the multi-source structure of low-light video. However, existing decompositions—by imaging domain [10], frequency band [11], or diffusion timestep [12]—partition by signal representation rather than by temporal behavior, and therefore place sources with fundamentally incompatible restoration requirements into the same branch.

An underestimated design flaw applies to parallel methods broadly: assigning branches to degradation sources in the architecture does not guarantee that training will specialize those branches. Under the standard training setup, every branch is supervised toward the same clean frame $X_t$. The resulting per-branch objective,

$$\min_{Y_k} \|Y_k - X_t\|_1, \quad k \in \{N, L, M\},$$

has a unique minimizer that any single branch can reach independently, so the optimizer has no incentive to distribute the work. We term this *branch-target collapse*: the three branches converge to the same solution and the parallel structure collapses to a single effective path. We verify this empirically: removing any one branch while redistributing its fusion weight changes the final metric by less than [X] dB (Sec. 4). The natural corrective—penalizing output similarity with $|\cos(Y_i, Y_j)|$—does not resolve collapse; it rewards dissimilarity, and a branch that outputs a constant image satisfies the penalty at zero cost. We observe exactly this failure mode, in which one branch degenerates to a constant map.

We propose **TSD-Net** (Triple Source Decoupled Network), an end-to-end LLVE model whose design is organized by the **Tri-Source Decoupled Restoration (TSDR)** principle. The key insight is that the three dominant low-light video degradation sources—imaging noise, illumination variation, and motion—differ in their temporal statistics in a way that prescribes distinct optimal estimators. Noise is i.i.d. across frames; its optimal suppressor is a temporal average. Illumination is a slowly varying global bias; its optimal estimator is a spatial low-pass filter. Motion is a structured cross-frame displacement; its estimator is frame warping. We enforce source specialization directly in the supervision via **Decoupled Target Supervision (DTS)**: BranchN is trained on a motion-compensated temporal average $X^\text{avg}_t$; BranchL on an illumination-normalized image $X^\text{illu}_t = X^\text{avg}_t / \hat{L}_t$, where $\hat{L}_t$ is a large-kernel low-pass estimate of the illumination field; and BranchM on the average of the two farthest neighboring frames after flow-based warping, $X^\text{align}_t$, which carries the strongest displacement signal without conflating with the noise-averaging target. The three targets have distinct optima by construction, so the branches are forced to specialize from the objective level rather than from regularization terms.

TSD-Net processes a $T=5$ frame clip through four successive stages. A **shared encoder** produces half-resolution per-frame features, and a **PixelTemporalAttentionSimple** module uses 3D convolutions over a three-frame window to output a center-aligned spatial feature $F_\text{spatial}$. A **MatrixRWKV** module operates at frame-summary level, maintaining matrix-valued recurrent states $S_t \in \mathbb{R}^{d \times d}$ with data-dependent decay to encode holistic inter-frame context at linear time complexity. The **Triple Query TCA** module then generates three source-aware features by pairing three independently learned queries with a shared key-value built from three temporal statistics—frame mean $\bar{F}$ (noise prior), spatially smoothed mean $\tilde{F}$ (illumination prior), and peak inter-frame difference $\Delta F$ (motion prior)—each gated by the MatrixRWKV context via a zero-initialized linear layer. RWKV-style bidirectional spatial scanning at linear complexity produces $F_N$, $F_L$, $F_M$ with a feature-level orthogonality regularizer. BranchN, BranchL, and BranchM are lightweight NAFBlock-based networks: BranchL includes a Retinex-inspired illumination head with learnable $\gamma$, and BranchM applies flow-based deformable alignment with confidence gating on the two adjacent frames. A **V7RFusion** module combines the three outputs with learned per-pixel softmax weights and a zero-initialized center-frame residual gate ($\tanh(\gamma) \cdot X_\text{center}$). The total model contains 3.74M parameters.

We evaluate TSD-Net on [N] benchmarks including SDSD-indoor and SDSD-outdoor [2], SMID [13], DID [14], and DAVIS [15]. TSD-Net achieves [X] dB PSNR on SDSD-indoor and outperforms state-of-the-art methods by up to [X] dB with [X]$\times$ fewer parameters than VLLVE++ [16], and attains the best temporal consistency across all tested datasets, as shown in Fig. 1.

In summary, our contributions are as follows:

- **An identified failure mode.** We identify and characterize *branch-target collapse* in parallel restoration architectures: when all branches are trained toward the same reconstruction target, they converge to the same solution regardless of upstream architectural decomposition, and this degeneracy is invisible in the final reconstruction metric. Output-similarity penalties fail to resolve it because constant-output branches satisfy them trivially.

- **A novel supervision principle.** We propose Decoupled Target Supervision (DTS) within the TSDR framework: each branch is trained on the classical estimator for its source—temporal averaging for noise, illumination-normalized low-pass for illumination, farthest-pair flow warping for motion. The three targets have distinct optima by construction and require no additional annotation.

- **TSD-Net.** We design an end-to-end 3.74M-parameter architecture that pairs source-aware temporal context (MatrixRWKV with matrix-valued states) and source-differentiating feature routing (Triple Query TCA with shared statistical key-value) with three source-specialized branches and adaptive pixel-wise fusion.

- **State-of-the-art performance.** Extensive experiments on [N] benchmarks show TSD-Net outperforms SOTA methods by up to [X] dB PSNR with strong temporal consistency. Code and models will be released to facilitate future research.

## 3 Proposed Method

### 3.1 Tri-Source Decoupled Restoration Framework

We formulate low-light video enhancement as a **tri-source decoupled restoration** problem. Let $\{I_{t-k}, \dots, I_t, \dots, I_{t+k}\}$ denote a sliding window of $T = 2k+1$ consecutive low-light frames centered at the target frame $I_t$. Following the low-light imaging model [11, 20], each observed frame $I_t$ is modeled as a composite of three underlying degradation sources acting upon a latent clean signal $X_t$:

$$I_t = \mathcal{D}\big(\underbrace{\mathcal{I}(X_t)}_{\text{illumination}},\; \underbrace{\mathcal{N}_t}_{\text{noise}},\; \underbrace{\mathcal{M}_t}_{\text{motion}}\big),$$

where $\mathcal{I}(\cdot)$ represents the illumination degradation operator that suppresses exposure and disrupts brightness constancy, $\mathcal{N}_t$ denotes the sensor noise term modeled as heteroscedastic Poissonian-Gaussian noise with variance dependent on local illumination level, and $\mathcal{M}_t$ accounts for motion-induced corruption including blur, misalignment, and rolling shutter artifacts. The composite operator $\mathcal{D}$ captures the entangled interaction of these three sources in the observed image.

More concretely, adopting the single-frame noise model of Foi *et al.* [11] and Wang *et al.* [3], the observed pixel intensity at spatial location $p$ satisfies:

$$I_t(p) = L_t(p) \cdot X_t(p) + n_t(p), \quad n_t(p) \sim \mathcal{N}\big(0,\; \sigma_s^2 \cdot L_t(p) \cdot X_t(p) + \sigma_r^2\big),$$

where $L_t(p) \in (0, 1]$ is the illumination attenuation at pixel $p$, and $\sigma_s^2$, $\sigma_r^2$ are the shot noise and read noise parameters respectively. Extending this to the multi-frame setting, the temporal dimension introduces motion displacement $\mathbf{d}_{t \to t'}(p)$ between frames $t$ and $t'$:

$$I_{t'}(p) \approx L_{t'}(p) \cdot X_t\big(p + \mathbf{d}_{t \to t'}(p)\big) + n_{t'}(p).$$

The key observation motivating our tri-source formulation is that **the three degradation sources exhibit fundamentally different temporal statistics**, as summarized in Table 1:

| Property | Illumination $\mathcal{I}$ | Noise $\mathcal{N}$ | Motion $\mathcal{M}$ |
|:---|:---|:---|:---|
| **Temporal type** | Bias (slowly varying) | Variance (i.i.d. across $t$) | Displacement (structured) |
| **Frame correlation** | $\|L_t - L_{t-1}\| \to 0$ | $\mathbb{E}[\mathcal{N}_t \cdot \mathcal{N}_{t'}] \approx 0,\; t \neq t'$ | $\mathbf{d}_{t \to t'} \neq 0$ in moving regions |
| **Optimal strategy** | Spatial smoothness prior | Temporal averaging under correspondence | Alignment / compensation |
| **Amplification chain** | Amplifies noise via $\nabla\mathcal{I}$ | Degrades $\mathcal{I}$ estimation | Breaks correspondence for denoising |

This taxonomy reveals why **a two-source split is insufficient**: any grouping that merges two of the three sources into a single branch conflates degradation types with incompatible temporal statistics.

- **Illumination + Noise** vs. Motion (Retinex-style): The "reflectance" branch $R = I / L$ absorbs both noise and clean texture. Noise amplification during the $I/L$ division makes subsequent denoising harder, as the noise variance becomes spatially dependent on the potentially inaccurate $L$ estimate.
- **Illumination + Motion** vs. Noise: The "structural" branch absorbs both slowly varying illumination and structured motion, preventing the model from exploiting the temporal smoothness of $L$ for flicker suppression.
- **Noise + Motion** vs. Illumination: The "residual" branch absorbs both random noise and structured motion. Since noise benefits from temporal averaging while motion requires compensation, the unified branch must simultaneously denoise *and* deblur—objectives whose optimal temporal processing strategies are diametrically opposed.

In contrast, our tri-source decomposition assigns each source to a dedicated branch with **matched temporal processing**: ISPN exploits cross-frame illumination smoothness for flicker-free light-up; NDPN leverages temporal averaging under $C_{t,\Omega}$-verified correspondence for noise suppression; MCPN applies motion-specific compensation where correspondence fails. The three branches interact through the shared $C_{t,\Omega}$ matrix and source priors $s_{\text{illu}}$, $s_{\text{noise}}$, ensuring that the decoupling does not sacrifice information flow.

The **sequential coupling** of these sources further demands explicit decoupling. Noise is amplified by illumination enhancement ($\nabla\mathcal{I}$ amplifies $\mathcal{N}_t$); motion breaks the temporal correspondence needed for multi-frame denoising ($\mathcal{M}_t$ degrades $C_{t,\Omega}$); and illumination estimation is misled by both noise and motion artifacts. This coupling motivates our design choice to perform decoupling *early* in the feature space—via wavelet-domain frequency routing and illumination-invariant temporal correspondence—before routing to source-specific branches.

### 3.2 Source-Decoupled Supervision

The framework above specifies how the three sources are separated in the representation space. It does not specify what each branch should reconstruct, and we find that this second question is not automatically answered by the first. This subsection states the supervision design, and motivates it by describing the degenerate solution that arises in its absence.

**Why a shared target fails.** Let the three branches output $Y_{\text{NDPN}}, Y_{\text{MCPN}}, Y_{\text{ISPN}} \in \mathbb{R}^{3 \times H \times W}$ and let $X_t$ be the ground truth. Under the conventional design, each branch is supervised against the same target:

$$\mathcal{L}_{\text{branch}} = \sum_{b \in \{N, M, I\}} w_b \, \| Y_b - X_t \|_1. \tag{4}$$

This objective separates across branches, and each term attains zero at $Y_b = X_t$. Consequently the global minimizer has $Y_{\text{NDPN}} = Y_{\text{MCPN}} = Y_{\text{ISPN}} = X_t$: the optimal solution of the decoupled architecture is an *undecoupled* one. Output similarity is not a training artifact to be suppressed but the exact optimum of the stated problem, which explains why regularizing it is unreliable. Adding a decorrelation penalty $\lambda |\cos(Y_a, Y_b)|$ produces a minimizer with $Y_a$ orthogonal to $Y_b$ in the mean-subtracted sense; a branch that outputs a constant image satisfies this at zero cost while contributing nothing to restoration, and we observe precisely this collapse in practice.

The difficulty is therefore not that the branches receive indistinguishable features—we verify in Section 4 that the branch-specific queries do produce measurably different representations—but that they are asked for indistinguishable outputs.

**Source-specific targets.** We replace the shared target with three targets constructed from the input sequence itself, requiring no additional annotation. Each target is the quantity that its designated source's optimal estimator would produce.

For the imaging noise source, the relevant property is temporal independence, so the optimal estimator is a temporal average—but computed over motion-compensated frames, since averaging uncompensated frames reintroduces motion as ghosting:

$$X^{\text{avg}}_t = \frac{1}{|\Omega|+1}\Big( I_t + \sum_{i \in \Omega} \mathcal{W}_{i \to t}\big(I_i\big) \Big), \tag{5}$$

where $\mathcal{W}_{i \to t}$ warps frame $i$ to the center time instant. NDPN is supervised on $X^{\text{avg}}_t$, which is by construction noise-reduced while retaining texture.

For the illumination source, the relevant property is that illumination is low-frequency and globally distributed, so it can be isolated by spatial low-pass filtering. Letting $\hat{L}_t = \mathrm{LPF}(X^{\text{avg}}_t)$ be the estimated illumination field, we define the illumination-normalized target

$$X^{\text{illu}}_t = X^{\text{avg}}_t \oslash \hat{L}_t, \tag{6}$$

and supervise ISPN on it. This target preserves relative reflectance and color while removing absolute exposure, so the branch is not required to reproduce texture that the other branches already handle.

For the motion source, the quantity to be corrected *is* the displacement, so the natural target is built from neighbouring frames brought into the center frame's coordinate system. We use the two outermost frames of the window,

$$X^{\text{align}}_t = \tfrac{1}{2}\big( \mathcal{W}_{t-k \to t}(I_{t-k}) + \mathcal{W}_{t+k \to t}(I_{t+k}) \big), \tag{7}$$

rather than an average over all of $\Omega$. The outermost frames carry the largest displacement and therefore the strongest alignment signal; averaging over all neighbours instead makes Eq. (7) numerically almost indistinguishable from Eq. (5)—we measure a mean absolute difference of $0.012$—which would leave NDPN and MCPN with effectively the same target and defeat the purpose of the construction. Supervising MCPN on $X^{\text{align}}_t$ makes its output the alignment residual rather than a full reconstruction.

The three targets differ by construction: $X^{\text{avg}}_t$ retains texture but not alignment, $X^{\text{illu}}_t$ retains relative structure but not absolute illumination, and $X^{\text{align}}_t$ retains geometry but not temporal aggregation. Their optima no longer coincide, so the branches cannot all converge to the same function, and the fusion stage receives three genuinely distinct candidates. The re-integration loss remains the original $\ell_1$ between the fused output $O_t$ and $X_t$; no additional penalty terms are introduced.

This formulation also clarifies the scope of the TSDR claim. The three-source decomposition is a statement about how degradation should be *partitioned*; the supervision design is what makes that partition *operational*. A decomposition whose parts are trained to agree is not a decomposition in any functional sense, and the two ingredients are needed together.

---

## Reference

[1] Cai, Yuanhao, et al. "Retinexformer: One-stage retinex-based transformer for low-light image enhancement." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2023.

[2] Wei, Kaixuan, et al. "A physics-based noise formation model for extreme low-light raw denoising." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2020.

[3] Wang, Ruixing, et al. "Seeing dynamic scene in the dark: A high-quality video dataset with mechatronic alignment." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2021.

[4] Yang, Wenhan, et al. "Sparse gradient regularized deep retinex network for robust low-light image enhancement." IEEE Transactions on Image Processing 30 (2021): 2072-2086.

[5] Li, Huaqiu, and Haoqian Wang. "Interpretable unsupervised joint denoising and enhancement for real-world low-light scenarios." International Conference on Learning Representations (ICLR). 2025.

[6] Wei, Chen, et al. "Deep retinex decomposition for low-light enhancement." Proceedings of the British Machine Vision Conference (BMVC). 2018.

[7] Liang, Jingyun, et al. "VRT: A video restoration transformer." IEEE Transactions on Image Processing 33 (2024): 2171-2182.

[8] Liang, Jingyun, et al. "Recurrent video restoration transformer with guided deformable attention." Advances in Neural Information Processing Systems 35 (2022): 378-393.

[9] Feng, Hansen, et al. "Learning physics-informed noise models from dark frames for low-light raw image denoising." IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).

[10] Li, Yuezhou, et al. "Perceptual decoupling with heterogeneous auxiliary tasks for joint low-light image enhancement and deblurring." IEEE Transactions on Multimedia 26 (2024): 6663-6675.

[11] Zhang, Ruikang, et al. "DualDn: Dual-domain denoising via differentiable ISP." European Conference on Computer Vision (ECCV). Springer, 2024.

[12] Xu, Xiaogang, et al. "SNR-aware low-light image enhancement." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022.

[13] Liu, Jiawei, et al. "Residual denoising diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2024.

[14] Foi, Alessandro, et al. "Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data." IEEE Transactions on Image Processing 17.10 (2008): 1737-1754.

[15] Zhu, Lingyu, et al. "Temporally consistent enhancement of low-light videos." International Journal of Computer Vision 132 (2024): 4015-4034.

[16] Li, Hao, et al. "Low-light video enhancement via spatial-temporal consistent illumination decomposition." Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI). 2025.
