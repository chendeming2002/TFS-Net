# TSD-Net: Triple Source Decoupled Restoration for Low-Light Video Enhancement

Anonymous Submission

Anonymous affiliation

## 1 Introduction

Low-light video enhancement (LLVE) aims to recover visibility, suppress noise, and maintain temporal coherence under insufficient illumination. It is a long-standing yet intricate problem, challenging not only human visual perception but also downstream vision tasks such as nighttime object detection [1]. Under low-light conditions, the extremely limited photon count leads to an inherently low signal-to-noise ratio (SNR), and amplifying illumination to improve visibility further magnifies the embedded noise [2]. Unique to video, independently realized per-frame noise introduces flickering artifacts that severely undermine enhancement quality [3].

To tackle these challenges, many deep-learning based methods for LLVE have been proposed, which can be broadly categorized into three paradigms: sequential [4, 5, 6], implicit [7, 8], and parallel [9, 10].

Sequential methods adopt a cascaded strategy that first enhances illumination and subsequently applies explicit denoising to the amplified frames. While this design offers conceptual clarity, it is vulnerable to error accumulation, as imperfections in the illumination correction are propagated and amplified in the denoising stage.

Implicit methods integrate noise suppression directly into the model architecture, eliminating the need for a separate denoising stage. Most of these approaches adopt Transformers to capture noise distributions and exploit multi-frame information to reduce stochastic noise and improve SNR. However, the computational cost of Transformers is already high, and the additional multi-frame attention required for video data further exacerbates this burden. As a result, both training and inference demand excessive hardware resources, which severely hinders their deployment in real-world applications.

Parallel methods attempt to address these limitations by disentangling the mixed degradation into multiple distinct components and assigning dedicated branches for source-specific restoration. This modular design offers several compelling advantages, including improved specialization, more flexible optimization, and enhanced interpretability. However, its effectiveness fundamentally depends on whether the underlying degradation sources can be properly characterized and decoupled. Due to the highly coupled nature of low-light video degradation, inappropriate decomposition may lead to incomplete modeling of degradation interactions and limit the ability of parallel frameworks to achieve comprehensive restoration.

Existing works exploring divide-and-conquer denoising strategies primarily fall into three categories: decomposition tailored to specific datasets [2, 11], partitioning based on mathematical or frequency-domain statistical characteristics [9, 12], and division in diffusion-based denoising methods that hinges on the progressive timesteps [13]. However, these paradigms face insurmountable challenges in low-light video denoising. Source-driven decomposition relies on inaccessible intermediate imaging pipeline representations in most real-world LLVE scenarios, where only sRGB frames are available [11]; spectral/statistical decomposition suffers from severe signal-noise spectral overlap and violated stationarity assumptions due to low signal energy and spatiotemporally varying noise in low-light conditions [12]; diffusion-based methods lack interpretable temporal-aware design while noise is frame-independently generated [13]. These limitations highlight the urgent need for a more adaptive and video-specific denoising decomposition strategy.

To address the above limitation of existing degradation decomposition strategies, we propose Triple Source Decoupled Network (TSD-Net) for low-light video enhancement. The proposed Triple Source Decoupled Restoration (TSDR) framework decomposes the entangled low-light video degradation into three physically meaningful sources—illumination, imaging noise, and motion—according to their distinct spatio-temporal characteristics. Each source is processed through a dedicated lightweight branch, and the restored components are progressively integrated through a physically motivated stage-wise fusion strategy.

For the first challenge—effectively decoupling highly coupled degradation sources while maintaining their complementary interactions—we adopt a parallel source-aware encoding strategy with task-specific branching. TSD-Net routes the shared shallow features extracted by a lightweight encoder simultaneously into three dedicated branches (ISPN, NDPN, and MCPN), each specializing in illumination, imaging noise, and motion restoration. This design prevents different degradation sources from interfering with each other's representations while preserving source-specific optimization. The branch outputs are subsequently re-integrated by the Stage-wise Guided Restoration and Fusion (SGRF) module according to their restoration dependencies, enabling physically consistent collaboration among multiple sources.

For the second challenge—effectively capturing global temporal interactions among multiple degradation sources—we introduce the Temporal Correspondence and Alignment (TCA) module. Although source-specific branches enable targeted degradation restoration, low-light video enhancement still requires comprehensive modeling of inter-frame relationships to maintain temporal consistency. Existing temporal modeling approaches often rely on explicit motion estimation or computationally intensive spatio-temporal attention, which may limit their ability to efficiently capture global frame-wise dependencies. To address this issue, TCA employs RWKV-based linear-complexity global interaction modeling to establish full-frame cross-frame correspondence, followed by an explicit temporal correspondence matrix to characterize frame-to-frame affinity. This design enables efficient global temporal interaction modeling without relying on optical flow or deformable alignment, providing effective temporal guidance for source-aware restoration.

A third challenge, however, is less often discussed and in our experience more decisive than either of the above: **a parallel architecture does not by itself produce a division of labor.** Assigning three branches to three sources is a statement about network topology, not about what the branches will actually learn. When the three branches are trained against a single reconstruction target—as they usually are—the objective has one global optimum that every branch can reach independently, and gradient descent has no reason to distribute the work. Each branch converges to a complete reconstruction, and the architecture collapses into three redundant copies of the same enhancer. We verify this empirically in Section 4: on our trained model the three branch outputs remain nearly collinear, and ablating any one of them while renormalizing the fusion weights leaves the final metric essentially unchanged. Notably, this degeneracy cannot be diagnosed from the final metric, which is why it has remained largely unattended; it is a property of the supervision, not of the decomposition.

This observation reframes the decoupling problem. The obstacle is not that the degradation sources are inseparable in the feature space—our measurements show that the branch-specific query representations do differ measurably—but that **the supervision signal does not ask the branches to do different things.** This distinction matters because it determines which remedies can work. Penalizing similarity between branch outputs, as we and others have attempted, does not establish a division of labor; it only rewards outputs that are dissimilar, and a branch that abandons structure altogether satisfies such a penalty at zero cost. We observed exactly this failure mode, in which one branch collapses to a constant image. We stress that the remedy is not to reduce the number of branches—a single-path enhancer is a well-established design with ample prior work—but to make the three-source decomposition operational, which is the question this paper addresses.

We therefore argue that source decoupling should be enforced at the **supervision** level rather than the architectural or regularization level. Instead of training all three branches against the same ground truth, each branch is assigned a distinct target derived from the input sequence itself: the noise branch is supervised on a motion-compensated temporal average, which is the optimal estimator for temporally independent noise; the illumination branch is supervised on an illumination-normalized low-frequency image, so that it is responsible for exposure and color rather than texture; and the motion branch is supervised on neighbouring frames warped to the center time instant, making its output the displacement correction itself. These targets require no additional annotation and, critically, have **distinct optima**. The three branches can no longer all reach the same solution, and the interaction between them is resolved at the level of the objective rather than through competing loss terms.

We note that this design also removes a tension that runs through the parallel-restoration literature. Prior work resolves inter-branch conflict either by partitioning the input (frequency bands, domains, timesteps) or by weighting competing losses; both operate on the representation or on the loss magnitude, leaving the targets shared. Our finding is that shared targets, not shared representations, are the binding constraint.

The decomposition of TSDR is motivated by the observation that different low-light video degradation sources exhibit distinct spatio-temporal characteristics and therefore require different restoration strategies. Specifically, we characterize degradation factors from two complementary perspectives: temporal independence, which describes whether a degradation exhibits statistical independence across consecutive frames, and spatial selectivity, which indicates whether the degradation is globally distributed or spatially localized. Based on these two dimensions, low-light video degradation can be categorized into four canonical degradation types [14, 15]:

(I) globally distributed and temporally independent degradation, corresponding to sensor-related random noise such as read noise;

(II) spatially selective and temporally independent degradation, corresponding to signal-dependent noise such as photon shot noise that becomes more prominent in low-intensity regions;

(III) globally distributed and temporally correlated degradation, corresponding to illumination variation across frames; and

(IV) spatially localized and temporally correlated degradation, corresponding to motion-induced disturbance caused by object or camera movement.

These four degradation types can be further grouped into three restoration-oriented sources according to their underlying recovery mechanisms. Types I and II, although differing in spatial distribution, share the common property of temporal independence and represent stochastic sensor noise. Both can be effectively suppressed through temporal aggregation and denoising strategies, and therefore are unified as the imaging noise source [14]. Type III corresponds to the illumination source, representing a slowly varying global bias that can be corrected through illumination estimation and temporal consistency constraints. Type IV forms the motion source, representing structured temporal displacement that requires explicit correspondence modeling or motion compensation rather than noise suppression [16]. Importantly, these sources cannot be further merged without losing restoration specificity: illumination variation and motion disturbance, although both temporally correlated, require fundamentally different processing strategies, while stochastic imaging noise cannot be handled through temporal modeling designed for structured degradation [15]. Therefore, TSDR adopts a three-source decomposition consisting of imaging noise, illumination, and motion, providing a compact yet physically meaningful representation of low-light video degradation.

Guided by this framework, TSD-Net instantiates the tri-source decomposition through a Wavelet Feature Router (WFR) for frequency-aware source routing, TCA for global cross-frame correspondence modeling, and a Degradation Prior Estimator (DPE) for source prior estimation. These components feed three parallel specialized branches (ISPN, NDPN, and MCPN), whose outputs are progressively re-integrated by SGRF following the physically motivated restoration order. Each branch is trained against a source-specific target as described above, so that the three-source separation established in the feature space is preserved in the outputs the fusion stage receives.

Our main contributions are as follows:

- We reformulate low-light video enhancement as a divide-and-conquer denoising problem under low-light scenarios and propose the TSDR framework. This framework decomposition extends beyond existing two-source paradigms.

- We identify and empirically demonstrate a failure mode of parallel restoration architectures: when multiple source-specific branches share a single reconstruction target, the architecture degenerates into redundant copies regardless of how the degradation is decomposed upstream. We show that penalizing output similarity does not resolve this, and instead offer a per-branch supervision design whose three targets have distinct optima.

- We design TSD-Net, an end-to-end architecture that instantiates the TSDR principle through wavelet-based subband routing, RWKV-style linear-complexity temporal correspondence, and three dedicated source-specific branches (ISPN, NDPN, MCPN) connected by a physically motivated stage-wise fusion (SGRF) that prevents noise amplification at its root.

- We introduce a two-phase progressive training schedule with perceptual decoupling to resolve gradient conflicts among competing restoration objectives.

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

For the motion source, the quantity to be corrected *is* the displacement, so the natural target is a neighbouring frame brought into the center frame's coordinate system:

$$X^{\text{align}}_t = \mathcal{W}_{i \to t}(I_i), \quad i \in \Omega. \tag{7}$$

Supervising MCPN on $X^{\text{align}}_t$ makes its output the alignment residual rather than a full reconstruction.

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
