# TSD-Net: Triple Source Decoupled Restoration for Low-Light Video Enhancement

Anonymous Submission

Anonymous affiliation

## 1 Introduction

Low-light video enhancement (LLVE) is a fundamental task in computer vision, aiming to recover visibility, suppress noise, and maintain temporal coherence for videos captured under insufficient illumination. Limited photon counts yield a low signal-to-noise ratio, and brightening the frames amplifies the embedded noise in proportion [2]. Since sensor noise is realized independently in each frame, per-frame enhancement also introduces temporal flicker [15]. Brightness recovery, noise suppression, and temporal consistency are therefore coupled in every low-light video, which makes LLVE considerably harder than its single-image counterpart, and joint denoising and enhancement has been studied as a remedy [5].

To tackle these challenges, many deep-learning based methods for LLVE have been proposed, which can be grouped by how they organize these demands. Sequential methods [4, 6] first estimate and correct illumination with a Retinex-style decomposition, then apply explicit denoising to the brightened result. Implicit methods [7, 8] embed all of the restoration within a single video network, typically a Transformer whose multi-frame attention learns brightening, denoising, and alignment jointly. Parallel methods [10, 12, 16] assign the degradation to dedicated branches, each responsible for one component of the restoration.

Each family carries a distinct limitation. The cascaded design of sequential methods accumulates errors, since imperfect illumination correction is amplified during denoising. Implicit methods achieve strong results, but their performance depends on the quality and generalization of the training data. Parallel methods offer the most natural match to the multi-source structure of low-light video, yet their decompositions are built on the imaging domain [11], frequency or SNR statistics [12], or the diffusion timestep [13]. These criteria partition the degradation by signal representation rather than by its temporal behavior, so sources with incompatible restoration requirements can fall into the same branch.

To address the above limitation of existing degradation decomposition strategies, we propose Triple Source Decoupled Network (TSD-Net) for low-light video enhancement. The proposed Triple Source Decoupled Restoration (TSDR) framework decomposes the entangled low-light video degradation into three physically meaningful sources—illumination, imaging noise, and motion—according to their distinct spatio-temporal characteristics. Illumination is a slowly varying global bias, imaging noise is independent across frames and is suppressed by temporal averaging, and motion is a structured displacement that must be aligned rather than averaged.

For the first challenge, effectively decoupling highly coupled degradation sources so that each branch specializes in its own source, architectural separation alone is not sufficient. When every branch regresses the same clean frame, the objective admits a single minimizer that every branch can reach independently, so nothing in the optimization asks the branches to differ. We therefore enforce the decomposition through supervision. Decoupled Target Supervision (DTS) gives each branch a target built from the input clip itself: the noise branch is trained toward the temporal average of shift-aligned frames; the illumination branch toward an illumination-normalized image, obtained by dividing that average by its large-kernel low-pass estimate; and the motion branch toward the average of the outermost frames of the window after alignment. These targets have distinct optima by construction, so the branches separate under the objective itself rather than under an auxiliary penalty.

For the second challenge, capturing global temporal interactions among the degradation sources without the cost of full spatio-temporal attention, we model frame-level context with MatrixRWKV, whose matrix-valued recurrent state and data-dependent decay encode inter-frame relations at linear complexity. This context gates three statistics that define a shared key-value in the Triple Query TCA: the frame mean (noise prior), the spatially smoothed mean (illumination prior), and the peak inter-frame difference (motion prior). Three independently learned queries attend to this shared key-value, so the source-specific features differ in what they read from a common view of the clip.

Guided by this framework, TSD-Net instantiates the tri-source decomposition as follows. A shared encoder with PixelTemporalAttentionSimple, which applies 3D convolutions over a three-frame window, produces center-aligned spatial features. MatrixRWKV then models the frame-level context, and the Triple Query TCA produces three source-aware features. BranchN, BranchL, and BranchM, each a lightweight NAFBlock network with a source-specific design (channel attention, a Retinex-style illumination head with learnable gamma, and flow-based deformable alignment with confidence gating), restore noise, illumination, and motion respectively. Finally, V7RFusion combines the three outputs with per-pixel softmax weights and a learned residual gate on the center frame. The full model contains 3.74M parameters.

We evaluate TSD-Net on [N] benchmarks, including SDSD-indoor and SDSD-outdoor [3], SMID [17], DID [18], and DAVIS [19]. TSD-Net achieves [X] dB PSNR on SDSD-indoor, outperforming the strongest competing method [CITE] by [X] dB with [X]$\times$ fewer parameters, as shown in Fig. 1.

Our main contributions are summarized as follows:

- We propose TSD-Net, a low-light video enhancement network built on the TSDR framework, which restores illumination, imaging noise, and motion with three source-specific branches.
- We propose Decoupled Target Supervision (DTS), which assigns each branch a distinct target constructed from the input clip, so that source specialization follows from the objective rather than from auxiliary penalties.
- We design a MatrixRWKV-based temporal module and a Triple Query TCA that gate shared statistical key-values with frame-level context, modeling global inter-frame interactions at linear complexity. To our knowledge, this is the first RWKV-based model for low-light video enhancement.
- We conduct extensive experiments on [N] benchmarks, where TSD-Net outperforms the strongest competing method by [X] dB PSNR with [X]$\times$ fewer parameters.

## 3 Proposed Method

### 3.1 Tri-Source Decoupled Restoration Framework

We formulate low-light video enhancement as a **tri-source decoupled restoration** problem. Let $\{I_{t-k}, \dots, I_t, \dots, I_{t+k}\}$ denote a sliding window of $T = 2k+1$ consecutive low-light frames centered at the target frame $I_t$. Each observed frame $I_t$ is modeled as a composite of three underlying degradation sources acting upon a latent clean signal $X_t$:

$$I_t = \mathcal{D}\big(\underbrace{\mathcal{I}(X_t)}_{\text{illumination}},\; \underbrace{\mathcal{N}_t}_{\text{noise}},\; \underbrace{\mathcal{M}_t}_{\text{motion}}\big),$$

where $\mathcal{I}(\cdot)$ represents the illumination degradation operator that suppresses exposure and disrupts brightness constancy, $\mathcal{N}_t$ denotes the sensor noise term, and $\mathcal{M}_t$ accounts for motion-induced corruption including blur and misalignment. The composite operator $\mathcal{D}$ captures the entangled interaction of these three sources in the observed image.

Adopting the signal-dependent noise formation model of Wei *et al.* [2] and Foi *et al.* [14], and the dark-frame noise modeling of Feng *et al.* [9], the observed pixel intensity at spatial location $p$ satisfies

$$I_t(p) = L_t(p) \cdot X_t(p) + n_t(p), \quad n_t(p) \sim \mathcal{N}\big(0,\; \sigma_s^2 \cdot L_t(p) \cdot X_t(p) + \sigma_r^2\big),$$

where $L_t(p) \in (0, 1]$ is the illumination attenuation at pixel $p$, and $\sigma_s^2$, $\sigma_r^2$ are the shot noise and read noise parameters. Across frames, motion introduces a displacement $\mathbf{d}_{t \to t'}(p)$ between frames $t$ and $t'$:

$$I_{t'}(p) \approx L_{t'}(p) \cdot X_t\big(p + \mathbf{d}_{t \to t'}(p)\big) + n_{t'}(p).$$

The sources are identified along two dimensions: temporal independence, whether a degradation is independent across consecutive frames, and spatial selectivity, whether it is globally distributed or spatially localized. Their combination yields four canonical degradation types [14, 15]: globally distributed and temporally independent degradation, corresponding to read noise; spatially selective and temporally independent degradation, corresponding to signal-dependent shot noise that dominates in dark regions; globally distributed and temporally correlated degradation, corresponding to illumination variation across frames; and spatially localized and temporally correlated degradation, corresponding to motion. The first two types share temporal independence and are both suppressed by temporal aggregation, so they form the imaging noise source. The third is the illumination source, a slowly varying global bias. The fourth is the motion source, a structured displacement that requires correspondence or compensation rather than noise suppression.

The three sources exhibit fundamentally different temporal statistics, as summarized in Table 1:

| Property | Illumination $\mathcal{I}$ | Noise $\mathcal{N}$ | Motion $\mathcal{M}$ |
|:---|:---|:---|:---|
| **Temporal type** | Bias (slowly varying) | Variance (i.i.d. across $t$) | Displacement (structured) |
| **Frame correlation** | $\|L_t - L_{t-1}\| \to 0$ | $\mathbb{E}[\mathcal{N}_t \cdot \mathcal{N}_{t'}] \approx 0,\; t \neq t'$ | $\mathbf{d}_{t \to t'} \neq 0$ in moving regions |
| **Optimal strategy** | Spatial smoothness prior | Temporal averaging under correspondence | Alignment / compensation |
| **Amplification chain** | Amplifies noise via $\nabla\mathcal{I}$ | Degrades $\mathcal{I}$ estimation | Breaks correspondence for denoising |

This taxonomy reveals why **a two-source split is insufficient**: any grouping that merges two of the three sources into a single branch conflates degradation types with incompatible temporal statistics.

- **Illumination + Noise** vs. Motion (Retinex-style): The "reflectance" component $R = I / L$ absorbs both noise and clean texture. Noise amplification during the $I/L$ division makes subsequent denoising harder, as the noise variance becomes spatially dependent on the potentially inaccurate $L$ estimate.
- **Illumination + Motion** vs. Noise: The "structural" component absorbs both slowly varying illumination and structured motion, preventing the model from exploiting the temporal smoothness of $L$ for flicker suppression.
- **Noise + Motion** vs. Illumination: The "residual" component absorbs both random noise and structured motion. Since noise benefits from temporal averaging while motion requires compensation, the unified component must simultaneously denoise *and* deblur, objectives whose optimal temporal processing strategies are diametrically opposed.

The sources are also coupled. Illumination enhancement amplifies noise; motion breaks the temporal correspondence that multi-frame denoising relies on; and illumination estimation is misled by both noise and motion artifacts. These couplings call for decoupling at the level of representation: each source needs its own processing path, while all paths draw on the same evidence from the clip.

We realize this decomposition with one dedicated branch per source. BranchN is a NAFBlock network with channel attention for the imaging noise source. BranchL adds an illumination prediction head with a Retinex-style decomposition and a learnable gamma for the illumination source. BranchM adds optical flow estimation, deformable alignment, and confidence gating for the motion source. All three branches read features from a shared encoder and temporal module, together with a Triple Query TCA that gives each branch its own query over three clip statistics: the frame mean, the spatially smoothed mean, and the peak inter-frame difference.

### 3.2 Source-Decoupled Supervision

The framework above specifies how the three sources are separated in the representation. It does not specify what each branch should reconstruct, and the architecture alone does not determine this. We first show that the conventional supervision answers the question with a single shared target, and then give each source its own target.

**Why a shared target fails.** Let the three branches output $Y_N, Y_L, Y_M \in \mathbb{R}^{3 \times H \times W}$ and let $X_t$ be the ground truth. Under the conventional design, each branch is supervised against the same target:

$$\mathcal{L}_{\text{branch}} = \sum_{b \in \{N, L, M\}} w_b \, \| Y_b - X_t \|_1. \tag{4}$$

This objective separates across branches, and each term attains zero at $Y_b = X_t$. Consequently the global minimizer has $Y_N = Y_L = Y_M = X_t$: the optimum of the decomposed architecture is an undecomposed one. Output similarity is therefore not an artifact to be suppressed but the exact optimum of the stated objective, which makes penalizing it work against the objective. A decorrelation penalty $\lambda |\cos(Y_a, Y_b)|$ does not remedy this. In our first decorrelation runs, the penalty was minimized by collapsing one branch to a constant map, which has no mean-subtracted correlation with any other output.

The difficulty is not that the branches receive indistinguishable features: the three query paths produce measurably different attention features. The difficulty is that they are asked for indistinguishable outputs.

**Source-specific targets.** We replace the shared target with three targets constructed from the input sequence itself, requiring no additional annotation. Each target is the quantity that its source's optimal estimator would produce. The targets are computed from the input rather than predicted by the network: alignment uses a global integer translation found by grid search within $\pm 4$ pixels, not the learned flow estimator. A target that depends on network parameters would drift as the network trains, and the supervision would no longer be a fixed signal.

For the imaging noise source, the relevant property is temporal independence, so the optimal estimator is a temporal average, computed over neighbouring frames aligned to the center frame. Otherwise averaging reintroduces motion as ghosting:

$$X^{\text{avg}}_t = \frac{1}{|\Omega|+1}\Big( I_t + \sum_{i \in \Omega} \mathcal{W}_{i \to t}\big(I_i\big) \Big), \tag{5}$$

where $\Omega$ is the set of neighbouring frames in the window and $\mathcal{W}_{i \to t}$ aligns frame $i$ to the center frame. BranchN is supervised on $X^{\text{avg}}_t$, which is noise-reduced by construction while retaining texture.

For the illumination source, the relevant property is that illumination is low-frequency and globally distributed, so it can be isolated by a large-kernel low-pass filter. Letting $\hat{L}_t = \mathrm{LPF}(X^{\text{avg}}_t)$ be the estimated illumination field, we define the illumination-normalized target

$$X^{\text{illu}}_t = \mathrm{clip}\big(X^{\text{avg}}_t \oslash \hat{L}_t,\ 0,\ 1\big), \tag{6}$$

and supervise BranchL on it. This target preserves relative structure and color while removing absolute exposure, so the branch is not required to reproduce texture that the other branches already handle.

For the motion source, the quantity to be corrected is the displacement, so the natural target is built from the outermost neighbouring frames brought into the center frame's coordinate system:

$$X^{\text{align}}_t = \tfrac{1}{2}\big( \mathcal{W}_{t-k \to t}(I_{t-k}) + \mathcal{W}_{t+k \to t}(I_{t+k}) \big), \tag{7}$$

rather than an average over all of $\Omega$. The outermost frames carry the largest displacement and therefore the strongest alignment signal. Averaging over all neighbours would make Eq. (7) almost indistinguishable from Eq. (5), with a mean absolute difference of $0.012$ on our data, leaving BranchM and BranchN with effectively the same target. Using the two outermost frames raises this difference to $0.127$ on random clips and $0.039$ on clips containing motion.

The three targets differ by construction. $X^{\text{avg}}_t$ averages out independent noise, $X^{\text{illu}}_t$ removes absolute exposure, and $X^{\text{align}}_t$ keeps the outermost frames without averaging over all neighbours. Their optima no longer coincide, so the branches cannot all converge to one function, and the fusion stage receives three genuinely distinct candidates. The training objective keeps an $\ell_1$ term between the fused output $O_t$ and $X_t$ and adds the three branch terms:

$$\mathcal{L} = \| O_t - X_t \|_1 + \sum_{b \in \{N, L, M\}} w_b \, \| Y_b - T_b \|_1 + \lambda \, \mathcal{L}_{\text{ortho}}, \qquad T_N = X^{\text{avg}}_t,\ T_L = X^{\text{illu}}_t,\ T_M = X^{\text{align}}_t, \tag{8}$$

where $\mathcal{L}_{\text{ortho}}$ is the feature-level orthogonality term.

This formulation also clarifies the scope of the TSDR claim. The three-source decomposition states how degradation should be *partitioned*, and the supervision design is what makes that partition *operational*. A decomposition whose parts are trained to agree is not a decomposition in any functional sense, and the two ingredients are needed together.

---

## Reference

[1] Cai, Yuanhao, et al. "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2023.

[2] Wei, Kaixuan, et al. "A physics-based noise formation model for extreme low-light raw denoising." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2020.

[3] Wang, Ruixing, et al. "Seeing dynamic scene in the dark: A high-quality video dataset with mechatronic alignment." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2021.

[4] Yang, Wenhan, et al. "Sparse gradient regularized deep retinex network for robust low-light image enhancement." IEEE Transactions on Image Processing 30 (2021): 2072-2086.

[5] Li, Huaqiu, et al. "Interpretable Unsupervised Joint Denoising and Enhancement for Real-World low-light Scenarios." International Conference on Learning Representations (ICLR). 2025.

[6] Wei, Chen, et al. "Deep Retinex Decomposition for Low-Light Enhancement." Proceedings of the British Machine Vision Conference (BMVC). 2018.

[7] Liang, Jingyun, et al. "VRT: A video restoration transformer." IEEE Transactions on Image Processing 33 (2024): 2171-2182.

[8] Liang, Jingyun, et al. "Recurrent video restoration transformer with guided deformable attention." Advances in Neural Information Processing Systems 35 (2022): 378-393.

[9] Feng, Hansen, et al. "Learning Physics-Informed Noise Models from Dark Frames for Low-Light Raw Image Denoising." IEEE Transactions on Pattern Analysis and Machine Intelligence 48.4 (2026): 3952-3969.

[10] Li, Yuezhou, et al. "Perceptual decoupling with heterogeneous auxiliary tasks for joint low-light image enhancement and deblurring." IEEE Transactions on Multimedia 26 (2024): 6663-6675.

[11] Li, Ruikang, et al. "DualDn: Dual-Domain Denoising via Differentiable ISP." European Conference on Computer Vision (ECCV). Springer, 2024, pp. 160-177.

[12] Xu, Xiaogang, et al. "SNR-Aware Low-Light Image Enhancement." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022, pp. 17693-17703.

[13] Liu, Jiawei, et al. "Residual denoising diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2024.

[14] Foi, Alessandro, et al. "Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data." IEEE Transactions on Image Processing 17.10 (2008): 1737-1754.

[15] Zhu, Lingyu, et al. "Temporally Consistent Enhancement of Low-Light Videos via Spatial-Temporal Compatible Learning." International Journal of Computer Vision 132 (2024): 4703-4723.

[16] Xu, Xiaogang, et al. "Low-Light Video Enhancement via Spatial-Temporal Consistent Decomposition." Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI). 2025, pp. 2134-2142.

[17] Chen, Chen, et al. "Seeing motion in the dark." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2019.

[18] Fu, Huiyuan, et al. "Dancing in the Dark: A Benchmark towards General Low-light Video Enhancement." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2023, pp. 12831-12840.

[19] Perazzi, Federico, et al. "A benchmark dataset and evaluation methodology for video object segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016.
