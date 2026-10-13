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

A low-light video window is a set of $T = 2k+1$ frames $\{I_{t-k}, \dots, I_t, \dots, I_{t+k}\}$ centered at $I_t$, and each frame is produced from a clean signal $X_t$ by two physical processes. Illumination attenuates the signal, and the sensor adds noise whose variance grows with the signal:

$$I_t(p) = L_t(p) \cdot X_t(p) + n_t(p), \quad n_t(p) \sim \mathcal{N}\big(0,\; \sigma_s^2 \cdot L_t(p) \cdot X_t(p) + \sigma_r^2\big), \tag{1}$$

where $L_t(p) \in (0, 1]$ is the attenuation at pixel $p$ and $\sigma_s^2$, $\sigma_r^2$ are the shot and read noise parameters, following Wei *et al.* [2], Foi *et al.* [14], and Feng *et al.* [9]. Motion adds a displacement between frames, so the content at $p$ in frame $t'$ comes from $p + \mathbf{d}_{t \to t'}(p)$ in frame $t$:

$$I_{t'}(p) \approx L_{t'}(p) \cdot X_t\big(p + \mathbf{d}_{t \to t'}(p)\big) + n_{t'}(p). \tag{2}$$

Together, these processes define a joint decomposition of the observation into its three sources:

$$I_t = \mathcal{D}\big(\mathcal{I}(X_t),\; \mathcal{N}_t,\; \mathcal{M}_t\big), \tag{3}$$

where $\mathcal{I}(X_t)$, $\mathcal{N}_t$, and $\mathcal{M}_t$ are the illumination, noise, and motion sources.

The question is how to group these processes for restoration, and the answer depends on how each one behaves across frames. Two questions sort every degradation: does it persist from one frame to the next, and does it act globally over the image or locally on a region? Crossing the answers gives four canonical types [14, 15]. Read noise is global and redrawn at every frame, and shot noise is local and redrawn at every frame, with strength set by the signal. Both are removed by temporal averaging, so they form one imaging-noise source. Illumination variation is global and persistent, a slowly changing bias over the clip. Motion is local and persistent, a displacement that must be compensated, because averaging it would only smear the moving object.

Each source therefore calls for a different operation: averaging over frames for noise, a smooth global estimate for illumination, and alignment for motion. Any two-source grouping forces one operation onto a source it does not fit. Pooling noise with motion averages misaligned frames and produces ghosting. Pooling illumination with motion estimates a bias from content that is still moving. Pooling illumination with noise estimates the gain from the noisy signal, so the subsequent correction amplifies the noise it was meant to remove. This is why a two-source split is insufficient and the three sources must be kept apart.

Accordingly, TSDR assigns one branch to each source: BranchN for imaging noise, BranchL for illumination, and BranchM for motion. The sources are coupled, since illumination gain amplifies noise and motion breaks the frame correspondence that denoising relies on, so the branches draw on one shared temporal evidence base. A shared encoder and a MatrixRWKV temporal module produce clip-level context, and the Triple Query TCA gives each branch its own query over three clip statistics: the frame mean, the spatially smoothed mean, and the peak inter-frame difference. V7RFusion then combines the three branch outputs with per-pixel softmax weights over the center frame. Sharing evidence while separating outputs is the design goal, and Section 3.2 explains how the outputs are made to differ.

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
