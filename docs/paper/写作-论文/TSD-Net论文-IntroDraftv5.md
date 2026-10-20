# TSD-Net: Triple Source Decoupled Restoration for Low-Light Video Enhancement

Anonymous Submission

Anonymous affiliation

## 1 Introduction

Low-light video enhancement (LLVE) is a fundamental task in computer vision, aiming to recover visibility, suppress noise, and maintain temporal coherence for videos captured under insufficient illumination. Limited photon counts yield a low signal-to-noise ratio, and brightening the frames amplifies the embedded noise in proportion [1]. Since sensor noise is realized independently in each frame, per-frame enhancement also introduces temporal flicker [2]. Brightness recovery, noise suppression, and temporal consistency are therefore coupled in every low-light video, which makes LLVE considerably harder than its single-image counterpart.

To tackle these challenges, many deep-learning based methods for LLVE have been proposed, which can be grouped by how they organize these demands. Sequential methods [3, 4] first estimate and correct illumination with a Retinex-style decomposition, then apply explicit denoising to the brightened result. Implicit methods [5, 6] embed all of the restoration within a single video network, typically a Transformer whose multi-frame attention learns brightening, denoising, and alignment jointly. Parallel methods [7, 8, 9] assign the degradation to dedicated branches, each responsible for one component of the restoration.

Each family carries a distinct limitation. The cascaded design of sequential methods accumulates errors, since imperfect illumination correction is amplified during denoising. Implicit methods achieve strong results, but their performance depends on the quality and generalization of the training data. Parallel methods offer the most natural match to the multi-source structure of low-light video, yet their decompositions are built on the imaging domain [10], frequency or SNR statistics [8], or the diffusion timestep [11]. These criteria partition the degradation by signal representation rather than by its temporal behavior, so sources with incompatible restoration requirements can fall into the same branch.

To address the above limitation of existing degradation decomposition strategies, we propose Triple Source Decoupled Network (TSD-Net) for low-light video enhancement. The proposed Triple Source Decoupled Restoration (TSDR) framework decomposes the entangled low-light video degradation into three physically meaningful sources—illumination, imaging noise, and motion—according to their distinct spatio-temporal characteristics. Illumination is a slowly varying global bias, imaging noise is independent across frames and is suppressed by temporal averaging, and motion is a structured displacement that must be aligned rather than averaged.

For the first challenge, effectively decoupling highly coupled degradation sources so that each branch specializes in its own source, architectural separation alone is not sufficient. When every branch regresses the same clean frame, the objective admits a single minimizer that every branch can reach independently, so nothing in the optimization asks the branches to differ. We therefore enforce the decomposition through supervision. Decoupled Target Supervision (DTS) gives each branch a target built from the input clip itself: the noise branch is trained toward the temporal average of shift-aligned frames; the illumination branch toward an illumination-normalized image, obtained by dividing that average by its large-kernel low-pass estimate; and the motion branch toward the average of the outermost frames of the window after alignment. These targets have distinct optima by construction, so the branches separate under the objective itself rather than under an auxiliary penalty.

For the second challenge, capturing global temporal interactions among the degradation sources without the cost of full spatio-temporal attention, we model frame-level context with MatrixRWKV, whose matrix-valued recurrent state and data-dependent decay encode inter-frame relations at linear complexity. This context gates three statistics that define a shared key-value in the Triple Query TCA: the frame mean (noise prior), the spatially smoothed mean (illumination prior), and the peak inter-frame difference (motion prior). Three independently learned queries attend to this shared key-value, so the source-specific features differ in what they read from a common view of the clip.

Guided by this framework, TSD-Net instantiates the tri-source decomposition as follows. A shared encoder with PixelTemporalAttentionSimple, which applies 3D convolutions over a three-frame window, produces center-aligned spatial features. MatrixRWKV then models the frame-level context, and the Triple Query TCA produces three source-aware features. BranchN, BranchL, and BranchM, each a lightweight network with a source-specific design, restore noise, illumination, and motion respectively. Finally, the Stage-wise Guided Restoration and Fusion (SGRF) block combines the three outputs with per-pixel softmax weights and a learned residual gate on the center frame.

Our main contributions are summarized as follows:

- We propose TSD-Net, a low-light video enhancement network built on the TSDR framework, which restores illumination, imaging noise, and motion with three source-specific branches.
- We propose Decoupled Target Supervision (DTS), which assigns each branch a distinct target constructed from the input clip, so that source specialization follows from the objective rather than from auxiliary penalties.
- We design a MatrixRWKV-based temporal module and a Triple Query TCA that gate shared statistical key-values with frame-level context, modeling global inter-frame interactions at linear complexity. To our knowledge, this is the first RWKV-based model for low-light video enhancement.

## 3 Proposed Method

### 3.1 Tri-Source Decoupled Restoration Framework

A low-light video window is a set of $T = 2k+1$ frames $\{I_{t-k}, \dots, I_t, \dots, I_{t+k}\}$ centered at $I_t$, and each frame is produced from a clean signal $X_t$ by two physical processes. Illumination attenuates the signal, and the sensor adds noise whose variance grows with the signal:

$$I_t(p) = L_t(p) \cdot X_t(p) + n_t(p), \quad n_t(p) \sim \mathcal{N}\big(0,\; \sigma_s^2 \cdot L_t(p) \cdot X_t(p) + \sigma_r^2\big), \tag{1}$$

where $L_t(p) \in (0, 1]$ is the attenuation at pixel $p$ and $\sigma_s^2$, $\sigma_r^2$ are the shot and read noise parameters, following Wei *et al.* [1], Foi *et al.* [12], and Feng *et al.* [13]. Motion adds a displacement between frames, so the content at $p$ in frame $t'$ comes from $p + \mathbf{d}_{t \to t'}(p)$ in frame $t$:

$$I_{t'}(p) \approx L_{t'}(p) \cdot X_t\big(p + \mathbf{d}_{t \to t'}(p)\big) + n_{t'}(p). \tag{2}$$

Together, these processes define a joint decomposition of the observation into its three sources:

$$I_t = \mathcal{D}\big(\mathcal{I}(X_t),\; \mathcal{N}_t,\; \mathcal{M}_t\big), \tag{3}$$

where $\mathcal{I}(X_t)$, $\mathcal{N}_t$, and $\mathcal{M}_t$ are the illumination, noise, and motion sources.

The question is how to group these processes for restoration, and the answer depends on how each one behaves across frames. Two questions sort every degradation: does it persist from one frame to the next, and does it act globally over the image or locally on a region? Crossing the answers gives four canonical types [12, 2]. Read noise is global and redrawn at every frame, and shot noise is local and redrawn at every frame, with strength set by the signal. Both are removed by temporal averaging, so they form one imaging-noise source. Illumination variation is global and persistent, a slowly changing bias over the clip. Motion is local and persistent, a displacement that must be compensated, because averaging it would only smear the moving object.

Each source therefore calls for a different operation: averaging over frames for noise, a smooth global estimate for illumination, and alignment for motion. Any two-source grouping forces one operation onto a source it does not fit. Pooling noise with motion averages misaligned frames and produces ghosting. Pooling illumination with motion estimates a bias from content that is still moving. Pooling illumination with noise estimates the gain from the noisy signal, so the subsequent correction amplifies the noise it was meant to remove. This is why a two-source split is insufficient and the three sources must be kept apart.

Accordingly, TSDR assigns one branch to each source: BranchN for imaging noise, BranchL for illumination, and BranchM for motion. The sources are coupled, since illumination gain amplifies noise and motion breaks the frame correspondence that denoising relies on, so the branches draw on one shared temporal evidence base. A shared encoder and a MatrixRWKV temporal module produce clip-level context, and the Triple Query TCA gives each branch its own query over three clip statistics: the frame mean, the spatially smoothed mean, and the peak inter-frame difference. V7RFusion then combines the three branch outputs with per-pixel softmax weights over the center frame. Sharing evidence while separating outputs is the design goal, and Section 3.2 explains how the outputs are made to differ.

---

## References

[1] Wei, Kaixuan, et al. "A physics-based noise formation model for extreme low-light raw denoising." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2020.

[2] Zhu, Lingyu, et al. "Temporally Consistent Enhancement of Low-Light Videos via Spatial-Temporal Compatible Learning." International Journal of Computer Vision 132 (2024): 4703-4723.

[3] Yang, Wenhan, et al. "Sparse gradient regularized deep retinex network for robust low-light image enhancement." IEEE Transactions on Image Processing 30 (2021): 2072-2086.

[4] Wei, Chen, et al. "Deep Retinex Decomposition for Low-Light Enhancement." Proceedings of the British Machine Vision Conference (BMVC). 2018.

[5] Liang, Jingyun, et al. "VRT: A video restoration transformer." IEEE Transactions on Image Processing 33 (2024): 2171-2182.

[6] Liang, Jingyun, et al. "Recurrent video restoration transformer with guided deformable attention." Advances in Neural Information Processing Systems 35 (2022): 378-393.

[7] Li, Yuezhou, et al. "Perceptual decoupling with heterogeneous auxiliary tasks for joint low-light image enhancement and deblurring." IEEE Transactions on Multimedia 26 (2024): 6663-6675.

[8] Xu, Xiaogang, et al. "SNR-Aware Low-Light Image Enhancement." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022, pp. 17693-17703.

[9] Xu, Xiaogang, et al. "Low-Light Video Enhancement via Spatial-Temporal Consistent Decomposition." Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI). 2025, pp. 2134-2142.

[10] Li, Ruikang, et al. "DualDn: Dual-Domain Denoising via Differentiable ISP." European Conference on Computer Vision (ECCV). Springer, 2024, pp. 160-177.

[11] Liu, Jiawei, et al. "Residual denoising diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2024.

[12] Foi, Alessandro, et al. "Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data." IEEE Transactions on Image Processing 17.10 (2008): 1737-1754.

[13] Feng, Hansen, et al. "Learning Physics-Informed Noise Models from Dark Frames for Low-Light Raw Image Denoising." IEEE Transactions on Pattern Analysis and Machine Intelligence 48.4 (2026): 3952-3969.
