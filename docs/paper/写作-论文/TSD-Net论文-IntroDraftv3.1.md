# TSD-Net: Triple Source Decoupled Restoration for Low-Light Video Enhancement

Anonymous Submission

Anonymous affiliation

## 1 Introduction

Low-light video enhancement (LLVE) seeks to recover visibility, suppress noise, and preserve temporal coherence in videos captured under insufficient illumination. The problem is intrinsically harder than its single-image counterpart. Limited photon counts yield a low signal-to-noise ratio, and brightening the frames amplifies the embedded noise together with the signal [2]. Because sensor noise is realized independently in each frame, naive per-frame processing also produces flicker [3]. An effective method must therefore reconcile three demands that pull in different directions: raise illumination, remove noise, and keep consecutive frames consistent.

Existing learning-based approaches can be grouped by how they organize these demands. Sequential methods first brighten and then denoise [4, 5, 6]; the design is transparent, but errors in the illumination stage are amplified before the denoiser sees them. Implicit methods fold all three demands into a single spatio-temporal network, typically a Transformer [7, 8]; this avoids a fixed order, but treats every temporal variation uniformly, so brightness drift, sensor noise, and object motion—which call for smoothing, averaging, and alignment respectively—receive the same treatment. Parallel methods decompose the degradation into components and assign each to a dedicated branch [9, 10]. In principle this is the most suitable organization for a coupled degradation, since each branch can apply the estimator its component requires. Our work belongs to this family.

A parallel design, however, rests on two prerequisites that are easy to conflate. The first is a *decomposition* that matches the degradation: the components must differ in a way that different estimators can exploit. The second is a *training signal* that actually drives each branch to its assigned component. Prior work has concentrated on the first prerequisite and has largely taken the second for granted. The central claim of this paper is that the second prerequisite is the binding one: without it, a well-chosen decomposition is not realized by training, regardless of how the branches are wired.

On the first prerequisite, we adopt a three-source decomposition into imaging noise, illumination, and motion. These sources are distinguished by their temporal behavior rather than by their spectral content or their position in the imaging pipeline. Noise is independent across frames and is best suppressed by temporal averaging; illumination is a slowly varying global bias and is best estimated by spatial smoothing; motion is a structured displacement and must be compensated by alignment rather than averaged away. Existing decompositions—by raw versus sRGB domain [11], by frequency band [9, 12], or by diffusion timestep [13]—do not partition along this axis, and therefore place components with incompatible temporal statistics into the same branch. Section 3.1 derives the three sources from a taxonomy of low-light degradation and shows that they cannot be merged further without losing this distinction.

On the second prerequisite, we report an observation that is invisible in the final metric and, to our knowledge, has not been documented. In the usual setting, every branch is supervised toward the same enhanced frame. The resulting objective separates across branches and has a single global optimum that each branch can reach on its own, so the optimizer has no incentive to divide the work. Each branch converges to a complete reconstruction, and the architecture degenerates into redundant copies of one enhancer. We confirm this on our own model: the three branch outputs are nearly collinear, and removing any one of them while renormalizing the fusion weights leaves the final PSNR essentially unchanged (Section 4). The final output is the same whether or not the branches specialize, which is precisely why the problem persists unnoticed.

This observation locates the difficulty in the supervision rather than in the representation. The branch-specific features do differ measurably; what they are asked to produce does not. The distinction determines which remedies can work. Penalizing similarity between branch outputs, a natural first attempt, does not create a division of labor: it rewards dissimilarity, and a branch that outputs a constant image satisfies the penalty at no cost while contributing nothing. We observe exactly this collapse in training.

We therefore enforce the decomposition at the level of the targets. Each branch is supervised toward a distinct target constructed from the input sequence, chosen to be what the optimal estimator for that source would produce: a motion-compensated temporal average for the noise branch, an illumination-normalized image for the illumination branch, and neighboring frames warped to the center time instant for the motion branch. The targets require no additional annotation and, by construction, have distinct optima. The branches can no longer all converge to the same function, and the conflict among them is resolved in the objective itself rather than by balancing competing loss terms.

We instantiate this design in TSD-Net. A shared encoder produces per-frame features; a wavelet feature router separates illumination-dominant from structure-dominant content; a temporal correspondence module establishes cross-frame affinity in linear time without optical flow; and three branches, trained on the targets above, restore noise, illumination, and motion in parallel. Their outputs are re-integrated in a stage-wise order—denoise, deblur, then brighten—so that noise is removed before it can be amplified by brightening.

Our contributions are as follows.

- We formulate low-light video enhancement as tri-source decoupled restoration, decomposing the degradation into imaging noise, illumination, and motion according to their temporal statistics, and show that this decomposition is both sufficient and irreducible.

- We identify a failure mode of parallel restoration: when branches share a reconstruction target, they converge to redundant copies regardless of the upstream decomposition, and similarity penalties do not repair this. We propose per-branch supervision on source-specific targets with distinct optima as the remedy.

- We present TSD-Net, which realizes the formulation with wavelet-based routing, linear-time temporal correspondence, three source-specific branches, and physically ordered stage-wise fusion, together with a progressive training schedule that resolves gradient conflicts among the branches.

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
