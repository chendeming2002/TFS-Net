# TSD-Net: Triple Source Decoupled Restoration for Low-Light Video Enhancement

Anonymous Submission

Anonymous affiliation

> 草稿说明: v7 以 IntroDraftv6 为底稿, 方法部分整体按 v8.5 当前实现重写
> (TemporalAggregator_G + SGRFv10 逐段约束融合 + kvmod 查询头,
> 对应配置 `configs/v8_5/g_sgrf10_qrb_kvmod_30ep.yaml`),
> 写作顺序为先 §3.3–§3.7 部件, 再据以整理 §3.1 与 §3.2, 最后改 §1。
> §3.4 按 kvmod 配置定稿; gatectx 对照训练结束后若结论变化, 只需改 §3.4 一处。
> §2 与 §4 仍为占位; Table 2 的 TSD-Net 行仍是旧 fullkv 配置的数字 (1.175M),
> 待按当前配置 (1.242M) 重测后更新。

## 1 Introduction

Low-light video enhancement (LLVE) is a fundamental task in computer vision, aiming to recover visibility, suppress noise, and maintain temporal coherence for videos captured under insufficient illumination. Limited photon counts yield a low signal-to-noise ratio, and brightening the frames amplifies the embedded noise in proportion [1]. Since sensor noise is realized independently in each frame, per-frame enhancement also introduces temporal flicker [2]. Brightness recovery, noise suppression, and temporal consistency are therefore coupled in every low-light video, which makes LLVE considerably harder than its single-image counterpart.

To tackle these challenges, many deep-learning based methods for LLVE have been proposed, which can be grouped by how they organize these demands. Sequential methods [3, 4] first estimate and correct illumination with a Retinex-style decomposition, then apply explicit denoising to the brightened result. Implicit methods [5, 6] embed all of the restoration within a single video network, typically a Transformer whose multi-frame attention learns brightening, denoising, and alignment jointly. Parallel methods [7, 8, 9] assign the degradation to dedicated branches, each responsible for one component of the restoration.

Each family carries a distinct limitation. The cascaded design of sequential methods accumulates errors, since imperfect illumination correction is amplified during denoising. Implicit methods achieve strong results, but their performance depends on the quality and generalization of the training data. Parallel methods offer the most natural match to the multi-source structure of low-light video, yet their decompositions are built on the imaging domain [10], frequency or SNR statistics [8], or the diffusion timestep [11]. These criteria partition the degradation by signal representation rather than by its temporal behavior, so sources with incompatible restoration requirements can fall into the same branch.

To address the above limitation of existing degradation decomposition strategies, we propose Triple Source Decoupled Network (TSD-Net) for low-light video enhancement. The proposed Triple Source Decoupled Restoration (TSDR) framework decomposes the entangled low-light video degradation into three physically meaningful sources—illumination, imaging noise, and motion—according to their distinct spatio-temporal characteristics. Illumination is a slowly varying global bias, imaging noise is independent across frames and is suppressed by temporal averaging, and motion is a structured displacement that averaging across frames would smear, so its branch reads the center frame alone.

Guided by this framework, TSD-Net instantiates the tri-source decomposition as follows. A FrameEncoder encodes each frame with shared weights into features at three scales. A TemporalAggregator averages the frames at every scale and fuses the result top-down into a clip-level representation at full resolution; the center frame passes through the same top-down path as a second stream with shared parameters, so that both aggregated and unaggregated evidence are available at full resolution. TaskQueryAttention sets a separate query for each branch at this resolution. NBranch, LBranch, and MBranch, each a lightweight network with a source-specific design, restore noise, illumination, and motion respectively. Finally, the Stage-wise Guided Restoration and Fusion (SGRF) block composes the three outputs stage by stage—noise, then illumination, then motion—into the restored frame. All global context in the network is carried by bidirectional WKV scans whose cost grows linearly with the sequence length; cross-frame combination is a plain average.

Our main contributions are summarized as follows:

- We propose the Triple Source Decoupled Restoration (TSDR) theory, which decomposes low-light video degradation into illumination, imaging noise, and motion according to their distinct spatio-temporal behavior, and assigns each source its own restoration branch.
- Guided by TSDR, we design TSD-Net, a low-light video enhancement network that restores the three sources with three source-specific branches and composes them stage by stage.
- We design a task-gated query RWKV block that lets each branch read the clip representation through a cross RWKV gate with query-conditioned keys, carrying global spatial context at linear complexity in the sequence length. To our knowledge, this is the first RWKV-based model for low-light video enhancement.

## 2 Related Work

[结构占位, 待写]

### 2.1 Low-light video enhancement

待写: 按 Sequential / Implicit / Parallel 三类方法综述, 对应 §1 第二段。

### 2.2 Degradation decomposition in low-light enhancement

待写: 已有分解准则 (成像域、频率或 SNR 统计、扩散时间步) 及其按信号表示而非时序行为划分的问题, 对应 §1 第三段。

### 2.3 RWKV-style linear recurrence for vision

待写: RWKV 系列模型在视觉低层任务中的使用, 对应 §3.3 中的空间 RWKV 块与 §3.4 中的查询块。

## 3 Proposed Method

### 3.1 Tri-Source Decoupled Restoration Framework

A low-light video window is a set of $T = 2k+1$ frames $\{I_{t-k}, \dots, I_t, \dots, I_{t+k}\}$ centered at $I_t$, and each frame is produced from a clean signal $X_t$ by two physical processes. Illumination attenuates the signal, and the sensor adds noise whose variance grows with the signal:

$$I_t(p) = L_t(p) \cdot X_t(p) + n_t(p), \quad n_t(p) \sim \mathcal{N}\big(0,\; \sigma_s^2 \cdot L_t(p) \cdot X_t(p) + \sigma_r^2\big), \tag{1}$$

where $L_t(p) \in (0, 1]$ is the attenuation at pixel $p$ and $\sigma_s^2$, $\sigma_r^2$ are the shot and read noise parameters, following Wei *et al.* [1], Foi *et al.* [12], and Feng *et al.* [13]. Motion adds a displacement between frames, so the content at $p$ in frame $t'$ comes from $p + \mathbf{d}_{t \to t'}(p)$ in frame $t$:

$$I_{t'}(p) \approx L_{t'}(p) \cdot X_t\big(p + \mathbf{d}_{t \to t'}(p)\big) + n_{t'}(p). \tag{2}$$

Together, these processes motivate a conceptual decomposition of the observation into three sources. It organizes the restoration problem rather than giving an exact equation for the observation:

$$I_t = \mathcal{D}\big(\mathcal{I}(X_t),\; \mathcal{N}_t,\; \mathcal{M}_t\big), \tag{3}$$

where $\mathcal{I}(X_t)$, $\mathcal{N}_t$, and $\mathcal{M}_t$ are the illumination, noise, and motion sources.

The question is how to group these processes for restoration, and the answer depends on how each one behaves across frames. Two questions sort every degradation: does it persist from one frame to the next, and does it act globally over the image or locally on a region? Crossing the answers gives four canonical types [12, 2]. Read noise is global and redrawn at every frame, and shot noise is local and redrawn at every frame, with strength set by the signal. Under aligned content and a signal that is stable over the window, both are zero-mean across frames given the signal, so temporal averaging reduces them, and they form one imaging-noise source. Illumination is global and, over a short window, approximately constant, so it can be estimated from the window as a slowly varying bias. Motion is local and persistent, a displacement whose averaging across frames would only smear the moving object.

Each source therefore calls for a different operation: averaging over frames for noise, a smooth global estimate for illumination, and a single-frame readout for motion. Any two-source grouping forces one operation onto a source it does not fit. Pooling noise with motion averages misaligned frames and produces ghosting. Pooling illumination with motion estimates a bias from content that is still moving. Pooling illumination with noise estimates the gain from the noisy signal, so the subsequent correction amplifies the noise it was meant to remove. This is why a two-source split is insufficient and the three sources must be kept apart.

The three sources also carry a dependency beyond the partition: the illumination sets the absolute strength of the shot noise, and both act on content that motion displaces. A restoration that respects this dependency removes the noise first, corrects the exposure of the denoised result, and repairs the local structure last. Section 3.6 builds this order into the composition of the three branches.

### 3.2 Overall Architecture

TSD-Net instantiates TSDR as a five-component pipeline. The FrameEncoder encodes each frame independently into features at three scales; it extracts spatial evidence frame by frame. The TemporalAggregator is the only component that combines frames: it averages the window at every scale and runs one shared-parameter top-down path twice—once over the window-mean stream, producing the multi-frame representation $F^1_{\text{multi}}$ at full resolution, and once over the center-frame stream, producing the center-frame representation $F^1_{\text{single}}$ together with its second-scale intermediate $F^2_{\text{single}}$—so that both aggregated and unaggregated evidence are available at full resolution. TaskQueryAttention produces one attended feature per source, $F_N$, $F_L$, and $F_M$, by having each source query the representation it can be estimated from. NBranch, LBranch, and MBranch map these features to the restored components $Y_N$, $Y_L$, and $Y_M$, each restoring one source only. SGRF composes the three components stage by stage, in the order noise, illumination, motion: each stage reads the partially restored result of the previous stages and adds its own gated correction, producing the restored frame $O_t$. Sections 3.3 to 3.6 describe each component in turn.

### 3.3 Frame Encoding and Temporal Aggregation

The FrameEncoder and the TemporalAggregator divide the feature construction into per-frame spatial encoding and cross-frame aggregation, respectively.

**FrameEncoder.** Each frame of the window is mapped to features at three scales by one encoder with weights shared across frames; the sharing follows from the observation model, in which every frame is produced by the same sensor and scene model. At each scale, one EncoderRWKVBlock processes the features while preserving their shape, and as in Re-WKV [15] the block is looped a fixed number of times, so the encoding depth is decoupled from the parameter count. An EncoderRWKVBlock is a spatial RWKV block that reads the same feature map as both its query and its key-value sources: an OmniShift operation [15] supplies local context, a bidirectional WKV scan [16] carries global context at a cost linear in the number of pixels, and a channel-mixing stage follows the scan. The encoder sees one frame at a time and operates entirely within it.

**TemporalAggregator.** The TemporalAggregator is the only stage that mixes frames. It first applies one further EncoderRWKVBlock to the coarsest-scale features of each frame. At every scale, the features then split into two streams: the mean over all $T$ frames, which Section 3.1 identifies as the estimator that suppresses the imaging noise, and the center frame alone. A single top-down path processes both streams with the same parameters: each of its two stages doubles the resolution with a 3×3 convolution and pixel shuffle, then fuses the result with the corresponding scale's features by concatenation and a 3×3 convolution with normalization and a SiLU activation. The mean stream yields the multi-frame representation $F^1_{\text{multi}}$ at full resolution; the center-frame stream yields $F^1_{\text{single}}$, with its second-scale intermediate $F^2_{\text{single}}$.

Writing the shared top-down mapping as $P$, the two outputs are

$$F^1_{\text{multi}} = P\Big(\tfrac{1}{T}\textstyle\sum_{t'} f_{t'}\Big), \qquad F^1_{\text{single}} = P\big(f_t\big), \tag{4}$$

so the window mean and the center frame are expressed under the same learned mapping, and their difference—used as a query in Section 3.4—compares like with like. The mean includes the center frame and is taken independently at each scale; the only cross-scale interaction is the shared top-down fusion.

### 3.4 Task-Gated Query Attention

The query, key-value, and residual base of each branch are chosen according to which representation its source can be estimated from.

**Imaging noise.** Under the conditions in Section 3.1, the noise in each frame is zero-mean given the signal and independent across frames, so the window mean suppresses it and the difference $F^1_{\text{single}} - F^1_{\text{multi}}$ carries the instantaneous residual of the center frame, which is dominated by the noise. The noise branch queries with this difference, takes the aggregate as the key-value that holds the stable content, and adds the attended result to the aggregate. The difference is an approximation of the noise rather than the noise itself, for three reasons: the mapping $P$ in Eq. (4) is nonlinear, so the zero-mean property of the pixel noise does not transfer exactly to the feature space; the noise variance depends on the signal, and the difference is not normalized by it; and content displaced by motion survives the mean and enters the difference, attenuated by the center frame's share $1/T$ of the mean.

**Illumination.** Over a short window, the attenuation $L_t$ is approximately constant. The aggregate $F^1_{\text{multi}}$ therefore provides a stable estimate of the illumination cues, and the illumination branch uses it as the query. The key-value and the residual base come from $F^1_{\text{single}}$: the correction acts on the content of the center frame, so the attended result is added back to the representation it was retrieved from.

**Motion.** Motion displaces content between frames, and the displacement persists across the window. At the frame rates typical of capture, however, the displacement between consecutive frames is large and irregular, so cross-frame motion structure is difficult to learn and to exploit. The motion branch therefore stays within the center frame, using $F^1_{\text{single}}$ as its query, key-value, and residual base.

Table 1 summarizes the query, key-value, and residual base of each branch and the assumption of Section 3.1 it relies on.

Table 1. Query, key-value, and residual base of each branch.

| Branch | Source | Assumption from Section 3.1 | Query | Key-value | Residual base |
|---|---|---|---|---|---|
| NBranch | Imaging noise | Noise zero-mean and independent across frames; aligned content | $F^1_{\text{single}} - F^1_{\text{multi}}$ | $F^1_{\text{multi}}$ | $F^1_{\text{multi}}$ |
| LBranch | Illumination | $L_t$ approximately constant over the window | $F^1_{\text{multi}}$ | $F^1_{\text{single}}$ | $F^1_{\text{single}}$ |
| MBranch | Motion | Displacement not averaged across frames | $F^1_{\text{single}}$ | $F^1_{\text{single}}$ | $F^1_{\text{single}}$ |

TaskQueryAttention realizes these assignments with one QueryRWKVBlock per branch, all at full resolution. A QueryRWKVBlock is a cross RWKV gate [14] in which the two sources play different roles: the query source produces the receptance and gate signals, which decide what to read, while the key-value source produces the key and value, which decide where the reading lands. The query also conditions the keys directly: a descriptor pooled from the query predicts a per-channel scale and shift applied to the keys before the scan, zero-initialized so that training starts from the unmodulated gate. The keys and values enter a bidirectional WKV scan [16], whose recurrent state carries global context at a cost linear in the sequence length; the scan output is multiplied by the receptance, normalized, and gated, then projected and passed through a channel-mixing stage, with an OmniShift [15] supplying local context at the input of each projection. The result is added to the branch's residual base through a zero-initialized learned scale and normalized, producing the branch features $F_N$, $F_L$, and $F_M$.

### 3.5 Source-Specific Branches

Each branch is a lightweight network that maps its attended features to a full-resolution output.

**NBranch** reads $F_N$ and outputs $Y_N$, the noise component of the center frame. The network stacks two NAF blocks with a channel-attention stage. Its supervision target (Section 3.7) is the noise share of the mid- and high-frequency bands of the clean signal.

**LBranch** follows the Retinex form: it estimates a single-channel illumination map $L_t$ from $F_L$ through a sigmoid, takes the center frame $X_t$ as the signal to be corrected, and produces the illumination-corrected output $Y_L = R_t \cdot L_t^{\gamma}$ with $R_t = X_t / L_t$ and a learnable exponent $\gamma$, plus a zero-initialized residual compensation, clamped to $[0, 1]$. Its supervision target is the low-frequency band of the clean signal.

**MBranch** reads $F_M$ alone and produces $Y_M$, restoring the local structure of the center frame. All its operations are intra-frame, and the restoration is direct. Its supervision target is the motion share of the mid- and high-frequency bands.

### 3.6 Output Composition

SGRF composes the three components stage by stage in the order noise, illumination, motion, following the dependency among the sources stated in Section 3.1: the noise strength is set by the illumination, and both act on content that motion displaces, so the composition removes the noise first, corrects the exposure of the denoised estimate, and repairs the local structure last. Each stage reads the partially restored result of the previous stages and adds its own gated correction. Because the three supervision targets of Section 3.7 sum to the clean signal, the accumulation starts from $X_0 = 0$:

$$X_k = X_{k-1} + s_k \, G_k \odot b_k(\Delta_k), \quad k \in \{N, L, M\}, \qquad O_t = \mathrm{clamp}(X_M,\, 0,\, 1), \tag{5}$$

where $\Delta_k$ is the correction of stage $k$ and the three factors $s_k$, $G_k$, $b_k$ shape how it enters. Writing $\tilde{F}_N = F_N$ and $\tilde{F}_k = F_k + \mathrm{proj}_k(X_{k-1})$ for $k \in \{L, M\}$—the current state mapped back to the feature space through a zero-initialized 1×1 convolution $\mathrm{proj}_k$—each stage reads $\tilde{F}_k$ in place of the raw branch feature.

**Correction.** $\Delta_N$ and $\Delta_M$ are the branch outputs passed through one per-channel 3×3 convolution, identity-initialized; $\Delta_L$ is the Retinex reconstruction of the illumination branch, which reads the center frame $X_t$ directly.

**Gate.** $G_k = \sigma(\mathrm{gate}_k([X_{k-1}, \tilde{F}_k]))$ is a per-pixel gate predicted from the current state and the stage features. The gate network is zero-initialized, so every stage starts at $G_k \equiv 0.5$; a regularizer (Section 3.7) keeps the gates away from saturation, where the sigmoid derivative vanishes and the stage would lose its ability to adjust.

**Bound and amplitude.** $b_k(\Delta) = \delta_k \tanh(\Delta / \delta_k)$ soft-limits the correction, and $s_k = 1 + (r_k/2)\tanh(a_k)$ is a bounded scalar amplitude initialized at 1. Each stage has its own limits, matched to the magnitude of its supervision target. The illumination target carries the full image content, so its stage skips the soft limit—its Retinex output is already clamped to $[0, 1]$—and takes a wider amplitude range ($r_L = 0.8$). The noise and motion targets are zero-mean bands whose mean magnitudes measure about 0.20 and 0.05, so their stages take $\delta_k$ at three times these values (0.60 and 0.16) and a narrower amplitude range ($r_N = r_M = 0.5$).

At initialization every stage reduces to half of its branch output added in sequence—$\mathrm{proj}_k = 0$, $s_k = 1$, $G_k \equiv 0.5$, and the per-channel convolutions are identities—so training starts from a plain additive composition and learns the staging jointly with the branches. The gradient path is unbroken across stages: the reconstruction loss on $O_t$ reaches every stage through the accumulation.

### 3.7 Loss Function

The training objective combines a final reconstruction term, per-branch supervision, a feature orthogonality penalty, and a gate regularizer:

$$\mathcal{L} = \big\lVert O_t - \mathrm{GT} \big\rVert_1 \;+\; \sum_{k} w_k(e)\, \frac{\lVert Y_k - S_k \rVert_1}{\overline{|S_k|}} \;+\; \lambda_{\text{o}}\, \mathcal{L}_{\text{ortho}} \;+\; \lambda_{\text{g}}\, \mathcal{L}_{\text{gate}}. \tag{6}$$

**Branch targets.** Each branch regresses its own target, built by splitting the clean center frame into frequency bands with two average-pooling low-pass filters $\mathrm{LP}_{31}$ and $\mathrm{LP}_{5}$:

$$S_L = \mathrm{LP}_{31}(\mathrm{GT}), \qquad S_N = \tfrac{1}{2}Q + \tfrac{4}{5}R, \qquad S_M = \tfrac{1}{2}Q + \tfrac{1}{5}R, \tag{7}$$

where $Q = \mathrm{LP}_{5}(\mathrm{GT}) - \mathrm{LP}_{31}(\mathrm{GT})$ is the mid band and $R = \mathrm{GT} - \mathrm{LP}_{5}(\mathrm{GT})$ is the high band. The mid and high bands are shared between the noise and motion targets in different proportions, and the three targets sum to $\mathrm{GT}$ exactly, which is what allows the composition of Eq. (5) to start from zero. The band magnitudes differ by up to two orders, so each branch loss is normalized by the mean absolute value $\overline{|S_k|}$ of its target, keeping the three contributions comparable. The branch weights ramp linearly over the first five epochs to $w_N = w_L = w_M = 0.3$.

**Feature orthogonality.** $\mathcal{L}_{\text{ortho}}$ is the mean absolute cosine similarity over the three pairs of branch features $(F_N, F_L, F_M)$, weighted by $\lambda_{\text{o}} = 0.01$.

**Gate regularization.** $\mathcal{L}_{\text{gate}} = \tfrac{1}{3}\sum_{k} \mathrm{mean}\big((G_k - 0.5)^2\big)$ keeps the stage gates of Eq. (5) in their adjustable range, weighted by $\lambda_{\text{g}} = 0.05$.

## 4 Experiments

[结构占位, 待写]

Table 2. Quantitative comparison of model complexity analysis. (TSD-Net 行为旧配置数字, 待按当前配置重测)

| Type | Model | Param. (M) ↓ | FLOPs (G) ↓ | Memory (GB) ↓ | Latency (ms) ↓ | FPS ↑ |
|---|---|---|---|---|---|---|
| Image | Retinexformer | <u>1.61</u> | **15.57** | — | — | — |
| Image | DarkIR | 12.96 | <u>26.91</u> | 0.31 | 13.5 | 74.1 |
| Video | SMOID | 5.43 | 67.70 | <u>0.20</u> | **6.8** | **146.8** |
| Video | SDSDNet | 3.95 | 34.48 | **0.13** | <u>8.5</u> | <u>118.2</u> |
| Video | Diff-L | 22.08 | 88.92 | — | — | — |
| Video | UDU-Net | 18.34 | 1098.00 | 0.67 | 485.3 | 2.1 |
| Video | TCE-Net | 9.06 | 99.14 | 0.43 | 177.8 | 5.6 |
| Video | TSD-Net (Ours) | **1.175** | 99.51 | 0.467 | 31.0 | 32.2 |

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

[14] Yin, Zhuowen, Chengru Li, and Xingbo Dong. "Video RWKV: Video Action Recognition Based RWKV." arXiv preprint arXiv:2411.05636 (2024).

[15] Yang, Zhiwen, et al. "Restore-RWKV: Efficient and Effective Medical Image Restoration with RWKV." arXiv preprint arXiv:2407.11087 (2024).

[16] Duan, Yuchen, et al. "Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures." arXiv preprint arXiv:2403.02308 (2024).
