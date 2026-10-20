# TSD-Net: Triple Source Decoupled Restoration for Low-Light Video Enhancement

Anonymous Submission

Anonymous affiliation

> 草稿说明: v6 以 IntroDraftv5 的 §1 与 §3.1 为框架, 按 v8 最新实现与命名 (docs/v8/08-命名对照.md) 改写。
> 带 `[待核实]` 的段落尚未定稿。

## 1 Introduction

Low-light video enhancement (LLVE) is a fundamental task in computer vision, aiming to recover visibility, suppress noise, and maintain temporal coherence for videos captured under insufficient illumination. Limited photon counts yield a low signal-to-noise ratio, and brightening the frames amplifies the embedded noise in proportion [1]. Since sensor noise is realized independently in each frame, per-frame enhancement also introduces temporal flicker [2]. Brightness recovery, noise suppression, and temporal consistency are therefore coupled in every low-light video, which makes LLVE considerably harder than its single-image counterpart.

To tackle these challenges, many deep-learning based methods for LLVE have been proposed, which can be grouped by how they organize these demands. Sequential methods [3, 4] first estimate and correct illumination with a Retinex-style decomposition, then apply explicit denoising to the brightened result. Implicit methods [5, 6] embed all of the restoration within a single video network, typically a Transformer whose multi-frame attention learns brightening, denoising, and alignment jointly. Parallel methods [7, 8, 9] assign the degradation to dedicated branches, each responsible for one component of the restoration.

Each family carries a distinct limitation. The cascaded design of sequential methods accumulates errors, since imperfect illumination correction is amplified during denoising. Implicit methods achieve strong results, but their performance depends on the quality and generalization of the training data. Parallel methods offer the most natural match to the multi-source structure of low-light video, yet their decompositions are built on the imaging domain [10], frequency or SNR statistics [8], or the diffusion timestep [11]. These criteria partition the degradation by signal representation rather than by its temporal behavior, so sources with incompatible restoration requirements can fall into the same branch.

To address the above limitation of existing degradation decomposition strategies, we propose Triple Source Decoupled Network (TSD-Net) for low-light video enhancement. The proposed Triple Source Decoupled Restoration (TSDR) framework decomposes the entangled low-light video degradation into three physically meaningful sources—illumination, imaging noise, and motion—according to their distinct spatio-temporal characteristics. Illumination is a slowly varying global bias, imaging noise is independent across frames and is suppressed by temporal averaging, and motion is a structured displacement that averaging across frames would smear, so its branch reads the center frame alone.

Guided by this framework, TSD-Net instantiates the tri-source decomposition as follows. A FrameEncoder encodes each frame with shared weights into features at three scales. A TemporalAggregator, composed of RWKV blocks, weights the frames and aggregates them top-down into a clip-level representation at full resolution. Its temporal weighting is computed recurrently across frames, so the cost of modeling inter-frame interactions grows linearly with the number of frames rather than quadratically as in full spatio-temporal attention. TaskQueryAttention sets a separate query for each branch at this resolution. NBranch, LBranch, and MBranch, each a lightweight network with a source-specific design, restore noise, illumination, and motion respectively. Finally, the Stage-wise Guided Restoration and Fusion (SGRF) block combines the three outputs over the center frame to produce the restored frame.

Our main contributions are summarized as follows:

- We propose the Triple Source Decoupled Restoration (TSDR) theory, which decomposes low-light video degradation into illumination, imaging noise, and motion according to their distinct spatio-temporal behavior, and assigns each source its own restoration branch.
- Guided by TSDR, we design TSD-Net, a low-light video enhancement network that restores the three sources with three source-specific branches.
- We design a task-gated query RWKV block that lets each branch query the clip, modeling global inter-frame interactions at linear complexity in the number of frames. To our knowledge, this is the first RWKV-based model for low-light video enhancement.

## 2 Related Work

[结构占位, 待写]

### 2.1 Low-light video enhancement

待写: 按 Sequential / Implicit / Parallel 三类方法综述, 对应 §1 第二段。

### 2.2 Degradation decomposition in low-light enhancement

待写: 已有分解准则 (成像域、频率或 SNR 统计、扩散时间步) 及其按信号表示而非时序行为划分的问题, 对应 §1 第三段。

### 2.3 RWKV-style linear recurrence for vision

待写: RWKV 系列模型在视觉低层任务中的使用, 对应 §3.1 中的时序模块与 RWKV 块。

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

Each source therefore calls for a different operation: averaging over frames for noise, a smooth global estimate for illumination, and an operation that does not average across frames for motion. Any two-source grouping forces one operation onto a source it does not fit. Pooling noise with motion averages misaligned frames and produces ghosting. Pooling illumination with motion estimates a bias from content that is still moving. Pooling illumination with noise estimates the gain from the noisy signal, so the subsequent correction amplifies the noise it was meant to remove. This is why a two-source split is insufficient and the three sources must be kept apart.

### 3.2 Overall Architecture

TSD-Net instantiates TSDR as a five-component pipeline. The FrameEncoder encodes each frame independently into features at three scales; it extracts spatial evidence and mixes no frames. The TemporalAggregator is the only component that combines frames: it produces a multi-frame representation $F^1_{\text{multi}}$ at full resolution and, through a bypass, the center-frame representation $F^1_{\text{single}}$, so that both aggregated and unaggregated evidence are available. TaskQueryAttention produces one attended feature per source, $F_N$, $F_L$, and $F_M$, by having each source query the representation it can be estimated from. NBranch, LBranch, and MBranch map these features to the restored components $Y_N$, $Y_L$, and $Y_M$, each restoring one source only. SGRF composes the three components with the center frame into the restored frame $O_t$ while keeping their roles separate. Sections 3.3 to 3.6 describe each component in turn.

### 3.3 Frame Encoding and Temporal Aggregation

The FrameEncoder and the TemporalAggregator divide the feature construction into per-frame spatial encoding and cross-frame aggregation, respectively.

**FrameEncoder.** Each frame of the window is mapped to features at three scales by one encoder with weights shared across frames; the sharing follows from the observation model, in which every frame is produced by the same sensor and scene model. At each scale, one EncoderRWKVBlock processes the features while preserving their shape, and the block is looped a fixed number of times, so the encoding depth is decoupled from the parameter count. The encoder sees one frame at a time and performs no cross-frame operation.

**TemporalAggregator.** The TemporalAggregator is the only stage that mixes frames. It first applies one further EncoderRWKVBlock to the coarsest-scale features of each frame. A frame descriptor is then computed per frame and fed to a TemporalMatrixRWKVBlock, which maintains a matrix-valued state along the frame dimension, $S_t = S_{t-1}\,\mathrm{diag}(w_t) + v_t^\top k_t$, with a data-dependent decay $w_t = \exp\!\big(-\exp(b + \mathrm{LoRA}(x_t))\big)$ predicted from the current descriptor. The block outputs a weight for each frame, and the recurrence keeps the cost of modeling inter-frame interactions linear in the number of frames. The weighted features are summed at the coarsest scale and fused top-down through the remaining scales, yielding the multi-frame representation $F^1_{\text{multi}}$ at full resolution. A bypass carries the center frame through the same top-down path without the frame weighting, yielding $F^1_{\text{single}}$ and $F^2_{\text{single}}$.

### 3.4 Task-Gated Query Attention

The query, key-value, and residual base of each branch are chosen according to which representation its source can be estimated from.

**Imaging noise.** Under the conditions in Section 3.1, the noise in each frame is zero-mean given the signal and independent across frames. The aggregate $F^1_{\text{multi}}$ therefore approximates the stable content, and the difference $F^1_{\text{single}} - F^1_{\text{multi}}$ approximates the instantaneous residual of the center frame, which carries the noise. The noise branch queries with this difference, takes the aggregate as the key-value that holds the stable content, and adds the attended result to the aggregate. The difference is an approximation rather than the noise itself, because the noise variance depends on the signal and the difference is not normalized by it.

**Illumination.** Over a short window, the attenuation $L_t$ is approximately constant. The aggregate $F^1_{\text{multi}}$ therefore provides a stable estimate of the illumination cues, and the illumination branch uses it as the query. The key-value and the residual base come from $F^1_{\text{single}}$: the correction acts on the content of the center frame, so the attended result is added back to the representation it was retrieved from.

**Motion.** Motion displaces content between frames, and the displacement persists across the window. At the frame rates typical of capture, however, the displacement between consecutive frames is large and irregular, so cross-frame motion structure is difficult to learn and to exploit. The motion branch therefore stays within the center frame, using $F^1_{\text{single}}$ as its query, key-value, and residual base.

Table 1 summarizes the query, key-value, and residual base of each branch and the assumption of Section 3.1 it relies on.

Table 1. Query, key-value, and residual base of each branch.

| Branch | Source | Assumption from Section 3.1 | Query | Key-value | Residual base |
|---|---|---|---|---|---|
| NBranch | Imaging noise | Noise zero-mean and independent across frames; aligned content | $F^1_{\text{single}} - F^1_{\text{multi}}$ | $F^1_{\text{multi}}$ | $F^1_{\text{multi}}$ |
| LBranch | Illumination | $L_t$ approximately constant over the window | $F^1_{\text{multi}}$ | $F^1_{\text{single}}$ | $F^1_{\text{single}}$ |
| MBranch | Motion | Displacement not averaged across frames | $F^1_{\text{single}}$ | $F^1_{\text{single}}$ | $F^1_{\text{single}}$ |

TaskQueryAttention realizes these assignments with one QueryRWKVBlock per branch, all at full resolution. In a QueryRWKVBlock, the two sources play different roles in a cross RWKV gate [14]: the query source produces the receptance and gate signals, which decide what to read, while the key-value source produces the key and value, which decide where the reading lands. The receptance, key, and value enter a bidirectional WKV scan over four spatial directions, whose recurrent state carries global context at a cost linear in the sequence length; an OmniShift operation supplies the local context that the recurrence does not see. The scan output is modulated by the gate signal, normalized, and added to the branch's residual base through a learned scale, producing the branch features $F_N$, $F_L$, and $F_M$.

### 3.5 Source-Specific Branches

Each branch is a lightweight network that maps its attended features to a full-resolution output.

**NBranch** reads $F_N$ and outputs $Y_N$, the stable content of the window plus the estimated noise correction of the center frame.

**LBranch** follows the Retinex form: it estimates an illumination map from $F_L$ through a sigmoid, with the center frame $X_t$ as an additional input, and produces the illumination-corrected output $Y_L$ at full resolution.

**MBranch** reads $F_M$ alone and produces $Y_M$, restoring the local structure of the center frame. It involves no cross-frame operation and estimates no displacement.

### 3.6 Output Composition

SGRF combines the three branch outputs with the center frame into the restored frame $O_t$. The outputs enter as additive corrections that complement one another across frequency bands, followed by a local refinement over the center frame. Because the composition is additive, a gradient reaching $O_t$ is passed to the three outputs equally.

### 3.7 Loss Function

待写

## 4 Experiments

[结构占位, 待写]

Table 2. Quantitative comparison of model complexity analysis.

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
