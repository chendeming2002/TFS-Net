# DPMambaIR: All-in-One Image Restoration via Degradation-Aware Prompt State Space Model

**作者**: Zhanwen Liu, Sai Zhou, Yuchao Dai, Yang Wang†, Yisheng An, Xiangmo Zhao

**来源**: arXiv:2504.17732v3 [cs.CV] 5 Feb 2026

---

## Abstract

All-in-One image restoration aims to address multiple image degradation problems using a single model, offering a more practical and versatile solution compared to designing dedicated models for each degradation type. Existing approaches typically rely on Degradation-specific models or coarse-grained degradation prompts to guide image restoration. However, they lack fine-grained modeling of degradation information and face limitations in balancing multi-task conflicts. To overcome these limitations, we propose DPMambaIR, a novel All-in-One image restoration framework that introduces a fine-grained degradation extractor and a Degradation-Aware Prompt State Space Model (DP-SSM). The DP-SSM leverages the fine-grained degradation features captured by the extractor as dynamic prompts, which are then incorporated into the state space modeling process. This enhances the model's adaptability to diverse degradation types, while a complementary High-Frequency Enhancement Block (HEB) recovers local high-frequency details. Extensive experiments on a mixed dataset containing seven degradation types show that DPMambaIR achieves the best performance, with 27.69 dB and 0.893 in PSNR and SSIM, respectively. These results highlight the potential and superiority of DPMambaIR as a unified solution for All-in-One image restoration.

**Keywords**: All-in-One, Image Restoration, State Space Model

---

## I. Introduction

Image restoration is a fundamental task in computer vision, aiming to recover high-quality images from degraded inputs. Recent advances in deep learning have significantly improved performance in degradation-specific restoration tasks, such as image denoising [1], [2], deblurring [3], deraining [4], dehazing [5], [6], and low-light enhancement [7]. However, traditional methods typically rely on degradation-specific models explicitly tailored to individual degradation types. These approaches often require prior knowledge of the degradation type, which limits their practicality in real-world scenarios, such as autonomous driving and nighttime surveillance, where diverse and unknown degradations frequently coexist.

To address this limitation, recent studies explore All-in-One image restoration frameworks that unify multiple restoration tasks into a single model. Existing methods can be broadly categorized into Mixture-of-Experts (MoE)-based and Prompt-based approaches. MoE-based methods, such as MoFME [8], dynamically route tasks to specialized experts using an uncertainty-aware router. However, these methods suffer from high computational costs and lack explicit degradation modeling, limiting adaptability to unseen degradations. Alternatively, Prompt-based methods like OneRestore [9] guide restoration using degradation prompts, integrating scene descriptors via cross-attention mechanisms.

However, existing prompt-based strategies [9], [10] suffer from fundamental limitations in how degradation information is represented and exploited. A key shortcoming lies in their reliance on static priors, where degradations are encoded as discrete class labels or fixed semantic tokens. Such categorical representations are inherently inadequate for modeling the continuous nature of real-world degradations and their spatial variability, for example, differentiating between light haze and dense, spatially accumulating fog. Furthermore, most of these methods follow an additive fusion paradigm in which degradation cues are treated merely as auxiliary information appended to visual features. This mechanism is intrinsically limited, as the underlying network architecture remains governed by fixed convolutional kernels or static attention patterns. Consequently, the model is compelled to address heterogeneous restoration demands, such as suppressing high-frequency noise while simultaneously enhancing global contrast, within a rigid geometric structure. This conflict ultimately degrades performance across diverse and complex degradation scenarios.

To address these challenges, we propose DPMambaIR, a novel All-in-One framework that shifts the design philosophy from conventional feature fusion to explicit dynamic parameter modulation. Our approach is built upon a fine-grained degradation extractor and a Degradation-Aware Prompt State Space Model (DP-SSM). Unlike methods that rely on discrete degradation labels, the proposed extractor employs a reconstruction objective to regress continuous degradation embeddings, enabling it to encode nuanced information regarding both degradation type and severity. These embeddings are utilized to directly modulate the core State Space Model parameters (Δ, B, C). Modulating the discretization step Δ effectively alters the integration step size of the underlying ordinary differential equation, allowing the model to realize distinct continuous dynamics conditioned on degradation characteristics. Specifically, the network learns to assign smaller Δ values to induce a high-inertia state favorable for smoothing high-frequency noise, whereas larger values are generated to enter a high-gain regime that amplifies weak signals in low-light conditions. Concurrently, the input projection matrix B is modulated to control how strongly external observations drive the latent state transitions, while the output matrix C is adjusted to inject global degradation priors. Furthermore, we observe that jointly optimizing heterogeneous degradations often induces an implicit bias toward low-frequency structures, making the consistent recovery of fine details challenging. To alleviate this issue, we incorporate a lightweight High-frequency Enhancement Block (HEB) that complements the proposed dynamic modulation and facilitates the restoration of local textures.

Extensive experiments on a mixed dataset containing seven degradation types demonstrate the efficacy of DPMambaIR. As shown in Fig. 1, DPMambaIR achieves state-of-the-art performance in terms of PSNR and SSIM, outperforming existing All-in-One methods such as AdaIR and OneRestore. Moreover, DPMambaIR achieves competitive results across individual tasks, including deraining, low-light enhancement, deblurring, and dehazing, demonstrating its potential as an effective and robust All-in-One image restoration solution.

The contributions of this paper are summarized as follows:

- We propose a degradation extractor capable of capturing fine-grained, continuous degradation features from complex degraded images via a regression-based reconstruction objective.
- We design a Degradation-aware Prompt State Space Model (DP-SSM) that introduces a dynamic parameter modulation mechanism. By dynamically modulating the state-space evolution parameters based on degradation priors, it enhances the model's physical adaptability to diverse degradation dynamics.
- Extensive experiments on a mixed dataset containing seven types of degradation demonstrate that DPMambaIR achieves state-of-the-art performance on PSNR and SSIM metrics, outperforming existing methods. The results validate the effectiveness and robustness of the proposed approach.

---

## II. Related Work

### A. All-in-One Image Restoration

Image restoration remains a foundational challenge in computer vision, striving to reconstruct high-fidelity content from corrupted inputs. While traditional approaches, such as Dark Channel Prior [11] and Color Line Prior [12], relied on handcrafted heuristics that often struggle in complex scenarios, the advent of deep learning [13] has revolutionized the field. Significant strides have been made in specific domains, including super-resolution [14]–[16], denoising [1], [2], deblurring [3], [17], [18], deraining [4], [19], dehazing [5], [20], desnowing [21], [22], and low-light enhancement [7]. Leveraging CNN-based [23], Transformer-based [24], or diffusion-based [25] architectures, these methods excel at feature representation. However, their reliance on degradation-specific designs fundamentally limits their utility in real-world environments where degradations are diverse, compounded, and often unknown.

Recent research has consequently pivoted towards All-in-One frameworks that emphasize flexibility and unified processing. These methods generally employ multi-task learning strategies, such as Mixture-of-Experts (MoE) or prompt-guided adaptation. For instance, AirNet [26] utilizes contrastive learning to derive degradation representations, while MoFME [8] implements an uncertainty-aware router to dynamically dispatch features to specialized experts. Although effective, MoE architectures often incur high computational overhead and lack explicit physical modeling of the degradation. In parallel, prompt-based methods such as PromptIR [10], OneRestore [9], and DA-CLIP [27] integrate degradation cues, ranging from learned visual prompts to text embeddings, via cross-attention mechanisms. IDR [28] further explores meta-learning for decomposition, and AdaIR [29] adopts frequency domain decoupling for adaptive restoration.

Despite these advancements, fundamental limitations persist in the representation and exploitation of degradation information. Most prompt-based strategies [9], [10] rely on coarse-grained priors or discrete class labels, which fail to capture the continuous nature and spatial variability inherent in real-world images. Furthermore, they typically adopt an additive fusion paradigm where degradation cues are treated merely as auxiliary inputs. This mechanism is intrinsically limited because the underlying network parameters remain fixed, governed by static weights or attention patterns that lack the flexibility to adapt to heterogeneous restoration demands. In contrast, our DPMambaIR framework employs a regression-based extractor to derive fine-grained, continuous degradation embeddings. We depart from simple feature concatenation and utilize these embeddings to dynamically re-parameterize the State Space Model. This shift from additive fusion to explicit dynamic parameter modulation allows the system to adapt its evolution dynamics to the specific physical properties of the degradation.

### B. State Space Model based Image Restoration

Convolutional Neural Networks (CNNs) [13] and Transformers [24] have long dominated image restoration. While CNNs excel at capturing local high-frequency details, their limited receptive fields hinder the modeling of long-range dependencies essential for correcting global degradations like haze or uneven lighting. Transformers address this with global self-attention but suffer from quadratic computational complexity relative to image resolution.

To reconcile long-range modeling with efficiency, researchers have revisited the State Space Model (SSM) [30]. SSMs map a 1D input sequence x(t) to an output y(t) via a latent state h(t) ∈ ℝ^N, governed by the linear ordinary differential equation (ODE):

$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Dx(t) \quad (1)$$

where A, B, C, D are learnable parameters. For digital implementation, this continuous system is discretized with a step size Δ, yielding the recurrence:

$$h_i = \bar{A}h_{i-1} + \bar{B}x_i, \quad y_i = \bar{C}h_i + \bar{D}x_i \quad (2)$$

Here, the discretized matrices are defined as $\bar{A} = \exp(\Delta A)$ and $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$. This formulation allows for linear-complexity inference while maintaining a global receptive field.

The Vision Mamba [31] first adapted SSMs for computer vision, demonstrating that the Vision State Space Module could outperform Vision Transformers [32] with significantly lower computational costs. This success has sparked adoption across tasks including object detection [33], segmentation [34], and classification [35]. In image restoration, MambaIR [36] pioneered the use of SSMs to achieve a balance between efficiency and perceptual quality, inspiring subsequent variants [37]–[39].

However, applying standard SSMs to All-in-One restoration faces two primary limitations. First, **Limited Global Context Utilization**: The causal nature of standard scanning means the i-th pixel only accesses information from preceding pixels. While multi-directional scanning can alleviate this, it increases computation without always yielding proportional gains in low-level tasks [36]. Second, and more critically, **Lack of Degradation-Awareness**: Existing SSM-based restoration methods rely on fixed transition dynamics (A, B, C) or purely input-driven modulation. They lack an explicit mechanism to adapt the system's evolution rules based on external degradation priors (e.g., adjusting the integration step Δ for different noise levels). Consequently, their ability to generalize across the diverse degradation types encountered in All-in-One settings remains constrained.

Motivated by these gaps, we propose the Degradation-aware Prompt State Space Model (DP-SSM). Unlike prior works, our method utilizes fine-grained degradation embeddings to dynamically modulate the discretization step Δ, along with the state-space transition and output matrices. This design establishes a dynamics-aware mechanism that harmonizes global context modeling with degradation-specific physical guidance, significantly enhancing robustness in complex restoration scenarios.

---

## III. Method

### A. Problem Definition

All-in-One Image Restoration aims to address a wide range of image degradation types within a unified framework. Specifically, given a degraded image $I_D \in \mathbb{R}^{3 \times H \times W}$, where D denotes the degradation process, our goal is to restore the clean image $\hat{O} \in \mathbb{R}^{3 \times H \times W}$ using a unified model F. This can be formulated as:

$$\hat{O} = F(I_D) \quad (3)$$

where F is designed to generalize across various degradation types, such as noise, blur, compression artifacts, and weather effects, without explicit knowledge of Deg. The degradation process Deg can be mathematically represented as:

$$I_D = Deg(O) + \eta \quad (4)$$

Here, $O \in \mathbb{R}^{3 \times H \times W}$ denotes the original clean image, Deg represents the degradation operator (e.g., blur kernel, brightness, or quality compression), and η signifies additive noise. The main challenge arises from the fact that Deg is usually unknown and varies across tasks, making it difficult to design a universal model capable of effectively handling all possible degradations.

Unlike degradation-specific restoration, the All-in-One setting requires the model to generalize across a broad spectrum of degradation types without prior knowledge of degradation parameters. This introduces two main challenges: (a) how to represent fine-grained degradation information in a unified way; and (b) how to dynamically adapt the model behavior based on such information during inference.

### B. DPMambaIR

We propose DPMambaIR, a degradation-aware framework for All-in-One image restoration. As illustrated in Fig. 2, our method is built upon a Mamba-based U-shaped architecture with an asymmetric encoder-decoder design. Initially, a 3×3 convolution extracts shallow features $F \in \mathbb{R}^{H \times W \times C}$ from the degraded input. The encoder comprises three stages, each utilizing downsampling to progressively halve the feature resolution while doubling the channel dimension. Correspondingly, the decoder employs upsampling and skip connections to fuse multi-scale features from the encoding path. Finally, the clean image is reconstructed via a refinement stage, utilizing a global residual connection to superimpose the learned residual onto the degraded input.

To address the challenges posed by unknown degradation types and severities in All-in-One image restoration, we propose a Degradation Extractor pre-trained via a self-supervised reconstruction objective. This module encodes heterogeneous degradation patterns into a compact embedding, effectively bridging the gap between blind and non-blind restoration by providing explicit degradation representations. To fully exploit these representations, we revisit the selective state-space modeling paradigm. Specifically, we propose a Degradation-Aware Prompt State Space Model, which dynamically modulates the restoration trajectory by injecting global degradation priors into the state transition process.

### C. Degradation Extractor

To effectively extract degradation priors, we formulate a generalized degradation model that encapsates the physical corruption process from image capture to transmission, as shown in Fig. 3(a). This process typically involves four degradation categories: (1) illumination degradation, (2) motion-induced blur, (3) transmission medium artifacts, and (4) compression artifacts. We unify these factors into a composite formation model:

$$I(x) = Q\big((\alpha(x)J(x) + \beta(x)\gamma(x)) \otimes K\big) \quad (5)$$

where J(x) denotes the latent clean image. The term α(x) models luminance variations, encompassing low-light conditions or uneven illumination caused by weather. γ(x) represents additive artifacts introduced by the transmission medium, such as rain streaks, snow, or haze aerosols, while β(x) modulates the spatial intensity of these artifacts. K denotes the blur kernel resulting from relative motion, and Q represents quantization or compression operations applied during storage.

Guided by this formulation, we design a degradation extractor capable of capturing heterogeneous corruption patterns, as illustrated in Fig. 3(b). Specifically, we employ Central Difference Convolution (CDC) [40]–[42] to identify gradient-based degradation cues, such as edges and textures associated with blur and noise. A multi-scale module aggregates features from four parallel CDC branches with varying kernel sizes, enabling the perception of degradations across different frequency bands. Additionally, we integrate a standard convolution branch to capture global intensity and brightness distributions.

To balance extraction accuracy with computational efficiency, we employ a re-parameterization technique during inference, merging the multi-branch structure into a compact representation, as shown in Fig. 3(c). This yields two model variants: a larger capacity DPMambaIR-L utilizing the full multi-branch structure, and a lightweight DPMambaIR employing the re-parameterized extractor (detailed in Table I).

To enable the learning of fine-grained degradation embeddings, we propose a reconstruction-based pre-training strategy. Unlike classification-based methods that rely on discrete labels, this approach forces the extractor to learn a continuous manifold of degradation types and severities. Formally, the extractor E maps a degraded input $I_D$ to a low-dimensional embedding $E_d$:

$$E_d = E(I_D) \quad (6)$$

To supervise this process, a reconstruction decoder R attempts to recover the degraded image $I_D$ by combining $E_d$ with a content embedding $E_c$ derived from the corresponding clean ground truth O:

$$E_c = \text{Restormer}(O), \quad \hat{I}_D = R(E_c, E_d) \quad (7)$$

This disentanglement ensures that $E_d$ captures pure degradation information necessary to reconstruct the corruption, independent of the image content. The optimization minimizes a hybrid objective combining L1 loss and the Learned Perceptual Image Patch Similarity (LPIPS) metric, defined as $L = \|I_D - \hat{I}_D\|_1 + \lambda_L L_{LPIPS}$, to ensure perceptually faithful reconstruction of the degradation patterns.

### D. Degradation-aware Prompt State Space Model

To address the limitations of SSM-based image restoration, specifically the insufficient use of global context and the lack of effective degradation-aware mechanisms, we propose a novel Degradation-aware Prompt State Space Model (DP-SSM) to effectively adapt SSM to All-in-One image restoration. Specifically, we propose a novel Degradation-aware Prompt Selective Scan, which incorporates degradation information into the State Space Model. This is achieved by leveraging a Degradation Embedding extracted from a pre-trained Degradation Extractor to provide supplementary degradation-related contextual information that the traditional SSM cannot capture. In addition, we modulate the parameter matrices in the state-space equations to establish a degradation-aware mechanism.

This mechanism enables SSM to adapt to multi-task learning and handle task competition. To better recover local high-frequency details, we design a lightweight High-frequency Enhancement Block (HEB). This component ensures the preservation of high-frequency details and enhances the overall model performance.

#### Degradation-aware Prompt Selective Scan

We revisit the traditional SSM, whose parameter settings are shown in Eq.(2). Matrices A, B, C, and D control the SSM output. The state transition matrix A compresses historical information, while the input matrix B maps input signals to the state space, determining their influence on historical states. The output matrix C maps historical states to the observable space, reflecting their impact on the output. Matrix D provides a direct pathway from input to output, similar to a residual connection. In traditional SSM, these matrices and Δ are derived from:

$$A = \text{init}(), \quad D = \text{init}(), \quad B = \text{Linear}(x), \quad C = \text{Linear}(x), \quad \Delta = \text{Linear}(x) \quad (8)$$

where init() is an initialization method that does not depend on the input x. However, this input-dependent design is agnostic to the degradation type. It processes high-frequency noise and low-frequency blur with the same initialization prior, lacking the ability to structurally adapt the system dynamics for specific restoration tasks. This motivates the introduction of a degradation-aware prompt mechanism, which guides the state-space modeling by incorporating specific degradation cues.

To integrate degradation awareness into the SSM, we first utilize a degradation extractor E to extract the degradation embedding $E_d$ from the input image $I_D$. This embedding is then employed to modulate Δ, the input matrix B, and the output matrix C, thereby enhancing the model's ability to compress historical information, process input signals, and capture global context, as illustrated in Fig. 4(a). The specific formulation is given as follows:

$$E_d = E(I_D)$$
$$\alpha_\Delta = M_\Delta(E_d), \quad \alpha_B = M_B(E_d), \quad \alpha_C = M_C(E_d)$$
$$\Delta_{dp} = \alpha_\Delta \cdot \Delta, \quad B_{dp} = \alpha_B \cdot B, \quad C_{dp} = \alpha_C \cdot C \quad (9)$$

Here, $M_\Delta$, $M_B$, and $M_C$ are combinations of linear layers used to map the degradation embedding $E_d$ into feature spaces, forming modulation vectors $\alpha_\Delta$, $\alpha_B$, and $\alpha_C$ for Δ, B, and C, respectively. $\Delta_{dp}$, $B_{dp}$, and $C_{dp}$ are the newly modulated SSM parameters.

Finally, the formulation of the Degradation-aware State Space Model can be expressed as follows:

$$\bar{A}_{dp} = \exp(\Delta_{dp} A)$$
$$\bar{B}_{dp} = (\Delta_{dp} A)^{-1}(\exp(\Delta_{dp} A) - I) \cdot \Delta_{dp} B_{dp}$$
$$h_i = \bar{A}_{dp_i} h_{i-1} + \bar{B}_{dp_i} x_i$$
$$y_i = C_{dp} h_i + D x_i \quad (10)$$

The proposed modulation mechanism fundamentally alters the underlying system dynamics, effectively recalibrating the trade-off between memory inertia and input sensitivity. Central to this adaptation is the step size $\Delta_{dp}$, whose learned distribution shown in Fig. 5 reveals a distinct physical correspondence to the degradation characteristics. For high-frequency degradations such as noise and JPEG artifacts, the model autonomously converges to a minimal $\Delta_{dp}$ regime. As $\Delta_{dp}$ approaches zero, the discretized transition matrix $\bar{A}_{dp}$ approximates the identity matrix while the input gain $\bar{B}_{dp}$ diminishes. This configuration transforms the SSM into a stable low-pass filter, creating a high-inertia state that effectively suppresses instantaneous pixel variance by prioritizing historical context over the unreliable current observation. In contrast, inputs characterized by low-frequency or structural deficiencies, such as blur and low-light conditions, trigger a marked increase in the learned $\Delta_{dp}$. This shift transitions the system into a high-gain regime, amplifying the magnitude of $\bar{B}_{dp}$ to enhance sensitivity towards weak input signals and facilitate the recovery of attenuated local structures. Complementing this dynamic discretization, the concurrent modulation of B acts as a gating mechanism for input observations, while the modulation of C supplements the feature readout with global degradation cues.

Through this strategy, the transformation matrix A and input matrix B are endowed with degradation-aware capabilities, allowing them to adaptively regulate memory inertia and input sensitivity according to the specific degradation severity. However, the standard SSM remains constrained by its strictly causal nature, where the i-th pixel can only access information from the preceding i−1 pixels, resulting in a restricted global receptive field. Although existing methods attempt to alleviate this via parallel multi-directional scanning, such approaches incur high computational costs and yield limited performance gains in low-level vision tasks [36]. By retaining the efficient single-directional scanning and instead modulating matrix C to inject global degradation information, our method achieves effective global degradation awareness with significantly lower computational overhead.

#### High-frequency Enhancement Block

All-in-One image restoration necessitates the recovery of image components distributed across diverse frequency bands. While our proposed Degradation-aware Prompt State Space Model enables dynamic adaptation to degradation characteristics and partially mitigates inter-task conflicts, the joint optimization of heterogeneous degradations inevitably induces an implicit bias toward low-frequency structures. As illustrated in Fig. 6, this spectral bias results in suboptimal performance regarding high-frequency details, such as edges and textures, compared to degradation-specific models. To address this limitation, we introduce a lightweight High-Frequency Enhancement Block (HEB) as a complementary module to explicitly facilitate the restoration of local textures.

For input features F, we first extract channel-wise low-frequency features using global average pooling, then subtract them from F to obtain high-frequency components, as illustrated in Fig. 4(b). Finally, the high-frequency features are added back to enhance F, formulated as:

$$F_{enhanced} = F + \alpha \cdot (F - G(F)) \quad (11)$$

Here, α represents a learnable scalar that controls the contribution of high-frequency enhancement, and G denotes the global average pooling operation. $F_{enhanced}$ is the enhanced feature map.

### E. Loss Function

We employ the commonly used L1 loss $L_1$ and L2 loss $L_2$ in low-level vision tasks. Additionally, following previous works [43], [44], we incorporate a frequency-domain loss $L_{fft}$ for training. The total loss is defined as:

$$L_{total} = \lambda_1 \cdot L_1(O, \hat{O}) + \lambda_2 \cdot L_2(O, \hat{O}) + \lambda_3 \cdot L_{fft}(O, \hat{O}) \quad (12)$$

where $\hat{O}$ and O denote the model output and the ground truth, respectively. The parameters $\lambda_1$, $\lambda_2$, and $\lambda_3$ are balancing factors, which we set to 1, 0.5 and 0.001, respectively.

---

## IV. Experiment and Analysis

We evaluate our method on both All-in-one image restoration tasks and Degradation-specific image restoration tasks. In the All-in-One Image Restoration setting, we train a single model on a dataset containing seven types of degradation. For Degradation-specific image restoration, we train separately on datasets corresponding to each type of degradation.

### A. Experimental Settings

**Training Details.** To ensure effective degradation-aware modeling, we adopt a two-stage training strategy. In the first stage, we pre-train the Degradation Extractor using the same mixed-degradation dataset employed for the All-in-One task. The training pipeline consists of the Degradation Extractor, a Content Extractor, and a Degradation Reconstructor. Specifically, we utilize the pre-trained Restormer from DegAE [45] as the Content Extractor with its parameters frozen, while optimizing only the Degradation Extractor and Reconstructor. This pre-training phase is conducted for 100,000 iterations with a batch size of 1, using input patches cropped to 256×256. We employ the AdamW [46] optimizer (β1 = 0.9, β2 = 0.999, weight decay 1e−4) with an initial learning rate of 1e−4, which is gradually reduced to 1e−6 through cosine annealing.

In the second stage, the pre-trained Degradation Extractor is integrated into the DPMambaIR framework with its parameters frozen to provide stable guidance. Building on previous works [10], [36], [43], we train the main restoration network using the same AdamW configuration for 300,000 iterations. The initial learning rate is set to 3e−4 and decayed to 1e−6 via cosine annealing. Following [43], we employ a progressive training strategy, cropping image patches of size 192×192. Horizontal and vertical flips are applied for data augmentation. Our experiments are conducted on a single NVIDIA RTX A100 GPU and implemented using the PyTorch platform.

**Datasets.** For All-in-One image restoration, we collect a large-scale mixed dataset covering seven common degradation types: haze, rain, snow, low-light, blur, noise, and JPEG compression, as shown in Fig. 7. Detailed information regarding these datasets is summarized in Table III. Additionally, degradation-specific evaluations are conducted using LOL [47] for low-light enhancement, GoPro [48] for deblurring, Rain100H [49] for deraining, and RESIDE6K [50] for dehazing.

**Table III: Datasets for Image Restoration Tasks**

| Degradation | Dataset | Train Pairs | Test Pairs |
|---|---|---|---|
| Haze | RESIDE6K [50] | 6000 | 1000 |
| Rain | Rain13K [51] | 13711 | - |
| | Rain100H [49] | - | 100 |
| Snow | Snow100K [52] | 5000 | 1000 |
| Low Light | LOL [47], LSRW [53] | 6085 | 65 |
| Blur | GoPro [48] | 2103 | 1111 |
| Noise | WaterlooED [54] | 4744 | - |
| | BSD400 [55] | - | 400 |
| JPEG | WaterlooED [54] | 4744 | - |
| | BSD400 [55] | - | 400 |

**Evaluation Metrics.** Following previous works [36], [43], we employ Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) to quantitatively evaluate restoration quality on RGB channels. Additionally, we utilize the Learned Perceptual Image Patch Similarity (LPIPS) to assess the perceptual consistency of the restored images.

**Comparisons with State-of-the-art Methods.** For All-in-One image restoration, we compare against eleven methods: MIRNet [56], NAFNet [57], MPRNet [58], Restormer [43], MambaIR [36], PromptIR [10], IDR [28], NDR-Restore [59], OneRestore [9], MoCEIR [60], and AdaIR [29]. The first four are general-purpose restoration models, while the latter four are designed specifically for All-in-One restoration: PromptIR uses visual prompts, OneRestore employs a pre-trained degradation encoder with cross-attention, and AdaIR leverages frequency-domain information for adaptive restoration. For degradation-specific restoration, we also include representative single-task methods such as JORDER [61] (deraining), MAXIM [62] (deblurring), EnlightenGAN [63] (low-light enhancement), DeepDeblur [64], AdaIR [29], DA-CLIP [27], among others.

### B. Quantitative Results

**All-in-one Image Restoration.** We evaluated two versions of our method for comparison: the normal version (DPMambaIR) and a large version (DPMambaIR-L). The normal version adopts the degradation extractor illustrated in Fig. 3(c), while the large version uses the architecture shown in Fig. 3(b).

**Table I: Quantitative Comparison on Seven Datasets for All-in-One Image Restoration (Format: PSNR ↑ / SSIM ↑ / LPIPS ↓)**

| Method | Lowlight | Snow | Rain | Haze | Jpeg | Noise | Blur | Average |
|---|---|---|---|---|---|---|---|---|
| MPRNet | 19.71/.659/.122 | 30.94/.938/.029 | 26.27/.912/.068 | 25.63/.947/.023 | 30.51/.938/.024 | 28.03/.897/.057 | 26.07/.869/.077 | 26.74/.880/.057 |
| MIRNet | 19.81/.668/.123 | 31.04/.941/.028 | 25.77/.907/.073 | 22.26/.882/.058 | 30.54/.939/.024 | 28.14/.900/.057 | 26.32/.873/.070 | 26.27/.873/.062 |
| NAFNet | 19.49/.650/.117 | 30.33/.935/.030 | 26.04/.906/.068 | 25.63/.946/.023 | 30.20/.936/.025 | 27.80/.895/.051 | 26.13/.870/.075 | 26.52/.877/.056 |
| Restormer | 19.99/.670/.113 | 31.67/.944/.025 | 26.91/.919/.061 | 26.18/.953/.020 | 30.56/.939/.024 | 28.17/.890/.056 | 26.62/.880/.063 | 27.16/.886/.052 |
| MambaIR | 19.81/.664/.115 | 30.43/.935/.032 | 25.52/.901/.078 | 25.90/.944/.024 | 30.38/.937/.025 | 27.91/.895/.058 | 26.07/.869/.077 | 26.57/.878/.058 |
| PromptIR | 19.90/.669/.116 | 31.52/.943/.024 | 26.80/.918/.061 | 26.13/.945/.022 | 30.57/.939/.024 | 28.18/.901/.054 | 26.36/.874/.072 | 27.07/.884/.053 |
| IDR | 19.68/.663/.115 | 31.16/.941/.027 | 26.39/.914/.065 | 26.11/.948/.023 | 30.51/.939/.024 | 28.10/.899/.054 | 26.29/.873/.073 | 26.89/.883/.054 |
| NDR-Restore | 19.79/.635/.206 | 30.00/.902/.052 | 26.00/.813/.158 | 27.00/.943/.025 | 30.19/.872/.056 | 27.78/.786/.133 | 25.77/.791/.193 | 26.64/.820/.117 |
| OneRestore | 19.77/.657/.127 | 29.95/.931/.034 | 25.37/.900/.080 | 27.74/.962/.016 | 30.10/.934/.026 | 27.65/.890/.060 | 25.90/.866/.082 | 26.64/.877/.061 |
| MoCEIR | 19.92/.674/.109 | 31.22/.943/.025 | 26.56/.920/.051 | 28.06/.963/.014 | 30.51/.939/.023 | 28.13/.900/.053 | 26.34/.874/.069 | 27.25/.888/.049 |
| AdaIR | 19.99/.672/.102 | 31.94/.923/.023 | 27.01/.922/.056 | 26.61/.952/.019 | 30.53/.939/.024 | 28.19/.900/.055 | 26.61/.879/.065 | 27.26/.887/.049 |
| **DPMambaIR** | **20.04/.680/.109** | **32.36/.949/.022** | **27.07/.926/.055** | 28.13/.963/.016 | **30.57/.939/.024** | **28.25/.902/.051** | 27.43/.894/.048 | **27.69/.893/.047** |
| DPMambaIR-L | 20.06/.677/.108 | 32.31/.950/.022 | 27.11/.925/.056 | **28.39/.963/.016** | 30.55/.939/.024 | 28.21/.902/.051 | **27.45/.895/.045** | 27.73/.893/.046 |

For subsequent evaluations, we primarily adopt the normal version unless otherwise specified. We compare our method against eleven state-of-the-art image restoration approaches. As shown in Table I, for the All-in-One image restoration task, our method achieves the best PSNR and SSIM on the aforementioned mixed datasets, outperforming existing methods. Specifically, our method achieves a PSNR of 27.69 dB and an SSIM of 0.893, surpassing AdaIR by 0.43 dB and 0.011, respectively. Additionally, DPMambaIR-L achieves a PSNR of 27.73 dB and an SSIM of 0.893, slightly outperforming the normal version.

We achieve the best performance across all seven sub-tasks. Experimental results indicate that multi-task learning often encounters challenges in balancing performance across different tasks. For instance, AdaIR demonstrates strong performance in deraining but performs poorly in dehazing. On the other hand, OneRestore excels in dehazing but delivers suboptimal results in tasks such as deraining and deblurring. In contrast, our method effectively tackles the balance issue in multi-task learning, achieving consistently optimal performance across all sub-tasks. These findings highlight the effectiveness and superiority of our proposed method.

**Degradation-Specific Image Restoration.** We evaluate our method on four Degradation-specific tasks, consistently achieving superior performance across multiple datasets compared to state-of-the-art CNN-based, transformer-based, and diffusion-based approaches, as shown in Table II.

**Table II: Quantitative Comparison on Degradation-Specific Image Restoration Tasks**

| Deraining | PSNR | SSIM | LLIE | PSNR | SSIM | Deblurring | PSNR | SSIM | Dehazing | PSNR | SSIM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| JORDER | 26.25 | 0.835 | EnlightenGAN | 17.61 | 0.653 | DeepDeBlur | 29.08 | 0.913 | GCANet | 26.59 | 0.935 |
| PReNet | 29.46 | 0.899 | MIRNet | 24.14 | 0.830 | DeBlurGAN | 28.70 | 0.858 | GridDehazeNet | 25.86 | 0.944 |
| MPRNet | 30.41 | 0.891 | URetinex-Net | 19.84 | 0.824 | DeBlurGANV2 | 29.55 | 0.934 | DeHazeFormer | 30.29 | 0.964 |
| MAXIM | 30.81 | 0.903 | MAXIM | 23.43 | 0.863 | MT-RNN | 31.15 | 0.945 | MAXIM | 29.12 | 0.932 |
| Restormer | 31.46 | 0.904 | IAT | 23.38 | 0.809 | IR-SDE | 30.70 | 0.901 | DA-CLIP | 30.16 | 0.936 |
| OneRestore | 29.36 | 0.944 | OneRestore | 22.97 | 0.835 | OneRestore | 28.76 | 0.915 | OneRestore | 30.16 | 0.974 |
| MoCEIR | 31.23 | 0.960 | MoCEIR | 23.64 | 0.818 | MoCEIR | 30.05 | 0.933 | MoCEIR | 30.62 | 0.975 |
| AdaIR | 31.46 | 0.961 | AdaIR | 23.47 | 0.831 | AdaIR | 30.57 | 0.939 | AdaIR | 30.87 | 0.975 |
| **DPMambaIR** | **32.30** | **0.967** | **DPMambaIR** | **24.20** | **0.852** | **DPMambaIR** | **31.42** | **0.948** | **DPMambaIR** | **30.89** | **0.977** |

On Rain100H [49], our method achieves a PSNR of 32.30 dB and SSIM of 0.967, surpassing AdaIR by 0.84 dB and 0.006, respectively. On the LOL dataset [47], it achieves 24.20 dB PSNR and 0.852 SSIM, outperforming MIRNet by 0.06 dB and 0.022. For deblurring on GoPro [48], it attains 31.42 dB PSNR and 0.948 SSIM, exceeding MT-RNN by 0.27 dB and 0.003. On RESIDE6K [50], it achieves 30.89 dB PSNR and 0.973 SSIM, outperforming AdaIR by 0.02 dB and 0.002. These results highlight the versatility and effectiveness of our method in addressing diverse degradation restoration tasks.

### C. Qualitative Results

Fig. 8 presents the visual results of deblurring, deraining, and dehazing under the All-in-One Image Restoration task setting. Our method demonstrates superior visual performance compared to other approaches. Specifically, in deblurring, our method restores sharper edge details; in deraining, it more effectively removes rain streaks while recovering background information; and in dehazing, it focuses on regions often overlooked by other methods, resulting in more comprehensive dehazing performance. Additional visual comparisons under both the All-in-One and Degradation-specific settings are provided in the Appendix.

### D. Model Complexity and Efficiency Analysis

**Table IV: Comparison of Complexity and Efficiency Against SOTA Methods**

| Method | Params (M) | Mem. (MB) | FLOPs (G) | Inf Time (ms) | PSNR (dB) |
|---|---|---|---|---|---|
| Restormer | 26.13 | 676.24 | 154.88 | 84.283 | 27.16 |
| IDR | 36.28 | 1433.24 | 270.76 | 81.977 | 26.89 |
| PromptIR | 35.59 | 720.10 | 172.71 | 90.964 | 27.07 |
| MoCEIR | 25.35 | 404.48 | 97.74 | 93.232 | 27.25 |
| AdaIR | 28.78 | 686.40 | 161.76 | 102.903 | 27.26 |
| **DPMambaIR** | 29.35 | 718.76 | 153.33 | 90.869 | **27.69** |

To accurately evaluate the practical applicability of DPMambaIR, we conducted a comprehensive efficiency comparison against state-of-the-art All-in-One methods. We measured the number of parameters, peak memory consumption, GFLOPs, and average inference time. All evaluations were performed on a single NVIDIA RTX A100 GPU with an input tensor size of 1×3×256×256. The statistics for DPMambaIR include the overhead of the Degradation Extractor.

As summarized in Table IV, DPMambaIR achieves the best performance while maintaining comparable computational overhead. Compared to the second-best method, AdaIR, our model delivers a significant performance gain of 0.43 dB with similar parameters and GFLOPs.

### E. Ablation Study

We conduct ablation studies to evaluate our key designs: the degradation-aware prompts and High-frequency Enhancement Block (HEB) (Table V), degradation extraction and utilization methods (Table VI), degradation embedding dimension (Table VII) and the global information supplementation strategy (Table VIII).

**Table V: Extended Ablation Studies on the Proposed Core Modules**

| Method | PΔ | PB | PC | HEB | PSNR | SSIM |
|---|---|---|---|---|---|---|
| Baseline | | | | | 26.96 | 0.884 |
| (a) | | | | ✓ | 27.25 | 0.887 |
| (b) | | ✓ | ✓ | | 27.56 | 0.891 |
| (c) | ✓ | | | | 27.58 | 0.892 |
| (d) | ✓ | ✓ | | | 27.60 | 0.892 |
| (e) | ✓ | | ✓ | | 27.61 | 0.892 |
| (f) | ✓ | ✓ | ✓ | | 27.67 | 0.893 |
| (g) | ✓ | ✓ | ✓ | ✓ | **27.69** | **0.893** |

Table V demonstrates the effectiveness of our proposed core modules. First, the High-frequency Enhancement Block (HEB) alone improves the baseline by 0.29 dB (Model (a)), validating its efficacy in enhancing local details. Second, to verify the importance of the degradation-aware mechanism, we conduct a detailed breakdown of the prompt components. Notably, employing PΔ alone (Model (c)) achieves 27.58 dB, which slightly outperforms the combination of PB and PC (Model (b), 27.56 dB). This observation underscores that dynamically modulating the discretization step Δ, which fundamentally alters the state transition dynamics, is the most critical factor for adapting to diverse degradation types. Furthermore, combining PΔ with PB or PC yields consistent improvements, and integrating all prompt modules (Model (f)) pushes the PSNR to 27.67 dB. Finally, the full model (Model (g)) demonstrates the synergy between prompt-based modulation and frequency enhancement, achieving the best performance of 27.69 dB.

**Table VI: Ablation Study on Different Degradation Extraction and Utilization Methods**

| Module | Method | PSNR | SSIM |
|---|---|---|---|
| **Degradation Extraction** | Classes | 27.58 | 0.892 |
| | OneRestore* | 27.56 | 0.891 |
| | Ours | **27.69** | **0.893** |
| **Degradation Insertion** | Concat | 26.96 | 0.882 |
| | Attention | 27.39 | 0.882 |
| | Modulation (Ours) | **27.69** | **0.893** |
| **Training Strategy** | Fine-tuning | 27.43 | 0.888 |
| | Frozen (Ours) | **27.69** | **0.893** |

Table VI presents the results for degradation extraction and utilization methods. For degradation extraction, we compare three approaches: (1) "Classes", which provides manually specified degradation types and can be regarded as a 100% accurate non-blind setting; (2) a pre-trained degradation classifier from OneRestore [9]; and (3) our fine-grained degradation representation via image reconstruction. The OneRestore extractor, optimized for classification, is insensitive to degradation strength or location, leading to poorer results. Our method surpasses the non-blind "Classes" baseline by 0.11 dB PSNR. Fig. 9 visualizes the t-SNE embeddings of the extracted degradation features. Significant gaps exist between different degradation types, indicating clear separability. Notably, although JPEG and noise share the same content during synthesis, the t-SNE visualization reveals that their separability is primarily driven by degradation information rather than image content.

For degradation utilization, we compare concatenation and attention-based fusion with our prompt modulation design. Our prompt-based modulation outperforms simple concatenation or attention-based fusion, showing better adaptability to local degradations.

For the training strategy, compared with jointly fine-tuning the degradation extractor together with the backbone network, freezing the parameters of the degradation extractor provides a more stable supply of degradation information. This design choice leads to more consistent guidance during restoration and results in a clear performance gain of 0.26 dB in PSNR, demonstrating the advantage of decoupling degradation representation learning from the restoration optimization process.

**Table VII: Ablation Study on the Dimension of Degradation Embedding (Demb)**

| Demb | Params (M) | FLOPs (G) | Rec. PSNR (dB) | Cls. Acc. (%) | AIO. PSNR (dB) |
|---|---|---|---|---|---|
| 256 | 28.71 | 153.33 | 21.51 | 72.0 | 27.55 |
| 384 | 29.03 | 153.33 | 24.53 | 84.3 | 27.63 |
| **512** | **29.35** | **153.33** | **25.67** | **92.0** | **27.69** |
| 1024 | 30.62 | 153.33 | 25.78 | 92.6 | 27.69 |

Table VII presents the ablation study on the Degradation Embedding dimension (Demb). Given that the Degradation Extractor serves as the sole source of degradation priors, the completeness and distinctiveness of the extracted information are paramount. To quantitatively evaluate the quality of the learned embeddings, we introduce two direct indicators: (1) Degradation Reconstruction PSNR (Rec. PSNR), calculated between the reconstructed degraded image and the target degraded input, which measures information integrity; and (2) Classification Accuracy (Cls. Acc.), which assesses the semantic separability of the degradation types. For the classification metric, we appended a simple MLP head to the frozen degradation extractor and trained it for only 1,000 iterations to map the embeddings to the seven degradation classes. As observed in Table VII, increasing Demb from 256 to 512 yields substantial gains: Rec. PSNR improves from 21.51 dB to 25.67 dB, and Cls. Acc. surges from 72.0% to 92.0%. This indicates that higher-dimensional embeddings effectively encode both the pixel-level details and the semantic categories of degradations. These improvements directly translate to the final restoration task, boosting AIO. PSNR to 27.69 dB. However, expanding to 1024 yields marginal returns despite higher costs. Thus, 512 is selected as the optimal balance.

**Table VIII: Ablation Study on Different Methods for Global Information Supplementation**

| Method | PSNR | SSIM | Params (M) | GFLOPs |
|---|---|---|---|---|
| Bi-direction Scan | 27.66 | 0.893 | 31.43 | 244.3 |
| Prompt-C (Ours) | **27.69** | **0.893** | 29.35 | 153.3 |

Table VIII presents the ablation study on global information supplementation. We compare our Prompt-C with Mamba's bi-directional scanning scheme. Compared to Mamba's bi-directional scan, our Prompt-C not only achieves slightly better performance but is also more efficient, reducing GFLOPs by nearly 37%. This highlights its effectiveness as a lightweight global information supplementation method.

### F. Generalization Capability and Failure Boundary Analysis

To investigate the robustness of DPMambaIR beyond the training distribution, we conducted Out-of-Distribution (OOD) evaluations on two unseen degradation types: Raindrop and Pixelation. Importantly, these evaluations were performed by directly applying the pre-trained model to the test images without any additional training or fine-tuning. This rigorous setting serves to delineate the generalization boundaries of our Degradation-Aware Prompt mechanism.

**Table IX: Quantitative Comparison on Unseen Degradation Types (OOD Testing)**

| Method | Pixelation (PSNR/SSIM/LPIPS) | Raindrop (PSNR/SSIM/LPIPS) |
|---|---|---|
| MambaIR | 20.80 / 0.7601 / 0.0568 | 19.29 / 0.7992 / 0.1051 |
| PromptIR | 19.69 / 0.7479 / 0.0582 | 19.26 / 0.7775 / 0.1146 |
| IDR | 20.64 / 0.7567 / 0.0494 | 17.40 / 0.7609 / 0.1276 |
| OneRestore | 19.75 / 0.7415 / 0.0628 | 19.75 / 0.7792 / 0.1100 |
| AdaIR | 21.39 / 0.7676 / 0.0512 | 19.07 / 0.7722 / 0.1225 |
| MoCEIR | 19.89 / 0.7355 / 0.0544 | 21.46 / 0.8254 / 0.0919 |
| **DPMambaIR** | **23.35 / 0.8003 / 0.0551** | **21.63 / 0.8337 / 0.0932** |

**Robustness on Unseen Degradations.** We evaluated the model on the Raindrop dataset [65] (58 images) and a synthetically generated Pixelated BSD68 dataset [66] (block size 2). As summarized in Table IX, DPMambaIR exhibits superior generalization, securing top-tier performance across all metrics. Specifically, on the Pixelation task, it achieves a PSNR of 23.35 dB, surpassing the second-best AdaIR by a significant margin of 1.96 dB. We attribute this robustness to our regression-based embedding strategy. Unlike static prompt approaches that risk model collapse on undefined inputs, DPMambaIR projects unseen degradations into a continuous latent space. This mechanism enables the model to approximate unknown corruption patterns within a learned manifold, preventing inference failure and yielding visually plausible results. A notable phenomenon is further observed in a specific low-light sample from the BSD68 dataset. When processing the pixelated version of this image, DPMambaIR automatically performs brightness enhancement alongside de-pixelation, capturing and correcting the underlying low-light degradation. In contrast, competing methods fail to address this compound issue. This capability underscores the potential of DPMambaIR in handling mixed degradations, suggesting that future work can further explore real-world restoration via more complex, fine-grained coupled degradation modeling.

**Failure Boundary Analysis.** Despite these promising results, DPMambaIR is not without limitations. We observed a distinct performance bottleneck when processing images with heavy raindrops. As visually analyzed in Fig. 10, while the model effectively removes light raindrops, it struggles with large, opaque water droplets that completely occlude the background. We attribute this to the physical nature of the degradation: removing heavy, opaque occlusions requires generative hallucination (inpainting) to synthesize missing content, whereas DPMambaIR operates on restoration cues. Future work could explore incorporating generative priors, such as Diffusion Models, to address such severe occlusions.

---

## V. Conclusion

We propose DPMambaIR, a novel All-in-One framework for image restoration capable of handling diverse degradation types. The core of our framework is a Degradation-Aware Prompt State Space Model (DP-SSM) that leverages a fine-grained degradation extractor. This design enables the dynamic integration of degradation features into the state-space modeling process, allowing for adaptive handling of complex degradation scenarios while maintaining global degradation awareness. Furthermore, a lightweight High-frequency Enhancement Block (HEB) is introduced to complement the main framework by enhancing high-frequency detail restoration with negligible computational overhead. This study underscores the importance of fine-grained degradation modeling and dynamic feature modulation in advancing All-in-One image restoration frameworks. Extensive experiments on a mixed dataset containing seven degradation types show that DPMambaIR achieves the best performance, establishing it as a promising approach for unified image restoration.

---

## References

[1] T. Zhang, R. Wang, Y. Niu, Z. Li, and T. Zhao, "Hugs-net: A lightweight and unified network for adverse weather image denoising," *IEEE Transactions on Multimedia*, pp. 1–10, 2025.

[2] C. Kim, T. H. Kim, and S. Baik, "Lan: Learning to adapt noise for image denoising," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 25 193–25 202.

[3] K. Zhuang, Q. Li, Y. Yuan, and Q. Wang, "Multi-domain adaptation for motion deblurring," *IEEE Transactions on Multimedia*, vol. 26, pp. 3676–3688, 2024.

[4] Y.-T. Peng, W.-H. Li, and Z. Chen, "Rain2avoid: Learning deraining by self-supervision," *IEEE Transactions on Multimedia*, vol. 27, pp. 4765–4779, 2025.

[5] Y. Su, N. Wang, Z. Cui, Y. Cai, C. He, and A. Li, "Real scene single image dehazing network with multi-prior guidance and domain transfer," *IEEE Transactions on Multimedia*, vol. 27, pp. 5492–5506, 2025.

[6] Z. Wang, H. Zhao, J. Peng, L. Yao, and K. Zhao, "Odcr: Orthogonal decoupling contrastive regularization for unpaired image dehazing," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 25 479–25 489.

[7] I. Morawski, K. He, S. Dangi, and W. H. Hsu, "Leveraging content and context cues for low-light image enhancement," *IEEE Transactions on Multimedia*, vol. 27, pp. 5337–5351, 2025.

[8] R. Zhang, Y. Luo, J. Liu, H. Yang, Z. Dong, D. Gudovskiy, T. Okuno, Y. Nakata, K. Keutzer, Y. Du et al., "Efficient deweahter mixture-of-experts with uncertainty-aware feature-wise linear modulation," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 38, no. 15, 2024, pp. 16 812–16 820.

[9] Y. Guo, Y. Gao, Y. Lu, H. Zhu, R. W. Liu, and S. He, "Onerestore: A universal restoration framework for composite degradation," in *European Conference on Computer Vision*. Springer, 2024, pp. 255–272.

[10] V. Potlapalli, S. W. Zamir, S. H. Khan, and F. Shahbaz Khan, "Promptir: Prompting for all-in-one image restoration," *Advances in Neural Information Processing Systems*, vol. 36, pp. 71 275–71 293, 2023.

[11] K. He, J. Sun, and X. Tang, "Single image haze removal using dark channel prior," *IEEE transactions on pattern analysis and machine intelligence*, vol. 33, no. 12, pp. 2341–2353, 2010.

[12] R. Fattal, "Dehazing using color-lines," *ACM transactions on graphics (TOG)*, vol. 34, no. 1, pp. 1–14, 2014.

[13] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *nature*, vol. 521, no. 7553, pp. 436–444, 2015.

[14] S. Liu, D. Suzhang, M. Yang, X. Zheng, and C. Zhu, "Depth map super-resolution via deep cross-modality and cross-scale guidance," *IEEE Transactions on Multimedia*, pp. 1–14, 2025.

[15] L. Peng, W. Li, R. Pei, J. Ren, J. Xu, Y. Wang, Y. Cao, and Z.-J. Zha, "Towards realistic data generation for real-world super-resolution," *arXiv preprint arXiv:2406.07255*, 2024.

[16] L. Peng, A. Wu, W. Li, P. Xia, X. Dai, X. Zhang, X. Di, H. Sun, R. Pei, Y. Wang et al., "Pixel to gaussian: Ultra-fast continuous super-resolution with 2d gaussian modeling," *arXiv preprint arXiv:2503.06617*, 2025.

[17] K. Zhang, W. Luo, Y. Zhong, L. Ma, W. Liu, and H. Li, "Adversarial spatio-temporal learning for video deblurring," *IEEE Transactions on Image Processing*, vol. 28, no. 1, pp. 291–301, 2018.

[18] K. Zhang, T. Wang, W. Luo, W. Ren, B. Stenger, W. Liu, H. Li, and M.-H. Yang, "Mc-blur: A comprehensive benchmark for image deblurring," *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 34, no. 5, pp. 3755–3767, 2023.

[19] K. Zhang, D. Li, W. Luo, W. Ren, and W. Liu, "Enhanced spatio-temporal interaction learning for video deraining: faster and better," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 45, no. 1, pp. 1287–1293, 2022.

[20] Z. Jin, Y. Qiu, K. Zhang, H. Li, and W. Luo, "Mb-taylorformer v2: improved multi-branch linear transformer expanded by taylor formula for image restoration," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2025.

[21] X. Guo, X. Wang, X. Fu, and Z.-J. Zha, "Deep unfolding network for image desnowing with snow shape prior," *IEEE Transactions on Circuits and Systems for Video Technology*, 2025.

[22] K. Zhang, R. Li, Y. Yu, W. Luo, and C. Li, "Deep dense multi-scale network for snow removal using semantic and depth priors," *IEEE Transactions on Image Processing*, vol. 30, pp. 7419–7431, 2021.

[23] Y. LeCun, B. Boser, J. Denker, D. Henderson, R. Howard, W. Hubbard, and L. Jackel, "Handwritten digit recognition with a back-propagation network," *Advances in neural information processing systems*, vol. 2, 1989.

[24] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," *Advances in neural information processing systems*, vol. 30, 2017.

[25] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," *Advances in neural information processing systems*, vol. 33, pp. 6840–6851, 2020.

[26] B. Li, X. Liu, P. Hu, Z. Wu, J. Lv, and X. Peng, "All-in-one image restoration for unknown corruption," in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2022, pp. 17 452–17 462.

[27] Z. Luo, F. K. Gustafsson, Z. Zhao, J. Sjölund, and T. B. Schön, "Controlling vision-language models for universal image restoration," *arXiv preprint arXiv:2310.01018*, 2023.

[28] J. Zhang, J. Huang, M. Yao, Z. Yang, H. Yu, M. Zhou, and F. Zhao, "Ingredient-oriented multi-degradation learning for image restoration," in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2023, pp. 5825–5835.

[29] Y. Cui, S. W. Zamir, S. Khan, A. Knoll, M. Shah, and F. S. Khan, "AdaIR: Adaptive all-in-one image restoration via frequency mining and modulation," in *The Thirteenth International Conference on Learning Representations*, 2025.

[30] R. E. Kalman, "A new approach to linear filtering and prediction problems," 1960.

[31] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, "Vision mamba: Efficient visual representation learning with bidirectional state space model," *arXiv preprint arXiv:2401.09417*, 2024.

[32] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., "An image is worth 16x16 words: Transformers for image recognition at scale," *arXiv preprint arXiv:2010.11929*, 2020.

[33] N. Yang, Y. Wang, Z. Liu, M. Li, Y. An, and X. Zhao, "Smamba: Sparse mamba for event-based object detection," *arXiv preprint arXiv:2501.11971*, 2025.

[34] C. Xiao, M. Li, Z. Zhang, D. Meng, and L. Zhang, "Spatial-mamba: Effective visual state space models via structure-aware state fusion," *arXiv preprint arXiv:2410.15091*, 2024.

[35] J. Sheng, J. Zhou, J. Wang, P. Ye, and J. Fan, "Dualmamba: A lightweight spectral-spatial mamba-convolution network for hyperspectral image classification," *IEEE Transactions on Geoscience and Remote Sensing*, 2024.

[36] H. Guo, J. Li, T. Dai, Z. Ouyang, X. Ren, and S.-T. Xia, "Mambair: A simple baseline for image restoration with state-space model," in *European conference on computer vision*. Springer, 2024, pp. 222–241.

[37] H. Guo, Y. Guo, Y. Zha, Y. Zhang, W. Li, T. Dai, S.-T. Xia, and Y. Li, "Mambairv2: Attentive state space restoration," *arXiv preprint arXiv:2411.15269*, 2024.

[38] L. Peng, X. Di, Z. Feng, W. Li, R. Pei, Y. Wang, X. Fu, Y. Cao, and Z.-J. Zha, "Directing mamba to complex textures: An efficient texture-aware state space model for image restoration," *arXiv preprint arXiv:2501.16583*, 2025.

[39] X. Di, L. Peng, P. Xia, W. Li, R. Pei, Y. Cao, Y. Wang, and Z.-J. Zha, "Qmambabsr: Burst image super-resolution with query state space model," *arXiv preprint arXiv:2408.08665*, 2024.

[40] Y. Wang, L. Peng, L. Li, Y. Cao, and Z.-J. Zha, "Decoupling-and-aggregating for image exposure correction," in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2023, pp. 18 115–18 124.

[41] L. Peng, Y. Cao, R. Pei, W. Li, J. Guo, X. Fu, Y. Wang, and Z.-J. Zha, "Efficient real-world image super-resolution via adaptive directional gradient convolution," *arXiv preprint arXiv:2405.07023*, 2024.

[42] L. Peng, Y. Cao, Y. Sun, and Y. Wang, "Lightweight adaptive feature de-drifting for compressed image classification," *IEEE Transactions on Multimedia*, vol. 26, pp. 6424–6436, 2024.

[43] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, "Restormer: Efficient transformer for high-resolution image restoration," in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2022, pp. 5728–5739.

[44] Y. Cui, W. Ren, X. Cao, and A. Knoll, "Focal network for image restoration," in *Proceedings of the IEEE/CVF international conference on computer vision*, 2023, pp. 13 001–13 011.

[45] Y. Liu, J. He, J. Gu, X. Kong, Y. Qiao, and C. Dong, "Degae: A new pretraining paradigm for low-level vision," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, pp. 23 292–23 303.

[46] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," *arXiv preprint arXiv:1412.6980*, 2014.

[47] C. Wei, W. Wang, W. Yang, and J. Liu, "Deep retinex decomposition for low-light enhancement," *arXiv preprint arXiv:1808.04560*, 2018.

[48] S. Nah, T. Hyun Kim, and K. Mu Lee, "Deep multi-scale convolutional neural network for dynamic scene deblurring," in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2017, pp. 3883–3891.

[49] W. Yang, R. T. Tan, J. Feng, J. Liu, Z. Guo, and S. Yan, "Deep joint rain detection and removal from a single image," in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2017, pp. 1357–1366.

[50] X. Qin, Z. Wang, Y. Bai, X. Xie, and H. Jia, "Ffa-net: Feature fusion attention network for single image dehazing," in *Proceedings of the AAAI conference on artificial intelligence*, vol. 34, no. 07, 2020, pp. 11 908–11 915.

[51] K. Jiang, Z. Wang, P. Yi, C. Chen, B. Huang, Y. Luo, J. Ma, and J. Jiang, "Multi-scale progressive fusion network for single image deraining," in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2020, pp. 8346–8355.

[52] Y.-F. Liu, D.-W. Jaw, S.-C. Huang, and J.-N. Hwang, "Desnownet: Context-aware deep network for snow removal," *IEEE Transactions on Image Processing*, vol. 27, no. 6, pp. 3064–3073, 2018.

[53] J. Hai, Z. Xuan, R. Yang, Y. Hao, F. Zou, F. Lin, and S. Han, "R2rnet: Low-light image enhancement via real-low to real-normal network," *Journal of Visual Communication and Image Representation*, vol. 90, p. 103712, 2023.

[54] K. Ma, Z. Duanmu, Q. Wu, Z. Wang, H. Yong, H. Li, and L. Zhang, "Waterloo exploration database: New challenges for image quality assessment models," *IEEE Transactions on Image Processing*, vol. 26, no. 2, pp. 1004–1016, 2016.

[55] P. Arbelaez, M. Maire, C. Fowlkes, and J. Malik, "Contour detection and hierarchical image segmentation," *IEEE transactions on pattern analysis and machine intelligence*, vol. 33, no. 5, pp. 898–916, 2010.

[56] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, "Learning enriched features for real image restoration and enhancement," in *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXV 16*. Springer, 2020, pp. 492–511.

[57] L. Chen, X. Chu, X. Zhang, and J. Sun, "Simple baselines for image restoration," *arXiv preprint arXiv:2204.04676*, 2022.

[58] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, "Multi-stage progressive image restoration," in *CVPR*, 2021.

[59] M. Yao, R. Xu, Y. Guan, J. Huang, and Z. Xiong, "Neural degradation representation learning for all-in-one image restoration," *IEEE Transactions on Image Processing*, 2024.

[60] E. Zamfir, Z. Wu, N. Mehta, Y. Tan, D. P. Paudel, Y. Zhang, and R. Timofte, "Complexity experts are task-discriminative learners for any image restoration," 2024.

[61] W. Yang, R. T. Tan, J. Feng, Z. Guo, S. Yan, and J. Liu, "Joint rain detection and removal from a single image with contextualized deep networks," *IEEE transactions on pattern analysis and machine intelligence*, vol. 42, no. 6, pp. 1377–1393, 2019.

[62] Z. Tu, H. Talebi, H. Zhang, F. Yang, P. Milanfar, A. Bovik, and Y. Li, "Maxim: Multi-axis mlp for image processing," *CVPR*, 2022.

[63] Y. Jiang, X. Gong, D. Liu, Y. Cheng, C. Fang, X. Shen, J. Yang, P. Zhou, and Z. Wang, "Enlightengan: Deep light enhancement without paired supervision," *IEEE Transactions on Image Processing*, vol. 30, pp. 2340–2349, 2021.

[64] S. Nah, T. H. Kim, and K. M. Lee, "Deep multi-scale convolutional neural network for dynamic scene deblurring," in *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, July 2017.

[65] R. Qian, R. T. Tan, W. Yang, J. Su, and J. Liu, "Attentive generative adversarial network for raindrop removal from a single image," in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2018, pp. 2482–2491.

[66] D. Martin, C. Fowlkes, D. Tal, and J. Malik, "A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics," in *Proceedings eighth IEEE international conference on computer vision. ICCV 2001*, vol. 2. IEEE, 2001, pp. 416–423.