
# VRT: A Video Restoration Transformer

## Abstract

Video restoration (e.g., video super-resolution) aims to restore high-quality frames from low-quality frames. Different from single image restoration, video restoration generally requires to utilize temporal information from multiple adjacent but usually misaligned video frames. Existing deep methods generally tackle with this by exploiting a sliding window strategy or a recurrent architecture, which either is restricted by frame-by-frame restoration or lacks longrange modelling ability. In this paper, we propose a Video Restoration Transformer (VRT) with parallel frame prediction and long-range temporal dependency modelling abilities. More specifically, VRT is composed of multiple scales, each of which consists of two kinds of modules: temporal mutual self attention (TMSA) and parallel warping. TMSA divides the video into small clips, on which mutual attention is applied for joint motion estimation, feature alignment and feature fusion, while self attention is used for feature extraction. To enable cross-clip interactions, the video sequence is shifted for every other layer. Besides, parallel warping is used to further fuse information from neighboring frames by parallel feature warping. Experimental results on five tasks, including video super-resolution, video deblurring, video denoising, video frame interpolation and space-time video super-resolution, demonstrate that VRT outperforms the state-of-the-art methods by large margins (up to 2.16dB) on fourteen benchmark datasets.

## Introduction

Video restoration, which reconstructs high-quality (HQ) frames from multiple low-quality (LQ) frames, has attracted much attention recently. Compared with image restoration, the key challenge of video restoration lies in how to make full use of neighboring highly-related but misaligned supporting frames for reconstructing reference frames.

Existing video restoration methods can be mainly divided into two categories: sliding window-based methods [5, 28, 30, 41, 44, 82, 92, 99, 123] and recurrent methods [8, 10, 20, 25, 27, 29, 31, 52, 67, 76, 80, 122]. As shown in Fig. 1a, sliding window-based methods generally input multiple frames to generate a single HQ frame and processes long video sequences in a sliding window fashion. Each input frame is processed for multiple times in inference, leading to inefficient feature utilization and increased computation cost.

Some other methods are based on a recurrent architecture. As shown in Fig. 1b, recurrent models mainly use previously reconstructed HQ frames for subsequent frame reconstruction. Due to the recurrent nature, they have three disadvantages. First, recurrent methods are limited in parallelization for efficient distributed training and inference. Second, although information is accumulated frame by frame, recurrent models are not good at long-range temporal dependency modelling. One frame may strongly affect the next adjacent frame, but its influence is quickly lost after few time steps [22,94]. Third, they suffer from significant performance drops on few-frame videos [6].

In this paper, we propose a Video Restoration Transformer (VRT) that allows for parallel computation and long-range dependency modelling in video restoration. Based on a multi-scale framework, VRT divides the video sequence into non-overlapping clips and shifts it alternately to enable inter-clip interactions. Specifically, each scale of VRT has several temporal mutual self attention (TMSA) modules followed by a parallel warping module. In TMSA, mutual attention is focused on mutual alignment between neighboring two-frame clips, while self attention is used for feature extraction. At the end of each scale, we further use parallel warping to fuse neighboring frame information into the current frame. After multi-scale feature extraction, alignment and fusion, the HQ frames are individually reconstructed from their corresponding frame features.

Compared with existing video restoration frameworks, VRT has several benefits. First, as shown in Fig. 1c, VRT is trained and tested on long video sequences in parallel. In contrast, both sliding window-based and recurrent methods are often tested frame by frame. Second, VRT has the ability to model long-range temporal dependencies, utilizing information from multiple neighbouring frames during the reconstruction of each frame. By contrast, sliding windowbased methods cannot be easily scaled up to long sequence modelling, while recurrent methods may forget distant information after several timestamps. Third, VRT proposes to use mutual attention for joint feature alignment and fusion. It adaptively utilizes features from supporting frames and fuses them into the reference frame, which can be regarded as implicit motion estimation and feature warping.

Our contributions can be summarized as follows:

1. We propose a new framework named Video Restoration Transformer (VRT) that is characterized by parallel computation and long-range dependency modelling. It jointly extracts, aligns, and fuses frame features at multiple scales.
    
2. We propose the mutual attention for mutual alignment between frames. It is a generalized "soft" version of image warping after implicit motion estimation.
    
3. VRT achieves state-of-the-art performance on video restoration, including video super-resolution, deblurring, denoising, frame interpolation and space-time video super-resolution. It outperforms state-of-the-art methods by up to 2.16dB on benchmark datasets.
    

---

# Low-Light Video Enhancement via Spatial-Temporal Consistent Decomposition

## Abstract

Low-Light Video Enhancement (LLVE) seeks to restore dynamic or static scenes plagued by severe invisibility and noise. In this paper, we present an innovative video decomposition strategy that incorporates view-independent and view-dependent components to enhance the performance of LLVE. We leverage dynamic cross-frame correspondences for the view-independent term (which primarily captures intrinsic appearance) and impose a scene-level continuity constraint on the view-dependent term (which mainly describes the shading condition) to achieve consistent and satisfactory decomposition results. To further ensure consistent decomposition, we introduce a dual-structure enhancement network featuring a cross-frame interaction mechanism. By supervising different frames simultaneously, this network encourages them to exhibit matching decomposition features. This mechanism can seamlessly integrate with encoder-decoder single-frame networks, incurring minimal additional parameter costs. Extensive experiments are conducted on widely recognized LLVE benchmarks, covering diverse scenarios. Our framework consistently outperforms existing methods, establishing a new SOTA performance.

## Introduction

Low-light enhancement aims to enhance underexposed images and videos captured in low-light conditions [1, 2], improving their visual quality while reducing noise. This technique can be applied in wide-ranging applications, such as portrait photography on mobile devices [3, 4], nighttime face recognition [5, 6] and vehicle detection [7, 8]. The key challenge when enhancing videos is the need for consistent enhancement results across corresponding locations in different frames, which vary both spatially and temporally. Moreover, there is a scarcity of real-world, high-quality, spatially aligned video pairs for dynamic scenes, resulting in limited generalization capabilities for unseen videos with varying conditions.

One solution is to incorporate parameterized enhancement models with physical significance into the enhancement process, reducing the reliance on training data. According to the linear Lambertian model for intrinsic image decomposition [9, 10], an observed image I can be expressed as the element-wise product of its albedo component and shading component. The albedo part is normally view-independent, capturing the intrinsic appearance, whereas the shading component is commonly view-dependent, influenced by lighting conditions and the surface normal's direction. Inspired by the Lambertian model, we propose that a frame in the LLVE task can similarly be decomposed as (I=L \otimes R), where L represents the view-dependent term, and R corresponds to the view-independent component.

In this paper, we introduce an innovative approach to attain our decomposition for normal-light video outputs, ensuring consistency in both spatial and temporal dimensions. We leverage cross-frame correspondences and real-world physical continuity constraints to achieve this decomposition without the need for explicit supervision. Furthermore, to enhance the consistency of the decomposition, we've developed an efficient dual-structure enhancement network featuring a novel cross-frame interaction mechanism.

Our approach aims to predict both the R and L of normallight images from the provided low-light inputs. It is important to note that the decomposition of view-independent and viewdependent terms is not unique, and our goal is to implement a suitable decomposition that enhances the performance of LLVE. In this paper, we aim to design appropriate priors directly from the video data to implement the decomposition. We recognize that R should represent the physical properties of objects, which should remain consistent across various observation perspectives. To achieve this, we first calculate corresponding locations in videos and utilize these computed spatial-temporal correspondences, along with uncertainty values, to link the predictions of R across different frames. Conversely, the terms of L are subject to a spatial smoothness loss to model real-world physical constraints (since the lighting field is continuous in practice). This approach allows us to establish a temporal-spatial consistency constraint in the enhanced videos, without requiring additional supervision. Note that we cannot guarantee the decomposition results are strictly view-independent or view-dependent, as this is not a 3D method. Our goal is to apply appropriate constraints to approximate desired decomposition properties, while focusing on optimizing the final LLVE performance with our approach.

Additionally, we introduce a dual-network structure to enforce the consistency of feature representations across frames, facilitating the consistent synthesis of decomposition in turn. Two frames are used as inputs, and their features are propagated within the designed Cross-Frame Interaction Module (CFIM). CFIM differs from other similar architectures in the restoration task [11] in terms of the fusion manner. CFIM facilitates interaction through a combination of long-range cross-frame attention computation and short-range channel-spatial fusion, ensuring the accurate and comprehensive merging of features from both global and local perspectives. Moreover, CFIM selectively incorporates knowledge from randomly chosen neighboring frames into the backbone (i.e., the part without feature propagation) during training. In this way, CFIM help the backbone in learning how to adaptively utilize knowledge from various cross-frame features, resulting in a more robust backbone.

We performed experiments on various widely recognized video enhancement datasets. Our results, both quantitative and qualitative, consistently demonstrate the effectiveness and state-of-the-art (SOTA) performance of our framework, as shown in Fig. 1. Furthermore, we conducted a large-scale user study involving 100 participants, which showcased the superiority of our results in terms of human subjective perceptions. In summary, our contributions are three-fold.

We introduce a novel canonical form for LLVE that predicts spatial-temporal consistent decomposition with view-independent and view-dependent terms for normal light outputs. This decomposition strategy leverages spatial-temporal correspondences and continuity. • We design a new LLVE network that facilitates interaction among the features of different frames and fits our consistent decomposition strategy. • We conduct extensive experiments on public datasets, illustrating the effectiveness of our proposed framework.

---

# Learning Temporal Consistency for Low Light Video Enhancement from Single Images

## Abstract

Single image low light enhancement is an important task and it has many practical applications. Most existing methods adopt a single image approach. Although their performance is satisfying on a static single image, we found, however, they suffer serious temporal instability when handling low light videos. We notice the problem is because existing data-driven methods are trained from single image pairs where no temporal information is available. Unfortunately, training from real temporally consistent data is also problematic because it is impossible to collect pixel-wisely paired low and normal light videos under controlled environments in large scale and diversities with noise of identical statistics. In this paper, we propose a novel method to enforce the temporal stability in low light video enhancement with only static images. The key idea is to learn and infer motion field (optical flow) from a single image and synthesize short range video sequences. Our strategy is general and can extend to large scale datasets directly. Based on this idea, we propose our method which can infer motion prior for single image low light video enhancement and enforce temporal consistency. Rigorous experiments and user study demonstrate the state-of-the-art performance of our proposed method. Our code and model will be publicly available at [https://github.com/zkawfanx/StableLLVE](https://github.com/zkawfanx/StableLLVE).

## Introduction

Illumination in sunny day and low light night can vary more than 10 orders of magnitude. In low light scenes, sensor noise is not negligible due to the low signal-to-noise ratio (SNR). Therefore, low light image enhancement is an important task which improves the SNR and enhances the image after modeling the noise and the signal. It enables various computer vision algorithms to perform properly.

Rather than explicitly modeling the noise and the signal, recent data-driven methods [3, 4, 9, 25, 26, 27] implicitly learn such models from image data and get satisfying results on a single static image. And they require pixel-wisely paired images of low and high SNR for training. However, we notice that it is impossible to collect pixel-wisely paired low and normal light videos under controlled environments in large scale and diversities with noise of identical statistics. Therefore, existing single image methods use either synthetic data or temporally inconsistent single image data for training. Thus, no temporal consistency can be learned through existing data. One can perceive serious artifacts and flickering from existing single image methods when handling low light videos.

In this paper, we aim to enforce temporal consistency even when training from static images. We propose a novel method to enforce the temporal stability in low light video enhancement with only static images. The key idea is to learn and infer motion field (optical flow) from a single image and synthesize short range video sequences. Our strategy is general and can extend to large scale datasets directly.

Based on this idea, we propose our method which can infer motion prior for single image low light video enhancement and enforce temporal consistency. In particular, we present an image-based method to achieve low light video enhancement and tackle temporal inconsistency problem by imposing consistency on the network. Specifically, we choose optical flow to mimic motions of dynamic scenes. It is more capable of representing both global and local motions. We first predict plausible optical flow from static images. Then we warp images with optical flow to be adjacent frames and impose consistency on deep model.

We conduct rigorous experiments to validate the effectiveness of our method. Experimental results on both synthetic and real data show that our method outperforms the state-of-the-art single image methods and achieves comparable results to video-based ones, which means our method can alleviate flickering problem without the need of videos. Furthermore, we also conduct a user study on 26 volunteers, of whom 78.9% prefer our method, suggesting the better temporal stability of our method.

Our main contributions are summarized as follows: • We present a novel solution to solving temporal inconsistency problem of low light video enhancement when using only single image data. • We propose to use optical flow prior to indicate potential motion from single image and thus enable us to model the temporal consistency. • We demonstrate the state-of-the-art performance of our method from rigorous experiments and user study.

---

# Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement

## Abstract

When enhancing low-light images, many deep learning algorithms are based on the Retinex theory. However, the Retinex model does not consider the corruptions hidden in the dark or introduced by the light-up process. Besides, these methods usually require a tedious multi-stage training pipeline and rely on convolutional neural networks, showing limitations in capturing long-range dependencies. In this paper, we formulate a simple yet principled One-stage Retinex-based Framework (ORF). ORF first estimates the illumination information to light up the low-light image and then restores the corruption to produce the enhanced image. We design an Illumination-Guided Transformer (IGT) that utilizes illumination representations to direct the modeling of non-local interactions of regions with different lighting conditions. By plugging IGT into ORF, we obtain our algorithm, Retinexformer. Comprehensive quantitative and qualitative experiments demonstrate that our Retinexformer significantly outperforms state-of-the-art methods on thirteen benchmarks. The user study and application on lowlight object detection also reveal the latent practical values of our method. Code is available at [https://github.com/caiyuanhao1998/Retinexformer](https://github.com/caiyuanhao1998/Retinexformer)

## Introduction

Low-light image enhancement is an important yet challenging task in computer vision. It aims to improve the poor visibility and low contrast of low-light images and restore the corruptions (e.g., noise, artifact, color distortion, etc.) hidden in the dark or introduced by the light-up process. These issues challenge not only human visual perception but also other vision tasks like nighttime object detection.

Hence, a large number of algorithms have been proposed for low-light image enhancement. However, these existing algorithms have their own drawbacks. Plain methods like histogram equalization and gamma correction tend to produce undesired artifacts because they barely consider the illumination factors. Traditional cognition methods rely on the Retinex theory [27] that assumes the color image can be decomposed into two components, i.e., reflectance and illumination. Different from plain methods, traditional methods focus on illumination estimation but usually introduce severe noise or distort color locally because these methods assume that the images are noise- and color distortion-free. This is inconsistent with real under-exposed scenes.

With the development of deep learning, convolutional neural networks (CNNs) have been applied in low-light image enhancement. These CNN-based methods are mainly divided into two categories. The first category directly employs a CNN to learn a brute-force mapping function from the low-light image to its normal-light counterpart, thereby ignoring human color perception. This kind of methods lack interpretability and theoretically proven properties. The second category is inspired by the Retinex theory. These methods [54, 65, 66] usually suffer from a multi-stage training pipeline. They employ different CNNs to decompose the color image, denoise the reflectance, and adjust the illumination, respectively. These CNNs are first trained independently and then connected together to be finetuned endto-end. The training process is tedious and time-consuming.

In addition, these CNN-based methods show limitations in capturing long-range dependencies and non-local selfsimilarity, which are critical for image restoration. The recently rising deep learning model, Transformer, may provide a possibility to address this drawback of CNN-based methods. However, directly applying original vision Transformers for low-light image enhancement may encounter an issue. The computational complexity is quadratic to the input spatial size. This computational cost may be unaffordable. Due to this limitation, some CNN-Transformer hybrid algorithms like SNR-Net [57] only employ a single global Transformer layer at the lowest spatial resolution of a Ushaped CNN. Thus, the potential of Transformer for lowlight image enhancement still remains under-explored.

To cope with the above problems, we propose a novel method, Retinexformer, for low-light image enhancement. Firstly, we formulate a simple yet principled One-stage Retinex-based Framework (ORF). We revise the original Retinex model by introducing perturbation terms to the reflectance and illumination for modeling the corruptions. Our ORF estimates the illumination information and uses it to light up the low-light images. Then ORF employs a corruption restorer to suppress noise, artifacts, under-/overexposure, and color distortion. Different from previous Retinex-based deep learning frameworks that suffer from a tedious multi-stage training pipeline, our ORF is trained end-to-end in a one-stage manner. Secondly, we propose an Illumination-Guided Transformer (IGT) to model the long-range dependencies. The key component of IGT is Illumination-Guided Multi-head Self-Attention (IG-MSA). IG-MSA exploits the illumination representations to direct the computation of self-attention and enhance the interactions between regions of different exposure levels. Finally, we plug IGT into ORF as the corruption restorer to derive our method, Retinexformer. As shown in Fig. 1, our Retinexformer surpasses state-of-the-art (SOTA) Retinexbased deep learning methods by large margins on various datasets. Especially on SID [9], SDSD [48]-indoor, and LOL-v2 [59]-synthetic, the improvements are over 6 dB.

Our contributions can be summarized as follows: • We propose the first Transformer-based algorithm, Retinexformer, for low-light image enhancement. • We formulate a one-stage Retinex-based low-light enhancement framework, ORF, that enjoys an easy onestage training process and models the corruptions well. • We design a new self-attention mechanism, IG-MSA, that utilizes the illumination information as a key clue to guide the modeling of long-range dependences. • Quantitative and qualitative experiments show that our Retinexformer outperforms SOTA methods on thirteen datasets. The results of user study and low-light detection also suggest the practical values of our method.

---

# Network In Network

## Abstract

We propose a novel deep network structure called "Network In Network"(NIN) to enhance model discriminability for local patches within the receptive field. The conventional convolutional layer uses linear filters followed by a nonlinear activation function to scan the input. Instead, we build micro neural networks with more complex structures to abstract the data within the receptive field. We instantiate the micro neural network with a multilayer perceptron, which is a potent function approximator. The feature maps are obtained by sliding the micro networks over the input in a similar manner as CNN; they are then fed into the next layer. Deep NIN can be implemented by stacking mutiple of the above described structure. With enhanced local modeling via the micro network, we are able to utilize global average pooling over feature maps in the classification layer, which is easier to interpret and less prone to overfitting than traditional fully connected layers. We demonstrated the state-of-the-art classification performances with NIN on CIFAR-10 and CIFAR-100, and reasonable performances on SVHN and MNIST datasets.

## Introduction

Convolutional neural networks (CNNs) consist of alternating convolutional layers and pooling layers. Convolution layers take inner product of the linear filter and the underlying receptive field followed by a nonlinear activation function at every local portion of the input. The resulting outputs are called feature maps.

The convolution filter in CNN is a generalized linear model (GLM) for the underlying data patch, and we argue that the level of abstraction is low with GLM. By abstraction we mean that the feature is invariant to the variants of the same concept. Replacing the GLM with a more potent nonlinear function approximator can enhance the abstraction ability of the local model. GLM can achieve a good extent of abstraction when the samples of the latent concepts are linearly separable, i.e. the variants of the concepts all live on one side of the separation plane defined by the GLM. Thus conventional CNN implicitly makes the assumption that the latent concepts are linearly separable. However, the data for the same concept often live on a nonlinear manifold, therefore the representations that capture these concepts are generally highly nonlinear function of the input. In NIN, the GLM is replaced with a "micro network" structure which is a general nonlinear function approximator. In this work, we choose multilayer perceptron as the instantiation of the micro network, which is a universal function approximator and a neural network trainable by back-propagation.

The resulting structure which we call an mlpconv layer is compared with CNN in Figure 1. Both the linear convolutional layer and the mlpconv layer map the local receptive field to an output feature vector. The mlpconv maps the input local patch to the output feature vector with a multilayer perceptron (MLP) consisting of multiple fully connected layers with nonlinear activation functions. The MLP is shared among all local receptive fields. The feature maps are obtained by sliding the MLP over the input in a similar manner as CNN and are then fed into the next layer. The overall structure of the NIN is the stacking of multiple mlpconv layers. It is called "Network In Network" (NIN) as we have micro networks (MLP), which are composing elements of the overall deep network, within mlpconv layers.

Instead of adopting the traditional fully connected layers for classification in CNN, we directly output the spatial average of the feature maps from the last mlpconv layer as the confidence of categories via a global average pooling layer, and then the resulting vector is fed into the softmax layer. In traditional CNN, it is difficult to interpret how the category level information from the objective cost layer is passed back to the previous convolution layer due to the fully connected layers which act as a black box in between. In contrast, global average pooling is more meaningful and interpretable as it enforces correspondance between feature maps and categories, which is made possible by a stronger local modeling using the micro network. Furthermore, the fully connected layers are prone to overfitting and heavily depend on dropout regularization, while global average pooling is itself a structural regularizer, which natively prevents overfitting for the overall structure.

Sliding a micro network over the input has been proposed in several previous works. For example, the Structured Multilayer Perceptron (SMLP) applies a shared multilayer perceptron on different patches of the input image; in another work, a neural network based filter is trained for face detection. However, they are both designed for specific problems and both contain only one layer of the sliding network structure. NIN is proposed from a more general perspective, the micro network is integrated into CNN structure in persuit of better abstractions for all levels of features.

---
# SPJFNet: Self-Mining Prior-Guided Joint Frequency Enhancement for Ultra-Efficient Dark Image Restoration

## Abstract

Current dark image restoration methods suffer from severe efficiency bottlenecks, primarily stemming from: (1) computational burden and error correction costs associated with reliance on external priors (manual or cross-modal); (2) redundant operations in complex multi-stage enhancement pipelines; and (3) indiscriminate processing across frequency components in frequency-domain methods, leading to excessive global computational demands. To address these challenges, we propose an Efficient Self-Mining Prior-Guided Joint Frequency Enhancement Network (SPJFNet). Specifically, we first introduce a Self-Mining Guidance Module (SMGM) that generates lightweight endogenous guidance directly from the network, eliminating dependence on external priors and thereby bypassing error correction overhead while improving inference speed. Second, through meticulous analysis of different frequency domain characteristics, we reconstruct and compress multi-level operation chains into a single efficient operation via lossless wavelet decomposition and joint Fourier-based advantageous frequency enhancement, significantly reducing parameters. Building upon this foundation, we propose a Dual-Frequency Guidance Framework (DFGF) that strategically deploys specialized high/low frequency branches (wavelet-domain high-frequency enhancement and Fourier-domain low-frequency restoration), decoupling frequency processing to substantially reduce computational complexity. Rigorous evaluation across multiple benchmarks demonstrates that SPJFNet not only surpasses state-of-the-art performance but also achieves significant efficiency improvements, substantially reducing model complexity and computational overhead. Code is available at [https://github.com/bywlzts/SPJFNet](https://github.com/bywlzts/SPJFNet).

## Introduction

Images taken in dark conditions often suffer from insufficient illumination, degrading visual quality and hindering downstream vision applications like autonomous driving and object detection. With the rise of deep learning, learning-based methods have surpassed traditional approaches but often come with high computational complexity. Given the limited cues in extremely dark images, many methods introduce pre-trained models or additional prediction networks to generate modal priors. However, pretrained models and the predictive network are likely to suffer from generalization issues, resulting in accumulated errors and high computational overhead.

Recently, Fourier-based methods have gained significant attention due to their efficiency in image restoration tasks. FourLLIE demonstrated that Fourier domain processing can handle global information as effectively as transformer while significantly reducing parameter overhead by replacing transformer layers.

Based on the limitations, we present our motivations: First, there are two primary approaches for introducing prior clues in dark image restoration. The first approach utilizes pre-trained models to generate modal priors, where DMFourLLIE leverages pre-trained models to generate infrared and depth map modal priors for enhanced guidance of phase component recovery, but neglects the generalization error issues introduced by pre-trained models. The second approach introduces prediction modules to generate priors, where SMGLLE designs a dedicated module to generate structural priors, but the predicted results are imprecise and introduce substantial additional overhead. This raises the question: How can we efficiently and accurately mine high-quality endogenous priors?

Motivated by the aforementioned questions, we propose an efficient Self-Mining Prior-Guided Joint Frequency Enhancement Network (SPJFNet). First, we utilize lossless wavelet downsampling to reduce images to a lower-resolution space, thereby reducing computational overhead. In this lower-resolution space, we develop a Self-Mining Guidance Module (SMGM), which consists of a lightweight gamma enhancement network that mines the image's intrinsic gradient information to generate high-quality endogenous priors. Subsequently, we introduce a Dual Frequency Guidance Framework (DFGF) that strategically exploits the complementary strengths of wavelet and Fourier domains through a dual-branch architecture. The key innovation lies in our frequency-adaptive processing strategy: wavelet low-frequency components are transformed into the Fourier domain where compact structures can efficiently handle global illumination recovery through precise amplitude and phase reconstruction guided by self-mined priors. Meanwhile, wavelet high-frequency components remain in their native domain, where they are adaptively enhanced through a gradient-prior-guided gating mechanism that preserves fine-grained textural details. This dual-domain approach maximizes the inherent advantages of each transform while mitigating their respective limitations, achieving superior reconstruction efficiency and quality.

Overall, our contributions are summarized as follows:

- We introduce SMGM, which extracts endogenous priors directly from dark images, eliminating external dependencies. Our DFGF enables frequency-adaptive enhancement through prior guidance, ensuring robust reconstruction without added computational burden.
    
- We propose SPJFNet, a novel dual-domain architecture that leverages the strengths of wavelet and Fourier transforms: efficiently modeling low-frequency components in the Fourier domain while preserving high-frequency details in the wavelet domain, thus achieving superior restoration with reduced complexity.
    
- Extensive experiments validate our state-of-the-art performance across multiple benchmarks, demonstrating significantly reduced model complexity. Our plug-and-play framework seamlessly integrates into existing frequency-domain methods, delivering both performance gains and computational efficiency.

---
# Towards Perfection: Building Inter-component Mutual Correction for Retinex-based Low-light Image Enhancement

## Abstract

In low-light image enhancement, Retinex-based deep learning methods have garnered significant attention due to their exceptional interpretability. These methods decompose images into mutually independent illumination and reflectance components, allows each component to be enhanced separately. In fact, achieving perfect decomposition of illumination and reflectance components proves to be quite challenging, with some residuals still existing after decomposition. In this paper, we formally name these residuals as inter-component residuals (ICR), which has been largely underestimated by previous methods. In our investigation, ICR not only affects the accuracy of the decomposition but also causes enhanced components to deviate from the ideal outcome, ultimately reducing the final synthesized image quality. To address this issue, we propose a novel Inter-correction Retinex model (IRetinex) to alleviate ICR during the decomposition and enhancement stage. In the decomposition stage, we leverage inter-component residual reduction module to reduce the feature similarity between illumination and reflectance components. In the enhancement stage, we utilize the feature similarity between the two components to detect and mitigate the impact of ICR within each enhancement unit. Extensive experiments on three low-light benchmark datasets demonstrated that by reducing ICR, our method outperforms state-of-the-art approaches both qualitatively and quantitatively. Our code is available at: [https://github.com/caoluyang0830/IRetinex.git](https://github.com/caoluyang0830/IRetinex.git).

## Introduction

Low-light image enhancement is an important yet challenging task in computer vision. It aims to improve the poor visibility and low contrast of low-light images and restore the corruptions hidden in the dark or introduced by the light-up process. Among existing approaches, the methods based on Retinex theory have garnered significant attention due to its promising results. Mathematically, Retinex theory decomposes the image I into illumination L and reflectance R components: I = L ⊙ R, where ⊙ represents element-wise multiplication. Based on Retinex theory, low-light image enhancements contain three stages: 1) a decomposition stage to decouple the low-light image into illumination component and reflectance component, 2) an enhancement stage to adjust illumination and reflectance components separately, and 3) a synthesis stage to combine two components for the normal-light image.

Despite their success, we found that existing methods are based on a fundamental yet idealized assumption: illumination and reflectance are mutually independent. Specifically, this assumption implies that, after decomposition, illumination and reflectance components should be perfectly separated, thus allowing their independent enhancement. However, in real cases, Retinex decomposition is a highly ill-posed problem and there is no available ground truth for the illumination and reflectance components, which makes perfect decomposition challenging to achieve.

To further illustrate this issue, we visualize the decomposition results of SOTA Retinex-based learning methods. Evidently, texture details (part of reflectance) bleed into the illumination component, and light and shadows (part of illumination) remain in the reflectance component. During the enhancement stage, due to the existence of these residuals, the enhancement components inevitably deviate from the ideal outcome, ultimately degrading the quality of the final synthesized image, manifesting as color distortion and detail loss. In this paper, we investigate this issue that has been largely underestimated by previous methods, providing the first formal definition of these residuals as inter-component residuals (ICR).

Based on above discussion, within the Retinex-based LLIE pipeline, reducing ICRs during the decomposition stage and mitigating their impact during the enhancement stage are crucial. Fortunately, we discovered that ICRs in the illumination component exhibit feature similarities to those in the corresponding reflectance component, and vice versa. This observation naturally prompts us to consider: Could we leverage these feature similarities to identify and mitigate ICRs?

Intuitively, during the decomposition stage, the low feature similarity between the illumination and reflectance components indicates that each component retains its inherent distinctive features, with no significant residuals. In the enhancement stage, the similar features actually reveal different component missed features. By supplementing these features into corresponding components, we could further drive the illumination and reflectance components closer to their ideal outcomes. Therefore, reducing feature similarity is beneficial to achieving perfect decomposition. Moreover, the similar features could facilitates mutual correction between the two components, thereby mitigating the ICR impact during the enhancement stage.

Based on the above analysis, we propose a novel inter-correction Retinex model (IRetinex) that systematically addresses ICR issues across both decomposition and enhancement stages. During the decomposition stage, we propose an innovative inter-component residual reduction module designed to minimize feature similarity between illumination and reflectance components, reducing initial decomposed ICR. During enhancement, we propose a novel module for residual mitigation and component enhancement. By leveraging feature similarity, this module identifies and mitigates ICRs across components and recovering buried details in low-light conditions. Additionally, we propose a Retinex-based multi-scale consistency loss that provides targeted supervision for the ICR identification process.

Our contributions are summarized as follows:

- An underestimated issue: Imperfect decomposition results in ICR, causing deviations in illumination and reflectance components from their ideal states, ultimately degrading synthesized image quality.
    
- An innovative solution: ICR could be identified and mitigated by leveraging feature similarities between illumination and reflectance components.
    
- A novel LLIE framework: We propose an inter-correction Retinex framework that addresses ICR issue during both decomposition and enhancement stages.
    
- Extensive experiments demonstrate that IRetinex outperforms state-of-the-art methods in both qualitative and quantitative evaluations. Moreover, it satisfies the requirements of computer vision applications for discriminative feature representation, achieving multimodal perception optimization.
    

---

# HVI: A New Color Space for Low-light Image Enhancement

## Abstract

Low-Light Image Enhancement (LLIE) is a crucial computer vision task that aims to restore detailed visual information from corrupted low-light images. Many existing LLIE methods are based on standard RGB (sRGB) space, which often produce color bias and brightness artifacts due to inherent high color sensitivity in sRGB. While converting the images using Hue, Saturation and Value (HSV) color space helps resolve the brightness issue, it introduces significant red and black noise artifacts. To address this issue, we propose a new color space for LLIE, namely Horizontal/Vertical-Intensity (HVI), defined by polarized HS maps and learnable intensity. The former enforces small distances for red coordinates to remove the red artifacts, while the latter compresses the low-light regions to remove the black artifacts. To fully leverage the chromatic and intensity information, a novel Color and Intensity Decoupling Network (CIDNet) is further introduced to learn accurate photometric mapping function under different lighting conditions in the HVI space. Comprehensive results from benchmark and ablation experiments show that the proposed HVI color space with CIDNet outperforms the state-of-the-art methods on 10 datasets. The code is available at [https://github.com/Fediory/HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet).

## Introduction

Under low-light conditions, imaging sensors often capture weak light signals with severe noise, resulting in poor visual quality for low-light images. Obtaining high-quality images from such degraded images necessitates Low-Light Image Enhancement (LLIE) that aims at improving the image brightness while simultaneously reducing the impact of noise and color bias.

The majority of existing LLIE approaches focus on finding an appropriate image brightness, which is typically done by employing deep neural networks to learn a mapping relationship between low-light images and normal-light images within the standard RGB (sRGB) space. However, the image brightness exhibits a strong coupling with the color from the three sRGB channels, a.k.a. high color sensitivity, causing an obvious color distortion of the restored image in these LLIE methods.

Inspired by Kubelka-Munk theory, recent methods have sought to transform images from the sRGB color space to the Hue, Saturation and Value (HSV) color space. These methods help achieve the brightness enhancement more accurately, but they amplify local color space noise, introducing severe artifacts in the results. The transformation from sRGB to the HSV disrupts the continuity of the red and black color, resulting in increased Euclidean distances for similar color and the introduction of artifacts in the final images. These two types of noise can cause severe artifacts in the enhancement of red-dominated or extremely dark images.

To address these issues, we introduce a new color space named Horizontal/Vertical-Intensity (HVI), designed specifically for the LLIE task. The key intuition is that minimizing color space noise, by reducing Euclidean distances in similar colors. To this end, we polarize in the Hue and Saturation (HS) plane to enforce smaller distances for similar red point coordinates, which eliminates the red discontinuity noise in the primary HSV space. For the black plane noise issue, we introduce a trainable darkness density parameter k and its corresponding adaptive intensity collapse function C_k, which compresses the radius of low-light regions to zero, with the flexibility to gradually expand to the value of one as the intensity increases. This helps remove the black noise artifacts while maintaining the primary image appearance.

We further propose an LLIE network, named Color and Intensity Decoupling Network (CIDNet), to leverage the chromatic and intensity information for optimizing the color and intensity in the HVI space. After transforming the image into the HVI space, CIDNet leverages two network branches, namely HV-branch and intensity-branch, to respectively model decoupled color and brightness information for learning accurate photometric mapping under different lighting conditions and restoring more natural colors.

Our contributions can be summarized as follows:

- We introduce a new HVI color space for the LLIE task, which is uniquely defined by polarized HS and trainable intensity. This offers an effective tool that eliminates the color space noise arising from the HSV space, largely enhancing the brightness of the low-light images.
    
- We further propose a novel LLIE network, CIDNet, to concurrently model the intensity and chromatic of low-light images in the HVI space. Despite being lightweight and computationally-efficient with relatively small parameters (1.88M) and computational loads (7.57GFLOPs), it enables the learning of accurate photometric mapping under different lighting conditions.
    
- Comprehensive results from benchmark and ablation experiments show that the proposed HVI color space with CIDNet outperforms the state-of-the-art methods on 10 datasets, demonstrating its effectiveness and superiority.

---
# References

## 1. VRT: A Video Restoration Transformer

[5] Jose Caballero, Christian Ledig, Andrew Aitken, Alejandro Acosta, Johannes Totz, Zehan Wang, and Wenzhe Shi. Real-time video super-resolution with spatio-temporal networks and motion compensation. In IEEE Conference on Computer Vision and Pattern Recognition, pages 4778–4787, 2017.

[6] Jiezhang Cao, Yawei Li, Kai Zhang, and Luc Van Gool. Video super-resolution transformer. arXiv preprint arXiv:2106.06847, 2021.

[8] Kelvin CK Chan, Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. Basicvsr: The search for essential components in video super-resolution and beyond. In IEEE Conference on Computer Vision and Pattern Recognition, pages 4947–4956, 2021.

[10] Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Basicvsr++: Improving video superresolution with enhanced propagation and alignment. arXiv preprint arXiv:2104.13371, 2021.

[20] Dario Fuoli, Shuhang Gu, and Radu Timofte. Efficient video super-resolution through recurrent latent space propagation. In IEEE International Conference on Computer Vision Workshop, pages 3476–3485, 2019.

[22] Alexander Greaves-Tunnell and Zaid Harchaoui. A statistical investigation of long memory in language and music. In International Conference on Machine Learning, pages 2394–2403, 2019.

[25] Muhammad Haris, Gregory Shakhnarovich, and Norimichi Ukita. Recurrent back-projection network for video superresolution. In IEEE Conference on Computer Vision and Pattern Recognition, pages 3897–3906, 2019.

[27] Yan Huang, Wei Wang, and Liang Wang. Bidirectional recurrent convolutional networks for multi-frame superresolution. Advances in Neural Information Processing Systems, 28:235–243, 2015.

[28] Yan Huang, Wei Wang, and Liang Wang. Video superresolution via bidirectional recurrent convolutional networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4):1015–1028, 2017.

[29] Takashi Isobe, Xu Jia, Shuhang Gu, Songjiang Li, Shengjin Wang, and Qi Tian. Video super-resolution with recurrent structure-detail network. In European Conference on Computer Vision, pages 645–660, 2020.

[30] Takashi Isobe, Songjiang Li, Xu Jia, Shanxin Yuan, Gregory Slabaugh, Chunjing Xu, Ya-Li Li, Shengjin Wang, and Qi Tian. Video super-resolution with temporal group attention. In IEEE Conference on Computer Vision and Pattern Recognition, pages 8008–8017, 2020.

[31] Takashi Isobe, Fang Zhu, Xu Jia, and Shengjin Wang. Revisiting temporal modeling for video super-resolution. arXiv preprint arXiv:2008.05765, 2020.

[41] Dongxu Li, Chenchen Xu, Kaihao Zhang, Xin Yu, Yiran Zhong, Wenqi Ren, Hanna Suominen, and Hongdong Li. Arvo: Learning all-range volumetric correspondence for video deblurring. In IEEE Conference on Computer Vision and Pattern Recognition, pages 7721–7731, 2021.

[44] Wenbo Li, Xin Tao, Taian Guo, Lu Qi, Jiangbo Lu, and Jiaya Jia. Mucan: Multi-correspondence aggregation network for video super-resolution. In European Conference on Computer Vision, pages 335–351, 2020.

[52] （注：原文参考文献中未完整提取该编号文献）

[67] （注：原文参考文献中未完整提取该编号文献）

[76] Mehdi SM Sajjadi, Raviteja Vemulapalli, and Matthew Brown. Frame-recurrent video super-resolution. In IEEE Conference on Computer Vision and Pattern Recognition, pages 6626–6634, 2018.

[80] （注：原文参考文献中未完整提取该编号文献）

[82] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data. In IEEE International Conference on Computer Vision Workshops, 2021.

[92] Tiancheng Tao, Xin Tao, Xiaoyong Shen, and Jiaya Jia. TDAN: Temporally-deformable alignment network for video super-resolution. In IEEE Conference on Computer Vision and Pattern Recognition, pages 3360–3369, 2020.

[94] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017.

[99] Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, and Yu Qiao. ESRGAN: Enhanced super-resolution generative adversarial networks. In European Conference on Computer Vision Workshops, 2018.

[122] Shangchen Zhou, Kelvin CK Chan, Xiangyu Xu, and Chen Change Loy. Spatio-temporal transformer for video super-resolution. In IEEE Conference on Computer Vision and Pattern Recognition, pages 10604–10613, 2021.

[123] Xiaotong Li, Xin Tao, and Jiaya Jia. Video super-resolution with dynamic filter generation. In IEEE Conference on Computer Vision and Pattern Recognition, pages 10614–10623, 2021.

## 2. Low-Light Video Enhancement via Spatial-Temporal Consistent Decomposition

[1] Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, and Jiaya Jia. SNR-Aware low-light image enhancement. In CVPR, 2022.

[2] Ruixing Wang, Xiaogang Xu, Chi-Wing Fu, Jiangbo Lu, Bei Yu, and Jiaya Jia. Seeing dynamic scene in the dark: High-quality video dataset with mechatronic alignment. In ICCV, 2021.

[3] Andrey Ignatov, Nikolay Kobyshev, Radu Timofte, Kenneth Vanhoey, and Luc Van Gool. DSLR-quality photos on mobile devices with deep convolutional networks. In ICCV, 2017.

[4] Samuel W. Hasinoff, Dillon Sharlet, Ryan Geiss, Andrew Adams, Barron Jonathan T., Florian Kainz, Jiawen Chen, and Marc Levoy. Burst photography for high dynamic range and low-light imaging on mobile cameras. ACM Transactions on Graphics (Proc. SIGGRAPH Asia), 2016.

[5] Long Ma, Tengyu Ma, Risheng Liu, Xin Fan, and Zhongxuan Luo. Toward fast, flexible, and robust low-light image enhancement. In CVPR, 2022.

[6] Wenjing Wang, Xinhao Wang, Wenhan Yang, and Jiaying Liu. Unsupervised face detection in the dark. IEEE TPAMI, 2022.

[7] Huiyuan Fu, Wenkai Zheng, Xicong Wang, Jiaxuan Wang, Heng Zhang, and Huadong Ma. Dancing in the dark: A benchmark towards general low-light video enhancement. In ICCV, 2023.

[8] Yirui Wu, Haifeng Guo, Chinmay Chakraborty, Mohammad Khosravi, Stefano Berretti, and Shaohua Wan. Edge computing driven low-light image dynamic enhancement for object detection. IEEE Transactions on Network Science and Engineering, 2022.

[9] A.L. Yuille. The lambertian reflectance model. [https://www.cs.jhu.edu/](https://www.cs.jhu.edu/) ayuille/courses/Stat238Winter12/lecture10.pdf, 2012.

[10] Takuya Narihira, Michael Maire, and Stella X Yu. Direct intrinsics: Learning albedo-shading decomposition by convolutional regression. In CVPR, 2015.

[11] Xiaoqian Lv, Shengping Zhang, Chenyang Wang, Weigang Zhang, Hongxun Yao, and Qingming Huang. Unsupervised low-light video enhancement with spatial-temporal co-attention transformer. TIP, 2023.

## 3. Learning Temporal Consistency for Low Light Video Enhancement from Single Images

[3] Chen Chen, Qifeng Chen, Minh N Do, and Vladlen Koltun. Seeing motion in the dark. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2019.

[4] Chen Chen, Qifeng Chen Chen, Jia Xu, and Vladlen Koltun. Learning to see in the dark. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2018.

[9] Haiyang Jiang and Yinqiang Zheng. Learning to see moving objects in the dark. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2019.

[25] Li Tao, Chuang Zhu, Jiawen Song, Tao Lu, Huizhu Jia, and Xiaodong Xie. Low-light image enhancement using cnn and bright channel prior. In IEEE International Conference on Image Processing (ICIP). IEEE, 2017.

[26] Li Tao, Chuang Zhu, Guoqing Xiang, Yuan Li, Huizhu Jia, and Xiaodong Xie. Llcnn: A convolutional neural network for low-light image enhancement. In Proceedings of the IEEE Visual Communications and Image Processing (VCIP), 2017.

[27] Ruixing Wang, Qing Zhang, Chi-Wing Fu, Xiaoyong Shen, Wei-Shi Zheng, and Jiaya Jia. Underexposed photo enhancement using deep illumination estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2019.

## 4. Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement

[9] Chen Chen, Qifeng Chen, Minh N Do, and Vladlen Koltun. Seeing motion in the dark. In ICCV, 2019.

[27] Edwin H Land. The retinex theory of color vision. Scientific american, 1977.

[48] Ruixing Wang, Xiaogang Xu, Chi-Wing Fu, Jiangbo Lu, Bei Yu, and Jiaya Jia. Seeing dynamic scene in the dark: A highquality video dataset with mechatronic alignment. In ICCV, 2021.

[54] Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. Deep retinex decomposition for low-light enhancement. In BMVC, 2018.

[57] Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, and Jiaya Jia. Snr-aware low-light image enhancement. In CVPR, 2022.

[59] Wenhan Yang, Wenjing Wang, Haofeng Huang, Shiqi Wang, and Jiaying Liu. Sparse gradient regularized deep retinex network for robust low-light image enhancement. TIP, 2021.

[65] Yonghua Zhang, Xiaojie Guo, Jiayi Ma, Wei Liu, and Jiawan Zhang. Beyond brightening low-light images. IJCV, 2021.

[66] Yonghua Zhang, Jiawan Zhang, and Xiaojie Guo. Kindling the darkness: A practical low-light image enhancer. In ACM MM, 2019.