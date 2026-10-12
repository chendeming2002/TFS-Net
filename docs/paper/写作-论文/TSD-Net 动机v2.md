# TSD-Net: Triple Source Decoupled Restoration for Low-Light Video Enhancement

Anonymous Submission

Anonymous affiliation

## 1 Introduction

Low-light video enhancement (LLVE) aims to recover visibility, suppress noise, and maintain temporal coherence under insufficient illumination. LLVE is a long-standing yet intricate problem, challenge not only human visual perception but also other vision tasks like nighttime object detection[1]. Under low-light conditions, the extremely limited photon count leads to an inherently low signal-to-noise ratio (SNR), and amplifying illumination to improve visibility further magnifies the embedded noise[2]. Unique to video, independently realized per-frame noise introduces flickering artifacts that severely undermine the enhancement quality[3].

To tackle these challenges, many deep-learning based methods for LLVE have been proposed, which can be broadly categorized into three paradigms: sequential[4,5,6], implicit[7,8], and parallel structure[9,10].

Sequential methods adopt a cascaded strategy that first enhances illumination and subsequently applies explicit denoising to the amplified frames. While this design offers conceptual clarity, it is vulnerable to error accumulation, as imperfections in the illumination correction are propagated and amplified in the denoising stage.

Implicit methods integrate noise suppression directly into the model architecture, eliminating the need for a separate denoising stage. Most of these approaches adopt Transformers to capture noise distributions and exploit multi-frame information to reduce stochastic noise and improve SNR. However, the computational cost of Transformers is already high, and the additional multi-frame attention required for video data further exacerbates this burden. As a result, both training and inference demand excessive hardware resources, which severely hinders their deployment in real-world applications.

Parallel methods attempt to address these limitations by disentangling the mixed degradation into multiple distinct components and assigning dedicated branches for source-specific restoration. This modular design offers several compelling advantages, including improved specialization, more flexible optimization, and enhanced interpretability. However, its effectiveness fundamentally depends on whether the underlying degradation sources can be properly characterized and decoupled. Due to the highly coupled nature of low-light video degradation, inappropriate decomposition may lead to incomplete modeling of degradation interactions and limit the ability of parallel frameworks to achieve comprehensive restoration.

Existing works exploring divide-and-conquer denoising strategies primarily fall into three categories: decomposition tailored to specific datasets[2, 11], partitioning based on mathematical or frequency-domain statistical characteristics[9, 12], and division in diffusion-based denoising methods that hinges on the progressive timesteps[13]. However, these paradigms face insurmountable challenges in low-light video denoising. Source-driven decomposition relies on inaccessible intermediate imaging pipeline representations in most real-world LLVE scenarios, where only sRGB frames are available[11]; spectral/statistical decomposition suffers from severe signal-noise spectral overlap and violated stationarity assumptions due to low signal energy and spatiotemporally varying noise in low-light conditions[12]; diffusion-based methods lack interpretable temporal-aware design while noise is frame-independently generated[13]. These limitations highlight the urgent need for a more adaptive and video-specific denoising decomposition strategy.

To address the above limitation of existing degradation decomposition strategies, we propose Triple Source Decoupled Network (TSD-Net) for low-light video enhancement. The proposed Triple Source Decoupled Restoration (TSDR) framework decomposes the entangled low-light video degradation into three physically meaningful sources—illumination, imaging noise, and motion—according to their distinct spatio-temporal characteristics. Each source is processed through a dedicated lightweight branch, and the restored components are progressively integrated through a physically motivated stage-wise fusion strategy.

For the first challenge—effectively decoupling highly coupled degradation sources while maintaining their complementary interactions—we adopt a parallel source-aware encoding strategy with task-specific branching. TSD-Net routes the shared shallow features extracted by a lightweight encoder simultaneously into three dedicated branches (ISPN, NDPN, and MCPN), each specializing in illumination, imaging noise, and motion restoration. This design prevents different degradation sources from interfering with each other's representations while preserving source-specific optimization. The branch outputs are subsequently re-integrated by the Stage-wise Guided Restoration and Fusion (SGRF) module according to their restoration dependencies, enabling physically consistent collaboration among multiple sources.

For the second challenge—effectively capturing global temporal interactions among multiple degradation sources—we introduce the Temporal Correspondence and Alignment (TCA) module. Although source-specific branches enable targeted degradation restoration, low-light video enhancement still requires comprehensive modeling of inter-frame relationships to maintain temporal consistency. Existing temporal modeling approaches often rely on explicit motion estimation or computationally intensive spatio-temporal attention, which may limit their ability to efficiently capture global frame-wise dependencies. To address this issue, TCA employs RWKV-based linear-complexity global interaction modeling to establish full-frame cross-frame correspondence, followed by an explicit temporal correspondence matrix to characterize frame-to-frame affinity. This design enables efficient global temporal interaction modeling without relying on optical flow or deformable alignment, providing effective temporal guidance for source-aware restoration.

The decomposition of TSDR is motivated by the observation that different low-light video degradation sources exhibit distinct spatio-temporal characteristics and therefore require different restoration strategies. Specifically, we characterize degradation factors from two complementary perspectives: temporal independence, which describes whether a degradation exhibits statistical independence across consecutive frames, and spatial selectivity, which indicates whether the degradation is globally distributed or spatially localized. Based on these two dimensions, low-light video degradation can be categorized into four canonical degradation types [14, 15]:

(I) globally distributed and temporally independent degradation, corresponding to sensor-related random noise such as read noise;

(II) spatially selective and temporally independent degradation, corresponding to signal-dependent noise such as photon shot noise that becomes more prominent in low-intensity regions;

(III) globally distributed and temporally correlated degradation, corresponding to illumination variation across frames; and

(IV) spatially localized and temporally correlated degradation, corresponding to motion-induced disturbance caused by object or camera movement.

These four degradation types can be further grouped into three restoration-oriented sources according to their underlying recovery mechanisms. Types I and II, although differing in spatial distribution, share the common property of temporal independence and represent stochastic sensor noise. Both can be effectively suppressed through temporal aggregation and denoising strategies, and therefore are unified as the imaging noise source[14]. Type III corresponds to the illumination source, representing a slowly varying global bias that can be corrected through illumination estimation and temporal consistency constraints. Type IV forms the motion source, representing structured temporal displacement that requires explicit correspondence modeling or motion compensation rather than noise suppression[16]. Importantly, these sources cannot be further merged without losing restoration specificity: illumination variation and motion disturbance, although both temporally correlated, require fundamentally different processing strategies, while stochastic imaging noise cannot be handled through temporal modeling designed for structured degradation[15]. Therefore, TSDR adopts a three-source decomposition consisting of imaging noise, illumination, and motion, providing a compact yet physically meaningful representation of low-light video degradation.

Guided by this framework, TSD-Net instantiates the tri-source decomposition through a Wavelet Feature Router (WFR) for frequency-aware source routing, TCA for global cross-frame correspondence modeling, and a Degradation Prior Estimator (DPE) for source prior estimation. These components feed three parallel specialized branches (ISPN, NDPN, and MCPN), whose outputs are progressively re-integrated by SGRF following the physically motivated restoration order.

Our main contributions are as follows:
- We reformulate low-light video enhancement as a divide-and-conquer denoising problem under low-light scenarios and propose the TSDR framework. This framework decomposition extends beyond existing two-source paradigms.
- We design TSD-Net, an end-to-end architecture that instantiates the TSDR principle through wavelet-based subband routing, RWKV-style linear-complexity temporal correspondence, and three dedicated source-specific branches (ISPN, NDPN, MCPN) connected by a physically motivated stage-wise fusion (SGRF) that prevents noise amplification at its root.
- We introduce a two-phase progressive training schedule with perceptual decoupling to resolve gradient conflicts among competing restoration objectives.

# Reference
[1]  Cai, Yuanhao, et al. "Retinexformer: One-stage retinex-based transformer 
     for low-light image enhancement." Proceedings of the IEEE/CVF International 
     Conference on Computer Vision (ICCV). 2023.

[2]  Wei, Kaixuan, et al. "A physics-based noise formation model for extreme 
     low-light raw denoising." Proceedings of the IEEE/CVF Conference on 
     Computer Vision and Pattern Recognition (CVPR). 2020.

[3]  Wang, Ruixing, et al. "Seeing dynamic scene in the dark: A high-quality 
     video dataset with mechatronic alignment." Proceedings of the IEEE/CVF 
     International Conference on Computer Vision (ICCV). 2021.

[4]  Yang, Wenhan, et al. "Sparse gradient regularized deep retinex network 
     for robust low-light image enhancement." IEEE Transactions on Image 
     Processing 30 (2021): 2072-2086.

[5]  Li, Huaqiu, and Haoqian Wang. "Interpretable unsupervised joint denoising 
     and enhancement for real-world low-light scenarios." International 
     Conference on Learning Representations (ICLR). 2025.

[6]  Wei, Chen, et al. "Deep retinex decomposition for low-light enhancement." 
     Proceedings of the British Machine Vision Conference (BMVC). 2018.

[7]  Liang, Jingyun, et al. "VRT: A video restoration transformer." IEEE 
     Transactions on Image Processing 33 (2024): 2171-2182.

[8]  Liang, Jingyun, et al. "Recurrent video restoration transformer with 
     guided deformable attention." Advances in Neural Information Processing 
     Systems 35 (2022): 378-393.

[9]  Feng, Hansen, et al. "Learning physics-informed noise models from dark 
     frames for low-light raw image denoising." IEEE Transactions on Pattern 
     Analysis and Machine Intelligence (2025).

[10] Li, Yuezhou, et al. "Perceptual decoupling with heterogeneous auxiliary 
     tasks for joint low-light image enhancement and deblurring." IEEE 
     Transactions on Multimedia 26 (2024): 6663-6675.

[11] Zhang, Ruikang, et al. "DualDn: Dual-domain denoising via differentiable 
     ISP." European Conference on Computer Vision (ECCV). Springer, 2024.

[12] Xu, Xiaogang, et al. "SNR-aware low-light image enhancement." 
     Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern 
     Recognition (CVPR). 2022.

[13] Liu, Jiawei, et al. "Residual denoising diffusion models." Proceedings 
     of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 
     (CVPR). 2024.

[14] Foi, Alessandro, et al. "Practical Poissonian-Gaussian noise modeling 
     and fitting for single-image raw-data." IEEE Transactions on Image 
     Processing 17.10 (2008): 1737-1754.

[15] Zhu, Lingyu, et al. "Temporally consistent enhancement of low-light 
     videos." International Journal of Computer Vision 132 (2024): 4015-4034.

[16] Li, Hao, et al. "Low-light video enhancement via spatial-temporal 
     consistent illumination decomposition." Proceedings of the International 
     Joint Conference on Artificial Intelligence (IJCAI). 2025.

