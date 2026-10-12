# Draft
## 1 Introduction
Low-light video enhancement is an important problem in computational photography, mobile imaging, and intelligent perception. Compared with low-light image enhancement, the video setting introduces an additional temporal dimension that is both a source of information and a source of difficulty. Neighboring frames provide redundant structure and context that can help recover details hidden in darkness, yet low-light videos also exhibit frame-to-frame brightness fluctuation, motion blur, sensor noise, rolling artifacts, and correspondence ambiguity.
【任务总括——弱光视频增强】弱光视频增强是计算摄影、移动成像及智能感知领域的重要课题。与弱光图像增强相比，视频场景引入了额外的时间维度——这一维度既是信息来源，也是挑战来源。相邻帧提供的冗余结构和上下文有助于恢复暗处隐藏的细节；但弱光视频还存在帧间亮度波动、运动模糊、传感器噪声、滚动伪影以及对应关系模糊等问题。

Single-image low-light enhancement methods, especially Retinex-based models, have shown strong ability to improve visibility by reasoning about illumination and reflectance. However, frame-wise application often produces temporal inconsistency. On the other hand, transformer-based video restoration methods such as VRT demonstrate the effectiveness of local spatio-temporal attention, but they are not explicitly designed for the illumination instability that dominates low-light videos.
【现有主流方法总括——单帧弱光增强方法】单帧弱光增强方法（尤其是基于Retinex模型的方法）通过分析光照条件和反射特性展现出显著的可见度提升能力。然而，逐帧应用这些方法往往会导致时间上的不一致性。另一方面，基于Transformer的视频修复方法（如 VRT）虽能有效利用局部时空注意力机制，但并未专门针对弱光视频中普遍存在的光照不稳定性进行设计。
【现有方法的问题】（）
We argue that low-light video enhancement should be reformulated as a source-decoupled restoration problem. In particular, illumination-source degradation and motion-source degradation behave differently across time and should not be mixed too early. Illumination changes mainly corrupt exposure consistency and invalidate brightness constancy, while motion degradation requires reliable temporal support and alignment.
我们认为，弱光视频增强应重新定义为一种源分离型修复问题：光照源劣化与运动源劣化在不同时间尺度上表现各异，不应过早混合处理；光照变化主要破坏曝光一致性并影响亮度恒定性，而运动劣化则需要可靠的时间支持与对齐处理。

Based on this observation, we propose MINS-Net, whose core is the Motion-Illumination Noise Separation (MINS) module. MINS first constructs illumination-invariant temporal correspondence, then derives source priors from correspondence entropy, and finally projects the center-frame priors back to neighboring frames to preserve multi-frame source-aware reasoning. The downstream restoration is divided into ISPN for illumination correction and MSPN for motion restoration, followed by a lightweight prior-guided reconstruction head.
【针对这些挑战，我们论文提出了】基于这一观察结果，我们提出了MINS-Net模型，其核心在于运动-光照噪声分离（MINS）模块。该模块首先构建光照不变的时间对应关系，随后从对应关系熵中推导出源信息先验，并将中心帧的先验信息反向投影至相邻帧，从而保持多帧源感知推理能力。下游图像恢复流程分为光照校正 ISPN 和运动恢复 MSPN 两个阶段，最后通过轻量级先验引导重建头完成最终处理。
【具体来说（针对各挑战）】（）
Our formulation offers a coherent narrative with clear physical interpretation and implementation feasibility. The main contributions are three-fold: first, we reformulate low-light video enhancement as a source-decoupled restoration problem; second, we design MINS for illumination-invariant correspondence learning, entropy-guided prior estimation, and backward prior projection; third, we build branch-specific restoration modules that turn these priors into temporally consistent final enhancement.
我们的框架具有清晰的物理意义和可实现性，形成了完整的理论体系。主要创新点体现在三个方面：首先，我们将弱光视频增强问题重新表述为源解耦的恢复任务；其次，设计了基于光照不变对应学习、熵引导先验估计及逆向先验投影的MINS模块；第三，构建了针对不同场景的专用恢复模块，将这些先验信息转化为时间一致性更强的最终增强效果。

---
# 框架
【a】概括现有弱光视频增强 # VRT: A Video Restoration Transformer
-->
【b】现有场景下的去噪——弱光视频/图片增强框架的去噪任务
-->
【c1】分治去噪——为什么要分治，以往的噪声是怎么分类的（分成几类来分治）？
【c2】挑战：现有的这些分类存在什么问题? it is still challenging to 干啥。下面具体解释这些 challenge 是什么：First … Second…Third
-->
【d】针对这些挑战，我们论文提出了
【e】具体来说，针对第一个挑战...
本文立足点：弱光视频增强，多分支去噪/分治去噪

---
# 各部分行文要求
【a】承上启下，概括弱光视频增强经典和现有SOTA方法。对VRT以及这类基于注意力的方法进行更详细的介绍，其他概括即可。
【b】为什么去噪对于弱光视频/图片增强任务很重要。然后是去噪功能在模型整体框架内的设计：1. 串行设计，先提亮再去噪。图片增强方面该框架的代表有RetinexNet(2018)、Retinexformer(2023)，视频增强方面你帮我检索找几篇影响力大的 2. 隐式设计，不设置一个专门去噪/修复细节的阶段，蕴含在模型框架（大多是由Transformer块组成的U-shaped Network，利用Transformer注意力机制的全局关联建模能力）内部 3. 以前两个思路类推，还能概括其他的设计范式吗？
【c1】1. 为什么要分治噪声？和不分治的做法相比有什么好处？在以往论文当中针对这一问题如何看待？我要引用哪些论文的成果来说明？ 2. 以往的噪声是怎么分类（1）按噪声源头分：这类方法从图像成像管道的物理过程出发，将噪声/退化按其来源位置、物理成因进行分离。DualDN(ECCV 2024)、PDHAT（TMM 2024）（2）按数学/统计/频域属性分源：这类方法通过对噪声/残差的统计和频域性质进行形式化分解。RDDM (Residual Denoising Diffusion Models) — CVPR 2024 、Invertible Disentanglement Network — IEEE TIP 2023、CRWKV — IJCAI 2025 （3）扩散过程重构解耦：许多基于扩散模型的去噪方法，根据扩散模型处理的阶段性特点解耦。Remix-DiT — NeurIPS 2024、DBBCS (Dual Brownian Bridge Coupled Sampling) — Electronics 2024、Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification — CVPR 2025

【c2】广泛检索，结合模型现有框架的设计出发点帮我概括
【d】、【e】根据模型已有理论框架，结合前文



