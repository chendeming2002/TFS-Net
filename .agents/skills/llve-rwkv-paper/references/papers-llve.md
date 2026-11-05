# LLVE 方法论文精读笔记（8 篇）

> 精读层级：**全文**＝arXiv/CVF 全文或本地全文笔记；**摘要级**＝仅摘要/出版商页；**未获取**＝全文不可得（引用措辞须保守）。
> 引用核对针对 v6/v7 草稿 [2]–[9]。

## Sequential 类（v7 归入 [3,4]）

### 1. RetinexNet [4]（摘要级，arXiv:1808.04560）

**Deep Retinex Decomposition for Low-Light Enhancement** · Wei Chen, Wenjing Wang, Wenhan Yang 等 · BMVC 2018 · arXiv:1808.04560（**旧稿若写 1808.04512 是错的**）

- Decom-Net：分解为反射率 + 光照；无真值分解，靠**成对低光/正常光图像的反射率一致性** + **光照平滑性**约束。
- Enhance-Net：对光照做亮度增强；"for joint denoising there is a denoising operation on reflectance"——去噪在反射率上，级联在增强之后。
- LOL 数据集（低光/正常光配对）的提出者。

**引用核对**：v7 归入 sequential（"first estimate and correct illumination ... then apply explicit denoising"）✅ 结构相符（分解 → 增强光照 → 反射率去噪）。

### 2. URetinex-Net [3]（未获取全文）

**Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement** · Yang Wenhan 等 · IEEE TIP 30 (2021): 2072–2086

- 按 715 引用核对笔记：稀疏梯度正则的 Retinex 分解；顺序式方法代表。
- 未上 arXiv；IEEE 全文不可得。**写 §2.1 时只用"Retinex 分解 + 稀疏梯度正则"这一层级描述，不写未核实的细节。**

**引用核对**：归入 sequential [3,4] ✅（与 715 笔记一致）。

## Implicit 类（v7 归入 [5,6]）

### 3. VRT [5]（本地全文笔记：摘要 + 引言）

**VRT: A Video Restoration Transformer** · Liang Jingyun 等 · IEEE TIP 33 (2024): 2171–2182 · arXiv:2206.09459（待核实）

- 多尺度框架，clip 划分 + 交替 shift；每尺度 TMSA（temporal mutual self attention，帧间互注意力做联合对齐与融合）+ parallel warping。
- 任务：**视频超分、去模糊、去噪、插帧、时空超分**——不含低光增强。35.6M 参数（RVRT 论文 Table 6）。

**引用核对**：v7 归入 implicit（"a Transformer whose multi-frame attention learns brightening, denoising, and alignment jointly"）⚠️ **措辞需限定**：VRT 学会的是 alignment/fusion/denoising，不是 brightening。建议改为 "learns denoising, alignment, and fusion jointly"，或注明 "general video restoration"。

### 4. RVRT [6]（全文精读，arXiv:2206.02146）

**Recurrent Video Restoration Transformer with Guided Deformable Attention** · Liang Jingyun 等 · NeurIPS 2022 · arXiv:2206.02146

- clip 内并行 + clip 间递归（双向信息累积）；GDA 用光流做 pre-alignment + 预测 offset 的 deformable attention；初始光流 SpyNet。
- 任务：SR（BI/BD）、去模糊（DVD/GoPro）、去噪（DAVIS/Set8）——**不含低光增强**；论文未讨论训练数据依赖/泛化。
- 10.8M 参数；REDS4 BI×4 32.75 dB（VRT 32.19）；GoPro 34.92。

**引用核对**：同 VRT，归入 implicit ✅ 但 "brightening" 措辞不适用。v7 对 implicit 的批评 "performance depends on the quality and generalization of the training data" —— RVRT 论文本身未讨论泛化，此批评更适合指向数据驱动的端到端方法整体，措辞时不要写成 RVRT 的自述局限。

## Parallel 类（v7 归入 [7,8,9]）

### 5. PDHAT [7]（未获取全文）

**Perceptual Decoupling with Heterogeneous Auxiliary Tasks for Joint Low-Light Image Enhancement and Deblurring** · Li Yuezhou 等 · IEEE TMM 26 (2024): 6663–6675

- 按 715 引用核对笔记：**三并行分支**架构（增强/去模糊等异构辅助任务解耦）。
- 未上 arXiv；IEEE 全文不可得。**只用"perceptual decoupling + 异构辅助任务 + 并行分支"层级描述。**

**引用核对**：归入 parallel（"assign the degradation to dedicated branches"）✅（与 715 笔记一致）。

### 6. SNR-Aware [8]（摘要级，CVF 开放页）

**SNR-Aware Low-Light Image Enhancement** · Xu Xiaogang 等 · CVPR 2022, pp. 17693–17703

- 摘要：SNR-aware transformers + 卷积模型；**极低 SNR 区域用长程操作、其他区域用短程操作**；SNR 先验引导特征融合；新自注意力避免极低 SNR 区域噪声 token 干扰。
- 七个基准超 SOTA；100 人用户研究。

**引用核对**：v7 "decompositions are built on ... frequency or SNR statistics [8]" ⚠️ **措辞偏松**。SNR-Aware 不是"把退化分解到不同分支"，而是**按 SNR 统计做空间自适应处理**。建议改为 "spatially adaptive processing driven by SNR statistics [8]"，或把它从 decomposition 句中移出、单独作为自适应增强代表。

### 7. STCD [9]（本地全文笔记：摘要 + 引言）

**Low-Light Video Enhancement via Spatial-Temporal Consistent Decomposition** · Xu Xiaogang 等 · IJCAI 2025, pp. 2134–2142

- 双源分解 $I=L\otimes R$：R 帧无关（固有外观，跨帧对应 + 不确定性链接）、L 帧可变（光照阴影，空间平滑约束）。
- CFIM（Cross-Frame Interaction Module）：长程跨帧注意力 + 短程通道-空间融合；双结构网络同时监督多帧。
- 动机句可引："a scarcity of real-world, high-quality, spatially aligned video pairs for dynamic scenes"。

**引用核对**：v7 归入 parallel [7,8,9] ⚠️ **归类需注意**：STCD 是**双源分解 + 跨帧交互**，不是"每源一个并行分支"的多分支架构。§2.1/§2.2 引用时建议表述为 "decomposes the video into view-independent and view-dependent components [9]"，把它作为"按信号表示（外观/光照）分解"的代表，而非 parallel-branch 代表。

## 时序一致性（v7 [2]）

### 8. TCE-Net [2]（未获取全文）

**Temporally Consistent Enhancement of Low-Light Videos via Spatial-Temporal Compatible Learning** · Zhu Lingyu 等 · IJCV 132 (2024)

- **页码两说**：v7 草稿写 4703–4723；715 引用核对笔记写 4015–4034。**[待核实]**——定稿前查 Springer 页面。
- 按 715 笔记：数据 + 模型双中心解决时序一致性；3D 卷积区分"时域随机噪声（帧独立）"与"时域一致信号"；简单时域融合在动态区域产生鬼影。

**引用核对**：
- "per-frame enhancement also introduces temporal flicker [2]" ✅（与 715 笔记一致）。
- "four canonical types [12, 2]"（两维度分类）⚠️ TCE-Net 支持"时序独立性"这一维度（噪声帧独立 vs 信号帧一致），但四分类本体是本文的构造；[12]（Foi）**不支持**四分类（见 papers-noise.md），建议此处只留 [2] 或改写为 "the distinction between frame-independent noise and temporally consistent signal [2]"。

## §2.1 组织建议（Sequential / Implicit / Parallel 三类）

- Sequential：RetinexNet [4]、URetinex [3] —— 误差传播（光照校正不准 → 去噪放大）。
- Implicit：VRT [5]、RVRT [6] —— 多帧注意力联合学习，但属通用视频修复；批评落在"依赖训练数据质量与泛化"。
- Parallel：PDHAT [7]（多分支）、SNR-Aware [8]（SNR 自适应）、STCD [9]（双源分解）—— 批评落在"分解准则按信号表示（成像域/频率/SNR/时间步），而非时序行为"。
- 注意 v7 §1 第三段的措辞已按此展开；写 §2.1 时每类 2–3 句，先机制后局限，引用位置与 §1 呼应。
