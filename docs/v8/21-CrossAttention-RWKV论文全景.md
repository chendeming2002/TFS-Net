# Cross-Attention + RWKV 论文全景（动态审计版）

> 创建于 2026-11-02；最近复核于 **2026-11-04**。
> 本轮采用五条检索链：关键词组合检索、已知论文的参考文献/被引文献滚雪球、作者与项目链、RWKV 社区月报/技术博文线索、官方代码与同源专利反查。博客和 README **只用于发现线索**，机制结论最终回到论文公式、正文/结构图、出版商页面、官方代码或专利。
> 本文是截至复核日的**高覆盖动态语料库**，不是可证明穷尽的封闭清单。付费全文、未来发表、撤稿和论文—代码不同步都会使绝对“全景”与“真空白”声明失真。
> 与本仓库的关联：`06-CrossRWKVGate-Q重构设计.md` 的来源论文是 arXiv:2411.05636（类型①的 Video RWKV / LCR）。

## 纳入与证据口径

- **核心纳入**：实际模型同时包含 RWKV/WKV 与跨输入交互；“输入”可以是两幅图、两路图特征、两个时刻/视角/模态、文本—视觉或编码器—解码器流。
- **边界相关**：只做同一输入的跨方向/跨层/跨频变换，或模块虽名为 cross-attention、公式却只是 add/concat/gate/CBAM/FiLM；单列，不能与跨流 Q/K/V 机制混计。
- **不计入正式数量**：撤稿、只有占位仓库、仅二手摘要且机制不可验证，或论文—代码的关键算子互相冲突；保留为风险线索。
- **证据等级**：**A**＝原文公式/算法并由代码或结构图交叉确认；**B**＝原文正文+结构图或出版商全文足以确认；**C**＝仅摘要/标题，暂不作细粒度结构判断；**X**＝撤稿或关键论文—代码冲突。

## 本轮高影响修订摘要

| 原描述/结论 | 2026-11-04 核实结果 |
|---|---|
| LCR 的 R/G 来自边缘、K/V 来自历史 | **方向相反**：R/G 来自当前主干+历史态，K/V 来自当前边缘 |
| MTRNet 是 T1↔T2 cross-attention，且与 VideoRWKV 几乎一样 | MTRNet 的 CA 在**各时相内部**做主体 Q↔边缘 K/V；仅输入来源拓扑近似，算子不同 |
| X2Bone 结构图没有双输入 | 代码有两个对称 `CrossBlock(a,b)/(b,a)`；双输入已确认，AP/LAT 的具体 R/K/V 标签有文码歧义 |
| ChangeRWKV 是标准二次型 CA | 它使用外置 **Cross-CBAM**，不是 QKᵀ，也不在 WKV 内 |
| 类型①在稠密重建/A 类仍空 | RS-RWKV（NeurIPS）与 MTRWKV（CVPR Highlight）均为 `R←MS, K/V←PAN` |
| 类型②尚无 A 类 | RWKVFusion（TPAMI）和 TIMotion（CVPR）已构成 A 类实例 |
| delta-rule 跨流状态无人认领 | SleepRWKVNet 的同源专利给出 RWKV-7 delta-rule 异源状态更新 |
| TE-MSTAD、PRFusion、CROWNFuse 的原分类 | 分别改为①、③、②；PRFusion 另标论文—代码不一致 |
| “44篇/类型③ 20篇/确认真空白” | 删除固定计数和绝对不存在声明，改为有日期与条件的动态审计 |

历史元数据修正仍有效：TMM 在 CCF 第七版为 B；JSTARS 不在 CCF 目录；CrossWKV/RrmTrack 仅自称 novel、无 first；CMGN 是文本假新闻检测而非图像融合。

## 分类体系（机制类型与输入关系分开记录）

| 类型 | 定义 | 判据 |
|---|---|---|
| **① WKV 算子内部跨流交叉 (机制级)** | Q/R 侧与 K/V 侧 (或 K 与 V 之间、衰减 w) 来自不同输入流, 交叉发生在 WKV 递归/门控内部 | 交叉算子 = WKV 本身 |
| **② RWKV 充当跨流交互算子 (替代 cross-attention)** | 无显式 cross-attention, RWKV 块 (time/channel-mix、扫描) 本身完成跨流交互或融合 | 交互算子 = RWKV 块, 无二次型 CA |
| **③ 混合范式 (RWKV + 显式交叉算子并存)** | 标准 QKᵀ cross-attention、线性交叉注意力、CBAM/互调制等显式跨流模块与 RWKV 分工嵌套 | RWKV 与外置交叉算子各司其职；**不等于全是二次型 CA** |
| **④ 特征级交叉 / 命名边界** | 交叉经通道统计、门控混合、损失对齐、证据融合或生成式监督完成；或名称含 CA 但公式无跨流 QK 匹配 | 既非 WKV 内部异源 R/K/V，也非 RWKV 替代 CA |
| **对照组** | 刻意不用交叉的 RWKV 多模态工作, 及常被引的"第一"基础设施 | 反向参照 |

输入关系（图—图、图特征—图特征、视觉—文本、双时相、跨层等）只描述数据来源，**不能反推机制类型**。

---

## 类型 ①: WKV 算子内部跨流交叉 (机制级)

### CrossWKV ⭐（无 first 自称）

**[Cross-attention for State-based model RWKV-7](https://arxiv.org/abs/2504.14260)** | 2025, arXiv:2504.14260 (2025-04-19, 仅 v1; Liu Xiao, Li Zhiyuan, Lin Yueyu) | 无 IF / 未入 CCF | 流关系: **跨模态, 文本→图像**

- **定位声明**: 摘要自称 "a **novel cross-attention mechanism** for the state-based RWKV-7 model"; **全文无 "first" 优先权声明**——"第一个把 cross-attention 形式化嵌入 WKV 递归"是外部归纳, 引用时只能引 "novel"。
- **设计** (全部核实属实): 文本嵌入 (CLIP 输出) 为 query 侧, 图像特征为序列主体; 文本 padding 到图像序列长度后生成 R, K/V 来自图像特征; **time-shift (δ = shift(x) − x) 保留且作用于图像 token 流**; 衰减 w 用 LoRA 分解 (decay/lr rank-64, value 16, gate 128); K 侧 L2Norm; 复杂度 O(T·N·H)。
- **理论**: "non-diagonal, input-dependent transition matrix... beyond the **TC0**"; "state-tracking tasks like **S5 permutation modeling**"。
- **评价**: DIR-7 (RWKV-7 扩散) ImageNet 256×256 FID 2.88、CLIP 0.33 属实, 但措辞是 "**matching** state-of-the-art / **rivaling** DiT-XL/2", 且自注 "preliminary results should be interpreted with caution"。未同行评审；动态引用数不用于支撑优先权。

### Video RWKV / LCR ⭐ (06 号文档来源)

**[Video RWKV: Video Action Recognition Based RWKV](https://arxiv.org/abs/2411.05636)** | 2024, arXiv:2411.05636 (2024-11-08; Zhuowen Yin, Chengru Li, Xingbo Dong) | 无 IF / 未入 CCF | 流关系: **跨时刻, 当前帧边缘→历史特征**

- **定位声明** (逐字核实): "this work **introduces RWKV to the video domain in a novel way**"; "set a new benchmark in video understanding"。模型名 **LCR**。
- **Cross RWKV gate（证据 A）**：`R,G = Cauconv(x_t,h_{t-1})`，来自当前主干特征与历史隐状态；`K,V` 来自当前帧边缘特征 `x_t^e`。随后以 R 门控 WKV 输出、再经逐头归一化与 `SiLU(G)` 调制。边缘信息还进入 LSTM 遗忘门以抑制背景。**旧表把 R/G 写成边缘、K/V 写成历史，方向相反，已修正。**
- **评价**: 线性复杂度, 代码公开；仍为预印本，"new benchmark" 是作者自评。**本仓库 `06-CrossRWKVGate-Q重构设计.md` 的任务 Q 机制 (式 2–8) 即移植自此**。

### OTN-RWKV / Asymmetric RWKV Fusion ⭐（类型①的 A 类代表）

**[RGB-Event based Pedestrian Attribute Recognition: A Benchmark Dataset and An Asymmetric RWKV Fusion Framework](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_RGB-Event_based_Pedestrian_Attribute_Recognition_A_Benchmark_Dataset_and_An_CVPR_2026_paper.html)** | 2026, CVPR, pp.32745–32755 (Xiao Wang 等; arXiv:2504.10018) | 无传统 IF | **CCF A** | 流关系: **跨模态, RGB↔Event**

- **定位**: 双模态各自 Vision-RWKV 编码 + OTN-RWKV 非对称融合; EventPAR 100K 数据集 (自称 "The first benchmark dataset for RGB-Event PAR")。
- **实现细节 (2026-11-02 核实 arXiv 全文 + 官方代码 Event-AHU/OpenPAR::otn_rwkv.py, MIT, 含 RWKV6-B 预训练权重)**: 融合在两编码器之后的**最后一层、单一尺度** (识别任务, 全局池化头); **WKV-6 算子内部跨流——R、K 投影来自 RGB 路, V 与衰减 w 来自 event 路**, 其中 **w = time_decay (参数) + tanh(MLP(event 特征))——输入条件化衰减**; 输出 `σ(R)⊙K⊙WKV + event 残差`; WKV-6 CUDA 核 (T_MAX=256)。**论文-代码出入**: 式 (5) 的 KNP top-K 相似度过滤 (阈值 0.75) 未随代码发布; 式 (6) 的 LN 链在代码中简化。

### EvRWKV (实现已由本仓库代码核实)

**[EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement](https://arxiv.org/abs/2507.03184)** | 2026, IEEE TCSVT, vol.36, no.7, pp.10201–10216 (DOI: 10.1109/TCSVT.2026.3672491) | IF 8.4 | **CCF B** | 流关系: **跨模态, Image↔Event, 复原域**

- **定位**: 连续跨模态交互 (反对 early/late fusion 二分): Cross-RWKV Module 捕捉细粒度时间与跨模态依赖 + EISFE 频域去噪/空域对齐。SDE/SDSD PSNR +1.79/+1.85 dB, 下游 mIoU +35.44% (均核实属实)。
- **实现细节 (2026-11-02 核实本仓库 `reference_repos/EvRWKV/egllie/models/Cross_Restore_RWKV.py`)**: **WKV 内部 K/V 跨流交替扫描**——`v_image = RUN_CUDA(..., k_event, v_image)`, `v_event = RUN_CUDA(..., k_image, v_event)` (**K 取对侧流, V 取本路流**), 偶次直扫/奇次转置扫 (双向); 融合权重自适应混合两路后过本路 `σ(R)` 门输出; ChannelMix 各路独立无交叉。**本仓库借用的 OmniShift 即出自该文件**。→ 归类型① (此前按摘要曾存疑归混合类, 已修正)。

### X2Bone (与 receptance 模式同型)

**[X2Bone: Reconstructing 3D Bone Structures from 2D Biplanar X-Rays via Cross-RWKV](https://papers.miccai.org/miccai-2026/1163-Paper0798.html)** | 2026, MICCAI, LNCS 16889 (open access) | 无传统 IF | **CCF B** (交叉/综合/新兴) | 流关系: **跨视角, AP↔LAT**

- **定位**: "Cross-RWKV module... long-range, bidirectional cross-view interaction with linear complexity"; 配套 3D Expand 视角相关深度调制。CTPelvic1K 最优 (DSC 87.96%, 11.02M 参数)。
- **实现细节（证据 A，但方向有文码歧义）**：双流共享权重 CNN 编码器，每尺度有两个对称 `CrossBlock(x_a,x_b)` / `CrossBlock(x_b,x_a)`，所以**结构图未画双箭头不代表没有双输入**。代码中 R 取第一个参数、K/V 取第二个参数；论文/rebuttal 对 AP/LAT 的文字归属与调用方向不完全一致。稳妥结论是“R 与 K/V 来自不同视角”已确认，具体 AP/LAT 标签须注明歧义。门控为 `sigmoid(r)⊙WKV`；论文称有 Cross-Shift，公开代码未见同名实现。

### RS-RWKV（NeurIPS 2025，新增关键漏项）

**[WKV-sharing embraced random shuffle RWKV high-order modeling for pan-sharpening](https://proceedings.neurips.cc/paper_files/paper/2025/file/d194e6b365e2cca61170445f25412122-Paper-Conference.pdf)** | 2025, NeurIPS 38 | **CCF A** | 流关系：**图—图 / MS↔PAN** | 证据 **A**

- Eq. (31) 明确：`R_s=X_M W_R`，`K_s=X_P W_K`，`V_s=X_P W_V`；即 MS 提供 receptance，PAN 提供 K/V，再以 `σ(R_s)⊙WKV(K_s,V_s)` 更新 MS。随后与 PAN 特征拼接进入 channel mixer。
- 随机打乱/逆打乱改变空间扫描次序；“WKV sharing”主要是**跨层缓存/复用 WKV 激活**，不要把它误写成跨模态 KV 交换。真正满足类型①的是 Eq. (31) 的异源 R/K/V。

### MTRWKV（CVPR 2026 Highlight，新增关键漏项；勿与 MTRNet 混淆）

**[Multigrain-aware Semantic Prototype Scanning and Tri-Token Prompt Learning Embraced High-Order RWKV for Pan-Sharpening](https://arxiv.org/abs/2604.14622)** | 2026, CVPR Highlight | **CCF A** | 流关系：**图—图 / MS↔PAN** | 证据 **A** | [代码](https://github.com/L1junfeng/MTRWKV)

- 预备式同样明确为 `R_s=X_MW_R, K_s=X_PW_K, V_s=X_PW_V`，因此属于 WKV 内部跨模态输入，而非普通双分支对照。
- 在此基础上引入语义 LSH 重排、prototype/global/register 三类 token、Bi-WKV 与高阶共享。它与 RS-RWKV 是同一作者链上的直接后续，说明“稠密重建域无类型①”已经不成立。

### SleepRWKVNet（由②改判①）

**[SleepRWKVNet](https://doi.org/10.1016/j.bspc.2025.108713)** | 2026, Biomedical Signal Processing and Control 113:108713 | 流关系：**EEG↔EOG/EMG** | 证据 **B**（论文摘要 + [同源专利 CN120873825A](https://patents.google.com/patent/CN120873825A/en)）

- Bi-IFM 中主 EEG 流生成 R、time-decay、in-context learning rate 与 output gate，辅助 EOG/EMG 流生成 remove key、replace key、value；以 RWKV-7 delta rule 更新辅助状态，再由主流 R 读取。两个方向/两种辅助模态并行交互。
- 因此它不是笼统“RWKV 做融合”，而是**状态更新内部异源输入**；同时推翻了无条件的“delta-rule 跨序列状态演化无人认领”。现阶段只能保留更窄的空白：视觉稠密预测 + CCF A + delta-rule 异源状态更新。

### TE-MSTAD（由④改判①）

**[TE-MSTAD](https://www.techscience.com/cmc/v88n2/67602/html)** | 2026, Computers, Materials & Continua 88(2), DOI:10.32604/cmc.2026.078282 | 流关系：**跨节点/模态** | 证据 **A**

- CFE 对第 i 路用本路 `r_i,v_i`，而 `k_cross^i` 是其余各路 K 投影的平均；随后计算 `WKV(k_cross^i,v_i)` 并由 `σ(r_i)` 门控。即“他路 K + 本路 R/V”的 WKV 内部交叉。

### 邻接机制参照（以下不计“不同输入流”核心数量）

#### RWKV-IR (Cross-Bi-WKV, 类型①变体)

**[RWKV-IR](https://arxiv.org/abs/2412.03814)** | 2024, arXiv 预印本 | 无 IF / 未入 CCF | 流关系: **跨方向 (同一输入的两个扫描序)**

- "Cross-Bi-WKV module that merges two Bi-WKV modules with different scanning orders for balanced horizontal and vertical attention"——WKV 内部跨**扫描方向**交互, 非跨输入流, 归本类型变体。

#### WuNeng (头间交叉变体)

**[WuNeng: Hybrid State with Attention](https://arxiv.org/abs/2504.19191)** | 2025, arXiv 预印本 (与 CrossWKV 同一作 Liu Xiao) | 无 IF / 未入 CCF | 流关系: **头间 (MHA 注意力头 × RWKV-7 状态头)**

- MHA 头与 RWKV-7 状态头做 cross-head 交互 (拼接/加性调制/门控融合); 跨流发生在**头间**而非序列间, 归本类型变体。

#### CrossMamba (同构前身, 非 RWKV 载体, 标注参照)

**[CrossMamba](https://arxiv.org/abs/2409.04803)** | 2024, arXiv (目标声音提取) | 无 IF / 未入 CCF | 流关系: **跨模态, 文本线索→音频**

- "utilize the clue to generate the query and the audio mixture to derive the key and value"——**Mamba 选择性状态空间递归内实现 cross-attention**, 是 CrossWKV 的直接思想前身 (CrossWKV 引用了它)。非 RWKV 载体, 仅作机制族参照。

### 类型①设计空间小结: 各工作从"另一路"取什么

| 工作 | 交叉流关系 | R/G 来源 | K 来源 | V 来源 | 衰减 w | 门控 |
|---|---|---|---|---|---|---|
| CrossWKV | 文本→图像 | 文本 (R) | 图像 | 图像 | LoRA(图像) 自适应 | LoRA gate |
| LCR Cross gate | 当前主干/历史态↔当前边缘 | 当前主干+历史态 (R,G) | **当前边缘** | **当前边缘** | 块参数；边缘另调 LSTM 遗忘门 | R⊙ + LN + SiLU(G) |
| OTN-RWKV | RGB↔Event | RGB (R,K) | RGB | **Event** | **参数 + tanh MLP(event) 输入条件化** | σ(R), K 外乘 |
| EvRWKV | Image↔Event | 本路 (R) | **对侧** | 本路 | 本路块参数 | σ(R) + 自适应融合权重 |
| X2Bone | AP↔LAT | 本视角 (R) | **对侧** | **对侧** | 块参数 | σ(R) |
| RS-RWKV | MS↔PAN | **MS** | **PAN** | **PAN** | 块参数 | σ(R) |
| MTRWKV | MS↔PAN | **MS** | **PAN** | **PAN** | 块参数 | σ(R) |
| SleepRWKVNet | EEG↔EOG/EMG | 主流 R | 辅流 remove/replace K | 辅流 V | 主流条件化；RWKV-7 delta rule | R 读状态 + output gate |
| TE-MSTAD | 当前路↔其余路 | 当前路 R | **其余路 K 均值** | 当前路 V | 块参数 | σ(R) |
| *(本仓库 QRB receptance)* | *统计源* | *查询路* | *KV 路* | *KV 路* | *头参数* | *σ(q)* |
| *(本仓库 QRB cross_rwkv)* | *统计源* | *查询路 (R,G)* | *KV 路* | *KV 路* | *头参数* | *R⊙ + 逐头LN + SiLU(G)* |

(斜体行为本仓库现状参照，非文献条目。已发表工作覆盖了 {R,K,V,w/状态更新} 异源的多个子集；SleepRWKVNet 已出现 RWKV-7 delta-rule 式跨流状态更新，故不再声称该机制整体空缺。)

---

## 类型 ②: RWKV 充当跨流交互算子 (替代 cross-attention)

### RrmTrack ⭐

**[RWKV-Inspired Multi-Modal Relation Modeling for Vision-Language Tracking](https://ieeexplore.ieee.org/document/11417442)** | 2026, IEEE TMM, vol.28, pp.6892–6901 (DOI: 10.1109/TMM.2026.3668700) | IF 9.7 | **CCF B** | 流关系: **跨模态, 语言↔图像**

- **定位声明**: "a **novel modality interaction method** specific to vision-language object tracking based on RWKV" (摘要无 "first" 自称)。
- **分工** (摘要逐字): "**time mixing module to model the relationship between language and image features and a channel mixing module to facilitate information interaction between images**"——跨模态走 time-mix、模态内走 channel-mix。
- 配套: Siamese + One-stream; 信息恢复模块; 并行训练 + RNN 推理; TNL2k/LaSOT/OTB-Lang/LaSOText/MGIT 五基准 "state-of-the-art results and speed"。

### RWKV-VG

**[RWKV-VG: visual grounding with RWKV-driven encoder-decoder framework](https://link.springer.com/article/10.1007/s00530-025-01720-w)** | 2025, Multimedia Systems, vol.31, art.124 | IF 3.1 (2024 JCR; 2025=2.8) | CCF C | 流关系: **跨模态, 视觉↔语言 (纯 RWKV, 无显式 CA)**

- **定位声明** (逐字核实): "**the first visual grounding framework entirely built on the RWKV architecture**"。RWKV 视觉/语言编码器 + RWKV 视觉语言解码器 (intra-modal + cross-modal reasoning), 可学习 [REG] token。
- **评价**: RefCOCO 系列评测; 摘要**自述 QRNet/CLIP-VG 等仍优于它** (预训练数据 15M vs 400M)。

### IESS-FusionNet (显式宣称替代 CA)

**[IESS-FusionNet](https://www.mdpi.com/2306-5354/13/1/57)** (EEG-EMG 婴儿癫痫痉挛检测) | 2026, Bioengineering 13(1):57 (MDPI) | IF 4.4 (2023 JCR; 2024 未核) | 不在 CCF | 流关系: **跨模态, EEG↔EMG**

- **Cross Time-Mixing 模块 "built upon the linear recurrent attention of the RWKV architecture... replacing the quadratic cross-attention"**——算子替代路线继 RrmTrack 后第二个**显式宣称"用 RWKV 替代 cross-attention"**的实例。

### OccRWKV

**[OccRWKV](https://arxiv.org/abs/2409.19987)** (3D 语义占据预测) | 2025, ICRA | 无传统 IF | **CCF B** | 流关系: **跨分支 (语义/几何双分支)**

- Sem-RWKV + Geo-RWKV 双分支分别建模, **BEV-RWKV 块充当跨分支融合算子** ("用 RWKV 做融合"而非给 RWKV 配 CA)。SemanticKITTI 25.1 mIoU @ 22.2 FPS。

### MMCGR

**[MMCGR](https://www.sciencedirect.com/science/article/abs/pii/S0925231226008210)** (对话情感识别) | 2026, Neurocomputing, vol.682, 133424 | IF 6.5 | CCF C | 流关系: **跨模态, 文本↔音频**

- 出版商全文摘要明确：unimodal RWKV 与 **cross-modal RWKV** 分别建模模态内/间关系；跨模态编码器使用两个输入窗口与参数共享，再由级联门控融合。可确认属于②；在取得公式前，暂不进一步声称其 R/K/V 如何异源。证据 **B**。

### RWKV-Enhanced IR-Vis Fusion

**[RWKV-Enhanced Infrared and Visible Image Fusion: Capturing Long-Range Dependencies with Cross-Modality Correlations](https://ieeexplore.ieee.org/document/11605729)** | 2026, PRMVAI (IEEE Int. Conf. on Pattern Recognition, Machine Vision and AI, 第 4 届) | 无 IF | 不在 CCF | 流关系: **跨模态, 红外↔可见光**

- 摘要逐字: "a **spatial-channel dual-hybrid RWKV-Fuser block**... simultaneously captures long-range dependencies and **cross-modal channel correlations**"——RWKV 块本身做跨模态融合。⚠️ 归②/④ 的边界待全文核 (无 "替代 CA" 表述, 亦无显式 CA)。

### VisualRWKV-HM（跨状态读取，新增）

**[VisualRWKV-HM: Enhancing linear visual-language models via hybrid mixing](https://doi.org/10.1016/j.inffus.2025.103336)** | 2025, Information Fusion 124:103336 | 流关系：**图像→文本状态读取** | 证据 **A** | [官方代码](https://github.com/howard-hou/VisualRWKV)

- 图像 token 先编码为矩阵状态 `S_img`；文本产生 `mem_read` 与 `mem_gate`，执行 `mem_read @ S_img`，再门控写入文本流。论文称 Cross State Mixing “analogous to cross-attention”，但它不是 QKᵀ，也不是同一次 WKV 内的异源 R/K/V。
- 纯 Cross State Mixing 未取得理想结果，最终模型采用 Hybrid Mixing（time mix + state readout）。因此对照组只能写“vanilla VisualRWKV 无 CA”，不能把整个 VisualRWKV 家族都归为无交叉。

### RWKVFusion（TPAMI 2025，新增）

**[An Efficient Image Fusion Network Exploiting Unifying Language and Mask Guidance](https://doi.org/10.1109/TPAMI.2025.3591930)** | 2025, IEEE TPAMI 47(11):9845–9862 | **CCF A** | 流关系：**视觉↔语言** | 证据 **A** | [作者全文](https://liangjiandeng.github.io/papers/2025/cao-tpami2025.pdf) / [代码](https://github.com/294coder/RWKVFusion)

- 默认 MFM 将图像与文本 token 交错/拼接为统一序列，再共同生成 k/v/r/g 并做 WKV5 混合；因此是“RWKV 作为跨流交互算子”的②。
- 论文中的两次标准 cross-attention 是**替换 RWKV 的消融对照**，不是主模型。此前据消融图把它归③会颠倒结论。

### TIMotion（CVPR 2025，新增）

**[TIMotion: Temporal and Interactive Framework for Efficient Human-Human Motion Generation](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_TIMotion_Temporal_and_Interactive_Framework_for_Efficient_Human-Human_Motion_Generation_CVPR_2025_paper.pdf)** | 2025, CVPR | **CCF A** | 流关系：**双主体动作序列** | 证据 **A**

- Eq. (6–8) 先把两人的帧序列按时间交错，再构造角色反转序列并沿通道拼接，统一送入 RWKV InteractionMixing，最后拆回两路输出。论文消融明确称 Causal Interactive Injection 取代原 Transformer 的 self-/cross-attention。
- 这与 RWKVFusion 同属“多输入先统一成一条序列，再由 RWKV 完成交互”，也直接推翻了“类型②尚无 A 类”。

### CROWNFuse（由③改判②）

**[CROWNFuse: Recurrent cross-modal memory interaction with edge-frequency recalibration for infrared and visible image fusion](https://doi.org/10.1016/j.neucom.2026.134876)** | 2026, Neurocomputing 134876 | CCF C | 流关系：**红外↔可见光** | 证据 **B**

- RWKV-HCMF 让每一模态的 WKV 聚合上下文作为 recurrent memory 注入另一模态，并用通道选择门控校准；出版商原文特别强调不做 dense pairwise token matching。
- 现有证据支持“RWKV 状态/记忆完成跨模态交互”，不支持把它写成 RWKV 外挂标准 CA；若以后取得完整公式并显示异源状态直接参与 WKV 更新，可再从②细化到①。

---

## 类型 ③: 混合范式 (RWKV + 显式交叉算子并存)

### CMRWNet

**[CMRWNet: Cross-Modal RWKV Network With Window-Guided Alignment for High-Fidelity Pansharpening](https://ieeexplore.ieee.org/document/11614160)** | 2026, IEEE JSTARS, vol.19, pp.24069–24084, DOI:10.1109/JSTARS.2026.3714757 | IF 5.3 (2024 JCR; 2025=6.3) | **不在 CCF 目录** | 流关系: **跨模态, MS↔PAN** | 证据 **B**

- 窗口级双向 cross-attention (subquadratic) 做精对齐 + 堆叠 PAN-RWKV ("row and columnwise parallel prefix-decay scans" + receptance gating) 做线性全局混合——CA 局部化与 RWKV 全局化的分工范式。IKONOS/Pléiades/WV-3 SOTA + QuickBird 跨传感器。

### Surv-RWKV

**[Surv-RWKV](https://www.sciencedirect.com/science/article/abs/pii/S0957417426004197)** (生存分析) | 2026, ESWA, vol.312, 131506 | IF 7.5 (2024 JCR; 2025=9.4) | CCF C | 流关系: **跨模态 (基因/病理)**

- RWKV 浅层融合 (空间-通道混合) + **cross-RWKV 深度交互 ("iterative cross-attention mechanisms")** + 最优传输对齐; 五个 TCGA 队列 (SOTA 逐字措辞因付费墙未取得)。

### ChangeRWKV（Cross-CBAM，而非 WKV 内 cross-attention）

**[Beyond Quadratic: Linear-Time Change Detection with RWKV](https://arxiv.org/abs/2603.19606)** | 2026, **AAAI**, Vol.40(14), pp.11811–11819 (DOI: 10.1609/aaai.v40i14.38167, Crossref 已核) | 无传统 IF | **CCF A** | 流关系: **跨时相 (双时相遥感)**

- 层级 RWKV 编码器 + 外置 **Cross-CBAM**：一条时相生成通道权重并施加到另一条，随后再做交叉空间注意。它没有 QKᵀ，也不改 WKV 本体，故属于③中的**显式跨流调制**，不能称“标准二次型 cross-attention”。删除缺乏完备检索依据的“第一个”优先权表述。LEVIR-CD 85.46% IoU。

### TemLo

**[TemLo: Temporal-Local Synergy Enhanced RWKV for Audio-Visual Segmentation](https://ieeexplore.ieee.org/document/11523565)** | 2026, IEEE TMM, vol.28, pp.9804–9815 | IF 9.7 | **CCF B** | 流关系: **跨模态, 音频↔视觉**

- 模态专属 RWKV 编码器双流 + **Acoustic Modulation Pyramid** 结构化音视频交互 + Vision-guided Modality Injector (摘要: "To fully enable cross-modal interaction and integration, we further design the Acoustic Modulation Pyramid...")。代码公开 (github.com/c1ircle/TemLo)。**音视频域此前误判为空白, 本轮补上**。⚠️ 调制金字塔非标准 CA, 归③/②存疑。

### CRWKV (类型③中的 A 类)

**[Multi-View Learning with Context-Guided Receptance for Image Denoising](https://doi.org/10.24963/ijcai.2025/86)** | 2025, IJCAI | 无传统 IF | **CCF A** | 流关系: **跨特征视图 (空域+FMix 频域)**

- Context-guided RWKV, 空域与频域表征经多视图学习融合, BiWKV 全像素交互; "多视图"是特征视图非模态。

### C2F-VRWKV

**[C2F-VRWKV](https://doi.org/10.1109/TGRS.2026.3693340)** (舰船检测) | 2026, IEEE TGRS | IF 8.6 (2024 JCR) | **CCF B** (交叉/综合/新兴) | 流关系: **跨频 (同一图像的频率带)**

- Vision-RWKV 骨干 + "Cross-Frequency Fusion Module... frequency-guided cross-attention strategy"。

### MTRNet (中文期刊)

**[MTRNet](https://doi.org/10.3788/lop251946)** | 2025, 《激光与光电子学进展》 | 无 IF/CCF (中文期刊) | 流关系: **双时相任务；CA 实际为各时相内部的主体特征↔边缘特征**

- **结构复核（证据 A）**：每个时相分支内部，VRWKV 主体特征生成 Q，PDC 边缘特征生成 K/V，经过标准多头 scaled dot-product Softmax cross-attention、拼接与 SimAM；T1/T2 两个时相直到解码阶段才做拼接。因此它不是“用 CA 直接做 T1↔T2”。
- **与 VideoRWKV 的关系**：二者的数据源拓扑相似，都是“主体/读取侧”对“边缘 K/V 侧”；但 MTRNet 是 RWKV 外的显式二次型 CA，VideoRWKV 是 WKV 内 R/G–K/V 门控递归。可说“结构图来源分配近似”，不能说注意力算子几乎相同。

### RWKV-CD（新增；与 MTRNet/VideoRWKV 任务和结构均接近）

**[RWKV-CD: A Vision-RWKV Framework for Building Change Detection From Remotely Sensed Imagery](https://doi.org/10.1109/JSTARS.2026.3677839)** | 2026, IEEE JSTARS 19:13986–14002 | 流关系：**主体特征↔PDC 边缘特征；双时相图像** | 证据 **B**

- EAM 顺序执行两步双向 CA：正文与 Fig. 4 描述为 EBAM 先用校准后的 VRWKV body 作 Q、PDC edge 作 K/V；EEAM 再反向用 edge 查询已更新的 body。
- 这是明确的“查询方向分两步调制”实例。Eq. (13) 的 `CrossAtt` 参数书写顺序与正文/Fig. 4 的 Q/K/V 说明疑似相反，引用时应保留该公式接口歧义。

### VAFTrack（新增）

**[VAFTrack: asynchronous feature fusion via visual receptive weighted key-value perceptual for visual tracking](https://doi.org/10.1007/s00530-025-01913-3)** | 2025, Multimedia Systems 31:348 | CCF C | 流关系：**模板图↔搜索图** | 证据 **A**

- 最终 template 特征与中间 search 特征先拼接；Eq. (5) 的异步显式注意力由 search 产生 Q，而 K/V 来自 search+template 级联序列。之后 VPFM 在统一特征上使用 Bi-WKV、Q-Shift 与通道模块。
- 因为标准 CA 与 RWKV/Bi-WKV 都存在，应归③，而不是把 VPFM 误判为 WKV 内异源 R/K/V。

### DRWKV（新增）

**[DRWKV: Focusing on Object Edges for Low-Light Image Enhancement](https://openaccess.thecvf.com/content/WACV2026/papers/Bai_DRWKV_Focusing_on_Object_Edges_for_Low-Light_Image_Enhancement_WACV_2026_paper.pdf)** | 2026, WACV | 流关系：**同图低层↔高层特征** | 证据 **A**

- Bi-SAB 在 RWKV U-Net 解码器外进行显式交叉：Algorithm 1 中 Q/V 来自低层 `F1`，K 来自高层 `F1'`，CA(Q,K) 后乘 K、残差 V，再做 Scharr 边缘与频谱对齐。它是图特征—图特征，但不是双图输入。

### REB-former（新增，语音域）

**[REB-former: RWKV-enhanced E-Branchformer for Speech Recognition](https://www.isca-archive.org/interspeech_2025/song25b_interspeech.pdf)** | 2025, Interspeech, DOI:10.21437/Interspeech.2025-1343 | 流关系：**编码器→解码器** | 证据 **A**

- 解码器每个 RWKV module 后接标准 encoder-decoder cross-attention，编码器输出作 K/V；这是最直接的“RWKV 与 CA 逐层串联”范式。

### Bridging Transformers and RWKV（新增，非录用论文）

**[Bridging Transformers and RWKV: Towards Efficient Multimodal Video Understanding](https://openreview.net/pdf?id=kmNqnwA4aV)** | ICLR 2026 OpenReview 投稿（公开状态显示未录用） | 流关系：**文本→scene tokens** | 证据 **B**

- hybrid decoder 并行运行 RWKV-v6 self-mix 与标准 cross-attention：文本 hidden states 作 Q，scene tokens 作 K/V，再用动态门融合两支。只作为公开投稿机制证据，不计入已录用论文统计。

### DCAF（新增，情感分析）

**[DCAF: Dynamic affective consistency-aware fusion with disentangled modality representations for multimodal sentiment analysis](https://doi.org/10.1016/j.neucom.2026.133655)** | 2026, Neurocomputing 686:133655 | CCF C | 流关系：**文本/音频/视觉** | 证据 **B**

- 轻量 RWKV 编码器负责模态内长序列，外置 fully symmetric Trimodal Cross-Attention 负责三模态交互，另配 CODL/CULD 解耦约束；属于③。

### PRFusion（由④改判③；论文—代码不一致）

**[A progressive receptance weighted key value network for multimodal medical image fusion](https://doi.org/10.1016/j.asoc.2026.116704)** | 2026, Applied Soft Computing, 116704 | 流关系：**多模态医学图像** | 证据 **B/X** | [官方仓库](https://github.com/CaiCy6/PRFusion)

- 出版商正文明确每个 encoder stage 嵌入线性复杂度 ECMA cross-modal attention，并以 PFFB 逐层累积对齐/融合结果，按论文应归③。
- 但当前官方 `Networks/FusionNet.py` 只见双流 self-RWKV、最深层与 skip 的直接相加，未见 ECMA/PFFB。故“论文机制已确认”，但**公开代码疑似未同步**，不能声称代码复现了论文结构。

### PestVL-Net

**[PestVL-Net](https://doi.org/10.1109/CVPRF71909.2026.00847)** (多模态虫情识别) | 2026, **CVPR FINDINGS Track (CVPRF)**, DOI 已核 | — | 同 CVPR 系 | 流关系: **视觉↔语言**

- RWKV 视觉通路 (显著性引导自适应窗口) + MLLM 文本先验, 经**标准显式 cross-attention** 融合 (Q=视觉, K/V=文本)。CVPR 2026 新设 Findings track, 引用时与主会区分。

### LADY

**[LADY](https://arxiv.org/abs/2512.15038)** (自动驾驶效率) | 2025-12, arXiv 预印本 | 无 IF / 未入 CCF | 流关系: **跨模态, camera↔LiDAR**

- RWKV-7 融合多帧相机+激光雷达 + **lightweight linear cross-attention (LICA)**。⚠️ 摘要不含 "RWKV" 字样, 全文检索必漏 (社区线索定位)。

### ITC-RWKV

**[ITC-RWKV](https://arxiv.org/abs/2510.21479)** (组织病理亚型分型) | 2025-10, arXiv 预印本 | 流关系: **跨尺度/跨粒度, 细胞↔组织**

- RWKV 式 receptance-weighted KV 聚合 (线性复杂度) + "bidirectional tissue-cell interaction module to enable **mutual attention** between localized cellular cues and their surrounding tissue environment"。

### DPC-QA Net

**[DPC-QA Net](https://arxiv.org/abs/2509.15802)** (病理图像无参考质量评估) | 2025, arXiv 预印本 | 流关系: **跨流 (感知质量↔细胞质量)**

- Aggr-RWKV 细胞流 + "**Cross-attention fusion** and multi-term losses align perceptual and cellular cues"。

### MSFusion

**[MSFusion](https://www.mdpi.com/2079-6374/16/8/423)** (多模态医学图像融合) | 2026-08, Biosensors 16(8):423 (MDPI) | IF 5.6 (2024 JCR) | 不在 CCF | 流关系: **跨模态医学影像**

- VRWKV/Bi-WKV 块 + MSCMCF "bidirectional cross-modal attention"。

### SGF-RWKV

**[SGF-RWKV](https://doi.org/10.1109/JBHI.2026.3739849)** (统一医学去噪) | 2026, IEEE JBHI | IF 6.8 | **CCF C** | 流关系: **语言→视觉引导**

- RWKV 统一去噪框架内嵌 "a semantic guided module leverages CLIP based language embeddings to generate task-aware attention maps"。

### RWPNet

**[RWPNet](https://doi.org/10.1109/JSTARS.2026.3734262)** (全色锐化, 与 CMRWNet 同赛道) | 2026, IEEE JSTARS | IF 5.3 | 不在 CCF | 流关系: **跨模态, MS↔PAN**

- SS-RWKV (spatial-spectral) + "dual-branch cross-fusion module... mutual feature modulation"。

### DME-RWKV

**[DME-RWKV](https://doi.org/10.3390/bioengineering13010012)** (抗 VEGF 疗效预测) | 2025, Bioengineering 13(1):12 (MDPI) | IF 4.4 (2023) | 不在 CCF | 流关系: **跨模态, OCT↔UWF**

- RWKV 整合双模态 + Causal Attention Learning + 课程学习。

### PMSRWKV-GCN

**[PMSRWKV-GCN](https://www.nature.com/articles/s41598-026-41091-4)** (多变量时序预测) | 2026, Scientific Reports | IF 3.9 (2024 JCR) | 不在 CCF | 流关系: **跨变量 (时序维↔变量维分工)**

- RWKV 管时间维 (通道独立预训练) + 注意力邻接 GCN 管跨变量维——与 RWKV-CVM (类型④) 门控混合并行的另一条跨变量路线。

### GoldFinch

**[GoldFinch](https://arxiv.org/abs/2407.12077)** | 2024, arXiv 预印本 | 流关系: **跨算子 (transformer 消费 RWKV 状态)**

- GOLD transformer 对 Finch (RWKV-6) 的 KV-cache 做 cross-attention (KV-cache 压缩导向)。

---

## 类型 ④: 特征级交叉 (非注意力算子)

### RWKV-CVM

**[RWKV-CVM: Gated Cross-Variate Mixing for Multivariate Power Load Forecasting](https://www.mdpi.com/2673-4826/7/2/58)** | 2026, Electricity (MDPI), vol.7 no.2 | IF 2.2 | 不在 CCF | 流关系: **跨变量**

- "gated, row-stochastic mixing matrix... adding only **65 trainable parameters**" (逐字核实)。

### RICEFuse

**[RICEFuse](https://www.sciencedirect.com/science/article/abs/pii/S1350449526001945)** | 2026, Infrared Physics & Technology, vol.156 | IF 3.4 (2024 JCR; 官网 3.3) | 不在 CCF | 流关系: **跨模态, 红外↔可见光**

- DFCEM 互补增强 + AMT-Shift + 余弦相似度跨模态对齐损失 (均逐字核实)。

### CMGN

**[CMGN: Text GNN and RWKV MLP-mixer combined with cross-feature fusion for fake news detection](https://www.sciencedirect.com/science/article/pii/S0925231225004837)** | 2025, Neurocomputing, vol.633, 129811 | IF 6.5 | CCF C | 流关系: **跨特征 (⚠️ 文本域: 假新闻检测, 非图像融合)**

- Text GNN + RWKV MLP-mixer + cross-feature fusion (标题核实; 摘要原文不可得)。

### RWKV-VIO

**[RWKV-VIO](https://www.mdpi.com/1424-8220/25/18/5737)** (视觉惯性里程计) | 2025, Sensors 25(18):5737 | 不在 CCF | 流关系: **跨模态, 视觉↔IMU**

- 双并行编码器 + RWKV 时序建模, 端到端 VIO; 摘要未见显式 CA。

### Compass

**[Compass](https://arxiv.org/abs/2608.03559)** (缺失模态裂缝分割) | 2026, arXiv 预印本 | 流关系: **跨模态证据融合**

- Needle RWKV 骨干 + 多模态证据融合 (ETPF/Dempster-Shafer, 非注意力算子)。

### Geometry-aware RWKV 光场 SR

**[Geometry-Aware RWKV for Heterogeneous Light Field Spatial Super-Resolution](https://doi.org/10.1109/LSP.2025.3555445)** | 2025, IEEE Signal Processing Letters | IF 未核 | 不在 CCF (AI/交叉页已查) | 流关系: **跨视角 (空间-角点相干性)**

- 异构融合 (高分辨率 2D 相机 + 光场, 蒸馏迁移) + geometry-aware RWKV。⚠️ 无显式 CA 证据。

### P-RWKV / PointER

**[P-RWKV/PointER](https://arxiv.org/abs/2606.10395)** (点云自监督) | 2026, arXiv 预印本 | 流关系: **"cross-modality" = 3D→2D 渲染生成式监督, 非注意力**

- 全文核实: 其 "cross-modality" 是 Photograph Module 把 3D 特征投影到 2D 渲染图做生成式监督。**引用时勿混淆为 cross-attention**。

### E²I-VRWKV（名称含 Cross-Modal Attention，但数学上属门控边界）

**[E²I-VRWKV: Explicit EPI-Representation and Interaction-Aware Vision-RWKV for Light Field Semantic Segmentation](https://proceedings.mlr.press/v306/zhang26bs.html)** | 2026, ICML, PMLR 306:155985–156000 | **CCF A** | 流关系：**中心视图/空间流↔辅助角度流** | 证据 **A**

- 模块名 LF-ECMA，但跨流主体不是标准 cross-QKV：DMS 先做空间↔角度双向残差调制；EPI prior 内的 Q/K/V 都来自同一个 EPI 张量（自注意力）；最终 `F_out=X_s+gate⊙(W_fuse(X_a)+B_epi)`。
- 因此应记录为“命名上的 cross-modal attention、公式上的跨流门控 + EPI 自注意力”，不能把它列为③中的标准外置 CA，也不能列为①。

## 待取得公式后再分类的核心线索（不计各类型数量）

### Learning to Prefer

**[Learning to Prefer: Reward-Driven Infrared-Visible Fusion via RWKV](https://doi.org/10.1016/j.inffus.2026.104685)** | 2026, Information Fusion 138:104685 | 流关系：**红外↔可见光** | 证据 **C**

- 出版商摘要可确认奖励驱动复原+融合框架、Cross-RWKV 与跨模态差异感知融合；尚未取得 Cross-RWKV 公式，不能在①和③之间定类。

### RWKVMatch

**[RWKVMatch: Vision RWKV-based Multi-scale Feature Matching Network for Unsupervised Deformable Medical Image Registration](https://doi.org/10.1109/ICASSP49660.2025.10888484)** | 2025, ICASSP | **CCF B** | 流关系：**fixed↔moving 医学图像** | 证据 **C**

- 摘要明确称“global attention and cross-fusion mechanism based on Vision-RWKV”，足以确认图—图/多尺度图特征交互；没有公式级证据区分②、③或①，不写具体 R/K/V 来源。

### MMSegRWKV

**[MMSegRWKV](https://doi.org/10.1109/JIOT.2026.3661927)** | 2026, IEEE IoT Journal | 流关系：**T1/T2/FLAIR MRI** | 证据 **C**

- 摘要可确认 U 形 RWKV 与 dual-view WKV，但“spatial/temporal factorization”是否等于模态间异源 WKV 尚不清楚，暂不按标题/模块名强行归类。

---

## 对照组: 刻意不用交叉的 RWKV 多模态 + "第一"基础设施

- **[RWKV-CLIP: A Robust Vision-Language Representation Learner](https://aclanthology.org/2024.emnlp-main.276/)** | 2024, **EMNLP 2024 主会** (2024.emnlp-main.276; arXiv:2406.06973) | CCF B | 摘要逐字自称 "**the first RWKV-driven vision-language representation learning model**"。双塔 + 对比学习, 塔间无 cross-attention——"不做 CA 的 RWKV 多模态"的对照组原型。
- **[ModRWKV](https://arxiv.org/abs/2505.14505)** | 2025, EMNLP 主会 | CCF B | 多模态 RWKV LLM 旗舰, 模态融合用 concat adapter, 无 CA——RWKV-CLIP 的 LLM 版对照组。
- **[VisualRWKV](https://arxiv.org/abs/2406.13362)** | 2024, arXiv | **vanilla VisualRWKV** 主路线不用 CA；此结论不外推到 VisualRWKV-HM，后者有 Cross State Mixing。
- **Otter** | 2026, AAAI Oral | 增强型 RWKV 小样本动作识别, 全文无 CA (原型匹配用范数距离)。Video RWKV 动作识别系列的同期延续, 厘清归属用。
- **MG-RWKV** | 2026, ECCV | CCF B | 音视频伪造定位; 全文核实其 "cross-attention feature pyramid" 属于被引的 UMMAFormer 而非自身。
- **VRWKV-Editor** | 2025, arXiv:2509.25998 | 视频编辑; 原文明确 "adding more cross-attentions or temporal attentions only increased the complexity"。
- **Delta-WKV** | 2025, MICCAI (arXiv:2502.20852) | CCF B | 医学 in-context 超分, 用 delta 规则做推理期动态权重——**RWKV 路线绕开 cross-attention 处理参考引导的实例, 与 delta-rule 论述直接相关的重要对照**。
- 其他低相关 (跨尺度/跨阶段/通道独立, 非本文义): HART、RWKV-X、MRT、MD-RWKV-UNet、SCRWKV、MFC-RFNet、FS-RWKV、RWKV-UNet、GDSR、MARCS、MDS-RWKV、Otter 前述等, 略。

---

## 复核后的整体图景（2026-11-04）

1. **类型①不再是少量预印本。** OTN-RWKV、RS-RWKV、MTRWKV 都已进入 CCF A；EvRWKV、X2Bone、SleepRWKVNet、TE-MSTAD 又覆盖了 K 异源、R/K/V 异源、delta-rule 状态更新和“他路 K 均值”等不同组合。
2. **类型②也已有 A 类。** RWKVFusion（TPAMI）与 TIMotion（CVPR）都把多输入统一成序列/状态后由 RWKV 替代原 CA；VisualRWKV-HM 则展示了“文本读图像矩阵状态”的另一种线性跨流读取。
3. **类型③不能统称标准二次型。** CMRWNet、MTRNet、RWKV-CD、VAFTrack、REB-former 是显式 QKᵀ CA；ChangeRWKV 是 Cross-CBAM；PRFusion 是论文声明的线性 ECMA；TemLo 等是调制模块。机制复杂度必须逐条写，不能由“cross-attention”名称推断。
4. **“图—图/图特征—图特征”已形成清晰子群。** 与本课题最接近的主线不只 CMRWNet、MTRNet、VideoRWKV，还包括 RS-RWKV、MTRWKV、RWKV-CD、X2Bone、EvRWKV、VAFTrack、RWKVMatch、PRFusion、MSFusion、CROWNFuse 等。
5. **不能再使用固定“44篇”“类型③共20篇”或“确认真空白”。** 本轮单靠引文、作者链、社区资讯与代码/专利反查就新增多篇标题不含 CrossAttention 的工作，说明数量会继续变化。

## 与本论文最相关的图—图 / 图特征—图特征对照

| 工作 | 两路输入 | 交叉位置 | 是否两步/双向查询 | 对本论文的直接启示 |
|---|---|---|---|---|
| VideoRWKV/LCR | 主干+历史态 / 当前边缘 | WKV 内 R/G 对 K/V | 不是对称两步；R/G 双门串联调制 | 门控深度与边缘条件化 |
| MTRNet | 单时相内 VRWKV 主体 / PDC 边缘 | RWKV 外标准 CA | 每个时相各自一次；非 T1↔T2 | 与 LCR 来源拓扑相似、算子不同 |
| RWKV-CD | VRWKV body / PDC edge | RWKV 外标准 CA | **是，body→edge 后 edge→body** | 最明确的顺序双向查询基线 |
| CMRWNet | MS / PAN | 窗口双向 CA + PAN-RWKV | 双向窗口对齐 | 局部 CA + 线性全局混合 |
| RS-RWKV / MTRWKV | MS / PAN | WKV 内 `R←MS, K/V←PAN` | 单向更新后再与 PAN 融合 | 稠密重建中异源 R/K/V 已有 A 类先例 |
| X2Bone | AP / LAT | WKV 内异源 R 与 K/V | 两个对称 CrossBlock | 结构图可隐藏双输入，须看 forward/公式 |
| EvRWKV | image / event | WKV 内交替异源 K/V | 双向交替扫描 | K 从对侧、V 留本路的另一种分工 |
| VAFTrack | template / search | 显式 CA 后接 Bi-WKV | search Q 读取拼接 K/V | CA 与 WKV 串联，不是 Cross-WKV |

### 对此前三个判断的最终结论

- **MTRNet 与 VideoRWKV**：可以说“输入来源分配/结构图外观近似”，不能说交叉注意力机制几乎相同。前者是外置 Softmax CA，后者是 WKV 内 R/G–K/V 门控递归。
- **X2Bone**：确有双输入 Cross-RWKV；结构图省略箭头造成了误判。R 与 K/V 的具体 AP/LAT 标签存在论文文字—代码调用方向差异，但“双视角异源输入”无疑。
- **ChangeRWKV**：用户判断正确。交叉发生在 Cross-CBAM，不在 RWKV/WKV；它也不是标准 QKᵀ CA。

## 只能条件化陈述的剩余空档

- 尚未发现：**视觉稠密预测 + CCF A + RWKV-7 delta-rule 异源状态更新**的完整组合；SleepRWKVNet 已覆盖非视觉睡眠分期，故不能写“delta-rule 跨流无人认领”。
- 本仓库的“同模态统计源 → 多个任务分解查询（N/L/M）→ WKV 内双门调制”这一完整组合，当前仍未见完全同型公开工作；但应称“截至复核日未发现”，不称“确认真空白”。
- 高光谱、SLAM/点云、RWKV×Mamba 等方向目前只见邻接工作或非注意力交叉；同样只作检索状态，不作不存在证明。

## 撤稿、边界与不计入项

- **[C3-OWD](https://openreview.net/pdf?id=u4vQPWy3Vk)**：论文公式是双向 Cross-RWKV（R 来自一路，K/V 来自另一路），但稿件已撤回，且官方代码的 `CrossModalExchange` 只是 GAP+FC 通道门控，没有论文所述 R/K/V 路径；标 **X**，不计正式语料数量。
- **[HMT](https://aclanthology.org/2025.naacl-long.410/)**：NAACL 2025 的模型无关长上下文插件，实验包含 RWKV backbone，当前 segment query 对历史 memory embeddings 做 CA 后把 prompt 喂给 RWKV。若口径是“任何 CA+RWKV 系统”可归③；若研究 RWKV 专用跨流机制，则仅作外围系统参考。
- RWKV-IR、WuNeng、CrossMamba 分别是跨扫描方向、跨头和非 RWKV 同构前身；保留作设计参照，不计“不同输入流的 CrossAttention-RWKV”核心数量。

## 方法学备注（供后续复查）

- 关键词搜索会漏掉标题/摘要不含 RWKV 的模块化工作（如 LADY），也会把“cross-modal loss / concat / CBAM / 自注意力”误收为跨流 CA；必须查看公式和数据流。
- 引文滚雪球的重要命中包括 RICEFuse→CROWNFuse/PRFusion/Learning to Prefer；作者链命中 RS-RWKV→MTRWKV；社区资讯命中 IESS-FusionNet/TE-MSTAD；代码和专利分别暴露 PRFusion/C3-OWD 不一致与 SleepRWKVNet 的 delta-rule 细节。
- Semantic Scholar/OpenAlex 的引用数会变化，本文不再用“零引用”支撑机制优先权；任何“first”只在作者原文明确声明时引用，并注明是作者自称。
- CCF 分级按 2026-03-31 第七版官网分类页核对；期刊 IF 易随年度变化，只作辅助信息，不作为机制证据。

## 本轮一手来源索引与复核优先级

- **公式级主来源**：NeurIPS PDF（RS-RWKV）、CVF/官方 PDF（TIMotion、DRWKV、MTRWKV）、PMLR（E²I-VRWKV）、ISCA Archive（REB-former）、Springer 全文（VAFTrack）、TechScience 全文（TE-MSTAD）、Optics Journal 全文（MTRNet）。
- **代码级主来源**：EvRWKV、X2Bone、OpenPAR/OTN-RWKV、VisualRWKV-HM、RWKVFusion、PRFusion、C3-OWD。代码只用于验证“已发布实现做了什么”，不能在论文与代码冲突时替代论文声明。
- **补充机制来源**：SleepRWKVNet 同源专利用于补足付费论文未公开的 Bi-IFM/delta-rule 细节；ScienceDirect/IEEE/Springer 出版商页用于核对题名、卷页、DOI、摘要和作者声明。
- **下一轮最高优先级**：取得 Learning to Prefer、RWKV-Enhanced IR-Vis、RWKVMatch、MMSegRWKV 的完整公式/代码；复查 Surv-RWKV 与 MMCGR 是否应从②/③进一步细分到①。

> 维护规则：新增论文先填“输入两路是什么—交叉发生在哪—R/Q/K/V/状态分别来自哪—是否显式 QKᵀ—证据等级”，再决定类型；不要先按论文标题里的 `Cross`、`Fusion`、`Attention` 分类。
