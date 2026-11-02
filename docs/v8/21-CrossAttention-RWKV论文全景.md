# Cross-Attention + RWKV 论文全景 (修订版, 已逐篇核实)

> 创建于 2026-11-02; 同日完成逐篇联网核实 (4 组并行检索, 每项判定附来源 URL, 见文末汇总)
> 同日第三轮: 以已知论文的前向/后向引用为线索 + arXiv/OpenAlex 全量扫描查漏 — 新增约 25 篇 (高相关 12), 见 F 节
> 性质: 文献调查记录。目的: 收集 Cross-Attention + RWKV 相关论文, 以及**论文自身对创新点的定位与评价** (如 "第一个..."), 并标记年份、venue、IF、CCF 分级
> 与本仓库的关联: `06-CrossRWKVGate-Q重构设计.md` 的来源论文 (arXiv 2411.05636, 即下文 B.2 的 Video RWKV / LCR) 在本轮补入并重新归位

## 核实修订要点 (相对上一轮调查原文)

上一轮原文有六处实质性错误或不准确, 已在正文中修正:

| # | 原文说法 | 核实结果 |
|:--:|---|---|
| 1 | RrmTrack 发表于 TMM, "CCF A (TMM 已升 A, 第七版)" | ❌ CCF 第七版已于 **2026-03-31 正式发布, TMM 仍为 B 类** (第六版亦为 B; MyHuiban 等第三方站标 A 与官网矛盾)。TMM IF 也非 8–9, **2024 JCR = 9.7** |
| 2 | CrossWKV 自称 "第一个把 cross-attention 形式化嵌入 RWKV-7 WKV 递归" | ❌ 论文全文**无 "first" 优先权声明**, 只自称 "novel"; "第一个" 是外部归纳。引用量 0 (Semantic Scholar, 2026-11-02) |
| 3 | RrmTrack 自称 "第一个把跨模态交互本身用 RWKV 算子实现引入视觉语言跟踪" | ⚠️ 摘要只自称 "a **novel** modality interaction method", 无 "first"; 正文付费墙未核 |
| 4 | CMRWNet "CCF C" | ❌ **JSTARS 不在 CCF 目录** (第六/七版均无) |
| 5 | CMGN 按 "GNN + RWKV 特征级融合" 列在边缘相关表 | ⚠️ 该文实为**文本假新闻检测**论文 (非图像融合), 保留但已标注领域 |
| 6 | Video RWKV "引用量约 7"; 多篇 IF 数值 | 引用量实为 **5** (Semantic Scholar); IF 已按 2024 JCR 全部更新 (Multimedia Systems 3.1 非 3.5; Neurocomputing 6.5 非 5.5; TCSVT 8.4 非 8.3 等) |

另有三处**措辞级修正**: CrossWKV 的实验结论是 "matching state-of-the-art / rivaling DiT-XL/2" 且自注为初步结果 (非 "称达 SOTA"); CVPR 2026 非对称融合的 "不对称" 在于事件 token top-K 筛选与两支路分别施加 Q-shift (摘要无 "移位配置不对称" 字样); RWKV-CLIP 自称 "the first RWKV-driven **vision-language representation learning model**" (wiki 转述的 "multimodal model" 与原文有出入)。

## A. 机制级创新

### 1. CrossWKV ⭐ (最核心)

**[Cross-attention for State-based model RWKV-7](https://arxiv.org/abs/2504.14260)** | 2025, arXiv:2504.14260 (2025-04-19 提交, 仅 v1; Liu Xiao, Li Zhiyuan, Lin Yueyu) | 无 IF / 未入 CCF | 引用量 0

- **定位声明**: 摘要自称 "a **novel cross-attention mechanism** for the state-based RWKV-7 model"。**注意: 全文无 "first" 优先权声明**——"第一个把 cross-attention 形式化嵌入 RWKV-7 WKV 递归" 是本调查的外部归纳, 不是论文自称; 引用时只能引 "novel"。
- **设计** (全部核实属实): 文本嵌入 (CLIP 编码器输出) 作为 query 侧源、图像特征作为序列主体, 填充对齐后单次前向完成跨模态全局交叉注意力; 文本 padding 到图像序列长度后线性投影生成 R, K/V 来自图像特征; **time-shift 操作 (δ = shift(x) − x) 保留并作用于图像 token 流**; 衰减项 w 用 LoRA 分解 (rank-64; 另 value rank-16、gate rank-128), 学习率 a 与门 g 也 LoRA 自适应; K 侧 L2Norm + 头级缩放。复杂度 O(T·N·H)。
- **理论卖点** (核实属实): "non-diagonal, input-dependent transition matrix enables it to represent complex functions **beyond the TC0** complexity class"; "state-tracking tasks like **S5 permutation modeling**"。
- **评价**: DIR-7 (RWKV-7 扩散) 在 ImageNet 256×256 上 FID 2.88、CLIP score 0.33 属实, 但原文措辞是 "**matching** state-of-the-art performance" / "**rivaling** DiT-XL/2", 且评估部分自注结果 "in progress"、"preliminary results should be interpreted with caution"。恒定内存 + 线性扩展。**注意**: 预印本未同行评审、零引用; 但它是后续 Cross-RWKV 命名工作的机制参照。

## B. 时空/视频任务 (本轮重点补充)

### 2. Video RWKV / LCR ⭐ (上轮遗漏, 本轮修正)

**[Video RWKV: Video Action Recognition Based RWKV](https://arxiv.org/abs/2411.05636)** | 2024, arXiv:2411.05636 (2024-11-08 提交, 仅 v1; Zhuowen Yin, Chengru Li, Xingbo Dong) | 无 IF / 未入 CCF | 引用量 5 (Semantic Scholar, 2026-11-02)

- **定位声明** (逐字核实): "this work **introduces RWKV to the video domain in a novel way**"; "These advantages enable LSTM CrossRWKV to **set a new benchmark in video understanding**"。模型名 **LCR** (LSTM-Cross RWKV) 核实属实。
- **Cross RWKV gate** (核实属实): 当前帧的边缘信息与过去帧特征通过 Cross RWKV 门交互——边缘特征引导"注意主体在哪里", 并跨帧全局聚合特征; 边缘信息充当 LSTM 的遗忘门 ("The edge information is passed through the forgetting gate of LSTM... to reduce the effect of background noise")。即跨时刻信息交互被门控化地写进了 RWKV/LSTM 的状态更新, 功能上等价于"跨流 cross-attention 的门控递归版"。
- **配套设计**: 增强 LSTM 递归存储长期记忆 (论文仅断言, 未展开); tube masking 降冗余、抗过拟合。
- **评价**: 线性复杂度; 代码模型全公开。**诚实说明**: 未正式发表的 arXiv 预印本, 引用量 5, "new benchmark" 的自评在社区影响力上并未兑现——但它确实是 "Cross + RWKV gate" 设计组合的**最早文字记载之一**, 比 CVPR 2026 的 Asymmetric RWKV Fusion 早了一年多。
- **归位修正**: 它的 Cross RWKV gate 本质上是"Q 侧信息 (当前帧边缘) 与 KV 侧信息 (历史特征)"的门控交叉, 与 CrossWKV 的 query 侧注入在结构思想上同源, 且都发生在 RWKV 状态更新内部。**本仓库 `06-CrossRWKVGate-Q重构设计.md` 的任务 Q 机制正是移植自该论文式 (2)–(8)。**

## C. 任务应用层

### 3. RrmTrack ⭐

**[RWKV-Inspired Multi-Modal Relation Modeling for Vision-Language Tracking](https://ieeexplore.ieee.org/document/11417442)** | 2026, IEEE TMM, vol.28, pp.6892–6901, 已正式刊出 (DOI: 10.1109/TMM.2026.3668700; Guangtong Zhang 等) | IF **9.7** (2024 JCR; 2025 JCR 第三方聚合为 9.9, 未经官方确认) | CCF **B** (第七版; TMM 未升 A)

- **定位声明**: 摘要逐字自称 "We introduce a **novel modality interaction method specific to vision-language object tracking based on RWKV**, providing customized interaction for different modalities... and effectively reducing the computational time cost of cross-modal interaction"。**摘要无 "first" 自称**; "第一个把跨模态交互本身用 RWKV 算子实现引入视觉语言跟踪" 是外部归纳 (正文付费墙未核)。
- **设计分工** (摘要逐字核实, 极有启发性): "**uses a time mixing module to model the relationship between language and image features and a channel mixing module to facilitate information interaction between images**"——跨模态走 time-mix、模态内走 channel-mix, 把 Spatial/Channel Mix 分工扩展到了模态维度。
- **配套** (全部摘要原文): Siamese + One-stream 混合特征提取; 信息恢复模块削减搜索图对模板图的交互干扰; "parallelized training with a linear attention mechanism and efficient RNN inference"; TNL2k/LaSOT/OTB-Lang/LaSOText/MGIT 五基准 "achieves state-of-the-art results and speed"。

### 4. Surv-RWKV

**[Surv-RWKV: Cross-modal receptance weighted key-value interaction with optimal transport feature alignment for survival analysis](https://www.sciencedirect.com/science/article/abs/pii/S0957417426004197)** | 2026, Expert Systems with Applications, vol.312, art.131506 | IF 7.5 (2024 JCR; 2025 JCR 9.4) | CCF C

- **定位**: RWKV 浅层融合模块 ("dynamically explores cross-modal dependencies through **spatial-channel hybrid operations**") + **cross-RWKV 深度交互模块 ("strengthen information synthesis via iterative cross-attention mechanisms")** + 最优传输对齐 (标题原文)。五个 TCGA 癌症队列验证 ("Extensive validation across five TCGA cancer cohorts")。"SOTA" 逐字措辞未能取得 (出版社 403), 语义上为 superior performance。

### 5. RWKV-VG

**[RWKV-VG: visual grounding with RWKV-driven encoder-decoder framework](https://link.springer.com/article/10.1007/s00530-025-01720-w)** | 2025, Multimedia Systems, vol.31, art.124 | IF 3.1 (2024 JCR; 3.5 为 2023 旧值; 2025 JCR 2.8) | CCF C

- **定位声明** (逐字核实): "**the first visual grounding framework entirely built on the RWKV architecture**"——纯 RWKV encoder-decoder (视觉编码器 + 语言编码器 + 视觉语言解码器, 无 CNN/Transformer), 解码器内做 intra-modal + cross-modal 推理; 可学习 [REG] token 回归框 (消融显示置于融合序列末端优于首位)。
- **评价**: RefCOCO/RefCOCO+/RefCOCOg/ReferItGame 评测; 摘要**自述 QRNet/CLIP-VG 等仍优于它** (归因于预训练数据量 15M vs 400M)。

### 6. CMRWNet

**[CMRWNet: Cross-Modal RWKV Network With Window-Guided Alignment for High-Fidelity Pansharpening](https://ieeexplore.ieee.org/document/11614160)** | 2026, IEEE JSTARS, vol.19, pp.24069–24084 | IF 5.3 (2024 JCR; 2025 JCR 6.3) | **不在 CCF 目录** (第六/七版均无 JSTARS; 遥感类在目录中的是 TGRS-B / GRSL-C)

- **定位**: **窗口级双向 cross-attention (subquadratic cost) 做 MS-PAN 精对齐 + 堆叠 PAN-RWKV ("row and columnwise parallel prefix-decay scans" + receptance gating) 做线性全局混合**——cross-attention 局部化与 RWKV 全局化的混合范式。IKONOS/Pléiades/WV-3 上 "outperforms state-of-the-art methods" + QuickBird 跨传感器验证。

### 7. EvRWKV

**[EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement](https://arxiv.org/abs/2507.03184)** | 2026, IEEE TCSVT, vol.36, no.7, pp.10201–10216 (DOI: 10.1109/TCSVT.2026.3672491) | IF 8.4 (IEEE 官方当前值; 8.3 为上一版 JCR) | CCF B

- **定位**: 连续跨模态交互 (反对 early/late fusion 二分: early 有模态异构性、late 切断特征关联): Cross-RWKV Module 捕捉细粒度时间与跨模态依赖 + EISFE (Event Image Spectral Fusion Enhancer, 频域去噪 + 空域对齐)。
- **数字核实属实**: SDE/SDSD 较 image-only 方法 PSNR +1.79/+1.85 dB; 下游语义分割 mIoU +35.44%。

### 8. X2Bone

**[X2Bone: Reconstructing 3D Bone Structures from 2D Biplanar X-Rays via Cross-RWKV](https://papers.miccai.org/miccai-2026/1163-Paper0798.html)** | 2026, MICCAI, LNCS 16889, Springer (open access) | 顶会无传统 IF | CCF B (第七版; 归"交叉/综合/新兴"类, 不在 AI 类)

- **定位**: Cross-RWKV 模块 (Bi-WKV 算子) 做**跨视角** (AP/侧位 X 光) "long-range, bidirectional cross-view interaction with linear complexity"; 配套 3D Expand "lifts the fused 2D features into volumetric space via view-dependent depth probability modulation"。CTPelvic1K 上最优 (DSC 87.96%, 11.02M 参数)。
- **实现细节 (2026-11-02 核实官方代码 github.com/pzhhhhh2263/X2Bone)**: 双流**共享权重 CNN 编码器** (MONAI UnetrBasicBlock, RWKV 只出现在交叉块), 每尺度两个对称 `CrossBlock(x_a, x_b)` / `(x_b, x_a)` (AP→LAT 与 LAT→AP); `CrossBlock` **双输入单输出**——R 投影来自一侧视角、K/V 来自对侧视角 (代码: `x_cross_a = cross_block1(x_a, x_b)`, R 取 x_a、K/V 取 x_b; 作者 rebuttal 的表述方向相反, 以代码为准), 残差回 R 侧; 门控为 **`sigmoid(r) ⊙ WKV`**——无带符号 R、无逐头 LN、无 SiLU(G) 第二门; 两路输入各过独立 OmniShift; Bi-WKV 为偶次直扫/奇次转置扫的双向扫描, 衰减按 1/T 缩放; 实现文件名 `Restore_RWKV_cross_r_new.py`——即 **Restore-RWKV 块的"R 改异源"改造**, 与本仓库同源路线。Cross-Shift (扫描前把对侧视角 token 经 1×1 卷积混入当前 token 的局部交换, 作者明确"非几何对齐") 据 rebuttal 存在, 发布代码中未见明确对应。**与本仓库的对照**: 与旧 `receptance` 模式 (`sigmoid(q)·wkv`, q 异源) **结构同型**; 比现行 `cross_rwkv` 模式 (LCR 式 R/G 双门: 带符号 R ⊙ WKV + 逐头 LN + SiLU(G)) **浅一层**; 无任务分解查询、无统计源分配。

### 9. Asymmetric RWKV Fusion

**[RGB-Event based Pedestrian Attribute Recognition: A Benchmark Dataset and An Asymmetric RWKV Fusion Framework](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_RGB-Event_based_Pedestrian_Attribute_Recognition_A_Benchmark_Dataset_and_An_CVPR_2026_paper.html)** | 2026, CVPR, pp.32745–32755 (Xiao Wang 等; arXiv:2504.10018) | 无传统 IF | CCF A (本表中唯一 CCF A)

- **定位**: 双模态各自 Vision-RWKV 编码 (Visible-RWKV + Event-RWKV) + **OTN-RWKV 非对称融合**: 用相似度矩阵从事件 token 中筛选 top-K 最具代表性的 token 对 (RGB 支路保留完整), 且两支路分别施加对应式 (1) 与式 (3) 的 Q-shift——"Q/KV 来自不同模态、token 处理不对称"的跨模态实例。**注意**: 摘要无 "移位配置不对称" 字样, 该表述是对正文机制的归纳。

## D. 边缘相关 (特征级"交叉", 非注意力)

| 论文 | 年份/Venue | IF (2024 JCR) | CCF | 交互形式 |
|---|---|---|---|---|
| [RWKV-Enhanced IR-Vis Fusion](https://ieeexplore.ieee.org/document/11605729) (全称 "RWKV-Enhanced Infrared and Visible Image Fusion: Capturing Long-Range Dependencies with Cross-Modality Correlations") | 2026, PRMVAI (IEEE Int. Conf. on Pattern Recognition, Machine Vision and AI, 第 4 届) | — | 不在目录 | spatial-channel dual-hybrid **RWKV-Fuser** 捕捉长程依赖与跨模态通道相关性 |
| [RICEFuse](https://www.sciencedirect.com/science/article/abs/pii/S1350449526001945) | 2026, Infrared Phys. Technol., vol.156 | 3.4 (官网 3.3) | 不在目录 | DFCEM 互补增强 + AMT-Shift + 余弦相似度跨模态对齐损失 (均逐字核实) |
| [CMGN](https://www.sciencedirect.com/science/article/pii/S0925231225004837) ("CMGN: Text GNN and RWKV MLP-mixer combined with cross-feature fusion for fake news detection") | 2025, Neurocomputing, vol.633 | 6.5 (Elsevier 官网快照 5.1, 来源有出入) | C | ⚠️ **文本域 (假新闻检测)**, 非图像融合: Text GNN + RWKV MLP-mixer + cross-feature fusion (标题核实; 摘要原文不可得) |
| [RWKV-CVM](https://www.mdpi.com/2673-4826/7/2/58) ("RWKV-CVM: Gated Cross-Variate Mixing for Multivariate Power Load Forecasting") | 2026, Electricity (MDPI), vol.7 no.2 | 2.2 | 不在目录 | "gated, row-stochastic mixing matrix... adding only **65 trainable parameters**" (逐字核实; 电力负荷预测) |

## E. 对照基础设施 (常被交叉引用的"第一")

- **[RWKV-CLIP: A Robust Vision-Language Representation Learner](https://aclanthology.org/2024.emnlp-main.276/)** | 2024, **EMNLP 2024 主会** (非 Findings; 2024.emnlp-main.276, pp.4799–4812; arXiv:2406.06973) | CCF B | 摘要逐字自称 "**the first RWKV-driven vision-language representation learning model**" (注意: 原文是 representation learning model, wiki 转述的 "multimodal model" 与原文有出入)。双塔 + CLIP 式对比学习对齐, 塔间无 cross-attention——它是"不做 cross-attention 的 RWKV 多模态"的对照组。

## 修订后的整体图景 (核实后)

1. **机制层**: CrossWKV (arXiv 2025) 仍是唯一在 WKV 递归内部定义 cross-attention 的工作, 但它**连 "first" 都未自称**且零引用; Video RWKV (LCR, arXiv 2024-11) 是时间上更早的 "Cross gate + RWKV" 尝试 (引用 5)。两条线索都是预印本——"机制级 RWKV-cross-attention"不仅没有定论性论文, 连优先权声明都还没有人正式写下。
2. **算子替代层**: RrmTrack (TMM 2026, **CCF B**, IF 9.7, 已正式刊出) 代表第三条路线——不引入 cross-attention, 直接用 RWKV 的 time-mix 充当跨模态交互算子、channel-mix 承担模态内交互。它是这条路线**发表层级最高**的工作; 本列表中唯一的 CCF A 是 CVPR 2026 的 Asymmetric RWKV Fusion, 但那属于编码器+融合层路线, 不动 WKV 本体。
3. **混合层**: CMRWNet (窗口 cross-attention + RWKV 扫描)、Surv-RWKV (迭代 cross-RWKV 深度交互)、EvRWKV (连续交互) 构成 "cross-attention 与 RWKV 分工/嵌套" 的谱系。
4. **应用层**: Cross-RWKV 命名已扩散到模态间 (EvRWKV、Surv-RWKV)、视角间 (X2Bone)、帧间 (Video RWKV 的 Cross RWKV gate), 但多数未动 WKV 本体。

**修订后的空白提示**: 把 CrossWKV 的 delta-rule 式跨序列状态演化正式用于视觉任务并发表于 CCF A 会议/期刊的工作**仍然空缺**——RrmTrack 最接近 (TMM, CCF B, 模态间 time-mix 交互), 但它用的是标准 time-mix 而非 CrossWKV 的广义 delta rule; LCR 的 Cross gate 思想最接近但停留在预印本。鉴于 CrossWKV 至今零引用、无优先权声明, 这个机制点目前**完全无人认领**, 仍是可切入的点。

> ⚠️ 本结论写于第三轮之前。第三轮发现 ChangeRWKV (AAAI 2026) 后需收窄为: 跨**时相**的标准二次型 cross-attention + RWKV 已进 CCF A, 但 **WKV 递归内部机制级 (delta-rule 式) cross-attention 发表于 CCF A 仍空缺**; 详见 F.5。

> 分级口径 (CCF 第七版, **2026-03-31 正式发布**, 官网 ccf.org.cn 可查): TMM 为 **B 类** (第六版亦为 B; MyHuiban 等第三方站标 A 属错误信息, 勿引用); CVPR = A; MICCAI = B (交叉/综合/新兴类); EMNLP = B; TCSVT = B; ESWA / Neurocomputing / Multimedia Systems = C; **JSTARS、Infrared Physics & Technology、Electricity、PRMVAI 不在目录**。Video RWKV 与 CrossWKV 为 arXiv 预印本, 无 IF 与 CCF 分级。

## F. 第三轮补充: 引文滚雪球与全量扫描的新发现 (2026-11-02)

方法: ①对全部已知论文做**前向引用追踪** (Semantic Scholar / OpenAlex citations API) + 对 CrossWKV/Video RWKV/RrmTrack/EvRWKV 做**后向引用**分析; ②arXiv API 全量枚举 (`all:RWKV` 133 篇逐条过滤 + 多组 cross 关键词); ③OpenAlex 关键词 300+ 条; ④按领域扫盲区 (医学/视频/遥感/时序/VLM/音视频/3D)。新发现的关键条目已二次核实 DOI/卷期/IF/CCF (CCF 均查官网第七版分类页)。

### F.1 高相关新增 (12 篇)

**IR-可见光融合赛道 (进一步拥挤):**

1. **Learning to Prefer: Reward-Driven Infrared-Visible Fusion via RWKV** | Jialing Huang 等 (通讯 Xiaotong Tu, Restore-RWKV 团队) | Information Fusion, vol.138, art.104685 (2026 在线) | **IF 14.7 (2024 JCR, 本主题最高 IF)** | 不在 CCF 目录 (AI/交叉两类已查)。奖励驱动的复原+融合统一框架, 含 **Cross-RWKV 模块** 与跨模态差异感知融合。⚠️ 摘要付费墙, Cross-RWKV 表述经 S2/ScienceDirect 索引交叉印证, 未逐字取得。发现路径: EvRWKV、RICEFuse 引用链 (两个独立代理各自命中)。
2. **CROWNFuse** | Pengchen Zhang 等 | Neurocomputing, art.134876 (2026) | IF 6.5 | CCF C。"Recurrent cross-modal memory interaction with edge-frequency recalibration"——RWKV-based Hierarchical Cross-Modal Fusion。发现路径: RICEFuse 引用链。

**音视频与生理信号 (原以为空白):**

3. **TemLo** | Yunzhi Zhuge 等 (大连理工) | IEEE TMM, vol.28, pp.9804–9815 (2026) | IF 9.7 | CCF B。Audio-Visual Segmentation: 模态专属 RWKV 编码器双流 + **Acoustic Modulation Pyramid** 结构化音视频跨模态交互 + Vision-guided Modality Injector。代码公开 (github.com/c1ircle/TemLo)。**音视频域此前认为是空白, 本轮补上**。
4. **SleepRWKVNet** | Hang Zhang | Biomedical Signal Processing and Control, vol.113, 108713 (2026) | IF 5.7 (Elsevier 官网现值) | 不在 CCF。多模态睡眠分期 (EEG/EOG/EMG): "bidirectional interactive RWKV" 整合三模态。已被引 6 次。
5. **IESS-FusionNet** | Junyuan Feng 等 (深圳大学) | Bioengineering 13(1):57 (MDPI, 2026) | IF 4.4 (2023 JCR; 2024 值未核实) | 不在 CCF。EEG-EMG 婴儿癫痫痉挛检测: **Cross Time-Mixing 模块 "built upon the linear recurrent attention of the RWKV architecture... replacing the quadratic cross-attention"**——算子替代路线继 RrmTrack 之后**第二个显式宣称"用 RWKV 替代 cross-attention"的实例**。

**遥感与检测:**

6. **ChangeRWKV (Beyond Quadratic: Linear-Time Change Detection with RWKV)** | Zhenyu Yang 等 (南京理工 Yazhou Yao 组) | **AAAI 2026, Vol.40(14), pp.11811–11819** (DOI: 10.1609/aaai.v40i14.38167, Crossref 已核) | CCF A。双时相变化检测: 层级 RWKV 编码器 + **Cross CBAM** 跨时相互注意力 ("channel attention weights are computed from one temporal feature and applied to the other, and vice-versa")。**cross-attention×RWKV 进入 CCF A 的第一个实例**——但跨的是时相不是模态, 且 cross-attention 是标准二次型, 不在 WKV 内部 (对空白提示的修正见 F.5)。
7. **C2F-VRWKV** | Shounan Wang | IEEE TGRS (2026) | IF 8.6 (2024 JCR) | CCF B (交叉/综合/新兴)。舰船检测: Vision-RWKV 骨干 + Cross-Frequency Fusion Module 的 "frequency-guided cross-attention" (跨频而非跨模态)。
8. **MTRNet** | 潘娅婷 | 《激光与光电子学进展》(2025), DOI: 10.3788/lop251946 | 中文期刊 (无 IF/CCF)。RWKV 编码器 + 交叉注意力做跨通道与空间联合建模, 双时相变化检测。中文文献遗漏项。

**医学影像 (本轮最大遗漏重灾区):**

9. **ITC-RWKV** | Yating Huang | arXiv:2510.21479 (2025-10 预印本)。组织病理亚型分型: RWKV 式 receptance-weighted KV 聚合 (线性复杂度) + **bidirectional tissue-cell interaction module** (细胞↔组织双向互注意力)。类型②③。
10. **DPC-QA Net** | Qijun Yang | arXiv:2509.15802 (2025 预印本)。病理图像无参考质量评估: Aggr-RWKV 细胞流 + "Cross-attention fusion" 对齐感知/细胞双流。
11. **MSFusion** | Liu Wang | Biosensors 16(8):423 (MDPI, 2026-08) | IF 5.6 (2024 JCR) | 不在 CCF。多模态医学图像融合: VRWKV/Bi-WKV 块 + MSCMCF 双向跨模态注意力。
12. **LADY** | Jihao Huang | arXiv:2512.15038 (2025-12 预印本)。自动驾驶: RWKV-7 融合多帧 camera+LiDAR + **lightweight linear cross-attention (LICA)**。⚠️ 摘要不含 "RWKV" 字样, 全文检索必漏 (由社区线索定位后核实)。

### F.2 中相关新增 (表格)

| 论文 | venue/年份 | IF (2024 JCR) | CCF | 机制要点 |
|---|---|---|---|---|
| [RWPNet](https://doi.org/10.1109/JSTARS.2026.3734262) | JSTARS 2026 | 5.3 | 不在 | 全色锐化 (与 CMRWNet 同赛道): SS-RWKV + dual-branch cross-fusion |
| [MMSegRWKV](https://doi.org/10.1109/JIOT.2026.3661927) | IEEE IoT-J 2026 | 9.6 | 不在 | 多模态 MRI 分割, dual-view WKV (DV-WKV) 空间/时间因式分解 |
| [渐进式 RWKV 多模态医学融合](https://doi.org/10.1016/j.asoc.2026.116704) | Applied Soft Computing 2026 | 6.6 | 不在 | 自称首个 RWKV 医学图像融合框架 |
| [SGF-RWKV](https://doi.org/10.1109/JBHI.2026.3739849) | IEEE JBHI 2026 | 6.8 | C | RWKV 统一医学去噪 + CLIP 语言嵌入生成任务感知注意力图 (语言→视觉引导) |
| [CRWKV](https://doi.org/10.24963/ijcai.2025/86) | IJCAI 2025 | — | A | Context-guided RWKV 多视图 (空域+FMix 频域) 融合去噪, BiWKV; "多视图"是特征视图非模态 |
| [PestVL-Net](https://doi.org/10.1109/CVPRF71909.2026.00847) | CVPR 2026 **FINDINGS Track (CVPRF)**, DOI 已核 | — | 同 CVPR 系 | RWKV 视觉通路 + 标准显式 cross-attention 融合文本 (Q=视觉, K/V=文本); CVPR 2026 新设 Findings track |
| [OccRWKV](https://arxiv.org/abs/2409.19987) | ICRA 2025 | — | B | Sem-RWKV + Geo-RWKV 双分支, **BEV-RWKV 充当跨分支融合算子** (RWKV 替代融合用注意力, 3D 语义占据) |
| [ModRWKV](https://arxiv.org/abs/2505.14505) | EMNLP 2025 主会 | — | B | 多模态 RWKV LLM 旗舰, 模态融合用 concat adapter, **无 cross-attention**——与 RWKV-CLIP 同属"不做 CA 的 RWKV 多模态"对照组 |
| [CrossMamba](https://arxiv.org/abs/2409.04803) | arXiv 2024 (目标声音提取) | — | — | **Mamba 状态空间递归内实现 cross-attention** (文本线索 Q→音频 K/V), CrossWKV 的直接思想前身 (CrossWKV 引用了它, 非 RWKV 载体) |
| [WuNeng](https://arxiv.org/abs/2504.19191) | arXiv 2025 (与 CrossWKV 同一作 Liu Xiao) | — | — | MHA 注意力头 × RWKV-7 状态头的 cross-head 交互 (跨流交互在头间而非模态间) |
| [GoldFinch](https://arxiv.org/abs/2407.12077) | arXiv 2024 | — | — | GOLD transformer 对 Finch RWKV 的 KV-cache 做 cross-attention (KV-cache 压缩导向) |
| [RWKV-IR](https://arxiv.org/abs/2412.03814) | arXiv 2024 | — | — | Cross-Bi-WKV: 融合两个不同扫描序的 Bi-WKV (WKV 内部跨方向交互, 类型①变体) |
| [MMCGR](https://www.sciencedirect.com/science/article/abs/pii/S0925231226008210) | Neurocomputing vol.682, 2026 | 6.5 | C | 对话情感识别: RWKV 跨模态表征 + 级联门控融合 (融合算子非显式 CA) |
| [PMSRWKV-GCN](https://www.nature.com/articles/s41598-026-41091-4) | Scientific Reports 2026 | 3.9 | 不在 | 多变量时序: RWKV 管时间维 + 注意力邻接 GCN 管跨变量维 (与 RWKV-CVM 门控混合**并行**的另一条跨变量路线) |
| TE-MSTAD | arXiv:2601.11951 | — | — | WSN 异常检测: RWKV-based Cross modal Feature Extraction (跨节点) |
| RWKV-VIO | Sensors 25(18):5737, 2025 | — | — | 视觉+IMU 双编码器 + RWKV 时序建模 (无显式 CA) |
| DME-RWKV | Bioengineering 13(1):12, 2025 | 4.4 (2023) | 不在 | OCT+UWF 双模态抗 VEGF 疗效预测 |
| Compass | arXiv:2608.03559 | — | — | Needle RWKV + 多模态证据融合 (Dempster-Shafer, 非注意力算子) |
| [Geometry-aware RWKV 光场 SR](https://doi.org/10.1109/LSP.2025.3555445) | IEEE SPL 2025 | 未核 | 不在 (AI/交叉页已查) | 异构光场融合, 恢复空间-角点 (跨视角) 相干性 |
| [P-RWKV/PointER](https://arxiv.org/abs/2606.10395) | arXiv 2026 | — | — | 点云自监督; "cross-modality" 经全文核实是 3D→2D 渲染生成式监督, **非** cross-attention |

### F.3 对照组与反向证据 (引用时勿混淆)

- **刻意不用 cross-attention 的 RWKV 工作**: VisualRWKV (arXiv:2406.13362, 多模态 LLM 主路线, 对比 LLaVA-1.5 提速 4×, 明确不用 CA)、Otter (AAAI 2026 Oral, 动作识别, 全文无 CA)、MG-RWKV (ECCV 2026, 音视频伪造定位, 其 "cross-attention feature pyramid" 属于被引的 UMMAFormer 而非自身)、VRWKV-Editor (arXiv:2509.25998, 视频编辑, 原文明确 "adding more cross-attentions or temporal attentions only increased the complexity")、HART (arXiv:2610.12883, 通道独立)。
- **重要对照 (与我们 delta-rule 论述相关)**: Delta-WKV (MICCAI 2025, arXiv:2502.20852) 在医学 in-context 超分中用 delta 规则做推理期动态权重——RWKV 路线**绕开** cross-attention 处理参考引导的实例。
- 其他低相关 (跨尺度/跨阶段, 非本文义): MD-RWKV-UNet、SCRWKV、MFC-RFNet、MG-RWKV 见上、RWKV-X、MRT、FS-RWKV、RWKV-UNet、GDSR、MARCS、MDS-RWKV 等, 略。

### F.4 确认的真空白 (多源枚举后仍无 CA×RWKV)

视频 VOS/复原/视频融合; 高光谱; SLAM/点云 (PointER 的 "cross-modality" 非注意力); **RWKV×Mamba 跨模态互操作**; 另外 **CrossWKV 前向引用经 S2+OpenAlex+WebSearch 三源确认仍为 0** (第三轮复验)。

### F.5 对整体图景与空白提示的修正

1. **"CCF A 空缺"需收窄**: ChangeRWKV (AAAI 2026, A) 与 CRWKV (IJCAI 2025, A) 出现后, CCF A 级已有两个 cross-attention×RWKV 实例, 但前者跨时相、后者跨特征视图, **均为标准二次型 cross-attention 挂在 RWKV 骨干上, 不动 WKV 本体**。"WKV 递归内部机制级 (delta-rule 式) cross-attention 发表于 CCF A"仍然空缺, CrossWKV 机制点仍无人认领 (零引用)。
2. **算子替代路线正在成形**: RrmTrack 之外, IESS-FusionNet 显式宣称 "replacing the quadratic cross-attention"、OccRWKV 用 RWKV 块充当跨分支融合算子、SleepRWKVNet 的双向交互 RWKV——该路线已从单点变成 4 个实例, 分布在 TMM/BSPC/Bioengineering/ICRA, 说明路线可行但尚无 A 类。
3. **IR-Vis 融合赛道最拥挤** (新增 Information Fusion 14.7 一篇), **医学影像是数量最多的应用域** (7+ 篇)。
4. CVPR 2026 新设 FINDINGS Track (CVPRF), PestVL-Net 属之; 引用时注意与 CVPR 主会区分。

### F.6 方法学备注 (供后续复查)

- arXiv 全文 token 检索会漏两类: 名称合成词不含 "RWKV" 的 (ModRWKV、LADY 摘要无 RWKV 字样)——必须靠引文滚雪球与 WebSearch 补。
- Semantic Scholar 公网 API 限流严重 (429), 建议带 key 复跑; OpenAlex 更稳定。
- 两条**未能核实、暂勿引用**的线索: IEEE TGRS 2026 建筑变化检测 Vision-RWKV 论文 (ieeexplore 11456675, 被拦); OpenReview kmNqnwA4aV "Bridging Transformers and RWKV" (称做了 RWKV±cross-attention 消融, CAPTCHA 拦截)。
- CCF 第七版补充核实 (2026-11-02, ccf.org.cn 官网分类页): AI 类会议 A 类共 7 个——AAAI、NeurIPS、ACL、CVPR、ICCV、ICML、IJCAI (**网传"IJCAI 降 B"不实; ICLR 不在目录任何级别**); ECCV、EMNLP、**ICRA** 均 B (AI 类); TGRS=B、JBHI=C (交叉/综合/新兴); Information Fusion、Applied Soft Computing、IEEE IoT-J、BSPC、IEEE SPL 均不在目录。

## 附: 逐篇核实判定汇总 (2026-11-02)


| 论文 | 存在性/venue/年份 | 机制声明 | 定位声明 ("first...") | IF | CCF |
|---|---|---|---|---|---|
| CrossWKV | ✅ arXiv:2504.14260, 2025-04-19 | ✅ 全部 (LoRA rank/O(T·N·H)/TC⁰/S₅) | ⚠️ 只自称 "novel", 无 "first" | — (预印本) | — |
| Video RWKV/LCR | ✅ arXiv:2411.05636, 2024-11-08 | ✅ 全部 (Cross RWKV gate/遗忘门/tube masking) | ✅ "introduces RWKV to the video domain in a novel way" | — (预印本) | — |
| RrmTrack | ✅ TMM vol.28 已刊出 | ✅ time/channel 分工逐字 | ⚠️ 只自称 "novel" | 9.7 (2024 JCR) | **B** (非 A) |
| Surv-RWKV | ✅ ESWA vol.312 | ✅ (SOTA 措辞未取得) | 无 first 自称 | 7.5 | C |
| RWKV-VG | ✅ Multimedia Systems vol.31 | ✅ | ✅ "the first... entirely built on RWKV" 逐字 | 3.1 | C |
| CMRWNet | ✅ JSTARS vol.19 | ✅ | 无 first 自称 | 5.3 | **不在目录** |
| EvRWKV | ✅ TCSVT vol.36 | ✅ (+1.79/1.85 dB, +35.44% 属实) | 无 first 自称 | 8.4 | B |
| X2Bone | ✅ MICCAI LNCS 16889 | ✅ 全部 | 无 first 自称 | — | B (交叉/新兴类) |
| Asymmetric RWKV Fusion | ✅ CVPR 2026 pp.32745-32755 | ✅ (不对称=事件 token top-K 筛选+双支路 Q-shift) | 无 first 自称 | — | A |
| RWKV-CLIP | ✅ EMNLP 2024 **主会** | ⚠️ 双塔无 CA 属正文细节 | ✅ "the first RWKV-driven vision-language representation learning model" | — | B |
| RWKV-Enhanced IR-Vis | ✅ PRMVAI 第 4 届 (IEEE) | ✅ RWKV-Fuser 逐字 | — | — | 不在目录 |
| RICEFuse | ✅ IPT vol.156 | ✅ 逐字 | — | 3.4 | 不在目录 |
| CMGN | ✅ Neurocomputing vol.633 (**假新闻检测**) | ⚠️ 标题证实, 摘要不可得 | — | 6.5 | C |
| RWKV-CVM | ✅ Electricity (MDPI) vol.7 | ✅ 65 参数逐字 | — | 2.2 | 不在目录 |

**核实来源** (一手页面): arXiv abs/html 页 (2504.14260, 2411.05636, 2507.03184, 2504.10018, 2406.06973); Semantic Scholar / OpenAlex / Crossref API; IEEE Xplore (11417442, 11614160, 11605729); Springer (RWKV-VG, X2Bone/miccai.org); ACL Anthology (2024.emnlp-main.276); CVF openaccess; MDPI; CCF 官网 ccf.org.cn/Academic_Evaluation (第七版公告 2026-03-31 + 各领域分类页); IF 来源: 各学会/出版社官网 + ablesci/impactfactor.cn 历年表。两处未决: RrmTrack 与 Surv-RWKV 的正文/完整摘要因付费墙未逐字核验, 摘要级声明均已核实。
