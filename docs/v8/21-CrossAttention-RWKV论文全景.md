# Cross-Attention + RWKV 论文全景 (按类型重组, 已逐篇核实)

> 创建于 2026-11-02; 同日完成逐篇联网核实 (三轮: 原始调查 → 逐篇核实 → 引文滚雪球)
> 本轮重组: 应要求把第三轮补充 (原 F 节) 整理进正文, **以"类型" (交叉机制与 RWKV 的结合方式) 作为唯一分类依据**, 不再按任务域/发现轮次分组; 任务域与流关系 (跨模态/跨视角/跨时刻/跨时相/跨频/跨变量等) 作为条目内属性标注
> 与本仓库的关联: `06-CrossRWKVGate-Q重构设计.md` 的来源论文 (arXiv 2411.05636, 即类型①条目 2 的 Video RWKV / LCR)

## 核实修订要点 (相对最初调查原文, 存档)

最初原文有六处实质性错误, 均已在正文修正:

| # | 原文说法 | 核实结果 |
|:--:|---|---|
| 1 | RrmTrack 发表于 TMM, "CCF A (TMM 已升 A, 第七版)" | ❌ CCF 第七版 (2026-03-31 发布) **TMM 仍为 B 类**; IF 非 8–9, **2024 JCR = 9.7** |
| 2 | CrossWKV 自称 "第一个..." | ❌ 全文无 "first" 优先权声明, 只自称 "novel"; 引用量 0 |
| 3 | RrmTrack 自称 "第一个..." | ⚠️ 摘要只自称 "novel", 无 "first" |
| 4 | CMRWNet "CCF C" | ❌ **JSTARS 不在 CCF 目录** |
| 5 | CMGN 按"GNN+RWKV 特征级融合"列图像融合 | ⚠️ 实为**文本假新闻检测**论文 |
| 6 | Video RWKV 引用"约 7"; 多篇 IF | 引用实为 5; IF 已按 2024 JCR 更新 (TMM 9.7 / MMS 3.1 / Neurocomputing 6.5 / TCSVT 8.4 等) |

## 分类体系 (唯一依据: 类型)

| 类型 | 定义 | 判据 |
|---|---|---|
| **① WKV 算子内部跨流交叉 (机制级)** | Q/R 侧与 K/V 侧 (或 K 与 V 之间、衰减 w) 来自不同输入流, 交叉发生在 WKV 递归/门控内部 | 交叉算子 = WKV 本身 |
| **② RWKV 充当跨流交互算子 (替代 cross-attention)** | 无显式 cross-attention, RWKV 块 (time/channel-mix、扫描) 本身完成跨流交互或融合 | 交互算子 = RWKV 块, 无二次型 CA |
| **③ 混合范式 (RWKV + 显式交叉算子并存)** | 标准/变体二次型 cross-attention、互注意力或调制模块与 RWKV 分工嵌套 | RWKV 与交叉算子各司其职 |
| **④ 特征级交叉 (非注意力)** | 交叉经通道统计、门控混合、损失对齐、证据融合、生成式监督完成 | 既非 CA, 也非以 RWKV 为交互算子 |
| **对照组** | 刻意不用交叉的 RWKV 多模态工作, 及常被引的"第一"基础设施 | 反向参照 |

---

## 类型 ①: WKV 算子内部跨流交叉 (机制级)

### 1. CrossWKV ⭐ (无 first 自称, 零引用)

**[Cross-attention for State-based model RWKV-7](https://arxiv.org/abs/2504.14260)** | 2025, arXiv:2504.14260 (2025-04-19, 仅 v1; Liu Xiao, Li Zhiyuan, Lin Yueyu) | 无 IF / 未入 CCF | 流关系: **跨模态, 文本→图像**

- **定位声明**: 摘要自称 "a **novel cross-attention mechanism** for the state-based RWKV-7 model"; **全文无 "first" 优先权声明**——"第一个把 cross-attention 形式化嵌入 WKV 递归"是外部归纳, 引用时只能引 "novel"。
- **设计** (全部核实属实): 文本嵌入 (CLIP 输出) 为 query 侧, 图像特征为序列主体; 文本 padding 到图像序列长度后生成 R, K/V 来自图像特征; **time-shift (δ = shift(x) − x) 保留且作用于图像 token 流**; 衰减 w 用 LoRA 分解 (decay/lr rank-64, value 16, gate 128); K 侧 L2Norm; 复杂度 O(T·N·H)。
- **理论**: "non-diagonal, input-dependent transition matrix... beyond the **TC0**"; "state-tracking tasks like **S5 permutation modeling**"。
- **评价**: DIR-7 (RWKV-7 扩散) ImageNet 256×256 FID 2.88、CLIP 0.33 属实, 但措辞是 "**matching** state-of-the-art / **rivaling** DiT-XL/2", 且自注 "preliminary results should be interpreted with caution"。未同行评审, Semantic Scholar 引用 0 (2026-11-02)。

### 2. Video RWKV / LCR ⭐ (06 号文档来源)

**[Video RWKV: Video Action Recognition Based RWKV](https://arxiv.org/abs/2411.05636)** | 2024, arXiv:2411.05636 (2024-11-08; Zhuowen Yin, Chengru Li, Xingbo Dong) | 无 IF / 未入 CCF | 流关系: **跨时刻, 当前帧边缘→历史特征**

- **定位声明** (逐字核实): "this work **introduces RWKV to the video domain in a novel way**"; "set a new benchmark in video understanding"。模型名 **LCR**。
- **Cross RWKV gate** (核实属实): 当前帧边缘信息与过去帧特征经门交互, 跨帧全局聚合; 边缘信息充当 LSTM 遗忘门 ("The edge information is passed through the forgetting gate of LSTM... to reduce the effect of background noise")。配套增强 LSTM、tube masking。
- **评价**: 线性复杂度, 代码公开; 预印本 (引用 5), "new benchmark" 自评未兑现。**本仓库 `06-CrossRWKVGate-Q重构设计.md` 的任务 Q 机制 (式 2–8) 即移植自此**。

### 3. OTN-RWKV / Asymmetric RWKV Fusion ⭐ (本类型已发表最高层级)

**[RGB-Event based Pedestrian Attribute Recognition: A Benchmark Dataset and An Asymmetric RWKV Fusion Framework](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_RGB-Event_based_Pedestrian_Attribute_Recognition_A_Benchmark_Dataset_and_An_CVPR_2026_paper.html)** | 2026, CVPR, pp.32745–32755 (Xiao Wang 等; arXiv:2504.10018) | 无传统 IF | **CCF A** | 流关系: **跨模态, RGB↔Event**

- **定位**: 双模态各自 Vision-RWKV 编码 + OTN-RWKV 非对称融合; EventPAR 100K 数据集 (自称 "The first benchmark dataset for RGB-Event PAR")。
- **实现细节 (2026-11-02 核实 arXiv 全文 + 官方代码 Event-AHU/OpenPAR::otn_rwkv.py, MIT, 含 RWKV6-B 预训练权重)**: 融合在两编码器之后的**最后一层、单一尺度** (识别任务, 全局池化头); **WKV-6 算子内部跨流——R、K 投影来自 RGB 路, V 与衰减 w 来自 event 路**, 其中 **w = time_decay (参数) + tanh(MLP(event 特征))——输入条件化衰减**; 输出 `σ(R)⊙K⊙WKV + event 残差`; WKV-6 CUDA 核 (T_MAX=256)。**论文-代码出入**: 式 (5) 的 KNP top-K 相似度过滤 (阈值 0.75) 未随代码发布; 式 (6) 的 LN 链在代码中简化。

### 4. EvRWKV (实现已由本仓库代码核实)

**[EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement](https://arxiv.org/abs/2507.03184)** | 2026, IEEE TCSVT, vol.36, no.7, pp.10201–10216 (DOI: 10.1109/TCSVT.2026.3672491) | IF 8.4 | **CCF B** | 流关系: **跨模态, Image↔Event, 复原域**

- **定位**: 连续跨模态交互 (反对 early/late fusion 二分): Cross-RWKV Module 捕捉细粒度时间与跨模态依赖 + EISFE 频域去噪/空域对齐。SDE/SDSD PSNR +1.79/+1.85 dB, 下游 mIoU +35.44% (均核实属实)。
- **实现细节 (2026-11-02 核实本仓库 `reference_repos/EvRWKV/egllie/models/Cross_Restore_RWKV.py`)**: **WKV 内部 K/V 跨流交替扫描**——`v_image = RUN_CUDA(..., k_event, v_image)`, `v_event = RUN_CUDA(..., k_image, v_event)` (**K 取对侧流, V 取本路流**), 偶次直扫/奇次转置扫 (双向); 融合权重自适应混合两路后过本路 `σ(R)` 门输出; ChannelMix 各路独立无交叉。**本仓库借用的 OmniShift 即出自该文件**。→ 归类型① (此前按摘要曾存疑归混合类, 已修正)。

### 5. X2Bone (与 receptance 模式同型)

**[X2Bone: Reconstructing 3D Bone Structures from 2D Biplanar X-Rays via Cross-RWKV](https://papers.miccai.org/miccai-2026/1163-Paper0798.html)** | 2026, MICCAI, LNCS 16889 (open access) | 无传统 IF | **CCF B** (交叉/综合/新兴) | 流关系: **跨视角, AP↔LAT**

- **定位**: "Cross-RWKV module... long-range, bidirectional cross-view interaction with linear complexity"; 配套 3D Expand 视角相关深度调制。CTPelvic1K 最优 (DSC 87.96%, 11.02M 参数)。
- **实现细节 (2026-11-02 核实官方代码 github.com/pzhhhhh2263/X2Bone)**: 双流共享权重 CNN 编码器 (RWKV 只在交叉块), 每尺度两个对称 `CrossBlock(x_a, x_b)`/`(x_b, x_a)`; **双输入单输出**——R 投影来自一侧视角、K/V 来自对侧 (代码: R 取 x_a、K/V 取 x_b; rebuttal 文字方向相反, 以代码为准); 门控 **`sigmoid(r) ⊙ WKV`**——无带符号 R/逐头 LN/SiLU(G); 实现文件 `Restore_RWKV_cross_r_new.py` 即 Restore-RWKV 的"R 改异源"改造, **与本仓库同源路线**。Cross-Shift (扫描前混入对侧 token 的局部交换) 据 rebuttal 存在, 发布代码未见。**与本仓库对照**: 与旧 `receptance` 模式 (`sigmoid(q)·wkv`, q 异源) 结构同型; 比现行 `cross_rwkv` (LCR 式 R/G 双门) 浅一层; 无任务分解查询。

### 6. RWKV-IR (Cross-Bi-WKV, 类型①变体)

**[RWKV-IR](https://arxiv.org/abs/2412.03814)** | 2024, arXiv 预印本 | 无 IF / 未入 CCF | 流关系: **跨方向 (同一输入的两个扫描序)**

- "Cross-Bi-WKV module that merges two Bi-WKV modules with different scanning orders for balanced horizontal and vertical attention"——WKV 内部跨**扫描方向**交互, 非跨输入流, 归本类型变体。

### 7. WuNeng (头间交叉变体)

**[WuNeng: Hybrid State with Attention](https://arxiv.org/abs/2504.19191)** | 2025, arXiv 预印本 (与 CrossWKV 同一作 Liu Xiao) | 无 IF / 未入 CCF | 流关系: **头间 (MHA 注意力头 × RWKV-7 状态头)**

- MHA 头与 RWKV-7 状态头做 cross-head 交互 (拼接/加性调制/门控融合); 跨流发生在**头间**而非序列间, 归本类型变体。

### 8. CrossMamba (同构前身, 非 RWKV 载体, 标注参照)

**[CrossMamba](https://arxiv.org/abs/2409.04803)** | 2024, arXiv (目标声音提取) | 无 IF / 未入 CCF | 流关系: **跨模态, 文本线索→音频**

- "utilize the clue to generate the query and the audio mixture to derive the key and value"——**Mamba 选择性状态空间递归内实现 cross-attention**, 是 CrossWKV 的直接思想前身 (CrossWKV 引用了它)。非 RWKV 载体, 仅作机制族参照。

### 类型①设计空间小结: 各工作从"另一路"取什么

| 工作 | 交叉流关系 | R/G 来源 | K 来源 | V 来源 | 衰减 w | 门控 |
|---|---|---|---|---|---|---|
| CrossWKV | 文本→图像 | 文本 (R) | 图像 | 图像 | LoRA(图像) 自适应 | LoRA gate |
| LCR Cross gate | 边缘→历史 | 边缘+LSTM 态 (R,G) | 历史 | 历史 | 边缘充当 LSTM 遗忘门 | R⊙ + LN + SiLU(G) |
| OTN-RWKV | RGB↔Event | RGB (R,K) | RGB | **Event** | **参数 + tanh MLP(event) 输入条件化** | σ(R), K 外乘 |
| EvRWKV | Image↔Event | 本路 (R) | **对侧** | 本路 | 本路块参数 | σ(R) + 自适应融合权重 |
| X2Bone | AP↔LAT | 本视角 (R) | **对侧** | **对侧** | 块参数 | σ(R) |
| *(本仓库 QRB receptance)* | *统计源* | *查询路* | *KV 路* | *KV 路* | *头参数* | *σ(q)* |
| *(本仓库 QRB cross_rwkv)* | *统计源* | *查询路 (R,G)* | *KV 路* | *KV 路* | *头参数* | *R⊙ + 逐头LN + SiLU(G)* |

(斜体行为本仓库现状参照, 非文献条目。可见已发表工作覆盖了 {R,K,V,w} 异源的不同子集; **"w 由查询路条件化"仅 OTN 一例且为跨模态识别任务, delta-rule 式演化全部空缺**。)

---

## 类型 ②: RWKV 充当跨流交互算子 (替代 cross-attention)

### 9. RrmTrack ⭐

**[RWKV-Inspired Multi-Modal Relation Modeling for Vision-Language Tracking](https://ieeexplore.ieee.org/document/11417442)** | 2026, IEEE TMM, vol.28, pp.6892–6901 (DOI: 10.1109/TMM.2026.3668700) | IF 9.7 | **CCF B** | 流关系: **跨模态, 语言↔图像**

- **定位声明**: "a **novel modality interaction method** specific to vision-language object tracking based on RWKV" (摘要无 "first" 自称)。
- **分工** (摘要逐字): "**time mixing module to model the relationship between language and image features and a channel mixing module to facilitate information interaction between images**"——跨模态走 time-mix、模态内走 channel-mix。
- 配套: Siamese + One-stream; 信息恢复模块; 并行训练 + RNN 推理; TNL2k/LaSOT/OTB-Lang/LaSOText/MGIT 五基准 "state-of-the-art results and speed"。

### 10. RWKV-VG

**[RWKV-VG: visual grounding with RWKV-driven encoder-decoder framework](https://link.springer.com/article/10.1007/s00530-025-01720-w)** | 2025, Multimedia Systems, vol.31, art.124 | IF 3.1 (2024 JCR; 2025=2.8) | CCF C | 流关系: **跨模态, 视觉↔语言 (纯 RWKV, 无显式 CA)**

- **定位声明** (逐字核实): "**the first visual grounding framework entirely built on the RWKV architecture**"。RWKV 视觉/语言编码器 + RWKV 视觉语言解码器 (intra-modal + cross-modal reasoning), 可学习 [REG] token。
- **评价**: RefCOCO 系列评测; 摘要**自述 QRNet/CLIP-VG 等仍优于它** (预训练数据 15M vs 400M)。

### 11. IESS-FusionNet (显式宣称替代 CA)

**[IESS-FusionNet](https://www.mdpi.com/2306-5354/13/1/57)** (EEG-EMG 婴儿癫痫痉挛检测) | 2026, Bioengineering 13(1):57 (MDPI) | IF 4.4 (2023 JCR; 2024 未核) | 不在 CCF | 流关系: **跨模态, EEG↔EMG**

- **Cross Time-Mixing 模块 "built upon the linear recurrent attention of the RWKV architecture... replacing the quadratic cross-attention"**——算子替代路线继 RrmTrack 后第二个**显式宣称"用 RWKV 替代 cross-attention"**的实例。

### 12. OccRWKV

**[OccRWKV](https://arxiv.org/abs/2409.19987)** (3D 语义占据预测) | 2025, ICRA | 无传统 IF | **CCF B** | 流关系: **跨分支 (语义/几何双分支)**

- Sem-RWKV + Geo-RWKV 双分支分别建模, **BEV-RWKV 块充当跨分支融合算子** ("用 RWKV 做融合"而非给 RWKV 配 CA)。SemanticKITTI 25.1 mIoU @ 22.2 FPS。

### 13. SleepRWKVNet

**[SleepRWKVNet](https://doi.org/10.1016/j.bspc.2025.108713)** (多模态睡眠分期) | 2026, Biomedical Signal Processing and Control, vol.113, 108713 | IF 5.7 (官网现值) | 不在 CCF | 流关系: **跨模态, EEG/EOG/EMG**

- "bidirectional interactive RWKV" 整合三模态特征 (经引用方 CSPT-SleepNet 转述); 已被引 6 次。⚠️ 内部结构未逐字核实, 归②存疑 (亦可能为①)。

### 14. MMCGR

**[MMCGR](https://www.sciencedirect.com/science/article/abs/pii/S0925231226008210)** (对话情感识别) | 2026, Neurocomputing, vol.682, 133424 | IF 6.5 | CCF C | 流关系: **跨模态, 文本↔音频**

- RWKV 计算模态内/间表征 + 级联门控融合 (融合算子非显式 CA)。⚠️ ScienceDirect 直抓被拒, 经 WebSearch+ResearchGate 交叉核实, 归②存疑。

### 15. RWKV-Enhanced IR-Vis Fusion

**[RWKV-Enhanced Infrared and Visible Image Fusion: Capturing Long-Range Dependencies with Cross-Modality Correlations](https://ieeexplore.ieee.org/document/11605729)** | 2026, PRMVAI (IEEE Int. Conf. on Pattern Recognition, Machine Vision and AI, 第 4 届) | 无 IF | 不在 CCF | 流关系: **跨模态, 红外↔可见光**

- 摘要逐字: "a **spatial-channel dual-hybrid RWKV-Fuser block**... simultaneously captures long-range dependencies and **cross-modal channel correlations**"——RWKV 块本身做跨模态融合。⚠️ 归②/④ 的边界待全文核 (无 "替代 CA" 表述, 亦无显式 CA)。

---

## 类型 ③: 混合范式 (RWKV + 显式交叉算子并存)

### 16. CMRWNet

**[CMRWNet: Cross-Modal RWKV Network With Window-Guided Alignment for High-Fidelity Pansharpening](https://ieeexplore.ieee.org/document/11614160)** | 2026, IEEE JSTARS, vol.19, pp.24069–24084 | IF 5.3 (2024 JCR; 2025=6.3) | **不在 CCF 目录** | 流关系: **跨模态, MS↔PAN**

- 窗口级双向 cross-attention (subquadratic) 做精对齐 + 堆叠 PAN-RWKV ("row and columnwise parallel prefix-decay scans" + receptance gating) 做线性全局混合——CA 局部化与 RWKV 全局化的分工范式。IKONOS/Pléiades/WV-3 SOTA + QuickBird 跨传感器。

### 17. Surv-RWKV

**[Surv-RWKV](https://www.sciencedirect.com/science/article/abs/pii/S0957417426004197)** (生存分析) | 2026, ESWA, vol.312, 131506 | IF 7.5 (2024 JCR; 2025=9.4) | CCF C | 流关系: **跨模态 (基因/病理)**

- RWKV 浅层融合 (空间-通道混合) + **cross-RWKV 深度交互 ("iterative cross-attention mechanisms")** + 最优传输对齐; 五个 TCGA 队列 (SOTA 逐字措辞因付费墙未取得)。

### 18. ChangeRWKV (类型③中的 A 类)

**[Beyond Quadratic: Linear-Time Change Detection with RWKV](https://arxiv.org/abs/2603.19606)** | 2026, **AAAI**, Vol.40(14), pp.11811–11819 (DOI: 10.1609/aaai.v40i14.38167, Crossref 已核) | 无传统 IF | **CCF A** | 流关系: **跨时相 (双时相遥感)**

- 层级 RWKV 编码器 + **Cross CBAM** 跨时相互注意力 ("channel attention weights are computed from one temporal feature and applied to the other, and vice-versa")。**cross-attention×RWKV 进入 CCF A 的第一个实例**——但为标准二次型挂 RWKV 骨干, 不动 WKV 本体。LEVIR-CD 85.46% IoU。

### 19. TemLo

**[TemLo: Temporal-Local Synergy Enhanced RWKV for Audio-Visual Segmentation](https://ieeexplore.ieee.org/document/11523565)** | 2026, IEEE TMM, vol.28, pp.9804–9815 | IF 9.7 | **CCF B** | 流关系: **跨模态, 音频↔视觉**

- 模态专属 RWKV 编码器双流 + **Acoustic Modulation Pyramid** 结构化音视频交互 + Vision-guided Modality Injector (摘要: "To fully enable cross-modal interaction and integration, we further design the Acoustic Modulation Pyramid...")。代码公开 (github.com/c1ircle/TemLo)。**音视频域此前误判为空白, 本轮补上**。⚠️ 调制金字塔非标准 CA, 归③/②存疑。

### 20. Learning to Prefer (本主题最高 IF)

**[Learning to Prefer: Reward-Driven Infrared-Visible Fusion via RWKV](https://doi.org/10.1016/j.inffus.2026.104685)** | 2026, Information Fusion, vol.138, 104685 (Jialing Huang 等, 通讯 Xiaotong Tu, Restore-RWKV 团队) | **IF 14.7 (2024 JCR)** | **不在 CCF 目录** | 流关系: **跨模态, 红外↔可见光, 复原+融合统一**

- 奖励驱动复原+融合框架, 含 **Cross-RWKV 模块**与跨模态差异感知融合。发现路径: EvRWKV、RICEFuse 引用链 (两个独立代理各自命中)。⚠️ 摘要付费墙, Cross-RWKV 内部结构 (①或③) 未能核实。

### 21. CROWNFuse

**[CROWNFuse](https://www.sciencedirect.com/science/article/abs/pii/S0925231226022745)** | 2026, Neurocomputing, 134876 | IF 6.5 | CCF C | 流关系: **跨模态, 红外↔可见光**

- "Recurrent cross-modal memory interaction with edge-frequency recalibration"——RWKV-based Hierarchical Cross-Modal Fusion。发现路径: RICEFuse 引用链。⚠️ 内部结构未核, 归③存疑。

### 22. CRWKV (类型③中的 A 类)

**[Multi-View Learning with Context-Guided Receptance for Image Denoising](https://doi.org/10.24963/ijcai.2025/86)** | 2025, IJCAI | 无传统 IF | **CCF A** | 流关系: **跨特征视图 (空域+FMix 频域)**

- Context-guided RWKV, 空域与频域表征经多视图学习融合, BiWKV 全像素交互; "多视图"是特征视图非模态。

### 23. C2F-VRWKV

**[C2F-VRWKV](https://doi.org/10.1109/TGRS.2026.3693340)** (舰船检测) | 2026, IEEE TGRS | IF 8.6 (2024 JCR) | **CCF B** (交叉/综合/新兴) | 流关系: **跨频 (同一图像的频率带)**

- Vision-RWKV 骨干 + "Cross-Frequency Fusion Module... frequency-guided cross-attention strategy"。

### 24. MTRNet (中文期刊)

**[MTRNet](https://doi.org/10.3788/lop251946)** | 2025, 《激光与光电子学进展》 | 无 IF/CCF (中文期刊) | 流关系: **跨时相 (双时相变化检测)**

- "以视觉 RWKV 作为主体编码器...运用交叉注意力实现跨通道与空间的联合注意力建模"。

### 25. PestVL-Net

**[PestVL-Net](https://doi.org/10.1109/CVPRF71909.2026.00847)** (多模态虫情识别) | 2026, **CVPR FINDINGS Track (CVPRF)**, DOI 已核 | — | 同 CVPR 系 | 流关系: **视觉↔语言**

- RWKV 视觉通路 (显著性引导自适应窗口) + MLLM 文本先验, 经**标准显式 cross-attention** 融合 (Q=视觉, K/V=文本)。CVPR 2026 新设 Findings track, 引用时与主会区分。

### 26. LADY

**[LADY](https://arxiv.org/abs/2512.15038)** (自动驾驶效率) | 2025-12, arXiv 预印本 | 无 IF / 未入 CCF | 流关系: **跨模态, camera↔LiDAR**

- RWKV-7 融合多帧相机+激光雷达 + **lightweight linear cross-attention (LICA)**。⚠️ 摘要不含 "RWKV" 字样, 全文检索必漏 (社区线索定位)。

### 27. ITC-RWKV

**[ITC-RWKV](https://arxiv.org/abs/2510.21479)** (组织病理亚型分型) | 2025-10, arXiv 预印本 | 流关系: **跨尺度/跨粒度, 细胞↔组织**

- RWKV 式 receptance-weighted KV 聚合 (线性复杂度) + "bidirectional tissue-cell interaction module to enable **mutual attention** between localized cellular cues and their surrounding tissue environment"。

### 28. DPC-QA Net

**[DPC-QA Net](https://arxiv.org/abs/2509.15802)** (病理图像无参考质量评估) | 2025, arXiv 预印本 | 流关系: **跨流 (感知质量↔细胞质量)**

- Aggr-RWKV 细胞流 + "**Cross-attention fusion** and multi-term losses align perceptual and cellular cues"。

### 29. MSFusion

**[MSFusion](https://www.mdpi.com/2079-6374/16/8/423)** (多模态医学图像融合) | 2026-08, Biosensors 16(8):423 (MDPI) | IF 5.6 (2024 JCR) | 不在 CCF | 流关系: **跨模态医学影像**

- VRWKV/Bi-WKV 块 + MSCMCF "bidirectional cross-modal attention"。

### 30. SGF-RWKV

**[SGF-RWKV](https://doi.org/10.1109/JBHI.2026.3739849)** (统一医学去噪) | 2026, IEEE JBHI | IF 6.8 | **CCF C** | 流关系: **语言→视觉引导**

- RWKV 统一去噪框架内嵌 "a semantic guided module leverages CLIP based language embeddings to generate task-aware attention maps"。

### 31. RWPNet

**[RWPNet](https://doi.org/10.1109/JSTARS.2026.3734262)** (全色锐化, 与 CMRWNet 同赛道) | 2026, IEEE JSTARS | IF 5.3 | 不在 CCF | 流关系: **跨模态, MS↔PAN**

- SS-RWKV (spatial-spectral) + "dual-branch cross-fusion module... mutual feature modulation"。

### 32. MMSegRWKV

**[MMSegRWKV](https://doi.org/10.1109/JIOT.2026.3661927)** (多模态 MRI 分割) | 2026, IEEE IoT Journal | IF 9.6 (2024 JCR) | 不在 CCF | 流关系: **跨模态 MRI (T1/T2/FLAIR)**

- U 形 RWKV + "dual-view weighted key–value (DV-WKV), which factorizes dependencies along spatial and temporal dimensions"。⚠️ DV-WKV 是否 WKV 内部跨流未核, 归③存疑。

### 33. DME-RWKV

**[DME-RWKV](https://doi.org/10.3390/bioengineering13010012)** (抗 VEGF 疗效预测) | 2025, Bioengineering 13(1):12 (MDPI) | IF 4.4 (2023) | 不在 CCF | 流关系: **跨模态, OCT↔UWF**

- RWKV 整合双模态 + Causal Attention Learning + 课程学习。

### 34. PMSRWKV-GCN

**[PMSRWKV-GCN](https://www.nature.com/articles/s41598-026-41091-4)** (多变量时序预测) | 2026, Scientific Reports | IF 3.9 (2024 JCR) | 不在 CCF | 流关系: **跨变量 (时序维↔变量维分工)**

- RWKV 管时间维 (通道独立预训练) + 注意力邻接 GCN 管跨变量维——与 RWKV-CVM (类型④) 门控混合并行的另一条跨变量路线。

### 35. GoldFinch

**[GoldFinch](https://arxiv.org/abs/2407.12077)** | 2024, arXiv 预印本 | 流关系: **跨算子 (transformer 消费 RWKV 状态)**

- GOLD transformer 对 Finch (RWKV-6) 的 KV-cache 做 cross-attention (KV-cache 压缩导向)。

---

## 类型 ④: 特征级交叉 (非注意力算子)

### 36. RWKV-CVM

**[RWKV-CVM: Gated Cross-Variate Mixing for Multivariate Power Load Forecasting](https://www.mdpi.com/2673-4826/7/2/58)** | 2026, Electricity (MDPI), vol.7 no.2 | IF 2.2 | 不在 CCF | 流关系: **跨变量**

- "gated, row-stochastic mixing matrix... adding only **65 trainable parameters**" (逐字核实)。

### 37. RICEFuse

**[RICEFuse](https://www.sciencedirect.com/science/article/abs/pii/S1350449526001945)** | 2026, Infrared Physics & Technology, vol.156 | IF 3.4 (2024 JCR; 官网 3.3) | 不在 CCF | 流关系: **跨模态, 红外↔可见光**

- DFCEM 互补增强 + AMT-Shift + 余弦相似度跨模态对齐损失 (均逐字核实)。

### 38. CMGN

**[CMGN: Text GNN and RWKV MLP-mixer combined with cross-feature fusion for fake news detection](https://www.sciencedirect.com/science/article/pii/S0925231225004837)** | 2025, Neurocomputing, vol.633, 129811 | IF 6.5 | CCF C | 流关系: **跨特征 (⚠️ 文本域: 假新闻检测, 非图像融合)**

- Text GNN + RWKV MLP-mixer + cross-feature fusion (标题核实; 摘要原文不可得)。

### 39. 渐进式 RWKV 多模态医学融合

**[A progressive RWKV network for multimodal medical image fusion](https://doi.org/10.1016/j.asoc.2026.116704)** | 2026, Applied Soft Computing | IF 6.6 (2024 JCR) | 不在 CCF | 流关系: **跨模态医学影像**

- 自称首个 RWKV 医学图像融合框架, RWKV 线性复杂度长程建模。⚠️ 无 CA 证据, 归④存疑。

### 40. TE-MSTAD

**[TE-MSTAD](https://arxiv.org/abs/2601.11951)** (WSN 异常检测) | 2026, arXiv 预印本 | 流关系: **跨节点**

- "a Cross modal Feature Extraction (CFE) module... building upon the RWKV model" + GNN + 时频双支融合。

### 41. RWKV-VIO

**[RWKV-VIO](https://www.mdpi.com/1424-8220/25/18/5737)** (视觉惯性里程计) | 2025, Sensors 25(18):5737 | 不在 CCF | 流关系: **跨模态, 视觉↔IMU**

- 双并行编码器 + RWKV 时序建模, 端到端 VIO; 摘要未见显式 CA。

### 42. Compass

**[Compass](https://arxiv.org/abs/2608.03559)** (缺失模态裂缝分割) | 2026, arXiv 预印本 | 流关系: **跨模态证据融合**

- Needle RWKV 骨干 + 多模态证据融合 (ETPF/Dempster-Shafer, 非注意力算子)。

### 43. Geometry-aware RWKV 光场 SR

**[Geometry-Aware RWKV for Heterogeneous Light Field Spatial Super-Resolution](https://doi.org/10.1109/LSP.2025.3555445)** | 2025, IEEE Signal Processing Letters | IF 未核 | 不在 CCF (AI/交叉页已查) | 流关系: **跨视角 (空间-角点相干性)**

- 异构融合 (高分辨率 2D 相机 + 光场, 蒸馏迁移) + geometry-aware RWKV。⚠️ 无显式 CA 证据。

### 44. P-RWKV / PointER

**[P-RWKV/PointER](https://arxiv.org/abs/2606.10395)** (点云自监督) | 2026, arXiv 预印本 | 流关系: **"cross-modality" = 3D→2D 渲染生成式监督, 非注意力**

- 全文核实: 其 "cross-modality" 是 Photograph Module 把 3D 特征投影到 2D 渲染图做生成式监督。**引用时勿混淆为 cross-attention**。

---

## 对照组: 刻意不用交叉的 RWKV 多模态 + "第一"基础设施

- **[RWKV-CLIP: A Robust Vision-Language Representation Learner](https://aclanthology.org/2024.emnlp-main.276/)** | 2024, **EMNLP 2024 主会** (2024.emnlp-main.276; arXiv:2406.06973) | CCF B | 摘要逐字自称 "**the first RWKV-driven vision-language representation learning model**"。双塔 + 对比学习, 塔间无 cross-attention——"不做 CA 的 RWKV 多模态"的对照组原型。
- **[ModRWKV](https://arxiv.org/abs/2505.14505)** | 2025, EMNLP 主会 | CCF B | 多模态 RWKV LLM 旗舰, 模态融合用 concat adapter, 无 CA——RWKV-CLIP 的 LLM 版对照组。
- **[VisualRWKV](https://arxiv.org/abs/2406.13362)** | 2024, arXiv | RWKV 多模态 LLM 主路线, 明确不用 CA (对比 LLaVA-1.5 提速 4×)。
- **Otter** | 2026, AAAI Oral | 增强型 RWKV 小样本动作识别, 全文无 CA (原型匹配用范数距离)。Video RWKV 动作识别系列的同期延续, 厘清归属用。
- **MG-RWKV** | 2026, ECCV | CCF B | 音视频伪造定位; 全文核实其 "cross-attention feature pyramid" 属于被引的 UMMAFormer 而非自身。
- **VRWKV-Editor** | 2025, arXiv:2509.25998 | 视频编辑; 原文明确 "adding more cross-attentions or temporal attentions only increased the complexity"。
- **Delta-WKV** | 2025, MICCAI (arXiv:2502.20852) | CCF B | 医学 in-context 超分, 用 delta 规则做推理期动态权重——**RWKV 路线绕开 cross-attention 处理参考引导的实例, 与 delta-rule 论述直接相关的重要对照**。
- 其他低相关 (跨尺度/跨阶段/通道独立, 非本文义): HART、RWKV-X、MRT、MD-RWKV-UNet、SCRWKV、MFC-RFNet、FS-RWKV、RWKV-UNet、GDSR、MARCS、MDS-RWKV、Otter 前述等, 略。

---

## 修订后的整体图景 (按类型)

1. **类型① 是主战场且已进入顶会**: OTN-RWKV (CVPR, CCF A) 占住 "WKV 内部跨流交叉 + 输入条件化衰减"; EvRWKV (TCSVT, B)、X2Bone (MICCAI, B) 各占一种 {R,K,V,w} 异源组合 (K 异源 / R 异源)。但两篇机制预印本 (CrossWKV 零引用且无 first 自称; LCR 引用 5) 说明 **delta-rule 级深交叉与门控级深交叉 (LCR 式 R/G 双门) 仍无人认领**。
2. **类型② 从单点变成多实例**: RrmTrack (TMM-B)、IESS-FusionNet (显式宣称替代)、OccRWKV (ICRA-B)、SleepRWKVNet、RWKV-VG、MMCGR、RWKV-Enhanced——路线可行但**尚无 A 类**。
3. **类型③ 数量最多 (20 篇)**: A 类两篇 (ChangeRWKV/AAAI、CRWKV/IJCAI) 均为标准二次型挂 RWKV 骨干; 医学影像与红外-可见光融合是该类型最拥挤的应用域。
4. **类型④ 与对照组**: 跨变量交互存在门控混合 (RWKV-CVM) 与注意力邻接 GCN (PMSRWKV-GCN) 两条并行路线; "刻意不用 CA" 的 RWKV 工作 (VisualRWKV/VRWKV-Editor/Otter 等) 构成反向参照——RWKV 与 CA 在多模态 LLM 侧是替代关系, 在视觉侧是结合关系。

**空白提示**: 把 CrossWKV 的 delta-rule 式跨序列状态演化正式用于视觉任务并发表于 CCF A 的工作**仍然空缺**。

> ⚠️ 此结论随核实进展两次收窄, 现行口径: CCF A 已有 OTN-RWKV (WKV 内部跨流交叉 + 输入条件化衰减, CVPR) 与 ChangeRWKV/CRWKV (二次型挂骨干, AAAI/IJCAI); **仍空缺且无人认领的四格**: ①delta-rule 式跨序列状态演化 (OTN 用 RWKV-6 衰减+调制, 非 delta rule; CrossWKV 零引用); ②同模态/同视角**统计源**流的交叉; ③任务分解多查询 (N/L/M 式); ④复原/稠密预测域的 WKV 内部交叉 (EvRWKV 是 B 类且跨模态; OTN 是识别任务、单尺度、全局池化)。

## 确认的真空白 (多源枚举后仍无 CA×RWKV)

视频 VOS/复原/视频融合; 高光谱; SLAM/点云 (PointER 的 "cross-modality" 非 CA); **RWKV×Mamba 跨模态互操作**; **CrossWKV 前向引用经 S2+OpenAlex+WebSearch 三源确认仍为 0**。

## 方法学备注 (供后续复查)

- arXiv 全文 token 检索会漏两类: 名称合成词不含 "RWKV" 的 (ModRWKV、LADY 摘要无 RWKV 字样)——必须靠引文滚雪球与 WebSearch 补。
- Semantic Scholar 公网 API 限流严重 (429), 建议带 key 复跑; OpenAlex 更稳定。
- 两条**未能核实、暂勿引用**的线索: IEEE TGRS 2026 建筑变化检测 Vision-RWKV 论文 (ieeexplore 11456675, 被拦); OpenReview kmNqnwA4aV "Bridging Transformers and RWKV" (称做了 RWKV±cross-attention 消融, CAPTCHA 拦截)。
- CCF 第七版核实 (2026-11-02, ccf.org.cn 官网分类页): AI 类会议 A 类共 7 个——AAAI、NeurIPS、ACL、CVPR、ICCV、ICML、IJCAI (**网传"IJCAI 降 B"不实; ICLR 不在目录任何级别**); ECCV、EMNLP、ICRA 均 B (AI 类); TGRS=B、JBHI=C (交叉/综合/新兴); TMM=B、TCSVT=B、Multimedia Systems=C (图形学与多媒体); ESWA/Neurocomputing=C (AI); **不在目录**: JSTARS、Information Fusion、Applied Soft Computing、IEEE IoT-J、BSPC、IEEE SPL、Infrared Physics & Technology、Electricity、PRMVAI。CVPR 2026 新设 FINDINGS Track (CVPRF)。

## 附: 逐篇核实判定汇总 (2026-11-02, 最初 14 篇; 新编号见正文)

| 论文 | 新编号 | 存在性/venue/年份 | 机制声明 | 定位声明 ("first...") | IF | CCF |
|---|---|---|---|---|---|---|
| CrossWKV | ①-1 | ✅ arXiv:2504.14260, 2025-04-19 | ✅ 全部 | ⚠️ 只自称 "novel", 无 "first" | — | — |
| Video RWKV/LCR | ①-2 | ✅ arXiv:2411.05636, 2024-11-08 | ✅ 全部 | ✅ "introduces RWKV... in a novel way" | — | — |
| RrmTrack | ②-9 | ✅ TMM vol.28 已刊出 | ✅ time/channel 分工逐字 | ⚠️ 只自称 "novel" | 9.7 | B |
| Surv-RWKV | ③-17 | ✅ ESWA vol.312 | ✅ (SOTA 措辞未取得) | 无 first 自称 | 7.5 | C |
| RWKV-VG | ②-10 | ✅ MMS vol.31 | ✅ | ✅ "the first... entirely built on RWKV" 逐字 | 3.1 | C |
| CMRWNet | ③-16 | ✅ JSTARS vol.19 | ✅ | 无 first 自称 | 5.3 | 不在目录 |
| EvRWKV | ①-4 | ✅ TCSVT vol.36 | ✅ (+1.79/1.85 dB, +35.44% 属实; 本仓库代码再核: K/V 跨流) | 无 first 自称 | 8.4 | B |
| X2Bone | ①-5 | ✅ MICCAI LNCS 16889 | ✅ 全部 | 无 first 自称 | — | B |
| Asymmetric RWKV Fusion | ①-3 | ✅ CVPR 2026 pp.32745-32755 | ✅ (R/K←RGB, V+w←event; KNP 未随代码发布) | 无 first 自称 | — | A |
| RWKV-CLIP | 对照组 | ✅ EMNLP 2024 主会 | ⚠️ 双塔无 CA 属正文细节 | ✅ "the first RWKV-driven vision-language representation learning model" | — | B |
| RWKV-Enhanced IR-Vis | ②-15 | ✅ PRMVAI 第 4 届 (IEEE) | ✅ RWKV-Fuser 逐字 | — | — | 不在目录 |
| RICEFuse | ④-37 | ✅ IPT vol.156 | ✅ 逐字 | — | 3.4 | 不在目录 |
| CMGN | ④-38 | ✅ Neurocomputing vol.633 (假新闻检测) | ⚠️ 标题证实, 摘要不可得 | — | 6.5 | C |
| RWKV-CVM | ④-36 | ✅ Electricity vol.7 | ✅ 65 参数逐字 | — | 2.2 | 不在目录 |

**核实来源** (一手页面): arXiv abs/html (2504.14260, 2411.05636, 2507.03184, 2504.10018, 2406.06973, 2504.19191, 2409.04803, 2412.03814, 2510.21479, 2509.15802, 2512.15038, 2601.11951, 2606.10395, 2608.03559); Semantic Scholar / OpenAlex / Crossref API; IEEE Xplore (11417442, 11614160, 11605729); Springer / papers.miccai.org; ACL Anthology; CVF openaccess; MDPI; Nature; CCF 官网 ccf.org.cn/Academic_Evaluation; 本仓库 reference_repos/EvRWKV 与官方 X2Bone/OpenPAR 代码。IF 来源: 学会/出版社官网 + ablesci/impactfactor.cn/bioxbio。未决项均已用 ⚠️ 标注 (Learning to Prefer 与 CROWNFuse 付费墙; SleepRWKVNet/MMCGR/MMSegRWKV 内部结构未逐字核)。
