# 领域论文知识库 (Paper Knowledge Base)

> LLIE / LLVE / 一般图像恢复 方向的研究动向与可迁移创新点
> 建立：2026-09-16
> 调研范围：CVPR2026(135篇图像恢复)、ICCV2025(31篇)、NeurIPS2025(26篇)

---

## 目录

| 文件 | 内容 | 用途 |
|------|------|------|
| [trends_llie_llve_2026.md](trends_llie_llve_2026.md) | 领域研究动向趋势综述 | 了解最新技术方向、判断我们的位置 |
| [ideas_top_transferable.md](ideas_top_transferable.md) | 可迁移创新点 idea 池（S/A/B 分级）| 候选改进方向、论文创新点素材 |
| [foxtrot_analysis.md](foxtrot_analysis.md) | Foxtrot 实验结果、架构分析、棋盘格诊断 | 本项目的问题诊断与改进方案 |
| [**sota_survey_2026.md**](sota_survey_2026.md) | **5篇SOTA论文深度代码调研（23个技术点）** | **完整实现分析+Golf改进路线图** |
| [Golf-plan.md](../v6/Golf-plan.md) | Golf 版本设计（Foxtrot 修复版）| 下一版架构设计文档 |

---

## 调研方法说明

**原则：不望文生义，关键代码直接阅读**

1. **论文清单**：从 [m0rtzz.github.io/paper-notes](https://m0rtzz.github.io/paper-notes/CVPR2026/image_restoration/) 获取 CVPR2026/ICCV2025/NeurIPS2025 图像恢复论文全列表
2. **适配度筛选**：按"视频/多帧/低光/时序"标准筛选相关论文，1-5 分打分
3. **代码精读**：对高适配度论文，克隆其 GitHub 仓库，直接阅读核心模块源码
   - MobileIE: `/tmp/MobileIE/` (ICCV2025, 4K参数实时LLIE)
   - SSGformer: `/tmp/SSGformer/` (ICCV2025, Sobel+SVD频谱先验)
4. **交叉验证**：论文声称 vs 代码实现，不一致处以代码为准

---

## 核心结论速览

### 对我们的任务（LLVE）最重要的三个发现

1. **LLVE 是相对空白区**：CVPR2026/ICCV2025/NeurIPS2025 的图像恢复板块几乎没有专门的 LLVE 论文（视频类归入"视频理解"板块）。当前 SOTA 仍是 VSRELL(CVPR2026)、RetinexMCNet(ICCV2025) 等少数几篇

2. **频域先验 + 退化感知路由是 2026 主流**：SSGformer 的 Sobel(高频边缘)+SVD(低频退化) 双路先验驱动空间分组，与我们 TSDR 分解的"噪声=高频 / 光照=低频 / 运动=高频+方向"高度对应

3. **扩散模型虽热但不适配 LLVE**：推理延迟 10-50×、时序一致性未内置、与 SDSD 的 PSNR 导向不完全对齐

### 我们当前架构的关键问题（来自代码审计）

| 问题 | 严重度 | 对应 idea |
|------|:---:|------|
| F1/F3 编码器特征完全闲置（60% 算力浪费）| 🔴 | foxtrot_analysis.md §4.2-A |
| 全流程 H/2 处理，无高分辨率 skip | 🔴 | S3 高分辨率 warp |
| 三分支同监督同一 GT，功能冗余 | 🟡 | A2 因果解耦 |
| 时序一致性损失缺失 | 🟡 | A3 双向一致性 |
| TCA-RWKV 的 K/V 来自聚合上下文而非邻帧 | 🟡 | S1 频谱路由 |

---

## 代码仓库状态

| 论文 | 仓库 | 克隆状态 | 已读模块 |
|------|------|:---:|------|
| MobileIE | github.com/AVC2-UESTC/MobileIE | ✅ | MBRConv/FST/HDPA |
| SSGformer | github.com/jeongyh98/SSGformer | ✅ | Sobel+SVD光谱先验/FGA_C分组注意力 |
| **MODEM** | **BasicSR框架** | ✅ | Morton-Order扫描/DAFM退化调制/两阶段蒸馏 |
| **DGAF-VSR** | **Diffusers集成** | ✅ | 超分辨率域warp/zero-init/全层级dense residual |
| **PRE-Mamba** | **PointCept框架** | ✅ | 双流MSSM门控/STDF时序差分/Hilbert序列化 |
| **LASQ** | **NeurIPS2026** | ✅ | MCMC-Gamma分层量化/特征空间扩散/对抗训练 |
| **AFUNet** | **ICCV2025** | ✅ | W-MCA跨帧对齐/Deep Unfolding/μ域FFTLoss |

**最新补完**（2026-09-16）：MODEM/DGAF-VSR/PRE-Mamba/LASQ/AFUNet 5篇完整代码精读完成，详见 [sota_survey_2026.md](sota_survey_2026.md)

---

## 参考来源

- CVPR2026 图像恢复(135篇): https://m0rtzz.github.io/paper-notes/CVPR2026/image_restoration/
- ICCV2025 图像恢复(31篇): https://m0rtzz.github.io/paper-notes/ICCV2025/image_restoration/
- NeurIPS2025 图像恢复(26篇): https://m0rtzz.github.io/paper-notes/NeurIPS2025/image_restoration/

---

## 维护说明

- 本知识库随研究进展更新，新增论文/idea 请标注日期
- 任何 idea 落地前必须先做 10-epoch 快速验证（吸取 Flight11 教训：未经小实验直接全量训练导致 47h GPU 浪费）
- 代码精读结论优先于论文摘要描述
