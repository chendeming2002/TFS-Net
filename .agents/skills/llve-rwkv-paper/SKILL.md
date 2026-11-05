---
name: llve-rwkv-paper
description: 低光视频/图像增强（LLVE/LLIE）论文写作与设计助手，面向 RWKV 架构方向（TSD-Net 项目）。当用户要求撰写、修改、润色论文任何章节（Abstract / Introduction / Related Work / Method / Experiments / Conclusion）、设计实验（数据集、指标、基线、消融、复杂度表）、核对引用文献、或讨论 TSDR 三源分解 / SGRF / QRB / 部件命名时使用。用户只说"写引言""改方法章""补实验""查引用""论文结构怎么搭"也应触发。
---

# LLVE/LLIE-RWKV 论文写作助手

面向 TSD-Net（低光视频增强，RWKV 架构）的论文写作与设计知识库。核心资产：

- **核心叙事**：TSDR 三源分解（illumination / noise / motion）+ 五部件架构 + RWKV 线性复杂度设计
- **参考文献库**：19 篇主要引用论文的精读笔记与引用核对结论（含已发现的旧稿引用错误）
- **逐章写作手册**：Abstract–Conclusion 的段落逻辑与措辞模板
- **实验设计思路**：数据集、指标、基线、消融、复杂度表、当前最佳数字

## 先读哪个文件

| 用户意图 | 读这个 |
|---|---|
| 讨论架构 / 叙事 / 命名 / 创新点 | `references/narrative.md` |
| 写 §2 Related Work、核对某条引用 | `references/papers-rwkv.md`、`references/papers-llve.md`、`references/papers-noise.md` |
| 写 §1 / §3 / §4 / Abstract 等具体章节 | `references/sections-playbook.md` |
| 设计实验 / 消融 / 复杂度表 / 图 | `references/experiments-design.md` |

## 写作工作流（默认顺序）

1. 读 `narrative.md` 校准术语与架构口径（命名以 `docs/v8/08-命名对照.md` 为唯一来源）。
2. 按 `sections-playbook.md` 的章节顺序草稿：**先 §3 部件（§3.3–§3.7）→ 再 §3.1–§3.2 框架与总体 → 再 §1 → 再 §2 → 最后 §4**。这与 v7 草稿的写作顺序一致：部件细节先定，框架与动机后写。
3. 每条引用回 `papers-*.md` 核对；标 `[待核实]` 的条目不得进入定稿。
4. 所有数字（PSNR/SSIM/LPIPS/参数量/FLOPs/延迟）必须来自 `docs/v8/13-实验结果汇总.md` 或训练日志，不得凭记忆填写。

## 红线（违反即返工）

1. **不得把"四方向 2D 扫描"归给 Vision-RWKV**。Vision-RWKV 的四方向是 Q-Shift（token 混合，发生在注意力前），其 Bi-WKV 仍是一维展平双向扫描；四方向 2D 扫描（SpatialWKV2D）出自 RSRWKV。引用 Vision-RWKV 时只说 "bidirectional WKV scan"。
2. **"首个 RWKV 低光视频增强"的表述必须带范围并核实**。Restore-RWKV 自称 "first RWKV-based model for image restoration"（医学图像，非低光视频）；Video RWKV/LCR 已把 RWKV 用于视频（动作识别）。TSD-Net 用 "the first RWKV-based model for low-light video enhancement"，并在 §2.3 交代 Restore-RWKV 的优先权范围。
3. **观测模型是 sRGB 域**。引用 RAW 域噪声文献（Foi/ELD/PNNP）时必须用一阶近似措辞，模板见 `papers-noise.md`。
4. **命名以 `docs/v8/08-命名对照.md` 为唯一口径**：论文名 SGRF ↔ 代码 `V7RFusion`；`QueryRWKVBlock` 是功能名（实现为 `CrossRWKVHeadV8`）；`summary.frame_encoder` 是块不是部件。
5. **引用条目错误以 `papers-*.md` 核对结论为准**。旧稿已知错误：ELD 编号（2003.12751 非 2004.14068）、RDDM 编号（2308.13712 非 2303.08530）、DualDn 第一作者（Ruikang Li 非 Zhang）、RetinexNet 编号（1808.04560）、Foi 2008 作者（4 人无 Astola）、TCE-Net 页码（4703-4723 与 4015-4034 两说，待核实）。
6. **不得把 VRT/RVRT 说成低光方法**。两者是通用视频修复 Transformer（SR/去模糊/去噪/插帧），§2.1 引用时措辞须限定任务范围。

## 输出约定

- 论文正文用**英文**；与用户的讨论用中文。
- 数学用 LaTeX：行内 `$...$`，独立公式 `$$...$$` 并编号 (1)(2)(3)…
- 表格用 Markdown；数字与 `docs/v8/13-实验结果汇总.md` 一致。
- 草稿中的未定项标 `[待核实]`，不写进终稿。
