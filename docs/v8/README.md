# v8: 多源噪声分割结构的整理与定稿

> 创建于 2026-10-15
> 目标: 整理、改进、定稿多源噪声分割结构的 RWKV 实现部分

## 目录内容

| 文件 | 内容 |
|------|------|
| `01-设计.md` | 现状梳理 (来源与缺口)、设计原则、Encoder 改造、扫描策略处置 |
| `02-实验计划.md` | 四组实验 (含零成本事实核查)、运行编排、判读规则 |

## 一句话概括本版本

v7.5 已定稿三分支的**融合方式与监督目标**; v8 处理它上游的 **TCA 与编码器的 RWKV 实现**
—— 把混杂的来源整理成规范结构, 补上缺失的机制, 并厘清我们的创新点边界。

## 四条工作主线

| # | 内容 | 关键点 |
|:--:|------|--------|
| 1 | 整理简化 RWKV 基线 | 以 Vision-RWKV / Restore-RWKV 为主要参考; 用 Omni-Shift 替代当前的"无局部上下文" |
| 2 | 探索 RWKV 编码器 | 把 `SharedEncoder` 内部的 NAFBlock 换成 RWKV block |
| 3 | 引入 RWKV-7 与 Value Residual | 先期直接搬运原始实现, 不做小修小改 |
| 4 | 厘清"只有 Spatial Mix"的原因 | 结论: 历史沿革造成, 无技术理由, 应当补回 Channel Mix |

## 已核实的关键事实

以下都已直接从代码或本地参考实现确认, 不是推测:

| # | 事实 | 依据 |
|:--:|------|------|
| 1 | 当前衰减 `w` 是逐通道可学习常量 | `pure_rwkv_sace.py::BiWKV`, `w = -softplus(self.spatial_decay)` |
| 2 | **Vision-RWKV 的衰减也是可学习常量** | `reference_repos/Vision-RWKV/.../vrwkv.py:164,219` —— 两者一致, 不是缺口 |
| 3 | TCA 空间头无 Channel Mix | `RWKVSpatialHead.forward`, `proj_out` 后直接 `post_norm` |
| 4 | 当前**没有**循环次数参数 M | MatrixRWKV 用"堆 2 个 block"; Re-WKV 的 M 是同一模块串行循环, 两者不同 |
| 5 | Omni-Shift 本地已有完整实现 | `reference_repos/EvRWKV/egllie/models/Cross_Restore_RWKV.py:103` |
| 6 | 2D 四方向扫描来自 RSRWKV | 该论文不针对 low-level vision, 属需要重新评估的部分 |

第 2 条尤其重要: 我最初认为当前实现丢掉了"数据相关衰减"这一 RWKV 核心机制, 查证
Vision-RWKV 源码后发现它的 `spatial_decay` 同样是 `nn.Parameter`, 两者做法一致。
**这个纠正避免了按错误前提去改代码。**

## 规模

0.5 h 代码核查/搬运 + 4 个 run × 6.8 h ≈ **28 h**。

## 前置条件

**BranchM 的尺度缺陷必须先修** (见 `../v7.5/03-design定稿.md` §三)。该缺陷与本轮改动无关,
但它使 v7.5 的判据 3 无法通过、判据 4 无法解释。在这个缺陷修复前开始 v8 对照, 所有 run
都会被同一个已知问题污染。

## 当前状态 (2026-10-16)

BranchM 修复与两个探针 (branchM_fix、phase1) 已跑完。裁决写在 `01-设计.md` §六·补:

- **b 方案 (residual 目标) 被否决** —— 它没有复活 M 路, 反而让 N、M 两路塌成常数图。
  根因是残差目标没有直流分量, 按带归一化后"输出常数 0"就是该路损失的下界。
- **`outputs/v8_readout/v8_ablation.log` 里 phase1 的消融数字作废** (旧口径错误, 见设计文档)。
- **R1 归因补测进行中** (固定 legacy 头、只换 residual 目标), 这是把"目标"与"空间头"两个变量
  分开的唯一途径, 也是决定 A 组编排的先决条件。
- **B1 与 C 的代码已就绪**: B1 (RWKV 编码器) 排在 R1 之后自动启动;
  C (RWKV-7 / Value Residual) 仅冒烟不训练。
- 43 项单测全绿。

接下来最重要的一件事是等 R1 跑完, 据其结论决定 A 组是合并实施还是拆成单变量。
详见 `02-实验计划.md` §零 与 §三。
