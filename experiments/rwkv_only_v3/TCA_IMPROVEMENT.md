# TCA 改进方案设计（保留多帧/帧间对齐框架）

> 依据：参考库对比分析（URWKV/DRWKV/EvRWKV）+ ablation 归因结论
> 约束：**保留**多帧信息运用与帧间对齐设计（可修改实现），不采用"逐帧独立"的回避路线
> 日期：2026-09-04

---

## 0. 问题回顾（改进靶点）

| # | 缺陷 | 证据 |
|---|------|------|
| P1 | `_scan_cumsum` 跨 chunk 状态传播数学错误（decay_state 方向反向 ew^{cs-1-j}，应为 ew^{j+1}；state 更新 off-by-one）| 代码审查 vs URWKV/DRWKV CUDA kernel |
| P2 | WKV 分支休眠：单层 + 16ch/方向 + proj_out/spatial_gamma 双重零初始化 | 参考模型为 15/32/20 层、全通道、无 gate 或 gamma=1 |
| P3 | C_omega 全局稠密 softmax warp @32²：极端低通 → 结构性模糊；帧门控 softmax 强制加权（无法弃权）| pair45 模糊；motion_map 退化 |
| P4 | C_omega 在原始特征上计算、却 warp 增强后特征（信号/载体错位）| pure_rwkv_sace.py:383-391 |

三个参考模型的共识：**RWKV 不做帧间对齐**。但本设计约束要求保留多帧利用 → 改造对齐机制本身，使其在数学上不再必然产生模糊。

---

## 方案 T-A：算子修复 + 解除休眠（地基工程，先行）

**不改架构，只修内核。后续所有实验的地基。**

### T-A1: WKV 算子替换（二选一）
- **选项 1（推荐）**：移植 URWKV/DRWKV 的双向 CUDA kernel（本地已有 `reference_repos/URWKV/model/modules/`，带 running max-subtraction，无 clamp 需求，fwd/bwd 融合）。我们的 4 方向扫描各自调用一次即可。
- **选项 2（无编译环境时的退路）**：纯 PyTorch 数学修复——
  ```python
  # 旧: decay_state = ew_pow[:, cs-1:cs] / ew_pow[:, :cs]     # ew^{cs-1-j} 方向反
  # 新: decay_state = ew_pow[:, 1:cs+1] / ew_pow[:, 0:1]      # ew^{j+1} 物理正确
  # 旧: state_num = ew_pow[:, cs-1:cs] * state_num + S_loc[:, -1:]
  # 新: state_num = ew_pow[:, cs:cs+1] * state_num + S_loc[:, -1:]
  ```

### T-A2: 解除休眠
- `spatial_gamma` 零初始化 → **ones**（EvRWKV 的 LayerScale=1 做法）
- 保留 `proj_out` 零初始化（仅一处，保稳定）——解除双重零门控
- 每方向通道 16 → 32（`channels=64` 改 2 方向×32ch，或 4 方向共享 heads）

### 预期与判读
- 若 ep10 PSNR 较同配置 v1 提升 ≥0.3dB：算子修复兑现，WKV 首次真正参与全局建模
- 若无变化：说明 TCA 的瓶颈确实全在时序路径（P3/P4），WKV 部分继续保留但不再是改进重点

---

## 方案 T-B：局部窗口相关对齐（替代全局稠密 warp，对齐机制核心改造）

**保留帧间对齐，但从"全帧凸组合"改为"窗口内局部对齐"——数学上消灭全局模糊源。**

### 设计
```
旧 (TemporalCorrespondence/Aggregation):
  特征 avg_pool → 32×32 → N×N 全局 softmax (N=1024) → warp → 上采样
  每格对应 8×8 原始像素, softmax 全局强制归一 → 输出=全帧凸组合 → 模糊

新 (LocalWindowAlignment):
  在 H/2 全分辨率上, 对中心帧每个位置 p:
    1. 取邻帧 p 的 R×R 邻域窗口 (R=9, 即 ±4px 位移容限)
    2. 窗口内余弦相似度 → 窗口内 softmax (温度可学习)
    3. warp(p) = 窗口内加权平均          ← 局部凸组合, 不跨远距离
  实现: F.unfold 提取邻帧窗口 (B, C·R², H·W) → reshape → bmm
  输出 = warp + 残差(center_feat)
```

### 关键性质变化
| 性质 | 旧全局 warp | 新窗口 warp |
|------|-----------|------------|
| 对齐分辨率 | 32²（每格 8×8px）| H/2=256 级（每格 2×2px）|
| 位移容限 | 理论无限但精度趋零 | ±4px（SDSD 实际运动范围）|
| 模糊来源 | 全帧凸组合（必然）| 仅窗口内 81 位置加权（局部）|
| 高频保留 | 池化摧毁 | 无池化，完整保留 |
| 显存 | N² 矩阵 (1024²) | N·R² (16384×81×4帧 ≈ 21MB) |

### 帧聚合改造
- 帧门控 `softmax(帧)` → **per-frame sigmoid gate**：允许"该帧不可靠→权重→0"的弃权（旧 softmax 必须把权重分给某帧）

### 预期
- 运动区域模糊显著缓解（pair45 类场景 PSNR/LPIPS 改善）
- 静止区域保留多帧降噪收益（窗口 softmax 峰值在中心 → 近似恒等）

---

## 方案 T-C：对齐置信度回退门控（信号源修复，可与 T-B 合并）

**v2 的 motion gate 失败根因是信号源（C_omega 对角线）无判别力。T-C 换用真实置信度信号。**

### 设计
```
置信度源 = 窗口 softmax 分布的形状（T-B 的副产物，零成本）:
  conf(p) = max_prob(p)   或   1 - normalized_entropy(p)

  conf 高 (峰值锐利=唯一明确匹配) → 信任对齐: 用邻帧聚合信息 (降噪)
  conf 低 (分布平坦=歧义/遮挡/运动) → 回退中心帧 (保锐度)

  F_out = conf · F_aligned + (1-conf) · F_center
```

### 与 v2 motion gate 的本质区别
| | v2 conf_proj | T-C |
|---|---|---|
| 信号源 | C_omega 对角线 4 个标量 | 窗口 softmax 分布形状（逐像素、真实）|
| 信号质量 | 退化为全局常数 0.64 | 峰值/熵直接反映匹配唯一性 |
| 作用位置 | 全局混合比 | 逐像素路由 |

### 合并训练建议
T-B 与 T-C 共享窗口 softmax 副产物，**建议合并为 T-BC 一次训练**（类比 BC 实验：2A+4 合并验证）。若 T-BC 有效再拆分归因。

---

## 实验执行计划

| 序 | 实验 | 内容 | 依赖 |
|:--:|------|------|------|
| 0 | BC 完成 | v2+2A+4 → 若 ≥20.2 确立新基线 | 🔄 训练中 ep6/20 |
| 1 | **T-A** | 算子修复 + gamma=ones + 通道扩充（在 BC 配置上叠加）| BC 完成 |
| 2 | **T-BC** | 局部窗口对齐 + sigmoid 帧门 + 置信度回退（替换 C_omega warp）| T-A |
| 3 | 归因拆分 | 视 T-BC 结果：T-B / T-C 单独消融 | T-BC |
| 4 | (备选) B/C 单独 | 2A 与 4 的单独贡献归因 | 视需要 |

全部沿用既有基础设施：`model_ablation.py` flag 机制 / `train_ablation.py` / `keepalive_ablation.sh` / 温度记录 / E-core 部署 / 20 epochs。
