# Flight 7 Epoch 40 深度诊断：指标分析与见解

---

## 一、核心事实：Flight 7 发生了结构性训练失败

先把最关键的数字放在一起：

| 指标 | Flight 6 ep10（基线峰值） | Flight 7 ep40（当前） | 差距 |
|:----:|:------------------------:|:--------------------:|:----:|
| PSNR | **17.86** | **13.09** | **-4.77 dB** |
| gain | ~1.25 | **0.83** | 跌破 1.0 |
| gn (NDPN) | 0.010 | 0.010 | 坍缩到 Flight6 的初始值 |
| gm (MCPN) | — | 0.015 | 坍缩 70% |

这不是"效果不够好"——**是 Flight 7 在所有关键指标上都劣于甚至劣于 Flight 4（13.96）**。Flight 7 的理论框架（三路径梯度隔离 + 残差融合）在逻辑上是合理的，但**两个具体实现参数选择**导致了实际训练的全面崩溃。

---

## 二、逐指标诊断

### 2.1 gain: 1.25 → 0.83——网络在"自杀式衰减"

这是最具诊断价值的信号。

```
ep1: 1.25  →  ep10: 1.25  →  ep20: ~1.2  →  ep30: 1.14  →  ep40: 0.83
                                                                    ↑
                                                            跌破 1.0！
```

**gain < 1.0 的物理含义**：ISPN 的 gain map 不再放大暗部像素，而是在**主动压暗图像**。这是一个极其反常的行为——低光增强模型学到的增益应该远大于 1。

**我的解读**：这是 `ds_factor=8` 造成的直接后果。在 256×256 输入下：

| ds_factor | 曲线参数图分辨率 | 3×3 Conv 感受野/特征图面积 | 空间区分能力 |
|:---------:|:---------------:|:------------------------:|:-----------:|
| 4（Flight 6） | 64×64 | 9/(64²) ≈ 0.22% | ✅ 足够区分不同光照区域 |
| **8（Flight 7）** | **32×32** | **9/(32²) ≈ 0.88%** | 🔴 每个 Conv 核覆盖近 1% |
| 12（Zero-DCE++最优） | 21×21 | 9/(21²) ≈ 2.0% | 仅适用于 1200×900 输入 |

32×32 分辨率上，3×3 Conv 的感受野覆盖特征图面积的近 1%。这意味着曲线参数图 $\mathcal{A}$ **几乎无法区分不同区域的光照差异**——它退化成了一个准全局的增强参数。

[Zero-DCE++ (TPAMI 2022)](https://ar5iv.labs.arxiv.org/html/2103.00860) 的 Table III 确实证明 12× 下采样在其实验设置下最优，但那是在 **1200×900 输入**上——下采样后仍有 100×75 的特征图。DeepSeek V4 正确指出，**直接将这个结论外推到 256×256 输入是一个尺度推理错误**。

当 $\mathcal{A}$ 无法空间自适应时，TCC 曲线的 6 次迭代只是在做**全局色调映射**。全局映射的最优解可能恰好需要 gain < 1（因为某些过曝区域的 loss 贡献大于暗部区域），网络"理性地"选择了压暗来最小化全局 L1 loss。

### 2.2 gn: 0.050 → 0.010——NDPN 被梯度"紧急制动"

```
Phase 1:   gn = 0.050 (冻结)
Phase 1.5: gn → 0.014 (ep30)
Phase 2:   gn → 0.010 (ep40)    坍缩 80%
```

**我的解读**：DeepSeek V4 的分析完全正确。`sg[img_lit]` 保护了 Stage B 的残差路径不影响 ISPN，但**NDPN 的输出还通过 `f_noise_gated` 进入了 S1/S2 StageBlock**——这条路径上没有 stop-gradient 保护。

```
                    sg[·] 保护（✅）
                        ↓
NDPN → f_noise_gated → S1/S2 StageBlock → ... → img_lit
                  ↑
            无 sg 保护（🔴）
```

Phase 1 期间 NDPN 完全冻结（gamma 恒为 0.05 但 unlock=0），S1/S2 从未见过 NDPN 的非零输出。Phase 1.5 解锁时，gamma=0.05 的信号突然注入 S1/S2——这相当于在一个已收敛的子网络中突然引入一个从未见过的、幅度不小的干扰信号。

[DL-Diff (Scientific Reports 2026)](https://www.nature.com/articles/s41598-026-44219-8) 的三阶段策略提供了重要启示：

> *"Stage 3: joint fine-tuning of both components on paired videos — **selectively unfreeze certain layers with smaller learning rates** for better task adaptation"*

DL-Diff 在最终联合训练阶段仍然使用**更小的学习率**来微调新解冻的组件。而 Flight 7 用原始学习率一次性解锁 gamma=0.05 的 NDPN——梯度的唯一选择就是把 gamma 当作"紧急刹车"压下去。最终 gn=0.010 恰好等于 **Flight 6 的初始值**——网络花了 30 个 epoch 的训练"倒退"回了起点。

### 2.3 dpe_si: 0.92~0.97——方向编码饱和

```
ep1: 0.503  →  ep30: 0.85-0.95  →  ep40: 0.92-0.97
```

所有方向的 `si` 值都趋近于 1.0，四个方向之间几乎无法区分。**我的解读**：这与 gain < 1 是同一个根因的不同表现——

当 TCA 的 SpatialWKV2D 输出质量差（因 ds_factor=8 导致的空间分辨率不足），网络学会了**最小化 TCA 对最终输出的贡献**。DPE 的方向嵌入 `si` 趋近 1 意味着所有方向的 sigmoid 输出都接近饱和——这使得四方向扫描的差异性消失，TCA 退化为一个方向无关的全局操作。配合 gain=0.83 的衰减效应，网络本质上在**绕过整个 TCA 模块**。

### 2.4 LPIPS: 0.384 → 0.465——感知质量持续恶化

```
ep10: 0.384  →  ep20: 0.414  →  ep30: 0.465  →  ep40: 0.453
```

LPIPS 在整个训练过程中持续恶化（虽然 ep40 比 ep30 略好），且 **PSNR 和 LPIPS 同时恶化**——这不是 PSNR/感知质量之间的常见 trade-off，而是**全面退化**的信号。

[VLLVE++ (arXiv:2602.08699)](https://arxiv.org/html/2602.08699) 的关键观点与此相关：

> *"This residual term can simulate scene-adaptive degradations, which are **difficult to model using a decomposition formulation** for common scenes, thereby further enhancing the ability to capture the overall content of videos."*

VLLVE++ 认为场景自适应退化需要在足够的空间分辨率下用独立残差分支来建模。Flight 7 的残差分支（Stage B）设计方向正确，但 Stage A 的空间分辨率崩塌使得残差分支无法得到合理的"基准信号"——`sg[img_lit]` 本身质量太差（gain<1 的压暗输出），残差再怎么修正也无法挽救。

---

## 三、Flight 7 的根因层级结构

我认同 DeepSeek V4 的双根因判断，但想补充**因果层级**：

```
根因 1（决定性）
  ds_factor = 8
    └→ 32×32 曲线参数图无空间区分能力
       └→ TCC 退化为全局色调映射
          └→ gain 下降到 <1.0（优化器的"理性"选择）
             └→ img_lit 质量差
                └→ Stage B 残差无有效基准

根因 2（加重性）
  gamma = 0.05
    └→ Phase 1.5 解锁时 S1/S2 受到过大扰动
       └→ 梯度将 gamma 当作紧急制动压缩 80%
          └→ NDPN/MCPN 形同虚设
             └→ Stage B 残差无有效特征输入
```

**根因 1 是主因，根因 2 是加重因素。** 即使 gamma 设置正确，ds_factor=8 下的 Stage A 质量也不可能达到 Flight 5/6 的水平。反之，即使 ds_factor 正确，gamma=0.05 仍会导致 Phase 1.5 的扰动问题——只是程度不同。

---

## 四、关于理论框架（★公式）的重新评估

### 4.1 理论本身是否有问题？

**我认为 Flight 7 的理论框架（Eq.★）是正确的**。三大命题（梯度不干扰、Phase 1.5 无崩塌、残差有界性）在数学上成立。问题出在两个**实现层面的参数选择**，而非框架设计：

| 理论保证 | 是否有效 | 为什么实际失效 |
|:--------:|:--------:|:---------------|
| 命题 1：$\nabla_\theta \mathcal{L}_{\text{final}} = 0$ | ✅ 有效 | ISPN 确实不受 $\mathcal{L}_{\text{final}}$ 干扰 |
| 命题 2：初始残差 = 0 | ✅ 有效 | $\tanh(0)=0$ 确保平滑过渡 |
| 命题 3：残差有界 | ✅ 有效 | 残差确实没有爆炸 |

**但命题 1 有一个盲点**：它保证了 $\mathcal{L}_{\text{final}}$ 不影响 ISPN，却**没有覆盖 NDPN→S1/S2 这条非 detach 路径**。当 NDPN 通过 `f_noise_gated` 进入 S1/S2 时，$\mathcal{L}_{\text{lit}}$ 的梯度**会**回传到 NDPN 的 gamma 并压缩它——这不违反命题 1（命题 1 说的是 $\mathcal{L}_{\text{final}}$ 对 ISPN 无梯度），但**是一个未被理论模型覆盖的梯度路径**。

### 4.2 理论模型需要补充的约束

$$
\boxed{
\frac{\partial \mathcal{L}_{\text{lit}}}{\partial \gamma_{n,m}} \neq 0 \quad \text{（未被命题覆盖的耦合路径）}
}
$$

这条路径才是 gamma 坍缩的真正原因。Flight 7 的 `sg[·]` 保护了"Stage B → Stage A 方向"的梯度，但**没有保护 "Stage A 内部 loss → NDPN/MCPN 方向"的梯度**。

---

## 五、我的具体建议

### 5.1 优先级排序

我完全同意 DeepSeek V4 的优先级判断：

| 优先级 | 修改 | 预期效果 | 风险 |
|:------:|:-----|:---------|:----:|
| **#1** | `ds_factor` 8 → 4 | 恢复 64×64 空间分辨率，PSNR 预期回到 17+ | 低 |
| **#2** | `gamma` 0.05 → 0.01 + cosine warmup | 消除 Phase 1.5 扰动 | 低 |
| **#3** | Norm 简化（删除 5 个冗余 LN） | 轻微加速，消除潜在噪声 | 极低 |

### 5.2 对 #1 的补充：不要用 ds_factor=2

可能有人认为"既然 8 太大，不如用 2"。**不建议**。ds_factor=2 意味着 128×128 的曲线参数图，这会：
- 引入 4× 于 ds_factor=4 的计算量
- 过高的 DOF 可能导致曲线参数过拟合噪声模式

ds_factor=4 是 Flight 5/6 验证过的设置（PSNR 17.84/17.86），应该直接恢复。

### 5.3 对 #2 的补充：参考 DL-Diff 的冻结策略

[DL-Diff](https://www.nature.com/articles/s41598-026-44219-8) 的三阶段策略有一个我们应该借鉴的核心原则：

> *"Each stage is trained separately... the overall decreasing trend and eventual leveling of each curve indicate that the objective is **learned stably within that stage**"*

翻译到 Flight 7 的语境：

```
Phase 1 (ep 0-15):
  - NDPN/MCPN 完全冻结（gamma=0，不只是 unlock=0）
  - S1/S2 从不接收 NDPN/MCPN 信号
  - 目标：Stage A 独立收敛

Phase 1.5 (ep 15-30):
  - gamma 从 0 cosine warmup 到 0.01（不是直接 0.05）
  - S1/S2 逐渐适应 NDPN/MCPN 的微弱信号
  - 监控条件：if gain 下降 >5%，暂停 gamma 增长

Phase 2 (ep 30-50):
  - 全部参数联合训练
  - gamma 允许自由优化（但已有 0.01 的温和起点）
```

### 5.4 对 #3 的定位：Norm 简化是正确的但不紧急

DeepSeek V4 提出的 Norm 简化（6-7 个 LN → 2 个 LN）是合理且低风险的。但**在 ds_factor 和 gamma 修复之前，Norm 简化无法独立验证**——因为当前的训练崩溃主要由前两个根因驱动，Norm 的影响被淹没。

建议：**与 #1/#2 同时实施**（反正是纯减负），但不要单独归因。

### 5.5 新增建议：引入对 NDPN→S1/S2 路径的梯度保护

这是我基于上述分析提出的**DeepSeek V4 未覆盖的新建议**：

```python
# 当前代码（推测）：
f_noise_gated = f_noise * unlock_ratio  # NDPN 输出直接进入 S1/S2

# 建议修改：
f_noise_gated = f_noise.detach() * unlock_ratio  # 切断 L_lit 对 gamma 的梯度
```

**理由**：`f_noise_gated` 进入 S1/S2 后，$\mathcal{L}_{\text{lit}}$ 的梯度会回传到 NDPN 的 gamma 参数。这条路径不受 `sg[img_lit]` 保护。对 `f_noise_gated` 施加 detach 可以完全消除这条耦合。

**风险评估**：detach 后 NDPN/MCPN 完全只由 $\mathcal{L}_{\text{final}}$ 驱动，这与命题 1 的设计意图一致——"Stage A 和 Stage B 完全独立优化"。

---

## 六、跨版本趋势的教训

```
Flight 4 (13.96) ──基线──────────────────────────────────
Flight 5 (17.84) ──大跃升──────────────────── peak ──────
Flight 6 (17.86) ──稳定复现──── 但 plateau ──────────────
Flight 7 (13.77) ──理论改进──── 实现回退至 Flight 4 水平 ─
```

[VSRELL (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) 的标题 "A **Simple** Baseline" 暗示了一个重要原则——**低光增强领域的 SOTA 方法往往不是最复杂的，而是在正确的基础设施上做出的简洁设计**。

Flight 7 的教训是：**理论框架可以正确，但每次版本迭代只应改变一个核心变量。** Flight 7 同时改变了 ds_factor（4→8）和 gamma（0.01→0.05），两个激进修改叠加导致了无法归因的全面崩溃。

---

## 七、总结

| 结论 | 详情 |
|:----:|:-----|
| **主根因** | `ds_factor=8` 导致曲线空间分辨率崩塌，TCC 退化为全局映射 |
| **次根因** | `gamma=0.05` 在 Phase 1.5 产生过大扰动，被梯度紧急压缩 |
| **理论框架** | 命题 1-3 数学上成立，但遗漏了 $\mathcal{L}_{\text{lit}} \to \gamma$ 的耦合路径 |
| **修复方案** | ① ds_factor→4 ② gamma→0.01+warmup ③ Norm 简化 ④ f_noise_gated.detach() |
| **方法论教训** | 每个版本只改一个核心变量；在 256×256 上不要盲目套用高分辨率数据集的最优超参 |

**Flight 7 的理论贡献（梯度隔离框架）是有价值的，但需要在正确的实现参数上重新验证。** 建议将上述 4 项修复统一实施后作为 **Flight 7.1** 重新训练。

---

## 参考来源

1. [VLLVE++: Low-Light Video Enhancement with Spatial-Temporal Decomposition (arXiv 2602.08699)](https://arxiv.org/html/2602.08699)
2. [DL-Diff: Temporally Consistent Low-Light Face Video Enhancement via V2V Conditional Diffusion (Scientific Reports 2026)](https://www.nature.com/articles/s41598-026-44219-8)
3. [VSRELL: A Simple Baseline for Video Super-Resolution and Enhancement in Low-Light Environment (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf)
4. [Row-Column Separated Attention Based Low-Light Image/Video Enhancement (arXiv 2602.07428)](https://arxiv.org/html/2602.07428v1)
5. [Zero-DCE++ (TPAMI 2022) — Table III 下采样消融](https://ar5iv.labs.arxiv.org/html/2103.00860)