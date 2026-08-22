# v2 概念模型分析与下一步改进方案

阅读了 v2 的训练结果和 Deepseek 建议，我认为**Deepseek 抓住了一个关键现象（motion_map 退化到 0.640）但对其含义的解读值得商榷**。让我先重新审视诊断，再给出改进方案。

---

## 1. 对现有诊断的重新审视

### 1.1 motion_map = 0.640 真的是"退化"吗？

Deepseek 的判断：pair20（静态）和 pair45（运动）motion_map 都是 0.640 → 学到全局平均值 → 门控失效。

但这个解读有问题：

**观察 1**：v2 相比 v1 提升了 +0.22 dB（19.80 → 20.02）。如果 motion_map 真的完全退化为常数，那 v2 应该退化为 "v1 + 一个 40K 参数的多余分支"，PSNR 不应该提升。**提升说明门控在某种意义上是有效的**。

**观察 2**：Deepseek 报告的是**图像平均值** 0.640，不是**空间分布**。两个场景平均都是 0.64，可能是：
- （a）motion_map 完全均匀（真退化）
- （b）motion_map 在空间上有分化，但两个场景的运动像素比例恰好接近

这两种情况截然不同，需要看 motion_map 的**直方图或方差**，而不只是均值。在没有这些数据前，"退化为全局偏置"的结论不成立。

**观察 3**：conf_proj 输入是 `(B, N, 4)` 的 per-pixel 4→1 mapping，其中 4 个值是 4 个邻帧的对角相关值。它**天然是 per-pixel 的**，不需要卷积就能空间分化——只要 C_omega 对角线本身有空间分化，motion_map 就有空间分化。

**真正的问题可能是**：C_omega 的对角线值本身在 SDSD 数据集上不够 discriminative，而不是 conf_proj 结构不够表达。**Deepseek 的 P1（加空间卷积）修的是错误的组件**。

### 1.2 pair45 PSNR=15.04 的真正原因

Deepseek 归因于"无光照估计 + 运动+低SNR+等权聚合"。这个判断部分对，但漏了一个更根本的原因：

**Sigmoid 输出对暗光的偏差**：head 最后用 `Sigmoid()` 输出 RGB。SDSD 的 GT 图像亮度范围广，暗场景的 GT 值可能集中在 [0.05, 0.3]。Sigmoid 在这个区间的梯度是 `y(1-y) ≈ 0.05×0.95 ≈ 0.05`——**梯度只有中值区的 1/5**。

**结果**：训练时暗场景的梯度信号被 Sigmoid 天然压制，模型对暗光的拟合能力系统性偏弱。这解释了为什么 pair45（PSNR=15）远低于 pair20（PSNR=26.7）——不完全是"能力不足"，而是"梯度不够"。

这个问题在完整 F10m5 里没有，因为 ISPN 通过 gain_map 显式处理亮度，输出前不经过 sigmoid 挤压。

---

## 2. 对 Deepseek 建议的评估

| 方案 | Deepseek 优先级 | 我的评估 | 理由 |
|------|:--:|:--:|------|
| P1 空间 conf_proj | ★★★ | ★ | conf_proj 本身已 per-pixel，加卷积解决的不是核心问题 |
| P2 通道级门控 | ★★☆ | ★★☆ | 有道理，但需要先验证单通道门控是否已工作 |
| P3 高频残差 | ★★☆ | ★★★ | 极低成本，直接解决 head 容量问题 |
| P4 显式帧权 | ★☆☆ | ★★☆ | 作为 ablation baseline 有价值 |
| T1-T4 三分支 | ★★★ | -- | 不属于概念模型范畴，暂不评估 |

**Deepseek 的最大盲点**：没有诊断 sigmoid 输出对暗光的梯度问题。这是 pair45 PSNR=15 的最直接原因，比 conf_proj 结构更值得优先处理。

---

## 3. 我建议的改进方案

按"最大 ROI + 最小改动 + 可清晰归因"排序：

### 方案 1：诊断先行 —— 可视化 motion_map 分布 ★★★

**成本**：零训练。只需在推理时保存 motion_map 并绘制直方图。

**必要性**：所有后续改动都依赖"motion_map 是否真的退化"的答案。如果它有空间分化只是均值巧合相同，那 P1 就是解决伪问题。

**具体操作**：
- 在 pair20 和 pair45 上分别保存 motion_map (H×W)
- 计算：均值、方差、直方图（0-1 分 10 桶）
- 可视化叠加到输入图像上，看是否与语义运动区域对齐

**判定分支**：
- 如果 motion_map 空间方差 < 0.01（真退化）→ 走方案 2A
- 如果 motion_map 有空间分化但两场景均值巧合相同 → 走方案 2B

### 方案 2A：如果 motion_map 真的退化 —— 修 C_omega 输入源 ★★★

**根因**：如果 C_omega 对角线在不同场景相似，问题在 TCA 的 temperature τ 太大导致 softmax 过于均匀。

**修改**：
```python
# TemporalCorrespondence 中
# 当前: tau = softplus(tau_raw) + 0.05
# 改为: tau = softplus(tau_raw).clamp(0.02, 0.5) + 0.02
```

同时**降低 conf_proj 的初始温度**，让 sigmoid 更 discriminative：
```python
# 当前 conf_proj 末端: Sigmoid()
# 改为: 输出前乘可学习标量 sigmoid(logits * scale), scale 初始化为 3.0
self.conf_scale = nn.Parameter(torch.tensor(3.0))
conf_map = torch.sigmoid(logits * self.conf_scale)
```

**成本**：+1 参数。**预期**：motion_map 分布拉开，方差增大。

### 方案 2B：如果 motion_map 有空间分化 —— 增强表达 ★★

保留当前 conf_proj 结构（它是对的），改为**在 motion_map 使用点做增强**：

```python
# 当前: tca_gated = tca_up × (1 - m) + l1_gate × m
# 改为: 允许非线性混合
mix_input = torch.cat([tca_up, l1_gate, motion_map.expand(-1, 64, -1, -1)], dim=1)
tca_gated = self.mix_conv(mix_input)  # Conv(64+64+64 → 64, 3×3)
```

**成本**：~37K 参数。**预期**：融合方式不再是硬性线性插值，允许在运动边界有更复杂的过渡。

### 方案 3：解决暗光梯度压制 ★★★

**问题**：Head 末端 Sigmoid 在暗区（y<0.3）梯度衰减。

**方案 A（最小改动）**：改用不饱和输出
```python
# 当前: nn.Conv2d(64, 3, 3, 1, 1), nn.Sigmoid()
# 改为: nn.Conv2d(64, 3, 3, 1, 1)  # 无激活，训练时 clamp
# forward 时: res_t = torch.clamp(logits, 0.0, 1.0)  # 只在推理时 clamp
```

**方案 B（较优）**：残差 + tanh
```python
# 假设有 img_center 作为基准
# res_t = image_center + torch.tanh(logits) * 0.5
# 输出以中心帧为基准做残差，tanh 允许双向修正
```

**成本**：零参数。**预期**：pair45 类场景 PSNR 提升 1-2 dB（这是我最有把握的预测）。

### 方案 4：高频残差注入（Deepseek P3）★★★

**保留 Deepseek 的建议，因为几乎零成本**：

```python
# 在 head 输出后
img_center = x[:, T//2]  # (B, 3, H, W)
hf = img_center - F.avg_pool2d(img_center, 5, stride=1, padding=2)
res_t = res_t + 0.1 * hf  # 或用可学习 α
```

**成本**：0 或 1 参数。**预期**：SSIM 提升，尤其对静态清晰场景（pair20 类）。

### 方案 5：诊断 TCA 到底提供多少价值（Deepseek P4 变体）★★

不作为改进，作为**必做的 ablation**：

```python
# 完全禁用 TCA，只用 concat[l1_center, l3_up]
# 训练同样 epochs
```

如果禁用 TCA 后 PSNR 掉 <0.3 dB，说明 470K 参数的 TCA 严重浪费；如果掉 >1 dB，TCA 确实有独立贡献。这个数据决定后续是否值得优化 TCA 内部结构。

---

## 4. 推荐的实验执行顺序

**阶段 1（诊断，1 天内完成）**：
1. 方案 1：可视化现有 v2 的 motion_map（不需重训）
2. 方案 5：训练一个禁用 TCA 的 baseline（RWKV-Only 骨架 - TCA）

这两步会给出三个关键数据点：
- motion_map 是否真的退化
- TCA 的实际边际贡献
- 当前架构的下限（无 TCA baseline）

**阶段 2（针对性修复）**：

基于阶段 1 结论选择：

- **情况 A**：motion_map 真退化 + TCA 有价值 → 优先执行 2A + 3 + 4
- **情况 B**：motion_map 有分化 + TCA 有价值 → 优先执行 3 + 4，观察增益，再考虑 2B
- **情况 C**：TCA 边际价值低 → 转向 RWKV-Only - TCA + 更强 head 的路线

**阶段 3（组合验证）**：

选取阶段 2 中单独有增益的方案，做组合训练，验证增益是否叠加。

---

## 5. 一个更深的问题

回看 v2 的训练曲线：ep30 = 19.66，ep40 = 20.02，ep49 = 19.77——**过拟合已经开始**。这意味着当前架构在 SDSD-indoor 上的表达容量已经**基本用尽**。

进一步优化 v2 的天花板可能在 20.3-20.5 dB。要真正突破，需要的是**范式变化**，不是继续堆门控/卷积。

范式变化的候选方向（不建议现在做，但值得思考）：
- 替换 C_omega 为可变形卷积对齐（deformable attention）
- 采用 flow-based warping（RAFT-lite 提取光流后 warp）
- 引入 diffusion prior（预训练模型做 refinement）

**但在这些之前**，先把方案 3（sigmoid 问题）做掉——**这个改动可能一次提供 0.5+ dB**，是当前最大的 free lunch。

---

## 6. 关键判断

- **Deepseek P1 是解决伪问题**（除非阶段 1 诊断证实退化）
- **暗光 sigmoid 梯度压制是被完全忽略的最大问题**（方案 3）
- **v2 已接近容量上限**，后续改动应聚焦于**释放已有容量**（方案 3）和**低成本增益**（方案 4），而不是**堆结构**（方案 2B）
- **必须先做 motion_map 可视化**再决定 conf_proj 相关改动，否则可能白干