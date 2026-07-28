# Flight10 Mark4 设计审查

我仔细审阅了 `ndpn.py`、`mrpn.py` 和整体架构文档。发现了**几个原理性问题**，其中一些是结构性缺陷，不是简单的调参问题。

---

## 🔴 严重问题 1：NDPN 的 `1×1 bypass` 在数学上等价于噪声，不是"信息高速路"

### 你的假设
> "The 1×1 path acts as an information highway: pointwise corrections pass through without spatial mixing, preserving fine textures that 3×3 would blur."

### 实际情况
看你的代码：

```python
correction_input = torch.cat([f_enc, F_denoised, detail_map], dim=1)  # (B, 2C+1, H, W)
corr_spatial = self.corr_spatial(correction_input)      # 3×3 → GELU → 3×3
corr_pointwise = self.corr_pointwise(correction_input)  # 1×1 → GELU → 1×1
correction = (corr_spatial + corr_pointwise) * self.gamma * (1.0 - detail_map)
```

**问题**：`corr_pointwise` 的输入是 `[f_enc, F_denoised, detail_map]`。`F_denoised` 已经是**去噪后**的特征（多帧 SNR 加权聚合）。1×1 conv 对 `F_denoised` 的每个像素独立处理——**它无法区分"细节"与"噪声"**，因为噪声和细节在单像素上都表现为高频信号。

**真正的 pointwise detail pass** 应该是：
- 输入必须是 **"含细节的干净参考"**（例如 `f_enc` 减去其低频版本）
- 或者 skip connection 从 `f_enc` 直接旁路（identity mapping）

现在的 `1×1 bypass` 实际上是**又一个 correction 头**，只是感受野更小。两个头都从相同的混合输入学习相同的目标 `correction`，只是空间尺度不同。这不是 detail preservation，是 **redundant correction**。

### 数学证明
设 $x = [f_{enc}, F_{denoised}, d]$。你的 correction:
$$c = (\phi_{3\times3}(x) + \phi_{1\times1}(x)) \cdot \gamma \cdot (1-d)$$

由于两个网络都零初始化，训练早期 $c \approx 0$；训练后两者都拟合同一残差目标 $r$，最优解是**任意分配** $\phi_{3\times3} + \phi_{1\times1} = r$。这不提供额外表征能力——只是把一个 3×3 网络的容量拆成 (3×3, 1×1) 两部分。

### 建议修改
若真想要 "pointwise detail highway"，应该是：

```python
# 真正的 highway：从原始 f_enc 提取 detail residual
detail_residual = f_enc - F_denoised  # 时序聚合去掉了什么？就是"帧间不一致 + 高频细节"
# 用 1×1 gate 决定 detail_residual 中哪些通道要保留
detail_gate = torch.sigmoid(self.detail_gate_1x1(detail_residual))  # (B, C, H, W)
preserved_detail = detail_residual * detail_gate * detail_map  # 高梯度区保留

correction = self.corr_spatial(correction_input) * self.gamma * (1.0 - detail_map)
f_noise_out = F_denoised + correction + preserved_detail + self.noise_proj(s_noise)
```

这才是符合 "detail 从 f_enc 高速旁路 F_denoised" 的物理直觉。

---

## 🔴 严重问题 2：`coarse_proj` 的双重矛盾

### 代码
```python
coarse_hint = self.coarse_proj(F.interpolate(l2_center, size=(H, W), ...))
correction = correction + coarse_hint * self.gamma * 0.5
```

### 问题 A：`coarse_hint` 是 H/2 上采样，本质上是**低频/粗尺度**信息
但它被加到 `correction` 里，而 `correction` 又乘以 `(1.0 - detail_map)`——等一下，让我再看代码：

```python
correction = (corr_spatial + corr_pointwise) * self.gamma * (1.0 - detail_map)  # 已经乘过detail门控
if l2_feats is not None:
    correction = correction + coarse_hint * self.gamma * 0.5  # coarse_hint 未经 detail 门控
```

**`coarse_hint` 没有经过 `(1-detail_map)` 门控**。这意味着在**纹理区（detail_map≈1）**，spatial/pointwise correction 被压制为 0，但 `coarse_hint` 仍然全强度注入。

**这是反的**：粗尺度信息（低频）应该更需要在平坦区注入（帮助全局去噪），在纹理区反而会**模糊细节**。

### 问题 B：`l2_lat` 本身就是 encoder 的中间层，不是 "coarse denoised prior"
`l2_lat[:, center_idx]` 只是中心帧的 H/2 特征——它是**含噪的、单帧的**表示。作为 "coarse guidance" 完全没有物理依据。

真正的 coarse guidance 应该是：
- 时序聚合后的粗尺度版本（例如对 `F_denoised` 做 avgpool→conv→upsample）
- 或者对 `F_aligned_list` 在 H/2 尺度做加权平均后上采样

### 建议
```python
# 从 F_denoised 生成真正的 coarse prior
coarse_prior = F.avg_pool2d(F_denoised, 2)         # H/2 平滑
coarse_hint = self.coarse_proj(F.interpolate(coarse_prior, size=(H,W), mode='bilinear'))
# 注入平坦区（低 detail），与主 correction 走一致门控
correction = correction + coarse_hint * self.gamma * 0.5 * (1.0 - detail_map)
```

---

## 🔴 严重问题 3：MCPN 的 `1×1 bypass` 逻辑混乱

### 代码
```python
motion_input = torch.cat([f_center, f_omega_aligned], dim=1)  # (B, 2C, H, W)
motion_delta = self.motion_refine_spatial(motion_input) + self.motion_refine_1x1(motion_input)
```

### 问题
`f_center` 和 `f_omega_aligned` 都是**已经通过 window correlation 对齐**的特征。它们的差异 `(f_omega - f_center)` 反映的是**未被 attention 完全对齐的运动残差**。

- 3×3 spatial：合理，捕获亚像素位移的空间模式
- 1×1 pointwise：**在这里毫无意义**。运动本质上是**空间位移**，pointwise 无法表达任何运动信息。1×1 只能做通道混合。

### 与 NDPN 的关键区别
- NDPN 处理噪声：噪声有 pointwise 成分（sensor noise 是 per-pixel），1×1 有意义（虽然如上面分析，实现方式不对）
- MCPN 处理运动：运动是纯 spatial，1×1 = 无操作

`motion_refine_1x1` 从零初始化开始，最终只会学到一个额外的 pointwise 通道 remap，与 `motion_refine_spatial` 的 3×3 中心权重耦合冗余。

### 建议
**移除** `motion_refine_1x1`。如果要增强 motion refinement 的表达力，应该是：
- 扩大 3×3 到 5×5（捕获更大位移）
- 或者加 dilation（多尺度 receptive field）
- 或者 depthwise 3×3 + pointwise 1×1（真正的 mobile-style separable）

---

## 🟡 中等问题 4：`detail_map` 只用图像梯度，忽略了噪声的高频性

```python
detail_raw = self._image_gradient(image_center)
detail_map = self.detail_proj(detail_raw)
```

在**低光图像**上，`image_center` 本身充满噪声。图像梯度会把**噪声也识别为"细节"**——尤其在低光暗区。这导致 `detail_map` 在噪声重的地方也接近 1，correction 被压制为 0，噪声得不到清除。

### 建议
用 `F_denoised` 的梯度（时序聚合后的干净版本）：

```python
# 从 F_denoised 反投影到图像空间的近似梯度
denoised_gray = F_denoised.mean(dim=1, keepdim=True)  # (B,1,H,W)
gx = (denoised_gray[..., 1:] - denoised_gray[..., :-1]).abs()
gy = (denoised_gray[:, :, 1:] - denoised_gray[:, :, :-1]).abs()
detail_raw = F.pad(gx,(0,1,0,0)) + F.pad(gy,(0,0,0,1))
detail_map = self.detail_proj(detail_raw)
```

这样 detail_map 反映的是"时序一致的高频"——真正的纹理，而不是随机噪声。

---

## 🟡 中等问题 5：`gamma_raw.clamp(max=0.03)` 与 correction 累加的双重压制

现在 correction 路径是：
$$c = [\phi_{3\times3}+\phi_{1\times1}](x)\cdot \underbrace{\gamma}_{\leq 0.03}\cdot(1-d) + \phi_{coarse}\cdot\gamma\cdot 0.5$$

$\gamma \leq 0.03$，且再乘 $(1-d)$，纹理区 $(1-d)\approx 0.1$，实际 correction 强度 $\leq 0.003$。这**几乎等于让 correction 分支不生效**。

从 m3 → m4，你增加了两条 correction 路径（1×1 bypass + coarse），但总体乘子没变，等于把同样的"预算"分给三条路径，每条路径的实际表达力反而**下降**。

### 建议
- 要么放开 gamma clamp（例如 0.1）
- 要么给不同路径独立的 gamma（`gamma_spatial`, `gamma_pointwise`, `gamma_coarse`），让模型自主分配

---

## 🟡 中等问题 6：`refine` 在 `F_denoised` 上的位置

```python
F_denoised = self.refine(F_denoised)  # 时序聚合后过 refine
# ...
f_noise_out = F_denoised + correction + self.noise_proj(s_noise)
```

`refine` 是 3×3 + 1×1 conv。它作用于**已经加权平均**的 `F_denoised` 上。但注意 `refine` 的第二个 conv 是 `act=False`，且没有 skip connection——这意味着 `refine` **改变了 F_denoised 的分布**。

后续的 `correction = f(f_enc, F_denoised, detail_map)`，其中 `F_denoised` 已经被 refine 变换过。`correction` 学的是"refined F_denoised 到 target 的残差"。

如果 refine 学到接近 identity（合理，因为 F_denoised 已经很好），一切正常。但没有 residual connection 保底，refine 完全依赖学习——早期训练可能破坏 F_denoised。

### 建议
```python
F_denoised_refined = F_denoised + self.refine(F_denoised)  # 加 skip
```

---

## 📋 总结与优先级

| 优先级 | 问题 | 建议 |
|:--:|------|------|
| **P0** | NDPN 1×1 bypass 逻辑错误 | 改为从 `f_enc - F_denoised` 提取 detail residual + gate |
| **P0** | MCPN 1×1 bypass 无物理意义 | 移除，或改为 depthwise 3×3 + 1×1 分离卷积 |
| **P1** | coarse_proj 输入源错误 + 门控缺失 | 用 F_denoised avgpool 作粗尺度先验，加 `(1-detail_map)` 门控 |
| **P1** | detail_map 被噪声污染 | 改用 F_denoised 的梯度而非 image_center |
| **P2** | gamma 总量预算不足 | 各路径独立 gamma，或放开 clamp 到 0.1 |
| **P2** | refine 无 skip | 加残差连接保底 |

## 🎯 核心洞察

Mark4 的三个"新增"（NDPN 1×1、l2 coarse、MCPN 1×1）看起来是**结构扩容**，但实际上：

1. **NDPN 1×1** 与 3×3 学同一目标（因输入相同、监督相同）→ 无效扩容
2. **l2 coarse** 引入了错误的信息源（含噪单帧）+ 错误的门控（未跟随 detail_map）→ 可能有害
3. **MCPN 1×1** 违背运动的空间本质 → 无效扩容

**m3 → m4 的 PSNR 提升如果发生，很可能来自 gamma_raw 三路共享带来的隐式正则化**（每条路径个体权重更小），而不是设计意图上的"细节保留 + 多尺度"。

建议先按 P0 修复，看是否能突破 m3 的 18.27。如果修复后反而下降，说明 m3 已经是当前架构的局部最优，需要考虑更大的结构调整（例如去掉多余分支、简化 NDPN 只保留 F_denoised + minimal correction）。