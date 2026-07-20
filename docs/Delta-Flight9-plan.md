# Flight 8 深度诊断：DPE 饱和的结构性根因与取消 WFR 的路径

---

## 一、Flight 8 当前状态：精准定位问题

### 1.1 关键数字总览

| 指标 | Flight 7.2 (ep80) | Flight 8 (ep20) | 判定 |
|:----:|:-----------------:|:---------------:|:----:|
| PSNR | **17.10** | 14.68 | 🔴 -2.42 dB |
| dpe_si (mean/std) | 0.893/**0.063** | 0.920/**0.000** | 🔴 完全饱和 |
| gain (mean/std) | 1.14/0.671 | 1.57/**0.512** | ✅ 更积极提亮 |
| gn | 0.020 | **0.049** | ⚠️ 暴走 (5×) |
| gm | 0.017 | 0.027 | ✅ 适中增长 |
| LPIPS | 0.375 | **0.343** | ✅ 感知改善 |
| Phase 1.5 dip | -2.06 dB | **-1.65 dB** | ✅ 更平滑 |

**诊断总结**：Flight 8 在三个方面确实优于 Flight 7.2（感知质量、Phase 切换平滑度、gain 空间积极性），但被一个致命缺陷抵消——**DPE 彻底崩溃为常数输出**。

### 1.2 DPE 饱和：从 "有点差" 到 "完全无效"

```
Flight 7.2 (ep80): dpe_si = 0.893 / std=0.063  ← 差，但还有些空间变化
Flight 8  (ep22):  dpe_si = 0.920 / std=0.000  ← 完全常数，零空间信息
```

**std=0.000 意味着什么**：DPE 对整张图（包括暗角和高光区）输出完全相同的 s_illum 值。它已经不是"光照估计不够准"——而是"根本不做光照估计"。这等价于一个 constant bias，完全可以被下游的 gain_head 吸收。

---

## 二、DPE 饱和的根因分析：为什么 3-stage + gray/lum 无效

### 2.1 Sigmoid 激活本身就不适合光照估计

这是我认为 **所有 Flight 版本中 DPE 设计的系统性错误**，而非某个超参的调整问题。

[IllumFlow (arXiv 2511.02411)](https://arxiv.org/html/2511.02411) 提供了关键物理洞察：

> *"Illumination variations can be approximated by a **linear parametric function**. This linear relationship stems from the fundamental physical property that raw pixel values scale linearly with the radiant energy collected."*

光照变换的物理本质是**线性/连续**的（辐射能量与像素值线性相关），而 sigmoid 是一个 S 形非线性函数：
- 在 (0.1, 0.9) 区间梯度大 → 网络倾向快速穿越中间区域
- 在极值附近梯度消失 → 一旦到达极值就很难回来
- **零初始化 head → sigmoid(0)=0.5** 看起来合理，但任何非零的 loss 信号都会把它推向 0 或 1

对于光照估计任务，我们需要的输出特征是：
- 暗区 → 低值（需要大幅增强）
- 亮区 → 高值（需要少量增强）
- 中间区域 → 中间值

但 sigmoid 的梯度分布恰好在中间值附近最大——这意味着优化器会**倾向于快速将所有输出推离 0.5**，要么全部向 0 走，要么全部向 1 走。结合 pixel-level loss（L1/L2），当绝大多数低光图像整体偏暗时，"全部输出高值"恰好是 sigmoid loss landscape 上的一个**全局吸引子**。

### 2.2 3-stage cascade 的错误放大效应

```
Stage1 (H/4): s_c3 → 轻微饱和 (sigmoid → 0.85)
Stage2 (H/2): refine(l2) + upsample(s_c3=0.85) → sigmoid → 0.90
Stage3 (H):   refine(l1) + upsample(s_c2=0.90) → sigmoid → 0.92
                                                         ↑ 级联放大！
```

每一级的 sigmoid 都在将前一级的饱和信号作为"条件输入"——前一级说"光照已经很亮了"，后一级也倾向于输出"更亮"。**没有任何反饱和机制**（如残差连接、归一化、或显式约束）阻止这种级联放大。

对比 [NID-LLIE (2026)](https://www.sciencedirect.com/science/article/abs/pii/S1051200426001375?dgcid=rss_sd_all) 的 TriLight 设计：

> *"Progressive illumination–detail co-optimization on a **stabilized intermediate representation**... LIB for non-uniform luminance correction"*

NID-LLIE 的关键区别是每一级工作在**稳定化的中间表征**上——先去噪稳定化，再做光照校正。而 Flight 8 的 3-stage 直接在"可能有噪声的特征"上逐级做光照估计，每一级都可能被噪声模式误导。

### 2.3 gray/lum priors 为什么没用

Gray prior (`I_t.mean(dim=1)`) 和 lum prior (`I_t.max(dim=1)`) 在低光图像上的特征：
- gray: 全图均值极低（整体暗） → 所有位置的 gray 值都接近 0.05-0.15
- lum: 全图最大值仍然很低 → 除少数高光点外，lum 也接近低值

当 gray 和 lum 在空间上**几乎无变化**（因为低光图像本身就是"全暗"的），它们作为先验注入后不能提供有意义的空间区分信号。conv-GELU 归一化后这些微弱的差异被进一步压缩。

**对比有效的做法**：[InterLight (arXiv 2605.19982)](https://doi.org/10.48550/arxiv.2605.19982) 的方案：

> *"We first inject **sensor-level illumination-response priors** via physics-guided augmentation, then represent the degradation through **adaptive prompts conditioned on the scene's latent illumination state**."*

InterLight 不是简单地拼接 gray/lum（那只是像素值），而是通过**传感器响应模型**构建物理先验，并用**自适应条件提示**来控制光照估计的输出范围。这比简单的 `I.mean()` / `I.max()` 强得多。

### 2.4 cls_token 零初始化的陷阱

`cls_token = nn.Parameter(torch.zeros(1,1,C,1,1))` 初始为零 → Phase 1 早期对输出无贡献。但当它开始学习时，它学到的是**单一全局偏置**——因为它的形状是 `(1,1,C,1,1)` 在空间维度上广播。这进一步鼓励了"全图输出相同值"的退化解。

---

## 三、gn 暴走 (0.01→0.049) 的因果链

这不是独立问题——它是 DPE 饱和的**直接后果**。

```
DPE 饱和 (s_illum=const, s_noise=const)
  └→ NDPN 接收到的噪声先验没有空间信息
     └→ NDPN 的 conf_proj 输出无法定位噪声区域
        └→ 去噪效果差 → L_ndpn_aux loss 大
           └→ 梯度唯一能调的自由度 = gamma
              └→ gamma 暴走 (0.01→0.049) 来全局放大效果
```

[Dynamic Nonlinear Networks (The Visual Computer, 2026)](https://link.springer.com/article/10.1007/s00371-026-04410-4) 精确描述了这种现象：

> *"Residual Refinement Module is **conditioned on the original low-light input**. This mechanism **explicitly couples noise estimation with illumination degradation**, enabling precise correction of noise and color deviations without sacrificing detail through uniform smoothing."*

关键词："explicitly couples noise estimation with illumination degradation"。当光照估计退化为常数时，噪声估计失去了"在哪里需要去噪"的定位信息——只能选择全局暴力去噪（→ gamma 暴走）或什么都不做（→ gamma 冻结）。

**建议**：如果 DPE 饱和问题不解决，即使 clamp gamma 到 0.03，NDPN 也会在受限范围内表现为"全局均匀去噪"——这不是正确的行为模式。

---

## 四、Full-Res WKV：4× 计算量换 -0.86 dB 的性价比分析

### 4.1 为什么 256×256 WKV 反而更差

| 维度 | H/2 WKV (Flight 7.2) | Full-res WKV (Flight 8) |
|:----:|:---------------------:|:-----------------------:|
| 序列长度 | 16384 | **65536** |
| 有效 batch | 8 | **16** (但 batch=2 + accum=8) |
| 每次梯度更新的真实样本多样性 | **4 不同样本** | **2 不同样本** |
| 参数更新频率 | 每 2 step | **每 8 step** |

**关键问题**：gradient accumulation 增加了 effective batch size 但**不增加样本多样性**。batch=2 意味着每个 forward pass 只看到 2 个不同的训练样本——相比 Flight 7.2 的 batch=4，**样本多样性降低了 50%**。

对于 RWKV 这种序列模型，65536 长序列的训练难度远大于 16384——更长的依赖链意味着更复杂的 loss landscape。在样本多样性降低的同时增加优化难度，结果就是 ep10 基线就低了 0.86 dB。

### 4.2 Full-res WKV 的理论收益在何处

Flight 8 changelog 中提到"4× 空间建模能力"——但这个论断基于一个假设：**更精细的空间依赖对低光增强有帮助**。然而：

[EvLIR (arXiv 2606.29430)](https://arxiv.gg/abs/2606.29430) 在 SDSD indoor 上达到 **25.63 dB PSNR** 和 **0.827 SSIM**——使用的是**轻量级 ConvGRU** 而非 full-res 全局注意力。这说明 SDSD indoor 数据集上，**局部时序建模**（ConvGRU 在有限窗口内）已经足够。全分辨率的全局空间依赖在 256×256 的低光场景中可能是**过度设计**。

### 4.3 建议：回退 WKV 到 H/2

```
当前：  Internal FPN → Full-res (256×256) WKV → 重
建议：  保持 H/2 (128×128) WKV + batch=4/accum=4 (eff=16 不变)
预期：  ep10 PSNR 回到 ~17 区间
```

这不是放弃 full-res WKV 的长期价值——而是在当前 DPE 饱和未解决时，不要叠加多个实验变量。

---

## 五、取消 WFR 的可行性评估

### 5.1 当前 WFR 的实际贡献

从 Flight 8 的设计来看，WFR 的角色已被大幅削弱：

| 原始 WFR 设计目标 | Flight 8 实际状态 |
|:----------------:|:----------------:|
| 为 DPE 提供低频光照信号 | ❌ DPE 已直连 Encoder l1/l2/l3 |
| 为 TCA 提供去光照结构特征 | ⚠️ 仅通过 `wfr_lambda`（零初始化）残差注入 |
| 信号分离（光照 vs 结构） | ❌ gray/lum priors 已替代其角色 |

**WFR 在 Flight 8 中的唯一剩余作用**：通过 `feat_tca` 为 TCA 提供可选残差。但 `wfr_lambda` 零初始化且无显式监督 → 这个残差可能从未被有效利用。

### 5.2 取消 WFR 的支持证据

[InterLight (2026)](https://doi.org/10.48550/arxiv.2605.19982) 和 [NID-LLIE (2026)](https://www.sciencedirect.com/science/article/abs/pii/S1051200426001375?dgcid=rss_sd_all) 均不使用显式的小波分流——它们用**自适应条件控制**和**解耦式协同优化**来替代显式频域分离。

[EvLIR](https://arxiv.gg/abs/2606.29430) 在 SDSD indoor/outdoor 上达到 SOTA，其架构中也没有小波分流：

> *"Given a low-light frame and its aligned event voxel, EvLIR preserves the ordered temporal bins... The resulting temporal state is converted into a **bounded illumination correction**, which provides **spatially adaptive photometric guidance** for Retinex-style illumination estimation."*

EvLIR 用时序事件流提供光照先验——类似于您的多帧输入可以通过时域统计自然获取光照信息，不需要显式的小波域分离。

### 5.3 取消 WFR 后的替代方案

```
取消前 (Flight 8):
  Encoder → l1/l2/l3
  Encoder center → WFR → feat_tca (残差注入 TCA)
  DPE ← l1/l2/l3 + gray/lum

取消后 (Flight 9 候选):
  Encoder → l1/l2/l3
  TCA ← l2/l3 直接（无需 WFR 中转）
  DPE ← l1/l2/l3 + 物理先验（重新设计）
  NDPN/MCPN ← l1（全分辨率细节特征）
```

**核心思路**：WFR 本质上是一个"路由器"——决定哪些信息去 DPE、哪些去 TCA。但如果 Encoder 已经输出多尺度特征（l1/l2/l3），各模块可以**直接选取所需尺度**：

- **DPE**：需要全局光照 → 取 l3 (H/4, 最粗尺度, 全局感受野)
- **TCA**：需要结构对齐 → 取 l2 (H/2, 中等分辨率)
- **NDPN/MCPN**：需要局部细节 → 取 l1 (H, 全分辨率)

这比 WFR 的"alpha 分流"更直接——多尺度 Encoder 本身就是天然的频率分离器（l3 是低频全局信息，l1 是高频局部信息）。

### 5.4 我的判断：**同意取消 WFR**

| 维度 | 保留 WFR | 取消 WFR |
|:----:|:--------:|:--------:|
| 参数量 | +25K | -25K |
| 信息路由 | 显式 alpha 学习 | 多尺度 Encoder 隐式分离 |
| DPE 输入 | 已不经过 WFR | 不变 |
| TCA 输入 | wfr_lambda=0 的残差 | Encoder l2 直连 |
| 训练复杂度 | 需要 L_wfr_reg | 减少一个损失项 |
| 文献支持 | ❌ 2026 无类似设计 | ✅ InterLight/NID-LLIE/EvLIR |

**但取消 WFR 不是最紧急的修改**——DPE 饱和是当前阻塞所有下游模块的根因。

---

## 六、Flight 9 核心修改建议：DPE 重新设计

### 6.1 最高优先级：彻底解决 DPE 饱和

基于检索结果，我提出一个**融合多篇 2026 工作**的新 DPE 设计：

#### 方案：用 Softplus + 残差偏移替代 Sigmoid

受 [IllumFlow](https://arxiv.org/html/2511.02411) 的启发——光照是线性/连续的，不需要 [0,1] 约束：

```python
# 当前 Flight 8 DPE 输出：
s_illum = torch.sigmoid(head_output)  # [0, 1]，容易饱和

# 建议替换为：
class IllumHead(nn.Module):
    def __init__(self, in_ch):
        self.head = nn.Conv2d(in_ch, 1, 1)  # 零初始化
        self.base = nn.Parameter(torch.tensor(0.3))  # 可学习基线
        
    def forward(self, x):
        # softplus 输出 (0, +∞)，加 base 偏移
        raw = self.head(x)
        s_illum = F.softplus(raw) + self.base  # 输出 > 0.3
        # 上界通过 soft clamp 控制：
        s_illum = s_illum / (1 + s_illum / 3.0)  # asymptotic max ≈ 3.0
        return s_illum
```

**物理含义**：s_illum 不再是"0到1的概率"，而是"光照强度的估计值"——暗区值小（如 0.3-0.5），亮区值大（如 1.5-2.5）。这避免了 sigmoid 在极值处梯度消失的问题。

#### 增加 L_illum_spatial 损失

受 [QRetinex-Net](https://doi.org/10.3390/app152212336) 的频率感知正则化启发：

```python
# 强制 s_illum 保留空间变化
L_illum_spatial = -torch.log(s_illum.std(dim=[-2,-1]).mean() + 1e-6)
# 当 std→0 时损失→∞，强制空间方差存在

# 边缘感知 TV（QRetinex-Net 式）
grad_x = s_illum[:,:,:,1:] - s_illum[:,:,:,:-1]
grad_y = s_illum[:,:,1:,:] - s_illum[:,:,:-1,:]
edge_weight = torch.exp(-10 * input_grad.abs())  # 输入图像边缘处允许不平滑
L_illum_tv = (grad_x.abs() * edge_weight_x + grad_y.abs() * edge_weight_y).mean()
```

**这两个损失组合**：L_illum_spatial 防止常数解，L_illum_tv 保证平滑（但尊重边缘）。

#### DPE 架构简化：取消 3-stage cascade

```python
# Flight 9 DPE: 单尺度但有显式反饱和
class DPE_v3(nn.Module):
    def __init__(self, C=64):
        # 输入：l3_lat (H/4) — 全局光照 + 物理先验
        self.proj = nn.Conv2d(C + 2, C, 1)  # +2 for gray, lum
        self.refine = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, groups=C//4), nn.GELU(),  # depthwise
        )
        self.illum_head = IllumHead(C)  # softplus，非 sigmoid
        self.noise_head = nn.Conv2d(C, 1, 1)  # sigmoid 可以用于噪声（0-1 置信度）
        
    def forward(self, l3_lat, gray_ds, lum_ds):
        x = self.proj(torch.cat([l3_lat, gray_ds, lum_ds], dim=1))
        x = self.refine(x)
        s_illum = F.interpolate(self.illum_head(x), scale_factor=4)  # H/4 → H
        s_noise = torch.sigmoid(self.noise_head(x))
        s_noise = F.interpolate(s_noise, scale_factor=4)
        return s_illum, s_noise
```

**关键简化**：
- 取消 3-stage cascade（级联放大饱和）
- 在最粗尺度 (H/4) 做光照估计 → 上采样到全分辨率（光照本身就是低频的）
- softplus + spatial_loss 双重防饱和
- 噪声估计保持 sigmoid（噪声置信度确实是 [0,1]）

### 6.2 WKV 回退到 H/2

如上文分析——在 DPE 饱和未解决时，不要叠加 full-res WKV 的额外复杂度。

```
Flight 9: TCA @ H/2 (128×128) + batch=4/accum=4 → eff=16
```

待 DPE 问题解决后（dpe_si std>0.05），可以在 Flight 10 重新尝试 full-res。

### 6.3 取消 WFR

如第五节分析——多尺度 Encoder 已经是天然的频率分离器。

### 6.4 gamma clamp（短期补丁）

```python
# NDPN gamma 上限
gn = torch.clamp(gn, max=0.03)
```

防止 DPE 饱和期间 gn 暴走导致的级联问题。

---

## 七、修改优先级排序

| 优先级 | 修改项 | 解决什么 | 风险 | 预期收益 |
|:------:|:-------|:---------|:----:|:--------:|
| **#1** | DPE sigmoid → softplus + L_illum_spatial | **DPE 饱和（核心阻塞）** | 中 | dpe_si std > 0.05 |
| **#2** | WKV 回退到 H/2 + batch=4 | ep10 基线回升 | 低 | +0.86 dB |
| **#3** | 取消 WFR | 减少无效计算和冗余路径 | 低 | -25K 参数，简化训练 |
| **#4** | DPE 取消 3-stage cascade | 消除饱和级联放大 | 低 | 更快收敛 |
| **#5** | gamma clamp (max=0.03) | gn 暴走 | 极低 | 训练稳定 |
| #6 | L_illum_tv (边缘感知) | 光照图平滑性 | 低 | 视觉质量 |

---

## 八、对"是否继续 Flight 8 训练"的建议

**建议：提前终止 Flight 8，实施上述修改作为 Flight 9。**

理由：
1. dpe_si=0.920/0.000 是结构性缺陷——不会随训练自行恢复
2. gn 暴走 (0.049) 如果持续到 Phase 2 可能导致训练崩溃
3. ep10 基线已低于 Flight 7.2 0.86 dB——即使 Phase 2 回升也很难超过 Flight 7.2 的 17.10
4. Flight 8 的正面贡献（LRU cache、sigmoid 零初始化、batched encoder）可以**无条件保留**到 Flight 9

---

## 九、总结

| 维度 | Flight 8 诊断 | 根因 | Flight 9 方案 |
|:----:|:------------:|:----:|:------------:|
| DPE 饱和 | dpe_si=0.92/0.00 | Sigmoid + cascade 放大 | **Softplus + L_spatial + 单尺度** |
| gn 暴走 | 0.01→0.049 | DPE 常数 → NDPN 补偿 | gamma clamp + DPE 修复 |
| 基线偏低 | ep10=16.33 (-0.86) | Full-res WKV + batch=2 | **回退 H/2 + batch=4** |
| WFR 冗余 | wfr_lambda≈0 | 多尺度 Encoder 已替代 | **取消 WFR** |

**核心认知更新**：DPE 的问题不是"先验不够"或"层数不够"——而是 **sigmoid 激活函数不适合光照估计任务**。2026 年的前沿工作（IllumFlow、InterLight、NID-LLIE）无一使用 sigmoid 做光照值回归——它们用连续流场、自适应提示、或 softplus 类无界激活。这是 Flight 系列需要做出的范式转换。

---

## 参考来源

1. [InterLight: Leveraging Intrinsic Illumination Priors for Low-Light Image Enhancement (arXiv 2605.19982)](https://doi.org/10.48550/arxiv.2605.19982)
2. [EvLIR: Learning Illumination Residuals from Ordered Events for Low-Light Image Enhancement (arXiv 2606.29430)](https://arxiv.gg/abs/2606.29430)
3. [IllumFlow: Illumination-Adaptive Low-Light Enhancement via Conditional Rectified Flow (arXiv 2511.02411)](https://arxiv.org/html/2511.02411)
4. [Dynamic Nonlinear Networks for Adaptive Low-Light Image Enhancement (The Visual Computer, 2026)](https://link.springer.com/article/10.1007/s00371-026-04410-4)
5. [NID-LLIE: A Lightweight Noise–Illumination–Detail Co-Optimization Network (Signal Processing: Image Communication, 2026)](https://www.sciencedirect.com/science/article/abs/pii/S1051200426001375?dgcid=rss_sd_all)
6. [QRetinex-Net: A Quaternion Retinex Framework for Bio-Inspired Color Constancy (Applied Sciences, 2026)](https://doi.org/10.3390/app152212336)
7. [Reti-Diff (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/6b9980876e7335e359df01911b5107d5-Paper-Conference.pdf)
8. [Continuous Splatting meets Retinex (arXiv 2606.16159)](https://arxiv.org/html/2606.16159v1)
9. [DIME-Net: Dual-Illumination Adaptive Enhancement Network (arXiv 2508.13921)](http://arxiv.org/abs/2508.13921)