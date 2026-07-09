# TSD-Net ISPN Mod4 ZeroDCE曲线增强改进分析

---

## 一、改进合理性分析

### 1.1 核心设计动机回顾

Mod4之前的ISPN面临的核心问题是：
- **DPE s_illum饱和至1.0** → ISPN失去调制信号
- **gain_head训练路径过长**（pix loss → res_t → SGRF S3 → gain_map的间接梯度）→ gain仅1.21×（需8-10×）
- **gain_map既要负责全局亮度提升又要负责空间精调** → 两个目标冲突

Mod4的解法是引入 **ZeroDCE曲线** 承担全局粗提亮，释放gain_map专注空间残差。

### 1.2 "全局曲线 + 空间残差" 分解的文献支持 ✅

根据联网检索（检索于2026-07-09），这一设计思路与多个方向高度吻合：

**（1）Zero-DCE原版** 提出 LE-curve：`LE(I(x); α) = I(x) + αI(x)(1 − I(x))`，具有三个关键性质——输出范围保证[0,1]、单调性保持对比度、形式简单可微分。[Zero-DCE CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)

**（2）LUT-GCE (2023)** 进一步提出 **全局曲线估计**：
> "To enable pixel-independent light enhancement and further lighten the computational burden, we propose a more flexible cubic curve formulation... a global curve estimation network (GCENet) with only 25.4k parameters"

— 这与Mod4用GAP压缩s_illum后预测per-image全局α的设计逻辑完全对齐。[LUT-GCE Paper](https://export.arxiv.org/pdf/2306.07083v1.pdf)

**（3）Recognition-Oriented LLIE (2025)** 提出 **GEM（全局增强）+ PAM（像素级调整）** 双模块架构：
> "GEM adjusts the brightness of the entire input image, producing a globally enhanced image. PAM estimates an optimal pixel-wise correction map."

Mod4的 `CurveBranch（全局曲线=GEM）+ gain_map（空间残差=PAM）` 正是这一范式在LLVE任务中的实例化。[Recognition-Oriented LLIE](https://arxiv.org/pdf/2501.04210)

**（4）DIME-Net (2025)** 使用 **MoE S-curve experts** 进行自适应亮度补偿：
> "16 S-curve enhancement networks with different parameters n... sparse gating mechanism based on average RGB values"

Mod4的做法更简洁（1条曲线3次迭代），但信息流结构类似：全局统计量→门控→曲线参数。[DIME-Net](http://arxiv.org/abs/2508.13921)

### 1.3 在TSD-Net整体架构中的合理性 ✅

从TSDR框架的角度分析：

| 要求 | Mod4满足情况 |
|------|-------------|
| **ISPN只负责illumination source** | ✅ ZeroDCE曲线本质是illumination correction，不影响noise/motion |
| **Phase 1零扰动** | ✅ α零初始化 → curve=identity → 四重零保证 |
| **SGRF物理顺序：denoise→deblur→brighten** | ✅ S2.5曲线位于S2(deblur)之后、S3(final brighten)之前 |
| **不引入噪声放大** | ✅ 曲线`I+αI(1-I)`在I<0.5时增益<2×，比直接gain×更温和 |
| **参数效率** | ✅ 仅增加9个参数（MLP: 1→16→9），不影响1.45M总参数 |

### 1.4 ZeroDCE曲线的数学优势

```
ZeroDCE: I' = I + α·I·(1-I)
         对暗区(I≈0): I' ≈ I(1+α) → 增益≈(1+α)
         对亮区(I≈1): I' ≈ I(1+α(1-I)) → 增益≈1 → 几乎不变

Retinex gain: I'' = I × gain
         对所有区域: 等比例缩放，亮区容易过曝
```

ZeroDCE曲线天然具有**暗区优先提亮、亮区保护**的物理意义，这在LLVE中正是需要的——低光图像的暗区需要大幅提亮但亮区（如路灯、天空）不应过曝。而简单的gain_map做不到这种亮度-非线性映射。

---

## 二、ISPN内部设计冗余分析

### 2.1 当前ISPN Mod4内部结构

```
输入: f_enc (B, 64, H, W), s_illum (B, 1, H/2, W/2)

refine: Conv(f_enc + s_illum, 65→64) → GELU → Conv(64→64) → GELU → h

├─ CurveBranch: s_illum → AvgPool(global) → MLP(1→16→9) → Tanh → α   [全局曲线]
├─ gain_head:   h → Conv(64→16) → GELU → Conv(16→1) → softplus → clamp [空间gain_map]  
└─ bias_head:   h → Conv(64→16) → GELU → Conv(16→3) → tanh×range       [空间bias_map]
```

### 2.2 冗余维度逐项分析

#### ❌ 冗余1：CurveBranch输入与refine路径无关

| 路径 | 输入 | 输出 | 信息流 |
|------|------|------|--------|
| refine → gain/bias | f_enc + s_illum | 空间变化的gain, bias | 含spatial context |
| CurveBranch | s_illum (直接) | 全局α | **跳过refine，与h无关** |

**问题**：CurveBranch直接从`s_illum`预测，完全不使用refine后的特征`h`。但`h`经过了`f_enc`（编码器特征）的空间信息融合。这意味着：
- CurveBranch只能获得DPE输出的**退化先验强度**，不知道当前帧的**实际空间内容**
- 如果s_illum饱和（之前的bug）→ CurveBranch输入恒定 → α恒定 → 失去自适应能力

**建议**：将CurveBranch输入改为 `h → GAP → MLP(64→16→9)`，这样曲线参数既包含全局亮度信息（来自s_illum已融入h），又包含内容感知信息（来自f_enc）。

#### ⚠️ 潜在冗余2：gain_map与曲线的功能边界

当曲线α足够大时（3次迭代后增益约 $(1+\alpha)^3 \approx 8\times$ 当α=1），全局亮度已被大幅提升。此时gain_map理论上只需做≈1.0的微调。但如果训练中曲线学习不足（α偏小），gain_map被迫重新承担全局提亮任务→回到Mod3的老问题。

**非冗余结论**：数学上两者不重叠（曲线是非线性、gain是线性），但存在**功能分配不稳定**的风险。需要L_gain_sup将gain_map的监督目标从`GT/I̅`调整为`GT/I_curved`（曲线处理后的目标比值），否则gain_map的监督目标没有考虑曲线的存在。

#### ✅ 非冗余：bias_head

bias_map做色差校正（∈[-0.1, 0.1]），与曲线/gain的功能完全正交。无冗余。

### 2.3 冗余总结表

| 组件对 | 冗余程度 | 理由 |
|--------|---------|------|
| CurveBranch vs gain_head | **低（数学正交）** | 曲线=非线性暗区优先；gain=线性等比例 |
| CurveBranch输入 vs refine路径 | **中等（信息冗余）** | 两路独立使用s_illum，CurveBranch不经过refine |
| S2.5位置 vs S3位置 | **无冗余** | 曲线在gain之前=非线性粗调后线性精调，逻辑正确 |
| gain_map + bias_map | **无冗余** | 乘性亮度调节 + 加性色差校正 |

---

## 三、What I Am Least Confident About 🔴

### 最不确定的点：**s_illum → AvgPool → MLP 的信息瓶颈是否过于严重**

**核心矛盾**：

1. **DPE的s_illum本身是空间图** `(B, 1, H/2, W/2)`，包含丰富的空间退化分布信息
2. CurveBranch立刻做**全局AvgPool** → 压缩为**单个标量** → 生成per-image的全局α
3. 对于非均匀光照场景（如：室内一盏台灯+大面积暗区），全局平均s_illum≈0.7，但：
   - 暗区真实需要α≈0.8（大幅提亮）
   - 灯下区域真实需要α≈-0.2（抑制高光）
   - 一个全局α≈0.5完全无法满足两个区域的需求

4. 虽然gain_map可以做空间补偿，但gain_map是**线性变换**——如果曲线在亮区提亮过度（因为全局α偏大），gain_map需要在亮区<1.0来压制，但clamp[1, max_gain]约束使gain≥1 → **亮区过曝无法通过gain_map修正**。

**进一步不确定**：

5. **3次迭代是否足够**？Zero-DCE用8次迭代，有些方法用6次。3次迭代在α∈[-1,1]时最大增益=(1+1)^3=8×。这对于极暗场景（需要16×-32×增益）可能不够。虽然gain_map补充了乘性增益，但两者的**配合训练稳定性**存疑——如果曲线快速收敛到8×，gain_map被压制；如果曲线学习过慢，gain_map又被迫承担大增益。

6. **s_illum饱和的老问题是否解决**？如果DPE的LayerNorm+零初始化修复仍未到位，s_illum→1.0 → AvgPool→1.0 → MLP(1.0)→某个固定α → CurveBranch退化为常数映射。文档中P0修复任务尚标记⬜。

### 次不确定的点：**L_gain_sup的监督目标未考虑曲线存在**

当前 `L_gain_sup = L1(gain_map, GT / I̅_t)`：
- 这个目标假设gain_map需要独立完成全部亮度提升
- 引入曲线后，实际监督目标应为 `GT / ZeroDCE(I̅_t, α_predicted)`
- 如果不修正，gain_map在训练前期被监督要求做大增益（因为loss不知道曲线的存在），与曲线产生梯度冲突

---

## 四、改进建议优先级

| 优先级 | 建议 | 理由 |
|--------|------|------|
| 🔴 P0 | **先解决DPE s_illum饱和问题** | CurveBranch依赖s_illum，饱和=无效 |
| 🟠 P1 | **L_gain_sup目标修正** 为 `GT / curve(img_s2, α)` | 消除gain与curve的监督冲突 |
| 🟡 P2 | **CurveBranch输入改为** `h → GAP → MLP(64→16→9)` | 利用refine后的内容感知特征 |
| 🟢 P3 | 评估 3 iter vs 5 iter 的效果差异 | 低风险实验，可后置 |
| 🟢 P4 | 放松 `gain clamp[1, max]` 为 `clamp[0.5, max]` | 允许gain在曲线过提亮区域压制 |

---

## 参考来源

1. [Zero-DCE: Zero-Reference Deep Curve Estimation (CVPR 2020 Paper)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)
2. [Zero-DCE++ Project Page](https://li-chongyi.github.io/Proj_Zero-DCE++.html)
3. [LUT-GCE: Lookup Table Global Curve Estimation](https://export.arxiv.org/pdf/2306.07083v1.pdf)
4. [Recognition-Oriented Low-Light Image Enhancement (GEM+PAM)](https://arxiv.org/pdf/2501.04210)
5. [DIME-Net: Dual-Illumination Adaptive Enhancement with MoE S-curves](http://arxiv.org/abs/2508.13921)
6. [Zero-DCE Keras Implementation](https://keras.io/examples/vision/zero_dce/)
7. [Zero-DiDCE: Rethinking Zero-DCE](https://link.springer.com/article/10.1007/s11063-024-11565-5)
8. [RICNET: Retinex-Inspired Illumination Curve Estimation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12390286/)
9. [Self-DACE++: Robust Low-Light Enhancement via Efficient Adaptive Curve Estimation](https://arxiv.org/html/2604.25367v2)
10. [Multi-stage Retinex + ViT Hybrid Framework (2026)](https://link.springer.com/article/10.1007/s00371-026-04485-z)