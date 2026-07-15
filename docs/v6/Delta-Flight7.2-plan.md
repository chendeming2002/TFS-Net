# Flight 7.1 深度诊断与 WFR 分流方案评估

---

## 第一部分：Flight 7.1 Epoch 40 指标分析

---

### 一、核心进展与遗留问题

Flight 7.1 相对于 Flight 7 确实取得了实质性进步：

| 修复项 | Flight 7 | Flight 7.1 | 判定 |
|:------:|:--------:|:----------:|:----:|
| gain 稳定性 | 0.83（崩溃） | **1.20 全程稳定** | ✅ 根因 1 已修复 |
| gain 空间性 | 无数据 | **std=0.63** | ✅ 曲线具有空间自适应能力 |
| 峰值 PSNR | 13.77 | **16.52** | ✅ +2.75 dB |
| ds_factor 影响 | 毁灭性 | 修复（回到 4） | ✅ |

**但产生了一个全新的、同样严重的问题——NDPN/MCPN 完全冻结（gn=gm=0.010，40 epoch 无变化）。** 三源分工框架在实际训练中退化为"ISPN 独揽一切"。

---

### 二、逐指标深度分析

#### 2.1 NDPN/MCPN gamma 零学习：detach 从"保护"变成了"扼杀"

```
ep1:  gn=0.010, gm=0.010
ep10: gn=0.010, gm=0.010
ep20: gn=0.010, gm=0.010
ep30: gn=0.010, gm=0.010
ep40: gn=0.010, gm=0.010    ← 40个epoch，零变化
```

**这是 Flight 7.1 最关键的失败信号。** 对比各版本：

| 版本 | gn 变化幅度 | gm 变化幅度 | NDPN/MCPN 状态 |
|:----:|:----------:|:----------:|:--------------:|
| Flight 6 | 0.010 → 0.0076（变化中） | — | ⚠️ 被压缩但在学习 |
| Flight 7 | 0.050 → 0.010（坍缩80%） | 0.050 → 0.015 | 🔴 被过度压缩 |
| **Flight 7.1** | **0.010 → 0.010（零变化）** | **0.010 → 0.010** | **❄️ 完全冻结** |

**我的诊断**：DeepSeek V4 的分析完全正确——`f_noise_out.detach()` + `residual_head` 零初始化 + `scale=0` 构成了**三重梯度阻断**：

```
NDPN 输出
  ├→ f_noise_gated.detach() → S1/S2    ← 路径1: detach 截断
  └→ residual_head (zero_init, scale=0) ← 路径2: 零输出，零梯度
                                         
结果：∂L/∂γ_n = 0（完全无梯度信号抵达 NDPN）
```

Flight 6 的"问题"（gamma 被 L_lit 压缩）**反而说明 NDPN 至少有梯度在流动**。Flight 7.1 的"修复"走了极端——完全切断了所有梯度通道，NDPN/MCPN 变成了永远不会被唤醒的死模块。

#### 2.2 Phase 2 持续退化：-0.82 dB 的异常

```
Phase 1.5 end (ep30):  PSNR=16.51, SSIM=0.581, LPIPS=0.432
Phase 2 early (ep40):  PSNR=15.69, SSIM=0.503, LPIPS=0.501
                        ↓-0.82 dB   ↓-0.078     ↑恶化 16%
```

对比 Flight 6 Phase 2：
```
Flight 6 ep30: PSNR=16.65 → ep40: 16.83 (+0.18 dB, 平稳微升)
Flight 7.1 ep30: PSNR=16.51 → ep40: 15.69 (-0.82 dB, 持续下跌)
```

**我的解读**：这不是简单的"Phase 2 损失项设置不当"，而是**整个网络能力瓶颈暴露**的表现：

1. Phase 2 加入感知损失（SSIM、LPIPS等），这些损失对**高频细节和结构保持**提出了更高要求
2. 但在 Flight 7.1 中，唯一能学习的模块就是 ISPN + Encoder 主干
3. ISPN（增益+曲线）的数学形式决定了它**只能做逐像素色调映射**——它无法修复结构、无法去噪、无法去模糊
4. Phase 2 的感知损失"要求"网络做结构恢复，但能做这件事的 NDPN/MCPN 已经冻结
5. 唯一能响应的 Encoder/ISPN 被迫在"维持亮度正确"和"改善结构"之间做不可能的妥协——**两者皆失**

这正是 [VSRELL (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) 中强调的问题："传统的串行处理管线中，如果光照增强和噪声抑制不作为耦合问题联合建模，则**误差会在后续阶段传播并累积**"。VSRELL 的 INCO 模块明确将光照分支和噪声分支通过**交叉调制融合**来协同工作——而 Flight 7.1 的 detach 策略恰好阻止了这种协同。

#### 2.3 SSIM 的 V 形但未恢复模式

```
ep10: 0.666 → ep20: 0.501 → ep30: 0.581 → ep40: 0.503
      峰值      跳水(-0.165)    回升(+0.080)   再次下跌(-0.078)
```

ep20 的 SSIM 跳水（-0.165）发生在 Phase 1.5 解锁点——即使有 detach 保护，Phase 切换时新的损失项仍然扰动了网络平衡。ep30 部分恢复，但 ep40 的 Phase 2 再次打破平衡。

**这暴露了分阶段训练的根本脆弱性**：每次 Phase 切换都是一次"小型灾难"，网络需要若干 epoch 来重新适应。在 50 epoch 的总预算中，**至少 10-15 epoch 浪费在了 Phase 切换的恢复期上**。

#### 2.4 LPIPS 持续恶化：最令人担忧的信号

```
ep10: 0.360 → ep20: 0.467 → ep30: 0.432 → ep40: 0.501
              恶化+0.107    小幅回升     再次恶化+0.069
```

LPIPS 从 ep10 到 ep40 净恶化 +0.141（39%↑）。这意味着**感知质量在整个训练过程中持续退化**。即使 PSNR 在某些节点有回升（ep30=16.51），感知质量从未恢复到 ep10 的水平。

**对比**：Flight 6 的 LPIPS 在 ep40 为 0.352——**比 Flight 7.1 好 30%**。这说明 Flight 6 虽然 PSNR 没有大幅突破，但至少保持了合理的感知质量。Flight 7.1 的激进架构修改反而损害了基本的图像质量。

---

### 三、Flight 7.1 vs Flight 6 的核心对比

| 维度 | Flight 6（ep40） | Flight 7.1（ep40） | 哪个更好 |
|:----:|:----------------:|:-----------------:|:--------:|
| PSNR | **16.83** | 15.69 | Flight 6 |
| SSIM | **0.717** | 0.503 | Flight 6（大幅领先） |
| LPIPS | **0.352** | 0.501 | Flight 6（大幅领先） |
| gain | 1.30 | 1.20 | Flight 6（更积极的提亮） |
| NDPN | 被压缩但在学习 | ❄️ 完全冻结 | Flight 6 |
| 训练稳定性 | 平台期 | 持续退化 | Flight 6 |

**残酷的结论**：Flight 7.1 在理论框架上正确，在实现上解决了 Flight 7 的两个根因，但**最终效果全面弱于 Flight 6**。这说明还有更深层的问题未被解决。

---

### 四、根因层级（更新版）

```
第一层根因：NDPN/MCPN 梯度完全断流
  └→ detach + zero_init + scale=0 三重阻断
     └→ 三源分工退化为"ISPN独揽"
        └→ ISPN 数学形式只能做色调映射
           └→ 无法回应 Phase 2 的结构/感知质量要求
              └→ PSNR/SSIM/LPIPS 持续退化

第二层根因：Phase 切换的损失冲击
  └→ Phase 1 → 1.5 → 2 每次都引入新损失项
     └→ 网络反复"被打乱又恢复"
        └→ 有效训练 epoch 被大量浪费
```

---

## 第二部分：WFR 分流方案评估

---

### 五、你的修改方案

> WFR 输出到 MCPN、NDPN 两个模块的是**分流后的特征**，输出到 DPE 仍为**完整的特征**（也不是输入到 TCA 的滤波特征）。

```
                    ┌─→ DPE（完整特征）
     Encoder ──→ WFR ├─→ NDPN（分流特征 A：噪声相关）
                    └─→ MCPN（分流特征 B：运动相关）
                    
                    ↓ 另一路（滤波后）
                    TCA
```

---

### 六、我的评价：方向正确，但不充分

#### ✅ 同意的部分

**6.1 DPE 保留完整特征——完全正确**

[DeAltHDR (ICLR 2026)](https://en.papernotes.org/ICLR2026/image_restoration/dealthdr_learning_hdr_video_reconstruction_from_degraded_alternating_exposure_se/) 明确证明了这一点：

> *"Short-exposure frames have heavy noise, while long-exposure frames have heavy blur; the nature of these degradations is fundamentally different. This paper provides two identical encoders with **completely independent parameters** for short and long exposures, allowing them to specialize in extracting features under their respective degradations."*

退化类型不同 → 特征提取策略不同。DPE 需要做全局空间推理（位姿/密度估计），这要求输入特征保留完整的空间结构和高频细节。分流后的特征可能丢失了 DPE 所需的关键线索。

[VSRELL (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) 的 ISFP 模块也强调：光照敏感的特征传播需要**完整的光照先验**（illumination map）来指导偏移预测——如果 DPE 获得的是分流后缺乏光照信息的特征，其估计精度会下降。

**6.2 MCPN/NDPN 接收分流特征——合理的归纳偏置**

[Fast-Slow Dual Branches (2026)](https://doi.org/10.20944/preprints202604.1276.v1) 的核心理念是：

> *"The video signal is decomposed into two complementary feature streams: a Slow branch with sparse temporal sampling and high spatial resolution... which focuses on long-range temporal denoising and high-frequency texture restoration; and a Fast branch with dense temporal sampling and low spatial resolution... which efficiently captures large-scale motion and rapid illumination changes."*

这正是"不同任务需要不同特征子空间"的直接证据。为 NDPN 提供噪声敏感特征、为 MCPN 提供运动敏感特征，本质上就是在编码层面施加任务感知的归纳偏置——让每个模块不需要自行从复杂的完整特征中解耦所需信息。

[HGD-Net (2026)](https://www.sciencedirect.com/science/article/abs/pii/S0020025526005979) 同样采用了这种范式：

> *"HGD-Net explicitly separates HV-plane completion from intensity-based structure recovery, applying a **generative chromatic prior** to the former and a **discriminative structural backbone** to the latter."*

不同退化类型 → 不同的归纳偏置 → 不同的特征输入。这是 2026 年低光增强领域的共识方向。

---

#### ⚠️ 不充分的部分：仅改 WFR 无法解决核心瓶颈

**6.3 WFR 分流不能解决梯度断流问题**

即使 NDPN 接收到了完美的"噪声特征"，如果没有梯度信号告诉它**如何利用这些特征**，它仍然会保持冻结状态。

```
当前：
WFR(分流) → NDPN → f_noise_out.detach() → S1/S2  ← ∂L/∂NDPN = 0
                   → residual_head(scale=0)         ← ∂L/∂NDPN = 0
                   
改进后仅改WFR：
WFR(更好的分流) → NDPN → f_noise_out.detach() → S1/S2  ← ∂L/∂NDPN 仍然 = 0 !!!
                        → residual_head(scale=0)         ← ∂L/∂NDPN 仍然 = 0 !!!
```

**更好的输入不等于有了学习能力。** 没有梯度，NDPN 的权重永远不会更新——不管它收到什么特征。

---

### 七、必须配套的关键修改：为 NDPN/MCPN 恢复梯度通路

结合 2026 年文献中的最佳实践，我提出以下配套方案：

#### 方案 A：辅助监督损失（**强烈推荐**）

受 [VSRELL](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) 的 INCO 模块启发——光照分支和噪声分支**各自有独立的优化目标**，然后通过交叉调制融合：

```python
# NDPN 辅助损失：多帧时域一致性（利用噪声帧间独立的先验）
L_denoise_aux = ||mean(aligned_frames) - NDPN_output||_1

# MCPN 辅助损失：光流一致性/运动补偿质量
L_motion_aux = ||warp(frame_t-1, flow) - MCPN_compensated||_1
```

**优势**：
- 绕过 detach 阻断——直接对 NDPN/MCPN 输出施加监督
- 每个模块有明确的优化目标，不依赖 Stage A/B 的反传
- 与 Stage A 的 L_lit 不冲突（目标正交）

**文献支持**：[DL-Diff (Scientific Reports 2026)](https://www.nature.com/articles/s41598-026-44219-8) 的三阶段训练中，每个阶段的子网络都有**独立的损失函数**：

> *"Each stage is trained separately... the overall decreasing trend and eventual leveling of each curve indicate that the objective is learned stably within that stage."*

#### 方案 B：residual_head scale 从 0 渐进解锁

```python
# 当前：scale = 0（永远关闭）
# 建议：scale = min(1.0, (epoch - warmup_epoch) / ramp_epochs)
residual_scale = torch.clamp((current_epoch - 20) / 10, 0.0, 1.0)
```

这给了 Stage B 残差路径一个**从零渐进到全开**的过程，让 NDPN/MCPN 在 ep20-30 之间逐步获得通过 L_final 回传的梯度。

#### 方案 C：软 detach（gradient scaling）

```python
# 当前：f_noise_gated = f_noise_out.detach() * unlock
# 建议：用梯度缩放替代二值detach
class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

# 使用：scale从0.0渐进到0.1（不是1.0——仍然限制回传强度）
f_noise_gated = GradScale.apply(f_noise_out, grad_scale) * unlock
```

**这允许 L_lit 的少量梯度（10%）流回 NDPN/MCPN**——既避免了 Flight 6 的"全额回传→压缩"问题，又避免了 Flight 7.1 的"零回传→冻结"问题。

---

### 八、综合修改建议优先级

| 优先级 | 修改 | 解决什么问题 | 风险 | 预期收益 |
|:------:|:-----|:------------|:----:|:--------:|
| **#1** | WFR 分流（你的方案） | DPE/NDPN/MCPN 信息需求差异 | 低 | NDPN/MCPN 输入质量提升 |
| **#2** | NDPN/MCPN 辅助损失 | **梯度断流（核心瓶颈）** | 低 | 模块从冻结恢复学习 |
| **#3** | residual_scale 渐进解锁 | Stage B 残差路径从零到开 | 低 | L_final 梯度可达 NDPN/MCPN |
| #4 | 软 detach (grad_scale=0.1) | L_lit 微量信号回传 | 中 | 进一步协同优化 |
| #5 | Phase 2 损失预热 | Phase 切换冲击 | 低 | 减少恢复期浪费 |

**#1 和 #2 必须同时实施**——单独的 WFR 分流无法解决梯度断流，单独的辅助损失在输入特征不够好时效果有限。两者组合才能真正让 NDPN/MCPN "看到正确的信号 + 知道如何利用"。

---

### 九、预期效果对比

| 指标 | Flight 7.1 ep40 | 预期（WFR+辅助损失） | 理由 |
|:----:|:--------------:|:-------------------:|:-----|
| PSNR | 15.69 | **17.0-17.5** | NDPN/MCPN 开始工作 + ISPN 减负 |
| SSIM | 0.503 | **0.62-0.68** | 运动补偿改善结构保持 |
| LPIPS | 0.501 | **0.38-0.42** | 去噪改善纹理质量 |
| gn | 0.010（冻结） | **0.012-0.018（增长）** | 辅助损失驱动学习 |
| gm | 0.010（冻结） | **0.012-0.020（增长）** | 辅助损失驱动学习 |

---

### 十、总结

#### 对 Flight 7.1 指标的核心见解：

1. **ds_factor 修复成功**（gain=1.20 稳定），但**detach 矫枉过正**（NDPN/MCPN 全冻结）
2. Phase 2 退化的本质是"ISPN 能力天花板"——逐像素色调映射无法满足结构/感知要求，而本应承担这些职责的模块已被冻结
3. Flight 7.1 的表现**全面弱于 Flight 6**——说明"完全隔离"不如"有控制的耦合"

#### 对 WFR 修改的评价：

| 维度 | 评分 | 说明 |
|:----:|:----:|:-----|
| 方向正确性 | **✅ 4.5/5** | 与 2026 年 CVPR/ICLR 趋势一致 |
| 充分性 | **⚠️ 2/5** | 仅解决输入质量，不解决梯度断流 |
| 必须配套的措施 | — | 辅助损失 + residual_scale 渐进 |

**最终建议**：实施 WFR 分流 + NDPN/MCPN 辅助监督损失 + residual_scale 渐进解锁，三者组合作为 **Flight 7.2** 的核心修改。

---

## 参考来源

1. [VSRELL: A Simple Baseline for Video Super-Resolution and Enhancement in Low-Light Environment (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf)
2. [Low-Light Video Enhancement via Fast–Slow Dual Branches and Flow-Guided Attention (Preprints 2026)](https://doi.org/10.20944/preprints202604.1276.v1)
3. [DeAltHDR: Learning HDR Video Reconstruction from Degraded Alternating Exposure Sequences (ICLR 2026)](https://en.papernotes.org/ICLR2026/image_restoration/dealthdr_learning_hdr_video_reconstruction_from_degraded_alternating_exposure_se/)
4. [HGD-Net: HVI-decoupled hybrid generative–discriminative network for low-light image enhancement (Information Sciences 2026)](https://www.sciencedirect.com/science/article/abs/pii/S0020025526005979)
5. [DL-Diff: Temporally consistent low-light face video enhancement via video-to-video conditional diffusion (Scientific Reports 2026)](https://www.nature.com/articles/s41598-026-44219-8)
6. [VLLVE++: Low-Light Video Enhancement with Spatial-Temporal Decomposition (arXiv 2602.08699)](https://arxiv.org/html/2602.08699)
7. [Low-Light Video Enhancement via Spatial-Temporal Consistent Decomposition (IJCAI 2025)](https://www.cse.cuhk.edu.hk/~byu/papers/C275-IJCAI2025-LLVE.pdf)
8. [DAF-IRNet: Degradation-aware feature disentanglement for task-conditioned all-in-one image restoration (Neurocomputing 2026)](https://www.sciencedirect.com/science/article/abs/pii/S0925231226016048)