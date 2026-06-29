# 三个架构设计点的深度讨论

## 实行状态 (2026-06-29)

| 设 计 | 简述 | 状态 | 变更文件 |
|--------|------|------|----------|
| D1 | s_noise 移除 SACE → NDPN 专属 | ✅ 已完成 | `pure_rwkv_sace.py` |
| D2 | IFPN 输入 encoder 特征 (非原始噪声帧) | ✅ 已完成 (Option A) | `tfs_net.py` L96-99 + L250-263 |
| D3 | IFPN 加 s_illum 调制门 | ✅ 已完成 | `ifpn.py` L103-106 + L163-167 |
| D4 | 单一 A_illu (VSRELL 风格) → IGRF | ✅ 已完成 (Option C 简化版) | `igrf.py` L119-140 |

训练启动: 加载 v6.5 pretrained → outputs/sdsd_v6_charlie2 (41 missing / 9 unexpected keys), loss 下降中

---

根据联网检索（检索于 2026-06-29T18:46:46+08:00）和您上传的代码/文档，以下对三个设计逐一展开。

---

## 设计点 1：s_noise（+phase_conf）输入到 SACE — 串行代价是否值得？

### 是否存在

**存在**。从 `tfs_net.py` 的 `PureRWKVSACE.forward` 可见：

```python
s_noise = tfsi_out.get("s_noise") if tfsi_out else None
# ...
f_t = out[:, t] + (1.0 - edge_weight) * f_raw_center
# s_noise 用于调制残差注入：(1-s_noise)·f_raw_center
```

TFSI 先完成 → 输出 s_noise → SACE 接收并使用，构成严格的**串行依赖链**。

### 提出目的

让 SACE 的帧间对齐具备"噪声感知"能力：高噪区域减少中心帧残差注入（因为中心帧本身也有噪声），避免将噪声当作"干净锚"传播到所有帧。

### 近年顶会论文参考

**遍查近两年顶会，未发现任何将噪声估计图直接馈入时序注意力/对齐模块的设计。** 主流策略如下：

| 论文 | 噪声估计与时序对齐的关系 | 架构模式 |
|------|------------------------|---------|
| [VSRELL (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) | 光照/噪声双分支**共享编码器**，在**解码阶段交叉调制融合** | **并行** |
| [RetinexMCNet (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_RetinexMCNet_A_Memory_Controller_Dominated_Network_for_Low-Light_Video_Enhancement_ICCV_2025_paper.html) | Stage 1 单帧增强（IAM+RDM），Stage 2 激活 MC 做时序一致性 | **两阶段解耦** |
| [MVSSM (CVPR 2026F)](https://openaccess.thecvf.com/content/CVPR2026F/papers/Zhou_MVSSM_Motion-aware_Visual_State_Space_Model_for_Efficient_Video_Deblurring_CVPRF_2026_paper.pdf) | 时序扫描仅依赖 optical flow 估计的运动方向，不接收噪声图 | **运动纯驱动** |
| [STCDiT (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_STCDiT_Spatio-Temporally_Consistent_Diffusion_Transformer_for_High-Quality_Video_Super-Resolution_CVPR_2026_paper.html) | 噪声通过扩散过程隐式处理，anchor-frame latent 仅提供结构先验 | **隐式** |

VSRELL 原文明确指出串行结构的害处：

> *"The two branches share encoder features and are fused through a cross-modulation mechanism in the decoding stage, **effectively avoiding error propagation in serial structures**."*

### 在问题建模中的地位

这是一个**辅助信号路由**设计，不是核心算法创新。s_noise 对 SACE 而言是"可选条件输入"，其作用是调制 `(1-s_noise)·f_raw_center` 的残差注入强度。从物理建模角度看，这意味着：

- 噪声大的区域 → s_noise 大 → `(1-s_noise)` 小 → 减少中心帧注入 → **矛盾**：高噪区域本应更依赖多帧平均去噪，但减少了帧间信息融合的参考锚
- 噪声小的区域 → s_noise 小 → 大量中心帧注入 → 合理但冗余（低噪区域本身不需要太多帧间补偿）

**这个设计处于"用错了位置"的状态**——s_noise 对去噪（NDPN）的指导价值远大于对帧间对齐（SACE）的指导价值。

### 权衡利弊

| 利 | 弊 |
|----|----|
| ✅ 理论上 RWKV 可区分噪声 vs 运动差异 | ❌ **级联误差放大**：单帧 FrequencyBranch 估计的 s_noise 本身不可靠（详见此前讨论），误差被 RWKV 放大传播到后续所有模块 |
| ✅ 为 SACE 提供额外先验 | ❌ **语义矛盾**：s_noise 大→减少中心帧注入→高噪区域反而失去了帧间平均去噪的锚参考 |
| | ❌ **梯度路径过长**：s_noise 的梯度需经 TFSI→SACE→NDPN→IGRF 反传，极易消失 |
| | ❌ **计算上串行瓶颈**：TFSI 必须完全完成后 SACE 才能开始，无法并行 |
| | ❌ **与 VSRELL 等 CVPR 2026 SOTA 的设计哲学明确相悖** |

**结论：代价远大于收益。建议将 s_noise 从 SACE 移除，改为 NDPN 的显式条件输入。SACE 应仅基于多帧特征自身做对齐，不依赖外部噪声估计。**

---

## 设计点 2：IFPN 输入 img_center

### 是否存在

**存在**。从 `tfs_net.py` 的 `IFPN` 和 `TFSNet.forward` 可见，IFPN 接收 `image_center`（原始 RGB 中心帧）作为输入之一。

### 提出目的

1. **Skip connection 理念**：为解码/重建阶段提供原始空间细节的快捷路径
2. **结构保持**：防止经过多层特征提取后中心帧的边缘/纹理信息完全丢失
3. **梯度辅助**：缩短从输出到输入的梯度路径，帮助训练收敛

### 近年顶会论文参考

| 论文 | 重建阶段是否输入原始帧像素 | 替代方案 |
|------|--------------------------|---------|
| [VSRELL (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) | **否**。原始帧仅用于**乘法增益** `I_bright = I_curr · Clamp(g·(1.5-α))` ，解码器输入是**多帧融合特征** | 乘法式增益 + 加法式去噪偏移 `I_denoise = Clamp(I_bright − O_denoise·M_noise)` |
| [STCDiT (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_STCDiT_Spatio-Temporally_Consistent_Diffusion_Transformer_for_High-Quality_Video_Super-Resolution_CVPR_2026_paper.html) | **否**。用 **VAE 编码后的 anchor-frame latent** 作为结构约束 | *"anchor-frame latent remains unaffected by temporal compression and retains richer spatial structural information"* |
| [RetinexMCNet (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_RetinexMCNet_A_Memory_Controller_Dominated_Network_for_Low-Light_Video_Enhancement_ICCV_2025_paper.pdf) | **否**。Stage 2 的 MC 接收的是 Stage 1 的**增强后特征**，非原始帧 | 编码→增强→MC时序融合→解码 |
| [MVSSM (CVPR 2026F)](https://openaccess.thecvf.com/content/CVPR2026F/papers/Zhou_MVSSM_Motion-aware_Visual_State_Space_Model_for_Efficient_Video_Deblurring_CVPRF_2026_paper.pdf) | **否**。输入经过 `local temporal fusion`（flow warp + ResBlock）后再进扫描模块 | 对齐后特征直接进解码器 |
| [DIVD (2024)](https://arxiv.org/html/2412.00773) | 输入是噪声+条件帧的**通道拼接**，但这是 diffusion model 的标准范式，非判别式网络设计 | 扩散模型特有 |

**核心观察**：**在判别式低光增强/去模糊网络中，没有一篇 SOTA 将原始有噪低光帧像素直接输入重建解码器。** VSRELL 的做法最有参考价值——用乘法增益处理原始帧，而非拼接输入。

### 在问题建模中的地位

IFPN 在您的数据流中承担**光照特征金字塔重建**角色：`IFPN → lit_up_map, f_illum_feat, ifpn_side`。它需要输出"增亮图"和"光照特征"。让它接收原始中心帧，等价于告诉 IFPN"这是未处理的低光图像，请参考它来生成增亮图"。

**问题在于**：原始中心帧是**低质量的**（噪声大、光照不足），直接作为参考会让网络学到一条"走捷径"——尤其当 VGG 感知损失权重高时（你的 Run 1 VGG=0.8），网络倾向于保留原始帧的结构以最小化感知差异，**代价是噪声也被保留**。

### 权衡利弊

| 利 | 弊 |
|----|----|
| ✅ 梯度快捷路径，训练初期稳定 | ❌ **噪声泄漏**：低光原始帧的 shot noise / read noise 通过 skip connection 直接注入重建结果 |
| ✅ 保留原始空间分辨率信息 | ❌ **训练捷径**：网络可能学会 `res ≈ img_center + 小偏移`，而非真正的增强与去噪 |
| ✅ 实现简单 | ❌ **与所有 2025-2026 SOTA 设计方向不符**：无一使用原始低质帧像素直接输入解码器 |
| | ❌ **阻碍去噪效果**：NDPN 辛苦去掉的噪声，可能通过 IFPN 的 img_center 路径被重新引入 |

**建议方案**（按推荐程度排序）：

1. **方案 A（推荐）**：将 `img_center` 替换为 **encoder 的浅层特征** `feats[:, T//2]`（已经过初步编码但保留空间细节）
2. **方案 B**：参考 VSRELL 的乘法式增益——不将原始帧拼接输入 IFPN，而是让 IFPN 输出一个增益图，通过 `res = img_center × gain_map + offset` 间接使用
3. **方案 C（最小改动）**：对 `img_center` 做一次 shallow conv（3×3 Conv + GELU）再输入 IFPN，至少让网络有机会过滤掉部分噪声

---

## 设计点 3：IFPN 无 s_illum 输入

### 是否存在

**存在**。从架构文档和代码看，`s_illum` 由 TFSI 的 `IntensityHead` 输出后：
- 用于 DWT-LFF 的 `illum_alpha` 间接控制
- 最终传入 IGRF 的 `BrightenStage`
- **但 IFPN 没有接收 s_illum**

### 提出目的

可能的设计意图是：
1. IFPN 的任务被定义为"从对齐后的特征重建光照特征金字塔"，认为 s_illum 信息已隐含在特征流中
2. 减少模块间耦合度，让 IFPN 独立工作
3. s_illum 被认为应在**最终重建阶段**（IGRF/BrightenStage）才使用

### 近年顶会论文参考

**在光照恢复/重建阶段显式引入光照图，是 2025-2026 的明确共识：**

| 论文 | 光照信息在重建中的使用方式 | 效果 |
|------|--------------------------|------|
| [VSRELL (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) | **最核心的设计之一**：光照图 `A_illu` 直接用于 deformable offset 调制 `∆O_fusion_k = ∆O_mod_k + F_smooth_k ◦ A_illu · (1 + A_illu)` | SDSD 上 PSNR 大幅领先 |
| [RetinexMCNet (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_RetinexMCNet_A_Memory_Controller_Dominated_Network_for_Low-Light_Video_Enhancement_ICCV_2025_paper.pdf) | **Illumination-guided RDM**：光照图直接引导反射分量去噪 | *"illumination-guided Reflectance Denoising Module (RDM) based on Retinex theory"*，SDSD 27.81 dB / DID 30.09 dB |
| [TempRetinex (2025)](https://arxiv.org/html/2511.09609v2) | 光照估计用于**时序反馈**——前帧光照 warp 到当前帧，作为重建辅助 | 时序光照一致性 |
| [Zero-TIG (2025)](https://arxiv.org/html/2503.11175) | 零样本光照引导：用估计的光照图显式引导增强过程 | 无需训练的光照引导 |

VSRELL 原文中关于光照图的具体用法极具参考价值：

> *"explicitly incorporating illumination information into the offset prediction and modulation processes. By explicitly modeling the relationship between illumination intensity and feature offsets, we introduce a normalized illumination map into the optical flow estimation network"*

> `A_illu = σ(BN(Conv2d(M_low, K=3, C_out=1)))` — 先从低光掩码生成光照注意力
> `F_smooth_k = F_k · (1-A_illu) + Blur(F_k, K_Gauss) · A_illu` — 低光区做高斯平滑
> `∆O_fusion_k = ∆O_mod_k + F_smooth_k ◦ A_illu · (1+A_illu)` — 光照调制偏移

RetinexMCNet 的消融实验也明确显示，IAM（光照调整模块）在 Stage 1 中贡献了 **+2.74 dB** 的提升，是所有模块中增益最大的。

### 在问题建模中的地位

**IFPN 承担的是"光照特征金字塔"的生成任务**——它负责输出 `lit_up_map` 和 `f_illum_feat`。然而这个专门负责光照重建的模块，却**没有接收光照估计信息**，这是一个显著的建模缺陷：

- IFPN 不知道当前场景**哪里暗、哪里亮**，只能从对齐后的特征中隐式推断
- 特征经过 Encoder → DWT-LFF → RWKV → NDPN 多层处理后，**原始光照的空间分布信息可能已大幅衰减**
- IFPN 生成的 `lit_up_map` 应该是"每个像素需要多大的增亮倍率"——这直接需要 s_illum 作为先验

### 权衡利弊

| 利 | 弊 |
|----|----|
| ✅ 架构简洁，IFPN 无额外输入依赖 | ❌ **职责与能力脱节**：IFPN 负责光照重建，但没有光照信息输入 — 这是设计上的"盲点" |
| ✅ 避免 s_illum 误差传播到 IFPN | ❌ **与 VSRELL/RetinexMCNet 等 SOTA 明确背离**：光照图引导重建已被证明是关键性能来源 |
| | ❌ **暗区过度平滑**：IFPN 无法区分"暗区细节"和"噪声"，倾向于对暗区均匀处理 |
| | ❌ **增亮图（lit_up_map）质量受限**：没有 s_illum 的空间引导，`lit_up_map` 可能生成不合理的增亮倍率（暗区增益不足/亮区过曝） |
| | ❌ **IFPN 监督信号不充分**：`L_ifpn_sup` 损失只有 0.1 权重，在缺少 s_illum 先验的情况下收敛困难 |

### 建议方案

参考 VSRELL 的 ISFP 机制和 RetinexMCNet 的 IAM 设计：

```python
class IFPN(nn.Module):
    def __init__(self, fused_channels, coarse_channels, img_channels):
        super().__init__()
        # ★ 新增: s_illum 调制层 (参考 VSRELL 的 A_illu 设计)
        self.illum_modulate = nn.Sequential(
            nn.Conv2d(1, fused_channels, 3, 1, 1),
            nn.Sigmoid()
        )
        # ... 原有组件 ...
    
    def forward(self, F_aligned, s_illum=None, ...):
        # ★ 光照感知调制
        if s_illum is not None:
            illum_gate = self.illum_modulate(s_illum)  # (B, C, H, W)
            # 低光区域: s_illum小 → gate值大 → 更激进的特征增强
            F_aligned = F_aligned * (1 + illum_gate * (1 - s_illum))
        # ... 原有金字塔重建 ...
```

这样 IFPN 的 `lit_up_map` 就有了空间自适应的光照先验引导。

---

## 三个设计点的系统性关联

三个问题不是孤立的，它们共同构成了一个**信息路由错位**问题：

```
当前设计（信息流错位）：
  TFSI ──s_noise──→ SACE (对齐) ← 噪声不该影响对齐
         ──s_illum──→ DWT-LFF (α)  ← 间接且微弱
  IFPN ← img_center (有噪原始帧)   ← 引入噪声
  IFPN ← 无 s_illum                 ← 光照模块缺光照信息

应有设计（参考 VSRELL/RetinexMCNet）：
  TFSI ──s_noise──→ NDPN (去噪)      ← 噪声信息服务于去噪
         ──s_illum──→ IFPN (光照重建) ← 光照信息服务于增亮
         ──s_illum──→ SACE (可选)     ← 光照引导对齐(VSRELL ISFP范式)
  SACE ← 纯多帧特征 (自行估计帧间差异) ← 解耦噪声依赖
  IFPN ← encoder特征 (非原始像素)     ← 避免噪声泄漏
```

**核心原则**（来自 VSRELL）：

> *"Through **joint modeling** of an illumination-sensitive branch and a noise estimation branch, end-to-end collaborative optimization is achieved... effectively avoiding error propagation in serial structures."*

**即：噪声图 → 去噪模块，光照图 → 增亮模块，各得其所，并行交叉而非串行堆叠。**

---

## 参考来源

1. [VSRELL: A Simple Baseline for Video Super-Resolution and Enhancement in Low-Light Environment (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf)
2. [VSRELL - CVPR 2026 Poster](https://cvpr.thecvf.com/virtual/2026/poster/36914)
3. [VSRELL - CVPR 2026 Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.html)
4. [RetinexMCNet (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_RetinexMCNet_A_Memory_Controller_Dominated_Network_for_Low-Light_Video_Enhancement_ICCV_2025_paper.html)
5. [RetinexMCNet - Full Paper (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_RetinexMCNet_A_Memory_Controller_Dominated_Network_for_Low-Light_Video_Enhancement_ICCV_2025_paper.pdf)
6. [STCDiT: Spatio-Temporally Consistent Diffusion Transformer (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_STCDiT_Spatio-Temporally_Consistent_Diffusion_Transformer_for_High-Quality_Video_Super-Resolution_CVPR_2026_paper.html)
7. [STCDiT - arXiv](https://arxiv.org/html/2511.18786)
8. [MVSSM: Motion-aware Visual State Space Model for Video Deblurring (CVPR 2026F)](https://openaccess.thecvf.com/content/CVPR2026F/papers/Zhou_MVSSM_Motion-aware_Visual_State_Space_Model_for_Efficient_Video_Deblurring_CVPRF_2026_paper.pdf)
9. [Zero-TIG: Temporal Consistency-Aware Zero-Shot Illumination-Guided LLVE (2025)](https://arxiv.org/html/2503.11175)
10. [TempRetinex: Retinex-Based Unsupervised Enhancement for Low-Light Video (2025)](https://arxiv.org/html/2511.09609v2)
11. [VRWKV-Editor: Reducing Quadratic Complexity in Video Editing (2025)](https://ar5iv.labs.arxiv.org/html/2509.25998)
12. [VideoFusion: Spatio-Temporal Collaborative Network (2025)](https://arxiv.org/html/2503.23359v2)
13. [DIVD: Deblurring with Improved Video Diffusion Model (2024)](https://arxiv.org/html/2412.00773)
14. [CMTA: Cross-Modal Temporal Alignment for Event-guided Video Deblurring (ECCV 2024)](https://arxiv.org/pdf/2408.14930)
15. [DiTVR: Zero-Shot Diffusion Transformer for Video Restoration (CVPR 2026 Workshop)](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/html/Gao_DiTVR_Zero-Shot_Diffusion_Transformer_for_Video_Restoration_CVPRW_2026_paper.html)
16. [Temporal-Consistent Video Restoration with Pre-trained Diffusion Models (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/37943)
17. [RetinexMCNet - ICCV 2025 Poster](https://iccv.thecvf.com/virtual/2025/poster/479)

# IFPN 光照建模：lit_up_map、f_illum、s_illum 的定位分析与简化可能性

根据联网检索（检索于 2026-06-29T18:58:29+08:00）和您上传的代码文档，以下对此设计点展开系统性讨论。

---

## 是否存在该设计

**存在，且涉及三条独立的光照信号路径。** 从 `tfs_net.py` 的数据流可追踪：

```
路径1: IFPN → lit_up_map_raw (像素级增亮图)
路径2: IFPN → f_illum_feat   (光照特征图)
路径3: TFSI → s_illum         (光照退化/闪烁掩码)

三者最终在 IGRF 的 BrightenStage 中汇合:
  res_t, lit_up_map = self.brighten(lit_up_map_raw, f_illum_feat, img_s2, s_illum=s_illum)
```

具体地，从代码可重构出 BrightenStage 的内部使用方式：

| 信号 | 来源 | 在 IGRF/Brighten 中的角色 |
|------|------|--------------------------|
| `lit_up_map_raw` | IFPN 的 `lit_up_proj` 输出 | 像素域增亮基底图 |
| `f_illum_feat` | IFPN 的 `feat_refine` 输出 | 特征域的光照修复残差 |
| `s_illum` | TFSI 的 `IntensityHead` 输出 | 调制增亮强度的标量/空间掩码 |

---

## 提出目的

设计者的意图是将光照恢复拆解为三个层次的分工：

| 信号 | 设计意图 | 语义层次 |
|------|---------|---------|
| **s_illum** | 回答"**该区域光照退化有多严重？**" — 是一个 0~1 的退化强度图 | **诊断层**：退化程度感知 |
| **lit_up_map_raw** | 回答"**该像素应该变多亮？**" — 是 RGB 域的增亮偏移/增益 | **处方层**：像素级增亮方案 |
| **f_illum_feat** | 回答"**增亮后的特征细节是什么？**" — 携带纹理/结构信息 | **内容层**：增亮后应有的纹理 |

从 Retinex 理论角度，这三者的对应关系：
- `s_illum` ≈ **光照图 L 的退化掩码**（哪里的 L 不对）
- `lit_up_map` ≈ **光照校正增益 L_corrected / L_degraded**
- `f_illum_feat` ≈ **反射率 R 经光照校正后的特征表示**

---

## 近年顶会论文参考

**遍查 2024-2026 顶会论文，未发现任何方法同时维护三个独立的光照相关输入。** 主流范式如下：

| 论文 | 光照建模策略 | 光照相关信号数量 | 具体实现 |
|------|------------|----------------|---------|
| [VSRELL (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) | INCO 光照-噪声协同优化 | **1 个**：`A_illu` | `A_illu = σ(BN(Conv2d(M_low)))` → 对齐偏移调制 + 特征平滑，**单一图同时指导增亮和对齐** |
| [Multinex (CVPR 2026)](https://github.com/albrateanu/multinex) | 多先验 Retinex 残差 | **1 组**：光照先验堆栈 | 从 HSV/YCbCr 等多色彩空间提取先验，但最终**融合为单一残差** |
| [ISALux (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/papers/Balmez_ISALux_Illumination_and_Semantics-Aware_Transformer_Employing_Mixture_of_Experts_for_WACV_2026_paper.pdf) | 光照先验金字塔 + 语义先验 | **1 个**：`P_i = 1 - argmax_c(I)` | 光照先验图直接注入每层 HISA-MSA 的注意力计算，**全程单一先验** |
| [M2Retinexformer (2026)](https://arxiv.org/html/2605.12556v1) | 多模态 Retinex | **1 个**：`L_p` | 光照先验图 `L_p` 只在入口处与 RGB 拼接，后续全部由交叉注意力融合 |
| [IS-SFD (2026)](https://openreview.net/forum?id=vc6pSwumm6) | GIE-Net 门控光照估计 | **1 个** | 多帧相似性引导的单一光照估计输出 |
| [TempRetinex (2025)](https://arxiv.org/html/2511.09609v2) | BCP + 时序反馈 Retinex | **1 个**：`R_0` | 初始反射率 `R_0 = γ·I`，后续 RE-Net 精化 |
| [Zero-TIG (2025)](https://arxiv.org/html/2503.11175v1) | 零样本光照引导 | **1 个** | IE-Net 估计光照 → 时序反馈 → RD-Net 去噪 |
| [FPN-Retinex (2024)](https://doi.org/10.23919/ccc63176.2024.10662161) | FPN 调整光照图 | **1 个** | 半监督：FPN 在不同尺度调整单一光照图 |

**最关键的对比——VSRELL 的极简设计**：

VSRELL 原文明确描述了它如何用 **单一 `A_illu`** 完成您的三个输入所做的全部工作：

> *"`A_illu ∈ R^{B×1×H×W}` serves as attention weights to enhance the feature contrast in LLR. `M_low = (1 − M_illu)²` is the mask weight and can non-linearly amplify the intensity of LLR."* — 这替代了您的 `s_illum`

> *"`F_smooth_k = F_k · (1 − A_illu) + Blur(F_k, K_Gauss) · A_illu`"* — 这替代了您的 `f_illum_feat`（光照感知的特征调制）

> *"`I_bright = I_curr · Clamp(g · (1.5 − α), 1, g_max)`"* — 这替代了您的 `lit_up_map`（乘法增益而非加法偏移）

**一张图完成了三件事**，而您用了三个独立分支。

---

## 在问题建模当中的地位

### 三者的功能边界模糊

从信息论角度分析这三个信号的**互信息**：

```
I(s_illum; lit_up_map) ≈ 高
  理由: 光照退化严重的区域 (s_illum 高) 必然需要更大的增亮增益 (lit_up_map 高)
        两者都是"暗区空间分布"的不同编码方式

I(lit_up_map; f_illum_feat) ≈ 中-高
  理由: IFPN 同时输出两者，共享相同的 coarse_adapter → illum_extract 路径
        f_illum_feat 本质上是 lit_up_map 的特征域版本

I(s_illum; f_illum_feat) ≈ 中
  理由: s_illum 由 TFSI 估计 (FrequencyBranch)
        f_illum_feat 由 IFPN 估计 (空间卷积)
        信息来源不同，但目标语义重叠
```

**核心问题**：三个信号之间的互信息过高，意味着存在大量**冗余计算**。网络需要学会"协调"三者的贡献，这在训练中增加了不必要的自由度：

```python
# IGRF 内部（概念化）
output = f_base * g(s_illum) + lit_up_map * h(f_illum_feat)
# g() 和 h() 有多种组合可以近似同一输出
# → 优化景观存在退化方向（degenerate directions）
# → 训练不稳定的根源之一
```

### 三个信号的时序一致性矛盾

| 信号 | 时序属性 | 问题 |
|------|---------|------|
| `s_illum` | TFSI 仅基于中心帧 → **单帧估计** | 无法感知帧间光照闪烁（这正是它要解决的问题） |
| `lit_up_map` | IFPN 基于对齐后特征 → **多帧融合后** | 已经过 SACE 对齐，但对齐本身受限 |
| `f_illum_feat` | 同 IFPN → **多帧融合后** | 与 lit_up_map 同源，时序特性相同 |

**矛盾所在**：`s_illum`（诊断层）是单帧的，但 `lit_up_map`（处方层）是多帧融合后的。诊断在前、处方在后，但诊断的信息量**严格少于**处方——这意味着 IGRF 用一个**信息更少的信号** (`s_illum`) 去调制一个**信息更丰富的信号** (`lit_up_map × f_illum_feat`)，这在信息论上是低效的。

参考 [VSRELL](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) 的做法：
> *"Through **joint modeling** of an illumination-sensitive branch and a noise estimation branch, end-to-end collaborative optimization is achieved... effectively avoiding error propagation in serial structures."*

VSRELL 的 `A_illu` 是基于**多帧窗口统计** (`S(W)` 包含方差、均值、空间最大/最小值) 生成的，因此单一输出就已包含了时序信息。

---

## 权衡设计利弊

### 当前设计（三个独立光照输入）的利与弊

| 维度 | 利 | 弊 |
|------|----|----|
| **表达能力** | ✅ 理论上三个独立信号可以编码更丰富的光照信息 | ❌ 但三者互信息过高，额外自由度被浪费在冗余编码上而非性能提升 |
| **物理可解释性** | ✅ "诊断-处方-内容"三层分工在直觉上合理 | ❌ 但实际训练中边界模糊——网络无法被约束到设计者期望的功能分工 |
| **训练稳定性** | — | ❌ 三个分支的梯度在 IGRF 的乘法/加法节点处耦合，容易产生梯度冲突 |
| **参数效率** | — | ❌ IFPN 的 `illum_extract` + `feat_refine` + `lit_up_proj` + TFSI 的 `IntensityHead` → 大量参数仅为光照建模服务 |
| **时序一致性** | — | ❌ `s_illum` 是单帧估计，但负责调制多帧融合后的特征——信息瓶颈 |
| **对标 SOTA** | — | ❌ 无一篇 2024-2026 顶会论文采用三个独立光照输入 |

### 简化方案对比

| 方案 | 保留信号 | 移除信号 | IGRF 公式 | 参考文献 |
|------|---------|---------|-----------|---------|
| **A：合并 s_illum 到 lit_up_map** | lit_up_map（增强版）, f_illum | s_illum | `res = f_base + lit_up_map' * f_illum` | [ISALux (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/papers/Balmez_ISALux_Illumination_and_Semantics-Aware_Transformer_Employing_Mixture_of_Experts_for_WACV_2026_paper.pdf)：光照先验直接参与注意力 |
| **B：合并 f_illum 到 lit_up_map** | lit_up_map, s_illum | f_illum | `res = f_base * (1 + s_illum * lit_up_map)` | [TempRetinex (2025)](https://arxiv.org/html/2511.09609v2)：光照图 + 反射率精化 |
| **C：单一光照图（推荐）** | A_illu（统一） | s_illum, lit_up_map 独立分支 | `res = f_base * (1 + A_illu) + f_illum` | [VSRELL (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf)：`A_illu` 统一引导 |

### 推荐方案 C 的具体实现

参考 VSRELL 的 INCO + ISFP 设计，将三个光照信号统一为**单一多帧光照注意力图**：

```python
class UnifiedIllumEstimator(nn.Module):
    """
    参考 VSRELL (CVPR 2026) 的 A_illu 设计:
    - 输入: 多帧融合特征 (非单帧)
    - 输出: 单一光照注意力图 A_illu
    - 同时编码: 退化程度 + 增亮强度 + 空间分布
    """
    def __init__(self, channels):
        super().__init__()
        # 多帧统计 → 光照感知 (参考 VSRELL 的 S(W))
        self.temporal_stats = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),  # mean + var 拼接
            nn.GELU(),
        )
        # 光照注意力生成 (参考 VSRELL 的 A_illu)
        self.illu_head = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, aligned_feats):
        """
        aligned_feats: (B, T, C, H, W) — SACE 对齐后的多帧特征
        """
        # 多帧统计 (类比 VSRELL 的 S(W))
        feat_mean = aligned_feats.mean(dim=1)       # (B, C, H, W)
        feat_var = aligned_feats.var(dim=1)          # (B, C, H, W)
        stats = torch.cat([feat_mean, feat_var], dim=1)
        
        feat = self.temporal_stats(stats)
        A_illu = self.illu_head(feat)  # (B, 1, H, W)
        return A_illu

# 简化后的 IGRF
class SimplifiedIGRF(nn.Module):
    def forward(self, f_base, f_illum_feat, A_illu):
        # 单一公式: 光照图引导增益 + 特征补偿
        # A_illu 高 → 该区域需要更多增亮
        enhanced = f_base * (1 + A_illu) + f_illum_feat * A_illu
        # ...后续 ResBlock 精化
```

**这个设计的优势**：

1. **A_illu 基于多帧统计**：`feat_var` 自然编码了帧间光照闪烁信息（高方差 = 闪烁），无需单独的 `s_illum`
2. **单一信号，双重作用**：`A_illu` 同时控制增益强度和特征补偿权重
3. **消除 TFSI→IGRF 的 s_illum 路径**：减少串行依赖，缩短梯度路径
4. **与 [VSRELL](https://openaccess.thecvf.com/content/CVPR2026/html/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.html) 设计对齐**：已在 SDSD 等基准上验证有效

### 对 IFPN 的连带影响

若采用方案 C：

| 模块 | 当前 | 简化后 |
|------|------|--------|
| **IFPN** | 输出 lit_up_map + f_illum_feat + L_t/L_ref/L_ratio + ifpn_side | 输出 **f_illum_feat** + 辅助监督（L_t/ifpn_side 可保留） |
| **TFSI** | 输出 s_illum + s_noise + phase_conf | 输出 **s_noise**（s_illum 移除或并入 UnifiedIllumEstimator） |
| **IGRF** | 输入 lit_up_map + f_illum_feat + s_illum | 输入 **f_illum_feat + A_illu** |
| **参数量变化** | — | 移除 `lit_up_proj` + `IntensityHead` 的 s_illum 分支 ≈ -15~20K 参数 |

---

## 总结

| 评估维度 | 评分 | 说明 |
|---------|------|------|
| 设计必要性 | ⚠️ **低** | 三个信号功能重叠严重，互信息高 |
| SOTA 对标 | ❌ **无** | 2024-2026 无论文使用三个独立光照输入 |
| 训练稳定性影响 | ❌ **负面** | 增加了优化景观的退化方向 |
| 简化可行性 | ✅ **高** | 参考 VSRELL 的 A_illu 单一表示即可 |
| 建议优先级 | 🔴 **P1** | 与"s_noise 路由错位"并列为最应修改的架构问题 |

**核心结论**：当前三输入设计**过度工程化**（over-engineered），将一个本质上单一的物理量（场景光照分布）拆分成三个耦合的信号，增加了训练难度而未带来性能收益。参考 VSRELL 的单一 `A_illu` 设计进行简化，可同时改善训练稳定性、参数效率和物理可解释性。

---

## 参考来源

1. [VSRELL: A Simple Baseline for Video Super-Resolution and Enhancement in Low-Light Environment (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf)
2. [VSRELL — CVPR 2026 Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.html)
3. [Multinex: Lightweight Low-light Image Enhancement via Multi-prior Retinex (CVPR 2026)](https://github.com/albrateanu/multinex)
4. [ISALux: Illumination and Semantics-Aware Transformer Employing MoE for LLIE (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/papers/Balmez_ISALux_Illumination_and_Semantics-Aware_Transformer_Employing_Mixture_of_Experts_for_WACV_2026_paper.pdf)
5. [M2Retinexformer: Multi-Modal Retinexformer for LLIE (arXiv 2026)](https://arxiv.org/html/2605.12556v1)
6. [IS-SFD: Illumination Smoothness and Semantic-frequency Denoising for LLVE (OpenReview 2026)](https://openreview.net/forum?id=vc6pSwumm6)
7. [TempRetinex: Retinex-Based Unsupervised Enhancement for Low-Light Video (2025)](https://arxiv.org/html/2511.09609v2)
8. [Zero-TIG: Temporal Consistency-Aware Zero-Shot Illumination-Guided LLVE (2025)](https://arxiv.org/html/2503.11175v1)
9. [FPN-Retinex: Semi-Supervised Low-Light Image Enhancement via FPN (CCC 2024)](https://doi.org/10.23919/ccc63176.2024.10662161)
10. [PyrLight: Laplacian-Pyramid-Guided Conditional Diffusion for LLIE (CVPR 2026 Workshop)](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/html/Yan_PyrLight_Laplacian-Pyramid-Guided_Conditional_Diffusion_for_Detail-Preserving_and_Artifact-Free_Low-Light_Enhancement_CVPRW_2026_paper.html)
11. [Low-light Image Enhancement with Retinex Decomposition in Latent Space (arXiv 2026)](https://arxiv.org/html/2603.15131)