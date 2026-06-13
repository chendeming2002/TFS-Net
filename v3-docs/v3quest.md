# TFS-Net v3 实现疑问与待澄清事项

> 本文档记录了根据 `TFSv3-result.md` 设计文档在现有代码基础上改写时，发现的模糊描述、缺失实现细节、以及需要参考论文代码的模块。
> **仅记录不修改的部分**——对于设计完全清晰、可直接改写的模块，不在此文件中出现。
> 
> **最后更新**：2026-06-12，第一轮改写实施后

---

## 一、各模块级疑问

### 1. TFSI — 频域分支（LFF 可学习频率滤波）

**对应现有文件**：`models/modules/mins.py` (MINSBlock → 全面改写为 TFSI)
**当前状态**：❌ 不可实现，频域分支未实现，TFSI 骨架中以零张量占位

**需要 FRBNet 论文代码参考的部分**：

1. **可学习径向基滤波器 (RBF) $H(u,v)$ 的具体实现**
   - 文档给出公式：$H(u,v) = \sum_{k=1}^{K} \omega_k \phi_k(\|(u,v)-\mu_k\|) \cdot \cos(\theta_{uv} - \theta_k)$
   - **疑问**：
     - $K$（RBF 基函数个数）的默认值是多少？FRBNet 论文中使用多少？
     - $\phi_k$ 的具体函数形式是什么？标准高斯 RBF $\exp(-\gamma r^2)$？还是其他？
     - $\mu_k$（中心位置）是固定网格初始化还是随机初始化？是否在频域二维坐标上学习？
     - $\theta_k$（方向角度参数）如何初始化？
     - $\omega_k$（权重）是可学习标量还是可学习空间 map？
   - **需求**：需要参考 FRBNet 官方 GitHub 仓库 (`Sing-Forevet/FRBNet`) 的 `LearnableFrequencyFilter` 或类似模块的 PyTorch 实现。

2. **零DC高斯窗 $W_g$ 的 $\sigma_g$ 初始化**
   - 公式：$W_g(u,v) = 1 - \exp(-(u^2+v^2)/(2\sigma_g^2))$
   - **疑问**：$\sigma_g$ 初始值设为多少？FRBNet 中如何初始化？是标量还是每通道独立？

3. **FFT 对多通道特征图的操作方式**
   - 现有编码器输出 $F_i \in \mathbb{R}^{C \times H \times W}$（$C=48$）。
   - **疑问**：FFT 是对每个通道独立做 2D-FFT（得到 $C$ 个频域图），还是对空间做 2D-FFT 后得到复数特征？$\tilde{\mathcal{F}}_i$ 的 shape 是 $(C, H, W)$ 复数还是 $(2C, H, W)$ 实数（幅度+相位拼接）？
   - 文档中 $F_f = \text{Conv}_{1\times1}(\text{Concat}[|\tilde{\mathcal{F}}_i|, \angle\tilde{\mathcal{F}}_i])$ 暗示是 $2C$ 通道输入，但需确认。

4. **对哪一帧做 LFF？**
   - 空间分支的 $\mu_t, \sigma_t$ 是所有帧的统计量。但频域分支公式中 $\tilde{\mathcal{F}}_i = W_g \odot H \odot \mathcal{F}_i$ 下标是 $i$。
   - **疑问**：是对每一帧分别做 LFF 再聚合，还是仅对中心帧 $t$ 做 LFF？如果对每帧都做，频域特征如何跨帧聚合（取均值？取中位值？）？

### 2. TFSI — 空间分支

**对应现有文件**：`models/modules/mins.py`
**当前状态**：✅ 已实现（`models/modules/tfsi.py` 中 `SpatialBranch`）

- 文档中空间分支使用 Median（中位值）和 Var（方差），现有 MINSBlock 使用的是 window attention 计算 entropy。
- **已实现**：μ_t = Median_i{F_i}, σ_t² = Var_i{F_i}, F_s = Conv3x3(Concat[μ_t, σ_t, μ_t/(σ_t+ε)])
- **小疑问**：$\mu_t / (\sigma_t + \epsilon)$ 作为第三通道拼接到 $[\mu_t, \sigma_t]$ 后面——这个 SNR 比和 TFSI 输出给 NDPN 的 SNR 是同一个量吗？还是 NDPN 使用自己独立计算的 SNR？文档中 NDPN 部分写的是用 $\mu_t^{clean}/\sigma_t$（光照归一化后的），而 TFSI 空间分支这里用的是原始 $\mu_t/\sigma_t$。两者是否不同？

### 3. SACE — 源感知对应估计（新增模块）

**对应现有文件**：无（全新模块），概念上替代了 MINSBlock 中的对应估计部分。
**当前状态**：❌ 不可实现，依赖 LFF（未澄清）+ 可变形 cross-attention（需 DAT 参考）+ 输出定义不明

**需要参考论文代码的部分**：

1. **可变形 cross-attention 的具体实现**
   - 文档引用 [DAT, CVPR 2023] 和 [EDVR, CVPR 2019]。
   - **疑问**：
     - 偏移量 $\Delta p_i$ 是通过什么网络预测的？是标准 Deformable Conv（DCNv2 风格，参考 EDVR）还是 Deformable Attention（DAT 风格，基于 sampling point 预测）？
     - 每个 query 使用多少个采样点？DAT 默认 4 或 8 个。
     - attention 是在 window 内做还是全局做？现有代码使用 window attention，是否沿用？
     - cross-attention 的 Q 来自中心帧 $\bar{F}_t$，K/V 来自邻帧 $\bar{F}_i$——邻帧是逐帧做 cross-attention 还是拼接后一次做？
   - **需求**：需要参考 DAT (CVPR 2023) 的 deformable attention 实现代码，或 torchvision 的 `DeformableAttention` 模块。

2. **SACE 输出是什么？**
   - 文档只描述了 attention 的计算 $\mathbf{A}_{t \to i}$，但没有明确说明 SACE 的输出特征是什么：
     - 是输出 attention-weighted 的聚合特征？
     - 还是仅输出 attention map $\mathbf{A}_{t \to i}$ 供下游 NDPN/MRPN 使用（类似现有 MSPN 的做法）？
   - 根据 v3 架构图 mermaid 中 "DefAttn --> NDPN & MRPN"，推测是输出 attention map。但需确认。

3. **SACE 依赖 LFF 做光照归一化**
   - Step 1 对每帧做 LFF 得到 $\bar{F}_i$，但 LFF 本身实现不明确（见 TFSI 频域分支疑问）。
   - SACE 中使用的 LFF 与 TFSI 频域分支中的 LFF 是否共享权重？（见整体框架 § 二）

### 4. IFPN — 双流光照估计（改写自 ISPN）

**对应现有文件**：`models/modules/ispn.py` (ISPN → IFPN)
**当前状态**：❌ 不可实现，IllumExtract 结构、L_ref 的 sim()、F_t^illum 定义、F_t^(L) 定义均不明

**需要参考论文代码的部分**：

1. **IllumExtract（光照估计器）的具体结构**
   - 文档写 $L_t = \text{IllumExtract}(\text{Concat}[I_t^{down}, \text{Conv}_{1\times1}(F_t^{(L)})])$，引用 [Retinexformer, ICCV 2023]。
   - **疑问**：
     - IllumExtract 是几层 Conv？Retinexformer 中的 Illumination Estimator 具体是什么结构（通常是一系列 Conv+ReLU 最后 sigmoid 输出单通道或多通道光照图）？
     - 输出 $L_t$ 的通道数是多少？单通道灰度光照图？3 通道 RGB 光照图？还是 $C$ 通道特征级光照？
   - **需求**：需要参考 Retinexformer 的 `IlluminationEstimator` 代码。

2. **邻帧参考光照 $L_{ref}$ 的计算**
   - 公式：$L_{ref} = \sum_{i \neq t} w_i L_i$，$w_i = \text{Softmax}(\text{sim}(F_i, F_t))$
   - **疑问**：
     - $\text{sim}(F_i, F_t)$ 的具体度量是什么？余弦相似度？点积？还是可学习的相似度函数？
     - 这个 sim 是在全局空间上做，还是逐像素做？如果是逐像素，$w_i$ 就是一个空间 map，需要用 Conv 来估计。
     - 现有 ISPN 用的是 `pairwise_cosine_logits`（全局平均后的余弦相似度），这里是否沿用？

3. **强度调制矫正公式的物理含义**
   - $F_t^{illum\_out} = s_{illum} \cdot F_t^{illum} \cdot \frac{L_{ref}}{L_t + \epsilon} + (1-s_{illum}) \cdot F_t^{illum}$
   - **疑问**：$F_t^{illum}$ 是什么？是编码器原始特征 $F_t$，还是 IFPN 内部某个中间特征？文档中没有定义 $F_t^{illum}$ 的来源。

4. **$F_t^{(L)}$ 的定义**
   - IFPN 双流需要最粗尺度特征 $F_t^{(L)}$。
   - **疑问**：$F_t^{(L)}$ 具体是 `l3`（stage3 直接输出，96 通道，H/8×W/8）还是 `p3`（lateral 投影后，48 通道，分辨率 H/4×W/4）？文档说"最粗尺度"，H/8×W/8 对应的是 `l3`（经过两次 stride=2 下采样）。
   - **编码器实际分辨率不一致**：当前 PyramidEncoder 仅有两次 stride=2（stage2 + stage3），l3 实际分辨率为 **H/4×W/4**（而非 v3 文档写的 H/8×W/8）。若要达到 H/8×W/8 需增加 stage4（stride=2），或将 stage3 改为 stride=4，或修改 IFPN 设计。这是一个需澄清的结构问题。

### 5. NDPN — SNR 自适应聚合（改写自 MSPN）

**对应现有文件**：`models/modules/mspn.py` (MSPN → NDPN)
**当前状态**：❌ 不可实现，依赖 SACE attention map，且 F_i 特征定义、Conv 结构、τ 初始化均不明

**疑问**：

1. **与 SACE 的关系**
   - NDPN 的 Step 1 写 $F_i^{aligned} = \mathbf{A}_{t \to i} \cdot W_V F_i$，说明 NDPN 复用 SACE 的 attention map。
   - 但 SACE 的 $\mathbf{A}_{t \to i}$ 是在光照归一化特征 $\bar{F}_i$ 上计算的，而 NDPN 用 $W_V F_i$（原始特征？还是也用 $\bar{F}_i$？）
   - **疑问**：$F_i$ 这里指的是原始编码器特征 $F_i$，还是 SACE 中光照归一化后的 $\bar{F}_i$？

2. **SNR 归一化的可学习参数**
   - $\tilde{s}_{SNR} = \sigma((\hat{SNR} - \tau_{mid}) / \tau_{scale})$
   - $\tau_{mid}, \tau_{scale}$ 是可学习标量——初始化为多少？是否有合理默认值？（例如 $\tau_{mid}=1.0, \tau_{scale}=1.0$？）

3. **聚合权重的卷积核**
   - $\alpha_i = \sigma(\text{Conv}(|F_i^{aligned} - F_t|)) \cdot (1 - \tilde{s}_{SNR})$
   - 这里的 Conv 是几层？$1\times1$ 还是 $3\times3$？输出通道数？是单层 Conv + Sigmoid 还是多层网络？

4. **$W_V$ 的定义**
   - $F_i^{aligned} = \mathbf{A}_{t \to i} \cdot W_V F_i$ 中 $W_V$ 是 1×1 卷积投影还是直接恒等？

### 6. MRPN — 运动残差处理（新增模块）

**对应现有文件**：无（全新模块）。文档写"残差驱动隐式遮挡处理，详见 v2，此处不重复"。
**当前状态**：❌ 完全不可实现，v2 文档不存在于本代码库，无任何设计细节

**疑问**：

1. **v2 设计细节完全缺失**
   - 当前代码库中没有 MRPN 的任何对应实现。v3 文档说"详见 v2"，但本代码库是从 v1 (MINS-Net) 开始，并不包含 v2 的 MRPN 实现。
   - **需求**：需要 v2 文档中 MRPN 的详细设计，包括：
     - 残差如何定义？（对齐帧与中心帧的差？）
     - 遮挡 map 如何从残差中估计？（网络结构？）
     - 输出特征如何计算？

2. **与 NDPN 的结构差异**
   - 两者都使用 SACE 的 attention map 做对齐，区别在于：NDPN 做 SNR 自适应去噪，MRPN 做遮挡处理。
   - **疑问**：MRPN 的聚合权重公式是什么？是否类似于 NDPN 但用"残差大→降权"替代"SNR低→加权"？

3. **MRPN 输出如何参与 IGRF 融合**
   - $F_t^{motion\_out}$ 的具体形式是特征级还是残差级？通道数是否为 $C=48$？

### 7. IGRF — 强度引导融合（改写自 FinalReconstruction）

**对应现有文件**：`models/modules/reconstruction.py` (FinalReconstruction → IGRF)
**当前状态**：⚠️ 部分可实现（融合公式清晰，已实现于 `models/modules/igrf.py`），但 $F_t^{base}$ 未定义，暂用 $F_t$（编码器原始特征）作为占位假设

**疑问**：

1. **$F_t^{base}$ 的定义**
   - 公式：$F_{fused} = s_{illum} F_t^{illum\_out} + s_{noise} F_t^{noise\_out} + s_{motion} F_t^{motion\_out} + F_t^{base}$
   - **疑问**：$F_t^{base}$ 是什么？是编码器原始特征 $F_t$？还是某种 base residual？文档中没有定义。
   - **当前假设**：暂用编码器融合特征 $F_t$（PyramidEncoder 输出）作为 $F_t^{base}$。

2. **三分支输出的通道数一致性**
   - IFPN、NDPN、MRPN 三个分支的输出特征通道数是否都是 $C$（fused_channels=48）？需确认以保证加法合法。

### 8. 损失函数

**对应现有文件**：`losses/losses.py` (MINSLoss → TFSNetLoss)
**当前状态**：⚠️ 部分可实现（L_recon 空间部分、L_perc、L_illum_smooth 清晰），但 L_temporal、L_consist、L_recon 频域部分均不可实现

**需要参考论文代码的部分**：

1. **时序一致性损失中的 RAFT 光流**
   - $\mathcal{L}_{temporal} = \sum_{i \neq t} \|\text{Warp}(\hat{I}_i, \text{Flow}_{i \to t}) - \hat{I}_t\|_1 \cdot (1-M_{occ})$
   - **疑问**：
     - RAFT 是预训练冻结的，还是联合训练的？（通常冻结）
     - 光流估计的输入是原图 $\hat{I}_t$ 还是某个中间特征？
     - 遮挡 mask $M_{occ}$ 如何计算？通常是 forward-backward flow consistency check，阈值是多少？
     - 需要对所有邻帧都算光流，还是仅对相邻帧？（$i = t\pm1$ 还是所有 $i \neq t$）
   - **需求**：需要 RAFT 的 PyTorch 推理代码（torchvision 已内置 `raft_large`）。

2. **频域重建损失**
   - $\mathcal{L}_{recon} = \|\hat{I}_t - I_t^{GT}\|_1 + 0.1 \cdot \|\text{FFT}(\hat{I}_t) - \text{FFT}(I_t^{GT})\|_1$
   - **疑问**：FFT 损失的 L1 是对复数的实部/虚部分别算，还是对幅度谱算？引用 [Cho et al., ICCV 2021] 通常是对幅度+相位分别计算。

3. **源强度时序一致性正则**
   - $\mathcal{L}_{consist} = \sum_s \|\text{Warp}(s^{(i)}, \text{Flow}) - s^{(t)}\|_1$
   - **疑问**：$s^{(i)}$ 是第 $i$ 帧经 TFSI 输出的强度 map。但 TFSI 只对中心帧 $t$ 输出强度（因为空间分支用 $\mu_t, \sigma_t$ 统计量），如何获得每一帧的 $s^{(i)}$？是否需要在训练时对每个帧都跑一遍 TFSI？这对计算开销的影响巨大。

### 9. PyramidEncoder 修改

**对应现有文件**：`models/modules/encoder.py`
**当前状态**：⚠️ 已修改为返回多尺度特征 `(fused_feat, coarse_feat)`，但 $F_t^{(L)}$ 的精确定义仍不明（见 IFPN 疑问 § 1.4.4）

**疑问**：

1. **IFPN 双流需要最粗尺度特征 $F_t^{(L)}$**
   - 当前 `PyramidEncoder.forward_single()` 已修改为返回 `(self.fuse(p1), l3)` 元组。
   - **疑问**：$F_t^{(L)}$ 具体是 `l3`（stage3 直接输出，96通道）还是 `p3`（lateral 投影后，48通道，但分辨率为 $H/4 \times W/4$）？文档说"最粗尺度"，$H/8 \times W/8$ 对应的是 `l3`（经过两次 stride=2 下采样）。
   - **当前假设**：暂用 `l3`（96 通道，H/4×W/4）作为 `coarse_feat` 返回。
   - **分辨率不一致**：当前编码器 l3 分辨率为 H/4×W/4，而 v3 设计文档写 H/8×W/8。需澄清是否增加 stage4 或修改 IFPN 双流设计。（见 IFPN 疑问 § 1.4.4）

---

## 二、整体框架与数据流相关问题

**当前状态**：TFSNet 主网络骨架已搭建（`models/tfs_net.py`），但 SACE/IFPN/NDPN/MRPN 均以 `NotImplementedError` 占位，无法完成前向传播。

### 2.1 前向传播数据流顺序

v3 设计有 5 个 Stage，但模块间的依赖关系存在循环引用风险：

- **TFSI** 需要多帧特征 $\{F_i\}$ → 来自 Encoder
- **SACE** 需要 LFF 做光照归一化 → LFF 在 TFSI 中定义，SACE "复用" TFSI 的 LFF
- **NDPN** 需要 SACE 的 attention map + TFSI 的 SNR + $\mu_t^{clean}$
- **IFPN** 需要 $I_t^{down}$（原图下采样）+ $F_t^{(L)}$（编码器粗特征）+ $s_{illum}$（TFSI 输出）
- **IGRF** 需要三个分支输出 + $s_*$（TFSI 输出）

**问题**：LFF 作为 TFSI 和 SACE 共享的子模块，是作为独立 `nn.Module` 提取出来由两者共享权重，还是 TFSI 和 SACE 各自实例化独立的 LFF？这影响参数量和训练行为。

**当前处理**：`tfsi.py` 中 `TFSI` 类已预留 `freq_branch` 属性（可为 `None` 或外部注入的 LFF 模块），供后续 SACE 共享时扩展。

### 2.2 训练时的多帧损失计算

v3 的时序一致性损失需要对邻帧的增强结果 $\hat{I}_i$ 做 warp。这意味着：
- 训练时需要对 **每个邻帧** 都通过完整网络得到 $\hat{I}_i$？
- 还是只对中心帧 $t$ 输出 $\hat{I}_t$，然后对 $\hat{I}_t$ 做反向 warp？

如果是前者，5 帧窗口需要 5 次前向传播，计算量巨大。如果是后者，文档中的公式 $\text{Warp}(\hat{I}_i, \text{Flow}_{i \to t})$ 就需要改写为只涉及 $\hat{I}_t$ 的形式。

**这是实现前必须澄清的关键问题。**

### 2.3 现有代码完全不需要修改的部分

以下模块在 v3 改写中**保持不变**：

| 文件 | 说明 |
|:---|:---|
| `models/modules/blocks.py` | 基础构建块（ConvBlock, ResBlock, LayerNorm2d, window 操作函数, safe_divide, pairwise_cosine_logits）全部保留，v3 各模块仍会用到 |
| `datasets/sdsd_dataset.py` | 数据集加载逻辑不变，多帧窗口采集方式不变 |
| `datasets/transforms.py` | 数据增强（随机裁剪、翻转、时序反转）不变 |
| `datasets/__init__.py` | 不变 |
| `utils/inference.py` | tiled_forward 推理逻辑不变（但需确保 model forward 返回格式兼容 `["res_t"]`） |
| `utils/io.py` | 检查点/图像 IO 不变 |
| `utils/metrics.py` | PSNR/SSIM 指标不变 |
| `utils/misc.py` | AverageMeter、seed、logger 不变 |
| `train.py` | 训练循环结构基本不变，但需更新 model/loss 的 import 和构建函数、以及 loss_dict 的 key |
| `infer.py` | 推理脚本结构不变，但需更新 model import 和构建函数 |
| `configs/sdsd_stage1.yaml` | 需更新 model 和 loss 相关配置项 |

### 2.4 需要新增的文件

| 新文件路径 | 内容 | 状态 |
|:---|:---|:---|
| `models/modules/tfsi.py` | TFSI 时频源指示器（替代 mins.py） | ✅ 骨架已实现（频域分支待补） |
| `models/modules/lff.py` | LFF 可学习频率滤波器（TFSI 和 SACE 共享子模块） | ❌ 依赖 LFF 实现细节 |
| `models/modules/sace.py` | SACE 源感知对应估计（新增） | ❌ 依赖 LFF + DAT 参考 |
| `models/modules/ifpn.py` | IFPN 双流光照估计（替代 ispn.py） | ❌ 依赖 Retinexformer 参考 |
| `models/modules/ndpn.py` | NDPN SNR 自适应聚合（替代 mspn.py） | ❌ 依赖 SACE attention map |
| `models/modules/mrpn.py` | MRPN 运动残差处理（新增，需 v2 设计） | ❌ 完全无设计 |
| `models/modules/deform_attn.py` | 可变形 cross-attention 实现（SACE 子模块） | ❌ 依赖 DAT 参考 |
| `models/modules/igrf.py` | IGRF 强度引导融合（替代 reconstruction.py） | ✅ 已实现（F_t^base 假设待确认） |
| `models/tfs_net.py` | TFSNet 主网络骨架 | ✅ 已实现（4模块待补） |

### 2.5 可删除的旧文件

| 文件路径 | 原因 | 状态 |
|:---|:---|:---|
| `models/modules/mins.py` | MINSBlock 被 TFSI + SACE 完全替代 | ⏳ 暂保留（TFSI 频域分支未完成） |
| `models/modules/ispn.py` | ISPN 被 IFPN 替代 | ⏳ 暂保留（IFPN 未实现） |
| `models/modules/mspn.py` | MSPN 被 NDPN + MRPN 替代 | ⏳ 暂保留（NDPN/MRPN 未实现） |
| `models/mins_net.py` | 被 `models/tfs_net.py` 替代 | ⏳ 暂保留（TFSNet 未完成） |
| `models/modules/reconstruction.py` | FinalReconstruction 被 IGRF 替代 | ⏳ 暂保留（IGRF 假设待确认） |

---

## 三、任务流指示（供 AI 助手执行）

> 本节为结构化任务流指令，指导 AI 助手如何系统性地解决 § 一、§ 二中的待澄清问题，
> 并完成 TFS-Net v3 剩余模块的实现。

### # Role（角色）

你是一名 **PyTorch 视频低光照增强研究员 + 工程师**，职责如下：

1. **论文代码调研员**：阅读 v3quest.md 中标注的参考论文/仓库，提取可落地的 PyTorch 实现细节
2. **架构实现者**：在现有 MINS-Net 代码基础上，逐模块补全 TFS-Net v3 的实现
3. **文档维护者**：每解决一个疑问，立即更新 v3quest.md 中对应条目的状态标记（❌→⚠️→✅）

### # Action（行动）

按以下优先级执行，**严格遵循依赖顺序**（上游模块未解决前不实现下游）：

```
Phase A：信息收集与答疑（只读，不改代码）
Phase B：按依赖顺序逐模块实现
Phase C：集成测试 + 损失函数 + 训练脚本适配
Phase D：清理旧文件 + 最终验证
```

### # Steps（步骤）

#### Phase A — 信息收集与答疑

> **目标**：解决 § 一中所有 ❌ 条目的疑问，使每个模块至少达到"可编码"状态。

**A.1 调研 FRBNet LFF 实现**（解决 §1.1 全部疑问）
- 访问 `https://github.com/Sing-Forevet/FRBNet`
- 定位 `LearnableFrequencyFilter` 或等价模块
- 提取以下参数：$K$ 默认值、$\phi_k$ 函数形式、$\mu_k$ / $\theta_k$ / $\omega_k$ 初始化方式、$\sigma_g$ 初始化
- 确认 FFT 对多通道特征图的操作方式（逐通道 2D-FFT vs 整体）
- **输出**：将提取的参数值写入 §1.1 对应疑问下方，标记为 `✅ 已澄清：...`

**A.2 调研 DAT 可变形注意力实现**（解决 §1.3.1）
- 访问 DAT (CVPR 2023) 官方代码或 `torchvision.ops.DeformableAttention`
- 确认：偏移预测网络结构、采样点数、window vs global、逐帧 vs 拼接
- **输出**：将实现方案写入 §1.3.1 对应疑问下方

**A.3 调研 Retinexformer IllumExtract**（解决 §1.4.1）
- 访问 Retinexformer (ICCV 2023) 官方代码
- 提取：IlluminationEstimator 的层数、激活函数、输出通道数
- **输出**：将结构描述写入 §1.4.1 对应疑问下方

**A.4 澄清设计文档歧义**（解决剩余疑问）
- 对每个 ❌ 条目，如果通过调研仍无法确定，向用户提问
- **提问格式**：`[模块名] 问题X：{具体描述}，选项A/B/C，推荐A因为{理由}`

#### Phase B — 按依赖顺序逐模块实现

> **目标**：按数据流依赖关系，逐个实现 §2.4 中标记为 ❌ 的模块。
> **实现顺序**（不可调换）：

```
B.1  lff.py          ← TFSI 频域分支 + SACE 共享子模块（依赖 A.1）
B.2  tfsi.py 补全    ← 将 FrequencyBranch 替换为真实 LFF 实现
B.3  deform_attn.py  ← 可变形 cross-attention（依赖 A.2）
B.4  sace.py         ← SACE 源感知对应估计（依赖 B.1 + B.3）
B.5  ifpn.py         ← IFPN 双流光照估计（依赖 A.3 + §1.4 澄清）
B.6  ndpn.py         ← NDPN SNR 自适应聚合（依赖 B.4）
B.7  mrpn.py         ← MRPN 运动残差处理（依赖 B.4 + v2 设计补充）
B.8  tfs_net.py 整合 ← 取消 NotImplementedError，接入所有模块
```

**每个模块的实现流程**：

1. **创建文件**：在 `models/modules/` 下创建对应 `.py` 文件
2. **编写代码**：包含完整的类定义、docstring（含公式 + shape 注释）、forward 方法
3. **单元测试**：用 `torch.randn` 构造假数据，验证输入输出 shape 正确
4. **更新 v3quest.md**：将 §2.4 表格中对应行状态从 ❌ 更新为 ✅
5. **更新 `__init__.py`**：在 `models/modules/__init__.py` 中导出新模块

#### Phase C — 集成测试与损失函数

> **目标**：端到端验证 + 补全损失函数 + 适配训练/推理脚本。

**C.1 TFSNet 端到端前向测试**
```python
# 验证：输入 (1, 5, 3, 256, 256) → 输出 dict 包含 res_t (1, 3, 256, 256)
model = TFSNet()
x = torch.randn(1, 5, 3, 256, 256)
out = model(x)
assert out["res_t"].shape == (1, 3, 256, 256)
```

**C.2 实现 TFSNetLoss**（解决 §1.8 疑问后）
- 在 `losses/losses.py` 中新增 `TFSNetLoss` 类
- 包含：L_recon（空间+频域）、L_temporal（RAFT warp）、L_consist、L_illum_smooth、L_perc
- RAFT 用 `torchvision.models.optical_flow.raft_large(pretrained=True)` 并冻结参数

**C.3 适配 train.py 和 infer.py**
- `train.py`：更新 `build_model()` → `TFSNet`，`build_loss()` → `TFSNetLoss`，更新 loss_dict key
- `infer.py`：更新 model import → `TFSNet`
- `configs/sdsd_stage1.yaml`：更新 model/loss 配置项

#### Phase D — 清理与最终验证

**D.1 删除旧文件**（仅在所有模块 ✅ 后）
- 删除 `models/modules/mins.py`、`ispn.py`、`mspn.py`、`reconstruction.py`
- 删除 `models/mins_net.py`
- 更新 `models/modules/__init__.py` 和 `models/__init__.py`，移除旧导出

**D.2 最终验证**
- 运行 `python train.py --config configs/sdsd_stage1.yaml --smoke` 确认训练循环可启动
- 运行 `python infer.py` 确认推理流程可走通（可跳过实际推理）

### # Context（上下文）

| 资源 | 路径/链接 | 用途 |
|:---|:---|:---|
| v3 设计文档 | `e:/TFS-Net/TFSv3-result.md` | 各模块公式、架构图、参考文献列表 |
| 本文档 | `e:/TFS-Net/v3quest.md` | 待澄清问题清单 + 实现状态追踪 |
| 现有代码库 | `e:/TFS-Net/models/` | MINS-Net v1 实现（参考结构，不直接修改） |
| 已实现 v3 模块 | `models/modules/tfsi.py`, `igrf.py`, `encoder.py` | TFSI 骨架、IGRF、多尺度编码器 |
| TFSNet 骨架 | `models/tfs_net.py` | 5-Stage 主网络（SACE/IFPN/NDPN/MRPN 待补） |
| FRBNet 仓库 | `https://github.com/Sing-Forevet/FRBNet` | LFF 频域滤波参考实现 |
| DAT 论文代码 | CVPR 2023 Deformable Attention Transformer | 可变形注意力参考 |
| Retinexformer | ICCV 2023 One-stage Retinex-based Transformer | IllumExtract 光照估计器参考 |
| RAFT 光流 | `torchvision.models.optical_flow.raft_large` | 时序一致性损失中的光流估计 |

### # Format（输出格式）

每个阶段完成后，按以下格式向用户汇报：

```
## Phase X 完成报告

### 已解决的问题
- [§N.M] 问题描述 → 答案摘要

### 已实现/修改的文件
| 文件 | 变更内容 | 验证状态 |
|:---|:---|:---|
| `path/to/file.py` | 新建/修改：... | ✅ shape 测试通过 |

### v3quest.md 状态更新
- §N.M : ❌ → ✅
- §2.4 表格：xxx 行更新

### 遗留问题（如有）
- [§N.M] 仍需用户确认：...
```

### # 约束条件

1. **只改设计清晰的模块**：如果某个模块的核心公式/结构仍有歧义，**不要凭猜测实现**，标记为 ❌ 并向用户提问
2. **向后兼容**：修改 `encoder.py` 等共享文件时，确保 `MINSNet`（v1）仍可正常运行（已通过 `return_coarse` 参数实现）
3. **Shape 注释**：所有 `forward()` 方法的输入/输出/中间张量必须标注 shape 注释，格式为 `# (B, C, H, W)`
4. **docstring 标注**：每个模块文件头部 docstring 必须标注当前实现状态（✅/⚠️/❌）和未解决疑问的 v3quest.md 章节引用
5. **不删除旧文件**：在 Phase D 之前，`mins.py` / `ispn.py` / `mspn.py` / `mins_net.py` / `reconstruction.py` 必须保留
6. **不修改以下文件**：`blocks.py`、`datasets/*`、`utils/*`（除非 tiled_forward 需适配新返回格式）
7. **参数量控制**：每个新模块实现后汇报参数量，TFSNet 总参数量目标 < 2M（与 MINS-Net v1 量级相当）
8. **先调研后编码**：Phase A 必须在 Phase B 之前完成；不允许在疑问未解决时"先写个大概"
9. **每模块单独验证**：每个新模块必须有独立的 shape 验证脚本或测试代码，不依赖其他未实现模块
10. **v3quest.md 是唯一真相源**：所有疑问、决策、状态变更都记录在此文件中，不在代码注释中记录设计决策
