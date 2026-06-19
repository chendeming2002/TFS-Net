# TFS-Net v3 — 用户行为需求清单 (v3require2)

> 本文档记录 Phase A 审查后，**需要用户（您）做出决策或提供输入**的事项。
> 每项标注优先级和影响范围，请按顺序处理。
>
> **生成日期**：2026-06-12
> **关联文档**：v3quest2.md（AI 侧遗留问题）

---

## 一、必须决策项（阻塞编码）

### R1: LFF 对特征图的适配策略 [P0]

**背景**：FRBNet 的 LFF 是对 3 通道 RGB 图像做处理（对数域 + 通道差 FFT），而 TFS-Net 需要对 C=48 通道的编码器特征做 FFT。这是根本性的应用场景差异，Claude 的 v3answer 没有讨论此问题。

**选项**：

| 选项 | 描述 | 优点 | 缺点 |
|:---|:---|:---|:---|
| **A（推荐）** | 逐通道 rfft2（48 次独立 FFT），LFF 参数共享，输出拼接后 Conv1x1 降到 C_f | 最自然的扩展；参数效率高；每通道都获得频域特征 | 48 次 FFT 有一定计算开销（但 rfft2 很快） |
| **B** | 先 1×1 Conv 降到 3 通道再做 FFT（模拟 FRBNet 的 RGB 通道差） | 最接近 FRBNet 原始设计 | 降维会丢失通道信息；与编码器 C=48 的设计意图矛盾 |
| **C** | 全通道一次性 fft2，LFF 滤波器对幅度谱 broadcast | 最简单，计算量最小 | 所有通道共享同一频谱表示，丧失通道特异性 |

**您的需求**：请选择 A/B/C 或提出替代方案。此决策直接影响 `lff.py` 的 forward 实现。

---

### R2: W_g (Zero-DC 窗) 的公式选择 [P0]

**背景**：v3 设计文档公式与 FRBNet 实际代码公式不同：

| 方案 | 公式 | 物理意义 |
|:---|:---|:---|
| v3 文档 | $W_g = 1 - \exp(-r^2/(2\sigma^2))$ | 标准高通滤波器（低频衰减，高频通过） |
| FRBNet 代码 | $W_g = \exp(-(r/\sigma)^2)$，$W_g[0,0]=0$ | 高斯带通（选取以 DC 为中心的频带，但强制排除 DC） |

**您的需求**：确认使用哪个公式。推荐使用 FRBNet 代码版本（已有实验验证），但这改变了 v3 设计文档的描述。

---

### R3: SACE 可变形 cross-attention 方案 [P0]

**背景**：v3quest 提到需要 DAT 参考，但 DAT 的 `DAttentionBaseline` 是 self-attention 模块，不能直接用作 cross-attention。已下载的 BasicSR/EDVR 中有 `PCDAlignment`（DCNv2 风格多尺度可变形对齐），更适合密集预测。

**选项**：

| 选项 | 描述 | 参数量 | 成熟度 |
|:---|:---|:---|:---|
| **A（推荐）** | DCNv2 风格（参考 EDVR PCD）：多尺度可变形卷积预测偏移，局部 K×K 窗口采样 | 中等 | 视频恢复领域最成熟 |
| **B** | 简化版 DAT：把 conv_offset 输入改为 Q-K 差值，全局 grid_sample | 较少 | 需要适配，无直接验证 |
| **C** | 标准 cross-attention（无 deformable）：Q=中心帧, K/V=邻帧，全局 softmax | 最少 | 最简单但无空间对齐能力 |

**您的需求**：请选择 A/B/C。此决策决定 `deform_attn.py` 和 `sace.py` 的核心实现。

---

### R4: IllumExtract (光照估计器) 适配方案 [P0]

**背景**：Retinexformer 实际的 `Illumination_Estimator` 结构与 Claude 描述完全不同：
- 实际是 1×1 Conv → 5×5 Depthwise Conv → 1×1 Conv（无 Sigmoid）
- 输入 4 通道（RGB + mean），输出 illu_fea（中间特征）+ illu_map（光照图）
- 中间通道数 n_fea_middle=31

**您的需求**：确认 IFPN 的 IllumExtract 设计：

| 参数 | 问题 | 建议 |
|:---|:---|:---|
| 输入通道 | `Concat[I_t^down(3ch), Conv1x1(F_t^L)]` 的总通道数？ | 3 + C_f = 51（若 F_t^L 经 Conv1x1 投影到 C_f=48） |
| 中间通道 | 保留 31 还是改为 32/48？ | 改为 32（对齐 TFS-Net 通道体系） |
| 输出通道 | 3（RGB 光照图）还是 1（灰度）？ | 3（与 Retinexformer 一致） |
| 是否输出 illu_fea | IFPN 是否需要中间特征？ | 首版仅输出 illu_map |
| Sigmoid | 是否添加 Sigmoid 约束光照图到 [0,1]？ | **需要确认**：Retinexformer 无 Sigmoid，但 v3 设计可能要求 [0,1] |

---

### R5: 时序一致性损失与 RAFT 光流 [P1]

**背景**：v3answer 建议简化 L_temporal，但仍需光流做 warp。RAFT 是大型预训练模型（~5.3M 参数），推理也需要额外显存。

**选项**：

| 选项 | 描述 | 显存开销 |
|:---|:---|:---|
| **A** | 使用 RAFT（torchvision raft_large，冻结），计算 Flow_{t→i}，遮挡 mask 用 forward-backward consistency | 高（需存储光流 + 前向后向流） |
| **B** | 使用简化光流（如 PWC-Net 或 LiteFlowNet），或直接省略遮挡 mask | 中 |
| **C（推荐）** | **首版省略 L_temporal**，仅用 L_recon + L_perc + L_illum_smooth 训练。后续实验再添加 | 无额外开销 |

**您的需求**：选择首版训练是否包含 L_temporal。推荐选项 C（先跑通基线再迭代）。

---

## 二、需要确认的设计偏好

### R6: LFF 角度调制项的设计选择 [P1]

FRBNet 代码中角度调制极其简单（N=1，无学习参数，λ=0.1 固定）。v3answer 错误地描述为 N=4 且有可学习系数。

**您的需求**：
- 选项 A：直接沿用 FRBNet 的简化设计（N=1，固定系数）→ 参数最少，已有验证
- 选项 B：扩展为可学习角度调制（N=2~4，c_n/s_n 可学习）→ 更灵活但增加参数
- **推荐 A**

### R7: 门控融合方式选择 [P2]

当前 TFSI `GatedFusion` 使用 `Conv1x1 + Sigmoid`。NAFNet 的 `SimpleGate`（通道分半相乘）是另一种无参数方案。

**您的需求**：
- 选项 A：保留当前 Conv1x1 + Sigmoid（有参数，表达力强）
- 选项 B：改用 SimpleGate（无参数，简洁）
- **推荐 A**（保持一致性，不引入新变化）

### R8: L_consist (源强度时序一致性正则) 的处理 [P1]

v3answer 建议首版移除 L_consist（因为需要多次 TFSI 前向）。

**您的需求**：确认首版是否移除 L_consist。
- 选项 A：移除（推荐），λ_2 重分配给 L_temporal 或 L_recon
- 选项 B：保留，用低分辨率近似计算

---

## 三、需要您提供的信息（如有）

### R9: v2 文档中 MRPN 的设计细节 [P2]

v3quest 指出"v2 文档不存在于本代码库"，MRPN 完全无设计。v3answer 给出了一个合理的新设计（残差降权 + 隐式遮挡），但这是 AI 自行设计的，**没有论文验证**。

**您的需求**：
- 若您有 v2 文档中 MRPN 的原始设计，请提供
- 若没有，确认是否接受 v3answer §2.5 的 MRPN 新设计作为首版实现

### R10: 训练超参数偏好 [P2]

以下超参数需要您确认默认值：

| 超参数 | 说明 | v3answer 建议 | 您的偏好 |
|:---|:---|:---|:---|
| τ_mid | NDPN SNR 归一化中心 | 1.0 | ? |
| τ_scale | NDPN SNR 归一化缩放 | 2.0 | ? |
| L_recon 频域权重 | FFT 损失系数 | 0.1 | ? |
| L_perc 权重 | 感知损失系数 | 待定 | ? |
| 学习率 | 初始学习率 | 沿用 v1 配置 | ? |
| batch_size | 训练 batch | 沿用 v1 配置 | ? |

---

## 四、参考代码已就绪确认

### R11: reference_repos 仓库完整性 [已确认 ✅]

以下 5 个仓库已下载到 `e:/TFS-Net/reference_repos/`：

| 仓库 | 状态 | 关键文件已读取 |
|:---|:---|:---|
| FRBNet | ✅ 完整 | `frbnet_utils.py` (RadialBasisFilter + LearnableFreFilter) |
| Retinexformer | ✅ 完整 | `RetinexFormer_arch.py` (Illumination_Estimator) |
| DAT | ✅ 完整 | `dat_blocks.py` (DAttentionBaseline) |
| BasicSR | ✅ 完整 | `edvr_arch.py` (PCDAlignment + TSAFusion) |
| NAFNet | ✅ 完整 | `NAFNet_arch.py` (SimpleGate + NAFBlock) |

**无需额外下载论文代码。** 5 篇参考论文的代码均已到位。

---

## 五、决策汇总表

| 编号 | 决策项 | Claude 选定方案 | 优先级 | 状态 |
|:---|:---|:---|:---|:---|
| R1 | LFF FFT 适配策略 | A 改进版（逐通道 rfft2 + 幅度/相位独立 RBF + Conv1x1 fuse） | P0 | ✅ Claude已决 |
| R2 | W_g 公式选择 | FRBNet 代码版本（高斯带 + DC 硬置零） | P0 | ✅ Claude已决 |
| R3 | SACE cross-attention | A 改进版（DCNv2 单尺度，K=3, n_groups=4） | P0 | ✅ Claude已决 |
| R4 | IllumExtract 适配 | 3层结构 + mean_channel + feat_proj(48→16) + 无Sigmoid | P0 | ✅ Claude已决 |
| R5 | L_temporal / RAFT | 三阶段策略（首版省略，阶段2用 raft_small） | P1 | ✅ Claude已决 |
| R6 | 角度调制设计 | A（沿用 FRBNet 简化版 N=1, λ=0.1 固定） | P1 | ✅ Claude已决 |
| R7 | 门控融合方式 | A（保留 Sigmoid Gate） | P2 | ✅ Claude已决 |
| R8 | L_consist 处理 | A（首版移除） | P1 | ✅ Claude已决 |
| R9 | MRPN 设计来源 | 接受 v3answer §2.5 新设计 | P2 | ✅ Claude已决 |
| R10 | 训练超参数 | 沿用 v3code3 §5.4 建议值 | P2 | ⏳ 用户确认 |

**所有 P0/P1 设计决策已由 Claude 给出方案。** 用户只需确认或提出异议。

---

## 六、v3code3 中发现的新问题（需 Claude 进一步思考）

> 以下问题在审查 v3code3.md 的伪代码时发现，需要在 Phase B 编码前解决。
> **注意：Claude 只有对话接口，无法直接读取文件，提问时需附上相关代码片段。**

### N1: Retinexformer depth_conv 的 groups 参数争议 [P0 阻塞]

**Claude 在 v3code3 §4.3 声称**："`depth_conv` 的 groups 改为 `n_fea_middle`，纠正 v3answer 错误（原 Retinexformer 写 `groups=n_fea_in=4`，但实际应是 `n_fea_middle` 才是真正的 depthwise）"

**但 Retinexformer 原始代码**（`RetinexFormer_arch.py` L102-103）：
```python
self.depth_conv = nn.Conv2d(
    n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
```
原始代码确实写的是 `groups=n_fea_in`（默认 4）。这是一个**分组卷积**（31 通道分成 4 组），不是严格意义的 depthwise（groups=in_channels）。

**问题**：Claude 的"纠正"是否真的正确？
- 若改为 `groups=n_fea_middle`（=31），变成严格 depthwise，**每通道独立卷积**，参数量从 `31*5*5*4/4 ≈ 775` 降到 `31*5*5 = 775`（碰巧相同），但语义不同
- 原始代码用 `groups=n_fea_in=4` 可能是有意设计（分组间有信息交流）
- **建议**：保留原始 `groups=4` 的写法（已验证有效），不"纠正"

### N2: FRBNet RadialBasisFilter 共享 basis 的设计被错误拆分 [P0 阻塞]

**FRBNet 原始代码**：一个 `RadialBasisFilter` 实例同时包含 `coeff_mag` 和 `coeff_phase`，**共享** `mu`、`log_bwh`、`angular_mod`（即共享 basis 函数），仅系数和门控独立：
```python
# frbnet_utils.py L7-47 关键逻辑
basis = torch.exp(-((r_hat - mu)^2) / (2*bwh^2))  # 共享 basis
diff_mag = (gate_mag * coeff_mag * basis).sum(0)    # mag 用独立 coeff
diff_phase = (gate_phase * coeff_phase * basis).sum(0)  # phase 用独立 coeff
```

**v3code3 的设计**：创建**两个独立的** `RadialBasisFilter` 实例 `lff_mag` 和 `lff_phase`，各自有独立的 `mu`、`log_bwh`。

**差异**：
- FRBNet：mag 和 phase 的 RBF **中心位置相同、带宽相同**，仅系数不同
- v3code3：mag 和 phase 的 RBF **中心位置和带宽独立学习**

**建议**：应沿用 FRBNet 的共享 basis 设计（一个 RBF 实例内同时输出 mag 和 phase 响应），物理含义是"同一组频带，不同的幅度/相位处理方式"。

### N3: FRBNet 的 r_hat 归一化到 [0,1] 被遗漏 [P0 阻塞]

**FRBNet 原始代码**（`frbnet_utils.py` L27-28）：
```python
r_hat = torch.sqrt(fx ** 2 + fy ** 2)
r_hat = r_hat / r_hat.max()  # normalize to 0..1
```
`r_hat` 被归一化到 [0,1]，这样 `mu = linspace(0, 1, K)` 才能覆盖整个频率范围。

**v3code3 的 RadialBasisFilter 实现**（§6.2）直接用未归一化的 `r_grid`：
```python
phi = torch.exp(-((r_grid.unsqueeze(0) - self.mu.view(-1, 1, 1)) / bwh) ** 2)
```
`r_grid = sqrt(fx^2 + fy^2)` 的值域是 `[0, ~0.707]`（Nyquist 频率），而 `mu = linspace(0, 1, K)` 的范围是 `[0, 1]`。**μ_k 和 r_grid 值域不匹配**，高频端的基函数永远不会被激活。

**修复**：在 LFF forward 中对 `r_grid` 做 `r_grid / r_grid.max()` 归一化，或在 RBF forward 中内部归一化。

### N4: DeformableCrossAttention 的双重循环效率问题 [P1]

v3code3 §3.3 的 `_deform_sample` 方法使用 `for g in range(G)` + `for k in range(Ks)` 双重 Python 循环：
- G=4, Ks=9 → **36 次 `F.grid_sample` 调用/帧**
- 每次 grid_sample 处理 (B, C/4, H, W)
- 这对训练速度影响极大（grid_sample 是 GPU 操作，Python 循环引入大量 kernel launch 开销）

**建议 Claude 思考**：
- 能否将所有采样点拼接为一次 `grid_sample` 调用？（将 Ks 维度合入 batch 维度）
- 或改用 `torchvision.ops.deform_conv2d` 的 C++ 实现（性能高 10-100 倍）
- 至少提供**优化版和简单版两种实现**，首版可用简单版验证正确性，再切换优化版

### N5: SACE 主类和 IFPN 主类缺少完整设计 [P1]

v3code3 仅给出了子模块代码（`DeformableCrossAttention` 和 `IllumExtract`），但**缺少主类的完整数据流**：
- **SACE 主类**：如何整合 LFF → 时域中位值 → 逐帧 DeformableCrossAttention → 输出 attention maps + mu_t_clean？
- **IFPN 主类**：如何整合 IllumExtract → sim() 计算 → L_ref → 强度调制公式？
- 这些在 v3answer §2.3 和 §2.5 有伪代码，但 v3code3 没有更新为与子模块一致的版本

**建议 Claude 在 Phase B 时提供**：SACE 和 IFPN 的完整 `forward()` 方法伪代码（含所有 shape 注释）

### N6: 训练配置更新细节缺失 [P2]

当前 `configs/sdsd_stage1.yaml` 包含 v1 的配置项（`mins_window_size`, `lambda_pix`, `lambda_ssim`, `lambda_tv`），v3 需要全面更新。v3code3 未给出新的完整 YAML 配置。

**需要 Claude 提供**：
- v3 的完整 `sdsd_stage1.yaml` 配置（model 参数、loss 参数、train 参数）
- `train.py` 和 `infer.py` 需要修改的具体位置和代码

---

## 七、最终状态

**Phase A 设计答疑**：✅ 全部完成（R1-R10 均已决策）
**Phase B 编码前提条件**：
- N1-N3 需 Claude 确认修正（P0 阻塞项，影响 LFF 和 IllumExtract 的正确性）
- N4 需 Claude 提供优化思路（P1，不阻塞但影响训练效率）
- N5-N6 可在编码过程中逐步解决
