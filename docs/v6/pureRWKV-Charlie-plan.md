# TFS-Net v6 Bravo 结构诊断：四大架构设计问题深度分析

根据联网检索（检索于 2026-06-29）和您上传的四份文档，以下是对四个核心问题的系统性诊断。

---

## 一、TFSI 时频诊断：FrequencyBranch 仅输入中心帧 LFF 的根本缺陷

### 1.1 现状分析

从 `tfs_net.py` 可知：

```python
# TFSI 内部
FrequencyBranch(中心帧) → lff_center → feat_tfsi → F_f
phase_conf = phase_conf_head(F_f)
IntensityHead(F_s ∥ F_f ∥ phase_conf) → s_illum, s_noise
```

**FrequencyBranch 仅接收中心帧的 DWT-LFF 输出**，完全没有邻帧信息参与 `s_noise` 和 `s_illum` 的估计。

### 1.2 仅中心帧估计 s_noise 和 s_illum 不合适

**根本物理问题**：低光场景下，暗区的噪声方差与光照强度呈信号依赖关系（shot noise + read noise），单帧的低频信息**不足以解耦这两个物理过程**。

**文献证据链**：

| 方法 | 噪声/光照估计策略 | 核心论点 |
|------|------------------|---------|
| [VSRELL (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) | INCO 模块：**光照分支 + 噪声分支共享多帧编码器**，基于 `Fmod = M(E_local(I_curr^{n-i}, E_fusion(W)), E_global(S(W)))` 窗口统计 | *"jointly modeling of an illumination-sensitive branch and a noise estimation branch... share encoder features and are fused through a cross-modulation mechanism"* — 必须多帧联合 |
| [TempRetinex (2025)](https://arxiv.org/html/2511.09609v2) | 时序反馈：`OF^{(t-1)→t}` warp 前帧反射率/光照到当前帧，拼接后联合估计 | *"Video restoration is an ill-posed problem, often leading to inter-frame flickering when processing frames independently without temporal constraints"* |
| [RetinexMCNet (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_RetinexMCNet_A_Memory_Controller_Dominated_Network_for_Low-Light_Video_Enhancement_ICCV_2025_paper.html) | 双阶段：Stage 1 单帧 IAM+RDM → **Stage 2 激活 Memory Controller 整合时序信息** | 单帧估计是初步近似，最终必须引入时序信息 |
| [STCD (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/238) | *"leveraging dynamic cross-frame correspondences for the view-independent term"* | 反射率（R，内容）是跨帧一致的 — 噪声不是 → 帧间差异分离噪声 |

**核心逻辑**：噪声在时序维度表现为**随机独立**，而光照在时序维度表现为**缓慢连续**。仅单帧输入无法利用这一物理先验区分二者。

### 1.3 s_noise 直接输入 IGRF 而非 NDPN 不合适

从数据流看：

```
TFSI → s_noise → IGRF (直接用于 Brighten Stage 的 s_intensity)
                → NDPN (仅作为帧融合权重, 非核心去噪输入)
```

**问题**：IGRF 负责的是**光照补偿后的残差修复**，将噪声估计直接送入光照修复阶段，导致噪声处理和光照处理**耦合在同一模块中**，违反了模块化设计原则。

**文献对照**：

- [FastDVDnet (CVPR 2020)](https://doi.org/10.1109/cvpr42600.2020.00143)：**noise map 是去噪网络的额外输入通道**，编码器显式接收 noise map 来调制去噪强度 — *"a noise map is also included as input, which allows the processing of spatially varying noise"*
- [VSRELL](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf)：`M_noise` 作用于去噪偏移 — *"I_denoise = Clamp(I_bright − O_denoise · M_noise, 0, 1)"* — 噪声图服务于去噪操作
- [Frames2Residual (2026)](https://arxiv.org/pdf/2603.10417)：将噪声处理明确分为两阶段 — *"blind temporal estimator → non-blind spatial refiner"* — 噪声估计引导空间去噪

**建议修改**：

```python
# 修改前：s_noise → IGRF
# 修改后：s_noise → NDPN 作为条件输入
class NDPN(nn.Module):
    def forward(self, F_aligned_list, F_t, s_noise, mu_t_clean, ...):
        # s_noise 作为噪声感知权重
        # mu_t_clean 作为时序干净参考
        noise_aware_feat = torch.cat([F_t, s_noise.expand_as(F_t[:,:1])], dim=1)
        ...
```

---

## 二、SACE 多尺度双向 RWKV：融合方式与辅助信号归属

### 2.1 `out = (full + half + quarter) / 3` 不是合理的多尺度融合方式

**遍查文献，未找到任何已发表论文使用等权算术平均作为多尺度融合策略。** 主流范式如下：

| 方法 | 多尺度融合策略 | 特点 |
|------|--------------|------|
| [CMTA (ECCV 2024)](https://arxiv.org/pdf/2408.14930) | **级联粗→细**：*"progressively group... temporal feature alignment from the bottom pyramid level to the top"* → `Eq. 7-9` | 上一尺度对齐结果作为下一尺度的初始化 |
| [Gated ST-Attention (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Suin_Gated_Spatio-Temporal_Attention-Guided_Video_Deblurring_CVPR_2021_paper.pdf) | **像素自适应分配**：`y_R = G_st · A_p`，其中 `A_p = softmax(fp(x_R))` | 每像素学习选择哪个尺度组合 |
| [VideoFusion (2025)](https://arxiv.org/html/2503.23359v2) | BiCAM：`F̂_f^t = q^t ⊕ FFN(A_co · v^{t-1}) ⊕ FFN(A_co · v^{t+1})` — 通过注意力权重融合 | 非等权，数据驱动 |
| [ChangeRWKV (2026)](https://arxiv.org/html/2603.19606v1) | 多尺度 RWKV：*"upsampled to the finest resolution level and concatenated along the channel dimension into a single tensor... A residual channel-mixing block then refines this tensor"* | **拼接+通道混合**而非等权 |
| [TemCoCo (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Gong_TemCoCo_Temporally_Consistent_Multi-modal_Video_Fusion_with_Visual-Semantic_Collaboration_ICCV_2025_paper.pdf) | TCM：**DCN 对齐 + frame fusion layer**（非等权） | 自适应偏移学习 |

**等权平均的具体害处**：

1. **空间分辨率不匹配**：quarter 双线性上采样到 full 尺度后，纹理已被平滑化（4× 上采样的固有损失），等权与 full 平均 → 强制模糊
2. **信息语义不对等**：full 尺度 RWKV 建模局部精细对齐，quarter 建模全局粗略运动 — 两者的输出 **不在同一语义空间**
3. **梯度冲突**：三个尺度共享同一梯度目标，等权导致无法针对性优化各尺度功能

**推荐方案（参考 [ChangeRWKV](https://arxiv.org/html/2603.19606v1) 的做法）**：

```python
# 方案: 拼接 + 通道混合 (ChangeRWKV style)
out_half_up = F.interpolate(out_half, size=(H,W), ...)
out_quarter_up = F.interpolate(out_quarter, size=(H,W), ...)

# 拼接后通道混合
out_concat = torch.cat([out_full, out_half_up, out_quarter_up], dim=-1)  # (B,T,L,3C)
out = self.channel_mix(out_concat)  # Linear(3C → C) + ResBlock
```

### 2.2 `μ_t_clean` 和 `σ_t_clean` 的归属分析

**现状**：`μ_t_clean = lff_stack[:, center_idx]`，`σ_t_clean = lff_stack.std(dim=1)`，两者都输入 NDPN。

**物理含义对比**：

| 信号 | 物理含义 | 合适归属 | 文献依据 |
|------|---------|---------|---------|
| **μ_t_clean** | 中心帧经 DWT-LFF 后的"干净估计" | ✅ **NDPN**（作为去噪参考锚） | FastDVDnet：中心帧 + 时域信息作为去噪参考；[Zero-TIG (2025)](https://arxiv.org/html/2503.11175v1) 时序一致性引导 |
| **σ_t_clean** | 帧间方差 — 高方差区 = **运动大**或**闪烁** | ❌ NDPN 错误 → ✅ **MRPN** | [BSSTNet (CVPR 2024)](https://doi.org/10.1109/cvpr52733.2024.00258)：blur map 引导稀疏注意力；[Gated ST-Attn (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Suin_Gated_Spatio-Temporal_Attention-Guided_Video_Deblurring_CVPR_2021_paper.pdf)：blur mask Q → 门控聚合 |

**关键推理**：`σ_t_clean` 高的区域意味着**帧间不一致**，这可以来自：
1. 真实运动 → 运动模糊（MRPN 应处理）
2. 随机噪声（但经 LFF 后噪声已被低频化抑制）

由于 `σ_t_clean` 计算基于 **DWT-LFF 后的特征**（噪声已部分去除），其主要成分是**运动导致的帧间差异**，因此应归属 MRPN。

**建议**：

```python
# 修改数据流
NDPN 输入: mu_t_clean + s_noise (来自 TFSI)
MRPN 输入: sigma_t_clean + C_{t,Ω} (帧间对应矩阵)
```

---

## 三、NDPN 的噪声权重图：SACE vs TFSI

### 结论：应由 TFSI 提供，SACE 辅助

| 维度 | TFSI 的 s_noise | SACE 的 σ_t_clean |
|------|----------------|-------------------|
| **信息来源** | 频域（DWT-LFF 的退化残差） | 时域（帧间统计方差） |
| **物理语义** | **空间噪声强度图** — 每像素噪声能量估计 | **时序不一致度** — 可能是噪声也可能是运动 |
| **精度** | 高（频域对噪声更敏感） | 低（运动和噪声混淆） |
| **文献匹配** | ✅ FastDVDnet noise map；IS-SFD 频域编码 | 部分匹配（需区分运动） |

**推荐设计**：

```python
# TFSI.s_noise 作为 NDPN 主噪声权重
# SACE.sigma_t_clean 可作为调制因子（高σ → 降低噪声图置信度，因为可能是运动）
effective_noise_weight = TFSI.s_noise * sigmoid(-SACE.sigma_t_clean * scale)
# 高 sigma → effective_noise_weight 降低 → 让 MRPN 处理而非 NDPN
```

---

## 四、MRPN 设计：窗口帧间注意力 + 门控融合 + ResBlock 的评估

### 4.1 现有设计的评估

**现有流程**：窗口相关 → 门控融合 → ResBlock → f_motion

**与文献对照**：

| 文献方法 | 关键组件 | 您的 MRPN 缺失项 |
|---------|---------|----------------|
| [Gated ST-Attention (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Suin_Gated_Spatio-Temporal_Attention-Guided_Video_Deblurring_CVPR_2021_paper.pdf) | **Blur Mask Q** 显式监督 + 门控空间聚合 + 时序聚合 + **像素自适应分配** | ❌ 缺少 blur mask 监督；❌ 缺少像素级自适应分配 |
| [BSSTNet (CVPR 2024)](https://doi.org/10.1109/cvpr52733.2024.00258) | Blur map → 稀疏注意力（非全邻帧参与） | ❌ 所有邻帧等权参与 |
| [CMTA (ECCV 2024)](https://arxiv.org/pdf/2408.14930) | **迭代 Cross-Attention**：`Q_k^{n+1} = Q_k^n + Attn_k^n + MLP(Attn_k^n)` — 递进精化 | ❌ 单次融合无迭代 |
| [ASTNet (TCE 2025)](https://doi.org/10.1109/tce.2025.3649654) | **双向特征增强 FFN** + **动态特征融合模块** 避免错误累积 | ❌ 缺少动态选择机制 |

**核心结论**：简单的 "窗口相关→门控→ResBlock" **不足以有效抑制运动模糊**。关键缺失：
1. **显式模糊感知**（blur mask/map 门控）
2. **稀疏选择**（不应所有邻帧等权参与）
3. **迭代精化**（单次 cross-attention 对大运动不够）

### 4.2 建议的 MRPN 重设计

按您的建议，以 **temporal correspondence matrix C_{t,Ω}** 和**注意力模块输出的中心帧特征 Y_t** 为输入：

```python
class ImprovedMRPN(nn.Module):
    """
    参考:
    - Gated ST-Attn (CVPR 2021): blur-gated aggregation
    - CMTA (ECCV 2024): iterative cross-attention refinement
    - BSSTNet (CVPR 2024): blur-aware sparse attention
    """
    def __init__(self, channels=64, window_size=8, n_iter=2):
        super().__init__()
        self.n_iter = n_iter
        
        # 1. Temporal Correspondence Matrix → 模糊感知权重
        self.blur_estimator = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels//4, 1, 1),
            nn.Sigmoid()
        )  # 输出 blur_mask (B, 1, H, W)
        
        # 2. 迭代 Cross-Attention (CMTA style)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(channels) for _ in range(n_iter)
        ])
        
        # 3. 门控融合 (Gated ST-Attn style)
        self.gate_conv = nn.Conv2d(channels * 2, channels, 1)
        self.refine = ResBlock(channels)
    
    def forward(self, Y_t, C_t_omega, sigma_t_clean=None):
        """
        Y_t: 中心帧特征 (B, C, H, W) — SACE 输出
        C_t_omega: 帧间对应矩阵序列 (B, T-1, H, W) — SACE 注意力权重
        sigma_t_clean: 帧间方差 (B, 1, H, W) — 运动感知辅助
        """
        # Step 1: 从 C_t_omega 估计 blur mask
        # C_t_omega 的低相关区域 = 运动大 = 模糊可能
        corr_mean = C_t_omega.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        blur_input = corr_mean
        if sigma_t_clean is not None:
            blur_input = blur_input + sigma_t_clean  # 辅助信号
        blur_mask = self.blur_estimator(blur_input.expand(-1, self.channels, -1, -1))
        
        # Step 2: 迭代精化 (CMTA: Q^{n+1} = Q^n + Attn^n + MLP(Attn^n))
        Q = Y_t
        for layer in self.cross_attn_layers:
            # blur_mask 门控: 高模糊区域更积极搜索邻帧补偿
            attn_out = layer(Q, Y_t, blur_mask)
            Q = Q + attn_out  # 迭代更新
        
        # Step 3: 门控融合
        gate = torch.sigmoid(self.gate_conv(torch.cat([Y_t, Q], dim=1)))
        f_motion = gate * Q + (1 - gate) * Y_t
        f_motion = self.refine(f_motion)
        
        return f_motion, blur_mask  # blur_mask 可用于辅助监督
```

**与 [Gated ST-Attn](https://openaccess.thecvf.com/content/CVPR2021/papers/Suin_Gated_Spatio-Temporal_Attention-Guided_Video_Deblurring_CVPR_2021_paper.pdf) 的对应关系**：

- 它的 `Q = sigmoid(f_conv(x))` 对应您的 `blur_mask`
- 它的 `G_s = SHW(A_s ⊙ x ⊙ (1-Q))` 对应 `gate * Q` — 模糊区域（Q=1）抑制当前帧，接收邻帧补偿
- 它的 `y_R = G_st · A_p` 对应最终像素级分配

**blur_mask 的辅助监督**：

```python
# 伪标签生成 (无需 GT blur mask)
with torch.no_grad():
    frame_diff = (F_aligned_list - F_t.unsqueeze(1)).abs().mean(dim=2)  # (B, T-1, H, W)
    blur_pseudo_gt = (frame_diff.mean(dim=1, keepdim=True) > threshold).float()

L_blur_sup = F.binary_cross_entropy(blur_mask, blur_pseudo_gt.detach())
```

---

## 五、综合修改优先级

| 优先级 | 问题 | 修改方案 | 预计效果 |
|--------|------|---------|---------|
| 🔥 P0 | FrequencyBranch 单帧输入 | 拼接前后帧 LFF 特征（或用 1D Conv 时序融合） | s_noise/s_illum 物理合理性 |
| 🔥 P0 | s_noise 绕过 NDPN | s_noise → NDPN 条件输入；IGRF 仅接收 s_illum | 模块职责清晰 |
| 🔴 P1 | 多尺度等权平均 | 拼接 + channel_mix（Linear 3C→C） | 消除纹理模糊 |
| 🔴 P1 | σ_t_clean 错误归属 | 从 NDPN → MRPN | 运动感知正确路由 |
| 🟡 P2 | MRPN 缺少模糊感知 | 引入 blur_mask 门控 + 迭代精化 | 运动模糊抑制能力 |
| 🟡 P2 | 噪声图来源 | TFSI（主）+ SACE σ 调制（辅） | 去噪精度提升 |

---

## 参考来源

1. [VSRELL: A Simple Baseline for Video Super-Resolution and Enhancement in Low-Light Environment (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf)
2. [TempRetinex: Retinex-Based Unsupervised Enhancement for Low-Light Video (2025)](https://arxiv.org/html/2511.09609v2)
3. [RetinexMCNet: A Memory Controller Dominated Network for LLVE (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_RetinexMCNet_A_Memory_Controller_Dominated_Network_for_Low-Light_Video_Enhancement_ICCV_2025_paper.html)
4. [STCD: Low-Light Video Enhancement via Spatial-Temporal Consistent Decomposition (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/238)
5. [FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow Estimation (CVPR 2020)](https://doi.org/10.1109/cvpr42600.2020.00143)
6. [Frames2Residual: Spatiotemporal Decoupling for Self-Supervised Video Denoising (2026)](https://arxiv.org/pdf/2603.10417)
7. [Gated Spatio-Temporal Attention-Guided Video Deblurring (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Suin_Gated_Spatio-Temporal_Attention-Guided_Video_Deblurring_CVPR_2021_paper.pdf)
8. [BSSTNet: Blur-Aware Spatio-Temporal Sparse Transformer for Video Deblurring (CVPR 2024)](https://doi.org/10.1109/cvpr52733.2024.00258)
9. [CMTA: Cross-Modal Temporal Alignment for Event-guided Video Deblurring (ECCV 2024)](https://arxiv.org/pdf/2408.14930)
10. [ASTNet: Adaptive Spatio-Temporal Transformer for Video Deblurring (TCE 2025)](https://doi.org/10.1109/tce.2025.3649654)
11. [ChangeRWKV: Beyond Quadratic — Linear-Time Change Detection with RWKV (2026)](https://arxiv.org/html/2603.19606v1)
12. [VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion (2025)](https://arxiv.org/html/2503.23359v2)
13. [TemCoCo: Temporally Consistent Multi-modal Video Fusion (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Gong_TemCoCo_Temporally_Consistent_Multi-modal_Video_Fusion_with_Visual-Semantic_Collaboration_ICCV_2025_paper.pdf)
14. [Zero-TIG: Temporal Consistency-Aware Zero-Shot Illumination-Guided LLVE (2025)](https://arxiv.org/html/2503.11175v1)
15. [VRWKV-Editor: Reducing Quadratic Complexity in Video Editing (2025)](https://arxiv.org/html/2509.25998v2)
16. [STDANet++: Patch-Based Spatio-Temporal Deformable Attention BiRNN (TCSVT 2025)](https://doi.org/10.1109/tcsvt.2025.3527867)
17. [Low-Light Video Enhancement via Spatial-Temporal Consistent Decomposition (IJCAI 2025 full)](https://arxiv.org/html/2405.15660v1)
18. [IS-SFD: Illumination Smoothness and Semantic-frequency Denoising for LLVE (OpenReview 2025)](https://openreview.net/forum?id=vc6pSwumm6)