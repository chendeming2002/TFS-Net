# TFS-Net v6 架构修改方案：基于已发表论文证据的系统性重设计

根据联网检索（检索于 2026-06-28）和您上传的三份文档，以下是针对四个修改点的完整方案。

---

## 一、Encoder 输出分流 + DWT-LFF 拆分 + SACE V 投影修改

### 1.1 设计依据（已发表论文）

| 论文 | 出处 | 核心证据 |
|------|------|---------|
| **VSRELL** | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) | "we take the **central frame as the anchor** to construct temporally symmetric windows around it" |
| **STCD** | [IJCAI 2025](https://www.ijcai.org/proceedings/2025/0238.pdf) | view-independent 项 R 跨帧一致 → 中心帧 R 可作为 V 投影的"干净内容基" |
| **VSRELL** | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) | "memory-gated cross-frame feature fusion strategy that **dynamically assesses the illumination confidence**" → 中心/邻居差异化处理 |

### 1.2 修改后的数据流

```
Encoder 共享权重 → F_stack (B, T, 64, H, W)
    ├── F_t = F_stack[:, T//2]                    # 中心帧 (B, 64, H, W)
    │    → SpatialDWTLFFAdapter_center            # ★ 独立分支
    │    → F_lff_t (B, 64, H, W) + feat_tfsi_t
    │
    └── F_Ω = F_stack[:, ≠ T//2]                  # 邻居帧 (B, T-1, 64, H, W)
         → SpatialDWTLFFAdapter_neighbor (per-frame) # ★ 独立分支
         → F_lff_Ω (B, T-1, 64, H, W)

SACE 注意力修改:
    Q = LayerNorm2d(F_lff_t)                      # Query: 中心帧锚定（VSRELL）
    K = LayerNorm2d(F_lff_Ω[t])  per frame        # Key: 邻居帧（用于对齐查询）
    V = ConvV(F_t)  ← ★ 直接取 Encoder 中心帧原始特征
                       而非 F_lff_t 或邻居帧
```

### 1.3 V 投影用 Encoder 原始中心帧的关键理由

**STCD 的视角分解原理**告诉我们：DWT-LFF 已经做了"光照归一化"（α·LL 软分配），但这是为了**对齐计算**（Q/K 空间），不应作为最终融合的"内容源"。V 应该携带**未被归一化破坏的原始结构信息**。

```python
class ModifiedSACE(nn.Module):
    def forward(self, f_stack_raw, lff_t, lff_omega, s_noise):
        # f_stack_raw: Encoder 原始输出 (B, T, C, H, W)
        center_idx = f_stack_raw.shape[1] // 2
        f_center_raw = f_stack_raw[:, center_idx]  # ★ V 直接取原始中心帧
        
        # Q/K 在 DWT-LFF 归一化空间计算（对齐用）
        q = self.q_proj(self.norm_q(lff_t))
        k_list = [self.k_proj(self.norm_k(lff_omega[:,t])) for t in range(T-1)]
        
        # ★ V 在 Encoder 原始空间（保留内容）
        v = self.v_proj(f_center_raw)
        
        # 多尺度双向 RWKV 聚合
        out = self.bidirectional_rwkv(q, k_list, v)
        
        # 边缘门控残差
        F_aligned = out + (1 - s_noise) * lff_omega
        return F_aligned
```

### 1.4 双 DWT-LFF 分支的差异化设计

| 分支 | 输入 | α 初始化 | 用途 |
|------|------|---------|------|
| `DWT_LFF_center` | 中心帧 F_t | α init = **0.6** (略偏 LL_ref) | 提供干净锚定特征供 Q + V 参考 |
| `DWT_LFF_neighbor` | 邻居帧 F_Ω | α init = **0.4** (略偏 LL_deg) | 强调退化诊断信号供 K + TFSI |

---

## 二、损失函数重设计：仅引用已发表论文

### 2.1 权重调整依据

根据 **BVI-Lowlight** [arXiv 2024](https://arxiv.org/html/2402.01970v2) 的实证结论：在未对齐数据上 L1 损失导致 PSNR 下降约 5dB，**对齐容忍度排序为 SSIM > VGG > L1**。

### 2.2 修改后的损失配方

| 损失项 | 原 λ | 新 λ | 已发表依据 |
|-------|------|------|-----------|
| **L1 / Charbonnier (L_pix)** | 1.0 | **0.3** | BVI-Lowlight [arXiv 2024](https://arxiv.org/html/2402.01970v2) 实证 L1 对错位最敏感 |
| **VGG 感知 (L_perc)** | 0.2 | **0.8** | TCE-Net [IJCV 2024](https://link.springer.com/article/10.1007/s11263-024-02084-w) + CIDN [TIP 2022](https://ar5iv.labs.arxiv.org/html/2201.03145) 均用 VGG，对齐容忍度高 |
| **SSIM (L_ssim)** | 0.1 | **0.5** | LAN [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Fu_Dancing_in_the_Dark_A_Benchmark_towards_General_Low-light_Video_ICCV_2023_paper.pdf) 主推，BVI 实证容忍度最佳 |
| **Focal Frequency (新增)** | — | **0.2** | FAN [IET IP 2025](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70264) 验证频域差异补偿 |
| **光流 Warp 时间损失 (新增)** | — | **0.15** | StableLLVE [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_Temporal_Consistency_for_Low_Light_Video_Enhancement_From_Single_CVPR_2021_paper.pdf) + TCE-Net [IJCV 2024](https://link.springer.com/article/10.1007/s11263-024-02084-w) |
| L_freq (FFT 幅度对齐) | 0.1 | **0.05** | 降权 — 因为相位不再做严格对齐（见 §3） |
| L_illum_smooth | 0.001 | 0.001 | 保留 |
| L_illum_sup | 0.02 | 0.02 | 保留（v5.9.2 验证有效） |
| L_inter | 0.3 | 0.2 | 略降 |
| L_ifpn_sup | 0.1 | 0.1 | 保留 |

### 2.3 关键变化

```python
L_total = 0.3·Charbonnier(res, GT)          # ↓ 原 1.0 → 0.3
        + 0.8·VGG(relu3_3)                   # ↑ 原 0.2 → 0.8 ★主导
        + 0.5·(1-SSIM)                       # ↑ 原 0.1 → 0.5
        + 0.2·FocalFreq(res, GT)             # ★新增
        + 0.15·WarpLoss(res_t, res_{t±1})    # ★新增
        + 0.05·L_freq                        # ↓ 降权
        + 0.001·L_illum_smooth + 0.02·L_illum_sup
        + 0.2·L_inter + 0.1·L_ifpn_sup
```

**移除项**：原 v5.9.2 设计中如有强像素级 L2/L1 中间监督，按 BVI-Lowlight 结论应一并降权或替换为 SSIM 版本。

---

## 三、TFSI 模块修改：处理低光视频的相位偏移问题

### 3.1 您的判断完全正确，已发表论文证据链

| 论文 | 出处 | 核心证据 |
|------|------|---------|
| **STA-SUNet** | [arXiv 2024](https://arxiv.org/html/2403.02408v3) | 直接承认低光视频 GT 存在"slightly shifted and distorted"运动模糊偏移 |
| **FDN** | [IEEE TIP 2025](https://doi.org/10.1109/tip.2025.3592559) | "**blur degradation is characterized by phase correlation**"——模糊退化在相位中 |
| **FourierDiff** | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Lv_Fourier_Priors-Guided_Diffusion_for_Zero-Shot_Joint_Low-Light_Enhancement_and_Deblurring_CVPR_2024_paper.pdf) | "motion information is encoded as **repeated image edges in phases**" |
| **VSRELL** | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) | "noise in dark regions directly **distorts color phases**"——低光噪声扭曲相位 |
| **FAN** | [IET IP 2025](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70264) | "brightness manifests as amplitude, while **noise is closely related to phase**" |

**核心结论**：在低光视频任务中，相位 = 结构 + 运动模糊 + 噪声扭曲三者混合，**不能假设 LL 与 GT 相位严格一致**。

### 3.2 修改方案：相位不确定性建模

当前 v6.4 的 TFSI 走的是 **空间域 DWT-LFF**（已经避开 FFT 全局性问题），但在 FrequencyBranch 仍有隐含的频域相位假设。修改如下：

```python
class ModifiedTFSI(nn.Module):
    """
    v6.5 TFSI: 相位不确定性建模
    参考: FAN [IET IP 2025] + FourierDiff [CVPR 2024] + FDN [TIP 2025]
    """
    def __init__(self, C=64):
        super().__init__()
        # SpatialBranch 完全保留（v5.6 soft_median 路径）
        self.spatial_branch = SpatialBranch(C)
        
        # FrequencyBranch 改造: 双轨幅度-相位分离
        self.dwt_lff = SpatialDWTLFFAdapter(C)  # 空间域处理（保留）
        
        # ★ 新增: 相位置信度估计（仅作用于 Conv-amp 路径）
        self.phase_conf_head = nn.Sequential(
            nn.Conv2d(C, C//2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(C//2, 1, 1),
            nn.Sigmoid()
        )
        
        # 幅度增强分支（亮度信号 — 保留严格一致性假设）
        self.amp_enhance = nn.Conv2d(C, C, 3, 1, 1)
        
        # IntensityHead 增加 phase_unreliability 通道
        self.intensity_head = nn.Conv2d(C * 2 + 1, 2, 1)  # +1 for phase_conf
    
    def forward(self, feats_center):
        # 1. SpatialBranch (保留)
        F_s = self.spatial_branch(feats_center)
        
        # 2. DWT-LFF 空间域（保留 v6.4 设计）
        out = self.dwt_lff(feats_center)
        feat_tfsi = out['feat_tfsi']  # IDWT((1-α)·LL, ½HF) 退化残差
        
        # 3. ★ 相位置信度估计 — 在残差特征上做
        #    高残差 + 高频成分丰富 → 相位可能含运动偏移
        phase_conf = self.phase_conf_head(feat_tfsi)  # (B, 1, H, W)
        
        # 4. 幅度路径增强（亮度信号严格 LL→GT 对应）
        F_amp = self.amp_enhance(feat_tfsi)
        
        # 5. 融合: phase_conf 作为额外通道
        F_fused = torch.cat([F_s, F_amp, phase_conf], dim=1)
        s_illum, s_noise_raw = self.intensity_head(F_fused).chunk(2, dim=1)
        s_illum = torch.sigmoid(s_illum)
        
        # ★ s_noise 由 phase_conf 调制: 相位不可靠区域 → s_noise 提高
        s_noise = torch.sigmoid(s_noise_raw) * (1 + 0.5 * (1 - phase_conf))
        s_noise = s_noise.clamp(0, 1)
        
        return s_illum, s_noise, phase_conf
```

### 3.3 修改前后对比

| 方面 | 原 v6.4 TFSI | 修改后 v6.5 TFSI |
|------|-------------|-----------------|
| 相位假设 | 隐含 LL-GT 一致 | 不一致，引入 phase_conf |
| 幅度处理 | 直接增强 | 保留（亮度仍主要在幅度） |
| s_noise 含义 | 仅时域噪声 | **时域噪声 + 相位不可靠度** |
| 下游用途 | 仅 NDPN 用 | s_noise 同时调制 SACE 残差 + NDPN |
| 文献依据 | — | FAN 2025 + FDN 2025 + FourierDiff 2024 |

### 3.4 配套修改：原 L_freq 损失项

原 L_freq = L1(|FFT(res_t)|, |FFT(GT)|) **只对齐幅度**，已经天然地规避了相位一致性问题——这一点是合理的，所以保留但降权。**不应**新增任何 FFT 相位 L1/L2 损失。

---

## 四、整体框架梳理与通道精简

### 4.1 v6.5 完整数据流（修改后）

```
输入: y (B, T, 3, H, W) 低光视频
  │
  ├─ Encoder (3级金字塔) → F_stack (B, T, 64, H, W)
  │     ├─ F_t = F_stack[:, T//2]  (中心帧原始特征, 保留供 SACE V)
  │     └─ F_Ω = F_stack[:, ≠T//2] (邻居帧原始特征)
  │
  ├─ DWT-LFF 分流 (★修改)
  │     ├─ DWT_LFF_center(F_t) → F_lff_t, feat_tfsi_t
  │     └─ DWT_LFF_neighbor(F_Ω, per-frame) → F_lff_Ω
  │
  ├─ TFSI (★修改: 相位不确定性)
  │     输入: feats_center = F_stack[:, T//2]
  │     输出: s_illum, s_noise, phase_conf  (B, 1, H, W) × 3
  │
  ├─ SACE (★修改: V取原始中心帧)
  │     Q ← F_lff_t (归一化锚定)
  │     K ← F_lff_Ω (邻居对齐键)
  │     V ← F_t (Encoder原始中心帧)  ★
  │     多尺度双向 RWKV + 边缘门控 (v6.5 PureRWKV) → F_aligned (B, T, 64, H, W)
  │
  ├─ IFPN → lit_up_map_raw, f_illum_feat, ifpn_side
  ├─ NDPN (用增强后的 s_noise) → f_noise_out
  ├─ MRPN → f_motion_out
  │
  └─ IGRF (3阶段) → res_t (B, 3, H, W)
```

### 4.2 可删除/精简的通道（综合精简）

| 原 v6.4 组件 | 处理 | 理由 |
|-------------|------|------|
| **DAT (DeformableCrossAttention)** | ❌ **删除** | v6.5 已论证纯 RWKV 多尺度可替代；省 130K 参数 |
| **soft-median (μ_t_clean 用)** | ❌ 已在 v6.3 移除 | 中心帧直接取代 |
| **AmpEnhance (未启用)** | ❌ **删除** | 文档显示未启用，清理代码 |
| **DWT-LFF 共享实例** | ✂ **拆为 2 个独立实例** | 中心/邻居语义不同，不应共享 |
| **TFSI 内 FrequencyBranch 的旧 FFT 路径残留** | ❌ 清理 | v6.4 已转空间域，遗留代码删除 |
| **L_freq 项** | ⬇ 降权（不删除） | 仍提供幅度对齐信号 |
| **FFT 相位 L1（如存在）** | ❌ 严禁 | 违反 §3 相位不一致结论 |
| **lit_up_delta 中的冗余 Conv** | ✂ 简化 | IFPN 内部精简 |
| **IGRF 内 illum_corr 的高维通道** | ✂ 减半 | 因 s_illum 已经更精确 |

### 4.3 v6.5 参数量重估

| 模块 | v6.4 | v6.5 (修改后) | 变化 |
|------|------|--------------|------|
| Encoder | 320K | 320K | — |
| TFSI | 100K | **108K** | +8K (phase_conf head) |
| SACE (DAT) | 130K | **0** | -130K ★大幅精简 |
| SACE (Cross-RWKV) | 110K | **160K** | +50K (多尺度双向) |
| DWT-LFF | 18K | **34K** | +16K (拆为2个实例) |
| IFPN | 150K | 140K | -10K (简化) |
| NDPN | 70K | 70K | — |
| MRPN | 50K | 50K | — |
| IGRF | 130K | 120K | -10K (illum_corr 减半) |
| AmpEnhance (未启用) | 91K | **0** | -91K ★清理 |
| **总计** | **1.169M** | **~1.00M** | **-14.4%** |

### 4.4 v6.5 设计原则总结

1. **中心帧锚定**（VSRELL [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf)）：F_t 全局贯穿 Q、V、TFSI 输入
2. **归一化-原始双空间分离**（STCD [IJCAI 2025](https://www.ijcai.org/proceedings/2025/0238.pdf)）：DWT-LFF 空间做对齐，Encoder 原始空间做内容融合
3. **相位不一致建模**（FDN [TIP 2025](https://doi.org/10.1109/tip.2025.3592559) + FourierDiff [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Lv_Fourier_Priors-Guided_Diffusion_for_Zero-Shot_Joint_Low-Light_Enhancement_and_Deblurring_CVPR_2024_paper.pdf) + FAN [IET IP 2025](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70264)）：phase_conf 通道贯穿 s_noise
4. **对齐容忍损失主导**（BVI-Lowlight [arXiv 2024](https://arxiv.org/html/2402.01970v2) + TCE-Net [IJCV 2024](https://link.springer.com/article/10.1007/s11263-024-02084-w)）：VGG/SSIM 主导，L1 降权
5. **纯 RWKV 对齐**（您的 pureRWKV.md 综合论证）：去 DAT 省 130K 参数

---

## 五、实施状态（2026-06-28 更新）

| 优先级 | 修改项 | 状态 | 文件 | 说明 |
|-------|-------|------|------|------|
| 🔥 P0 | 损失权重: VGG 0.4 + SSIM 0.3 + pix 0.5 | ✅ 已实施 | configs/v6_bravo.yaml, losses/losses.py | 均衡版(1.4:1), VGG+SSIM温和主导 |
| 🔥 P0 | TFSI phase_conf + s_noise 调制 | ✅ 已实施 | tfsi.py | phase_conf_head → IntensityHead |
| ⚡ P1 | DWT-LFF 拆分 (α 0.6/0.4) | ✅ 已实施 | pure_rwkv_sace.py:50-51 | lff_center + lff_neighbor 独立实例 |
| ⚡ P1 | V 源 = Encoder 原始中心帧 | ✅ 已实施 | pure_rwkv_sace.py:150 | f_raw_center 替换 lff_stack |
| ⚡ P1 | 删除 DAT | ✅ v6.5 已移除 | pure_rwkv_sace.py | PureRWKVSACE |
| ⚡ P1 | 删除 AmpEnhance | ⏸ 代码保留 | — | config 已关闭 |
| 🔧 P2 | IFPN/IGRF 精简 | ⏸ | — | 边际收益 |
| — | size-mismatch ckpt 加载 | ✅ 已实施 | train.py | 过滤 shape 不匹配的 key |

### 五-B：训练实证（2026-06-28）

**Run 1（崩塌）**: lr=0.001, VGG 0.8+SSIM 0.5+pix 0.3
- ep5 PSNR=19.79 → ep10 暴跌 11.32 (-8.5dB) → ep15 锁定 10.85
- **教训**: VGG+SSIM=1.3 碾压 pix=0.3 (4.3:1), warmup 结束后 lr 跳升触发放大。**损失权重必须逐项消融。**

**Run 2（均衡版，进行中）**: lr=0.0008, VGG 0.4+SSIM 0.3+pix 0.5, warmup=5
- VGG+SSIM=0.7 : pix=0.5 = 1.4:1, 温和主导
- loss 正常下降 (0.489→0.446), 无崩塌迹象
- ep5 验证约 4h 后出

---

## 参考来源

1. [VSRELL: A Simple Baseline for Video Super-Resolution and Enhancement in Low-Light Environment (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf)
2. [Low-Light Video Enhancement via Spatial-Temporal Consistent Decomposition / STCD (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/0238.pdf)
3. [Fourier-Based Decoupling Network for Joint Low-Light Image Enhancement and Deblurring / FDN (IEEE TIP 2025)](https://doi.org/10.1109/tip.2025.3592559)
4. [FDN GitHub Repository](https://github.com/Jabruson/FDN-TIP2025)
5. [Fourier Priors-Guided Diffusion for Zero-Shot Joint LLE and Deblurring / FourierDiff (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Lv_Fourier_Priors-Guided_Diffusion_for_Zero-Shot_Joint_Low-Light_Enhancement_and_Deblurring_CVPR_2024_paper.pdf)
6. [FAN: A Fourier Alignment Network for Low-Light Image Enhancement (IET Image Processing 2025)](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70264)
7. [A Spatio-Temporal Aligned SUNet Model for Low-Light Video Enhancement (arXiv 2024)](https://arxiv.org/html/2403.02408v3)
8. [BVI-Lowlight: Fully Registered Benchmark Dataset (arXiv 2024)](https://arxiv.org/html/2402.01970v2)
9. [Temporally Consistent Enhancement of LLV / TCE-Net (IJCV 2024)](https://link.springer.com/article/10.1007/s11263-024-02084-w)
10. [Learning Temporal Consistency for LLVE from Single Images / StableLLVE (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_Temporal_Consistency_for_Low_Light_Video_Enhancement_From_Single_CVPR_2021_paper.pdf)
11. [Dancing in the Dark: A Benchmark towards General LLVE / LAN (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Fu_Dancing_in_the_Dark_A_Benchmark_towards_General_Low-light_Video_ICCV_2023_paper.pdf)
12. [Enhancing Low-Light Images via Cross-Image Disentanglement / CIDN (TIP 2022)](https://ar5iv.labs.arxiv.org/html/2201.03145)