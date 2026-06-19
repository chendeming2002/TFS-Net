# TFS-Net v4 架构设计文档

## 1. 总体概述

TFS-Net v4 是针对 v3.2 训练停滞（23 epoch，PSNR=4.83 dB，loss 恒定 0.963）的根本性架构修复版本。

**核心问题**：v3 的 IGRF 归一化加权融合（w_base≈0.44，branches≈0.12-0.32）严重衰减三源分支梯度，导致 IFPN/NDPN/MRPN 梯度比 IGRF 弱 13-90 倍，分支形同虚设，模型退化为 encoder+IGRF 两层网络。

**v4 核心修复**：移除加权融合，改为 concat+conv 拼接融合；将共享 aux_proj 替换为 3 个独立 BranchReconHead。

---

## 2. 五阶段数据流

```
输入: x (B, T=5, 3, H, W)
│
├── Stage 0: PyramidEncoder
│   多帧 → 三尺度金字塔特征 + 全分辨率融合
│   feats (B,T,64,H,W)  +  coarse_feats (B,T,128,H/4,W/4)
│
├── Stage 1: TFSI（时频源指示器）
│   feats → SpatialBranch(μ_t, σ_t, SNR) + FrequencyBranch(LFF)
│         → GatedFusion → IntensityHead
│   输出: F_fused, s_illum/s_noise/s_motion ∈ [0,1], μ_t, σ_t, SNR
│
├── Stage 2: SACE（可变形跨帧对齐，共享 LFF）
│   feats + TFSI输出 → 可变形卷积对齐
│   输出: μ_t_clean, F_aligned_list, attn_maps
│
├── Stage 3: 三源恢复分支（并行）
│   ├── IFPN: 光照恢复，条件=s_illum，输入=coarse_feats, feats
│   │         输出: f_illum_out (B,64,H,W)
│   ├── NDPN: 噪声去除，条件=s_noise，输入=F_aligned_list, μ_t_clean, σ_t
│   │         输出: f_noise_out (B,64,H,W)
│   └── MRPN: 运动补偿，条件=s_motion，输入=F_aligned_list
│             输出: f_motion_out (B,64,H,W)
│
└── Stage 4: IGRF v4（拼接融合重建）
    [F_fused, f_illum_out, f_noise_out, f_motion_out]
    → concat (B,256,H,W) → Conv(256→64)+GELU → 4×ResBlock → delta
    → res_t = clamp(image_center + delta, 0, 1)
```

---

## 3. 各模块详细说明

### 3.1 TFSI（时频源指示器）

**功能**：估计每个空间位置的三源退化强度。

**数据流**：
```
feats (B,T,64,H,W) → LayerNorm（逐帧）
    ├── SpatialBranch:
    │   μ_t = Median_t(feats)       # 时域中位值，结构先验
    │   σ_t = Std_t(feats)          # 时域标准差，噪声/运动度量
    │   SNR = μ_t / (σ_t + ε)      # 信噪比估计
    │   F_s = Conv(Concat[μ_t, σ_t, SNR])  # (B,64,H,W)
    │
    └── FrequencyBranch:
        F_f = LFF(feats[:, center])  # 可学习频域滤波 (B,64,H,W)
            ↓
    GatedFusion: g = σ(Conv1x1(concat[F_s, F_f]))
                 F_fused = g⊙F_s + (1-g)⊙F_f
            ↓
    IntensityHead: Conv1x1(F_fused) → Sigmoid×3
        → s_illum, s_noise, s_motion ∈ [0,1]（独立，不互斥）
```

**三源强度图在 v4 中的角色**：

| 用途 | v3 | v4 |
|------|----|----|
| IGRF 融合加权 | softmax 归一化后加权 → **梯度衰减** | **已移除** |
| IFPN 条件调制 | s_illum 传入 | 保留 |
| NDPN 条件调制 | s_noise 传入 | 保留 |
| MRPN 条件调制 | s_motion 传入 | 保留 |
| Loss 正则 | L_illum 平滑 | L_illum 边缘感知平滑 |

---

### 3.2 SACE（可变形跨帧对齐）

**功能**：基于 TFSI 输出的退化先验，进行可变形卷积跨帧对齐。

**v3.2 改进**：`_soft_median` 替代 `median(dim=1)`，使梯度沿通道维分散（而非仅1帧获得梯度），梯度改善 2.1x。

---

### 3.3 IFPN（Illumination-Filtering Pyramid Network）

**功能**：多帧光照参考估计与光照恢复。

**关键机制**：
- `IllumExtract`: Retinexformer 风格光照图提取（groups=4 分组卷积）
- 帧间相似度加权：cosine similarity + softmax → L_ref（邻帧光照参考）
- `L_ratio = clamp(L_ref/L_t, -4, 4)`（v3.2 从 [-6.9, 25.5] 裁剪到 [-4, 4]，数值稳定）
- `F_t_illum = F_t × (1 + ratio_proj(L_ratio))` → refine → f_illum_out

**v3.2 改进**：`L_ratio.clamp(-4, 4)` 防止梯度爆炸。

---

### 3.4 NDPN（Noise-Dispelling Pyramid Network）

**功能**：利用时域统计量进行噪声去除。

**输入**：F_aligned_list, μ_t_clean, σ_t, s_noise

---

### 3.5 MRPN（Motion-Recovery Pyramid Network）

**功能**：运动补偿恢复。

**输入**：feats, F_aligned_list, s_motion

---

### 3.6 IGRF v4（拼接融合重建）— 核心修改

**v3 架构（已废弃）**：
```python
# 归一化加权融合
w_base, w_illum, w_noise, w_motion = softmax([s_base, s_illum, s_noise, s_motion])
f_fused = w_base*f_base + w_illum*f_illum + w_noise*f_noise + w_motion*f_motion
# 问题：w_base≈0.44, branches≈0.12-0.32 → 梯度被乘以 <1 的权重 → 衰减
```

**v4 架构（当前）**：
```python
# 拼接融合：所有分支等权参与，梯度无衰减
f_cat = torch.cat([f_base, f_illum, f_noise, f_motion], dim=1)  # (B, 4C, H, W)
f_fused = Conv(4C→C) + GELU  # 学习最优融合权重
delta = 4×ResBlock + Conv(C→3)
res_t = clamp(image_center + delta, 0, 1)
```

**梯度改善效果（smoke test）**：

| 模块 | v3.2 梯度 | v4 梯度 | 改善倍数 |
|------|-----------|---------|---------|
| SACE | 0.0019 | 0.034 | **17.9x** |
| IFPN | 0.0017 | 0.014 | **8.4x** |
| MRPN | 0.0055 | 0.037 | **6.7x** |
| NDPN | 0.0046 | 0.018 | **3.9x** |

---

## 4. 损失函数设计

### 4.1 主重建损失（监督 IGRF 输出 res_t）

```
L_recon = L_pix + λ_freq × L_freq
    L_pix  = Charbonnier(res_t, GT)          # 空间域鲁棒重建
    L_freq = L1(|FFT(res_t)|, |FFT(GT)|)    # 频域幅度谱约束

L_ssim = 1 - SSIM(res_t, GT)                # 结构相似性
L_perc = PerceptualLoss(res_t, GT)          # VGG16 relu3_3 特征距离
L_illum = EdgeAwareSmoothness(s_illum, GT)  # 光照场边缘感知平滑
```

### 4.2 独立辅助分支损失（监督 IFPN/NDPN/MRPN）

**v3.2（共享 aux_proj）**：
```python
aux_proj = nn.Conv2d(64, 3, 1, 1)  # 195 参数，三分支共享
L_aux = mean(charbonnier(aux_proj(f_branch), GT))  # 三分支用同一投影
# 问题：梯度冲突，一组参数无法同时学三种不同功能的重建
```

**v4（独立 BranchReconHead × 3）**：
```python
# 每个分支有专属的 3 层重建网络（~3,200 参数/head）
aux_head_illum = BranchReconHead(64, 3)  # Conv(64→32)+GELU+Conv(32→32)+GELU+Conv(32→3)
aux_head_noise = BranchReconHead(64, 3)
aux_head_motion = BranchReconHead(64, 3)

L_aux = mean([
    Charbonnier(aux_head_illum(f_illum_out), GT),
    Charbonnier(aux_head_noise(f_noise_out), GT),
    Charbonnier(aux_head_motion(f_motion_out), GT),
]) / 3.0
```

### 4.3 为什么不能用更大规模的共享 aux_proj 替代多 aux_head？

| 方案 | 参数量 | 梯度冲突 | 能否学到独立功能 |
|------|--------|---------|---------------|
| 共享 1×1 conv | 195 | 严重（三方向梯度叠加） | 否 |
| 共享大 CNN（10层）| >10,000 | **依然存在**（同一参数组） | 否 |
| 独立 3×BranchReconHead | ~9,600 | **无**（参数完全分离）| 是 |

**根本原因**：这是多任务学习中的**梯度干扰问题**（Gradient Interference），不是网络容量问题。一组共享参数无论多大，都无法同时为 3 个不同功能的输入学到最优映射——梯度方向相互矛盾，导致哪个都学不好。

### 4.4 总损失公式

```
L_total = L_recon
        + λ_ssim × L_ssim          (λ_ssim=0.2)
        + λ_perc × L_perc          (λ_perc=0.2)
        + λ_illum × L_illum        (λ_illum=0.01)
        + λ_aux × L_aux            (λ_aux=0.5)
```

---

## 5. 训练配置

| 参数 | 值 |
|------|-----|
| 数据集 | SDSD indoor（70序列，2064样本） |
| 输入 | window_size=5, crop_size=256 |
| 编码器 | 4级金字塔 [32,64,96,128], fused=64 |
| Batch Size | 4 |
| 学习率 | 5e-4（CosineAnnealing + LinearWarmup 5epoch） |
| 优化器 | AdamW, weight_decay=1e-4 |
| AMP | 开启（GradScaler） |
| 梯度裁剪 | max_norm=1.0 |
| 总 Epochs | 100 |
| 验证间隔 | 每 5 epoch |
| 推理方式 | tiled_forward（tile=256, overlap=32）|

---

## 6. 参数量

| 模块 | 参数量（约） |
|------|-------------|
| PyramidEncoder | ~480K |
| TFSI | ~200K |
| SACE | ~180K |
| IFPN | ~120K |
| NDPN | ~80K |
| MRPN | ~80K |
| IGRF (含 fuse_proj) | ~130K |
| **主网络总计** | **~1.28M** |
| aux_head × 3（loss 模块）| ~9.6K |
| **含 loss 头总计** | **~1.29M** |

---

## 7. v3 → v3.2 → v4 演进历史

### v3（基线）
- 归一化加权融合，共享 aux_proj（195参数），无 SSIM loss
- 问题：分支梯度弱 13-90x，训练 23 epoch 完全停滞

### v3.2（梯度修复）
- `_soft_median` 替代 `median`（SACE 梯度 2.1x）
- `L_ratio.clamp(-4,4)`（数值稳定）
- 新增 Charbonnier + SSIM + 共享 aux_proj
- warmup scheduler + grad_clip
- **结果**：梯度改善 1.5-2.2x，但仍不足以突破架构瓶颈

### v4（架构重构）
- IGRF：移除加权融合 → concat+conv（梯度无衰减）
- Loss：共享 aux_proj → 独立 BranchReconHead × 3
- lambda_aux=0.5, lambda_perc=0.2
- **结果**：梯度改善 3.9-17.9x，pix loss 从 0.503 降至 0.482（epoch 2）

---

## 8. 文件对应关系

| 文件 | 内容 |
|------|------|
| `models/modules/igrf.py` | IGRF v4 拼接融合（核心修改） |
| `losses/losses.py` | TFSNetLoss v4 + BranchReconHead |
| `models/tfs_net.py` | 主网络，输出 dict 含 f_illum/noise/motion_out |
| `models/modules/tfsi.py` | TFSI 三源强度估计 |
| `models/modules/ifpn.py` | IFPN 光照恢复（L_ratio clamp v3.2） |
| `models/modules/sace.py` | SACE 对齐（soft_median v3.2） |
| `models/modules/ndpn.py` | NDPN 去噪 |
| `models/modules/mrpn.py` | MRPN 运动补偿 |
| `configs/sdsd_stage1.yaml` | 训练超参数配置 |
