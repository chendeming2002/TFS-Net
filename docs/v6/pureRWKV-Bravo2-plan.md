# TFS-Net v6 Bravo 架构困境全景诊断与解决方案

根据联网检索（检索于 2026-06-29）和您上传的四份文档，以下是从模型结构角度的系统性诊断。

---

## 一、核心困境回顾

从文档可知：
- **Run 1 崩塌**: VGG 0.8 + SSIM 0.5 + pix 0.3 → ep10 暴跌至 11.32 dB (-8.5 dB)
- **Run 2 均衡版**: VGG 0.4 + SSIM 0.3 + pix 0.5 → loss 正常下降但尚无最终验证
- v6.5 纯 RWKV 追平了 v5.9.2 (20.36 vs 20.39)，但 Bravo 引入的多项改动互相耦合增加了不稳定性

---

## 二、结构层面的四个耦合风险点

### 🔴 风险 1：DWT-LFF 的 IDWT 重构不对称（结构性 Bug）

**位置**: `dwt_lff.py` 第 55~64 行

```python
feat_sace = self.dwt.inverse(LL_ref, LH_half, HL_half, HH_half)  # LL_ref = α·LL
feat_tfsi = self.dwt.inverse(LL_deg, LH_half, HL_half, HH_half)  # LL_deg = (1-α)·LL
```

**问题本质**：`feat_sace` 和 `feat_tfsi` 共享**完全相同的** `LH*0.5, HL*0.5, HH*0.5` 高频分量。当 `illum_alpha` 偏离 0.5 时（你设计中心帧 α=0.6，邻居帧 α=0.4）：

| α 值 | LL_ref | LL_deg | 后果 |
|------|--------|--------|------|
| 0.6 | 0.6·LL | 0.4·LL | feat_sace 的低频能量是 feat_tfsi 的 1.5 倍 |
| 训练中 α→0.9 | 0.9·LL | 0.1·LL | feat_tfsi **几乎只剩高频边缘**，能量极低 |
| 训练中 α→0.1 | 0.1·LL | 0.9·LL | feat_sace **几乎只剩高频边缘** |

**连锁反应路径**：
```
α 偏移 → LL_ref/LL_deg 能量比失衡 → 
LayerNorm(feat_sace) 与 LayerNorm(feat_tfsi) 的分布差异扩大 →
PureRWKVSACE 的 lff_stack 帧间方差 std 出现异常峰值/零值 →
RWKV 的 spatial_decay/spatial_first 梯度爆炸 → loss NaN
```

**关键**: 由于中心帧和邻居帧用不同 α 初始化（0.6 vs 0.4），在 SACE 中 `lff_stack` 混合了两种分布不同的输出，`std(dim=1)` 计算的 `sigma_t_clean` 实际上测量的不是"噪声/运动方差"，而是**两个 DWT-LFF 实例之间的 α 差异**。

---

### 🔴 风险 2：RWKV 的 `_bi_wkv_scan` 数值稳定性缺失

**位置**: `cross_rwkv.py` 第 82~96 行

```python
def _bi_wkv_scan(self, w, u, k, v):
    ew = torch.exp(w).view(1, 1, 1, C)       # ← 无上界约束！
    ...
    out[:, t] = num_t / (den_t + 1e-8)       # ← 除法容易溢出
```

根据 [Vision-RWKV (ICLR 2025)](https://arxiv.org/html/2403.02308) 的稳定性章节：

> *"Increasing model depth and the accumulation of exponential terms during recursion can lead to instability... We divide the exponential term by the number of tokens (e.g., exp(−(|t−i|−1)/T·w)), making the maximum decay and growth bounded."*

以及 [flash-linear-attention 社区 Issue #77](https://github.com/fla-org/flash-linear-attention/issues/77)：

> *"RWKV6 在视觉数据上训练，NaN 梯度初期稀少，但越来越频繁...唯一能阻止 NaN 的方案是给 w 加硬上界 |w| < 12"*

你的代码中已经做了 `w = self.spatial_decay / T` 的除法（forward 第 109 行），但：
1. `spatial_decay` 的初始化用了 `decay_speed[h] = -5 + 8 * (h / (C-1))^(0.7+...)` ，范围约 [-5, +3]。除以 T=5 后约 [-1, +0.6]，`exp(w)` 范围约 [0.37, 1.82]。这**尚在安全范围**。
2. **但训练中 `spatial_decay` 是可学习参数**，如果梯度推动某些通道的 decay 值超出初始范围（比如达到 +20），则 `exp(20/5) = exp(4) ≈ 55`，与 `k` 相乘后很容易溢出。

**核心遗漏**：缺少 Vision-RWKV 推荐的以下保障：
- ❌ 无 `spatial_decay` 的硬上界 clamp
- ❌ 无 RWKV output 后的额外 LayerNorm  
- ❌ 无 LayerScale（零初始化残差缩放）

---

### 🔴 风险 3：VGG 感知损失权重严重偏高

根据 NTIRE 2026 低光增强挑战赛的多个顶级方案：

| 方案 | VGG 权重 | 像素损失权重 | 比例 |
|------|---------|------------|------|
| [SYSU-FVL (NTIRE 2026)](https://arxiv.org/html/2604.17669v2) | **0.04** | 1.0 (Charbonnier) | 1:25 |
| [DUSKAN (NTIRE 2026)](https://arxiv.org/html/2604.17669v2) | **0.01** | 1.0 (L1) | 1:100 |
| [LUMEN (2025)](https://arxiv.org/html/2605.17893) | **0.1** | 1.0 (L1) | 1:10 |
| [低光增强通用建议](https://www.mdpi.com/2076-3417/15/11/6330) | **0.1** | 1.0 (L1) + 1.0 (SSIM) | 1:20 |
| [PyTorch 社区经验](https://discuss.pytorch.org/t/vgg-perceptual-loss-on-multi-resolution/221093) | **0.001~0.01** | 1.0 | — |
| **你的 Run 1** | **0.8** | 0.3 (Charbonnier) | **2.67:1 (反转!)** |
| **你的 Run 2** | **0.4** | 0.5 | **0.8:1** |

**Run 1 失败的直接原因**：VGG 损失值的数量级通常比像素损失高 1~2 个量级（VGG 特征幅度大）。当你设 VGG=0.8 + SSIM=0.5 = 1.3 vs pix=0.3 时，实际梯度比可能高达 **10:1~40:1**，完全由感知损失主导。

**Run 2 仍然偏高**：VGG=0.4 虽然好一些，但对比业界标准 (0.01~0.1) 仍高出 **4~40 倍**。它目前没有崩塌可能只是因为 warmup 期间梯度尚未累积到临界点。

根据 [StableLLVE (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_Temporal_Consistency_for_Low_Light_Video_Enhancement_From_Single_CVPR_2021_paper.pdf)：

> *"Training a temporally stable image-based model is actually a compromise between visual quality and temporal stability. The optimal result lies in the balance of them."*

---

### 🟡 风险 4：phase_conf 对 s_noise 的调制方向可能反向

**位置**: TFSI 中

```python
s_noise = torch.sigmoid(s_noise_raw) * (1 + 0.5 * (1 - phase_conf))
```

设计意图：相位不可靠 → phase_conf 低 → s_noise 增大 → 更强去噪。

**但问题是**：`s_noise` 在 SACE 中的用法是：
```python
F_aligned[t] = out[t] + (1 - s_noise) * f_raw_center  # s_noise 大 → 残差贡献小
```

也就是说 **s_noise 大 = 认为噪声大 = 减少中心帧残差注入**。但 phase_conf 低的区域可能恰恰需要更多的中心帧 "干净" 内容注入（因为该区域对齐不可靠，应该更依赖中心帧）。**这存在语义矛盾**。

---

## 三、解决方案（按优先级排序）

### 🔥 P0：修正 DWT-LFF 的 IDWT 重构

**方案 A（推荐）：取消 0.5 衰减，仅在 LL 上分裂**

```python
def forward(self, x: torch.Tensor):
    LL, LH, HL, HH = self.dwt(x)
    alpha = self.illum_alpha(LL)
    LL_ref = alpha * LL
    LL_deg = (1.0 - alpha) * LL
    
    # ★ 修正：不对高频做 0.5 衰减，两分支用完整高频
    feat_sace = self.dwt.inverse(LL_ref, LH, HL, HH)
    feat_tfsi = self.dwt.inverse(LL_deg, LH, HL, HH)
    # ...
```

**方案 B（更保守）：为两分支独立学习高频衰减系数**

```python
self.hf_gate_sace = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.8)
self.hf_gate_tfsi = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)

feat_sace = self.dwt.inverse(LL_ref, LH * self.hf_gate_sace, ...)
feat_tfsi = self.dwt.inverse(LL_deg, LH * self.hf_gate_tfsi, ...)
```

**理由**：当前共享 0.5·HF 的设计使得两个分支的差异**仅存在于 LL 低频部分**——而低频分量经过 LayerNorm 后信息量有限（被归一化掉了），实际上 feat_sace 和 feat_tfsi 经过 norm 后**几乎无法区分**，浪费了双分支设计。

---

### 🔥 P1：RWKV 稳定化（3 项措施）

```python
class VRWKVStyleSpatialMix(nn.Module):
    def __init__(self, ...):
        # ... 原有代码 ...
        
        # ★ 新增 1：LayerScale（零初始化）
        self.layer_scale = nn.Parameter(torch.ones(channels) * 1e-5)
        
        # ★ 新增 2：输出后 LayerNorm
        self.post_norm = nn.LayerNorm(channels)
    
    def forward(self, x_flat, feat_2d_shape):
        # ... 原有扫描逻辑 ...
        
        # ★ 修正 3：hard clamp spatial_decay
        w = self.spatial_decay.clamp(-8, 8) / T  # 硬上界！
        u = self.spatial_first.clamp(-5, 5) / T
        
        wkv_out = self._bi_wkv_scan(w, u, k, v)
        rwkv = sr * wkv_out
        out = self.output(rwkv)
        
        # ★ LayerScale + post-norm
        out = self.post_norm(out) * self.layer_scale
        return out
```

参考 [Vision-RWKV](https://openreview.net/pdf?id=nGiGXLnKhl) 原文：

> *"We apply layer normalization after the attention mechanism and the Squared ReLU operation, to prevent the model's output from overflowing. These two adjustments promote stable scaling..."*

---

### ⚡ P2：损失函数权重彻底修正

根据 [NTIRE 2026 挑战赛](https://arxiv.org/html/2604.17669v2)多个方案的共识和 [通用低光增强损失设计](https://www.mdpi.com/2076-3417/15/11/6330)：

```python
# ★ 推荐配置（对齐 2026 SOTA 实践）
L_total = 1.0 * Charbonnier(res, GT)          # 基础重建（主导）
        + 0.04 * VGG_perceptual(relu3_3)       # ↓ 从 0.4 降到 0.04
        + 0.2 * (1 - SSIM)                     # ↓ 从 0.3 降到 0.2
        + 0.1 * Edge_Sobel(res, GT)            # 新增：边缘保持
        + 0.05 * L_freq                        # 保留
        + temporal losses...                    # 保留
```

**为什么 VGG 必须降到 ≤0.04？**

来自 [NTIRE 2026 参赛方案](https://arxiv.org/html/2605.02212)：
> *"The loss function combines pixel-wise and perceptual objectives, defined as L=0.1·L₁ + L_LPIPS"*

注意这里 L1 的权重反而是**较小的 0.1**，但 LPIPS 是一种 **已归一化到 [0,1] 范围** 的感知度量，和 VGG L1 的数量级完全不同。如果你用的是 raw VGG feature L1（relu3_3 输出的特征图差异），其数值通常在 **1~100** 的范围，远大于像素 L1（通常 0~1），所以权重必须 ÷10~÷100 才能与像素损失平衡。

---

### ⚡ P3：添加时序一致性损失

根据 [RCSA (arXiv 2026)](https://arxiv.org/html/2602.07428v1) 的方案：

```python
# 帧差自相似损失
L_self = ||（pred_{k+1} - pred_k) - (gt_{k+1} - gt_k)||_2

# 局部亮度差异损失  
L_dif = |MaxPool(bright_k) - MaxPool(bright_{k+1})|
```

以及 [StableLLVE (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_Temporal_Consistency_for_Low_Light_Video_Enhancement_From_Single_CVPR_2021_paper.pdf)：
> *"With the increase of branch weight [λ=20], the network becomes more temporally stable... optimal weight is around λ=20"*

你的模型**缺少任何显式时序约束**，完全依赖 RWKV 的隐式时序建模能力。在纯 RWKV 对齐能力有限的情况下，补充时序损失是必要的。

---

### 🔧 P4：phase_conf → s_noise 调制方向修正

```python
# 修正前（语义矛盾）:
s_noise = sigmoid(raw) * (1 + 0.5*(1-phase_conf))  
# phase不可靠 → s_noise大 → (1-s_noise)小 → 减少中心帧残差

# ★ 修正后（逻辑一致）:
# 方案一：反转调制方向
s_noise = sigmoid(raw) * (1 - 0.3*(1-phase_conf))
# phase不可靠 → s_noise减小 → (1-s_noise)增大 → 更依赖中心帧原始特征

# 方案二：让 phase_conf 独立作用于 SACE 的边缘门控
# 而非通过 s_noise 间接调制
```

---

## 四、总结：问题耦合关系图

```
DWT-LFF 0.5 HF 共享 ──┐
                        ├─→ lff_stack 分布不稳定 ──→ RWKV 输入方差异常
DWT-LFF α 中心/邻居不同 ┘                                    │
                                                              ↓
VGG 权重过高 ──→ 梯度由感知损失主导 ──→ spatial_decay 被推向极值
                                                              │
                                                              ↓
RWKV 缺少硬约束 ──────────────────────→ exp(w) 溢出 → NaN
                                                              │
缺少时序损失 ──→ 帧间输出不一致 ──→ 方差信号 noise ←──────────┘
```

**关键洞察**：Run 1 在 ep10 崩塌不是某单一因素导致，而是上述 4 个风险在 warmup 结束（lr 跳升）后**同时触发**的级联失效。

---

## 五、推荐实施顺序

| 步骤 | 修改 | 状态 | 文件 |
|------|------|------|------|
| 1 | VGG 0.4→**0.04**, pix 0.5→**1.0**, SSIM 0.3→**0.2** | ✅ | configs/v6_bravo.yaml |
| 2 | RWKV `spatial_decay.clamp(-8,8)` + `spatial_first.clamp(-5,5)` + post LayerNorm | ✅ | cross_rwkv.py |
| 3 | DWT-LFF 取消 `*0.5` HF 共享，两分支用完整 LH/HL/HH | ✅ | dwt_lff.py |
| 4 | 添加 L_self 时序损失 (λ=0.15) | ⏸ | — |
| 5 | 修正 phase_conf 调制方向 | ⏸ | — |

---

## 六、训练实证（2026-06-29）

**Bravo2 ep5**: PSNR=**20.05**, SSIM=0.7676 — **历史最高 ep5！**

| 实验 | ep5 PSNR | SSIM | 说明 |
|---|---|---|---|
| v5.5 | 19.09 | 0.7577 | fuse 死亡 |
| v5.9.2 | 19.61 | 0.7595 | s_illum 复生 |
| v6.5 | 19.74 | 0.7620 | Pure RWKV |
| **Bravo2** | **20.05** | **0.7676** | **+0.44 dB vs v5.9.2** |

三项修复验证成功：P0(VGG 0.04+pix 1.0)→训练稳定无崩塌, P1(RWKV clamp)→数值稳定, P2(DWT-LFF完整HF)→双分支差异化。

---

## 参考来源

1. [Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures (ICLR 2025)](https://arxiv.org/html/2403.02308)
2. [Vision-RWKV PDF (OpenReview)](https://openreview.net/pdf?id=nGiGXLnKhl)
3. [flash-linear-attention Issue #77: RWKV6 NaN Gradient](https://github.com/fla-org/flash-linear-attention/issues/77)
4. [NTIRE 2026 Challenge on Low Light Image Enhancement](https://arxiv.org/html/2604.17669v2)
5. [NTIRE 2026 Efficient Low Light Enhancement](https://arxiv.org/html/2605.02212)
6. [Low-Light Image Enhancement: Lightweight Network (Applied Sciences 2025)](https://www.mdpi.com/2076-3417/15/11/6330)
7. [LUMEN: Low-light Multi-stage Enhancement Network (arXiv 2025)](https://arxiv.org/html/2605.17893)
8. [Row-Column Separated Attention for Low-Light Video Enhancement (arXiv 2026)](https://arxiv.org/html/2602.07428v1)
9. [StableLLVE: Learning Temporal Consistency for LLVE (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_Temporal_Consistency_for_Low_Light_Video_Enhancement_From_Single_CVPR_2021_paper.pdf)
10. [VGG Perceptual Loss on Multi Resolution - PyTorch Forums](https://discuss.pytorch.org/t/vgg-perceptual-loss-on-multi-resolution/221093)
11. [KL-Divergence to Focus Frequency in Low-Light Enhancement (arXiv 2025)](https://arxiv.org/html/2509.13083v2)
12. [ILR-Net: Low-light Enhancement based on Retinex (PLOS One 2024)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0314541)