# Flight 10 完整修改方案 — 基于推理测试结果的彻底重构

---

## 一、核心诊断：S3 提亮为什么彻底失效

推理数据揭示的不是"提亮不够"——而是**提亮管线从根本上不工作**：

```
Input:    mean=0.079, max=0.478   ← 极暗
S1 去噪:  mean=0.067, max=1.000   ← 变更暗，且部分像素爆到1.0
S2 去模糊: mean=0.033, max=0.525   ← 再次减半！
S3 提亮:  mean=0.073, max=0.663   ← gain×0.033 ≈ 0.073，增益仅×2.2
res_t:    mean=0.073, max=0.663   ← 残差≈0，Stage B 无贡献
```

这里有**三个独立的致命故障**，必须全部解决：

### 故障 1：S1→S2 亮度折半（0.067→0.033）

S2 = soft_clamp(S1 + δ₂(f_motion))。当 δ₂ 为负值时，S2 < S1。这说明 **MCPN 的运动补偿残差在推理时产生了大幅负值**，相当于 S2 在"去亮"而非"去模糊"。

根因：MCPN 在训练时通过 `L_mcpn_aux = L1(img_s2, GT)` 学习。但 GT 是正常曝光（mean≈0.45），img_s1 是极暗的（mean≈0.067）。MCPN 没有能力在一步 δ 中弥合这个巨大的亮度差距——它学到的 δ₂ 是一个**混乱的信号**，有些像素加、有些减，整体效果是进一步压低亮度。

### 故障 2：gain_map 增益不足（mean=1.19）

gain_map = GT / img_s2 的监督目标要求增益约为 0.45/0.033 ≈ **13.6×**。但实际 gain_map mean=1.19——仅为目标值的 **8.7%**。ISPN 的 gain_head 几乎没有学到有效的增益估计。

根因组合：
- `L_gain_sup` 权重 0.5 看似足够，但 gain 的目标值跨度极大（暗区需要 20×，亮区需要 2×），L1 损失在高增益区的绝对误差主导了优化，导致 gain_head 学到了"保守中间值"
- TCC 曲线在训练的 256×256 分辨率上可能部分有效，但推理的 1080×1920 分辨率上 s_illum 的空间分布与训练时完全不同

### 故障 3：Stage B 残差≈0（res_t = img_lit）

res_t mean=0.073 = img_lit mean=0.073。Stage B 的残差路径 `δ = h_θ(f_noise + f_motion) · tanh(β)` 输出了近零值。

根因：tanh(β) 的 β 在零初始化后没有有效学习——当 img_lit 本身就极暗时（mean=0.073），L_pix = Charbonnier(res_t, GT) 的梯度信号被 tanh(β) 的近零放大因子阻断，形成了**死区**。

---

## 二、问题因果链与修复优先级

```
根因 A: SGRF Stage A 的 S1→S2 序列中，MCPN δ₂ 无约束地压低亮度
  └→ img_s2 过暗 (mean=0.033)
     └→ gain 目标值被推到极端 (13.6×)
        └→ ISPN gain_head 学不到正确增益 (只输出 1.19×)
           └→ img_lit ≈ 0.033 × 1.19 ≈ 0.039 (经 TCC 推到 0.073)
              └→ 提亮近乎失效

根因 B: Stage B 的 tanh(β) 零初始化 → 残差路径死区
  └→ res_t = img_lit + 0 ≈ img_lit
     └→ 最终输出无法通过残差补偿提亮不足

根因 C: 训练/推理分辨率不匹配 (256² vs 1080×1920)
  └→ s_illum 空间分布失真
     └→ gain_map 在全分辨率推理时进一步退化
```

---

## 三、Flight 10 具体修改项

### 3.1 ❗ 最高优先级：约束 S1/S2 亮度不得低于输入

**问题**：S1 和 S2 都比输入更暗——这在物理上不合理。去噪和去模糊不应该降低整体亮度。

**方案**：为 NDPN 和 MCPN 的 δ 输出添加**下界约束**：

```python
# SGRF Stage A 修改
# S1: 去噪不应让图像变暗
delta_1 = self.stage1_conv(f_noise_out)
delta_1_clamped = torch.where(
    img_input + delta_1 < img_input * 0.8,  # 如果 S1 比输入暗超过 20%
    -img_input * 0.2,  # 最多只允许暗 20%
    delta_1
)
img_s1 = soft_clamp(img_input + delta_1_clamped)

# S2: 去模糊同样不应大幅降低亮度
delta_2 = self.stage2_conv(f_motion_out)
delta_2_clamped = torch.where(
    img_s1 + delta_2 < img_s1 * 0.7,  # S2 比 S1 暗超过 30%
    -img_s1 * 0.3,
    delta_2
)
img_s2 = soft_clamp(img_s1 + delta_2_clamped)
```

**文献依据**：[VSRELL (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) 的 INCO 模块将光照增强和噪声抑制作为**耦合问题联合建模**——其去噪分支显式使用 clamp 操作：

> $I_{denoise} = \text{Clamp}(I_{bright} - O_{denoise} \cdot M_{noise},\ 0,\ 1)$

去噪残差被约束在 `[0, 1]` 范围内，防止产生不合理的负值。

**更优方案（推荐）**：改为直接约束 δ 的范围，而非条件裁剪：

```python
# 方案 B：软约束 —— 使用 tanh 限幅 δ 的幅度
delta_1 = self.stage1_conv(f_noise_out)
delta_1 = delta_1 * torch.sigmoid(self.delta1_scale)  # 可学习缩放，初始≈0.5
# delta_1 的幅度被 sigmoid 限制在 [0,1) × delta_range
img_s1 = soft_clamp(img_input + delta_1)
```

### 3.2 ❗ 最高优先级：Stage B 残差的 Delta Scaling

**问题**：`tanh(β)` 零初始化导致残差路径死区——梯度信号被 `tanh(0)=0` 乘以后消失。

**方案**：参考 [NTIRE 2026 冠军方案 DUSKAN](https://arxiv.org/html/2604.17669v2) 的关键设计：

> $\mathbf{y} = \mathbf{x}_m \odot \sigma(\text{softplus}(\mathbf{W}_g \mathbf{x}_m)) + \mathbf{x}_m \odot \mathbf{d}$
>
> where **$\mathbf{d}$ is a learnable skip scale** ensuring a stable residual gradient path.
>
> Both paths follow a **pre-norm residual pattern with near-zero initialization**.

DUSKAN 的核心洞察是：残差路径需要一个**独立的可学习 skip scale**，与主路径的门控分离。它不是 `tanh(0)=0` 的"零初始化后等待学习"——而是 `d` 初始化为小正值（如 0.1），**保证从训练第一步就有非零梯度流**。

```python
# 替换当前的 tanh(β) 机制
class StageB(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.residual_net = nn.Sequential(
            nn.Conv2d(C*2, C, 3, 1, 1), nn.GELU(),
            nn.Conv2d(C, C, 3, 1, 1), nn.GELU(),
            nn.Conv2d(C, 3, 3, 1, 1),  # 输出 RGB 残差
        )
        # 关键：learnable delta scale，初始化为 0.1
        self.delta_scale = nn.Parameter(torch.tensor(0.1))
        # 零初始化最后一层 conv 的权重 → 残差初始为零
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)
    
    def forward(self, f_noise_out, f_motion_out, img_lit):
        delta = self.residual_net(torch.cat([f_noise_out, f_motion_out], 1))
        # delta_scale=0.1 × delta≈0 → 初始残差≈0 但梯度 ≠ 0
        res_t = img_lit + self.delta_scale * delta
        return res_t
```

**与旧设计的关键区别**：
- `tanh(β)` 零初始化：β=0 → tanh(0)=0 → **梯度被完全乘以零**
- `delta_scale=0.1` + 零初始化 conv：delta≈0 → 输出≈img_lit → **但梯度 = 0.1 × ∂L/∂delta ≠ 0**

### 3.3 ❗ 高优先级：ISPN gain_head 增益范围扩展

**问题**：gain_map mean=1.19 但目标应该是 ~13.6×。

**根因**：当前 gain_head 可能使用了 sigmoid 或 softplus 后加 1 的形式，输出范围天然受限。

**方案**：

```python
# 当前可能的实现：
gain_map = 1.0 + F.softplus(self.gain_head(features))  # 输出 > 1.0

# 修改为：允许更大增益
gain_map = 1.0 + F.softplus(self.gain_head(features)) * self.gain_scale
# gain_scale 初始化为 5.0，允许增益最高到 ~20×
self.gain_scale = nn.Parameter(torch.tensor(5.0))
```

**文献依据**：[VSRELL 的 INCO 模块](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf)使用动态增益：

> $I_{bright} = I_{curr} \cdot \text{Clamp}(g \cdot (1.5 - \alpha),\ 1,\ g_{max})$

其中 $g_{max}$ 是显式的增益上界参数。VSRELL 的增益不是无界的——而是有一个**可控的上限**，防止过曝的同时允许足够的提亮。

建议 Flight 10 的增益范围：**[1.0, 20.0]**，通过 clamp 实现：

```python
gain_map = torch.clamp(
    1.0 + F.softplus(self.gain_head(features)) * self.gain_scale,
    min=1.0, max=20.0
)
```

### 3.4 ❗ 高优先级：S1 去噪质量修复（max=1.0 饱和像素）

S1 max=1.000 说明去噪过程中某些像素被推到了最大值——这是 NDPN δ₁ 在某些区域产生了过大的正值。

**方案**：将 soft_clamp 的上界从 1.0 收紧到输入 max 的 1.5 倍：

```python
# 自适应上界
upper_bound = torch.clamp(img_input.max() * 1.5, max=1.0)
img_s1 = soft_clamp(img_input + delta_1, min_val=0.0, max_val=upper_bound)
```

这防止了去噪步骤意外引入"亮斑"——在低光场景中，去噪后的像素不应该比输入的最亮像素亮太多。

### 3.5 损失函数修改

#### 确认删除 3 项死代码 ✅

| 删除项 | 原因 |
|:------:|:-----|
| `L_wfr_reg` | WFR 已移除，恒为 0 |
| `L_gamma_reg` | relu(0.005-\|γ\|) 恒为 0，γ 远超阈值 |
| `L_dpe_prior` | 与 L_illum_spatial 直接冲突，推动 s_illum 退化 |

#### 新增/修改损失项

**A. L_brightness_preserve（新增，防止 S1/S2 降亮）**：

```python
# S1 不应比输入暗太多
L_bright_s1 = F.relu(img_input.mean() * 0.8 - img_s1.mean())
# S2 不应比 S1 暗太多
L_bright_s2 = F.relu(img_s1.mean() * 0.7 - img_s2.mean())
L_brightness_preserve = L_bright_s1 + L_bright_s2
# 权重: 0.5（固定，Phase 1+1.5+2 全程）
```

**设计理由**：这是一个**单边约束**——只在亮度下降超过阈值时产生惩罚，不阻止合理的微小亮度变化。[Zero-TIG (arXiv 2503.11175)](https://arxiv.org/html/2503.11175v2) 对首帧使用零向量初始化残差，其设计哲学是"去噪/去模糊操作应该是保守的"：

> *"For the first frame in a sequence, $R^{(t-1)\rightarrow t}_{RD}$ and $S^{(t-1)\rightarrow t}_{RD}$ are initialised as zero vectors."*

**B. L_gain_range（新增，鼓励增益覆盖目标范围）**：

```python
# 目标增益：GT.mean() / max(img_s2.mean(), 0.01)
target_gain_mean = GT.mean() / torch.clamp(img_s2.mean(), min=0.01)
# 鼓励 gain_map 的均值接近目标增益
L_gain_range = F.l1_loss(gain_map.mean(), target_gain_mean.detach())
# 权重: 0.3（固定，Phase 2 启用）
```

**C. 启用 Perceptual Decoupling**：

| 输出 | 损失类型 | 理由 |
|:----:|:--------:|:-----|
| img_s1 (去噪) | **SSIM_cs** (无亮度分量) | S1 暗但应结构清晰 |
| img_s2 (去模糊) | **SSIM_cs** (无亮度分量) | S2 更暗但应无运动模糊 |
| img_lit (提亮) | **L1** (已有 L_lit=0.5) | 亮度匹配 |
| res_t (最终) | **VGG + SSIM + L1** | 全套监督 |

**文献验证**：[ISALux (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/papers/Balmez_ISALux_Illumination_and_Semantics-Aware_Transformer_Employing_Mixture_of_Experts_for_WACV_2026_paper.pdf) 的消融实验确认 L2+VGG+MS-SSIM 三合一提供 **+0.68 dB** 增益——但**前提是各损失作用于正确的目标**。

**D. L_illum_spatial 改为单边**：

```python
# 当前：-log(std + eps)，高 std 时仍有微弱梯度推动增长
# 修改：单边惩罚，std > 1.0 时不再推动
L_illum_spatial = F.relu(1.0 - s_illum.std(dim=[-2,-1])).mean()
# 权重：0.03（从 0.1 降低）
```

std=19.6 远超合理范围。单边设计在 std>1.0 后**完全放手**，由 L_illum_tv 自然约束平滑度。

**E. L_align_warp 处置**：

ifpn_sup=5.35 持续 20+ epoch 不下降——从 UW 移出，改为固定权重 0.005。Phase 2 后期（ep60+）完全移除。

### 3.6 推理管线修复：Tiled Forward

```python
def tiled_forward(self, frames, tile_size=256, overlap=32):
    """分块推理，解决训练/推理分辨率不匹配"""
    B, T, C, H, W = frames.shape
    stride = tile_size - overlap
    output = torch.zeros(B, C, H, W, device=frames.device)
    weight = torch.zeros(B, 1, H, W, device=frames.device)
    
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = y_end - tile_size  # 保证 tile 大小一致
            x_start = x_end - tile_size
            
            tile = frames[:, :, :, y_start:y_end, x_start:x_end]
            tile_out = self.forward(tile)  # 标准 256×256 推理
            
            # 余弦权重窗口（减少边界伪影）
            w = self._cosine_window(tile_size, overlap).to(frames.device)
            output[:, :, y_start:y_end, x_start:x_end] += tile_out * w
            weight[:, :, y_start:y_end, x_start:x_end] += w
    
    return output / (weight + 1e-8)
```

**这直接解决了推理时 s_illum (H/4=270×480) 与训练时 (H/4=64×64) 空间分布不匹配的问题**。每个 tile 都在训练的 256×256 分辨率上运行，DPE 和 ISPN 的行为与训练一致。

---

## 四、完整 Flight 10 修改清单与优先级

| 优先级 | 修改项 | 解决什么故障 | 风险 | 预期收益 |
|:------:|:-------|:------------|:----:|:--------:|
| **#1** | **S1/S2 亮度下界约束 + L_brightness_preserve** | S1→S2 亮度折半 | 低 | img_s2 从 0.033→>0.06 |
| **#2** | **Stage B delta_scale=0.1 替换 tanh(β)** | 残差路径死区 | 低 | res_t ≠ img_lit |
| **#3** | **ISPN gain_scale=5.0 + clamp [1,20]** | gain_map=1.19 不足 | 中 | gain→5-15× |
| **#4** | **删除 3 项死代码 + L_illum_spatial 单边化** | 冲突梯度 | 极低 | 训练稳定性 |
| **#5** | **启用 perceptual_decoupling** | SSIM 结构差距 | 低 | SSIM +0.03-0.06 |
| **#6** | **tiled_forward 推理** | 分辨率不匹配 | 低 | 推理质量对齐训练 |
| **#7** | **L_gain_range 新增** | 增益学习不足 | 中 | gain 覆盖目标范围 |
| **#8** | **L_align_warp 降权→固定 0.005** | ifpn 停滞 | 低 | 减少无效梯度 |

---

## 五、完整损失表 (Flight 10)

### Phase 1 (ep 0-10)

| # | 损失项 | 权重类型 | 值 |
|---|--------|:--------:|:--:|
| 1 | L_pix (Charbonnier, res_t) | UW | — |
| 2 | L_ssim (res_t) | UW | — |
| 3 | L_illum_smooth (边缘感知 TV) | UW | — |
| 4 | L_gain_sup (gain vs GT/img_s2) | fixed | 0.5 |
| 5 | L_illum_spatial (单边, std>1.0 不罚) | fixed | **0.03** |
| 6 | L_illum_tv (边缘感知 TV) | fixed | 0.05 |
| 7 | L_lit (img_lit vs GT) | fixed | 0.5 |
| 8 | **L_brightness_preserve** 🆕 | fixed | **0.5** |

### Phase 1.5 (ep 11-35)

| # | 损失项 | 权重类型 | 值 |
|---|--------|:--------:|:--:|
| 9 | L_ndpn_aux (SSIM_cs, img_s1 vs GT) | fixed | 0.2 |
| 10 | L_mcpn_aux (L1, img_s2 vs GT) | fixed | 0.1 |
| 11 | L_residual_reg | fixed | **0.05** ↓ |

### Phase 2 (ep 36-100)

| # | 损失项 | 权重类型 | 值 |
|---|--------|:--------:|:--:|
| 12 | **L_ssim_s1** (SSIM_cs, img_s1) 🆕 | UW | — |
| 13 | **L_ssim_s2** (SSIM_cs, img_s2) 🆕 | UW | — |
| 14 | L_perc (VGG, res_t) | UW | — |
| 15 | L_freq (FFT L1, res_t) | UW | — |
| 16 | L_inter (S2×gain vs GT) | UW | — |
| 17 | **L_gain_range** 🆕 | fixed | **0.3** |
| 18 | L_align_warp | fixed | **0.005** ↓↓ |

**已删除**：L_wfr_reg ❌, L_gamma_reg ❌, L_dpe_prior ❌
**总计**：15-18 项（从 17→18 但无死代码，每项都有活跃梯度）

---

## 六、预期指标

| 指标 | Flight 9 (ep70) | Flight 10 目标 | 改进来源 |
|:----:|:---:|:---:|:---|
| **S1 mean** | 0.067 | **>0.065** | 亮度下界约束 |
| **S2 mean** | 0.033 🔴 | **>0.050** | 亮度下界约束 + L_brightness_preserve |
| **gain_map mean** | 1.19 🔴 | **5-15×** | gain_scale=5.0 + L_gain_range |
| **img_lit mean** | 0.073 🔴 | **>0.30** | gain 修复 + TCC |
| **res_t mean** | 0.073 🔴 | **>0.35** | delta_scale 打通残差路径 |
| **PSNR** | 16.33 | **>17.0** | 提亮管线恢复 + 感知解耦 |
| **SSIM** | 0.616 | **>0.66** | SSIM_cs 直接监督 S1/S2 |
| **LPIPS** | 0.364 | **<0.36** | VGG 仅作用于亮度匹配输出 |
| **dpe_si std** | 19.6 | **1.0-5.0** | 单边 L_spatial 停止过度推高 |

---

## 七、实施顺序

```
Step 1: 删除死代码 (L_wfr_reg, L_gamma_reg, L_dpe_prior) + 清理 UW log_var
        → 无风险，立即执行

Step 2: Stage B delta_scale 替换 tanh(β)
        → 简单改动，立即验证残差路径是否有非零输出

Step 3: S1/S2 亮度下界约束 + L_brightness_preserve
        → 架构改动，需要验证不影响训练稳定性

Step 4: ISPN gain_scale + gain clamp [1,20] + L_gain_range
        → 关键改动，直接影响提亮效果

Step 5: 启用 perceptual_decoupling (SSIM_cs→S1/S2, VGG→res_t)
        → 损失配置变更

Step 6: tiled_forward 推理
        → 仅影响推理，不影响训练

Step 7: 训练 10 epoch → 256×256 验证
        → 如果 S3 img_lit mean > 0.25 → 继续训练
        → 如果仍 < 0.10 → 执行备用方案
```

## 八、备用方案（如果 Flight 10 的提亮管线仍然失效）

如果在 Flight 10 的修改后，img_lit mean 仍然 < 0.15：

**彻底简化 Stage A**：取消 S1→S2→TCC→S3 的四步级联，改为**单步提亮**：

```python
# 直接从输入到提亮，跳过中间暗态
img_lit = img_input * gain_map  # 一步增益
res_t = img_lit + delta_scale * residual_net(f_noise, f_motion)
```

这是 [VSRELL](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf) 的做法——**先增益提亮，再去噪去模糊**，而非当前的"先去噪去模糊（在暗态），再提亮"。当前管线的根本问题可能就是**在极暗图像上做去噪/去模糊，然后期望后续的增益来补偿**——但增益学不到足够大的值。

如果反转顺序为"先提亮再去噪"，NDPN 和 MCPN 在亮态上工作，其 δ 残差的尺度与 GT 匹配，学习难度大幅降低。

---

## 参考来源

1. [VSRELL — Video Super-Resolution and Enhancement in Low-Light (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.pdf)
2. [NTIRE 2026 Challenge on Efficient Low Light Image Enhancement — DUSKAN 方案](https://arxiv.org/html/2604.17669v2)
3. [Zero-TIG — Temporal Consistency-Aware Zero-Shot LLVE (arXiv 2503.11175)](https://arxiv.org/html/2503.11175v2)
4. [VLLVE++ — Spatial-Temporal Decomposition (arXiv 2602.08699)](https://arxiv.org/html/2602.08699)
5. [Dynamic Nonlinear Networks for Low-Light Enhancement (The Visual Computer, 2026)](https://link.springer.com/article/10.1007/s00371-026-04410-4)
6. [NTIRE 2026 Efficient LLIE Methods and Results](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Yan_NTIRE_2026_Challenge_on_Efficient_Low_Light_Image_Enhancement_Methods_CVPRW_2026_paper.pdf)
7. [Low-Light Video Enhancement — Fast-Slow Dual Branches (Preprints, 2026)](https://doi.org/10.20944/preprints202604.1276.v1)
8. [Content-Illumination Coupling Guided Enhancement (Scientific Reports, 2024)](https://doi.org/10.1038/s41598-024-58965-0)
9. [ISALux — Illumination and Semantics-Aware Transformer (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/papers/Balmez_ISALux_Illumination_and_Semantics-Aware_Transformer_Employing_Mixture_of_Experts_for_WACV_2026_paper.pdf)