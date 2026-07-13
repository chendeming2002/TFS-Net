# Flight6 设计：基于 Flight5 实证数据 + Zero-DCE++ 理论的 DOF 再平衡方案

---

## 一、Flight5 数据的精确诊断

### 1.1 核心矛盾已锁定

Flight5 的四个 epoch 检查点讲述了一个完整的故事：

```
时间线叙事：
ep5:   gain=1.37 🎉 → 首次突破1.0，TCC给了gain空间
ep10:  gain≈1.05   → 开始回落，曲线DOF优势显现
ep20:  gain<1.0    → 曲线已开始排挤（但PSNR仍在涨→17.84）
ep30:  gain=0.63   → 排挤完成，PSNR回落→17.32
ep40:  gain=0.55-0.77 → 锁死在<1，PSNR停滞→17.33
```

**关键转折点在 ep10-20 之间**：曲线完成了对 gain 的排挤。此后无论训练多久，PSNR 都无法突破 17.84——因为 gain 已退化为被动缩放器（<1），而曲线独占了亮度调节但受限于其固有的非线性形状无法精确匹配 GT。

### 1.2 MCPN/NDPN 回落的因果链

```
因果链：
曲线DOF(18HW) >> gain DOF(1HW)
  → 优化器优先通过曲线降低L_pix
  → gain被边缘化（<1 = 纯衰减）
  → gain不再传递有效梯度给下游MCPN/NDPN
  → MCPN gamma: 0.017→0.011（回落）
  → NDPN gamma: 0.011→0.009（回落）
```

这解释了为什么 MCPN 在 Phase 1.5 短暂激活（ep30: 0.017）后又回落——不是 MCPN 本身有问题，而是**其上游信号源（gain 分支）被饿死了**。

### 1.3 定量：曲线还能改进多少？

Flight5 当前状态：
- 曲线输出（curve_out）≈ 0.45-0.50（TCC 正常工作）
- gain ≈ 0.63 → 最终输出 ≈ 0.45 × 0.63 ≈ **0.28**
- GT 均值 ≈ 0.45-0.50
- 差距来源：gain 在做 **不必要的衰减**

如果 gain 能稳定在 1.0-1.2：
- 最终输出 ≈ 0.45 × 1.1 = **0.50**（匹配 GT）
- 预期 PSNR 提升：消除 0.28→0.50 的系统偏差 ≈ **+2-3 dB**

---

## 二、Zero-DCE++ 的关键启示

根据联网检索（检索于 2026-07-12T15:22:07+08:00），[Zero-DCE++ (TPAMI 2022)](https://ar5iv.labs.arxiv.org/html/2103.00860) 提供了三条直接适用的实证：

### 启示 1：下采样反而提升性能

| 下采样倍率 | PSNR | 趋势 |
|:----------:|:----:|:----:|
| 原始分辨率 | 16.09 | baseline |
| 4×↓ | 16.29 | +0.20 |
| **12×↓** | **16.42** | **+0.33 🏆** |
| 50×↓ | 15.85 | -0.24 |
| 300×↓ | 14.53 | -1.56 |

论文解释：

> "pixels in a local region have the same intensity (also the same adjustment curves)"  
> ——[Zero-DCE++](https://ar5iv.labs.arxiv.org/html/2103.00860)

曲线参数本质上是**区域级属性**——全分辨率预测引入的像素级噪声反而有害。

### 启示 2：迭代间参数共享无性能损失

> "the estimated curve parameters in different iteration stages are similar... the curve parameter map can be reused"  
> ——[Zero-DCE++](https://ar5iv.labs.arxiv.org/html/2103.00860)

Zero-DCE 原始：8 iter × 3 ch = **24 参数图**
Zero-DCE++：3 参数图复用 8 次 = **3 参数图**

性能差异"negligible"。

### 启示 3：组合降维的效果

[论文 Table IV](https://ar5iv.labs.arxiv.org/html/2103.00860) 显示：DSconv + Pshared + 12×↓ 三者叠加，参数从 79K→10K，FLOPs 从 84G→0.115G，PSNR 仅损失约 0.5 dB（非监督设置下）。

**直接映射到我们的问题**：Flight5 的曲线 DOF = 6 iter × 3 ch × H × W = 18HW。如果应用 Zero-DCE++ 的两个降维：
- 迭代间共享：18HW → 3HW（6× 降）
- 4× 空间下采样：3HW → 3HW/16 ≈ 0.19HW（再 16× 降）
- 最终曲线 DOF：**0.19HW** vs gain 的 **1HW** → gain 反而有 **5× DOF 优势**

---

## 三、Flight6 完整设计

### 3.1 设计原则

基于以上分析，Flight6 的核心目标是：

```
目标：让 gain 在 DOF 层面获得优势，迫使优化器通过 gain 分支传递亮度调节信号

约束：
  1. 不破坏 TCC 的收敛性质（已验证有效）
  2. 不引入新的训练不稳定性
  3. 保持 soft_clamp + warmup + clip 的稳定性组合
  4. 渐进式改动，可回溯
```

### 3.2 架构改动清单

#### 改动 1（核心）：TCC 迭代间参数共享

```python
# ============================================
# 文件: ispn.py — TargetConvergentCurve
# ============================================

# Flight5 原始：curve_net 输出 6×3=18 通道
# self.curve_net = nn.Sequential(
#     nn.Conv2d(feat_ch, 32, 3, padding=1),
#     nn.ReLU(inplace=True),
#     nn.Conv2d(32, 18, 3, padding=1),  # 18 通道 = 6 iter × 3 ch
# )

# Flight6 修改：curve_net 输出 3 通道，复用 6 次
class TargetConvergentCurve_v2(nn.Module):
    def __init__(self, feat_ch=64, n_iter=6, ds_factor=4):
        super().__init__()
        self.n_iter = n_iter
        self.ds_factor = ds_factor
        
        # 曲线网络：只输出 3 通道的 A maps
        self.curve_net = nn.Sequential(
            nn.Conv2d(feat_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),   # ← 从 18 减到 3
        )
        
        # α_target（可学习全局参数）
        self.alpha_raw = nn.Parameter(torch.zeros(1))
        
        # 初始化（保持 Flight5 验证过的 bias=0.25）
        nn.init.zeros_(self.curve_net[-1].weight)
        nn.init.constant_(self.curve_net[-1].bias, 0.25)
    
    @property
    def alpha_target(self):
        return torch.sigmoid(self.alpha_raw)  # (0, 1)
    
    def forward(self, feat, img):
        """
        feat: (B, feat_ch, H, W) — 编码器特征
        img:  (B, 3, H, W) — S2 输出（TCC 输入）
        """
        B, C, H, W = img.shape
        
        # === 改动 2：低分辨率特征预测 ===
        # 对特征做空间下采样
        H_ds, W_ds = H // self.ds_factor, W // self.ds_factor
        feat_ds = F.adaptive_avg_pool2d(feat, (H_ds, W_ds))
        
        # 在低分辨率上预测 A maps
        A_raw_ds = self.curve_net(feat_ds)          # (B, 3, H/ds, W/ds)
        
        # 上采样回原始分辨率
        A_raw = F.interpolate(A_raw_ds, size=(H, W), 
                              mode='bilinear', align_corners=False)
        
        # A 范围映射 [-4, 4]
        A_map = 4.0 * torch.tanh(A_raw)            # (B, 3, H, W)
        
        # === TCC 迭代：同一 A_map 复用 n_iter 次 ===
        alpha = self.alpha_target
        le = img
        for _ in range(self.n_iter):
            le = le + A_map * le * (1 - le) * (alpha - le)
        
        return le
```

**DOF 计算**：
```
Flight5 曲线 DOF = 18 × H × W = 18HW
Flight6 曲线 DOF = 3 × (H/4) × (W/4) = 3HW/16 = 0.1875HW
降幅 = 96×
```

#### 改动 2（核心）：gain 分支路径对称化

```python
# ============================================
# 文件: ispn.py — GainPredictor
# ============================================

# Flight5 原始：
# gain = softplus(fc(gap(h))) + offset
# 路径: h → gap → fc(32→1) → softplus → ×curve_out
# 问题: softplus 导数在 0 附近 = 0.5（梯度衰减），且全局 GAP 失去空间信息

# Flight6 修改：gain 保持像素级预测，移除 softplus
class GainPredictor_v2(nn.Module):
    def __init__(self, feat_ch=64, gain_range=(0.5, 2.0)):
        """
        gain_range: (min, max) — 输出范围
        """
        super().__init__()
        self.g_min = gain_range[0]
        self.g_max = gain_range[1]
        
        # 像素级预测（与曲线对称的 conv 结构）
        self.gain_net = nn.Sequential(
            nn.Conv2d(feat_ch, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),   # 输出 1 通道
        )
        
        # 初始化：输出≈0 → gain≈(min+max)/2 = 1.25
        nn.init.zeros_(self.gain_net[-1].weight)
        nn.init.zeros_(self.gain_net[-1].bias)
    
    def forward(self, feat):
        """
        feat: (B, feat_ch, H, W)
        returns: gain_map (B, 1, H, W) in [g_min, g_max]
        """
        raw = self.gain_net(feat)  # (B, 1, H, W)
        # sigmoid 映射到 [g_min, g_max]
        gain = self.g_min + (self.g_max - self.g_min) * torch.sigmoid(raw)
        return gain
```

**关键改动说明**：
- 移除 `softplus`（梯度在 0 附近 = 0.5，衰减信号）
- 移除 `GAP`（全局平均池化丢失空间信息，限制 gain 为标量）
- gain 变为 **像素级** 预测（DOF = 1HW）
- 输出通过 `sigmoid` 映射到 `[0.5, 2.0]`，梯度在所有位置非零
- 初始化：raw=0 → sigmoid(0)=0.5 → gain = 0.5 + 1.5×0.5 = **1.25**（合理起点）

**梯度路径对比**：

| | Flight5 | Flight6 |
|---|---|---|
| 曲线路径 | h → conv32 → conv18 → tanh → 6×iter | h → pool → conv32 → conv3 → tanh → bilinear → 6×iter |
| gain 路径 | h → GAP → fc1 → softplus | h → conv16 → conv1 → sigmoid |
| 曲线路径深度 | 2 conv + 6 iter | 2 conv + upsample + 6 iter |
| gain 路径深度 | 1 fc + 1 activation | 2 conv + 1 activation |
| **梯度衰减** | 曲线≈0.8, gain≈0.5(softplus) | **曲线≈0.6(远+pool), gain≈0.8(近+直接)** |

Flight6 中 **gain 的梯度路径更短、更直接**，优化器会自然优先利用 gain。

#### 改动 3：gain 监督适配

```python
# ============================================
# 文件: losses.py — gain_supervision_loss
# ============================================

def compute_gain_supervision(gain_map, gt, curve_out, phase, epoch):
    """
    Flight6 版本：
    - gain_map 现在是 (B, 1, H, W) 而非标量
    - target_gain 也是像素级
    """
    # 计算像素级 target
    eps = 1e-3
    target_gain = gt / (curve_out.detach() + eps)    # (B, 3, H, W)
    target_gain = target_gain.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # clamp 到合理范围
    target_gain = target_gain.clamp(0.5, 2.0)
    
    # L1 监督
    loss = F.l1_loss(gain_map, target_gain)
    
    return loss
```

#### 改动 4：最终数据流整合

```python
# ============================================
# 文件: ispn.py — ISPN.forward (Flight6)
# ============================================

def forward(self, img_s2, feat, gt=None, phase='P1', epoch=0):
    """
    img_s2: S2 输出 (B, 3, H, W)
    feat:   编码器特征 (B, 64, H, W)
    """
    # 1. TCC 曲线（低分辨率参数，高分辨率执行）
    curve_out = self.tcc(feat, img_s2)           # (B, 3, H, W)
    
    # 2. Gain（像素级预测）
    gain_map = self.gain_predictor(feat)          # (B, 1, H, W)
    
    # 3. 合成
    enhanced = curve_out * gain_map               # (B, 3, H, W)
    
    # 4. Soft clamp（保持 Flight5 验证过的版本）
    output = soft_clamp(enhanced)                 # (B, 3, H, W)
    
    # 5. 损失计算
    if gt is not None:
        losses = {}
        losses['L_pix'] = F.l1_loss(output, gt)
        
        if phase in ['P1', 'P1.5']:
            losses['L_gain'] = 0.5 * compute_gain_supervision(
                gain_map, gt, curve_out, phase, epoch)
        
        return output, losses
    
    return output
```

### 3.3 DOF 最终对比

| 组件 | Flight5 | Flight6 | 变化 |
|------|:-------:|:-------:|:----:|
| 曲线 | 18HW | **0.1875HW** | ↓ 96× |
| gain | ~1（标量） | **1HW** | ↑ HW× |
| **比率** | **18:1 曲线碾压** | **1:5.3 gain 优势** | **反转** |

---

## 四、为什么选择 4× 而不是 12×？

Zero-DCE++ 使用 12× 是在**输入图像尺寸 1200×900** 的条件下。我们的训练图像尺寸可能是 256×256 或 512×512：

| 图像尺寸 | 12×↓ 后 | 特征图尺寸 | 是否足够 |
|:--------:|:-------:|:---------:|:--------:|
| 1200×900 | 100×75 | 充足 | ✅ |
| 512×512 | 43×43 | 勉强 | 🟡 |
| 256×256 | 21×21 | 边缘 | ⚠️ |
| **256×256 + 4×↓** | **64×64** | **充足** | ✅ |

对于 256×256 输入，**4× 下采样得到 64×64 特征图**——足够保留区域级亮度分布信息，同时有效降低 DOF。

如果训练图像是 512×512，可以提升到 8×↓（得到 64×64），效果可能更优。

---

## 五、训练策略

### 5.1 Phase 设计

| Phase | Epochs | 解锁 | 损失 | 目的 |
|:-----:|:------:|:----:|:----:|:-----|
| **P1** | 0-15 | TCC + gain（NDPN/MCPN 冻结） | L_pix + L_gain_sup(×0.5) | 建立 curve-gain 分工 |
| **P1.5** | 15-30 | + NDPN/MCPN 渐进解锁 | + L_ssim(×0.1) | 精细化 + 辅助分支激活 |
| **P2** | 30-50 | 全部 | 全损失 | 最终收敛 |

**关键区别 vs Flight5**：
- 不再需要 P0 阶段冻结曲线——因为 DOF 已经重新平衡，不存在"曲线排挤 gain"的结构性动力
- gain_sup 权重从 Flight5 的 1.0 降到 0.5——因为 gain 现在有 DOF 优势，不需要那么强的外部推力

### 5.2 学习率与稳定性

```python
# 保持 Flight5 验证过的稳定性组合
optimizer = Adam(params, lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
warmup_steps = 100
grad_clip_norm = 1.0

# 曲线和 gain 使用相同学习率（DOF 已平衡，不需要差异化 LR）
```

### 5.3 初始化验证

Flight6 初始状态（epoch 0, 未训练）：

```
A_map: bias=0.25 → tanh(0.25)=0.245 → A=4×0.245=0.98
curve_out: img=0.10, A=0.98, α=0.5, 6 iter → ≈0.36

gain: raw=0 → sigmoid(0)=0.5 → gain=0.5+1.5×0.5=1.25

output: 0.36 × 1.25 = 0.45
soft_clamp(0.45) ≈ 0.44

预期 epoch 0 PSNR: ~13-14（合理起点，不暗不亮）
```

---

## 六、预期效果与监控指标

### 6.1 预期轨迹

| Epoch | PSNR（预期） | gain（预期） | 对比 Flight5 |
|:-----:|:-----------:|:-----------:|:------------:|
| 5 | 15-16 | 1.1-1.3 | F5: 16.79 @ ep10 |
| 10 | 17-18 | 1.0-1.3 | **匹配 F5 峰值** |
| 20 | **18.5-19.5** | **0.9-1.3** | **超越 F5 峰值 1-2 dB** |
| 30 | 19-20 | 0.9-1.2 | 持续上升（无排挤） |
| 40 | 19.5-20.5 | 0.9-1.2 | **接近 Mod6 虚假值** |

**核心预期**：gain 在整个训练过程中**保持 >0.9**，不再回落到 <0.7。

### 6.2 Epoch 10 检查清单

```
✅ 方案验证成功:
  - PSNR > 17.0
  - gain_mean: 0.9 - 1.4 (保持活跃)
  - curve_out_mean: 0.35 - 0.55
  - alpha_target: 0.40 - 0.65

🟡 部分成功:
  - PSNR 15.5-17.0 (低于 Flight5 ep10)
  - gain_mean: 0.7-0.9 (偏低但非锁死)
  - → 可能 4×↓ 过于激进，考虑 2×↓

🔴 需要干预:
  - PSNR < 15.0
  - gain_mean = 1.25 不变（gain 没学到东西）
  - curve_out < 0.15（TCC 回归 identity）
  - → 检查 feat_ds 是否有有效信号传到 curve_net
```

### 6.3 关键对比指标

**Flight6 成功的定义**：
1. ep30 PSNR > 18.5（超越 Flight5 峰值 17.84）
2. ep30 gain_mean > 0.85（不被排挤）
3. MCPN gamma 持续 > 0.012（不回落）
4. 训练-推理 PSNR 差 < 1.0 dB（一致性）

---

## 七、备选方案（如果 4×↓ 过于激进）

如果 Flight6-A（4×↓）在 ep10 时 PSNR < Flight5 的 16.79：

| 备选 | 改动 | 曲线DOF | 比率 |
|------|------|:-------:|:----:|
| Flight6-B | 2×↓ + 参数共享 | 0.75HW | 1:1.3 |
| Flight6-C | 仅参数共享（不下采样） | 3HW | 3:1 |

优先级：A > B > C。理由是 Zero-DCE++ 的 [Table III](https://ar5iv.labs.arxiv.org/html/2103.00860) 明确显示 4-12× 范围内性能不降反升。

---

## 八、Flight6 改动总览（实施检查表）

| # | 文件 | 改动 | 行数 |
|:-:|:----:|:----:|:----:|
| 1 | ispn.py | TCC curve_net 输出 18ch → 3ch | ~5 行 |
| 2 | ispn.py | TCC forward: 特征下采样 + A_map 上采样 + 单 A 复用 6 iter | ~15 行 |
| 3 | ispn.py | GainPredictor: GAP+fc+softplus → conv+sigmoid，像素级输出 | ~20 行 |
| 4 | ispn.py | ISPN.forward: gain_map shape 适配（标量→HW） | ~3 行 |
| 5 | losses.py | gain_supervision: 适配像素级 gain + target | ~5 行 |
| 6 | train.py | Phase 配置更新 | ~5 行 |
| 7 | — | 超参：ds_factor=4, gain_range=[0.5, 2.0] | 配置 |

**总改动量**：~55 行代码，核心是 #1-3 的架构重组。

**保持不变**：
- soft_clamp ✓
- warmup 100 步 ✓
- grad_clip 1.0 ✓
- bias=0.25 初始化 ✓
- α_target 可学习 ✓
- Phase 渐进解锁逻辑 ✓

---

## 参考来源

1. [Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation (Zero-DCE++, TPAMI 2022)](https://ar5iv.labs.arxiv.org/html/2103.00860)
2. [Zero-DCE++ 论文 DOI](https://doi.org/10.48550/arxiv.2103.00860)
3. [IEEE Computer Society 发表版](https://www.computer.org/csdl/journal/tp/2022/08/09369102/1rFvZd8MAKI)
4. [Zero-DCE 项目主页](https://li-chongyi.github.io/Proj_Zero-DCE.html)
5. [Zero-Reference Deep Curve Estimation (CVPR 2020 原始论文)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)