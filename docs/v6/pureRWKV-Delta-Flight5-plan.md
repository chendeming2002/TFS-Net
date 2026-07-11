

# Flight5 终极方案：基于 Flight4 Ep20 数据 + 文献证据的精确设计

---

## 一、诊断共识：三个 AI 分析的交叉验证

在设计 Flight5 之前，先把三轮分析中**已达成共识的结论**锁定：

| 判断 | 我（前两轮） | DeepSeek V4 | 本轮联网证据 | 共识 |
|------|:-:|:-:|:-:|:-:|
| gain=0.50 是结构性问题，非训练不足 | ✅ | ✅ | — | **确认** |
| curve 8次迭代导致输出饱和 | ✅ | ✅ | [ResNetDCE (2026)](https://doi.org/10.1109/iccraids67816.2026.11519577): 12iter PSNR下降 | **确认** |
| LE-curve 无收敛目标（不动点=1.0） | ✅ | — | [Zero-DiDCE (2024)](https://link.springer.com/article/10.1007/s11063-024-11565-5): 20iter=15.56 vs 7iter=17.15 | **确认** |
| gain 监督目标与数据流不一致 | ✅ | ✅（明确指出） | — | **确认** |
| Mod6 的 19.77 dB 含 clamp 伪收益 | ✅ | ✅ | — | **确认** |
| NDPN 健康、MCPN 受限于数据集 | ✅ | ✅ | — | **确认** |
| Phase 2 无法补救结构性分工错误 | ✅ | ✅ | — | **确认** |

**一句话总结**：Flight4 的 14.02 dB 不是"学得慢"，是**两个结构性缺陷叠加**——曲线公式发散 + gain 监督错位。这两个问题不解决，训练 1000 epoch 也没用。

---

## 二、Flight4 病灶的精确量化

### 2.1 LE-curve 饱和验证

取 Flight4 ep20 实际参数：α mean ≈ 0.76, input mean ≈ 0.10（S2 output）

| 迭代 | LE 值 | 增量 | 状态 |
|:----:|:-----:|:----:|:----:|
| 0 | 0.100 | — | 暗 |
| 1 | 0.168 | +0.068 | |
| 2 | 0.275 | +0.106 | |
| 3 | 0.426 | +0.152 | |
| 4 | 0.612 | +0.186 | ← GT 范围 |
| 5 | 0.793 | +0.181 | ⚠️ 过亮 |
| 6 | 0.917 | +0.124 | 🔴 过曝 |
| 7 | 0.975 | +0.058 | 🔴 |
| 8 | **0.994** | +0.019 | 🔴 **全白** |

**curve 输出 0.994，gain 必须压到 0.50 才能得到 0.994 × 0.5 ≈ 0.50**——这就是 gain=0.50 的物理原因。

### 2.2 gain 监督错位

当前代码的逻辑：

```
数据流:  img_s2 → curve() → img_curved → ×gain → output
监督:    target_gain = GT / img_s2     ← 监督点在curve之前！
```

这意味着：
- target_gain ≈ GT/img_s2 ≈ 0.5/0.1 = **5.0**
- 但 gain 的实际作用点在 curve 之后，img_curved ≈ 0.99
- 实际需要的 gain ≈ GT/img_curved ≈ 0.5/0.99 = **0.505**

**监督告诉 gain "你应该是 5.0"，但数据流让 gain "你只能当 0.5"——这两个信号互相矛盾**，训练自然无法收敛。

---

## 三、Flight5 方案：三项精确修改

> **设计原则：只改有确定证据支撑的部分，每项修改对应一个已确认的病灶。**

### 修改 1：Gain 监督重定向（修复监督-数据流错位）

**病灶**：target_gain 计算在 curve 之前
**修复**：target_gain 计算在 curve 之后，Phase 1 用 detach 隔离

```python
# ============================================
# 文件: losses.py — L_gain_sup 计算
# ============================================

# ---- Flight4（当前，错误）----
# target_gain = GT / (img_s2 + eps)
# gain_loss = L1(gain_map, clamp(target_gain, G_min, G_max))

# ---- Flight5（修复）----
def compute_gain_loss(gain_map, img_curved, gt, G_min, G_max, 
                      phase, eps=1e-4):
    """
    gain 监督应基于 curve 输出后的残差。
    
    Phase 1/1.5: detach img_curved，防止 gain_loss 反传到 curve
    Phase 2:     去掉 detach，允许联合微调
    """
    if phase < 2:
        # Phase 1: curve 梯度不受 gain_loss 影响
        target_gain = gt / (img_curved.detach() + eps)
    else:
        # Phase 2: 允许联合优化
        target_gain = gt / (img_curved + eps)
    
    target_gain = target_gain.clamp(G_min, G_max)
    return F.l1_loss(gain_map, target_gain)
```

**预期效果**：
- curve 输出 ~0.5 时，target_gain ≈ GT/0.5 ≈ 1.0 → gain 学到正常值
- curve 输出 ~0.99（仍饱和）时，target_gain ≈ 0.5 → gain 仍会做刹车，**但至少监督和实际行为一致**
- 与修改 2 配合时，gain 将完全脱离底限

### 修改 2：Target-Convergent Curve（修复 LE-curve 发散）

**病灶**：LE-curve 唯一正不动点 = 1.0，无法停在中间亮度
**修复**：添加 (α_target − LE) 收敛因子

**数学形式**：

$$TCC_n(x) = TCC_{n-1}(x) + A_n(x) \cdot TCC_{n-1}(x) \cdot (1 - TCC_{n-1}(x)) \cdot (\alpha_{target} - TCC_{n-1}(x))$$

**不动点**：$TCC^* = 0$, $TCC^* = 1$, 或 $TCC^* = \alpha_{target}$

当 $LE$ 从 0.1 向上增加时，经过 $\alpha_{target}$ 后 $(\alpha_{target} - LE)$ 变负，增量变负，**曲线自动回落**——这就是收敛机制。

根据 [Zero-DiDCE](https://link.springer.com/article/10.1007/s11063-024-11565-5) 的实验结论：

> *"ALE-curve 可以将输入图像光照逐步收敛到 α，即使使用 20 次迭代也不会导致过曝。"*

```python
# ============================================
# 文件: ispn.py — TargetConvergentCurve
# ============================================

class TargetConvergentCurve(nn.Module):
    """
    Flight5: 目标收敛曲线，替换原 ZeroDCE 8-iter LE-curve。
    
    关键改进：
    1. 添加 (α_target - LE) 收敛因子
    2. A 范围扩展到 [-4, 4]（补偿收敛因子导致的增量缩小）
    3. 迭代次数 8→6
    4. α_target 可学习，初始化 0.5
    """
    
    def __init__(self, feat_ch=64, n_iter=6):
        super().__init__()
        self.n_iter = n_iter
        
        # 可学习目标亮度：sigmoid(0.0) = 0.5
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        
        # A maps 预测网络（per-pixel, per-iter, per-channel）
        self.curve_net = nn.Sequential(
            nn.Conv2d(feat_ch, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3 * n_iter, 3, padding=1),  # → 18 ch
        )
        # 零初始化最后一层 → Phase 1 输出 = identity
        nn.init.zeros_(self.curve_net[-1].weight)
        nn.init.zeros_(self.curve_net[-1].bias)
    
    @property
    def alpha_target(self):
        """目标亮度，(0, 1) 范围"""
        return torch.sigmoid(self.alpha_raw)
    
    def forward(self, img, feat):
        """
        Args:
            img:  B,3,H,W — S2 输出（~0.1 mean）
            feat: B,64,H,W — refine features
        Returns:
            le:   B,3,H,W — 曲线增强后的图像
        """
        # 预测 A maps，范围 [-4, 4]
        A_raw = self.curve_net(feat)                    # B, 18, H, W
        A_maps = 4.0 * torch.tanh(A_raw)               # [-4, 4]
        A_maps = A_maps.view(
            img.shape[0], self.n_iter, 3, *img.shape[2:]
        )  # B, 6, 3, H, W
        
        alpha = self.alpha_target  # scalar in (0, 1)
        
        le = img
        for i in range(self.n_iter):
            A_i = A_maps[:, i]     # B, 3, H, W
            # Target-Convergent Curve
            delta = A_i * le * (1.0 - le) * (alpha - le)
            le = le + delta
        
        # 安全网：soft clamp（正常情况下不应被激活）
        le = 0.5 + 0.5 * torch.tanh(4.0 * (le - 0.5))
        
        return le
```

**A 范围为什么是 [-4, 4]？** 数学验证：

取 α_target=0.5, A=3.0, 从 LE=0.10 出发：

| Iter | LE | 增量 = 3.0 × LE(1−LE)(0.5−LE) | 说明 |
|:----:|:----:|:----:|:----:|
| 0 | 0.100 | +0.108 | |
| 1 | 0.208 | +0.144 | |
| 2 | 0.352 | +0.101 | |
| 3 | 0.453 | +0.035 | 接近 α |
| 4 | 0.488 | +0.005 | **收敛** |
| 5 | 0.493 | +0.001 | |
| 6 | **0.494** | — | ≈ α_target ✅ |

**6 次迭代从 0.10 → 0.494，且迭代 7/8/9/100 次仍然 ≈ 0.5**——永不饱和。

对比原 LE-curve 8 次迭代（A=0.76）→ **0.994（全白）**。

### 修改 3：迭代次数 8→6 + Gain 初始化调整

```python
# ============================================
# 迭代次数（已包含在修改 2 的 n_iter=6 中）
# ============================================
# Conv2d 输出通道: 24 (8×3) → 18 (6×3)
# 参数减少: 32×24 - 32×18 = 192 参数（忽略不计）

# ============================================
# Gain 初始化（ispn.py — GainBranch）
# ============================================

# Flight4: gain_head[-1].bias = 0.8
#   → raw=0.8, softplus(0.8)/softplus(Gmax) * (Gmax-0.5) + 0.5
#   → G ≈ 0.8（偏高，但被曲线饱和压到 0.5）

# Flight5: gain_head[-1].bias = 0.0
#   → raw=0.0, softplus(0.0)/softplus(4.0) * (4.0-0.5) + 0.5
#   → G ≈ 0.5 + 0.693/4.018 * 3.5 ≈ 1.10
# 
# 含义：初始 gain ≈ 1.1（近似 identity）
# 配合 TCC 输出 ≈ 0.5，最终 0.5 × 1.1 ≈ 0.55，合理起点
nn.init.zeros_(self.gain_head[-1].bias)
```

---

## 四、三项修改的协同效应

```
Flight4 数据流（当前）：
  img_s2(0.10) → LE-curve-8iter → img_curved(0.99) → ×gain(0.50) → output(0.50)
  gain_sup target = GT/img_s2 = 5.0（与实际 gain=0.5 矛盾）
  → PSNR 14.02

Flight5 数据流（设计）：
  img_s2(0.10) → TCC-6iter → img_curved(0.50) → ×gain(1.0-1.2) → output(0.50-0.60)
  gain_sup target = GT/img_curved = 1.0-1.2（与实际 gain 一致）
  → 预期 PSNR 17-20
```

关键变化：
1. TCC 输出 ≈ 0.5（不再饱和） → **gain 不需要刹车**
2. gain 监督 ≈ GT/0.5 ≈ 1.0 → **与 gain 实际范围一致**
3. gain 可以自由学习 0.8-1.5 的微调 → **curve/gain 真正分工**

---

## 五、完整改动清单

| # | 文件 | 修改内容 | 行数 | 对应病灶 |
|---|------|---------|:----:|---------|
| 1 | `ispn.py` | `SpatialCurveBranch` → `TargetConvergentCurve` | ~40 | 曲线饱和 |
| 2 | `ispn.py` | A 范围: `tanh` → `4*tanh` | 1 | TCC 增量补偿 |
| 3 | `ispn.py` | 新增 `self.alpha_raw = nn.Parameter(0.0)` | 1 | 可学习目标 |
| 4 | `ispn.py` | 迭代: 8→6, Conv2d out: 24→18 | 2 | 过迭代 |
| 5 | `ispn.py` | gain bias: 0.8 → 0.0 | 1 | gain 初始值 |
| 6 | `losses.py` | `target_gain = GT/(img_curved.detach()+eps)` | 3 | 监督错位 |
| **总计** | | | **~50行** | **+1 参数** |

**不改的东西**：
- Encoder / TCA / SGRF：无问题
- NDPN（gn=0.021）：健康运转
- MCPN（gm=0.007）：受限于数据集，暂搁
- Phase 调度逻辑：保持不变
- soft_clamp：保留作安全网
- 学习率 / batch size / 数据增强：保持不变

---

## 六、验证标准

### Ep10 决策树

```
Flight5 Ep10 检查点：
│
├─ 检查 curve output mean
│   ├─ 0.35-0.65 → ✅ TCC 工作正常
│   ├─ > 0.85    → 🔴 A 范围不够或 α_target 学偏
│   └─ < 0.20    → 🔴 A 太小，未能提亮
│
├─ 检查 gain
│   ├─ > 0.70    → ✅✅ 脱离底限，分工修复成功
│   ├─ 0.55-0.70 → 🟡 部分改善，可能需要调 α_target
│   └─ = 0.50    → 🔴 仍锁死（可能 detach 未生效或其他问题）
│
├─ 检查 α_target（sigmoid(alpha_raw)）
│   ├─ 0.40-0.65 → ✅ 学到合理目标
│   ├─ > 0.85    → 🔴 趋向 LE-curve 退化（目标太高）
│   └─ < 0.20    → 🔴 欠增强
│
└─ 检查 PSNR
    ├─ > 16.0    → ✅ 方案验证成功
    ├─ 14.5-16.0 → 🟡 有改善但不够
    └─ < 14.0    → 🔴 退步，紧急排查
```

### 关键监控量（建议新增到 TensorBoard）

```python
# 每个 validate epoch 记录：
log('curve/output_mean', img_curved.mean())
log('curve/output_std', img_curved.std())
log('curve/saturated_ratio', (img_curved > 0.95).float().mean())
log('curve/alpha_target', torch.sigmoid(model.ispn.curve.alpha_raw))
log('gain/mean', gain_map.mean())
log('gain/p10', gain_map.quantile(0.1))
log('gain/p90', gain_map.quantile(0.9))
log('gain/target_mean', target_gain.mean())
```

---

## 七、预期训练轨迹

| Epoch | PSNR（预期） | gain（预期） | curve out（预期） | 说明 |
|:-----:|:-----------:|:-----------:|:----------------:|------|
| 1 | ~10 | ~1.1 | ~0.10 | 零初始化，curve=identity |
| 5 | ~13 | ~0.9 | ~0.30 | TCC 开始提亮 |
| 10 | **16-17** | **0.8-1.2** | **0.40-0.55** | ✅ 第一个验证点 |
| 20 | **18-19** | 0.9-1.3 | 0.45-0.55 | 可能追上 Mod6 |
| 30 | 19-20 | 1.0-1.4 | ~0.50 | Phase 1.5/2 加速 |
| 50 | **21-23** | 1.0-1.5 | ~0.50 | 目标峰值 |

**对比 DeepSeek V4 的预测**：DeepSeek 认为 Flight4 在 ep50-60 能追上 Mod6（19.77）。我认为 Flight4 不改是追不上的，但**Flight5 在 ep20-30 就应该追上或超越 Mod6 峰值**。

---

## 八、风险与缓解

| 风险 | 概率 | 缓解 |
|------|:----:|------|
| TCC 增量不足（A=4 仍不够） | 低 | ep5 监控 curve_output_mean；若 <0.2 则扩到 6×tanh |
| α_target 学偏到 0.9+ | 低 | sigmoid 已限制 (0,1)；若偏高加弱正则 (α-0.5)² |
| gain_loss detach 导致 Phase 1 curve 太自由 | 中 | L_pix 仍约束最终输出；curve 不可能完全跑偏 |
| Phase 2 去掉 detach 后梯度冲突 | 低 | Phase 2 lr 已降低；监控 gain 是否突变 |
| 零初始化 + TCC = 前几 epoch 很暗 | 确定 | 正常现象，Phase 1 L_pix 会快速推动 A 增大 |

---

## 九、与之前方案的演化关系

```
Flight3 Mod6: hard clamp → 训练好看(19.77)但推理差(13.85)
     ↓ 修复 clamp
Flight4:      soft clamp → 训练真实(14.02)但曲线仍饱和、gain仍锁死
     ↓ 修复曲线公式 + gain监督
Flight5:      TCC + gain重定向 → 预期曲线不饱和、gain自由学习
```

每一步只修复了**前一步暴露出的真实问题**：
- Mod6 暴露了 clamp 伪收益 → Flight4 修复
- Flight4 暴露了曲线饱和 + gain 监督错位 → Flight5 修复

---

## 参考来源

1. [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf) — 原始 LE-curve 公式、迭代次数消融（n=8 为平衡点）、L_exp 曝光控制损失设计
2. [Rethinking Zero-DCE for Low-Light Image Enhancement (Zero-DiDCE, Neural Processing Letters, 2024)](https://link.springer.com/article/10.1007/s11063-024-11565-5) — ALE-curve 收敛性证明、α=0.63 最优、动态迭代器 18.23 vs 固定 20iter 15.56
3. [Analysis of DCE Iterations Using ResNetDCE for Low-Light Enhancement (ICCRAIDS, 2026)](https://doi.org/10.1109/iccraids67816.2026.11519577) — ResNet18+DCE-Net 混合架构、9iter=21.05dB 最优、12iter 开始下降、LPIPS 与 PSNR 的 trade-off
4. [Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation (TPAMI, 2022)](https://doi.org/10.1109/tpami.2021.3063604) — ZeroDCE++ 轻量化扩展
5. [Self-DACE++: Robust Low-Light Enhancement via Efficient Adaptive Curve Estimation (arXiv, 2025)](https://arxiv.org/html/2604.25367v2) — 去噪+曲线联合框架
6. [Stereo-IDCE: Interactive Dual-Curve Estimation for Unsupervised Low-Light Stereo Image Enhancement (2026)](https://doi.org/10.47760/ijcsmc.2026.v15i01.005) — 曲线复用增强（CRLE）、跨视角一致性损失
7. [I-INR: Iterative Implicit Neural Representations (2025)](https://arxiv.org/html/2504.17364v4) — 迭代步数消融：PSNR 峰值在 steps=4，超过后下降但感知质量继续改善
8. [Zero-DCE Keras 参考实现](https://keras.io/examples/vision/zero_dce/) — 标准实现参考