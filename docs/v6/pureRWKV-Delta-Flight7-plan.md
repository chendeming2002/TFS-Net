# Flight6 实证分析与 Flight7 设计

---

## 一、Flight6 结果的精确诊断

### 1.1 成功与失败的分离

Flight6 实现了其**设计目标**（gain DOF 优势），但未实现其**性能目标**（PSNR 提升）。这两件事必须分开理解：

| 维度 | 结果 | 判定 |
|------|------|:----:|
| gain 全程 >1 | ✅ 1.14→1.30，从未 <1 | 🟢 架构目标达成 |
| gain 有空间信息 | ✅ std=0.5 | 🟢 像素级 sigmoid 生效 |
| PSNR 超越 Flight5 | ❌ 峰值 17.86 ≈ Flight5 的 17.84 | 🟡 持平 |
| PSNR 持续提升 | ❌ ep10→ep50: 17.86→16.38 (-1.48) | 🔴 严重退化 |
| MCPN 保持活跃 | ❌ gm: 0.010→0.0076 | 🔴 持续衰减 |

### 1.2 关键转折点：ep10→ep20 的 -1.13 dB 崩塌

```
时间线：
ep10 (Phase1 结束): PSNR=17.86, gain=1.14
  ← Phase1: 只有 TCC+gain+L_pix+L_gain_sup, NDPN/MCPN=0
  
ep20 (Phase1.5 中): PSNR=16.73, gain=~1.2
  ← Phase1.5: NDPN/MCPN 渐进解锁 (unlock_ratio 从0→1)
  ← 崩塌 = -1.13 dB
```

**对比 Flight5 的同一转折**：
- Flight5 ep10→ep20: 16.79→17.84 = **+1.05 dB（上升）**
- Flight6 ep10→ep20: 17.86→16.73 = **-1.13 dB（崩塌）**

**同样是 Phase1.5 解锁 NDPN/MCPN，为什么 Flight5 受益而 Flight6 受损？**

### 1.3 根因分析：不是曲线 DOF 不够，而是梯度路径冲突

根据联网检索（检索于 2026-07-13T14:42:08+08:00），[Zero-DCE++ (TPAMI 2022)](https://ar5iv.labs.arxiv.org/html/2103.00860) 的 Table III 明确证明：

| 下采样倍率 | PSNR |
|:----------:|:----:|
| 原始分辨率 | 16.09 |
| 4× ↓ | 16.29 |
| **12× ↓** | **16.42** 🏆 |
| 50× ↓ | 15.85 |

> **"downsampling the sizes of input has unnoticeable effect on the enhancement performance but significantly saves computational cost"** —— [Zero-DCE++](https://ar5iv.labs.arxiv.org/html/2103.00860)

这完全推翻了"4× 下采样过于激进"的假设。**曲线估计对空间分辨率极其鲁棒**——在 1200×900 上 12× 下采样（→ 100×75）反而 PSNR 最高。对于我们的 256×256 输入，4× 下采样到 64×64 远远足够。

**真正的根因在别处。** 让我重新审视数据流：

```
Flight6 数据流（Phase 1.5/2）：
                    ┌─ feat_ds(64×64) → curve_net → A_map → bilinear↑ → TCC 6iter → curve_out
f_enc_center(64ch) ─┤
                    └─ h(64ch) → gain_net(conv16→conv1→sigmoid) → gain_map
                    
                    curve_out × gain_map → SGRF → res_t
                                              ↑
                    f_noise_gated + f_motion_gated ─────┘
```

**关键问题**：当 NDPN/MCPN 解锁时，`f_noise_gated` 和 `f_motion_gated` 进入 SGRF。但是看 `tfs_net.py` 中的代码：

```python
sgrf_out = self.sgrf(
    gain_map=gain_map,
    f_noise_out=f_noise_gated,
    f_motion_out=f_motion_gated,
    image_center=image_center,
    curve_A=curve_A,
    alpha_target=alpha_target,
    curve_iter=self.ispn.curve_iter,
)
```

SGRF 同时接收 `gain_map`、`curve_A`（曲线参数）、`f_noise_out`、`f_motion_out`。当 NDPN/MCPN 从 0 渐变到非零时，SGRF 的输出发生突变——但此时 **gain 和曲线已经在 Phase1 收敛到了一个不需要 NDPN/MCPN 的平衡点**。

这就是 Flight5 与 Flight6 的关键差异：

| | Flight5 | Flight6 |
|---|---|---|
| Phase1 的 gain | 被曲线排挤到 <1 | 成功保持 >1 |
| Phase1 的曲线+gain 状态 | **未完全收敛**（gain<1 = 亮度不足） | **已充分收敛**（gain>1 + curve 协同 = 亮度匹配） |
| Phase1.5 解锁时 | NDPN/MCPN 填补亮度缺口 → **有用** → PSNR↑ | NDPN/MCPN 没有亮度缺口可填 → **干扰已有平衡** → PSNR↓ |

**Flight6 的悖论**：Phase1 太成功了，以至于 Phase1.5 的 NDPN/MCPN 解锁变成了纯粹的噪声源。

### 1.4 NDPN/MCPN 为什么持续衰减？

看 `ndpn.py` 中的输出：
```python
f_noise_out = f_enc - noise_feat * strength * self.gamma + self.noise_proj(s_noise)
```

看 `mrpn.py` 中的输出：
```python
hat_f_t = self.refine(f_t_fuse) + f_center * self.out_scale
# out_scale 初始化为 0，refine.conv2 初始化为 0
# → 初始时 hat_f_t ≈ f_t_fuse ≈ g_t * f_center（因为 startup_gate=1 → g_t=1）
```

MCPN 的 `startup_gate=1` 使得 `g_t=1`，输出 ≈ `f_center`。这意味着 MCPN 在初始状态下**是恒等映射**。当它进入 SGRF 时，如果 SGRF 已经在 Phase1 学会了"忽略运动分支"，那么 MCPN 永远得不到有效梯度。

**gm (0.010→0.0076) 说明 MCPN 在主动缩小自己**——因为它的任何输出变化都会破坏已收敛的 Phase1 平衡，而 L_pix 梯度会惩罚这种破坏。

---

## 二、Flight5 vs Flight6 的结构性差异

从代码层面理解两者的不同：

| 特性 | Flight5 | Flight6 |
|------|---------|---------|
| 曲线 DOF | 18HW（高） | 0.1875HW（低） |
| gain DOF | ~1（标量/GAP） | 1HW（像素级） |
| Phase1 收敛程度 | **不充分**（gain<1 = 系统性偏暗） | **充分**（gain>1 + curve = 亮度匹配） |
| NDPN/MCPN 角色 | 填补 Phase1 的亮度缺口 | **无缺口可填** → 成为噪声 |
| Phase1.5 趋势 | PSNR↑ | PSNR↓ |

**核心洞察**：Flight6 的 DOF 再平衡解决了 gain 排挤问题，但**创造了新问题**——Phase1 的 TCC+gain 组合已经足够好，以至于多帧信息（NDPN/MCPN）找不到自己的角色。

---

## 三、Flight7 设计：分离提亮与去噪/去模糊的梯度路径

### 3.1 核心问题重述

```
Flight6 的根本矛盾：
  1. 提亮路径（TCC+gain）在 Phase1 已收敛到好的亮度
  2. 去噪路径（NDPN）和去模糊路径（MCPN）需要修改已有输出来做精细化
  3. 但任何修改都会增加 L_pix，导致这些分支被惩罚性地缩小 gamma
  
  → 需要的不是"更好的 DOF 平衡"，而是"让 NDPN/MCPN 有独立的梯度目标"
```

### 3.2 设计原则

```
Flight7 的核心思路：

  P1（ep0-15）: TCC + gain 独立训练亮度提升 → 输出 "img_lit"
  
  P1.5/P2（ep15+）: NDPN + MCPN 在 "img_lit" 基础上做 **残差精修**
                     ↓
                     独立监督目标：
                       - NDPN: 减少 img_lit 与 GT 之间的高频噪声差异
                       - MCPN: 减少 img_lit 与 GT 之间的结构/运动伪影

  关键区别：NDPN/MCPN 不参与"提亮"决策，只做"去噪+去模糊"残差
```

### 3.3 架构修改

#### 修改 1：SGRF 输出拆分为两阶段

当前 SGRF 将 gain_map、curve_A、f_noise_out、f_motion_out 一次性混合。Flight7 拆分为：

```python
# ============================================
# 文件: igrf.py — SGRF (Flight7 修改)
# ============================================

class SGRF(nn.Module):
    """Flight7: 两阶段输出
    
    Stage A: TCC curve + gain → img_lit (提亮完成的中间结果)
    Stage B: img_lit + NDPN/MCPN residual → res_t (最终输出)
    """
    def __init__(self, channels=64, out_channels=3, ...):
        super().__init__()
        # ... 保持现有结构 ...
        
        # Flight7 新增：残差融合头
        # 将 f_noise_gated + f_motion_gated 转换为 RGB 残差
        self.residual_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        # 零初始化：初始时残差=0，输出=img_lit
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        
        # Flight7: 残差强度控制（可学习，初始=0）
        self.residual_scale = nn.Parameter(torch.zeros(1))
    
    def forward(self, gain_map, f_noise_out, f_motion_out, 
                image_center, curve_A, alpha_target, curve_iter):
        # === Stage A: 提亮 ===
        # S1: 初始提亮（现有逻辑）
        img_s1 = image_center * gain_map  # 或更复杂的 S1
        
        # S2: TCC 曲线
        alpha = alpha_target
        le = img_s1
        for n in range(curve_iter):
            A_n = curve_A[:, n]
            le = le + A_n * le * (1 - le) * (alpha - le)
        img_lit = soft_clamp(le)  # 提亮完成
        
        # === Stage B: 去噪+去模糊残差 ===
        # 融合 NDPN/MCPN 特征
        f_combined = f_noise_out + f_motion_out  # 简单相加
        residual = self.residual_head(f_combined)  # (B, 3, H, W)
        
        # 缩放控制
        scale = torch.tanh(self.residual_scale)  # (-1, 1)
        
        # 最终输出 = 提亮 + 残差
        res_t = soft_clamp(img_lit + residual * scale)
        
        return {
            "res_t": res_t,
            "img_s1": img_s1,
            "img_s2": img_lit,      # Stage A 输出
            "img_curved": img_lit,
            "img_lit": img_lit,     # Flight7: 用于独立监督
            "residual": residual * scale,  # Flight7: 残差可视化
            "lit_up_map": gain_map,
        }
```

#### 修改 2：损失函数分离

```python
# ============================================
# 文件: losses.py — Flight7 损失设计
# ============================================

def flight7_loss(outputs, gt, phase, epoch):
    """
    核心思路：提亮和去噪/去模糊有独立的监督信号
    """
    img_lit = outputs["img_lit"]      # Stage A 输出（纯提亮）
    res_t = outputs["res_t"]          # Stage B 输出（提亮+残差）
    residual = outputs["residual"]    # 残差本身
    
    losses = {}
    
    # === 提亮损失（全程作用于 img_lit） ===
    losses["L_lit"] = F.l1_loss(img_lit, gt)
    
    # === 去噪/去模糊损失（Phase1.5+ 作用于 res_t） ===
    if phase in ('phase1_5', 'phase2'):
        # 最终输出的 L1
        losses["L_final"] = F.l1_loss(res_t, gt)
        
        # 残差正则：鼓励残差只修正高频细节，不改变整体亮度
        # （防止残差分支"接管"提亮功能）
        losses["L_residual_reg"] = residual.abs().mean() * 0.1
        
        # SSIM 损失只作用于最终输出
        losses["L_ssim"] = 1.0 - ssim(res_t, gt)
    else:
        # Phase1: 最终输出 = img_lit（因为残差=0）
        losses["L_final"] = losses["L_lit"]
    
    # === gain 监督（保持 Flight6 验证过的） ===
    if phase in ('phase1', 'phase1_warmup', 'phase1_5'):
        losses["L_gain_sup"] = compute_gain_supervision(...)
    
    return losses
```

**关键设计**：
- `L_lit` 全程作用于 `img_lit`（TCC+gain 的纯输出）。这保证了 Phase2 不会通过去噪残差来"绕过"提亮分支。
- `L_final` 作用于最终输出 `res_t = img_lit + residual`。NDPN/MCPN 的梯度只通过 `residual` 传播，不会反向影响 TCC+gain。
- `L_residual_reg` 正则化残差幅度，防止残差分支膨胀到"重新做提亮"。

#### 修改 3：梯度隔离（stop_grad）

```python
# ============================================
# SGRF.forward 中的关键一行
# ============================================

# Stage B 中，img_lit 对残差分支的输入做 detach
# → 残差分支的梯度不会回传到 TCC/gain

# 方案 A：完全隔离
res_t = soft_clamp(img_lit.detach() + residual * scale)

# 方案 B：部分隔离（允许微弱梯度，帮助残差理解上下文）
# res_t = soft_clamp(img_lit * 0.1 + img_lit.detach() * 0.9 + residual * scale)
```

**推荐方案 A**。理由：Flight6 证明了 TCC+gain 在 Phase1 能收敛到 17.86 PSNR。梯度隔离保护这个成果不被后续分支破坏。

#### 修改 4：ISPN 曲线参数——进一步下采样验证

根据 [Zero-DCE++ Table III](https://ar5iv.labs.arxiv.org/html/2103.00860) 的证据，4× 下采样已经足够（对 256×256 输入得到 64×64），但 **12× 可能更优**。考虑到我们的输入是 256×256：

```python
# Flight7: 尝试 8× 下采样（256→32）
# 比 4×(→64) 更低，但比 12×(→21) 留更多余量

class ISPN(nn.Module):
    def __init__(self, ..., ds_factor: int = 8):  # 从4改为8
        ...
```

对于 256×256：
- 4× → 64×64 → DOF = 3×64×64/65536 = 0.1875 HW
- **8× → 32×32 → DOF = 3×32×32/65536 = 0.047 HW**
- 12× → 21×21 → DOF = 3×21×21/65536 = 0.020 HW

Zero-DCE++ 的数据（在 1200×900 上）表明 12× 最优对应输出分辨率 100×75。映射到 256×256：
- 100/1200 = 0.083 → 256×0.083 ≈ 21 像素
- 但 21×21 可能对卷积过小（3×3 conv 需要至少 7×7 输入才有意义）

**结论：Flight7 使用 ds_factor=8（→32×32）**，在 Zero-DCE++ 的最优区间内，且保证卷积有效。

#### 修改 5：NDPN/MCPN gamma 重新初始化

```python
# Flight7: NDPN/MCPN gamma 初始化更大
# 理由：在 Flight6 中 gm=0.01→0.0076 = 被压制
# Flight7 中有残差隔离保护，gamma 可以更大

# ndpn.py
self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 0.05))  # 从0.01→0.05

# mrpn.py  
self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 0.05))  # 从0.01→0.05
```

### 3.4 完整数据流（Flight7）

```
输入: x (B,T,3,H,W)
  │
  ├─ Encoder → feats (B,T,64,H,W)
  │
  ├─ SWD → feat_tfde, feat_tca
  │
  ├─ DPE(feat_tfde) → s_illum, s_noise
  │
  ├─ TCA(feat_tca) → F_aligned_list, F_t_aligned, mu_t_clean, sigma_t_clean
  │
  ├─ ISPN(f_enc_center, s_illum):
  │     ├─ refine → h (64ch, H×W)
  │     ├─ h → gain_net → gain_map (1ch, H×W) ∈[0.5, 2.0]
  │     └─ h → AvgPool(8×) → curve_net(32×32) → bilinear↑ → A_map (3ch, H×W)
  │
  ├─ SGRF Stage A (提亮):
  │     img_s1 = image_center × gain_map  (或 + residual S1)
  │     img_lit = TCC(img_s1, A_map, α, 6iter)
  │     img_lit = soft_clamp(img_lit)
  │     ← L_lit = L1(img_lit, gt)  [全程监督]
  │
  ├─ NDPN(feats, F_aligned, s_noise...) → f_noise_out  [Phase1.5+]
  ├─ MCPN(F_aligned, sigma_t_clean...) → f_motion_out  [Phase1.5+]
  ├─ CXG(f_noise_out, f_motion_out) → f_noise_gated, f_motion_gated
  │
  └─ SGRF Stage B (残差精修):
        f_combined = f_noise_gated + f_motion_gated
        residual = residual_head(f_combined)  (3ch, H×W)
        res_t = soft_clamp(img_lit.detach() + residual × tanh(scale))
        ← L_final = L1(res_t, gt)  [Phase1.5+]
        ← L_residual_reg = |residual|.mean() × 0.1
```

### 3.5 Phase 策略

| Phase | Epochs | 活跃模块 | 损失 | 目标 |
|:-----:|:------:|:--------:|:----:|:-----|
| P1 | 0-15 | ISPN(TCC+gain) + SGRF-A | L_lit + L_gain_sup×0.3 | 提亮收敛 |
| P1.5 | 15-30 | + NDPN/MCPN 渐进 + SGRF-B | + L_final + L_res_reg | 残差学习 |
| P2 | 30-50 | 全部 | L_lit + L_final + L_ssim + L_res_reg | 精细化 |

**关键区别 vs Flight6**：
- P1 的 `L_gain_sup` 权重从 0.5 降到 0.3（gain 已证明不需要强监督）
- P1.5 解锁时，**`img_lit.detach()` 保护提亮分支不被扰动**
- P2 的 `L_lit` 持续监督 → TCC+gain 全程有明确梯度，不会被 NDPN/MCPN 干扰

---

## 四、预期效果分析

### 4.1 为什么 Flight7 能解决 Flight6 的退化？

| Flight6 问题 | Flight7 解法 | 机制 |
|:------------|:-----------|:-----|
| Phase1.5 解锁后 PSNR 崩塌 | `img_lit.detach()` 梯度隔离 | 残差分支的梯度不会回传到 TCC/gain |
| MCPN gamma 持续下降 | gamma 初始 0.05 + 残差正则保护 | 残差分支有独立 L_final 目标 |
| gain 虽 >1 但 PSNR 不涨 | L_lit 全程独立监督 img_lit | TCC+gain 始终有直接梯度信号 |
| NDPN/MCPN "无事可做" | 残差分支只需消除 img_lit 与 GT 的差 | 明确的角色：去噪+去模糊 |

### 4.2 预期轨迹

| Epoch | Phase | PSNR 预期 | gain | 机理 |
|:-----:|:-----:|:---------:|:----:|:-----|
| 10 | P1 end | 17.5-18.0 | 1.1-1.3 | 复现 Flight6 Phase1 |
| 20 | P1.5 mid | **17.8-18.5** | 1.1-1.3 | **不崩塌**（detach 保护） |
| 30 | P1.5 end | **18.5-19.5** | 1.0-1.3 | 残差开始贡献 |
| 40 | P2 early | 19.0-20.0 | 1.0-1.3 | SSIM 损失精修 |
| 50 | P2 mid | **19.5-20.5** | 1.0-1.3 | 收敛 |

**核心预期**：
1. ep10 ≈ Flight6 ep10（~17.86）——Phase1 部分完全相同
2. ep20 **不崩塌**（Flight6: -1.13, Flight7 预期: +0~0.5）
3. ep30+ **持续上升**（残差分支逐步消除噪声/模糊）

### 4.3 Epoch 10 & 20 检查清单

```
✅ Flight7 验证成功:
  ep10: PSNR > 17.5, gain > 1.0
  ep20: PSNR ≥ ep10 (不崩塌!)
  ep20: residual.abs().mean() < 0.05 (残差小但非零)
  ep20: NDPN gamma ≥ 0.04 (未被压缩)

🟡 部分成功:
  ep20: PSNR = ep10 ± 0.3 (未崩塌但也未提升)
  → 可能需要更大的 gamma 初始值或更小的 L_res_reg

🔴 需要干预:
  ep10: PSNR < 17.0 (说明 ds_factor=8 过于激进)
  ep20: residual.abs().mean() > 0.2 (残差分支接管了提亮)
  → detach 可能没生效，检查代码
```

---

## 五、代码实施清单

| # | 文件 | 改动 | 估计行数 |
|:-:|:----:|:-----|:--------:|
| 1 | `ispn_v2.py` | `ds_factor` 从 4→8 | 1 行 |
| 2 | `igrf.py` (SGRF) | 新增 `residual_head` + `residual_scale` + 两阶段 forward | ~30 行 |
| 3 | `igrf.py` (SGRF) | `img_lit.detach()` 梯度隔离 | 1 行 |
| 4 | `losses.py` | 新增 `L_lit`(全程)、`L_final`(P1.5+)、`L_residual_reg` | ~20 行 |
| 5 | `ndpn.py` | gamma 初始 0.01→0.05 | 1 行 |
| 6 | `mrpn.py` | gamma 初始 0.01→0.05 | 1 行 |
| 7 | `tfs_net.py` | forward 返回新增 `img_lit`、`residual` 字段 | ~5 行 |
| 8 | `train.py` | Phase 配置适配新损失 | ~10 行 |

**总改动：~70 行**，核心是 #2（SGRF 两阶段）和 #3（detach）。

### 保持不变

- ✅ TCC 公式（6 iter, α_target 可学习）
- ✅ gain: 像素级 sigmoid [0.5, 2.0]
- ✅ soft_clamp
- ✅ warmup 100 步 + grad_clip 1.0
- ✅ 曲线 bias=0.25 初始化
- ✅ 参数共享（单 A_map 复用 6 次）
- ✅ Phase 渐进解锁逻辑

---

## 六、与 DeepSeek V4 方案对比

| DeepSeek 方案 | 我的评估 | Flight7 采纳？ |
|:-------------|:---------|:-------------:|
| A: 12× 下采样 | 方向正确但非根因。Flight6 的 4× 已足够（Zero-DCE++ 证据） | 部分：8× |
| B: 双分支解耦 | **方向最正确**——梯度解耦是核心 | ✅ 核心采纳 |
| C: TV loss + gamma 增大 | 治标手段，不解决梯度冲突 | gamma 增大采纳 |

**Flight7 的核心设计与 DeepSeek 方案 B 一致**，但实现方式不同：

- DeepSeek B：提亮分支完全独立（不共享编码器特征）
- **Flight7：共享编码器但在 SGRF 内部用 detach() 隔离梯度**

我选择 detach 方案的原因：
1. 改动更小，可回溯
2. 编码器特征对 NDPN/MCPN 的输入是必要的（它们需要 `f_enc_center`、`feats` 等）
3. 只需隔离"提亮结果→残差分支"这一条路径的梯度

---

## 七、总结

**Flight6 的根因不是"曲线 DOF 不够"，而是"提亮分支太成功导致去噪/去模糊分支无法介入"。**

Flight7 的解法：
1. **梯度隔离**：`img_lit.detach()` 让残差分支自由优化而不破坏已有的提亮质量
2. **独立监督**：`L_lit`（全程）+ `L_final`（Phase1.5+）给两条路径各自明确的目标
3. **残差正则**：`L_residual_reg` 防止残差分支膨胀为"第二个提亮器"
4. **更大的 gamma**：让 NDPN/MCPN 从一开始就有足够的输出幅度

---

## 参考来源

1. [Zero-DCE++ (TPAMI 2022) — 论文全文及下采样消融实验](https://ar5iv.labs.arxiv.org/html/2103.00860)
2. [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)
3. [Zero-DCE OpenReview 版本](https://openreview.net/pdf?id=k0E7LwLz6qz)
4. [Keras Zero-DCE 实现](https://keras.io/examples/vision/zero_dce/)

