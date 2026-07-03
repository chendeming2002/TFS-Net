# TSD-Net 两阶段渐进训练方案分析与设计

## 一、可行性批判性分析

### 1.1 方案核心逻辑评估

你的方案本质是**架构级的课程学习 (Architectural Curriculum Learning)**——先训练"简化子网"（仅光照+结构），再解锁全网络。这个思路在原则上是合理的，但需要仔细审视几个潜在陷阱。

### 1.2 支持性证据

| 文献 | 结论 | 与你方案的对应 |
|------|------|--------------|
| [D3Fusion (Applied Sciences 2025)](https://www.mdpi.com/2076-3417/15/16/8918) | 明确指出端到端训练"不同任务损失项在梯度反传中相互干扰"，需分阶段训练 | ✅ 你的动机与此一致 |
| [Progressive Training in LLIE](https://doi.org/10.48550/arxiv.2409.01641) (AFD-LLIE) | Laplace 金字塔逐层解锁，先低频光照后高频细节 | ✅ 你先训练 ISPN 光照路径符合此思路 |
| [Curriculum Learning (Bengio et al., 2009)](https://doi.org/10.1145/1553374.1553380) (被引 12000+) | 从简单任务开始逐步增加复杂度可显著加速收敛 | ✅ 理论基础 |
| [D²RNet (AAAI 2024)](https://doi.org/10.1609/aaai.v38i4.28144) | UDC 视频解耦训练时分支分别预训练效果更佳 | ✅ 支持分支独立预训练 |

### 1.3 潜在风险与需要修正的地方

**风险 1：SWD 在 Phase 1 会学到"偏斜"的分流策略** ⚠️

SWD 的可学习参数 $\alpha$ 和 $g_n$ 是**协同决定 TFDE 与 TCA 的分流比例的**。如果 Phase 1 完全关闭 TFDE 通路：
- $\alpha$ 会退化为 **0**（因为 LL_tfde 没有梯度信号）→ 全部 LL 都流向 TCA
- $g_n$ 会退化为 **0** → 全部 HF 都流向 TCA
- Phase 2 解锁后，SWD 需要**重新学习分流**，此时其他模块已经收敛，反而更难调整

**修正**：Phase 1 时**保留 TFDE→ISPN 的通路**（因为你说要训练 ISPN），SWD 的两个输出都应该有梯度回传。真正需要截断的只是 **NDPN、MCPN 的最终输出**。

**风险 2：SGRF 三阶段结构与训练顺序不匹配** ⚠️

你的 SGRF 顺序是 **S1 (去噪) → S2 (去模糊) → S3 (提亮)**。如果 Phase 1 截断 NDPN 和 MCPN 输出：
- $f_{\text{noise}}^{\text{out}}$ 为 0 → S1 的 $\delta_1(\cdot) = 0$ → $I_1 = I_t$（等于直通）
- $f_{\text{motion}}^{\text{out}}$ 为 0 → S2 的 $\delta_2(\cdot) = 0$ → $I_2 = I_1 = I_t$
- 只有 S3 提亮起作用

这实际上等价于**只训练了一个"简单的 Retinex 光照增强"模型**，S1/S2 的残差模块 $\delta_1, \delta_2$ 完全没有训练信号。Phase 2 解锁时，$\delta_1, \delta_2$ 是随机初始化状态，会引入巨大梯度冲击。

**修正**：Phase 1 时也应该让 $\delta_1, \delta_2$ 得到一些训练——可以通过**将 NDPN/MCPN 输出替换为 zero-init 的可学习特征**或**让 SGRF 直接接收 TCA 的 $\hat{F}_t$ 作为伪 noise/motion 特征**。

**风险 3：CXG 门控在 Phase 1 无法训练** ⚠️

CXG 是训练时的交叉门控，需要 $f_{\text{noise}}$ 和 $f_{\text{motion}}$ 都有意义才能学习。Phase 1 截断后 CXG 完全没有训练信号。

**修正**：Phase 1 直接**旁路 CXG**（identity 通过），Phase 2 才启用。

**风险 4：TCA 的 $C_{t,\Omega}$ 在 Phase 1 缺乏运动监督** ⚠️

$C_{t,\Omega}$ 的时序对应矩阵是 NDPN 和 MCPN 共同需要的。Phase 1 截断这两者，TCA 只被 ISPN 使用——而 ISPN 主要用 $\hat{F}_t$（聚合后的中心帧），并不强烈依赖 $C_{t,\Omega}$ 的对角线结构。**TCA 在 Phase 1 可能学到的对齐质量较低**。

**修正**：Phase 1 增加辅助损失监督 $C_{t,\Omega}$ 的合理性（详见 2.3 节的辅助损失设计）。

### 1.4 关于"从根本上改进"的判断

**结论：能显著缓解，但不能完全根本解决**。

- ✅ **对当前"结构损失/感知损失难下降"的问题有效**——因为 Phase 1 只优化 ISPN 主导的光照恢复，梯度冲突大幅减少
- ✅ **对"梯度冲突"和"三分支耦合"的问题有效**——符合 D3Fusion 和 D²RNet 的经验
- ⚠️ **对"batch=1 梯度噪声大"的问题无直接改善**——这是优化器层面的问题
- ⚠️ **对"感知-失真权衡"的根本 tradeoff 无改善**——这是理论下界

**根本改进需要三管齐下**：
1. **两阶段架构训练**（你提出的方案）
2. **Kendall 不确定性加权**（Mark2 已实施）
3. **感知解耦到 SGRF 阶段**（Mark2 的进一步方向）

三者结合才能"根本改进"。

---

## 二、修正后的两阶段训练方案设计

### 2.1 阶段总体规划

| 阶段 | Epoch | 训练模块 | 冻结/截断模块 | 学习目标 |
|------|-------|---------|--------------|---------|
| **Phase 1: Structural-Illumination Pretraining** | 0-19 | Encoder, SWD, TFDE, TCA, ISPN, SGRF (S3 主导) | NDPN 输出置零, MCPN 输出置零, CXG 旁路 | 基础结构对齐 + 光照恢复 |
| **Phase 1.5: Warm Unlock** | 20-24 | 全部（渐进解锁） | 无（NDPN/MCPN 逐步解冻） | 平滑过渡 |
| **Phase 2: Full Tri-Source Restoration** | 25-49 | 全部模块 | 无 | 三源联合优化 |

### 2.2 Phase 1 详细设计

#### 2.2.1 前向传播修改

在 `models/tfs_net.py` 的 `TFSNet.forward` 中增加 `phase` 参数：

```python
def forward(self, frames, phase='phase1'):
    """
    Args:
        frames: (B, T, 3, H, W)
        phase: 'phase1' | 'phase1_5' | 'phase2'
    """
    # ─── 通用前向 ───
    B, T, _, H, W = frames.shape
    F_seq = self.encoder(frames.flatten(0,1)).view(B, T, -1, H, W)
    
    # SWD 分流（Phase 1 也需要，保持 TFDE 通路开放）
    F_tfde, F_tca = [], []
    for t in range(T):
        f_tfde_t, f_tca_t = self.swd(F_seq[:, t])
        F_tfde.append(f_tfde_t); F_tca.append(f_tca_t)
    F_tfde = torch.stack(F_tfde, dim=1)
    F_tca = torch.stack(F_tca, dim=1)
    
    # TFDE 估计退化强度
    s_illum, s_noise = self.tfde(F_tfde)
    
    # TCA 时序对齐
    F_out_list, C_omega, F_hat, mu, sigma = self.tca(F_tca)
    
    # ─── ISPN(始终激活) ───
    lit_up_map, f_illum_feat, A_illu, ifpn_side = self.ispn(
        F_out_list, s_illum, F_hat
    )
    
    # ─── NDPN / MCPN(条件激活) ───
    if phase == 'phase1':
        # Phase 1: 截断 NDPN 和 MCPN 输出
        # 使用 zero_like 保持形状，detach 阻断梯度
        f_noise_out = torch.zeros_like(F_hat)
        f_motion_out = torch.zeros_like(F_hat)
        
        # CXG 旁路（identity）
        f_noise_gated = f_noise_out
        f_motion_gated = f_motion_out
        
    elif phase == 'phase1_5':
        # Phase 1.5: 渐进解锁（使用 epoch-dependent 缩放因子）
        f_noise_out = self.ndpn(F_out_list, s_noise, mu, sigma, C_omega, F_hat)
        f_motion_out = self.mcpn(F_out_list, sigma, C_omega, F_hat)
        
        # 使用 unlock_ratio 逐步引入(由训练循环传入)
        unlock_ratio = getattr(self, '_unlock_ratio', 0.0)
        f_noise_out = f_noise_out * unlock_ratio
        f_motion_out = f_motion_out * unlock_ratio
        
        # CXG 也逐步启用
        if unlock_ratio > 0.3:
            f_noise_gated, f_motion_gated = self.cxg(f_noise_out, f_motion_out)
        else:
            f_noise_gated, f_motion_gated = f_noise_out, f_motion_out
    
    else:  # phase2
        f_noise_out = self.ndpn(F_out_list, s_noise, mu, sigma, C_omega, F_hat)
        f_motion_out = self.mcpn(F_out_list, sigma, C_omega, F_hat)
        f_noise_gated, f_motion_gated = self.cxg(f_noise_out, f_motion_out)
    
    # ─── SGRF 三阶段(始终执行，但根据 phase 有不同行为) ───
    res_t, img_s1, img_s2 = self.sgrf(
        f_illum_feat, f_noise_gated, f_motion_gated, 
        lit_up_map, frames[:, T//2], A_illu
    )
    
    return {
        'res_t': res_t,
        'img_s1': img_s1,
        'img_s2': img_s2,
        's_illum': s_illum,
        's_noise': s_noise,
        'lit_up_map': lit_up_map,
        'A_illu': A_illu,
        'ifpn_side': ifpn_side,
        'C_omega': C_omega,
        'F_hat': F_hat,
        'F_out_list': F_out_list,
        'phase': phase,
    }
```

#### 2.2.2 关键设计要点

**① SWD 保持全通路（不截断 TFDE 侧）**——避免风险 1，让 SWD 的 $\alpha$ 和 $g_n$ 在 Phase 1 就学到合理分流。

**② SGRF 在 Phase 1 退化为"光照增强 + 微弱残差"**：
- $\delta_1(0) \approx \delta_1.\text{bias}$（残差模块的 bias 项）
- $\delta_2(0) \approx \delta_2.\text{bias}$
- 主要输出由 S3 的 $I_2 \cdot \text{lit\_up\_map} \cdot (1+A_{\text{illu}})$ 决定
- $\delta_1, \delta_2$ 的 bias 会得到微弱训练（不会完全"僵死"）

**③ 使用 `torch.zeros_like` 而非 `detach`**——因为我们要的是"没有信号"而非"有信号但阻断梯度"，前者更干净。

**④ CXG 在 Phase 1 完全旁路**——避免风险 3，CXG 的门控只在 Phase 1.5 及以后学习。

---

### 2.3 Phase 1 损失函数设计

**核心原则**：Phase 1 只监督 **ISPN + SGRF-S3 + TCA + SWD** 相关的信号，**不监督 NDPN/MCPN 相关的信号**（避免向截断的分支反传梯度）。

#### 2.3.1 Phase 1 损失项

$$\mathcal{L}_{\text{phase1}} = \mathcal{L}_{\text{illum}} + \mathcal{L}_{\text{align}} + \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{swd\_reg}}$$

**(A) 光照恢复损失** $\mathcal{L}_{\text{illum}}$：

$$\mathcal{L}_{\text{illum}} = \underbrace{\lambda_1 \cdot \text{PECharbonnier}(\hat{X}_t, X_t^{\text{gt}})}_{\text{像素级重建}} + \underbrace{\lambda_2 \cdot \|\text{lit\_up\_map} - \text{lit\_up\_map}^*\|_1}_{\text{光照图监督}} + \underbrace{\lambda_3 \cdot \text{TV}(s_{\text{illu}}) \cdot e^{-|\nabla I|}}_{\text{光照平滑}}$$

其中 $\text{lit\_up\_map}^* = \bar{I}^{\text{gt}} / (\bar{I}^{\text{input}} + \epsilon)$ 为伪监督光照比（用 GT 和输入的平均亮度估计）。

**(B) 时序对齐损失** $\mathcal{L}_{\text{align}}$（新增，直接监督 TCA）：

$$\mathcal{L}_{\text{align}} = \lambda_4 \cdot \sum_{t' \in \Omega} \|W(F_{t'}^{\text{out}}, C_{t,t'}) - F_t^{\text{out}}\|_1 + \lambda_5 \cdot \mathcal{L}_{\text{diag}}(C_{t,\Omega})$$

其中：
- 第一项是**warping consistency**——用 $C_{t,t'}$ 把邻帧 warp 到中心帧后应与中心帧特征一致
- 第二项 $\mathcal{L}_{\text{diag}}(C_{t,\Omega})$ 是**对角线先验损失**：

$$\mathcal{L}_{\text{diag}}(C_{t,\Omega}) = -\frac{1}{|\Omega|}\sum_{t'\in\Omega}\left[\mathbb{1}_{\text{static}} \cdot \log C_{t,t'}[i,i]\right]$$

即：对于**静止区域**（GT 帧间差分 $|X_t^{\text{gt}} - X_{t'}^{\text{gt}}| < \tau_{\text{static}}$，$\tau_{\text{static}}=0.02$），$C_{t,t'}$ 的对角线元素应接近 1（鼓励静态对应）。这个损失让 TCA 在 Phase 1 就学到合理的对应结构，避免风险 4。

**(C) 结构重建损失** $\mathcal{L}_{\text{recon}}$：

$$\mathcal{L}_{\text{recon}} = \lambda_6 \cdot (1 - \text{SSIM}(\hat{X}_t, X_t^{\text{gt}})) + \lambda_7 \cdot \|\text{IFPN\_side} - \mathcal{D}(X_t^{\text{gt}})\|_1$$

其中 $\mathcal{D}$ 是下采样算子。注意 Phase 1 **不使用 VGG 感知损失**——因为此时输出的高频细节几乎全部由 SGRF-S3 简单放大产生，感知损失会向 SGRF 施加"不切实际"的高频恢复压力。

**(D) SWD 分流正则** $\mathcal{L}_{\text{swd\_reg}}$（新增，防止分流退化）：

$$\mathcal{L}_{\text{swd\_reg}} = \lambda_8 \cdot \left[(\bar{\alpha} - 0.5)^2 + (\bar{g_n} - 0.5)^2\right]$$

其中 $\bar{\alpha}$、$\bar{g_n}$ 是 SWD 分流权重的空间平均。这个损失**阻止 SWD 因为 TFDE 侧梯度较弱而退化到全部导向 TCA**——保持 $\alpha, g_n$ 靠近 0.5（均衡分流）作为先验。

#### 2.3.2 Phase 1 权重设置（Kendall UW 初始化）

延续 Mark2 的 Kendall 不确定性加权，Phase 1 的 `log_var` 初始化：

```python
log_var_phase1 = {
    'pix':          0.0,   # weight ≈ 0.5
    'lit_up_sup':  -1.0,   # weight ≈ 1.35，重点监督光照图
    'illum_smooth': 1.0,   # weight ≈ 0.18，弱约束
    'align_warp':  -0.5,   # weight ≈ 0.82，重点监督对齐
    'diag_prior':   0.5,   # weight ≈ 0.30，中等约束
    'ssim':        -1.0,   # weight ≈ 1.35，结构重要
    'ifpn_sup':     0.0,   # weight ≈ 0.5
    'swd_reg':      2.0,   # weight ≈ 0.07，弱正则
    # ↓ Phase 1 屏蔽项（不计算，不参与）
    # 'perc':   —
    # 'freq':   —
    # 'inter':  —
}
```

---

### 2.4 Phase 1.5 过渡设计（5 epoch 平滑解锁）

**核心机制**：使用一个单调递增的 `unlock_ratio` $\in [0, 1]$ 逐步引入 NDPN/MCPN 的贡献：

$$\text{unlock\_ratio}(e) = \frac{e - 20}{5}, \quad e \in [20, 25]$$

**前向修改**（已在 2.2.1 中给出）：`f_noise_out *= unlock_ratio`, `f_motion_out *= unlock_ratio`。

**损失渐进**：Phase 1.5 期间逐步引入 Phase 2 的新损失项：

```python
def get_phase15_loss_weights(unlock_ratio):
    """Phase 1.5 损失渐进权重"""
    return {
        # Phase 1 损失(保持全权重)
        'pix':          1.0,
        'lit_up_sup':   1.0,
        'illum_smooth': 1.0,
        'align_warp':   1.0,
        'diag_prior':   1.0,
        'ssim':         1.0,
        'ifpn_sup':     1.0,
        'swd_reg':      max(0.0, 1.0 - 2*unlock_ratio),  # 逐步撤销
        
        # Phase 2 新损失(逐步引入)
        'perc':   unlock_ratio,           # 感知损失渐进引入
        'freq':   unlock_ratio,           # 频域损失渐进引入
        'inter':  unlock_ratio,           # 中间监督渐进引入
    }
```

**学习率降级**：

$$\text{lr}(e) = \text{lr}_{\text{base}} \cdot (1 - 0.5 \cdot \text{unlock\_ratio}(e))$$

Phase 1 结束时 lr=8e-4，Phase 1.5 期间线性降到 4e-4，Phase 2 起进入 cosine 退火。

---

### 2.5 Phase 2 详细设计

Phase 2 从 Epoch 25 开始，等价于当前 Mark2 的完整损失，但**继承 Phase 1 训练好的 ISPN/SGRF/TCA 权重**。

#### 2.5.1 损失函数

$$\mathcal{L}_{\text{phase2}} = \mathcal{L}_{\text{phase1}}^{\text{reduced}} + \mathcal{L}_{\text{noise}} + \mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{perc-decouple}}$$

**(A) 继承的 Phase 1 损失**（权重降低）：

- $\mathcal{L}_{\text{pix}}$ (PECharbonnier), $\mathcal{L}_{\text{ssim}}$：正常权重
- $\mathcal{L}_{\text{lit\_up\_sup}}$：权重降低到 0.3（因为 lit_up_map 已收敛）
- $\mathcal{L}_{\text{align\_warp}}$, $\mathcal{L}_{\text{diag\_prior}}$：权重降低到 0.2（TCA 已训练好）
- $\mathcal{L}_{\text{swd\_reg}}$：**完全移除**（让 SWD 自由调整分流）

**(B) 噪声分支损失** $\mathcal{L}_{\text{noise}}$（新增）：

$$\mathcal{L}_{\text{noise}} = \lambda_9 \cdot (1 - \text{SSIM}(I_1, X_t^{\text{gt}})) + \lambda_{10} \cdot \text{PECharbonnier}(I_1, X_t^{\text{gt}} \cdot \ell_t^{\text{est}})$$

其中：
- **第一项**：SGRF-S1 输出 $I_1$（去噪后中间结果）应与 GT 结构相似
- **第二项**：$I_1$ 应等于"未提亮的干净帧"——用 GT 乘以估计的光照 $\ell_t^{\text{est}} = 1 / \text{lit\_up\_map}$ 得到伪监督
- 这是**感知解耦到 SGRF-S1 阶段**（Mark2 提到的方向）

**(C) 运动分支损失** $\mathcal{L}_{\text{motion}}$（新增）：

$$\mathcal{L}_{\text{motion}} = \lambda_{11} \cdot \text{VGG}_{\text{relu3\_3}}(I_2, X_t^{\text{gt}} \cdot \ell_t^{\text{est}}) + \lambda_{12} \cdot \text{PECharbonnier}(I_2, X_t^{\text{gt}} \cdot \ell_t^{\text{est}})$$

- **第一项**：SGRF-S2 输出 $I_2$（去模糊后）应与"未提亮 GT"感知一致——**这是感知损失监督 MCPN 的自然位置**
- **第二项**：像素级监督

**(D) 最终感知损失（弱化版）**：

$$\mathcal{L}_{\text{perc-decouple}} = \lambda_{13} \cdot \text{VGG}_{\text{relu3\_3}}(\hat{X}_t, X_t^{\text{gt}})$$

**注意**：由于感知损失已经在 $\mathcal{L}_{\text{motion}}$ 中解耦到 S2，这里的最终感知损失可以**大幅降低权重**（$\lambda_{13} \approx 0.3 \lambda_{11}$），主要用于最终提亮阶段 S3 的一致性保证。

#### 2.5.2 Phase 2 Kendall UW 初始化

```python
log_var_phase2 = {
    # 从 Phase 1 继承(读取 checkpoint)
    'pix':          checkpoint,
    'lit_up_sup':   checkpoint + 1.0,   # 手动降权(σ² 增大)
    'illum_smooth': checkpoint,
    'align_warp':   checkpoint + 1.0,   # 手动降权
    'diag_prior':   checkpoint + 1.0,   # 手动降权
    'ssim':         checkpoint,
    'ifpn_sup':     checkpoint,
    
    # 新引入的损失(初始化)
    'freq':        -0.5,   # weight ≈ 0.82
    'noise_ssim':  -1.0,   # weight ≈ 1.35，突出 S1 结构监督
    'noise_pix':    0.0,   # weight ≈ 0.5
    'motion_perc': -1.5,   # weight ≈ 2.24，突出 S2 感知监督
    'motion_pix':   0.0,   # weight ≈ 0.5
    'perc':         1.0,   # weight ≈ 0.18，弱化最终感知
    'inter':        0.0,   # weight ≈ 0.5
}
```

---

### 2.6 完整训练调度表

| Epoch | 阶段 | 前向配置 | 主要损失 | 学习率 | 备注 |
|-------|------|---------|---------|--------|------|
| 0-4 | Phase 1 Warmup | NDPN/MCPN 截断 | $\mathcal{L}_{\text{pix}}$ + $\mathcal{L}_{\text{ssim}}$ + $\mathcal{L}_{\text{lit\_up}}$ | 8e-4 (linear warmup) | 简化损失热身 |
| 5-19 | Phase 1 Main | NDPN/MCPN 截断 | 全部 Phase 1 损失 | 8e-4 → 6e-4 (cosine) | 光照+结构+对齐学习 |
| 20-24 | Phase 1.5 Transition | NDPN/MCPN 逐步解锁 | Phase 1 + Phase 2 新损失（渐进） | 6e-4 → 4e-4 (linear) | 5 epoch 平滑过渡 |
| 25-44 | Phase 2 Main | 全网络 | 全部 Phase 2 损失 | 4e-4 → 1e-4 (cosine) | 三源联合优化 |
| 45-49 | Phase 2 Finetune | 全网络 | 全部 Phase 2 损失 | 1e-4 → 3e-5 | 精调 |

---

### 2.7 训练循环代码框架

在 `train.py` 中修改主循环：

```python
def get_phase(epoch):
    if epoch < 5:
        return 'phase1_warmup'
    elif epoch < 20:
        return 'phase1'
    elif epoch < 25:
        return 'phase1_5'
    else:
        return 'phase2'

def get_unlock_ratio(epoch):
    if epoch < 20:
        return 0.0
    elif epoch < 25:
        return (epoch - 20) / 5.0
    else:
        return 1.0

def get_lr(epoch, base_lr=8e-4):
    if epoch < 5:
        # Linear warmup
        return base_lr * (0.01 + 0.99 * epoch / 5)
    elif epoch < 20:
        # Cosine within Phase 1
        progress = (epoch - 5) / 15
        return base_lr * (0.75 + 0.25 * math.cos(math.pi * progress))
    elif epoch < 25:
        # Linear降级到 Phase 1.5
        return base_lr * 0.75 * (1 - (epoch - 20) / 5 * 0.33)
    elif epoch < 45:
        # Phase 2 cosine
        progress = (epoch - 25) / 20
        return base_lr * 0.5 * (0.25 + 0.75 * math.cos(math.pi * progress))
    else:
        # Phase 2 finetune
        return base_lr * 0.125 * (0.3 + 0.7 * (49 - epoch) / 4)


# 主训练循环
for epoch in range(50):
    phase = get_phase(epoch)
    unlock_ratio = get_unlock_ratio(epoch)
    lr = get_lr(epoch)
    
    # 更新学习率
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    
    # Phase 转换时的特殊处理
    if epoch == 20:
        # 进入 Phase 1.5：解冻 NDPN/MCPN 参数(如果之前 requires_grad=False)
        for p in model.ndpn.parameters(): p.requires_grad = True
        for p in model.mcpn.parameters(): p.requires_grad = True
        for p in model.cxg.parameters(): p.requires_grad = True
        print(f"[Phase Transition] Unlocking NDPN/MCPN/CXG at epoch {epoch}")
    
    if epoch == 25:
        # 进入 Phase 2：调整 Kendall UW 的 log_var
        with torch.no_grad():
            loss_fn.log_vars['lit_up_sup'] += 1.0  # 降权
            loss_fn.log_vars['align_warp'] += 1.0
            loss_fn.log_vars['diag_prior'] += 1.0
            # 新增损失的 log_var 已在 loss_fn.__init__ 中定义
        print(f"[Phase Transition] Entering Phase 2 at epoch {epoch}")
    
    # 训练一个 epoch
    for batch in train_loader:
        model._unlock_ratio = unlock_ratio  # 传入 forward
        output = model(batch['frames'], phase=phase)
        loss = loss_fn(output, batch['gt'], phase=phase, 
                       unlock_ratio=unlock_ratio)
        loss.backward()
        # ... grad accum + optimizer step ...
```

**关键实现细节**：

**① NDPN/MCPN 参数冻结** (可选，推荐)：
```python
# 训练开始时(Phase 1 之前)
for p in model.ndpn.parameters(): p.requires_grad = False
for p in model.mcpn.parameters(): p.requires_grad = False
for p in model.cxg.parameters(): p.requires_grad = False
```
好处：**Phase 1 不会因为 zero forward 意外产生梯度**（比如 batchnorm 的统计量更新），且节省内存和计算。

**② Optimizer 分组**：
```python
optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() 
                if not any(m in n for m in ['ndpn','mcpn','cxg'])], 
     'lr': 8e-4, 'name': 'main'},
    {'params': model.ndpn.parameters(), 'lr': 8e-4, 'name': 'ndpn'},
    {'params': model.mcpn.parameters(), 'lr': 8e-4, 'name': 'mcpn'},
    {'params': model.cxg.parameters(),  'lr': 8e-4, 'name': 'cxg'},
], weight_decay=1e-4)
```
好处：可以对 NDPN/MCPN 使用**独立学习率**，Phase 1.5 时它们可以用更大 lr 快速追赶。

---

## 三、损失函数完整代码框架

在 `losses/losses.py` 中新增：

```python
class TFSNetLossPhase(nn.Module):
    """支持两阶段训练的损失函数"""
    
    def __init__(self, ...):
        super().__init__()
        # 保留 Mark2 的全部组件
        self.pe_charb = PECharbonnierLoss(edge_weight=2.0)
        self.ssim_fn = SSIMLoss()
        self.perc_fn = VGGPerceptualLoss(layer='relu3_3')
        # ... 其他损失 ...
        
        # ─── Phase 1 专用参数 ───
        self.static_thresh = 0.02  # diag_prior 的静止阈值
        
        # ─── Kendall UW: 包含 Phase 1 & Phase 2 的所有键 ───
        self.log_vars = nn.ParameterDict({
            # Phase 1 & 2 共享
            'pix':          nn.Parameter(torch.tensor(0.0)),
            'ssim':         nn.Parameter(torch.tensor(-1.0)),
            'lit_up_sup':   nn.Parameter(torch.tensor(-1.0)),
            'illum_smooth': nn.Parameter(torch.tensor(1.0)),
            'align_warp':   nn.Parameter(torch.tensor(-0.5)),
            'diag_prior':   nn.Parameter(torch.tensor(0.5)),
            'ifpn_sup':     nn.Parameter(torch.tensor(0.0)),
            'swd_reg':      nn.Parameter(torch.tensor(2.0)),
            # Phase 2 专属
            'freq':         nn.Parameter(torch.tensor(-0.5)),
            'noise_ssim':   nn.Parameter(torch.tensor(-1.0)),
            'noise_pix':    nn.Parameter(torch.tensor(0.0)),
            'motion_perc':  nn.Parameter(torch.tensor(-1.5)),
            'motion_pix':   nn.Parameter(torch.tensor(0.0)),
            'perc':         nn.Parameter(torch.tensor(1.0)),
            'inter':        nn.Parameter(torch.tensor(0.0)),
        })
    
    def _uw(self, loss, key):
        log_var = self.log_vars[key]
        return 0.5 * torch.exp(-log_var) * loss + 0.5 * log_var
    
    def _compute_static_mask(self, gt, gt_neighbors):
        """检测帧间静止区域(用于 diag_prior 损失)"""
        # gt: (B, 3, H, W), gt_neighbors: list of (B, 3, H, W)
        masks = []
        for gt_nb in gt_neighbors:
            diff = torch.abs(gt - gt_nb).mean(dim=1, keepdim=True)
            mask = (diff < self.static_thresh).float()
            masks.append(mask)
        return masks  # list of (B, 1, H, W)
    
    def _diag_prior_loss(self, C_omega, static_masks):
        """对角线先验损失"""
        # C_omega: (B, T-1, HW, HW), static_masks: list of (B, 1, H, W)
        loss = 0.0
        B, T_1, HW, _ = C_omega.shape
        for t_idx in range(T_1):
            C = C_omega[:, t_idx]  # (B, HW, HW)
            diag = torch.diagonal(C, dim1=-2, dim2=-1)  # (B, HW)
            
            mask = static_masks[t_idx].flatten(2).squeeze(1)  # (B, HW)
            # 只在静止区域施加对角线约束
            loss = loss - (mask * torch.log(diag + 1e-8)).sum() / (mask.sum() + 1e-6)
        return loss / T_1
    
    def _warp_consistency_loss(self, F_out_list, C_omega):
        """时序 warping 一致性损失"""
        # F_out_list: list of T 个 (B, C, H, W)
        # C_omega: (B, T-1, HW, HW)
        T = len(F_out_list)
        center = T // 2
        F_t = F_out_list[center]
        B, C, H, W = F_t.shape
        
        loss = 0.0
        neighbor_idx = [i for i in range(T) if i != center]
        for k, t_prime in enumerate(neighbor_idx):
            F_tp = F_out_list[t_prime].flatten(2)  # (B, C, HW)
            C_ttp = C_omega[:, k]                  # (B, HW, HW)
            # Warp: F_tp @ C_ttp^T → (B, C, HW)
            F_warped = torch.bmm(F_tp, C_ttp.transpose(-1, -2))
            F_warped = F_warped.reshape(B, C, H, W)
            loss = loss + F.l1_loss(F_warped, F_t)
        return loss / len(neighbor_idx)
    
    def _swd_reg_loss(self, model):
        """SWD 分流均衡正则(需要访问 model.swd 的中间统计)"""
        # 假设 SWD forward 中保存了 self._alpha_mean, self._gn_mean
        alpha_mean = model.swd._alpha_mean
        gn_mean = model.swd._gn_mean
        return (alpha_mean - 0.5)**2 + (gn_mean - 0.5)**2
    
    def forward(self, output, gt, gt_neighbors=None, model=None,
                phase='phase2', unlock_ratio=1.0, epoch=0):
        """
        Args:
            output: dict from TFSNet.forward
            gt: (B, 3, H, W) center frame ground truth
            gt_neighbors: list of neighbor frame GTs (用于 static mask)
            model: TFSNet instance (用于访问 SWD 中间统计)
            phase: 'phase1_warmup' | 'phase1' | 'phase1_5' | 'phase2'
            unlock_ratio: Phase 1.5 的解锁比例
            epoch: 当前 epoch
        """
        res_t = output['res_t']
        losses = {}
        
        # ═══ Phase 1 Warmup: 仅像素+SSIM ═══
        if phase == 'phase1_warmup':
            losses['pix'] = self.pe_charb(res_t, gt)
            losses['ssim'] = 1.0 - self.ssim_fn(res_t, gt)
            # 轻度光照约束
            losses['illum_smooth'] = illum_smooth_tv(output['s_illum'], gt)
            
            total = (self._uw(losses['pix'], 'pix')
                   + self._uw(losses['ssim'], 'ssim')
                   + 0.01 * losses['illum_smooth'])
            return total, losses
        
        # ═══ Phase 1: 光照+结构+对齐 ═══
        if phase == 'phase1':
            # 光照相关
            losses['pix'] = self.pe_charb(res_t, gt)
            losses['ssim'] = 1.0 - self.ssim_fn(res_t, gt)
            losses['illum_smooth'] = illum_smooth_tv(output['s_illum'], gt)
            
            # lit_up_map 伪监督
            gt_mean = gt.mean(dim=[2,3], keepdim=True)
            input_mean = output.get('input_mean', gt_mean * 0.3)  # fallback
            lit_up_target = gt_mean / (input_mean + 1e-6)
            losses['lit_up_sup'] = F.l1_loss(
                output['lit_up_map'].mean(dim=[2,3], keepdim=True), 
                lit_up_target.clamp(1.0, 8.0)
            )
            
            # 对齐损失
            losses['align_warp'] = self._warp_consistency_loss(
                output['F_out_list'], output['C_omega'])
            
            if gt_neighbors is not None:
                static_masks = self._compute_static_mask(gt, gt_neighbors)
                losses['diag_prior'] = self._diag_prior_loss(
                    output['C_omega'], static_masks)
            else:
                losses['diag_prior'] = torch.tensor(0.0, device=gt.device)
            
            # IFPN 侧输出
            gt_down = F.avg_pool2d(gt, 2)
            losses['ifpn_sup'] = F.l1_loss(output['ifpn_side'], gt_down)
            
            # SWD 分流正则
            if model is not None:
                losses['swd_reg'] = self._swd_reg_loss(model)
            else:
                losses['swd_reg'] = torch.tensor(0.0, device=gt.device)
            
            # Kendall UW 组合
            total = (self._uw(losses['pix'],          'pix')
                   + self._uw(losses['ssim'],         'ssim')
                   + self._uw(losses['illum_smooth'], 'illum_smooth')
                   + self._uw(losses['lit_up_sup'],   'lit_up_sup')
                   + self._uw(losses['align_warp'],   'align_warp')
                   + self._uw(losses['diag_prior'],   'diag_prior')
                   + self._uw(losses['ifpn_sup'],     'ifpn_sup')
                   + self._uw(losses['swd_reg'],      'swd_reg'))
            return total, losses
        
        # ═══ Phase 1.5: 渐进解锁 ═══
        if phase == 'phase1_5':
            # 计算所有 Phase 1 损失(全权重)
            total_p1, losses = self.forward(
                output, gt, gt_neighbors, model, 
                phase='phase1', unlock_ratio=1.0, epoch=epoch)
            
            # 逐步引入 Phase 2 新损失
            losses['freq'] = freq_loss(res_t, gt)
            losses['perc'] = self.perc_fn(res_t, gt)
            
            # 中间监督(SGRF S1/S2)
            img_s1 = output['img_s1']
            img_s2 = output['img_s2']
            # 估计 pseudo unlit GT
            lit = output['lit_up_map'].detach()
            pseudo_gt_unlit = (gt / (lit * (1 + output['A_illu'].detach()))
                              ).clamp(0, 1)
            losses['noise_ssim'] = 1.0 - self.ssim_fn(img_s1, pseudo_gt_unlit)
            losses['motion_perc'] = self.perc_fn(img_s2, pseudo_gt_unlit)
            
            total_new = (self._uw(losses['freq'], 'freq')
                       + self._uw(losses['perc'], 'perc')
                       + self._uw(losses['noise_ssim'], 'noise_ssim')
                       + self._uw(losses['motion_perc'], 'motion_perc'))
            
            # SWD 正则逐步撤销
            swd_reg_factor = max(0.0, 1.0 - 2 * unlock_ratio)
            
            total = total_p1 + unlock_ratio * total_new
            return total, losses
        
        # ═══ Phase 2: 完整损失 ═══
        if phase == 'phase2':
            # 基础项(从 Phase 1 继承但降权，由 Kendall UW 的 log_var 已调整承担)
            losses['pix'] = self.pe_charb(res_t, gt)
            losses['ssim'] = 1.0 - self.ssim_fn(res_t, gt)
            losses['freq'] = freq_loss(res_t, gt)
            losses['perc'] = self.perc_fn(res_t, gt)
            losses['illum_smooth'] = illum_smooth_tv(output['s_illum'], gt)
            
            # SGRF 中间阶段损失
            img_s1 = output['img_s1']
            img_s2 = output['img_s2']
            lit = output['lit_up_map'].detach()
            pseudo_gt_unlit = (gt / (lit * (1 + output['A_illu'].detach())
                              + 1e-6)).clamp(0, 1)
            
            # S1 去噪监督
            losses['noise_ssim'] = 1.0 - self.ssim_fn(img_s1, pseudo_gt_unlit)
            losses['noise_pix']  = self.pe_charb(img_s1, pseudo_gt_unlit)
            
            # S2 去模糊监督
            losses['motion_perc'] = self.perc_fn(img_s2, pseudo_gt_unlit)
            losses['motion_pix']  = self.pe_charb(img_s2, pseudo_gt_unlit)
            
            # 中间乘法路径
            losses['inter'] = self.pe_charb(
                (img_s2 * output['lit_up_map']).clamp(0, 1), gt)
            
            # IFPN 侧输出
            gt_down = F.avg_pool2d(gt, 2)
            losses['ifpn_sup'] = F.l1_loss(output['ifpn_side'], gt_down)
            
            # 对齐损失(降权保持)
            losses['align_warp'] = self._warp_consistency_loss(
                output['F_out_list'], output['C_omega'])
            if gt_neighbors is not None:
                static_masks = self._compute_static_mask(gt, gt_neighbors)
                losses['diag_prior'] = self._diag_prior_loss(
                    output['C_omega'], static_masks)
            else:
                losses['diag_prior'] = torch.tensor(0.0, device=gt.device)
            
            # Kendall UW 组合(所有项)
            total = (
                self._uw(losses['pix'],          'pix')
              + self._uw(losses['ssim'],         'ssim')
              + self._uw(losses['freq'],         'freq')
              + self._uw(losses['perc'],         'perc')
              + self._uw(losses['illum_smooth'], 'illum_smooth')
              + self._uw(losses['noise_ssim'],   'noise_ssim')
              + self._uw(losses['noise_pix'],    'noise_pix')
              + self._uw(losses['motion_perc'],  'motion_perc')
              + self._uw(losses['motion_pix'],   'motion_pix')
              + self._uw(losses['inter'],        'inter')
              + self._uw(losses['ifpn_sup'],     'ifpn_sup')
              + self._uw(losses['align_warp'],   'align_warp')
              + self._uw(losses['diag_prior'],   'diag_prior')
            )
            return total, losses
```

---

## 四、方案效果预期与风险控制

### 4.1 预期效果

| 指标 | 当前 (Mark2) | Phase 1 结束 (ep20) | Phase 2 结束 (ep49) |
|------|------------|------------------|-------------------|
| PSNR | ~20.5 dB | ~20.0 dB (光照恢复主导) | **21.5-22.5 dB** |
| SSIM | ~0.70 | ~0.75 (结构显著提升) | **0.82-0.85** |
| L_perc 下降幅度 | 停滞 | Phase 1 不监督 | Phase 2 显著下降 |
| L_ssim 下降幅度 | 缓慢 | **快速下降** | 继续下降 |

**为什么能改进结构/感知损失**：
- Phase 1 移除了 NDPN/MCPN 的干扰，SSIM 损失的梯度可以**清晰地流向 ISPN 和 SGRF-S3**
- Phase 2 中感知损失被**解耦到 SGRF-S2 (motion_perc)**，专门监督去模糊阶段——这正是感知损失最有效的位置（去模糊本质是感知任务）
- diag_prior + warp_consistency 让 TCA 学到更好的对齐 → Phase 2 时 NDPN 的置信度 c 更可靠

### 4.2 风险控制清单

| 风险 | 缓解措施 |
|------|--------|
| Phase 转换时 loss 突增 | Phase 1.5 的 5 epoch 平滑过渡；Kendall UW 会自动调整权重 |
| NDPN/MCPN 在 Phase 2 冷启动困难 | Optimizer 分组给它们独立更高的 lr（可选） |
| SWD 分流退化 | Phase 1 的 `swd_reg` 正则 + Phase 1.5 逐步撤销 |
| TCA 在 Phase 1 学不到运动结构 | 新增 `diag_prior` 静止区域监督 + `warp_consistency` 一般性对齐监督 |
| 伪监督 `pseudo_gt_unlit` 不准 | 用 `.detach()` 阻断梯度回传到 lit_up_map；即使不准，只是噪声监督，不至于误导 |

### 4.3 消融实验建议

如果最终写论文，这个两阶段方案本身就是一个 **强 ablation 支撑**：

| 消融项 | 预期结果 |
|-------|---------|
| w/o Phase 1 (直接 Phase 2) | 复现当前 Mark2 的问题（L_perc/L_ssim 停滞） |
| w/o Phase 1.5 (直接跳转) | Phase 2 初期 loss 剧烈波动 |
| w/o `diag_prior` | TCA 对齐质量下降，NDPN 置信度不可靠 |
| w/o SGRF 中间监督（S1/S2） | 感知损失回到停滞状态 |

---

## 五、总结

**你的方案在正确方向上，但需要 5 个关键修正**：

1. ✅ **SWD 保持全通路**：不截断 TFDE→ISPN 侧，避免 SWD 分流退化
2. ✅ **CXG 在 Phase 1 旁路**：而非"训练但无信号"
3. ✅ **增加 TCA 辅助损失**（$\mathcal{L}_{\text{align\_warp}}$ + $\mathcal{L}_{\text{diag\_prior}}$）：确保 TCA 在 Phase 1 就学到有意义的时序对应
4. ✅ **Phase 1.5 过渡阶段**（5 epoch 平滑解锁）：避免 Phase 2 冷启动冲击
5. ✅ **Phase 2 引入 SGRF 中间监督**（S1 SSIM + S2 感知）：这是**根本解决感知损失停滞的关键**——把感知损失从"监督最终输出"转移到"监督去模糊阶段"

**结合 Mark2 已有的 Kendall UW + PE Loss，加上本方案的两阶段训练 + 感知解耦到 SGRF，才能从根本上解决当前的收敛问题。**

## 参考来源

1. [Bengio et al., Curriculum Learning, ICML 2009](https://doi.org/10.1145/1553374.1553380)
2. [D3Fusion: Progressive Three-Stage Training, Applied Sciences 2025](https://www.mdpi.com/2076-3417/15/16/8918)
3. [AFD-LLIE: Advanced Frequency Disentanglement Paradigm](https://doi.org/10.48550/arxiv.2409.01641)
4. [D²RNet: Decoupling Degradations, AAAI 2024](https://doi.org/10.1609/aaai.v38i4.28144)
5. [PDHAT: Perceptual Decoupling, TMM 2024](https://doi.org/10.1109/tmm.2024.3355634)
6. [Kendall et al., Multi-Task Learning Using Uncertainty, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)
7. [Blau & Michaeli, Perception-Distortion Tradeoff, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.pdf)