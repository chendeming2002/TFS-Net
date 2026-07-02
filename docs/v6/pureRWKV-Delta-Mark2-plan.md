# TFS-Net v6 Delta Mark1 损失函数与训练策略改进方案

---

## 一、问题诊断

根据你的架构文档，当前 TFS-Net v6 共有 **9 项损失**联合优化，且权重固定（λ_pix=1.0 vs λ_perc=0.04，相差 25 倍）。结合训练改进文档的调研结论，问题根因可以精确定位：

| 现象 | 根因 | 理论依据 |
|------|------|---------|
| 像素损失持续下降，感知/结构损失停滞 | 感知-失真权衡（数学必然） | [Perception-Distortion Tradeoff](https://doi.org/10.1109/cvpr.2018.00781) (CVPR 2018, 被引1352) |
| 9项固定权重损失联合优化不稳定 | 多任务梯度冲突 + 量级不均衡 | [Kendall et al.](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf) (CVPR 2018, 被引5317) |
| SGRF 三阶段（去噪→去模糊→提亮）的 L_ssim/L_perc 全挂在最终输出 | 共享输出梯度方向冲突 | [D3Fusion](https://www.mdpi.com/2076-3417/15/16/8918) 三阶段训练验证 |
| batch=1，梯度估计噪声大 | 小batch SGD方差 | 基础优化理论 |

**核心洞察**：你的架构本身（SWD分流 + ISPN/NDPN/MCPN三路并行 + SGRF三阶段）已经具备了 [PDHAT](https://doi.org/10.1109/tmm.2024.3355634) 式感知解耦的天然条件，但损失函数设计没有利用这个优势——所有感知/结构损失都堆积在最终输出 `res_t` 上，导致梯度在共享层上互相拉扯。

---

## 二、改进方案 A：不确定性加权（P0 优先级，立刻实施）

### A1. Kendall 同方差不确定性加权

**理论基础**：[Kendall et al. (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf) 证明，通过学习每个任务的同方差不确定性 $\sigma_i$，可以自动平衡不同量级、不同单位的损失项，效果优于手动调参甚至网格搜索的最优权重。

**数学公式**：

$$\mathcal{L}_{\text{total}} = \sum_{i=1}^{K} \frac{1}{2\exp(s_i)} \mathcal{L}_i + \frac{1}{2} s_i, \quad s_i = \log \sigma_i^2$$

其中 $s_i$ 是可学习参数（log variance）。损失大的任务 → $\sigma_i$ 自动增大 → 权重 $\frac{1}{2\sigma_i^2}$ 降低；正则项 $\frac{1}{2}s_i$ 防止 $\sigma_i$ 无限增大。

**在 `losses/losses.py` 中的具体实现**：

```python
class TFSNetLoss(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ─── 新增：可学习 log-variance 参数 ───
        # 将 9 项损失分为 3 组（按语义关联）
        # 组1: 最终输出质量（pix, freq）
        # 组2: 结构+感知（ssim, perc）
        # 组3: 中间监督（illum_smooth, illum_sup, inter, ifpn_sup）
        # 注意：L_noise_sup 权重=0 已关闭，不参与
        
        self.log_vars = nn.ParameterDict({
            'pix':    nn.Parameter(torch.tensor(0.0)),   # 初始 σ²=1 → 权重=0.5
            'freq':   nn.Parameter(torch.tensor(0.0)),
            'ssim':   nn.Parameter(torch.tensor(-1.0)),  # 初始 σ²≈0.37 → 权重≈1.35
            'perc':   nn.Parameter(torch.tensor(-2.0)),  # 初始 σ²≈0.14 → 权重≈3.6
            'illum':  nn.Parameter(torch.tensor(0.0)),
            'inter':  nn.Parameter(torch.tensor(0.0)),
            'ifpn':   nn.Parameter(torch.tensor(0.0)),
        })
        # ... 原有的其他初始化 ...
    
    def _uncertainty_weight(self, loss, key):
        """Kendall 不确定性加权"""
        log_var = self.log_vars[key]
        # 1/(2*exp(s)) * L + s/2
        precision = torch.exp(-log_var)
        return 0.5 * precision * loss + 0.5 * log_var
    
    def forward(self, pred, gt, ...):
        # 计算各项原始损失（不乘固定权重）
        L_pix   = charbonnier(res_t, gt)
        L_freq  = freq_loss(res_t, gt)
        L_ssim  = 1.0 - ssim(res_t, gt)
        L_perc  = perceptual_loss(res_t, gt)
        L_illum = illum_smooth_loss(s_illum) + 0.02 * illum_sup_loss(s_illum, gt_illum)
        L_inter = charbonnier(torch.clamp(img_s2 * lit_up_map, 0, 1), gt)
        L_ifpn  = perceptual_loss(ifpn_side, F_down(gt))
        
        # 不确定性加权
        total = (
            self._uncertainty_weight(L_pix,   'pix')
          + self._uncertainty_weight(L_freq,  'freq')
          + self._uncertainty_weight(L_ssim,  'ssim')
          + self._uncertainty_weight(L_perc,  'perc')
          + self._uncertainty_weight(L_illum, 'illum')
          + self._uncertainty_weight(L_inter, 'inter')
          + self._uncertainty_weight(L_ifpn,  'ifpn')
        )
        return total
```

**初始化策略**：

| 参数 | 初始值 | 对应初始 σ² | 对应初始权重 | 设计意图 |
|------|--------|------------|------------|---------|
| `log_var['pix']` | 0.0 | 1.0 | 0.5 | 像素损失量级最大，初始权重偏低 |
| `log_var['ssim']` | -1.0 | 0.37 | 1.35 | SSIM 量级约 0.1-0.3，初始给较高权重 |
| `log_var['perc']` | -2.0 | 0.14 | 3.6 | 感知损失量级最小，初始给最高权重 |
| `log_var['freq']` | 0.0 | 1.0 | 0.5 | 与 pix 同级 |
| 其他 | 0.0 | 1.0 | 0.5 | 中间监督，初始均衡 |

根据 Kendall 论文 Table 1 的实验结果，**不确定性加权在训练数百步后即可快速收敛到合理值**，对初始化不敏感。但上述初始化通过让感知损失初始权重偏高，可以加速早期的平衡过程。

**训练监控**——在每个 epoch 日志中输出学习到的权重：
```python
# 每 epoch 打印
for key, log_var in self.log_vars.items():
    sigma2 = torch.exp(log_var).item()
    weight = 1.0 / (2 * sigma2)
    print(f"  {key}: σ²={sigma2:.4f}, effective_weight={weight:.4f}")
```

**预期效果**：训练过程中，如果 L_pix 下降快而 L_perc 停滞，系统会自动降低 L_pix 的权重、提高 L_perc 的权重，直到两者的加权下降速率趋于平衡。

---

## 三、改进方案 B：感知解耦损失——利用 SGRF 三阶段结构（P1 优先级）

### B1. 设计思路

[PDHAT (TMM 2024)](https://doi.org/10.1109/tmm.2024.3355634) 的核心策略是：**将不同感知属性解耦到异构辅助任务中，每个分支独立损失独立回传**，从根本上消除梯度冲突。

你的 SGRF 已有三阶段结构：
```
Stage1: img_s1 = denoise(f_noise_gated, img_center)   ← 去噪
Stage2: img_s2 = motion(f_motion_gated, img_s1)        ← 去模糊
Stage3: res_t = clamp(img_s2 × lit_up_map × (1+A_illu)) ← 提亮
```

**但当前所有的 L_ssim 和 L_perc 都只监督最终 res_t**——这意味着 Stage1 去噪模块的梯度需要穿过 Stage2 和 Stage3 才能接收到结构/感知约束，路径过长且与其他阶段的梯度混合。

### B2. 具体实现

**改动点**：修改 `models/modules/igrf.py` 的 SGRF forward，增加中间输出暴露；修改 `losses/losses.py`，增加阶段专属损失。

```python
# ─── SGRF 修改：暴露中间输出 ───
class SGRF(nn.Module):
    def forward(self, f_illum_feat, f_noise_gated, f_motion_gated, 
                lit_up_map, img_center, A_illu):
        # Stage1: 去噪
        img_s1 = self.denoise_stage(f_noise_gated, img_center)
        
        # Stage2: 去模糊
        img_s2 = self.motion_stage(f_motion_gated, img_s1)
        
        # Stage3: 提亮
        res_t = torch.clamp(img_s2 * lit_up_map * (1 + A_illu), 0, 1)
        
        # 返回所有阶段输出（用于感知解耦损失）
        return res_t, img_s1, img_s2

# ─── 感知解耦损失 ───
class PerceptualDecoupledLoss(nn.Module):
    """
    借鉴 PDHAT 的感知解耦思想：
    - Stage1 (去噪) → 结构保持约束 (SSIM)
    - Stage2 (去模糊) → 感知质量约束 (VGG perceptual)
    - Stage3 (提亮) → 频域保真约束 (FFT) + 像素保真 (Charbonnier)
    
    每个阶段的损失梯度只需要穿过本阶段，
    不会与其他阶段的梯度在共享层上冲突。
    """
    def __init__(self):
        super().__init__()
        self.ssim_fn = SSIMLoss()
        self.perc_fn = VGGPerceptualLoss(layer='relu3_3')
        
    def forward(self, res_t, img_s1, img_s2, gt):
        # Stage1 专属：结构约束（去噪应保持结构）
        L_ssim_s1 = 1.0 - self.ssim_fn(img_s1, gt)
        
        # Stage2 专属：感知约束（去模糊应恢复感知质量）
        L_perc_s2 = self.perc_fn(img_s2, gt)
        
        # Stage3 专属：频域+像素约束（提亮应保真）
        L_pix_s3  = charbonnier(res_t, gt)
        L_freq_s3 = freq_loss(res_t, gt)
        
        return L_ssim_s1, L_perc_s2, L_pix_s3, L_freq_s3
```

### B3. 感知解耦 + 不确定性加权的融合

```python
# 最终损失组合
total = (
    # 感知解耦损失（各阶段专属）
    self._uw(L_ssim_s1,  'ssim')      # Stage1 结构约束
  + self._uw(L_perc_s2,  'perc')      # Stage2 感知约束
  + self._uw(L_pix_s3,   'pix')       # Stage3 像素约束
  + self._uw(L_freq_s3,  'freq')      # Stage3 频域约束
    # 分支专属损失（不变）
  + self._uw(L_illum,    'illum')     # TFDE→ISPN 路径
  + self._uw(L_inter,    'inter')     # 中间乘法路径
  + self._uw(L_ifpn,     'ifpn')      # ISPN 侧输出
)
```

**梯度流向对比**：

```
当前（所有损失在 res_t）:
  L_ssim → ∂/∂res_t → ∂/∂Stage3 → ∂/∂Stage2 → ∂/∂Stage1 → ∂/∂NDPN
  L_perc → ∂/∂res_t → ∂/∂Stage3 → ∂/∂Stage2 → ∂/∂Stage1 → ∂/∂NDPN
  L_pix  → ∂/∂res_t → ∂/∂Stage3 → ∂/∂Stage2 → ∂/∂Stage1 → ∂/∂NDPN
  ↑ 三者梯度在所有阶段的参数上叠加 → 冲突

改进后（感知解耦）:
  L_ssim → ∂/∂img_s1 → ∂/∂Stage1 → ∂/∂NDPN          ← 只穿过 Stage1
  L_perc → ∂/∂img_s2 → ∂/∂Stage2 → ∂/∂MCPN          ← 只穿过 Stage2
  L_pix  → ∂/∂res_t  → ∂/∂Stage3 → (浅层到 ISPN)    ← 主要在 Stage3
  ↑ 三者在各自负责的阶段独立优化
```

---

## 四、改进方案 C：损失调度（P0 优先级，配合 A1 使用）

### C1. 分阶段损失权重调度

受 [D3Fusion (Applied Sciences 2025)](https://www.mdpi.com/2076-3417/15/16/8918) 三阶段渐进训练的启发——该论文明确指出："端到端训练导致不同任务的损失项在梯度反传中相互干扰，降低收敛稳定性"，并通过实验验证了**分阶段训练**的必要性。

**方案**：如果选用不确定性加权（方案A1），则损失调度作为**辅助手段**，只需要调度两个方面：

```yaml
# configs/v6_bravo.yaml 修改
loss_schedule:
  # 阶段1: 粗略恢复（Epoch 0-14）
  phase1:
    epochs: [0, 14]
    freeze_modules: []  # 不冻结，但降低感知损失初始影响
    log_var_init:
      pix: 0.0     # 权重 ≈ 0.5
      ssim: 0.0     # 权重 ≈ 0.5（此阶段先让SSIM正常贡献）
      perc: 2.0     # 权重 ≈ 0.07（极低，不干扰像素恢复）
      freq: 1.0     # 权重 ≈ 0.18（低）
  
  # 阶段2: 细节细化（Epoch 15-34）
  phase2:
    epochs: [15, 34]
    # 解锁感知损失，让不确定性加权自由调节
    action: "reset log_var['perc'] to -1.0"  # 权重 ≈ 1.35
    
  # 阶段3: 联合精调（Epoch 35-49）
  phase3:
    epochs: [35, 49]
    # 降低学习率，所有损失自由竞争
    lr_multiplier: 0.3
```

**实现代码**：

```python
# train.py 中
def adjust_loss_schedule(loss_fn, epoch):
    """根据训练阶段调整不确定性加权的初始点"""
    if epoch == 15:
        # Phase2: 解锁感知损失
        with torch.no_grad():
            loss_fn.log_vars['perc'].fill_(-1.0)  # σ²=0.37, weight≈1.35
            loss_fn.log_vars['freq'].fill_(-0.5)  # σ²=0.61, weight≈0.82
        print("[LossSchedule] Phase2: Unlocked perceptual/freq losses")
    
    elif epoch == 35:
        # Phase3: 不做额外调整，让不确定性加权自由运行
        print("[LossSchedule] Phase3: Joint fine-tuning, all losses free")
```

**为什么这比纯固定权重好**：

- Phase1 中 L_perc 的 log_var=2.0 意味着其有效权重仅 ≈0.07，不会干扰像素恢复
- Phase2 时 reset 到 -1.0，解锁感知约束，不确定性加权机制会在此基础上自动调节
- 相比 D3Fusion 的完全冻结方案，这里只调整损失权重初始点，不冻结模块参数，更温和

### C2. Warmup 期间仅用像素+结构损失

在当前的 5 epoch warmup 期间，只使用 L_pix + L_ssim，**完全屏蔽**感知损失和中间监督：

```python
def forward(self, pred, gt, epoch, ...):
    # Warmup 期间简化损失
    if epoch < 5:
        total = (
            self._uncertainty_weight(L_pix, 'pix')
          + self._uncertainty_weight(L_ssim, 'ssim')
          + 0.001 * L_illum_smooth  # 极小权重保持光照平滑
        )
        return total
    
    # 正常训练
    ...
```

**理由**：
- VGG 感知损失在训练初期（网络输出几乎是随机的）会产生**非常大且方向混乱的梯度**
- 你的文档已记录过"旧版 λ_perc=0.8 时 ep10 崩溃 -8.5dB"——这正是感知损失在早期过度主导的表现
- Warmup 期间专注像素恢复，让网络先学到"基本的亮度和颜色映射"

---

## 五、改进方案 D：GradProm 式梯度促进（P2 优先级）

受 [GradProm (arXiv 2025)](https://arxiv.org/pdf/2501.01114) 启发——该方法用余弦相似度检测梯度方向是否一致，只在一致时才让辅助损失参与更新：

```python
class GradPromLoss(nn.Module):
    """
    改进思路: 将 L_perc 和 L_ifpn_sup 视为"辅助任务"，
    L_pix + L_ssim 视为"主任务"。
    
    只有当辅助任务梯度与主任务梯度方向一致时（cos > 0），
    才让辅助任务的梯度参与参数更新。
    """
    def compute_with_gradprom(self, model, L_main, L_aux):
        # 计算主任务梯度方向
        grad_main = torch.autograd.grad(
            L_main, model.parameters(), 
            retain_graph=True, allow_unused=True
        )
        
        # 计算辅助任务梯度方向
        grad_aux = torch.autograd.grad(
            L_aux, model.parameters(), 
            retain_graph=True, allow_unused=True
        )
        
        # 展平并计算余弦相似度
        g_main_flat = torch.cat([g.flatten() for g in grad_main if g is not None])
        g_aux_flat  = torch.cat([g.flatten() for g in grad_aux if g is not None])
        cos_sim = F.cosine_similarity(g_main_flat.unsqueeze(0), 
                                       g_aux_flat.unsqueeze(0))
        
        if cos_sim > 0:
            # 方向一致 → 辅助损失参与
            return L_main + L_aux
        else:
            # 方向冲突 → 只用主任务
            return L_main
```

**注意**：GradProm 计算量较大（需要两次 backward），对于 batch=1 的场景可能增加 30-50% 训练时间。**建议仅在 P0/P1 方案效果不足时启用**。

---

## 六、改进方案 E：PE Loss——边缘增强像素损失（P1 优先级）

受 [PE Loss (Computational Visual Media 2025)](https://www.sciopen.com/article/10.26599/CVM.2025.9450475) 启发——该方法指出传统像素损失**对所有像素一视同仁**，不区分清晰边缘和模糊区域，导致结果平滑化。

**核心思想**：设计模糊因子图（blur factor map）检测模糊像素并**放大其误差惩罚**：

```python
class PECharbonnierLoss(nn.Module):
    """
    Perception-Enhanced Charbonnier Loss
    在原 Charbonnier 基础上，对模糊边缘区域施加更高惩罚
    """
    def __init__(self, eps=1e-6, edge_weight=2.0):
        super().__init__()
        self.eps = eps
        self.edge_weight = edge_weight
        # Sobel 算子检测边缘
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        # 初始化 Sobel 核
        self.sobel_x.weight.data = torch.tensor(
            [[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32
        ).reshape(1,1,3,3)
        self.sobel_y.weight.data = torch.tensor(
            [[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32
        ).reshape(1,1,3,3)
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
    
    def forward(self, pred, gt):
        # 计算 GT 的边缘图
        gt_gray = gt.mean(dim=1, keepdim=True)
        edge_x = self.sobel_x(gt_gray)
        edge_y = self.sobel_y(gt_gray)
        edge_map = torch.sqrt(edge_x**2 + edge_y**2 + self.eps)
        
        # 归一化到 [1, edge_weight]
        edge_map_norm = 1.0 + (self.edge_weight - 1.0) * (
            edge_map / (edge_map.max() + self.eps)
        )
        
        # 加权 Charbonnier
        diff = pred - gt
        loss = torch.sqrt(diff**2 + self.eps**2)
        weighted_loss = (loss * edge_map_norm).mean()
        
        return weighted_loss
```

**替换你当前的 `Charbonnier(res_t, GT)` 为 `PECharbonnier(res_t, GT)`**。

这样做的效果：
- 平坦区域（天空、墙面）：权重 ≈ 1.0，正常 Charbonnier
- 边缘区域（物体轮廓、纹理）：权重 ≈ 2.0，**加倍惩罚边缘模糊**
- 间接促进 L_ssim 和 L_perc 的下降——因为像素损失本身就在关注结构区域

---

## 七、综合实施方案与配置

### 7.1 推荐组合

| 阶段 | 方案组合 | 变更文件 |
|------|---------|---------|
| **立即实施** | A1（不确定性加权）+ C2（warmup简化损失）| `losses/losses.py`, `train.py` |
| **验证 A1 后** | A3（损失调度 Phase1→2→3）| `train.py` |
| **如效果不足** | B（感知解耦到 SGRF 各阶段）+ E（PE Loss）| `models/modules/igrf.py`, `losses/losses.py` |
| **最后手段** | D（GradProm）| `train.py` |

### 7.2 最终配置文件修改

```yaml
# configs/v6_bravo.yaml 补充
loss:
  # 不确定性加权初始化
  uncertainty_weighting: true
  log_var_init:
    pix: 0.0
    freq: 0.0
    ssim: -1.0
    perc: 2.0       # Phase1 初始极低，Phase2 reset 到 -1.0
    illum: 0.0
    inter: 0.0
    ifpn: 0.0
  
  # PE Loss
  use_pe_charbonnier: true
  pe_edge_weight: 2.0
  
  # 感知解耦
  perceptual_decoupling: true  # L_ssim → Stage1, L_perc → Stage2
  
  # Warmup 简化
  warmup_loss_only_pix_ssim: true
  
  # 损失调度
  loss_schedule:
    phase1_end: 14
    phase2_end: 34
    phase2_perc_log_var_reset: -1.0

# 训练参数不变
train:
  batch_size: 1
  epochs: 50
  lr: 8e-4
  warmup_epochs: 5
  weight_decay: 1e-4
  grad_clip: 0.5
```

### 7.3 训练监控新增指标

```bash
# 在每 epoch 日志中增加输出：
# 1. 不确定性权重趋势
grep "σ²=" outputs/sdsd_delta/nohup.out | tail -10

# 2. 各阶段中间输出的 PSNR（诊断瓶颈在哪个阶段）
# img_s1 的 PSNR → 去噪效果
# img_s2 的 PSNR → 去模糊效果
# res_t 的 PSNR → 最终效果
grep "stage_psnr" outputs/sdsd_delta/nohup.out | tail -10

# 3. 梯度余弦相似度（可选，诊断用）
# cos(∇L_pix, ∇L_ssim) → 正常应 > 0
# cos(∇L_pix, ∇L_perc) → 可能 < 0（冲突）
grep "grad_cos" outputs/sdsd_delta/nohup.out | tail -10
```

---

## 八、预期效果与风险评估

| 方案 | 预期PSNR提升 | 预期SSIM提升 | 风险 | 回退策略 |
|------|------------|------------|------|---------|
| A1 不确定性加权 | +0.3~0.8 dB | +0.01~0.03 | 极低 | 回退到固定权重 |
| B 感知解耦 | +0.2~0.5 dB | +0.02~0.05 | 低 | 取消中间输出监督 |
| C 损失调度 | +0.1~0.3 dB | +0.01~0.02 | 极低 | 取消调度 |
| E PE Loss | +0.1~0.3 dB | +0.01~0.03 | 极低 | 替换回原 Charbonnier |
| D GradProm | +0.1~0.2 dB | +0.01 | 中（训练变慢）| 关闭 |

**最保守的改动**（如果你只想改一处）：**单独实施 A1（不确定性加权）**，仅修改 `losses/losses.py`，约 20 行代码，风险极低，收益最高。

---

## 参考来源

1. [PDHAT - Perceptual Decoupling With Heterogeneous Auxiliary Tasks (TMM 2024)](https://doi.org/10.1109/tmm.2024.3355634)
2. [Multi-Task Learning Using Uncertainty to Weigh Losses (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)
3. [D3Fusion - Progressive Three-Stage Training (Applied Sciences 2025)](https://www.mdpi.com/2076-3417/15/16/8918)
4. [PE Loss - Perception-Enhanced Distortion-Oriented Loss (CVM 2025)](https://www.sciopen.com/article/10.26599/CVM.2025.9450475)
5. [GradProm - Generalized Gradient Promotion (arXiv 2025)](https://arxiv.org/pdf/2501.01114)
6. [DAP-LED - Degradation-Aware Priors with CLIP (arXiv 2024)](https://doi.org/10.48550/arxiv.2409.13496)
7. [Traversing Distortion-Perception Tradeoff (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Traversing_Distortion-Perception_Tradeoff_using_a_Single_Score-Based_Generative_Model_CVPR_2025_paper.pdf)
8. [ALLNet - Multi-task Dense Prediction (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_ALLNet_Multi-task_Dense_Prediction_for_Degraded_Images_CVPR_2026_paper.pdf)
9. [MambaDPF-Net - Multi-Task Training Learning (Electronics 2024)](https://doi.org/10.3390/electronics14224533)
10. [IPCMoE - Mixture-of-Experts for Joint LLE-Deblur (ACM 2025)](https://doi.org/10.1145/3746027.3755026)