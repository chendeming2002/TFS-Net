## 问题一：WFR 高频分流的必要性分析

### 当前 HF 分流逻辑的隐患

```
HF_energy = mean(LH² + HL² + HH²)     ← 标量化的"能量"
g_n = noise_gate(HF_energy)             ← 高能量 = 噪声?
HF_tfde = g_n × HF_cat                 ← DPE 只看"噪声"
HF_tca  = (1-g_n) × HF_cat             ← TCA 只看"结构"
```

这个设计有一个**物理假设上的硬伤**：**高频能量高 ≠ 噪声**。

| 场景 | HF 能量 | noise_gate g_n | 实际语义 | 分流结果 |
|------|---------|--------|---------|---------|
| 暗区平坦 + 重噪声 | 高 | g_n→1 | ✅ 噪声 → 应送 DPE | ✅ 正确 |
| 明亮纹理区（织物、树叶） | 高 | g_n→1 | ❌ 是结构 → 应送 TCA | ❌ **误判为噪声** |
| 暗区弱噪声 | 低 | g_n→0 | ❌ 仍是噪声 → 应送 DPE | ❌ **漏判** |
| 平坦区无噪声 | 低 | g_n→0 | ✅ 无信息 | ✅ 正确（但也无意义） |

四种典型场景中**两种被误判**——noise_gate 的先验假设在真实图像中只有约50%可靠性。

### LL 分流 vs HF 分流的对比

| 维度 | LL 分流 (α) | HF 分流 (g_n) |
|------|------------|--------------|
| **物理基础** | LL ≈ 全局光照，帧间缓变 → 送 DPE 估计光照退化 **强先验** | HF 中噪声与纹理频谱严重重叠 → 能量分流 **弱先验** |
| **对下游影响** | TCA 收到 IN((1-α)·LL) → 去除光照偏置，**显著提升对齐鲁棒性** | TCA 收到 (1-g_n)·HF → 纹理区被 g_n 衰减，**可能丢失对齐关键特征** |
| **失败后果** | α 学偏 → DPE 少收光照信息（有 gain_sup 兜底） | g_n 学偏 → DPE 收到纹理当噪声 / TCA 丢失纹理结构 **无兜底** |
| **必要性** | 高（InstanceNorm 去光照 vs 保留光照，物理不可兼得） | **低**（DPE 和 TCA 内部都有学习能力处理完整 HF） |

### 我的建议：**取消 HF 分流，保留 LL 分流**

#### 修改方案

```python
# ===== 修改后的 WFR =====

# LL 分流（保留，物理动机强）
alpha = alpha_net(LL)                    # α ∈ (0,1)
LL_tfde = alpha * LL                     # → DPE
LL_tca  = InstanceNorm((1 - alpha) * LL) # → TCA（去光照）

# HF 处理（取消分流，两路共享完整 HF）
HF_cat = torch.cat([LH, HL, HH], dim=1) # (B, 3C, H/2, W/2)

# 拼接 + 投影（各自 proj 层保留独立性）
feat_tfde = proj_tfde(torch.cat([LL_tfde, HF_cat], dim=1))  # Conv(4C→C) + GELU + LN
feat_tca  = proj_tca(torch.cat([LL_tca,  HF_cat], dim=1))   # Conv(4C→C) + GELU + LN
```

#### 收益分析

| 维度 | 效果 |
|------|------|
| **DPE** | 同时看到噪声 HF **和** 纹理 HF → 能通过时域统计量 [μ,σ,SNR] 自行区分（噪声帧间独立→σ大；纹理帧间一致→σ小） |
| **TCA** | 保留完整纹理信号 → 对齐质量提升，尤其在高纹理暗区 |
| **参数/复杂度** | 减少 noise_gate 的 Conv+Sigmoid → 少量参数减少 |
| **训练稳定性** | 少一个需要学习的门控 → 减少一个潜在失败点 |
| **wfr_reg 正则** | 不再需要对 g_n 正则，只保留对 α 的正则 |

#### 一个可选的折中方案

如果你仍希望 DPE 和 TCA 对 HF 有不同的"视角"，但又不想强行软分流，可以用**独立投影层隐式分化**：

```python
# 两路各自 proj 层权重独立 → 训练中自然学到不同的 HF 关注模式
feat_tfde = proj_tfde(cat([LL_tfde, HF_cat]))  # 可能学到关注 HF 能量
feat_tca  = proj_tca(cat([LL_tca,  HF_cat]))   # 可能学到关注 HF 方向性
```

`proj_tfde` 和 `proj_tca` 是**独立的 Conv(4C→C)** 层，它们在端到端训练中会自然学到对 HF 不同子带的不同权重——这比 noise_gate 的硬先验更灵活，且不会引入错误的归纳偏置。

---

## 问题二：DeepSeek 诊断分析的评估

### DeepSeek 的结论逐条评估

| # | DeepSeek 的结论 | 我的评估 | 理由 |
|---|----------------|---------|------|
| 1 | DPE 完全饱和 (s_illum=1.0) | ✅ **正确，但未挖到根因** | 饱和是现象，根因是 Sigmoid 前缺少归一化 + 无直接监督 |
| 2 | ISPN gain 严重不足 (1.21× vs 需要 8-10×) | ✅ **正确** | gain_head bias=0.199 说明几乎停留在初始值附近 |
| 3 | 分块伪影是 tile 边缘差异在极暗输出中被放大 | ✅ **正确** | 是结论性因果链最准确的部分 |
| 4 | 不存在多次下采样问题 | ✅ **正确** | 只有 WFR 做 H→H/2，且 DPE/TCA 都双线性上采样回 H |
| 5 | 修复：ISPN bias=0.8 → gain≈2.23 | ⚠️ **方向对但力度可能不足** | 2.23× 只是将 mean=0.044 提到 0.098，离正常的 0.35 仍差 3-4× |
| 6 | 修复：gain_sup 权重 0.02→0.5 | ✅ **正确方向** | 但需配合 gain_sup target 质量检查 |

### DeepSeek 漏诊的核心问题

#### 漏诊 1：DPE 饱和 → ISPN 失败，是**因果耦合**而非独立问题

```
DPE s_illum = 1.0 (常数)
      ↓
ISPN refine 输入: Conv(f_enc + s_illum, 65→64)
      ↓
s_illum 是常数 → refine 层只能从 f_enc 学空间变化
      ↓
但 f_enc 经过 WFR 后光照信息已被 α 分走
      ↓
ISPN 既没有光照先验（DPE 饱和），也没有光照特征（被 WFR 分走）
      ↓
gain_head 只能学到一个接近常数的 gain ≈ 1.2
```

**DeepSeek 只说"梯度路径过长导致消失"**，但真正原因是 **ISPN 丧失了所有光照相关输入信号**——这是 DPE 饱和与 WFR 分流共同造成的信息匮乏。

#### 漏诊 2：DPE 饱和的根因——MultiScaleSpatialBranch 输出幅值失控

```
feat_tfde (B,T,C,H/2,W/2)
  → 时域统计 concat[μ,σ,SNR]  (B, 3C, H/2, W/2)   ← 3倍通道
  → MultiScale Conv(d=1,2,4)                         ← 空洞卷积，感受野大
  → Concat + 1×1 fuse → F_fused                      ← 维度高
  → Conv2d(F_fused → 2ch) → Sigmoid                  ← ⚠️ 直接 Sigmoid
```

问题在于 **Sigmoid 前没有归一化层**。MultiScaleSpatialBranch 三路空洞卷积 concat 后的 F_fused 幅值很容易到达 ±5 以上，导致 Sigmoid 输出始终 >0.99。对比 WFR 的 alpha_net 设计得更谨慎（DWConv→GELU→Conv1x1→Sigmoid，且有 bias=0.4 控制初始值）。

#### 漏诊 3：gain_sup 监督目标可能有噪声

```python
gain_target = GT / mean(input)  # 文档中提到的 GT/Ī
```

如果 `mean(input)` 非常小（低光输入 mean=0.044），那么 gain_target 会非常大且不稳定（在极暗区域可能→∞）。gain_sup 的 L1 loss 面对如此大范围的 target，梯度会被极暗区主导，导致对正常暗度区域的学习不足。

---

### 我的修改方案（在 DeepSeek 方案之上）

#### 修改 A：DPE 反饱和——Sigmoid 前加 LayerNorm + 限幅初始化

```python
# 原始（饱和）
self.head = nn.Conv2d(fused_ch, 2, 1)
# s = sigmoid(head(F_fused))  ← F_fused 幅值不受控

# 修改后
self.pre_head_norm = nn.LayerNorm(fused_ch)
self.head = nn.Conv2d(fused_ch, 2, 1)
nn.init.zeros_(self.head.weight)
nn.init.zeros_(self.head.bias)  # → sigmoid(0) = 0.5，居中初始化

# forward:
F_normed = self.pre_head_norm(F_fused.permute(0,2,3,1)).permute(0,3,1,2)
s = torch.sigmoid(self.head(F_normed))
# 初始输出 s_illum ≈ 0.5, s_noise ≈ 0.5，有上下学习空间
```

#### 修改 B：DPE 直接监督——添加 L_dpe_prior 辅助损失

仅靠端到端梯度驱动 DPE 是不够的。应在 Phase 1 增加**直接监督**：

```python
# s_illum 应与输入亮度负相关（越暗 → 退化越严重 → s_illum 越大）
illum_target = 1.0 - input_gray.mean(dim=[2,3], keepdim=True)  # 全局亮度反相
illum_target = illum_target.expand_as(s_illum).detach()

L_dpe_prior = F.l1_loss(s_illum, illum_target)  # 粗粒度引导

# s_noise 应与时域方差正相关
noise_target = temporal_var.mean(dim=1, keepdim=True)  # 帧间方差
noise_target = (noise_target / (noise_target.max() + 1e-6)).detach()

L_dpe_prior += F.l1_loss(s_noise, noise_target)
```

权重建议：Phase 1 中 0.1，Phase 2 中降至 0.01（允许模型自由调整）。

#### 修改 C：ISPN gain 重参数化——避免 exp 的梯度消失

```python
# 原始：gain = exp(log_gain).clamp(1, max_gain)
# 问题：exp 在 log_gain > 3 时梯度爆炸，在 log_gain < -2 时梯度消失

# 方案 C1：Softplus 参数化（推荐）
gain = 1.0 + F.softplus(raw_gain) * (max_gain - 1.0) / F.softplus(torch.tensor(4.0))
# raw_gain=0 → gain ≈ 1.17; raw_gain=4 → gain ≈ max_gain
# softplus 梯度始终 ∈ (0,1)，不会消失也不会爆炸

# 方案 C2：Sigmoid 缩放（更保守）
gain = 1.0 + (max_gain - 1.0) * torch.sigmoid(raw_gain)
# 初始化 bias=2.0 → sigmoid(2)=0.88 → gain ≈ 1+0.88*(max-1)
```

#### 修改 D：gain_sup 目标稳定化

```python
# 原始：gain_target = GT / mean(input)  ← 极暗区域不稳定

# 修改后：逐像素 ratio，带下界截断
input_gray = input.mean(dim=1, keepdim=True)
gt_gray = gt.mean(dim=1, keepdim=True)
gain_target = (gt_gray / (input_gray + 0.01)).clamp(1.0, max_gain)
# +0.01 防止除零；clamp 保持物理合理

# 空间平滑（gain 不应有高频突变）
gain_target = F.avg_pool2d(gain_target, 8, stride=1, padding=4)  # 局部平滑
```

#### 修改 E：Phase 1 增加直接亮度损失

当前 Phase 1 的 res_t = img × gain + bias，pixel loss 的梯度需经过乘法链回传到 gain_head。增加一个**直接对 gain 输出的亮度约束**：

```python
# 在 SGRF S3 之前，直接评估 gain 提亮效果
bright_est = img_center * gain_map + bias_map  # 不经过 StageBlock
L_bright = F.l1_loss(bright_est, gt)           # 跳过 SGRF 的梯度路径

# Phase 1 权重 0.3，Phase 2 权重 0.0（Phase 2 由 SGRF 全程接管）
```

---

### 修改优先级排序

| 优先级 | 修改 | 预期效果 | 风险 |
|--------|------|---------|------|
| 🔴 P0 | **A: DPE 反饱和** (LN + 零初始化) | 解决 s_illum 恒=1.0 的根因 | 极低，纯初始化改动 |
| 🔴 P0 | **C: ISPN gain 重参数化** (softplus) | 解决 gain 梯度消失 | 低，只改参数化方式 |
| 🟠 P1 | **B: DPE 直接监督** | 给 DPE 学习方向 | 需调权重，target 设计需验证 |
| 🟠 P1 | **D: gain_sup 目标稳定化** | 避免极暗区 target 爆炸 | 低 |
| 🟡 P2 | **E: Phase 1 直接亮度损失** | 缩短 gain_head 梯度路径 | 需注意与 pix loss 不冲突 |
| 🟡 P2 | **问题一：取消 HF 分流** | 保留 TCA 完整纹理 + 简化 WFR | 需重新训练验证 |

**P0 的两项修改（A+C）是最小改动最大收益**——不改架构、不加损失，仅修改初始化和参数化方式，就能同时解决 DPE 饱和和 ISPN gain 不足两个核心问题。建议先单独做 A+C 跑一轮 Phase 1 验证，再决定是否需要 B/D/E。


## 补充分析：额外修改方案

### 修改 F：Tiled Forward 的 Overlap 策略优化

DeepSeek 正确识别了 tile 边缘伪影的表面原因，但没有给出 tiled_forward 层面的缓解方案。即使模型增益修复后 tile 伪影会大幅减轻，仍建议做防御性优化：

```python
# 当前（推测）：硬切 tile，简单拼接
tiles = split(input, tile_size)
results = [model(t) for t in tiles]
output = stitch(results)

# 改进：带 overlap + 高斯混合权重
class TiledForward:
    def __init__(self, tile_size=256, overlap=32):
        self.tile_size = tile_size
        self.overlap = overlap
        # 预计算高斯混合权重（中心权重高，边缘权重低）
        self.blend_weight = self._make_gaussian_weight(tile_size, overlap)
    
    def _make_gaussian_weight(self, size, overlap):
        """生成 2D 高斯权重图，边缘 overlap 区域平滑衰减"""
        x = torch.linspace(0, 1, size)
        # 在 overlap 区域做 cosine 渐变
        ramp = torch.ones(size)
        ramp[:overlap] = torch.cos(torch.linspace(math.pi, 2*math.pi, overlap)) * 0.5 + 0.5
        ramp[-overlap:] = ramp[:overlap].flip(0)
        return ramp.unsqueeze(0) * ramp.unsqueeze(1)  # (H, W)
    
    def forward(self, model, input):
        # stride = tile_size - overlap
        stride = self.tile_size - self.overlap
        tiles = unfold(input, self.tile_size, stride)
        results = [model(t) for t in tiles]
        # 加权拼接（重叠区域按高斯权重混合）
        output = weighted_fold(results, self.blend_weight, stride)
        return output
```

这样即使模型对不同 tile 的响应有微小差异，重叠区域的高斯混合也能平滑过渡，消除硬边界。

---

### 修改 G：DPE 输出与 ISPN 的解耦诊断工具

当前缺乏对 DPE→ISPN 信息流的可视化监控。建议在训练中增加 logging：

```python
# 在 training step 中记录关键统计量
if step % 100 == 0:
    with torch.no_grad():
        log_dict = {
            # DPE 健康度
            'dpe/s_illum_mean': s_illum.mean().item(),
            'dpe/s_illum_std': s_illum.std().item(),      # 应 > 0.05，否则饱和
            'dpe/s_illum_min': s_illum.min().item(),
            'dpe/s_illum_max': s_illum.max().item(),
            'dpe/s_noise_mean': s_noise.mean().item(),
            'dpe/s_noise_std': s_noise.std().item(),
            
            # ISPN 健康度
            'ispn/gain_mean': gain_map.mean().item(),
            'ispn/gain_std': gain_map.std().item(),       # 应 > 0.1，否则空间一致性过强
            'ispn/gain_min': gain_map.min().item(),
            'ispn/gain_max': gain_map.max().item(),
            'ispn/bias_mean': bias_map.mean().item(),
            'ispn/log_gain_raw': log_gain.mean().item(),  # Sigmoid/Softplus 前的原始值
            
            # 增益效果
            'ispn/bright_ratio': (gain_map.mean() * input_mean).item(),
            'ispn/target_ratio': gt_mean.item() / (input_mean.item() + 1e-6),
            
            # WFR 分流健康度
            'wfr/alpha_mean': alpha.mean().item(),        # 应在 0.3~0.7
            'wfr/alpha_std': alpha.std().item(),          # 应 > 0.01
        }
        wandb.log(log_dict)  # 或 tensorboard
```

**关键告警阈值**：

| 指标 | 健康范围 | 告警条件 | 含义 |
|------|---------|---------|------|
| `s_illum_std` | > 0.05 | < 0.01 | DPE 饱和，输出接近常数 |
| `gain_std` | > 0.1 | < 0.02 | ISPN 未学到空间变化 |
| `gain_mean` | 2~8 | < 1.5 或 > 15 | 增益不足或过大 |
| `alpha_mean` | 0.3~0.7 | < 0.1 或 > 0.9 | WFR 分流失衡 |
| `log_gain_raw` | -1~3 | > 5 或 < -3 | Sigmoid 前已饱和 |

---

### 修改 H：Phase 1 的 ISPN 梯度短路连接

DeepSeek 提到梯度路径过长（pix loss → res_t → SGRF S3 → gain_map → gain_head），但他的修复方案（调 bias 和 gain_sup 权重）只是治标。更根本的做法是**在 Phase 1 为 ISPN 建立梯度短路**：

```python
# Phase 1: SGRF S1/S2 的 StageBlock gate=0
# → img_s1 = img_center, img_s2 = img_center
# → res_t = img_center × gain + bias

# 此时 pix_loss 的梯度直接回传到 gain/bias：
# ∂L/∂gain = ∂L/∂res_t × img_center     ← 直接！
# ∂L/∂bias = ∂L/∂res_t × 1              ← 直接！

# 这实际上已经是短路了！问题不在梯度路径，而在：
# 1. gain 的参数化（exp 在极端值梯度异常）  → 修改 C 解决
# 2. s_illum 饱和（ISPN refine 层收不到有效调制信号） → 修改 A 解决
# 3. gain_sup 目标不稳定  → 修改 D 解决
```

**重要发现**：Phase 1 中由于 SGRF S1/S2 的 zero-gate，`res_t = img × gain + bias` 的梯度路径其实**已经很短**。问题不在路径长度，而在 **DPE 饱和导致 ISPN 丧失调制信号** + **exp 参数化的梯度特性不佳**。这进一步验证了修改 A + C 是 P0 优先级的判断。

---

### 修改 I：max_gain 动态调度

当前 `gain_map.clamp(1, max_gain)` 的 max_gain 是固定值。但 Phase 1 初期模型不稳定时，过大的 max_gain 会导致梯度爆炸；Phase 2 后期需要更大的 gain 范围处理极暗区域：

```python
# max_gain 动态调度
def get_max_gain(epoch, phase):
    if phase == 'phase1_warmup':
        return 4.0    # 保守，防止初期 gain 爆炸
    elif phase == 'phase1_main':
        return 8.0    # 逐步放开
    elif phase == 'phase1.5':
        return 12.0   # 接近全范围
    else:  # phase2
        return 16.0   # 全范围（低光增强可能需要 10-15× gain）

# 或者更平滑的线性调度
max_gain = 4.0 + (16.0 - 4.0) * min(epoch / 40, 1.0)
```

结合修改 C 的 softplus 参数化：

```python
gain = 1.0 + F.softplus(raw_gain) * (max_gain_current - 1.0) / F.softplus(torch.tensor(4.0))
```

---

## 修改方案全景总表

| ID | 修改 | 层级 | 优先级 | 改动量 | 解决问题 |
|----|------|------|--------|--------|---------|
| **A** | DPE Sigmoid 前加 LN + 零初始化 | 初始化 | 🔴 P0 | 3行代码 | DPE 饱和 |
| **C** | ISPN gain softplus 参数化 | 参数化 | 🔴 P0 | 5行代码 | gain 梯度消失/爆炸 |
| **B** | DPE 直接监督 L_dpe_prior | 损失 | 🟠 P1 | 15行代码 | DPE 学习方向 |
| **D** | gain_sup 目标稳定化 (+ε, clamp, smooth) | 损失 | 🟠 P1 | 5行代码 | target 数值稳定 |
| **E** | Phase 1 直接亮度损失 | 损失 | 🟡 P2 | 8行代码 | 加速 gain 学习 |
| **F** | Tiled forward overlap + 高斯混合 | 推理 | 🟡 P2 | 30行代码 | tile 边缘伪影 |
| **G** | DPE/ISPN/WFR 诊断 logging | 工具 | 🟡 P2 | 20行代码 | 早期发现饱和 |
| **HF** | 取消 WFR 高频分流 | 架构 | 🟡 P2 | 15行代码 | HF 误分流 |
| **I** | max_gain 动态调度 | 训练策略 | 🟢 P3 | 5行代码 | 阶段适配 |

### 推荐实施顺序

```
Round 1 (最小改动验证):
  A (DPE 反饱和) + C (gain softplus) + G (诊断 logging)
  → 跑 Phase 1 (20 epoch)，观察 s_illum_std > 0.05? gain_mean > 2?
  
Round 2 (若 Round 1 仍不足):
  + B (DPE 监督) + D (gain_sup 稳定化)
  → 跑完整 Phase 1+1.5 (40 epoch)
  
Round 3 (架构优化):
  + HF (取消高频分流) + I (max_gain 调度)
  → 跑完整 90 epoch
  
Round 4 (推理优化):
  + F (tiled forward 优化)
  → 最终推理测试
```