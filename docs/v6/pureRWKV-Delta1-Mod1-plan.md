# TFS-Net v6 Delta NaN问题深入分析与风险排查

## (1) 本次NaN根因深入分析

### 核心爆炸链路

DeepSeek V4的诊断完全准确。以下结合您的模型具体结构做深入展开：

```
Encoder输出 (B,T,64,H,W)
  → SACE Downsample (H/2 × W/2)
    → MVC-Shift (三分支空洞DWConv，不改变数值范围)
      → SpatialWKV2D
        → proj_k (nn.Linear, 无归一化)  ← 🔴 k值来源
          → exp(k)  ← 🔴 爆炸点
            → cumsum(exp(k) * v)  ← 🔴 累积上溢
              → NaN 传播到整个模型
```

### 为什么此模型比标准RWKV更容易爆炸

**序列长度极长**：您的SpatialWKV2D扫描的是空间维度H×W。假设输入256×256，降采样后128×128=16384个token。而标准Vision-RWKV的token数为196（14×14）~3136（56×56）。您的序列长度是Vision-RWKV典型值的**5~80倍**，cumsum累积步数成比例增加，指数爆炸风险呈指数级放大。

根据[Vision-RWKV论文](https://arxiv.org/abs/2403.02308)明确指出：

> "As input resolution increases, both exponential decay and growth can quickly exceed the range of floating-point numbers. To address this, we divide the exponential term by the number of tokens (e.g., exp(−(|t−i|−1)/T·w)), making the maximum decay and growth bounded."

您的BiWKV实现虽然有 `(-w.abs() / total_tokens).exp()` 对衰减项做了归一化，但**关键的k值并未按token数归一化**——这正是Vision-RWKV所要求的。

**通道数过小**：根据[Issue #309 (BlinkDL/RWKV-LM)](https://github.com/BlinkDL/RWKV-LM/issues/309)中的讨论：

> "it seems that the problem has something to do with the 'dims', I found that there would be Nan value only if the dims is less than 128."

您的模型通道数C=64（split后每head仅16维），正好落在NaN高发区。小维度下Linear投影矩阵的奇异值分布更窄，条件数更差，k值更容易漂移到极端。

**累积路径分析**：以cumsum实现为例：
```python
ew_pow = ew.pow(arange_L)  # (1, L, C/4), L=16384时
S_fwd = (ekv / ew_pow).cumsum(dim=1) * ew_pow
```
当L=16384且`exp(k)`中k>5时：`exp(5)^{16384步累加}` → **天文数字**，fp32直接溢出到inf→NaN。

### 修复措施的数值验证

参考[RWKV-v4neo CUDA kernel](https://github.com/BlinkDL/RWKV-LM/blob/acc4293f/RWKV-v4neo/cuda/wkv_cuda.cu)的log-space稳定化实现：

```c
// aa and bb are running sums divided by exp(pp) (to avoid overflow)
F aa = 0, bb = 0, pp = MIN_VALUE;
F ww = u + kk;
F p = max(pp, ww);
F e1 = exp(pp - p);
F e2 = exp(ww - p);
y[ii] = (e1 * aa + e2 * vv) / (e1 * bb + e2);
```

RWKV官方kernel通过**维护log-space最大值pp**来避免直接计算大指数值。而您的cumsum实现**没有**这层保护——这是NaN发生的根本数学原因。

DeepSeek V4的 `k.clamp(-8,8) + pre_norm` 方案是一个**近似修复**：不走log-space，而是约束输入range。这在大多数情况下有效，但存在信息损失风险（被clamp的k值梯度为0）。

---

## (2) 模型内部其他风险点全面排查

### 🔴 P0 — 可能直接导致NaN或训练崩溃

#### 风险1：BiWKV的cumsum实现缺少log-space稳定化

**位置**：`BiWKV.forward`中的cumsum方案

**当前代码**：
```python
ek, ekv = k.exp(), k.exp() * v  # k已clamp(-8,8)，但exp(8)=2981
arange_L = torch.arange(L, device=k.device).float().view(1, L, 1)
ew_pow = ew.pow(arange_L)  # 衰减项的L次幂
S_fwd = (ekv / ew_pow).cumsum(dim=1) * ew_pow  # ← 风险
```

**分析**：即使k已clamp到±8，`exp(8)=2981`。在L=16384的序列上做cumsum，分母`D_fwd = (ek / ew_pow).cumsum(dim=1) * ew_pow` 最坏情况可达 `2981 × 16384 ≈ 4.9×10^7`（单通道），虽不会溢出fp32，但**梯度回传**时step大时仍可能spike。

更关键的是 `ew_pow = ew.pow(arange_L)`：当`arange_L[0,-1,0]=16383`时，如果`ew=0.999`（接近1的衰减），`0.999^16383 ≈ 7.5e-8`。而如果`ew=1.001`（训练后w学到负decay），`1.001^16383 ≈ 1.2e7`。**只要w学到任何使ew>1的值，ew_pow就会爆炸**。

参考[RWKV官方推荐](https://github.com/blinkdl/rwkv-lm)：
> "Note: In [state = kv + w * state] everything must be in fp32 because w can be very close to 1."

以及[RWKV-7 clampw.cu](https://github.com/BlinkDL/RWKV-LM/blob/6fa41b95/RWKV-v7/train_temp/cuda/rwkv7_clampw.cu)中：
```c
constexpr float W_SCALE = -0.6065306597f; // -exp(-0.5)
w[i] = __expf(W_SCALE / (1.0f + __expf(-to_float(w_[idx]))));
```
RWKV-7通过sigmoid将w约束在(0, exp(-0.5))范围内，**确保decay始终<1**。

**修复建议**：
```python
# 确保 ew 严格 < 1（衰减必须衰减）
w_clamped = self.spatial_decay.clamp(-8, -0.01)  # 强制负数
ew = (-w_clamped.abs() / total_tokens).exp()  # 必然 < 1
# 或更安全：
ew = torch.sigmoid(self.spatial_decay) * 0.99 + 0.001  # ∈ (0.001, 0.991)
```

#### 风险2：ew_pow在长序列上的数值下溢

**位置**：`BiWKV.forward`的反向扫描

**分析**：当ew<1且L=16384，`ew^L` 极小（如`0.99^16384≈0`），导致`ekv / ew_pow` 变成超大值（有效是/0），然后cumsum后再乘回`ew_pow`理论上应恢复，但fp32精度有限，**中间值溢出后乘回来也是NaN**。

这正是[RWKV CUDA kernel](https://github.com/BlinkDL/RWKV-LM/blob/acc4293f/RWKV-v4neo/cuda/wkv_cuda.cu)使用log-space的原因——中间值永远不需要实际计算超大/超小值。

**修复建议**：

方案A（推荐）：改写为chunk-wise递推，避免极长cumsum：
```python
CHUNK = 256  # 每256个token递推一次state
state = torch.zeros(B, 1, C, device=k.device)
outputs = []
for i in range(0, L, CHUNK):
    chunk_k = k[:, i:i+CHUNK]
    chunk_v = v[:, i:i+CHUNK]
    # 在chunk内做short-range cumsum（仅256步，安全）
    # 跨chunk用state递推
    chunk_out, state = self._chunk_wkv(chunk_k, chunk_v, state, ew)
    outputs.append(chunk_out)
```

方案B：改为log-space实现（参考[数值稳定WKV实现](https://www.youngju.dev/blog/ai-papers/2026-03-07-ai-papers-rwkv-architecture-rnn-transformer-hybrid.en)）：
```python
def log_space_wkv(k, v, w, u, L):
    """参考RWKV官方CUDA kernel的Python等价"""
    aa, bb, pp = 0., 0., -1e38
    outputs = []
    for i in range(L):
        ww = u + k[i]
        p = max(pp, ww)
        e1 = exp(pp - p)
        e2 = exp(ww - p)
        y = (e1 * aa + e2 * v[i]) / (e1 * bb + e2)
        outputs.append(y)
        # state update
        ww = w + pp
        p = max(ww, k[i])
        e1 = exp(ww - p)
        e2 = exp(k[i] - p)
        aa = e1 * aa + e2 * v[i]
        bb = e1 * bb + e2
        pp = p
    return outputs
```

---

### 🟡 P1 — 训练不稳定/收敛困难

#### 风险3：R/K/V投影初始化问题

**位置**：`SpatialWKV2D.__init__`中的`proj_r, proj_k, proj_v`

**问题**：PyTorch的`nn.Linear`默认使用Kaiming uniform初始化，其scale为 `1/sqrt(in_features) = 1/sqrt(64) ≈ 0.125`。这对k投影来说**太大了**。

参考[RWKV-7训练参考实现](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/rwkv7_train_simplified.py)：
```python
self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
# 对于C=64: ±0.05/8 = ±0.00625
```

而您当前默认初始化的范围是 ±0.125，比RWKV-7推荐值**大20倍**。

**修复建议**：
```python
# 在SpatialWKV2D.__init__末尾添加
nn.init.uniform_(self.proj_k.weight, -0.05/math.sqrt(channels), 0.05/math.sqrt(channels))
nn.init.uniform_(self.proj_r.weight, -0.5/math.sqrt(channels), 0.5/math.sqrt(channels))
nn.init.uniform_(self.proj_v.weight, -0.5/math.sqrt(channels), 0.5/math.sqrt(channels))
nn.init.zeros_(self.proj_out.weight)  # 已有 ✓
```

#### 风险4：weight decay应用范围不当

**位置**：训练配置`configs/v6_bravo.yaml`

[RWKV官方](https://github.com/blinkdl/rwkv-lm)明确强调：
> "**Make sure you only apply weight decay to large matrix parameters** (basically projections) in your model instead of all parameters. **THIS IS VERY IMPORTANT.**"

不应对以下参数施加weight decay：
- `spatial_decay` / `spatial_first`（BiWKV的w/u参数）
- `spatial_gamma`
- LayerNorm的weight/bias
- `tau`（温度参数）
- MVCShift中的DWConv参数

**建议**：在optimizer中区分参数组：
```python
decay_params = [p for n, p in model.named_parameters() if 'weight' in n and p.dim() >= 2 and 'norm' not in n]
no_decay_params = [p for n, p in model.named_parameters() if p not in decay_params_set]
```

#### 风险5：Adam优化器eps和beta2设置

**位置**：训练配置

参考[RWKV-6 spike修复建议](https://github.com/blinkdl/rwkv-lm)：
> 1. `--adam_eps 1e-18`
> 2. `--beta2 0.95` if you see spikes
> 3. warmup: `lr = lr * (0.01 + 0.99 * step / w_step)`

您当前配置（`lr=8e-4, epochs=50, warmup=5`）的warmup策略可能不够平滑。**建议将warmup初始lr比例从默认0.2降到0.01**。

#### 风险6：四方向扫描head间数值不平衡

**位置**：SpatialWKV2D中四方向concat

**问题**：四个扫描方向共享同一个BiWKV实例（`self.bi_wkv`），但各方向的token排列顺序不同，导致：
- 对角线扫描的有效"邻域"关系比水平/垂直更稀疏
- 同一个`spatial_decay`参数对不同方向的语义不同
- Concat后4个head的scale/variance可能差异数倍

**建议**：为每个方向使用独立的BiWKV实例（增加参数极少，仅4×C/4个w/u参数）：
```python
self.bi_wkv_list = nn.ModuleList([BiWKV(self.head_dim) for _ in range(4)])
```

或至少在concat前对各head做独立LayerNorm：
```python
self.head_norms = nn.ModuleList([nn.LayerNorm(self.head_dim) for _ in range(4)])
```

---

### 🟢 P2 — 长期稳定性/收敛质量

#### 风险7：TemporalCorrespondence温度参数tau退化

**位置**：`TemporalCorrespondence.__init__`中 `self.tau = nn.Parameter(torch.ones(1) * 0.07)`

**问题**：初始tau=0.07，cosine similarity ∈ [-1,1]，则logit范围 = [-14.3, 14.3]。如果tau进一步减小到0.01，logit范围变为[-100,100]，`sim.clamp(-20,20)`虽然限制了范围，但**梯度已被截断**，tau的梯度信号丢失。

**建议**：
```python
# 用softplus保证tau>0且有下界
self.tau_raw = nn.Parameter(torch.zeros(1))  # raw parameter
@property
def tau(self):
    return F.softplus(self.tau_raw) + 0.05  # ∈ (0.05, +∞)
```

#### 风险8：MVC-Shift的三分支叠加导致幅值膨胀

**位置**：`MVCShift.forward`

**当前代码**：
```python
def forward(self, x):
    out = x
    for branch in self.branches:
        out = out + branch(x)  # 3次加法，输出 = x + b1(x) + b2(x) + b3(x)
    return out
```

**问题**：假设每个branch(x)的scale与x相当，输出scale ≈ 4x。这在送入SpatialWKV2D的R/K/V投影前**没有归一化**，相当于给k投影的输入加了4倍scale，间接加速了k值漂移。

**建议**：在MVC-Shift输出后加LayerNorm2d（或直接在SpatialWKV2D的pre_norm中处理——如果已加pre_norm则此风险已被覆盖）：
```python
def forward(self, x):
    out = x
    for branch in self.branches:
        out = out + branch(x)
    return out / 4.0  # 简单scale down
    # 或依赖 SpatialWKV2D 的 pre_norm
```

**验证**：如果DeepSeek V4添加的`pre_norm`（LayerNorm）在R/K/V投影前，则此风险**已被覆盖**。确认pre_norm位置即可。

#### 风险9：NDPN噪声减法路径

**位置**：`f_noise = f_enc - noise × strength × gamma`

**问题**：减法操作可能产生负值特征，传到IGRF Stage1后 `img_s1 = clamp(img_center + δ(f_noise_gated), 0, 1)` 中的δ可能是极端负值，导致img_s1=0（纯黑），后续lit_up_map的乘法无法恢复。

**建议**：对`strength`加sigmoid约束：
```python
strength = torch.sigmoid(self.denoise_strength(noise_feat + conf_map))  # ∈ (0,1)
```

#### 风险10：IGRF Stage3乘法链的梯度消失

**位置**：`res_t = clamp(img_s2 × lit_up_map × (1+A_illu), 0, 1)`

**问题**：三个量相乘，如果img_s2暗区≈0（低光场景常见），则梯度 `∂loss/∂lit_up_map ≈ 0` → IFPN分支梯度消失。

**建议**：在Stage3前对img_s2加bias防止纯零：
```python
img_s2_safe = img_s2 + 0.01  # 保证最小亮度
res_t = clamp(img_s2_safe * lit_up_map * (1 + A_illu), 0, 1)
```

---

## 综合修复优先级

| 优先级 | 风险 | 严重程度 | 状态 | 建议 |
|--------|------|----------|------|------|
| 🔴 P0 | exp(k)爆炸 | NaN直接原因 | ✅ 已修复(clamp+pre_norm) | 监控k值分布 |
| 🔴 P0 | ew_pow长序列下溢/上溢 | NaN隐患 | ⚠️ 需确认 | 强制decay<1或改chunk递推 |
| 🟡 P1 | K投影初始化过大 | 加速NaN复现 | ❌ 未修复 | `±0.05/sqrt(C)` |
| 🟡 P1 | weight decay范围不当 | 训练不稳 | ❌ 未确认 | 仅对大矩阵decay |
| 🟡 P1 | adam_eps/beta2 | 训练spike | ❌ 未确认 | eps=1e-18, beta2=0.95 |
| 🟡 P1 | 四方向共享BiWKV | 数值不均 | ❌ 未修复 | 独立实例或per-head LN |
| 🟢 P2 | tau退化 | C_omega质量 | ⚠️ 部分修复(clamp) | softplus+下界 |
| 🟢 P2 | MVC-Shift幅值4x | pre_norm可覆盖 | ✅ 若pre_norm在位 | 确认位置 |
| 🟢 P2 | NDPN减法溢出 | 输出异常 | ❌ | sigmoid(strength) |
| 🟢 P2 | IGRF乘法梯度消失 | 收敛慢 | ❌ | 加最小bias |

---

## 立即行动建议

### 紧急确认项（不需要重启训练）

1. **确认`spatial_decay`是否可能学到使ew>1的值**：
```python
# 在训练循环中加hook监控
with torch.no_grad():
    w = model.sace.spatial_wkv.bi_wkv.spatial_decay
    ew = (-w.abs() / total_tokens).exp()
    print(f"ew range: [{ew.min():.6f}, {ew.max():.6f}]")  # 必须 < 1
```

2. **确认pre_norm的位置**：应在MVC-Shift输出之后、R/K/V投影之前。

### 下次重启时建议的代码修改

```python
# BiWKV: 强制衰减 < 1
def forward(self, k, v, total_tokens):
    B, L, C = k.shape
    # 安全clamp
    k = k.clamp(-8, 8)
    v = v.clamp(-8, 8)
    
    # 强制 decay ∈ (0, 1)
    w = -F.softplus(self.spatial_decay)  # 确保负数
    ew = (w / total_tokens).exp().view(1, 1, C)  # 必然 < 1
    assert (ew < 1).all(), f"ew must < 1, got max={ew.max()}"
    
    u = self.spatial_first.clamp(-5, 5)
    u_coef = (u / total_tokens).exp().view(1, 1, C)
    
    # 分chunk避免超长cumsum的精度问题
    CHUNK = min(512, L)
    # ... chunk-wise实现 ...
```

---

## 参考来源

1. [[Bug] RWKV7Attention returns nan · Issue #150 · fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/issues/150)
2. [BlinkDL/RWKV-LM — Fixing RWKV-6 Spikes & Training Tips](https://github.com/blinkdl/rwkv-lm)
3. [Question about the loss backward with RWKV7 · Issue #309 · BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM/issues/309)
4. [RWKV State Management — Numerical Stability](https://github.com/zechenzhangAGI/AI-research-SKILLs/blob/main/01-model-architecture/rwkv/references/state-management.md)
5. [RWKV Architecture Deep Dive: Numerical Stability via Log-Space Computation](https://www.youngju.dev/blog/ai-papers/2026-03-07-ai-papers-rwkv-architecture-rnn-transformer-hybrid.en)
6. [RWKV-v4neo CUDA WKV Kernel (log-space reference)](https://github.com/BlinkDL/RWKV-LM/blob/acc4293f/RWKV-v4neo/cuda/wkv_cuda.cu)
7. [WKV Operation | DeepWiki — Numerical Stability Considerations](https://deepwiki.com/BlinkDL/RWKV-LM/2.2-wkv-operation)
8. [RWKV-7 Train Reference — rwkv7_train_simplified.py](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/rwkv7_train_simplified.py)
9. [RWKV-7 clampw CUDA kernel](https://github.com/BlinkDL/RWKV-LM/blob/6fa41b95/RWKV-v7/train_temp/cuda/rwkv7_clampw.cu)
10. [Vision-RWKV: Efficient and Scalable Visual Perception (ICLR 2025)](https://arxiv.org/abs/2403.02308)
11. [OpenGVLab/Vision-RWKV GitHub](https://github.com/opengvlab/vision-rwkv)
12. [Improving RWKV with Parallel Cumulative Sums — Log-space Exponentially Weighted WKV](https://jackd.github.io/posts/improving-rwkv/)

*根据联网检索（检索于 2026-06-30T20:52:52+08:00）*