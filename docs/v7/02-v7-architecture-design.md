# Golf v7 重新设计: 矩阵状态 RWKV + 多元噪声分割

> 日期: 2026-09-28 | 仓库: TFS-Net/docs/v7/02-v7-architecture-design.md  
> 前置文档: `01-RWKV-mechanism-survey.md` (RWKV-4~8 机制调研)  
> 目标: 在多元噪声分割结构（N/L/M 三分支）框架内，用 RWKV-5/6 级别的矩阵状态机制替换当前 naive grouped RWKV-4，形成完整设计方案

---

## 一、当前 v7 架构诊断

### 1.1 当前架构 (正在训练的 baseline)

```
Input [B, T=5, 3, H, W]
  ↓
Stage 1: SharedEncoder (逐帧, 0.41M)
  → F1 [B,T,32,H,W], F2 [B,T,64,H/2,W/2], F3 [B,T,128,H/4,W/4] (F3 未用)
  ↓
Stage 2: feature_proj (1×1 conv 64→128) + PixelTemporalAttentionSimple
  → feat_aligned [B, 128, H/2, W/2] (中心帧, 3D conv window=3)
  ↓
Stage 3: FrameLevelRWKV (8 独立 RWKV-4 头, FiLM 调制)
  → feats_rwkv [B, T, 128, H/2, W/2] + frame_ctx [B, T, 128]
  ↓
Stage 4: feat_aligned + feat_center_rwkv → 2×NAFBlock → Upsample3x3 → to_rgb
  → out [B, 3, H, W] + residual_gamma × X_center
```

**总参数: 1.04M** | **RWKV 部分: ~0.6M** (8头 × 16维 × 2块)

### 1.2 关键问题

| 问题 | 严重性 | 说明 |
|------|--------|------|
| **RWKV-4 向量状态** | 致命 | 每头状态仅 16 标量 (head_dim=16)，几乎无法存储有意义的帧间关系 |
| **头间零通信** | 严重 | 8 头完全独立，无法学习跨通道模式 |
| **无噪声分割** | 严重 | 单分支解码器，未利用 N/L/M 退化分解的物理先验 |
| **Global Pool 信息损失** | 中等 | AdaptiveAvgPool2d(1) 丢失所有空间信息，FiLM 只能做全局调制 |
| **F3 未使用** | 轻微 | Encoder 花参数算了 F3 但被丢弃 |

### 1.3 当前训练进度 (baseline 参照)

- Epoch 9/60 运行中 (2026-09-28 11:00)
- Loss 在 0.03~0.36 范围波动
- 预计 epoch 10 首次验证 PSNR
- Golf R2 baseline: val PSNR=20.14 (使用三分支 + TCA-RWKV, 3.50M 参数)

---

## 二、重新设计方案

### 2.1 设计原则

1. **RWKV 用对地方**: 帧级语义建模（非像素级对齐）— 延续 v7 已验证的正确方向
2. **矩阵状态是必需品**: 升级到 RWKV-5/6 的 $S \in \mathbb{R}^{d \times d}$ 矩阵状态
3. **恢复三分支结构**: N/L/M 退化分解是 Golf 家族的核心物理先验
4. **RWKV 多头 = 退化分量分离**: 多头自然对应 N/L/M，不需要独立的 TCA 解耦模块
5. **参数预算 ≤ 3.5M**: 与 Golf R2 可比，RTX 4090 可训练

### 2.2 架构总览: **Golf v7-R (RWKV-Redesigned)**

```
Input [B, T=5, 3, H, W]
  │
  ├── Stage 1: SharedEncoder (逐帧独立, 0.41M, 复用 Golf R2)
  │   → F1 [B,T,32,H,W]
  │   → F2 [B,T,64,H/2,W/2]
  │
  ├── Stage 2: PixelTemporalAttention (像素级对齐, ~0.01M)
  │   Feature Proj: 1×1 conv 64→128
  │   3D Conv (窗口=3): 捕捉局部运动
  │   → feat_aligned [B, 128, H/2, W/2]
  │
  ├── Stage 3: MatrixRWKV-TCA ⭐ (帧级多头, 矩阵状态, ~0.55M)
  │   │
  │   │  3.1 Spatial Summary (替代 Global Pool)
  │   │      AdaptiveAvgPool2d(4) → [B, T, C, 4, 4]
  │   │      Flatten → [B, T, C×16] → Linear → [B, T, D=192]
  │   │
  │   │  3.2 Multi-Head Matrix RWKV (h=6, d=32, 矩阵状态 32×32)
  │   │      ┌─ Head 1-2: Noise-aware (衰减快, 捕捉 i.i.d. 噪声模式)
  │   │      ├─ Head 3-4: Illumination-aware (衰减慢, 捕捉光照趋势)
  │   │      └─ Head 5-6: Motion-aware (动态衰减, 运动自适应)
  │   │      每头: s_t = s_{t-1} · diag(w_t) + v_t^T · k_t
  │   │      w_t 数据相关 (RWKV-6 风格)
  │   │      → frame_ctx [B, T, D=192]
  │   │
  │   │  3.3 Context Decomposition (替代 TCA 解耦)
  │   │      Linear 192 → {ctx_N(64), ctx_L(64), ctx_M(64)}
  │   │      Ortho regularization on {ctx_N, ctx_L, ctx_M}
  │   │
  │   │  3.4 Spatial Injection (FiLM per branch)
  │   │      ctx_X → scale_X, shift_X (1×1 conv)
  │   │      F_X = feat_aligned * (1 + scale_X) + shift_X
  │   │      → F_N, F_L, F_M [B, 128, H/2, W/2]
  │
  ├── Stage 4: 三分支处理 (~1.85M, 复用/简化 Golf R2)
  │   ├── BranchN (去噪, ~0.60M): 简化版, 2×NAFBlock + Upsample3x3
  │   ├── BranchL (光照, ~0.55M): Retinex + gamma + 2×NAFBlock + Upsample3x3
  │   └── BranchM (运动, ~0.70M): 简化流估计 + 2×NAFBlock + Upsample3x3
  │   → Y_N, Y_L, Y_M [B, 3, H, W]
  │
  └── Stage 5: AdaptiveFusion (~0.06M, 复用 Golf R2)
      → O [B, 3, H, W]
```

**总参数预算: ~2.9M** (SharedEncoder 0.41 + PixelTemp 0.01 + MatrixRWKV-TCA 0.55 + 三分支 1.85 + Fusion 0.06)

### 2.3 Stage 3 详细设计: MatrixRWKV-TCA

这是本次重新设计的核心创新——用矩阵状态 RWKV 替代原有 TCA-RWKV 的解耦机制。

#### 2.3.1 Spatial Summary (替代 Global Pool)

**问题**: 当前 `AdaptiveAvgPool2d(1)` 将 H/2×W/2 压缩到 1×1，丢失所有空间结构。

**方案**: 使用 `AdaptiveAvgPool2d(4)` 保留 4×4=16 个空间位置，再线性投影。

```python
class SpatialSummary(nn.Module):
    def __init__(self, in_dim=128, out_dim=192, spatial_size=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(spatial_size)
        self.proj = nn.Linear(in_dim * spatial_size * spatial_size, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        tokens = []
        for t in range(T):
            pooled = self.pool(x[:, t])          # [B, C, 4, 4]
            flat = pooled.flatten(1)              # [B, C*16]
            tokens.append(self.proj(flat))        # [B, D]
        tokens = torch.stack(tokens, dim=1)       # [B, T, D]
        return self.ln(tokens)
```

**参数**: 128×16×192 + 192 + 192×2 ≈ 393K

#### 2.3.2 Matrix RWKV Time Mixing (RWKV-6 风格)

**核心改进: 从向量状态升级到矩阵状态**

```python
class MatrixRWKVTimeMix(nn.Module):
    """
    RWKV-6 风格 Time Mixing: 矩阵值状态 + 数据相关衰减
    
    状态更新: S_t = S_{t-1} · diag(w_t) + v_t^T · k_t
    w_t = exp(-exp(base_decay + decay_lora(shifted_x)))  # 数据相关
    """
    def __init__(self, dim: int, num_heads: int = 6, head_size: int = 32):
        super().__init__()
        self.dim = dim           # 192
        self.num_heads = num_heads
        self.head_size = head_size  # d = 32, 状态 S ∈ R^{32×32}
        assert dim == num_heads * head_size
        
        # Projections (shared across heads, then reshape to multi-head)
        self.W_r = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        
        # Token Shift (V7 简单 lerp 风格)
        self.mix_r = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.mix_k = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.mix_v = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        
        # 数据相关衰减 (RWKV-6 风格)
        self.base_decay = nn.Parameter(torch.ones(num_heads, head_size) * 0.5)
        self.decay_lora_down = nn.Linear(dim, dim // 4, bias=False)
        self.decay_lora_up = nn.Linear(dim // 4, dim, bias=False)
        
        # 门控 (V5 SiLU gate)
        self.W_g = nn.Linear(dim, dim, bias=False)
        
        # Per-head GroupNorm (V5 风格, 替代归一化分母)
        self.group_norm = nn.GroupNorm(num_heads, dim)
        
    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        H, d = self.num_heads, self.head_size
        
        # Token shift: x_{t-1} 与 x_t 的 lerp
        x_shifted = F.pad(x[:, :-1], (0, 0, 1, 0))  # [B, T, D], 第0帧 shift=0
        
        xr = x * self.mix_r + x_shifted * (1 - self.mix_r)
        xk = x * self.mix_k + x_shifted * (1 - self.mix_k)
        xv = x * self.mix_v + x_shifted * (1 - self.mix_v)
        
        # Projections
        r = self.W_r(xr).view(B, T, H, d)  # receptance
        k = self.W_k(xk).view(B, T, H, d)  # key
        v = self.W_v(xv).view(B, T, H, d)  # value
        g = torch.sigmoid(self.W_g(x))      # gate [B, T, D]
        
        # 数据相关衰减 w_t
        decay_delta = self.decay_lora_up(
            torch.tanh(self.decay_lora_down(xk))
        ).view(B, T, H, d)
        # w_t = exp(-exp(base + delta)), 保证 w_t ∈ (0, 1)
        w = torch.exp(-torch.exp(
            self.base_decay.unsqueeze(0).unsqueeze(0) + decay_delta
        ))  # [B, T, H, d]
        
        # 递归: 矩阵状态更新
        # S_t = S_{t-1} · diag(w_t) + v_t^T · k_t
        state = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(T):
            k_t = k[:, t]           # [B, H, d]
            v_t = v[:, t]           # [B, H, d]
            r_t = r[:, t]           # [B, H, d]
            w_t = w[:, t]           # [B, H, d]
            
            # 状态更新: S = S · diag(w) + v^T · k (外积)
            state = state * w_t.unsqueeze(-2) + torch.einsum('bhd,bhe->bhde', v_t, k_t)
            # 读取: o = r · S → [B, H, d]
            o_t = torch.einsum('bhd,bhde->bhe', r_t, state)
            outputs.append(o_t)
        
        out = torch.stack(outputs, dim=1)  # [B, T, H, d]
        out = out.reshape(B, T, D)
        
        # GroupNorm per head + gate
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        out = out * g  # SiLU-like gate
        out = self.W_o(out)
        
        return out
```

**与当前 RWKV-4 对比**:

| | 当前 (V4) | 新设计 (V6) |
|---|---|---|
| 状态形状 | 向量 $\mathbb{R}^{16}$ (per head) | 矩阵 $\mathbb{R}^{32 \times 32}$ (per head) |
| 总状态容量 | 8×16 = 128 标量 | 6×32×32 = 6,144 标量 (**48×**) |
| 衰减 | 固定 `time_decay` | 数据相关 `w_t = f(x_t)` (LoRA) |
| 状态更新 | $s_t = e^{-w} s_{t-1} + k_t v_t$ (逐元素) | $S_t = S_{t-1} \text{diag}(w_t) + v_t^T k_t$ (外积) |
| 头间通信 | 无 | 通过共享 W_r/W_k/W_v + GroupNorm |
| 参数量 | ~0.6M (含 8 个独立头) | ~0.35M (参数更高效) |

**参数估算**: 4 × (192×192) + 192×48 + 48×192 + 192×32 + GroupNorm ≈ 166K

#### 2.3.3 Channel Mix (V7 ReLU² MLP)

```python
class ReLUSquaredMLP(nn.Module):
    """V7 风格 Channel Mix: 简洁的 2-layer MLP"""
    def __init__(self, dim, hidden_ratio=3):
        super().__init__()
        hidden = dim * hidden_ratio
        self.W1 = nn.Linear(dim, hidden, bias=False)
        self.W2 = nn.Linear(hidden, dim, bias=False)
        self.mix = nn.Parameter(torch.ones(1, 1, dim) * 0.5)

    def forward(self, x):
        x_shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        xm = x * self.mix + x_shifted * (1 - self.mix)
        return self.W2(F.relu(self.W1(xm)).square())
```

**参数**: 192×576 + 576×192 ≈ 221K

#### 2.3.4 MatrixRWKVBlock (Time Mix + Channel Mix)

```python
class MatrixRWKVBlock(nn.Module):
    def __init__(self, dim=192, num_heads=6, head_size=32):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.time_mix = MatrixRWKVTimeMix(dim, num_heads, head_size)
        self.channel_mix = ReLUSquaredMLP(dim)

    def forward(self, x):
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x
```

**单块参数**: ~387K → **2 块 × 387K = 774K**

#### 2.3.5 Context Decomposition (替代 TCA 解耦)

**核心思想**: RWKV 多头输出自然包含不同退化分量的信息，通过线性投影 + 正交正则化，显式解耦为 N/L/M 三个上下文向量。

```python
class ContextDecomposition(nn.Module):
    """将 RWKV 输出解耦为 N/L/M 三个上下文"""
    def __init__(self, dim=192, ctx_dim=64):
        super().__init__()
        self.proj_N = nn.Linear(dim, ctx_dim)
        self.proj_L = nn.Linear(dim, ctx_dim)
        self.proj_M = nn.Linear(dim, ctx_dim)
        self.ln_N = nn.LayerNorm(ctx_dim)
        self.ln_L = nn.LayerNorm(ctx_dim)
        self.ln_M = nn.LayerNorm(ctx_dim)

    def forward(self, frame_ctx):
        # frame_ctx: [B, T, D] → 取中心帧
        center = frame_ctx[:, frame_ctx.shape[1] // 2]  # [B, D]
        ctx_N = self.ln_N(self.proj_N(center))  # [B, 64]
        ctx_L = self.ln_L(self.proj_L(center))
        ctx_M = self.ln_M(self.proj_M(center))
        return ctx_N, ctx_L, ctx_M

    def ortho_loss(self, ctx_N, ctx_L, ctx_M):
        """正交约束: 三个上下文应尽量正交"""
        cos_NL = F.cosine_similarity(ctx_N, ctx_L, dim=-1).abs().mean()
        cos_NM = F.cosine_similarity(ctx_N, ctx_M, dim=-1).abs().mean()
        cos_LM = F.cosine_similarity(ctx_L, ctx_M, dim=-1).abs().mean()
        return (cos_NL + cos_NM + cos_LM) / 3
```

**参数**: 3 × (192×64 + 64 + 64×2) ≈ 37.5K

#### 2.3.6 Spatial Injection (FiLM per branch)

```python
class BranchFiLM(nn.Module):
    """将上下文注入空间特征 (FiLM 调制)"""
    def __init__(self, ctx_dim=64, feat_dim=128):
        super().__init__()
        self.to_scale = nn.Linear(ctx_dim, feat_dim)
        self.to_shift = nn.Linear(ctx_dim, feat_dim)
        # 零初始化 → 初始时 FiLM 是恒等变换
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, feat, ctx):
        # feat: [B, 128, H/2, W/2], ctx: [B, 64]
        scale = self.to_scale(ctx)[:, :, None, None]  # [B, 128, 1, 1]
        shift = self.to_shift(ctx)[:, :, None, None]
        return feat * (1 + scale) + shift
```

**参数**: 3 × (64×128 + 128) × 2 ≈ 49.5K

### 2.4 三分支简化方案

为控制参数预算，对 Golf R2 的三分支进行简化:

#### BranchN (去噪): 0.60M
- 去掉 var_map 通路（矩阵 RWKV 的 Noise 头已捕捉噪声统计）
- 保留: 2×NAFBlock(128) + Upsample3x3 + to_rgb
- 不使用 F1 skip（节省参数，简化流程）

#### BranchL (光照): 0.55M
- 保留 Retinex 物理先验 (L_t, R_t, gamma)
- 简化: 去掉 illum_refine（低分辨率 L 直接 bilinear 到全分辨率）
- 保留: illum_head (zero-init) + 2×NAFBlock + Upsample3x3 + gamma

#### BranchM (运动): 0.70M
- 简化 FlowEstimator: 只用中心帧与相邻 2 帧（而非全部 4 帧）
- 保留: confidence gating + fusion_conv + 2×NAFBlock + Upsample3x3
- 去掉 F1 skip

### 2.5 完整参数预算

| 模块 | 参数量 | 说明 |
|------|--------|------|
| SharedEncoder | 0.41M | 复用 Golf R2，不修改 |
| feature_proj | 0.01M | 1×1 conv 64→128 |
| PixelTemporalAttention | 0.01M | 3D conv 窗口=3 |
| SpatialSummary | 0.39M | AvgPool(4) + Linear |
| MatrixRWKVBlock ×2 | 0.77M | V6 矩阵状态, 6 头 |
| ContextDecomposition | 0.04M | 线性解耦 N/L/M |
| BranchFiLM ×3 | 0.05M | 上下文注入 |
| BranchN (简化) | 0.60M | 2×NAFBlock + Upsample |
| BranchL (简化) | 0.55M | Retinex + 2×NAFBlock |
| BranchM (简化) | 0.70M | Flow + Conf + 2×NAFBlock |
| AdaptiveFusion | 0.06M | 复用 Golf R2 |
| **总计** | **~3.6M** | 与 Golf R2 (3.50M) 可比 |

---

## 三、与 Golf R2 的关键差异

| 方面 | Golf R2 (TCA-RWKV) | Golf v7-R (MatrixRWKV-TCA) |
|------|-------------------|-----------------------------|
| **RWKV 版本** | RWKV-4 向量状态, 单头 | RWKV-6 矩阵状态, 6 多头 |
| **RWKV 操作层级** | 像素级 (逐像素时序) | 帧级 (全局语义) |
| **退化解耦** | 独立 TCA 模块 (三查询解耦) | RWKV 多头 + 线性投影 + 正交约束 |
| **帧间对齐** | TCA 隐式对齐 | 显式 PixelTemporalAttention + BranchM flow |
| **衰减机制** | 固定 | 数据相关 (RWKV-6 LoRA) |
| **状态容量** | ~128 标量 | ~6,144 标量 (48×) |
| **参数量** | 3.50M | ~3.6M |
| **物理先验** | Retinex (BranchL) | Retinex + 分层时序 + 正交解耦 |

---

## 四、训练方案

### 4.1 Loss 函数

```python
loss = L1(output, GT)
     + 0.1 * L1(Y_N, GT)   # 分支监督
     + 0.1 * L1(Y_L, GT)
     + 0.1 * L1(Y_M, GT)
     + 0.01 * ortho_loss    # 正交约束
```

如 pytorch_msssim 可安装:
```python
loss += 0.1 * (1 - MS_SSIM(output, GT))
```

### 4.2 训练超参 (第一阶段)

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 1 | RTX 4090 内存约束 |
| lr | 6e-4 | AdamW |
| epochs | 60 | 与 baseline 一致 |
| scheduler | CosineAnnealing (T_min=1e-6) | |
| crop_size | 256×256 | |
| T | 5 | |
| val_interval | 5 | 比 baseline 更频繁 |

### 4.3 实施路径

**Phase 1: 矩阵 RWKV 核心 (先验证 RWKV 升级)**
1. 实现 `MatrixRWKVTimeMix` + `MatrixRWKVBlock`
2. 替换 `frame_rwkv.py` 中的 `RWKVTimeMix`
3. 保持单分支解码器，先验证矩阵状态的效果
4. 成功标准: ep10 PSNR > 当前 baseline

**Phase 2: 三分支恢复 (加入退化分解)**
1. 实现 `ContextDecomposition` + `BranchFiLM`
2. 简化并集成 BranchN/L/M
3. 集成 AdaptiveFusion
4. 成功标准: ep30 PSNR > Golf R2 baseline (20.14)

**Phase 3: 精调与消融**
1. 头数消融 (4 vs 6 vs 8)
2. 空间摘要消融 (AvgPool 1×1 vs 4×4 vs 8×8)
3. 衰减类型消融 (固定 vs 数据相关)
4. 正交约束强度消融

---

## 五、风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 矩阵状态 T=5 收益不大 | 中 | Phase 1 快速验证; 若无增益则回退到 V5 固定衰减 |
| 3.6M 超出 GPU 内存 | 低 | batch=1 已验证 1.04M 可行; 3.6M 估计 ~10GB 可控 |
| 三分支增加训练不稳定性 | 中 | 零初始化 FiLM + 分支 loss 权重渐增 (warmup) |
| 正交约束过强限制表达 | 低 | λ_ortho = 0.01, 可调节 |
| SpatialSummary 参数过多 (393K) | 中 | 可降低 spatial_size=2 (4 位置 vs 16), 参数降至 ~100K |

---

## 六、文件规划

```
models/golf_v7r/                   # 新目录, "r" = redesigned
├── __init__.py
├── golfnet_v7r.py                 # 主网络
├── matrix_rwkv.py                 # MatrixRWKVTimeMix + Block
├── spatial_summary.py             # SpatialSummary
├── context_decomp.py              # ContextDecomposition + BranchFiLM
├── branch_n_simple.py             # 简化 BranchN
├── branch_l_simple.py             # 简化 BranchL  
├── branch_m_simple.py             # 简化 BranchM
├── loss.py                        # 多分支 + 正交 loss
└── pixel_temporal.py              # 复用 golf_v7 版本

configs/golf_v7r.yaml              # 训练配置
train_golf_v7r.py                  # 训练脚本
```

---

## 七、关键数学总结

### 矩阵状态更新 (RWKV-6 风格, 每头)

$$S_t = S_{t-1} \cdot \text{diag}(w_t) + v_t^T \cdot k_t$$

其中:
- $S_t \in \mathbb{R}^{d \times d}$ (矩阵值状态, d=32)
- $w_t = \exp(-\exp(w_{\text{base}} + \text{LoRA}(x_t))) \in (0, 1)^d$ (数据相关衰减)
- $v_t, k_t \in \mathbb{R}^d$ (值向量、键向量)
- $v_t^T \cdot k_t \in \mathbb{R}^{d \times d}$ (秩-1 外积更新)

### 输出读取

$$o_t = r_t \cdot S_t \in \mathbb{R}^d$$

其中 $r_t$ 是 receptance 向量 (控制读取)

### 退化解耦

$$[\text{ctx}_N, \text{ctx}_L, \text{ctx}_M] = [W_N, W_L, W_M] \cdot \text{RWKV}(x)_{\text{center}}$$

$$\mathcal{L}_{\text{ortho}} = \frac{1}{3}\sum_{(i,j) \in \{NL, NM, LM\}} |\cos(\text{ctx}_i, \text{ctx}_j)|$$

---

## 附录: 为什么不用 RWKV-7 (Goose) 的广义 Delta 规则?

RWKV-7 的核心升级是转移矩阵从 $\text{diag}(w_t)$ 扩展为 $\text{diag}(w_t) - \hat{\kappa}_t^T (a_t \odot \hat{\kappa}_t)$。这在 NLP 的长上下文（T=4K~32K）中非常有价值，因为矩阵状态会逐渐饱和，需要主动"忘记"过时信息。

但 LLVE 的 T=5:
1. **状态不会饱和**: 5 个秩-1 更新到 32×32 矩阵，远未满秩
2. **无需主动遗忘**: 5 帧全部有用，不需要移除旧帧的 KV 关联
3. **实现复杂度翻倍**: 需要额外的 $\kappa, a_t$ 投影和归一化
4. **V6 的数据相关衰减已足够**: 运动自适应衰减覆盖了 LLVE 的核心需求

**结论**: V6 级别 (矩阵状态 + 数据相关衰减) 是 T=5 LLVE 的最优选择点，V7 的边际收益不值得额外复杂度。

---

## 八、实现状态 (2026-09-28)

### 8.1 已实现文件

```
models/golf_v7r/
├── __init__.py                 ✅ 模块导出
├── matrix_rwkv.py              ✅ MatrixRWKVTimeMix (204K) + ReLUSquaredMLP (221K) + MatrixRWKVBlock (426K)
├── spatial_summary.py          ✅ SpatialSummary
├── context_decomp.py           ✅ ContextDecomposition + BranchFiLM (零初始化验证通过)
├── branch_n_simple.py          ✅ BranchNSimple (556K)
├── branch_l_simple.py          ✅ BranchLSimple (609K, Retinex 保留)
├── branch_m_simple.py          ✅ BranchMSimple (804K, flow + confidence)
├── upsample.py                 ✅ 复用 golf_v7 Upsample3x3
├── loss.py                     ✅ GolfV7RLoss (多分支 warmup + 正交)
└── golfnet_v7r.py              ✅ 主网络

configs/golf_v7r.yaml           ✅ batch=1, lr=4e-4, 60ep, val_interval=5
train_golf_v7r.py               ✅ 含 grad_clip + cosine scheduler + best.pth
```

### 8.2 实测参数分布 (总 3.51M)

| 模块 | 实测 | 设计预估 |
|------|------|---------|
| encoder | 0.410M | 0.41M |
| pixel_temporal | 0.024M | 0.01M |
| spatial_summary (s=2) | 0.099M | 0.10M |
| matrix_rwkv (2 blocks) | 0.852M | 0.77M |
| ctx_decomp | 0.037M | 0.04M |
| film (×3) | 0.050M | 0.05M |
| branch_N | 0.556M | 0.60M |
| branch_L | 0.609M | 0.55M |
| branch_M | 0.804M | 0.70M |
| fusion | 0.063M | 0.06M |
| feature_proj | 0.008M | 0.01M |
| **总计** | **3.51M** | ~3.6M |

与 Golf R2 (3.50M) 几乎一致，便于公平对比。

### 8.3 实测资源

- **GPU 显存**: 4.57 GB (batch=1, 256² crop, 训练峰值)
- **单步耗时**: ~0.055s (steady state)
- **单 epoch**: ~30 min (8253 steps)
- **60 epoch**: ~30 h

### 8.4 已验证的设计特性

- `BranchFiLM` 零初始化 → 初始恒等变换 (单元测试通过)
- `ReLUSquaredMLP` W2 零初始化 → 初始 channel mix 输出为零
- `ContextDecomposition.ortho_loss` 初始 ≈ 0.04-0.11 (随机初始化下已较低)
- 全模型前向 + 反向传播梯度检查通过 (358 params 有梯度)
- 训练自 2026-09-28 11:51 启动，稳定运行 (PID 2379543)

### 8.5 发现并修复的 bug

baseline `train_golf_v7_simple.py` 在 epoch 10 首次验证时崩溃:
`AttributeError: 'float' object has no attribute 'item'`。
根因: `utils/metrics.py` 的 `tensor_psnr` / `tensor_ssim` 返回 Python float，
但脚本调用 `.item()`。已修复 `train_golf_v7_simple.py` 与 `train_golf_v7r.py`。
这解释了 baseline 为何从未产出验证指标。

### 8.6 参照基线结果 (2026-09-28 12:15)

独立评估脚本 `eval_checkpoint.py` 补测了 baseline 的 epoch-10 表现:

| 模型 | Checkpoint | Val PSNR | Val SSIM | 参数量 |
|------|-----------|---------|---------|--------|
| Golf v7 (naive RWKV-4, 单分支) | epoch 10 | **19.42 dB** | 0.7525 | 1.04M |
| Golf R2 (TCA-RWKV) | — | 20.14 dB | — | 3.50M |
| Golf v7r (MatrixRWKV-TCA) | 训练中 | 待测 (ep5 首次验证) | — | 3.51M |

v7r 的成功判据:
- **ep5**: PSNR ≥ 19.0 (追上 baseline v7 ep10)
- **ep30**: PSNR ≥ 20.14 (超越 Golf R2)
- **ep60**: PSNR ≥ 21.0 (目标)

注: 评估在完整 1080p 分辨率下进行 (1120 样本)，单样本前向 ~1.06s，全量评估 ~6-8 min (与训练并发时)。

### 8.7 ⚠️ 关键修正: 矩阵状态读出索引错误 (2026-09-28 12:34)

首轮训练启动后，用数值测试 (对照 RWKV-6 朴素参考实现) 发现 `MatrixRWKVTimeMix`
的**输出读出索引方向错误**:

```python
# ❌ 错误 (首版): r 与 value 维收缩, 输出在 key 维
state = state * w_t.unsqueeze(-2) + einsum('bhd,bhe->bhde', v_t, k_t)
o_t = einsum('bhd,bhde->bhe', r_t, state)

# ✅ 正确 (修正后): r 与 key 维收缩, 输出在 value 维
state = state * w_t.unsqueeze(-2) + einsum('bhi,bhj->bhij', v_t, k_t)
o_t = einsum('bhj,bhij->bhi', r_t, state)
```

**错误后果**: 状态矩阵 `S[i,j]` 的语义被颠倒。RWKV 的设计是
`S[i,j] = v[i]·k[j]` (i=value 通道, j=key 通道), 读出时 receptance `r` 必须
与 **key 维 j** 收缩, 输出落在 **value 维 i**。首版把 `r` 与 value 维收缩、
输出在 key 维，导致读出 `o = S·r` 而非 `o = r·S`，等价于把矩阵状态当成了
转置使用 — 记忆的"寻址"方向完全反了。

**验证方法** (`/tmp/opencode/test_matrix_rwkv.py`): 手写朴素递推
```
S_t = S_{t-1}·diag(w_t) + v_t^T k_t
o_t = r_t · S_t
```
与模块 `MatrixRWKVTimeMix.recurrence()` 逐元素对比。修正后最大差异
`0.000000`，修正前 `1.65`。

**处置**: 首轮 (buggy) 训练产物移至 `outputs/golf_v7r_buggy_v1/`，
修正后于 12:34 重启训练 (PID 2411612)。为便于复现，`recurrence()`
被抽为 `@staticmethod`，可直接单元测试。
