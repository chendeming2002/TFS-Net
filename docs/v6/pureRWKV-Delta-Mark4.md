

# 分析与改进方案

## 一、两个意见的合理性分析

### 意见1：各阶段解冻模块以"无叠加启动"

**完全合理，且是当务之急。**

文档中的训练日志已经给出了铁证：Phase 2 PSNR 从 18.24 暴跌至 8.70，且持续无法恢复。根本原因链条非常清晰：

1. Phase 1 中 SGRF 的 StageBlock 只见过零输入（`f_noise=0, f_motion=0`），其卷积权重针对 `0→0` 映射优化
2. Phase 2 激活时，NDPN 输出 `f_enc`、MCPN 输出随机值，StageBlock 收到从未见过的非零输入
3. `Conv→GELU→Conv(非零输入)` 产生任意输出，直接叠加到 `img_s1/img_s2` 上，摧毁 Phase 1 学到的光照补偿

这本质上是**分阶段训练中的分布漂移（distribution shift）问题**。参考文献支持：

- **Progressive Training in GAN**（Karras et al., ICLR 2018）：渐进式增长网络在添加新层时使用 fade-in（α从0线性增长到1），确保新层初始输出为零，不扰动已有训练结果
- **Lottery Ticket Hypothesis 的 rewinding**（Frankle et al., 2019）：重置权重到早期状态以保持训练轨迹兼容
- **Adapter/LoRA 初始化**（Hu et al., 2022）：LoRA 的 B 矩阵初始化为零，确保添加 adapter 时初始输出不变

核心原则一致：**新激活模块的初始输出必须等效于激活前的状态（零张量），然后渐进释放。**

### 意见2：简化 TFDE 和 ISPN

**合理，理由充分：**

**TFDE 简化理由：**
- 当前 FrequencyBranch 包含 LFF（可学习频率滤波）、DWT-LFF adapter、相位置信度估计等多个复杂子模块，参数量和计算图深度过大
- 频域分支的 phase_conf 调制 s_noise 的逻辑（`s_noise *= 1 - 0.3*(1-phase_conf)`）引入了难以调试的耦合
- 在低光条件下，频域特征本身就被噪声严重污染（信噪比极低时频谱几乎被噪声主导），频域分支的有效信息有限
- **多尺度空洞卷积**是更稳定的替代方案：3×3 捕获局部纹理/噪声统计，5×5（或 dilation=2 的 3×3）捕获中尺度光照渐变，两者拼接后融合，既保留多尺度感受野又避免频域变换的训练不稳定性

**ISPN 简化理由：**
- 当前 ISPN 使用 pairwise cosine logits + softmax attention 做多帧光照对齐，这在 Phase 1 warmup 阶段就引入了复杂的注意力计算
- 光照增强的物理模型本质上就是 **Retinex**：`I = R × L`，恢复就是 `R = I / L`，即乘法操作
- 加一个加法修正项处理光照相关的系统偏差（如暗电流、色偏）是物理合理的
- 输出为 `(gain_map, bias_map)` → `enhanced = input × gain + bias`，形式简洁，梯度路径清晰

---

## 二、具体改进方案

### 方案 A：无叠加启动的阶段解冻机制

**核心思想：不仅要保证 NDPN/MCPN 的输出为零，还要保证 SGRF StageBlock 在收到非零输入时也能 pass-through。**

```python
# ============================================================
# 方案 A: SGRF StageBlock 零初始化 + MCPN 静默启动
# ============================================================

# --- A1: SGRF StageBlock 增加残差缩放门 ---
class StageBlock(nn.Module):
    """
    改进: 添加 learnable zero-init gate，确保 Phase 2 启动时
    StageBlock(非零输入) ≈ 0，不扰动 Phase 1 的输出。
    
    物理含义: gate 从 0 渐进学习到有效值，
    等效于 LoRA 的 B=0 初始化策略。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        # 零初始化门：初始时 StageBlock 输出 = 0
        self.gate = nn.Parameter(torch.zeros(1))  
        
    def forward(self, x_main: torch.Tensor, x_guide: torch.Tensor) -> torch.Tensor:
        """
        x_main: 主路径（img_s1 或 img_s2）
        x_guide: 来自 NDPN/MCPN 的引导特征
        """
        h = self.conv1(x_guide)
        h = self.act(h)
        h = self.conv2(h)
        # gate 初始=0 → 输出=0 → x_main 不受扰动
        return x_main + h * self.gate


# --- A2: MCPN 静默启动 ---
class MCPN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # ... 原有结构 ...
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        # 静默启动: refine 最后一层零初始化
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        # gamma 初始为 0（与 NDPN 一致）
        self.gamma = nn.Parameter(torch.zeros(1))
        # startup_bias: 初始=5 → sigmoid(5)≈0.993 → g_t≈1 → 纯中心帧
        self.startup_bias = nn.Parameter(torch.tensor(5.0))
        
    def forward(self, F_aligned_list, center_idx, **kwargs):
        f_center = F_aligned_list[center_idx]
        # gate 倾向于 1 → 输出 ≈ f_center
        # startup_bias 初始=5, 可学习下降
        g_t = torch.sigmoid(self.raw_gate + self.startup_bias)
        f_fused = g_t * f_center + (1 - g_t) * f_neighbor_agg
        
        delta = self.refine(f_fused) * self.gamma  # gamma=0 → delta=0
        f_motion_out = f_center + delta  # ≈ f_center
        
        return {"f_motion_out": f_motion_out, "G_t": g_t}


# --- A3: NDPN 确认已有的静默机制 (无需改动) ---
# gamma=0, noise_proj 零初始化 → f_noise_out = f_enc
# 但需在 SGRF-S1 的 StageBlock 也加 zero-gate (A1)
```

**阶段解冻时序表：**

| Epoch | Phase | NDPN 输出 | MCPN 输出 | SGRF StageBlock gate | 等效行为 |
|-------|-------|-----------|-----------|---------------------|----------|
| 0-19 | Phase 1 | 强制 0 | 强制 0 | gate=0（不重要） | 纯光照学习 |
| 20-29 | Phase 1.5 | unlock×(≈0) | unlock×(≈f_center) | gate≈0 → 渐进学习 | 三个零初始化同时渐进释放 |
| 30+ | Phase 2 | 正常输出 | 正常输出 | gate 已学到有效值 | 全功能协同 |

关键改进：**三重保险**——NDPN 零输出 + MCPN 零增量 + SGRF StageBlock 零门控，任何一层都能独立保证 Phase 2 启动不崩溃。

---

### 方案 B：简化 TFDE

**设计思路：移除频域分支，用多尺度空洞卷积替代，保持双源强度估计（s_illum, s_noise）的输出接口不变。**

```python
class MultiScaleSpatialBranch(nn.Module):
    """
    替代原 FrequencyBranch + SpatialBranch 的统一空域分支。
    
    设计依据:
    - 3×3 conv: 局部纹理/噪声粒度统计 (1-pixel 邻域)
    - 3×3 dilated conv (d=2): 中尺度光照渐变 (5-pixel 有效感受野)
    - 3×3 dilated conv (d=4): 大尺度光照区域 (9-pixel 有效感受野)
    
    多尺度拼接后融合，替代频域分析的多频带分解功能。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = out_channels // 3  # 每个尺度分配 1/3 通道
        
        self.branch_local = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, 1, 1),
            nn.GELU(),
        )
        self.branch_mid = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, 1, padding=2, dilation=2),
            nn.GELU(),
        )
        self.branch_wide = nn.Sequential(
            nn.Conv2d(in_channels, out_channels - 2 * mid, 3, 1, padding=4, dilation=4),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_local = self.branch_local(x)
        f_mid = self.branch_mid(x)
        f_wide = self.branch_wide(x)
        return self.fuse(torch.cat([f_local, f_mid, f_wide], dim=1))


class TFDE_v2(nn.Module):
    """
    简化版 TFDE：纯空域，移除 FrequencyBranch/LFF/phase_conf。
    
    数据流:
        feats (B,T,C,H,W)
            → 时域统计量: μ_t, σ_t, SNR  (沿 T 维计算)
            → 拼接 [μ_t, σ_t, SNR] (B, 3C, H, W)
            → MultiScaleSpatialBranch → F_fused (B, C_f, H, W)
            → IntensityHead → s_illum, s_noise (B, 1, H, W)
    
    参数量对比:
        原 TFDE: SpatialBranch + FrequencyBranch + LFF + ConcatFusion 
                 + IntensityHead + phase_conf ≈ 大
        新 TFDE_v2: 时域统计 + MultiScaleSpatial + 1×1 Head ≈ 小
    """
    def __init__(self, channels: int = 64, fused_channels: int = 64, eps: float = 1e-6,
                 use_soft_median: bool = True, soft_median_tau: float = 0.1):
        super().__init__()
        self.eps = eps
        self.use_soft_median = use_soft_median
        self.soft_median_tau = soft_median_tau
        
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        
        # 时域统计量 → 多尺度空域特征
        self.ms_branch = MultiScaleSpatialBranch(
            in_channels=channels * 3,  # [μ_t, σ_t, SNR]
            out_channels=fused_channels,
        )
        
        # 双源强度头 (与原接口兼容)
        self.head = nn.Conv2d(fused_channels, 2, 1)  # 2 通道: s_illum, s_noise
        
    @staticmethod
    def _soft_median(x: torch.Tensor, dim: int = 1, tau: float = 0.1) -> torch.Tensor:
        with torch.no_grad():
            med = x.median(dim=dim).values.unsqueeze(dim)
        dist = (x - med).abs()
        weights = F.softmax(-dist / tau, dim=dim)
        return (weights * x).sum(dim=dim)
        
    def forward(self, feats: torch.Tensor) -> dict:
        """
        Args:
            feats: (B, T, C, H, W)
        Returns:
            兼容原 TFDE 输出接口的 dict
        """
        B, T, C, H, W = feats.shape
        
        # 逐帧 GroupNorm
        feats_norm = self.norm(feats.reshape(B * T, C, H, W)).reshape(B, T, C, H, W)
        
        # 时域统计量
        if self.use_soft_median:
            mu_t = self._soft_median(feats_norm, dim=1, tau=self.soft_median_tau)
        else:
            mu_t = feats_norm.median(dim=1).values
            
        sigma_t_sq = feats_norm.var(dim=1, unbiased=False)
        sigma_t = torch.sqrt(sigma_t_sq + self.eps)
        snr = mu_t / (sigma_t + self.eps)
        
        # 拼接 → 多尺度空域卷积
        stats = torch.cat([mu_t, sigma_t, snr], dim=1)  # (B, 3C, H, W)
        f_fused = self.ms_branch(stats)                   # (B, C_f, H, W)
        
        # 双源强度
        raw = torch.sigmoid(self.head(f_fused))
        s_illum = raw[:, 0:1]
        s_noise = raw[:, 1:2]
        
        return {
            "F_fused": f_fused,
            "F_s": f_fused,      # 兼容旧接口
            "F_f": f_fused,      # 兼容旧接口（无频域分支）
            "mu_t": mu_t,
            "sigma_t": sigma_t,
            "snr": snr,
            "s_illum": s_illum,
            "s_noise": s_noise,
        }
```

**简化收益：**
- 移除 LFFFeatureAdapter、SpatialDWTLFFAdapter、phase_conf_head 等模块
- 移除 ConcatFusion（只有一个分支，无需融合）
- 梯度路径从 `feats → LFF → DWT → phase_conf → s_noise 调制` 简化为 `feats → 统计量 → Conv → head`
- 可以移除 SWD 对 TFDE 的 DWT 分流（TFDE 直接在原始分辨率特征上工作）

---

### 方案 C：简化 ISPN

**设计思路：输出 `(gain_map, bias_map)`，增强公式为 `enhanced = input × gain + bias`。**

```python
class ISPN_v2(nn.Module):
    """
    简化版 ISPN：Retinex-inspired 光照增强。
    
    物理模型:
        低光图像 I = R × L + n_dark  (Retinex + 暗电流噪声)
        增强目标: I_enhanced = I × G + B
        
        其中:
        - G (gain_map): 乘法提亮，≈ 1/L_estimated，补偿光照不足
        - B (bias_map): 加法修正，补偿暗电流、色偏等光照相关系统偏差
    
    输入:
        f_enc_center: (B, C, H, W) 中心帧编码器特征
        s_illum:      (B, 1, H, W) TFDE 估计的光照强度图
        
    输出:
        gain_map: (B, 1, H, W) 乘法增益，范围 [1, max_gain]
        bias_map: (B, 3, H, W) 加法偏置，范围 [-bias_range, +bias_range]
    
    与 SGRF 的接口:
        SGRF 亮度增强阶段: img_enhanced = img_deblurred × gain_map + bias_map
    """
    def __init__(self, channels: int = 64, img_channels: int = 3,
                 max_gain: float = 10.0, bias_range: float = 0.1):
        super().__init__()
        self.max_gain = max_gain
        self.bias_range = bias_range
        
        # 轻量特征精炼 (2层 ResBlock 足够)
        self.refine = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, 1, 1),  # +1 for s_illum
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
        )
        
        # Gain head: 输出乘法增益
        # 初始化使 gain ≈ 1 (不改变亮度)
        self.gain_head = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
        )
        # 零初始化 → sigmoid(0)=0.5 → gain = 1 + 0.5*(max-1) ≈ 中间值
        # 改用: 输出 log-gain, 初始=0 → gain=exp(0)=1
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.zeros_(self.gain_head[-1].bias)
        
        # Bias head: 输出加法偏置 (3通道, RGB 独立)
        # 零初始化 → bias=0 (不修正)
        self.bias_head = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, img_channels, 1),
        )
        nn.init.zeros_(self.bias_head[-1].weight)
        nn.init.zeros_(self.bias_head[-1].bias)
        
    def forward(self, f_enc_center: torch.Tensor, s_illum: torch.Tensor) -> dict:
        """
        Args:
            f_enc_center: (B, C, H, W) 中心帧编码器特征
            s_illum:      (B, 1, H, W) 光照强度先验
            
        Returns:
            gain_map: (B, 1, H, W) 乘法增益 ∈ [1, max_gain]
            bias_map: (B, 3, H, W) 加法偏置 ∈ [-bias_range, +bias_range]
            f_illum_feat: (B, C, H, W) 光照特征 (供 SGRF 使用)
        """
        # 拼接特征和光照先验
        h = self.refine(torch.cat([f_enc_center, s_illum], dim=1))
        
        # Gain: exp(·) 确保正值, clamp 防止过大
        # 初始: log_gain=0 → gain=1 (恒等映射, Phase 1 友好)
        log_gain = self.gain_head(h)
        gain_map = torch.exp(log_gain).clamp(1.0, self.max_gain)
        
        # Bias: tanh 限制范围, 初始=0 (无偏置)
        bias_map = torch.tanh(self.bias_head(h)) * self.bias_range
        
        return {
            "gain_map": gain_map,    # (B, 1, H, W)
            "bias_map": bias_map,    # (B, 3, H, W)
            "f_illum_feat": h,       # (B, C, H, W) 供 SGRF 残差路径
        }
```

**对应的 SGRF 修改：**

```python
class SGRF_v2(nn.Module):
    """
    简化版 SGRF: denoise → deblur → brighten
    
    Stage 1 (Denoise):  img_s1 = image_center + StageBlock_1(f_noise_out)
    Stage 2 (Deblur):   img_s2 = img_s1 + StageBlock_2(f_motion_out)  
    Stage 3 (Brighten): res_t  = img_s2 × gain_map + bias_map
    
    StageBlock 使用 zero-gate 初始化 (方案 A)。
    """
    def __init__(self, channels: int, out_channels: int = 3):
        super().__init__()
        
        # Stage 1: Denoise
        self.stage1 = StageBlock(channels)  # 方案A的零门控版本
        self.to_img1 = nn.Conv2d(channels, out_channels, 3, 1, 1)
        
        # Stage 2: Deblur  
        self.stage2 = StageBlock(channels)
        self.to_img2 = nn.Conv2d(channels, out_channels, 3, 1, 1)
        
        # Stage 3: Brighten (直接用 ISPN 的 gain/bias, 无需额外参数)
        # 可选: 一个轻量残差修正
        self.final_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        nn.init.zeros_(self.final_refine[0].weight)
        nn.init.zeros_(self.final_refine[0].bias)
        self.refine_gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, f_noise_out, f_motion_out, gain_map, bias_map, 
                image_center, f_illum_feat):
        # Stage 1: Denoise
        img_s1 = image_center + self.to_img1(self.stage1(f_noise_out))
        
        # Stage 2: Deblur
        img_s2 = img_s1 + self.to_img2(self.stage2(f_motion_out))
        
        # Stage 3: Brighten (Retinex-style)
        img_bright = img_s2 * gain_map + bias_map
        
        # 可选残差修正 (zero-gate)
        res_t = img_bright + self.final_refine(img_bright) * self.refine_gate
        res_t = res_t.clamp(0, 1)
        
        return {
            "res_t": res_t,
            "img_s1": img_s1,
            "img_s2": img_s2,
            "gain_map": gain_map,
            "bias_map": bias_map,
        }
```

---

## 三、整体改进路线图

```
Phase 1 (Epoch 0-19): 只训练 Encoder + TFDE_v2 + ISPN_v2
  ├── NDPN: 强制输出 0
  ├── MCPN: 强制输出 0
  ├── SGRF StageBlock gate = 0 (即使意外激活也安全)
  ├── ISPN_v2: gain≈1, bias≈0 → 渐进学习提亮
  └── Loss: L1 + SSIM (只监督 res_t 的亮度)

Phase 1.5 (Epoch 20-29): 渐进解冻 NDPN/MCPN
  ├── unlock_ratio: 0→1 线性增长 (10 epoch)
  ├── NDPN: gamma=0, 零投影 → 输出≈f_enc × unlock → ≈0
  ├── MCPN: startup_bias=5, gamma=0 → 输出≈f_center × unlock → ≈0
  ├── SGRF StageBlock gate ≈ 0 → 渐进学习
  └── 三重零保证: 分支零 × unlock零 × gate零

Phase 2 (Epoch 30+): 全功能训练
  ├── 所有模块正常工作
  ├── CXG 交叉激励 (unlock_ratio > 0.3 时激活)
  └── 多 Loss 联合: L1 + SSIM + VGG + Charbonnier
```

**预期效果：**
- Phase 1 → Phase 1.5 过渡：PSNR 不会下降（三重零保证）
- Phase 1.5 中：NDPN/MCPN/StageBlock 的 gate 逐渐从 0 学到有效值
- Phase 2：三源协同，PSNR 在 Phase 1 基线上继续提升