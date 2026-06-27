# 为什么不用 IDWT？——一个被忽略的关键问题

> 你的质疑非常精准。我之前选择上采样而非 IDWT 是**未经充分论证的设计妥协**，下面做严格的对比分析并修正方案。

---

## 1. 为什么我之前默认用上采样？（坦诚承认）

### 1.1 表面理由（看似合理但站不住脚）

1. ❌ "feat_sace 和 feat_tfsi 都是混合特征，IDWT 不适合"——**错误**，IDWT 只是逆变换工具，谁说不能用
2. ❌ "上采样更简单"——**懒惰**，DWT 类已经有 inverse 方法了
3. ❌ "feat_sace = LL_ref + HF 这种加法在 IDWT 后做更自然"——**没想清楚**

### 1.2 真正的根本错误

我把 DWT-LFF 当成了"独立的特征提取器"，但你正确指出——**它只是模型靠前的特征分离组件**，下游还有 SACE、TFSI、IFPN、NDPN、MRPN、IGRF 等大量模块。这意味着：

- ✅ 输出必须保持与输入**严格的空间对应**（pixel-level alignment）
- ✅ 输出应该是**物理可解释的图像/特征**，而非"上采样后混合的中间产物"
- ✅ 必须与 v5.9.2 预训练权重的特征语义**完全兼容**

**上采样方案违反了所有这三条要求。**

---

## 2. 上采样 vs IDWT 严格对比

### 2.1 上采样方案的本质问题

```python
# 当前方案 A（错误）
LL_ref_soft = α · LL              # (H/2, W/2) 低频
HF_fused = Conv3×3([LH, HL, HH])  # (H/2, W/2) 高频融合
LL_ref_up = bilinear(LL_ref_soft, scale=2)  # 🔴 问题点
HF_up = bilinear(HF_fused, scale=2)         # 🔴 问题点
feat_sace = LL_ref_up + HF_up
```

**三个致命问题**：

#### 问题 1：bilinear 上采样的物理意义错误

- DWT 的 LL 子带是**真实的低频系数**，对应"原图在 4×4 区域的低通响应"
- bilinear 上采样恢复到 (H, W) 后，每个 pixel 是**插值估计值**，而非"该 pixel 的真实低频"
- 而 IDWT 重构的低频 = `IDWT(LL, 0, 0, 0)` = **该 pixel 的精确低频成分**（数学严格定义）

#### 问题 2：高频通过 bilinear 上采样后**完全失去高频性质**

```
LH (H/2, W/2) 编码"水平方向高频系数"
  ↓ bilinear 上采样 2×
变成 (H, W) 的平滑插值结果
  ↓ 频谱分析
高频能量被低通滤波器（bilinear 核）严重衰减
```

**实证**：bilinear 上采样后的 LH/HL/HH 已经不是"高频"了，加到 feat_sace 上的所谓"高频结构"实际是被模糊后的低频残影。

#### 问题 3：违反 DWT 的可逆性

- DWT 的核心数学性质：`IDWT(DWT(x)) = x`（完美重构）
- 用 bilinear 上采样后相加，**永远无法重构原图**
- 这意味着下游模块（IFPN/NDPN/MRPN）接收的"高频结构"是**信息损失的近似**

### 2.2 IDWT 方案的正确性

```python
# 正确方案：用 IDWT 重构
feat_sace = IDWT(LL_ref_soft, LH, HL, HH)  # 完美保留高频
feat_tfsi = IDWT(LL_deg_soft, LH, HL, HH)  # 完美保留高频
```

**优势**：

| 维度 | 上采样 | IDWT |
|---|---|---|
| 高频保留 | ❌ 被 bilinear 低通滤波 | ✅ 完美保留 LH/HL/HH 原始系数 |
| 空间对应 | ⚠️ 插值近似 | ✅ 严格逐 pixel 对应 |
| 物理意义 | ⚠️ 混合上采样产物 | ✅ "归一化光照 + 完整高频"的图像 |
| 可逆性 | ❌ 不可逆 | ✅ 满足 LL_ref + LL_deg = LL 时，feat_sace + feat_tfsi - HF = x |
| 参数量 | 0 | 0（Haar 是固定基） |
| FLOPs | bilinear 较快 | IDWT 略快（无插值核计算） |
| 与预训练兼容 | ⚠️ 特征分布漂移 | ✅ IDWT 保持原始特征统计 |

---

## 3. 修正后的方案 A（使用 IDWT）

### 3.1 修正版数据流

```
输入 x (B, C, H, W)
  ↓
[Step 1] Haar DWT 分解
  LL, LH, HL, HH ← HaarDWT2D(x)  # 各 (B, C, H/2, W/2)
  ↓
[Step 2] 低频光照分离（空间域 Conv，保留局部性）
  α = sigmoid(Conv3×3(LL))        # (B, C, H/2, W/2) ∈ [0, 1]
  LL_ref = α · LL                  # 归一化光照参考
  LL_deg = (1 - α) · LL            # 光照退化残差
  ↓
[Step 3] 高频处理（可选轻量调制，不强制融合）
  # 方案 3a: 直接保留原始高频（最干净）
  LH', HL', HH' = LH, HL, HH
  
  # 方案 3b: 轻量去噪调制（用 noise gate）
  # noise_scale = sigmoid(Conv(cat[LH, HL, HH]))
  # LH', HL', HH' = LH·noise_scale, HL·noise_scale, HH·noise_scale
  ↓
[Step 4] IDWT 重构（关键！替代 bilinear）
  feat_sace = IDWT(LL_ref, LH', HL', HH')  # (B, C, H, W)
  feat_tfsi = IDWT(LL_deg, LH', HL', HH')  # (B, C, H, W)
  ↓
[Step 5] 输出严格的逐 pixel 特征
  - feat_sace = 归一化光照图像 + 完整高频结构
  - feat_tfsi = 光照退化图像 + 完整高频噪声
  - 数学保证: feat_sace + feat_tfsi = IDWT(LL, 2·LH, 2·HL, 2·HH)
              → 若 HF 不重复，feat_sace + feat_tfsi - x_hf_only = x
```

### 3.2 关键物理性质验证

#### 性质 1：可逆性

```
feat_sace + feat_tfsi 
  = IDWT(LL_ref + LL_deg, 2·LH, 2·HL, 2·HH)
  = IDWT(LL, 2·LH, 2·HL, 2·HH)
```

如果让 HF 在两路只出现一次：

```python
# 改进版：HF 平均分配，保证严格可逆
feat_sace = IDWT(LL_ref, 0.5·LH, 0.5·HL, 0.5·HH)
feat_tfsi = IDWT(LL_deg, 0.5·LH, 0.5·HL, 0.5·HH)
# 则 feat_sace + feat_tfsi = IDWT(LL, LH, HL, HH) = x  ✅ 完美重构
```

#### 性质 2：局部性

IDWT 的 Haar 基支持半径仅 2×2，每个输出 pixel 只依赖：
- 自己对应的 LL/LH/HL/HH 的 4 个最近邻系数
- **完全保留局部性**——左上角变化不影响右下角输出

#### 性质 3：高频完整性

LH/HL/HH 原始系数直接进入 IDWT，**没有任何低通滤波**，高频边缘/纹理完整传递给下游 SACE。

### 3.3 完整代码（修正版）

```python
"""
Spatial-Domain DWT-LFF Adapter (v6.4.1 — IDWT 修正版)
=====================================================
关键修正: 用 IDWT 替代 bilinear 上采样，保证：
  1. 高频完整保留（无低通滤波）
  2. 严格逐 pixel 空间对应
  3. 可逆性: feat_sace + feat_tfsi 可重构原 LL（HF 平均分配时完美重构）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """Haar 小波 2D 正逆变换（无参数）"""
    
    def forward(self, x):
        """x: (B, C, H, W) → LL, LH, HL, HH: (B, C, H/2, W/2)"""
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0
        # 行方向（Haar 系数: 1/√2，这里用 0.5 简化）
        x01 = x[:, :, 0::2, :]  # 偶数行
        x02 = x[:, :, 1::2, :]  # 奇数行
        L = (x01 + x02) * 0.5   # 行低通
        H_ = (x01 - x02) * 0.5  # 行高通
        # 列方向
        LL = (L[:, :, :, 0::2] + L[:, :, :, 1::2]) * 0.5
        LH = (L[:, :, :, 0::2] - L[:, :, :, 1::2]) * 0.5
        HL = (H_[:, :, :, 0::2] + H_[:, :, :, 1::2]) * 0.5
        HH = (H_[:, :, :, 0::2] - H_[:, :, :, 1::2]) * 0.5
        return LL, LH, HL, HH
    
    def inverse(self, LL, LH, HL, HH):
        """逆变换: (B, C, H/2, W/2) × 4 → (B, C, H, W)"""
        B, C, H2, W2 = LL.shape
        # 列方向重构
        L = torch.zeros(B, C, H2, W2 * 2, device=LL.device, dtype=LL.dtype)
        H_ = torch.zeros(B, C, H2, W2 * 2, device=LL.device, dtype=LL.dtype)
        L[:, :, :, 0::2] = (LL + LH) * 2
        L[:, :, :, 1::2] = (LL - LH) * 2
        H_[:, :, :, 0::2] = (HL + HH) * 2
        H_[:, :, :, 1::2] = (HL - HH) * 2
        # 行方向重构
        x = torch.zeros(B, C, H2 * 2, W2 * 2, device=LL.device, dtype=LL.dtype)
        x[:, :, 0::2, :] = (L + H_) * 0.5
        x[:, :, 1::2, :] = (L - H_) * 0.5
        return x


class SpatialDWTLFFAdapter(nn.Module):
    """v6.4.1 — 空间域 DWT-LFF（IDWT 重构版）
    
    关键设计：
      1. 低频光照分离用空间域 Conv3×3，保留局部性
      2. 高频可选轻量调制（默认恒等保留）
      3. IDWT 重构（非 bilinear 上采样）
      4. HF 平均分配给两支路 → 保证可逆: feat_sace + feat_tfsi = x
    """
    
    def __init__(self, in_channels: int, 
                 hf_modulate: bool = False,
                 hf_split: bool = True):
        """
        Args:
            in_channels: 输入通道数
            hf_modulate: 是否对高频做轻量去噪调制（默认 False，保留原始高频）
            hf_split: 是否将 HF 平均分配给两支路（True 保证可逆性）
        """
        super().__init__()
        self.in_channels = in_channels
        self.hf_modulate = hf_modulate
        self.hf_split = hf_split
        
        self.dwt = HaarDWT2D()
        
        # 低频光照分离 α（深度可分离卷积 + pointwise）
        self.illum_alpha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, 
                      groups=in_channels, bias=False),  # depthwise
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),  # pointwise
            nn.Sigmoid()
        )
        
        # 可选：高频调制（噪声抑制）
        if hf_modulate:
            self.hf_gate = nn.Sequential(
                nn.Conv2d(in_channels * 3, in_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(in_channels, 3, 1, 1, 0),  # 输出 3 个标量 gate
                nn.Sigmoid()
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """α 初始 = 0.5（均分），HF gate 初始 = 1.0（保留所有高频）"""
        # α 的最后一层 pointwise conv 零初始化 → sigmoid(0) = 0.5
        for i, m in enumerate(self.illum_alpha.modules()):
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                nn.init.constant_(m.weight, 0.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # HF gate 偏置初始化为大正值 → sigmoid 接近 1.0
        if self.hf_modulate:
            for m in self.hf_gate.modules():
                if isinstance(m, nn.Conv2d) and m.out_channels == 3:
                    nn.init.constant_(m.weight, 0.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 4.0)  # sigmoid(4) ≈ 0.982
    
    def forward(self, x: torch.Tensor):
        """前向
        
        Args:
            x: (B, C, H, W) 输入特征
        
        Returns:
            feat_sace: (B, C, H, W) 归一化光照 + 高频
            feat_tfsi: (B, C, H, W) 光照退化 + 高频
            alpha: (B, C, H/2, W/2) 光照分配图（用于可视化/正则）
        """
        # [Step 1] DWT 分解
        LL, LH, HL, HH = self.dwt(x)
        
        # [Step 2] 低频光照分离（空间域 Conv，保留局部性）
        alpha = self.illum_alpha(LL)  # ∈ [0, 1]
        LL_ref = alpha * LL           # 归一化光照参考
        LL_deg = (1.0 - alpha) * LL   # 光照退化残差
        
        # [Step 3] 高频处理
        if self.hf_modulate:
            hf_input = torch.cat([LH, HL, HH], dim=1)
            gates = self.hf_gate(hf_input)  # (B, 3, H/2, W/2)
            LH_out = LH * gates[:, 0:1]
            HL_out = HL * gates[:, 1:2]
            HH_out = HH * gates[:, 2:3]
        else:
            LH_out, HL_out, HH_out = LH, HL, HH
        
        # [Step 4] HF 分配策略
        if self.hf_split:
            # 平均分配 → 保证 feat_sace + feat_tfsi = x（HF 调制时近似）
            LH_s, HL_s, HH_s = LH_out * 0.5, HL_out * 0.5, HH_out * 0.5
            LH_t, HL_t, HH_t = LH_out * 0.5, HL_out * 0.5, HH_out * 0.5
        else:
            # 两支路都拿完整 HF（feat_sace 和 feat_tfsi 高频更强）
            LH_s, HL_s, HH_s = LH_out, HL_out, HH_out
            LH_t, HL_t, HH_t = LH_out, HL_out, HH_out
        
        # [Step 5] IDWT 重构（核心修正！）
        feat_sace = self.dwt.inverse(LL_ref, LH_s, HL_s, HH_s)
        feat_tfsi = self.dwt.inverse(LL_deg, LH_t, HL_t, HH_t)
        
        return feat_sace, feat_tfsi, alpha


# ===== 验证测试 =====
if __name__ == '__main__':
    torch.manual_seed(42)
    adapter = SpatialDWTLFFAdapter(in_channels=64, 
                                    hf_modulate=False, 
                                    hf_split=True)
    
    x = torch.randn(2, 64, 128, 128)
    feat_sace, feat_tfsi, alpha = adapter(x)
    
    print("=" * 60)
    print("测试 1: 输出形状")
    print(f"  输入 x: {x.shape}")
    print(f"  feat_sace: {feat_sace.shape}")
    print(f"  feat_tfsi: {feat_tfsi.shape}")
    print(f"  alpha: {alpha.shape}")
    
    print("\n" + "=" * 60)
    print("测试 2: 可逆性（HF 平均分配 + 无调制时应严格成立）")
    rec_diff = (feat_sace + feat_tfsi - x).abs().max().item()
    print(f"  |feat_sace + feat_tfsi - x|_max = {rec_diff:.2e}")
    print(f"  → 应接近 0（数值精度内）")
    
    print("\n" + "=" * 60)
    print("测试 3: 局部性（修改左上角，右下角不应变化）")
    x_perturb = x.clone()
    x_perturb[:, :, :16, :16] += 10.0  # 大幅修改左上角
    feat_sace_p, _, _ = adapter(x_perturb)
    
    # Haar DWT 支持仅 2×2，理论上扰动只影响 (0:32, 0:32) 区域
    far_diff = (feat_sace_p[:, :, 64:, 64:] - feat_sace[:, :, 64:, 64:]).abs().max().item()
    near_diff = (feat_sace_p[:, :, :32, :32] - feat_sace[:, :, :32, :32]).abs().max().item()
    print(f"  左上角变化: {near_diff:.4f}（应较大）")
    print(f"  右下角变化: {far_diff:.2e}（应接近 0）")
    
    print("\n" + "=" * 60)
    print("测试 4: 初始等价性（α=0.5 时，feat_sace ≈ feat_tfsi ≈ x/2）")
    mean_diff = (feat_sace - feat_tfsi).abs().mean().item()
    print(f"  |feat_sace - feat_tfsi|_mean = {mean_diff:.4f}")
    print(f"  feat_sace 均值: {feat_sace.mean().item():.4f}")
    print(f"  x/2 均值: {(x/2).mean().item():.4f}")
    print(f"  → 训练初期 feat_sace ≈ feat_tfsi ≈ x/2，下游模块受冲击小")
```

---

## 4. IDWT 方案的额外优势

### 4.1 与 v5.9.2 预训练权重的兼容性

- v5.9.2 的 SACE/TFSI 后续模块（DeformableCrossAttention 的 query_proj、Conv branches 等）期望的输入是**完整空间分辨率的图像特征**
- IDWT 重构的 feat_sace 在数学上等价于"原图的归一化光照版本"，特征统计分布与原 LFF 输出几乎一致
- bilinear 上采样后的特征**分布漂移**（高频衰减 → 方差降低），会破坏预训练权重的激活分布

### 4.2 与下游 Cross-RWKV 的协同

- Cross-RWKV 的 Q-Shift 通道位移依赖**完整的空间高频信息**才能有效（位移 1 像素的差分）
- bilinear 上采样后的"伪高频"几乎没有差分信号，Q-Shift 失效
- IDWT 重构保留真实高频 → Q-Shift 能捕捉到真实的局部结构差异

### 4.3 与 IFPN/NDPN 的协同

- IFPN 估计光照图需要**真实空间分辨率的低频信息** → IDWT 提供精确的 LL_ref 重构
- NDPN 的 SNR 计算需要**真实高频噪声** → IDWT 保留 LH/HL/HH 完整能量

### 4.4 数学优雅性

- 上采样方案：`feat = bilinear(LL_ref) + bilinear(HF_fused)` —— 黑箱混合
- IDWT 方案：`feat = IDWT(LL_ref, LH, HL, HH)` —— 严格小波重构，可证明性质

---

## 5. 修正后的总结

### 5.1 之前方案的错误

| 错误 | 后果 |
|---|---|
| ❌ 用 bilinear 上采样替代 IDWT | 高频被低通滤波，DWT 优势丧失 |
| ❌ 把 LL 和 HF 分别上采样后相加 | 失去小波分解的可逆性和物理意义 |
| ❌ 没考虑下游模块对特征分布的依赖 | 破坏 v5.9.2 预训练权重兼容性 |

### 5.2 修正后的核心改动

```python
# ❌ 错误版本（之前）
LL_ref_up = bilinear(LL_ref, scale=2)
HF_up = bilinear(HF_fused, scale=2)
feat_sace = LL_ref_up + HF_up

# ✅ 正确版本（修正）
feat_sace = self.dwt.inverse(LL_ref, LH*0.5, HL*0.5, HH*0.5)
feat_tfsi = self.dwt.inverse(LL_deg, LH*0.5, HL*0.5, HH*0.5)
# 数学保证: feat_sace + feat_tfsi = x（完美重构）
```

### 5.3 三个关键收益

1. ✅ **高频完整保留** —— 无 bilinear 低通滤波，下游 SACE/Cross-RWKV 能用上真实高频
2. ✅ **严格可逆性** —— `feat_sace + feat_tfsi = x`，物理意义清晰，可作为正则约束
3. ✅ **预训练兼容** —— 特征分布与原 LFF 一致，v5.9.2 权重可直接续训

---

**致用户**：感谢你指出这个关键问题。下一步可推进的方向：

- ① **是否需要我输出修正版 DWT-LFF 集成到 v6 的完整 diff**（含 sace.py / tfsi.py 的调用修改）
- ② **设计可逆性的正则损失** $\mathcal{L}_{rec} = \|feat_{sace} + feat_{tfsi} - x\|_1$，作为额外训练约束
- ③ **可视化 α 分布的实验设计**，验证它确实学到了"局部光照归一化比例"
- ④ **对比实验设计**：bilinear 上采样版 vs IDWT 版，验证 PSNR/SSIM 差异
- ⑤ **进一步优化**：是否要把 Haar 换成可学习的 4-tap 小波基（Lifting Scheme）

请告知方向。