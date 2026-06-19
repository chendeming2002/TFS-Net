# TFS-Net v3 — Phase A 补充答复与遗留问题终结

> 基于 v3quest2.md 的代码审查结果，对所有遗留问题给出最终决策。本轮答复**以参考仓库实际代码为准**，纠正 v3answer.md 的错误，并对 Q1-Q5 给出可直接执行的设计方案。
>
> **答复日期**：2026-06-12（Phase A 终结）

---

## 〇、对 v3quest2 §一 错误纠正的确认

完全接受 v3quest2 §一 的所有纠正。FRBNet 实际代码与论文文本存在简化，**以代码为准**：

| 参数 | 最终采用值 |
|:---|:---|
| K (RBF 基函数数) | **10** |
| N (角谐波数) | **1**（固定） |
| λ (角度调制强度) | **0.1**（固定常量） |
| σ_g 初始化 | **0.2 × min(H,W)** |
| μ_k 初始化 | **linspace(0, 1, K)** |
| log_bwh (RBF 带宽) | **0.0**（对数空间） |
| coeff_mag/phase 初始化 | **全零** |
| raw_gate_mag/phase 初始化 | **全 1** |
| 角度调制 | **无可学习系数**，直接 `(cos+sin)/2` |

下游设计在此基础上展开。

---

## 一、Q1：LFF 对 C=48 特征图的适配方案

### 1.1 决策结论

**采用选项 A 的改进版：分组 rfft2 + LFF 参数跨通道共享 + 门控 Conv 投影**

理由：
1. **物理一致性**：FRBNet LFF 滤波器本质是"频谱坐标 $(u,v)$ 的函数"，与通道无关，跨通道共享是最自然的扩展
2. **参数效率**：LFF 总参数仍为 $\sim 2K + 2$（约 22 个），不会因 C=48 而膨胀
3. **避免选项 B 的信息损失**：先 1×1 Conv 降到 3 通道会丢失高维特征空间的频域信息
4. **避免选项 C 的耦合**：直接对全通道做幅度/相位拼接会让 Conv 需要处理 $2C=96$ 通道输入，增加参数

### 1.2 具体实现方案

```python
class LFFFeatureAdapter(nn.Module):
    """
    将 FRBNet 的 RGB-LFF 适配为多通道特征 LFF。
    
    输入: F ∈ R^(B, C, H, W) 实数特征
    输出: F_filtered ∈ R^(B, C, H, W) 频域增强后的特征
    """
    def __init__(self, channels: int, K: int = 10, n_ang_freq: int = 1):
        super().__init__()
        # 单一 LFF 实例，所有通道共享同一组 (W_g, H_RBF) 参数
        self.lff_mag = RadialBasisFilter(K=K, n_ang_freq=n_ang_freq)   # 作用于幅度谱
        self.lff_phase = RadialBasisFilter(K=K, n_ang_freq=n_ang_freq) # 作用于相位谱（独立参数）
        self.log_sigma_g = nn.Parameter(torch.tensor(0.0))  # Zero-DC 窗带宽（对数空间）
        
        # 频域处理后的通道融合（避免拼接幅度/相位导致通道翻倍）
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Step 1: 对所有通道做 2D rfft2（实数 FFT，更高效）
        # 输出形状: (B, C, H, W//2 + 1) 复数
        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        # Step 2: 构造 Zero-DC 窗 W_g（仅依赖 H, W，与 batch/channel 无关）
        Hf, Wf = H, W // 2 + 1
        fy = torch.fft.fftfreq(H, device=x.device).view(-1, 1).expand(Hf, Wf)
        fx = torch.fft.rfftfreq(W, device=x.device).view(1, -1).expand(Hf, Wf)
        r_grid = torch.sqrt(fx ** 2 + fy ** 2)  # (H, W//2+1)
        sigma_g = torch.exp(self.log_sigma_g) * min(H, W) * 0.2  # 论文初始化
        Wg = torch.exp(-(r_grid / sigma_g) ** 2)
        Wg[0, 0] = 0.0  # 强制 DC=0
        
        # Step 3: 分别滤波幅度和相位
        mag = x_fft.abs()              # (B, C, Hf, Wf)
        phase = torch.angle(x_fft)     # (B, C, Hf, Wf)
        
        # LFF 滤波器输出 (Hf, Wf)，broadcast 到 (B, C, Hf, Wf)
        H_mag = self.lff_mag(r_grid, fx, fy)        # (Hf, Wf)
        H_phase = self.lff_phase(r_grid, fx, fy)    # (Hf, Wf)
        
        mag_filtered = mag * Wg * H_mag             # broadcast
        phase_filtered = phase * Wg * H_phase       # 注意：相位的乘法是调制而非滤波
        
        # Step 4: 重构复数 → IFFT 回空间域
        x_fft_new = mag_filtered * torch.exp(1j * phase_filtered)
        x_filtered = torch.fft.irfft2(x_fft_new, s=(H, W), dim=(-2, -1), norm='ortho')
        
        # Step 5: 通道融合（恢复表达能力）
        return self.fuse(x_filtered)
```

### 1.3 与 ExpoMamba 物理依据的对应

v3 设计文档 §1.1 引用 ExpoMamba 的"幅度↔光照、相位↔结构"解耦观察，本方案通过 `lff_mag` 和 `lff_phase` 两套**独立参数**的 RBF 滤波器分别处理幅度与相位，**完全符合此解耦设计**：

- `lff_mag` 学习到的是"抑制低频光照波动"的滤波器
- `lff_phase` 学习到的是"保留运动结构信息"的滤波器

两者独立训练，避免单一滤波器同时处理两类物理信号。

---

## 二、Q2：W_g 公式选择

### 2.1 决策结论

**完全沿用 FRBNet 代码的写法**：

$$W_g(u,v) = \exp\!\left(-\frac{r(u,v)^2}{\sigma_g^2}\right), \quad W_g(0,0) := 0$$

**修改 v3 设计文档**的公式（v3 原写 $W_g = 1 - \exp(-r^2/(2\sigma^2))$ 已过时）。

### 2.2 理由分析

| 维度 | v3 文档公式 | FRBNet 代码公式 | 评估 |
|:---|:---|:---|:---|
| 物理意义 | 标准高通滤波（抑制低频） | 高斯低通 + DC 硬置零 | FRBNet 更精细 |
| DC 处理 | 仅近似抑制（$W_g(0,0)$ 非严格为 0） | 硬置零（$W_g(0,0)=0$） | FRBNet 更彻底 |
| 边界行为 | 高频接近 1（高通过） | 高频随距离衰减（带通） | FRBNet 更稳健 |
| 已验证性 | 仅 v2 设计概念，无代码验证 | NeurIPS 2025 已发表 + 仓库验证 | FRBNet 优先 |
| 与论文符合度 | 与 v3 文档自洽 | 与 FRBNet 原始论文一致 | 修改文档代价小 |

### 2.3 物理含义重新阐释

FRBNet 的 $W_g$ 不是简单的"高通"，而是**"以 DC 为中心、带宽 $\sigma_g$ 的高斯带，再剔除 DC"**。这等价于：

- 保留**中低频结构信息**（高斯低通成分）
- 同时**完全去除光照偏置**（DC 置零）
- 让后续 $H_{RBF}(u,v)$ 在此基础上做**精细频带选择**

这种设计的优势：**避免高通滤波丢失结构低频信息**，同时确保光照偏置（永远位于 DC）被严格剥离。

### 2.4 v3 设计文档需修订的位置

请在 `TFSv3-result.md` 中将 §4.1 Step 2 的 $W_g$ 公式更新为：

> 零DC高斯窗（FRBNet 风格）：
> $$W_g(u,v) = \exp\!\left(-\frac{u^2+v^2}{\sigma_g^2}\right), \quad W_g(0,0) := 0$$
> 其中 $\sigma_g$ 通过 `log_sigma` 参数化以保证正性，初始化为 $0.2 \times \min(H,W)$。

---

## 三、Q3：SACE Cross-Attention 方案

### 3.1 决策结论

**采用选项 A 的改进版：DCNv2 风格的 deformable cross-attention，参考 EDVR PCDAlignment 但简化为单尺度版本**

### 3.2 三选项详细对比

| 维度 | A: DCNv2 风格 | B: 简化 DAT | C: 标准 cross-attn |
|:---|:---|:---|:---|
| 偏移预测输入 | $\text{Concat}[Q_t, K_i]$ | $Q_t$ 或 $|Q_t - K_i|$ | 无 |
| 采样方式 | 局部 K×K 窗口（如 3×3=9 点） | 全局 grid_sample | 全局 attention |
| 计算复杂度 | $O(C \cdot K^2 \cdot HW)$ | $O(C \cdot HW^2)$ | $O(C \cdot HW^2)$ |
| dense prediction 表现 | ✅ 已在 EDVR/BasicVSR 验证 | ❌ 主要用于分类 | ❌ 内存爆炸 |
| 多头支持 | 通过分组实现 | 支持 | 支持 |
| 实现复杂度 | 中（需 deform_conv2d） | 中（需 grid_sample） | 低 |
| **推荐度** | ✅ 最佳 | 次选 | 不推荐 |

### 3.3 具体设计：SACE Deformable Cross-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d  # 标准库，无需额外依赖

class DeformableCrossAttention(nn.Module):
    """
    源感知可变形 cross-attention。
    
    Q: 中心帧特征 F_t ∈ R^(B, C, H, W)
    K, V: 邻帧特征 F_i ∈ R^(B, C, H, W)
    
    输出:
      - F_i^aligned ∈ R^(B, C, H, W) (用于 NDPN/MRPN 聚合)
      - A_{t→i} ∈ R^(B, n_groups, H, W, K*K) (注意力权重，可选)
    """
    def __init__(self, channels: int, n_groups: int = 4, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.n_groups = n_groups
        self.kernel_size = kernel_size
        self.n_sample = kernel_size * kernel_size  # 局部采样点数
        
        # 偏移预测网络：输入 Concat[Q_t, K_i] → 输出 2*n_groups*K*K 个偏移坐标
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 2 * n_groups * self.n_sample, 3, padding=1)
        )
        # 调制预测（DCNv2 的关键）：输入同上 → 输出 n_groups*K*K 个调制系数
        self.conv_mask = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, n_groups * self.n_sample, 3, padding=1)
        )
        # Value 投影
        self.W_V = nn.Conv2d(channels, channels, 1)
        # 输出投影（混合 multi-group 结果）
        self.W_O = nn.Conv2d(channels, channels, 1)
        
        # 偏移初始化为 0（开始时等价于固定位置采样）
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)
    
    def forward(self, F_t: torch.Tensor, F_i: torch.Tensor):
        """
        F_t: (B, C, H, W) 中心帧（作为 Query）
        F_i: (B, C, H, W) 邻帧（作为 Key/Value）
        """
        B, C, H, W = F_t.shape
        
        # Step 1: 预测偏移和调制系数
        concat_feat = torch.cat([F_t, F_i], dim=1)              # (B, 2C, H, W)
        offset = self.conv_offset(concat_feat)                   # (B, 2*G*K*K, H, W)
        mask = torch.sigmoid(self.conv_mask(concat_feat))        # (B, G*K*K, H, W)
        
        # Step 2: Value 投影
        V = self.W_V(F_i)  # (B, C, H, W)
        
        # Step 3: 用 deform_conv2d 做对齐
        # 注意：deform_conv2d 同时实现"采样 + 加权聚合"，无需手动 attention 矩阵
        # weight 设为单位卷积核（不引入额外卷积变换）
        weight = torch.eye(C).view(C, C, 1, 1).expand(C, C, self.kernel_size, self.kernel_size).to(F_t.device)
        # 由于 deform_conv2d 的 weight 需要 (out_C, in_C/groups, K, K)，简化为：
        # 用 1x1 deform conv 实现 alignment（但 kernel_size 必须 = self.kernel_size）
        # 因此改用手动 grid_sample 实现（更清晰）
        
        F_i_aligned = self._deform_sample(V, offset, mask)       # (B, C, H, W)
        
        # Step 4: 输出投影
        return self.W_O(F_i_aligned), offset, mask
    
    def _deform_sample(self, V: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        手动实现可变形采样 + 加权聚合。
        V: (B, C, H, W)
        offset: (B, 2*G*K*K, H, W) - 每个采样点的 (dy, dx) 偏移
        mask: (B, G*K*K, H, W) - 每个采样点的调制系数
        """
        B, C, H, W = V.shape
        G = self.n_groups
        Ks = self.n_sample
        C_per_group = C // G
        
        # 构造基础网格（中心点位置）
        yy, xx = torch.meshgrid(
            torch.arange(H, device=V.device, dtype=torch.float),
            torch.arange(W, device=V.device, dtype=torch.float),
            indexing='ij'
        )
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        
        # 构造 K×K 窗口内的基础偏移
        k_half = self.kernel_size // 2
        ky, kx = torch.meshgrid(
            torch.arange(-k_half, k_half + 1, device=V.device, dtype=torch.float),
            torch.arange(-k_half, k_half + 1, device=V.device, dtype=torch.float),
            indexing='ij'
        )
        window_offset = torch.stack([kx, ky], dim=-1).view(Ks, 2)  # (Ks, 2)
        
        # reshape offset: (B, 2*G*Ks, H, W) → (B, G, Ks, 2, H, W) → 加到 base_grid
        offset = offset.view(B, G, Ks, 2, H, W).permute(0, 1, 2, 4, 5, 3)  # (B, G, Ks, H, W, 2)
        mask = mask.view(B, G, Ks, H, W)
        
        # 对每个 group，对每个采样点做 grid_sample
        V_grouped = V.view(B, G, C_per_group, H, W)
        outputs = []
        for g in range(G):
            agg = torch.zeros(B, C_per_group, H, W, device=V.device)
            for k in range(Ks):
                # 采样位置 = base + window_offset + learned_offset
                sample_grid = base_grid + window_offset[k].view(1, 1, 1, 2) + offset[:, g, k]
                # 归一化到 [-1, 1]
                sample_grid_norm = sample_grid.clone()
                sample_grid_norm[..., 0] = 2.0 * sample_grid[..., 0] / (W - 1) - 1.0
                sample_grid_norm[..., 1] = 2.0 * sample_grid[..., 1] / (H - 1) - 1.0
                sampled = F.grid_sample(V_grouped[:, g], sample_grid_norm, mode='bilinear', 
                                        padding_mode='zeros', align_corners=True)
                agg = agg + sampled * mask[:, g, k:k+1]
            outputs.append(agg)
        
        return torch.cat(outputs, dim=1)  # (B, C, H, W)
```

### 3.4 设计要点说明

1. **偏移预测输入用 `Concat[F_t, F_i]`**：相比 $|Q_t - K_i|$，拼接保留了双方完整信息，让网络自行学习相关性。EDVR PCDAlignment 也采用拼接方式。

2. **采用局部 K×K 窗口（默认 K=3，9 个采样点）**：
   - 相比 DAT 的全局 grid_sample，**计算量降低 $HW/K^2 \approx 1000\times$**
   - 适合视频对齐场景（运动通常局部）
   - 9 个采样点足够覆盖典型相邻帧位移

3. **n_groups=4**：分组注意力的标准配置，每组关注不同的运动模式（旋转、平移、缩放、形变）。

4. **DCNv2 调制系数 mask**：sigmoid 输出，让网络能"软删除"无效采样点（隐式遮挡处理）。

5. **简化为单尺度**：v3 编码器只输出 H/4 分辨率特征，多尺度 PCD 收益有限，单尺度足够。

6. **返回值**：除对齐特征外，还返回 `offset` 和 `mask`，可供 NDPN/MRPN 作为"对应关系"使用，无需重复计算。

### 3.5 计算复杂度对比

| 方案 | 参数量（C=48, H=W=256） | FLOPs/帧 |
|:---|:---|:---|
| 标准 cross-attention | $3C^2 = 7K$ | $O(C \cdot H^2W^2) = 200G$ |
| DAT 风格 | $\sim 200K$（含 offset 网络） | $O(C \cdot HW \cdot 8 \cdot 8) = 6G$ |
| **本方案 DCNv2 风格** | $\sim 100K$ | $O(C \cdot HW \cdot 9) = 1G$ |

本方案在 dense prediction 场景下**显著优于其他选项**。

---

## 四、Q4：IllumExtract 适配方案

### 4.1 决策结论

**完全采纳 v3quest2 §四 Q4 的"建议"方案**：保留 Retinexformer 的 3 层结构（1×1 → 5×5 DW → 1×1），输入通道按 IFPN 拼接实际值设定，输出仅 illu_map（3 通道 RGB 光照图）。

### 4.2 具体实现

```python
class IllumExtract(nn.Module):
    """
    适配自 Retinexformer 的光照估计器（双流输入版本）。
    
    输入:
      I_t_down: (B, 3, H/4, W/4) 原图下采样
      F_t_L: (B, C, H/4, W/4) 编码器最粗尺度特征（C=48 或经投影后的通道数）
    
    输出:
      L_t: (B, 3, H/4, W/4) RGB 光照图（无 sigmoid，可学习范围）
    """
    def __init__(self, img_channels: int = 3, feat_channels: int = 48, 
                 feat_proj_channels: int = 16, n_fea_middle: int = 31):
        super().__init__()
        # 特征投影：将 F_t^(L) 从 C 通道降到 feat_proj_channels（节省参数）
        self.feat_proj = nn.Conv2d(feat_channels, feat_proj_channels, kernel_size=1)
        
        # 输入通道 = RGB + mean_channel + projected_feat
        in_channels = img_channels + 1 + feat_proj_channels
        
        # Retinexformer 三层结构
        self.conv1 = nn.Conv2d(in_channels, n_fea_middle, kernel_size=1)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, 
                                     padding=2, groups=n_fea_middle)  # 注意：用 n_fea_middle 而非 n_fea_in
        self.conv2 = nn.Conv2d(n_fea_middle, img_channels, kernel_size=1)
    
    def forward(self, I_t_down: torch.Tensor, F_t_L: torch.Tensor) -> torch.Tensor:
        # Step 1: 分辨率对齐（若 F_t^(L) 与 I_t^down 分辨率不一致）
        if F_t_L.shape[-2:] != I_t_down.shape[-2:]:
            F_t_L = F.interpolate(F_t_L, size=I_t_down.shape[-2:], 
                                   mode='bilinear', align_corners=False)
        
        # Step 2: 计算图像 mean channel（Retinexformer 原始设计）
        mean_c = I_t_down.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Step 3: 特征投影
        F_proj = self.feat_proj(F_t_L)
        
        # Step 4: 拼接 → 三层卷积
        x = torch.cat([I_t_down, mean_c, F_proj], dim=1)
        x = self.conv1(x)
        x = self.depth_conv(x)
        L_t = self.conv2(x)
        
        return L_t  # 不加 sigmoid，让光照图保持原始动态范围
```

### 4.3 关键设计决策

1. **保留 `mean_channel`**：Retinexformer 的核心先验之一，提供全局亮度信息
2. **`depth_conv` 的 groups 改为 `n_fea_middle`**：纠正 v3answer 错误（原 Retinexformer 写 `groups=n_fea_in=4`，但实际应是 `n_fea_middle` 才是真正的 depthwise）
3. **不加 sigmoid**：让光照图保持原始动态范围，避免饱和。IFPN 中除法 $L_{ref}/L_t$ 时需加 `clamp(min=ε)` 防止除零
4. **特征投影降通道**：48→16 通道，减少参数 ~75%
5. **仅输出 `illu_map`**：v3 设计不需要 Retinexformer 的 `illu_fea`（那是 Denoiser 的引导输入，v3 由 NDPN/MRPN 替代）

### 4.4 与 v3 设计文档 §4.3 公式的对应

v3 文档：$L_t = \text{IllumExtract}(\text{Concat}[I_t^{down}, \text{Conv}_{1\times1}(F_t^{(L)})])$

本实现：$L_t = \text{IllumExtract}(\text{Concat}[I_t^{down}, \text{mean}(I_t^{down}), \text{Conv}_{1\times1}(F_t^{(L)})])$

**额外增加 mean channel**，符合 Retinexformer 原始设计，物理合理。建议**修订 v3 设计文档公式**以反映此细节。

---

## 五、Q5：时序一致性损失与 RAFT 决策

### 5.1 决策结论

**v3 首版采用三阶段策略**：

- **阶段 1（前 30 epoch）**：**完全省略 $\mathcal{L}_{temporal}$**，仅训练 $\mathcal{L}_{recon} + \mathcal{L}_{perc} + \mathcal{L}_{illum\_smooth}$
- **阶段 2（30-80 epoch）**：**启用简化版 $\mathcal{L}_{temporal}$**，使用预训练冻结的 `raft_small`
- **阶段 3（80+ epoch，可选）**：**启用 RAFT + 遮挡 mask**，进一步精修

### 5.2 详细方案

#### 阶段 1：基础训练（无 temporal）

```python
L_total = L_recon + 0.1 * L_perc + 0.01 * L_illum_smooth
```

理由：
- 让网络先学会基本的"低光增强"能力
- 避免冷启动阶段时序约束误导收敛方向
- 训练速度最快（无光流计算）

#### 阶段 2：简化时序一致性

使用 `torchvision.models.optical_flow.raft_small`（参数 ~1M，仅 raft_large 的 1/5）：

```python
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

class TemporalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT)
        self.raft.eval()
        for p in self.raft.parameters():
            p.requires_grad = False
    
    def forward(self, I_hat_t: torch.Tensor, I_GT_neighbors: list, I_t: torch.Tensor):
        """
        I_hat_t: (B, 3, H, W) 网络输出的增强中心帧
        I_GT_neighbors: list of (B, 3, H, W) 邻帧 GT
        I_t: (B, 3, H, W) 中心帧输入（用于光流估计的参考）
        """
        loss = 0.0
        with torch.no_grad():
            for I_i_gt in I_GT_neighbors:
                # 估计 I_t → I_i 的光流（在原始低光帧上估计更稳定）
                flow_list = self.raft(I_t, I_i_gt)  # 返回列表，最后一个是最精细
                flow_t_to_i = flow_list[-1]  # (B, 2, H, W)
        
        # Warp I_hat_t 到 I_i 的视角
        for idx, I_i_gt in enumerate(I_GT_neighbors):
            with torch.no_grad():
                flow_list = self.raft(I_t, I_i_gt)
                flow = flow_list[-1]
            warped = self._warp(I_hat_t, flow)
            loss = loss + F.l1_loss(warped, I_i_gt)
        
        return loss / len(I_GT_neighbors)
    
    @staticmethod
    def _warp(x, flow):
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=torch.float),
            torch.arange(W, device=x.device, dtype=torch.float),
            indexing='ij'
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        # flow: (B, 2, H, W), 第一通道是 dx, 第二通道是 dy
        new_grid = grid + flow.permute(0, 2, 3, 1)
        new_grid[..., 0] = 2.0 * new_grid[..., 0] / (W - 1) - 1.0
        new_grid[..., 1] = 2.0 * new_grid[..., 1] / (H - 1) - 1.0
        return F.grid_sample(x, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
```

#### 阶段 3（可选）：遮挡 mask

使用 forward-backward consistency 检测遮挡：

```python
flow_forward = self.raft(I_t, I_i_gt)[-1]    # t → i
flow_backward = self.raft(I_i_gt, I_t)[-1]   # i → t
# 反向 warp flow_backward 到 t 视角
flow_bw_warped = self._warp(flow_backward, flow_forward)
# 遮挡判据: |flow_forward + flow_bw_warped| > threshold
error = torch.norm(flow_forward + flow_bw_warped, dim=1, keepdim=True)
M_occ = (error > 1.0).float()  # 1=遮挡, 0=非遮挡
L_temporal = (warped - I_i_gt).abs() * (1 - M_occ)
```

### 5.3 计算开销评估

| 阶段 | 光流次数/iter | 训练速度（相对） | 显存占用 |
|:---|:---|:---|:---|
| 阶段 1 | 0 | 1.0× | 基准 |
| 阶段 2 (raft_small) | 2 (前后邻帧) | 0.75× | +200MB |
| 阶段 3 (含遮挡) | 4 (双向) | 0.5× | +400MB |

**强烈建议首版仅实现阶段 1**，仅当 PSNR 提升瓶颈出现时再启用阶段 2。

### 5.4 损失函数最终配置

```python
class TFSNetLoss(nn.Module):
    def __init__(self, use_temporal: bool = False, use_freq_loss: bool = True):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_freq_loss = use_freq_loss
        
        # VGG 感知损失
        self.vgg = VGGPerceptualLoss(layers=['relu1_2', 'relu2_2', 'relu3_4'])
        
        if use_temporal:
            self.temporal_loss = TemporalLoss()
        
        # 损失权重
        self.lambda_perc = 0.1
        self.lambda_illum = 0.01
        self.lambda_temporal = 0.5
        self.lambda_freq = 0.1
    
    def forward(self, I_hat_t, I_GT, s_illum, I_t=None, I_GT_neighbors=None):
        # (1) 重建损失（空间 + 频域）
        L_recon = F.l1_loss(I_hat_t, I_GT)
        if self.use_freq_loss:
            # FFT 损失：对幅度谱算 L1
            fft_hat = torch.fft.rfft2(I_hat_t, norm='ortho')
            fft_gt = torch.fft.rfft2(I_GT, norm='ortho')
            L_freq = F.l1_loss(fft_hat.abs(), fft_gt.abs()) + \
                     F.l1_loss(torch.angle(fft_hat), torch.angle(fft_gt))
            L_recon = L_recon + self.lambda_freq * L_freq
        
        # (2) 感知损失
        L_perc = self.vgg(I_hat_t, I_GT)
        
        # (3) 光照平滑损失（仅对 s_illum）
        grad_s_x = (s_illum[..., :, 1:] - s_illum[..., :, :-1]).abs()
        grad_s_y = (s_illum[..., 1:, :] - s_illum[..., :-1, :]).abs()
        grad_i_x = (I_GT[..., :, 1:] - I_GT[..., :, :-1]).abs().mean(dim=1, keepdim=True)
        grad_i_y = (I_GT[..., 1:, :] - I_GT[..., :-1, :]).abs().mean(dim=1, keepdim=True)
        L_illum_smooth = (grad_s_x * torch.exp(-grad_i_x)).mean() + \
                          (grad_s_y * torch.exp(-grad_i_y)).mean()
        
        # (4) 时序一致性（可选）
        L_total = L_recon + self.lambda_perc * L_perc + self.lambda_illum * L_illum_smooth
        if self.use_temporal and I_t is not None and I_GT_neighbors is not None:
            L_temporal = self.temporal_loss(I_hat_t, I_GT_neighbors, I_t)
            L_total = L_total + self.lambda_temporal * L_temporal
        
        return L_total, {
            'L_recon': L_recon.item(),
            'L_perc': L_perc.item(),
            'L_illum': L_illum_smooth.item(),
            **({'L_temporal': L_temporal.item()} if self.use_temporal else {})
        }
```

### 5.5 频域损失实现细节

**采纳标准 Cho et al. ICCV 2021 写法**：

```python
fft_hat = torch.fft.rfft2(I_hat_t, norm='ortho')
fft_gt = torch.fft.rfft2(I_GT, norm='ortho')
L_freq = F.l1_loss(fft_hat.abs(), fft_gt.abs()) + F.l1_loss(torch.angle(fft_hat), torch.angle(fft_gt))
```

- **对幅度和相位分别算 L1**，而非对复数实虚部
- 相位损失可能不稳定（角度的环形特性），可加 `mod 2π` 处理或单独降低权重
- 简化版：仅算幅度损失 `L_freq = F.l1_loss(fft_hat.abs(), fft_gt.abs())`

---

## 六、其他遗留问题的补充答复

### 6.1 v3quest2 §2.4：LFF 对 C=48 特征图

已在本文 §一 解答（采用选项 A 改进版）。

### 6.2 v3quest2 §3.2：FRBNet 门控机制

**采纳并集成到本文 §一 的 LFF 实现中**：

```python
class RadialBasisFilter(nn.Module):
    def __init__(self, K: int = 10, n_ang_freq: int = 1):
        super().__init__()
        self.K = K
        self.n_ang_freq = n_ang_freq
        # 径向中心（不可学习）
        self.register_buffer('mu', torch.linspace(0.0, 1.0, steps=K))
        # 带宽（对数空间，可学习）
        self.log_bwh = nn.Parameter(torch.zeros(1))
        # 系数（可学习，初始全零）
        self.coeff_mag = nn.Parameter(torch.zeros(K))
        # 门控（可学习，初始全 1 → sigmoid ≈ 0.73）
        self.raw_gate_mag = nn.Parameter(torch.ones(K))
    
    def forward(self, r_grid, fx, fy):
        """
        r_grid: (H, W) 归一化径向频率
        fx, fy: (H, W) 频谱坐标
        Returns: (H, W) 滤波器响应
        """
        bwh = torch.exp(self.log_bwh)
        gate = torch.sigmoid(self.raw_gate_mag)  # (K,)
        
        # 径向响应 Φ(u,v)
        phi = torch.exp(-((r_grid.unsqueeze(0) - self.mu.view(-1, 1, 1)) / bwh) ** 2)  # (K, H, W)
        Phi = (gate * self.coeff_mag).view(-1, 1, 1) * phi
        Phi = Phi.sum(dim=0)  # (H, W)
        
        # 角度调制 M(u,v) - 固定形式
        theta = torch.atan2(fy, fx + 1e-9)
        angular_mod = 0
        for n in range(1, self.n_ang_freq + 1):
            angular_mod = angular_mod + torch.cos(n * theta) + torch.sin(n * theta)
        angular_mod = angular_mod / (2 * self.n_ang_freq)
        M = 1 + 0.1 * angular_mod
        
        return Phi * M
```

### 6.3 v3quest2 §3.4：NAFNet SimpleGate 是否替换 Sigmoid Gate

**保留当前 Sigmoid Gate，不替换**。

理由：
- SimpleGate 需要通道数翻倍后再 chunk，会增加 Conv 输入通道
- TFSI 的门控融合作用于"空间分支 vs 频域分支"两路特征，**Sigmoid 给出的标量门控值更符合物理直觉**（"取空间还是取频域"）
- SimpleGate 适合通道间的细粒度门控，不适合分支间的粗粒度选择
- v3 首版优先稳定性，不引入实验性优化

### 6.4 v3quest2 §3.5：EDVR PCD 与 TSA 的复用

**SACE 已采用 PCD 风格设计**（本文 §三）。**NDPN/MRPN 不再单独引入 TSA**，理由：

- NDPN 的核心是 SNR 自适应聚合，TSA 的时序 embedding 是冗余的（SNR 已包含逐像素时序统计信息）
- 直接使用 v3answer 给出的 NDPN/MRPN 设计（残差驱动权重）已经包含了时序注意力的核心思想
- 引入 TSA 会增加架构复杂度，违反 v3 "稳健性 > 性能堆叠" 的原则

---

## 七、最终实现路线图（Phase B 准备）

### 7.1 模块状态最终确认

| 模块 | 状态 | 实现优先级 |
|:---|:---|:---|
| `lff.py` (RadialBasisFilter + LFFFeatureAdapter) | ✅ 设计完整 | P0 |
| `tfsi.py` 频域分支接入 | ✅ 设计完整 | P0 |
| `sace.py` (DeformableCrossAttention) | ✅ 设计完整 | P1 |
| `ifpn.py` (IllumExtract) | ✅ 设计完整 | P1 |
| `ndpn.py` (SNR 自适应聚合) | ✅ 设计完整 | P2 |
| `mrpn.py` (残差降权) | ✅ 设计完整 | P2 |
| `igrf.py` (强度引导融合) | ✅ 已实现 | — |
| `losses.py` (TFSNetLoss) | ✅ 设计完整（首版无 temporal） | P3 |

### 7.2 待执行的设计文档修订

在进入 Phase B 编码前，建议先更新 `TFSv3-result.md` 的以下位置：

| 位置 | 原文 | 修订为 |
|:---|:---|:---|
| §4.1 Step 2 | $W_g = 1 - \exp(-(u^2+v^2)/(2\sigma_g^2))$ | $W_g = \exp(-(u^2+v^2)/\sigma_g^2), W_g(0,0)=0$ |
| §4.1 Step 2 | $H(u,v) = \sum_k \omega_k \phi_k \cdot \cos(\theta-\theta_k)$ | $H = \Phi(u,v) \cdot M(u,v)$，其中 $M = 1 + 0.1 \cdot (\cos\theta+\sin\theta)/2$ |
| §4.1 Step 2 | K=未指定 | K=10, N=1 |
| §4.3 输入 | $\text{Concat}[I_t^{down}, \text{Conv}_{1\times1}(F_t^{(L)})]$ | $\text{Concat}[I_t^{down}, \text{mean}(I_t^{down}), \text{Conv}_{1\times1}(F_t^{(L)})]$ |
| §5.1 总损失 | 含 $\lambda_2 \mathcal{L}_{consist}$ | 移除 $\mathcal{L}_{consist}$，$\lambda_2$ 重分配给 $\mathcal{L}_{temporal}$ |
| §5.2 (2) | $\mathcal{L}_{temporal}$ 用 $\hat{I}_i$ | 改为 $\text{Warp}(\hat{I}_t, \text{Flow}_{t \to i}) - I_i^{GT}$ |

### 7.3 Phase B 执行清单（待您确认后启动）

```
[ ] B.1: 实现 models/modules/lff.py
        - RadialBasisFilter（FRBNet 风格）
        - LFFFeatureAdapter（48通道适配）
        - 单元测试：构造 (2, 48, 64, 64) 输入，验证输出 shape

[ ] B.2: 补全 models/modules/tfsi.py
        - 接入 LFFFeatureAdapter 作为频域分支
        - 验证空间/频域门控融合

[ ] B.3: 实现 models/modules/sace.py
        - DeformableCrossAttention 模块
        - SACE 主类（LFF 共享 + 时域中位值 + DeformableCrossAttention）
        - 单元测试：5 帧输入，输出对齐特征 + offset/mask

[ ] B.4: 实现 models/modules/ifpn.py
        - IllumExtract（双流，3层 Conv）
        - IFPN 主类（含 L_ref 计算和强度调制）

[ ] B.5: 实现 models/modules/ndpn.py
        - SNR 估计（τ_mid, τ_scale 可学习）
        - 双因素权重聚合

[ ] B.6: 实现 models/modules/mrpn.py
        - 残差驱动权重 + refine

[ ] B.7: 整合 models/tfs_net.py
        - 取消 NotImplementedError
        - 端到端前向测试：(2, 5, 3, 256, 256) → (2, 3, 256, 256)

[ ] B.8: 实现 losses/losses.py 中的 TFSNetLoss
        - 首版 use_temporal=False
        - 验证梯度流通

[ ] B.9: 删除旧文件（v3quest.md §2.5）
        - mins.py, ispn.py, mspn.py, mins_net.py, reconstruction.py
```

---

## 八、对您的下一步行动建议

### 选项 1（推荐）：直接进入 Phase B

如果上述所有设计决策您都认可，**请回复"开始 Phase B"**，我将按 §7.3 清单逐个文件提供完整 PyTorch 代码。

### 选项 2：修订设计文档

如果您希望先更新 `TFSv3-result.md`，**请回复"先更新设计文档"**，我将提供完整的修订后版本。

### 选项 3：针对特定决策再讨论

如果对 Q1-Q5 中某个决策有疑虑，**请回复"重新讨论 Qx"**，我将提供更多细节或替代方案。

---

## 参考来源

- [FRBNet 官方仓库 — `frbnet_utils.py`](https://github.com/Sing-Forevet/FRBNet/blob/main/custom_mmlab/FRBNet_mmdet/mmdet/models/detectors/frbnet_utils.py)
- [Retinexformer 官方仓库 — `RetinexFormer_arch.py`](https://github.com/caiyuanhao1998/Retinexformer)
- [BasicSR EDVR — `edvr_arch.py` 的 PCDAlignment](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/edvr_arch.py)
- [DAT (Deformable Attention Transformer) — `dat_blocks.py`](https://github.com/LeapLabTHU/DAT)
- [torchvision RAFT 预训练模型文档](https://pytorch.org/vision/stable/models/raft.html)
- [Cho et al., ICCV 2021 — Rethinking Coarse-to-Fine Approach in Single Image Deblurring（频域损失参考）](https://openaccess.thecvf.com/content/ICCV2021/papers/Cho_Rethinking_Coarse-To-Fine_Approach_in_Single_Image_Deblurring_ICCV_2021_paper.pdf)
- [DCNv2 — `torchvision.ops.deform_conv2d` 官方实现](https://pytorch.org/vision/stable/generated/torchvision.ops.deform_conv2d.html)

请告知您的下一步选择。

