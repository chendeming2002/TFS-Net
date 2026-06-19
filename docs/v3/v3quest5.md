# TFS-Net v3 — B.3 修正 + B.4-B.6 实现指令

> **日期**：2026-06-12
> **前置**：v3answer5.md（你的 B.1/B.2/B.3 实现）

---

## 一、B.3 offset reshape 的 P0 级 bug（必须修正）

`DeformableCrossAttention.forward()` L366：

```python
# 你的代码（错误）
offset = offset.view(B, G, K, 2, H, W)
# offset 原始 shape: (B, G*K*K*2, H, W)
# 通道内存布局：[g0_x0, g0_x1, ..., g0_x{K²-1}, g0_y0, g0_y1, ..., g0_y{K²-1}, g1_x0, ...]
# 即：前 K² 个通道全是 x 偏移，后 K² 个通道全是 y 偏移
# 但 view(B, G, K, 2, H, W) 把连续两个通道当作 (x, y) 对 → 数据错乱！
```

**修正**：先按 `(2, K)` 拆分，再 permute 到 `(K, 2)`：

```python
# 修正后
# 内存 [x_0,...,x_{K²-1}, y_0,...,y_{K²-1}] → view(B, G, 2, K, H, W)
# 然后 permute 到 (B, G, K, 2, H, W)，使 offset[:, g, k, 0] = x偏移_k, offset[:, g, k, 1] = y偏移_k
offset = offset.view(B, G, 2, K, H, W).permute(0, 1, 3, 2, 4, 5).contiguous()
```

**注意**：你的简单版 `_sample_simple` 和优化版 `_sample_optimized` 都用了相同的错误 reshape，所以两版"数值一致"——但**一致地错误**。修正后需要同时更新两版的 sample_loc 构造逻辑。

同时确认 `_build_base_grid` 返回的 `(n_points, 2)` 是 `[x, y]` 格式（你的代码是正确的），与 `grid_sample` 的 `grid[..., 0]=x, grid[..., 1]=y` 一致。

---

## 二、对 Claude 三个问题的答复

1. **SACE 数据流**（LFF → median → 逐帧 DCA + 残差）：**确认符合预期**
2. **OffsetMaskHead 输入**（拼接 query+key_value）：**合理**，保留当前设计
3. **默认 `use_optimized=True`**：**同意**

---

## 三、B.4: `models/modules/ifpn.py`

### 3.1 设计公式（来自 TFSv3-result.md §4.3）

```
输入:
  I_t_down = Bicubic(I_center, 1/4)     → (B, 3, H/4, W/4)
  F_t_L = coarse_feats[:, center_idx]    → (B, 96, H/4, W/4)  ← PyramidEncoder 最粗尺度

L_t = IllumExtract(Concat[I_t_down, mean(I_t_down), Conv1x1(F_t_L)])
     → (B, 3, H/4, W/4) RGB 光照图（无 sigmoid）

L_ref = Σ_{i≠t} w_i · L_i,   w_i = Softmax(sim(F_i, F_t))
     其中 L_i = IllumExtract(I_i_down, F_i_L)

F_t^illum_out = s_illum · F_t^illum · L_ref/(L_t+ε) + (1-s_illum) · F_t^illum
```

**注意**：设计文档写 H/8，但当前 PyramidEncoder 两次 stride=2 → **实际是 H/4**。请以 H/4 实现。

### 3.2 IllumExtract 实现（N3: groups=4 必须保留）

```python
class IllumExtract(nn.Module):
    def __init__(self, img_channels=3, feat_channels=96, feat_proj_channels=16, n_fea_middle=31):
        # in_channels = img_channels(3) + mean_channel(1) + feat_proj_channels(16) = 20
        # conv1: 20 → 31
        # depth_conv: 31 → 31, kernel=5, groups=4 ← N3: 不是 groups=31！
        # conv2: 31 → 3
        # 不加 sigmoid
```

**Retinexformer 原始代码**（`RetinexFormer_arch.py` L102-103）：
```python
self.depth_conv = nn.Conv2d(
    n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
# n_fea_in = 4（即 groups=4，分组卷积，不是 depthwise！）
```

### 3.3 IFPN 主类接口

```python
class IFPN(nn.Module):
    def __init__(self, fused_channels=48, coarse_channels=96):
        ...

    def forward(self, I_t_down, F_t_L, s_illum, feats, coarse_feats, center_idx):
        """
        Args:
            I_t_down    : (B, 3, H/4, W/4) 中心帧下采样图
            F_t_L       : (B, 96, H/4, W/4) 中心帧最粗尺度特征（= coarse_feats[:, center_idx]）
            s_illum     : (B, 1, H, W) 光照强度
            feats       : (B, T, C_f=48, H, W) 全分辨率融合特征
            coarse_feats: (B, T, c3=96, H/4, W/4) 所有帧粗尺度特征
            center_idx  : int

        Returns:
            dict:
                f_illum_out: (B, C_f=48, H, W)
                L_t        : (B, 3, H/4, W/4) 中心帧光照图（供调试）
        """
```

### 3.4 sim() 函数

用 `blocks.py` 中已有的 `pairwise_cosine_logits(center, neighbors)` 计算帧间相似度，或简单实现：
```python
def sim(F_i, F_t):
    # F_i: (B, C, H, W), F_t: (B, C, H, W)
    return F.cosine_similarity(F_i.flatten(2), F_t.flatten(2), dim=-1).mean(dim=-1)  # (B,)
```

### 3.5 强度调制

```python
# L_t 和 L_ref 都是 (B, 3, H/4, W/4)
# 需要上采样到 (B, 3, H, W) 才能与 F_t (B, 48, H, W) 交互
# 建议: L_ratio = F.interpolate(L_ref / (L_t + eps), scale_factor=4, mode='bilinear')
# 然后: F_t_illum = Conv1x1(F_t) * L_ratio 或类似方式
# 最终 f_illum_out shape = (B, 48, H, W)
```

### 3.6 验证

```python
ifpn = IFPN(fused_channels=48, coarse_channels=96)
I_t_down = torch.randn(2, 3, 64, 64)
F_t_L = torch.randn(2, 96, 64, 64)
s_illum = torch.rand(2, 1, 256, 256)
feats = torch.randn(2, 5, 48, 256, 256)
coarse_feats = torch.randn(2, 5, 96, 64, 64)

out = ifpn(I_t_down, F_t_L, s_illum, feats, coarse_feats, center_idx=2)
assert out["f_illum_out"].shape == (2, 48, 256, 256)
```

---

## 四、B.5: `models/modules/ndpn.py`

### 4.1 设计公式（来自 TFSv3-result.md §4.4）

```
SNR 估计:
  SNR_hat = |μ_t_clean| / (σ_t + ε)     逐像素
  s_SNR = sigmoid((SNR_hat - τ_mid) / τ_scale)   τ_mid, τ_scale 可学习

Step 1: 利用 SACE 的 attn_maps 做对齐
  F_i^aligned = SACE 输出的 F_aligned_list[i]

Step 2: 双因素动态权重
  α_i(x,y) = sigmoid(Conv(|F_i^aligned - F_t|)) · (1 - s_SNR(x,y))
  含义: 对齐残差小 + SNR 低 → 邻帧权重大

Step 3: 聚合
  F_t^denoised = Σ_i α_i · F_i^aligned / (Σ_i α_i + ε)
  F_t^noise_out = s_noise · F_t^denoised + (1 - s_noise) · F_t
```

### 4.2 NDPN 接口

```python
class NDPN(nn.Module):
    def __init__(self, channels=48):
        # 可学习 SNR 归一化参数
        self.tau_mid = nn.Parameter(torch.tensor(0.0))
        self.tau_scale = nn.Parameter(torch.tensor(1.0))
        # 权重预测卷积
        self.alpha_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),  # 输出单通道权重
        )

    def forward(self, feats, F_aligned_list, mu_t_clean, sigma_t, s_noise, center_idx):
        """
        Args:
            feats          : (B, T, C, H, W) 编码器特征
            F_aligned_list : List[(B, C, H, W)] SACE 对齐后的特征
            mu_t_clean     : (B, C, H, W) SACE 时域中位值
            sigma_t        : (B, C, H, W) TFSI 时域标准差
            s_noise        : (B, 1, H, W) 噪声强度
            center_idx     : int

        Returns:
            dict:
                f_noise_out: (B, C, H, W)
                s_snr      : (B, 1, H, W) SNR 归一化图（供调试）
        """
```

### 4.3 验证

```python
ndpn = NDPN(channels=48)
feats = torch.randn(2, 5, 48, 64, 64)
F_aligned_list = [torch.randn(2, 48, 64, 64) for _ in range(5)]
mu_t_clean = torch.randn(2, 48, 64, 64)
sigma_t = torch.rand(2, 48, 64, 64) + 0.1
s_noise = torch.rand(2, 1, 64, 64)

out = ndpn(feats, F_aligned_list, mu_t_clean, sigma_t, s_noise, center_idx=2)
assert out["f_noise_out"].shape == (2, 48, 64, 64)
```

---

## 五、B.6: `models/modules/mrpn.py`

### 5.1 设计公式

MRPN 处理运动退化（运动模糊/遮挡），核心思想是**残差驱动隐式遮挡**：

```
Step 1: 运动残差计算
  R_i = |F_i^aligned - F_t|     对齐残差（大残差 = 大运动/遮挡）

Step 2: 残差驱动降权
  w_i = sigmoid(-Conv(R_i))     残差越大 → 权重越小（隐式遮挡）
  w_i = w_i / (Σ_i w_i + ε)    归一化

Step 3: 加权聚合
  F_t^motion_agg = Σ_i w_i · F_i^aligned

Step 4: 运动强度调制 + refine
  F_t^motion_out = s_motion · Conv_Refine(F_t^motion_agg) + (1 - s_motion) · F_t
```

### 5.2 MRPN 接口

```python
class MRPN(nn.Module):
    def __init__(self, channels=48):
        # 残差权重预测
        self.weight_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 1),  # 单通道权重
        )
        # refine 卷积
        self.refine = nn.Sequential(
            ConvBlock(channels, channels, 3, 1, 1, act=True),
            ConvBlock(channels, channels, 3, 1, 1, act=False),
        )

    def forward(self, feats, F_aligned_list, s_motion, center_idx):
        """
        Args:
            feats          : (B, T, C, H, W)
            F_aligned_list : List[(B, C, H, W)] SACE 对齐后特征
            s_motion       : (B, 1, H, W) 运动强度
            center_idx     : int

        Returns:
            dict:
                f_motion_out: (B, C, H, W)
                motion_weights: (B, T-1, H, W) 各帧权重图（供调试）
        """
```

### 5.3 验证

```python
mrpn = MRPN(channels=48)
feats = torch.randn(2, 5, 48, 64, 64)
F_aligned_list = [torch.randn(2, 48, 64, 64) for _ in range(5)]
s_motion = torch.rand(2, 1, 64, 64)

out = mrpn(feats, F_aligned_list, s_motion, center_idx=2)
assert out["f_motion_out"].shape == (2, 48, 64, 64)
```

---

## 六、B.4-B.6 完成后直接进入 B.7

B.7 是 `models/tfs_net.py` 整合阶段，需要：
1. 取消 `raise NotImplementedError`
2. 实例化所有模块并串联 forward
3. SACE 的 LFF 与 TFSI 共享（构造时传入 `self.tfsi.freq_branch.lff`）
4. 端到端验证：`(2, 5, 3, 256, 256) → dict{res_t: (2, 3, 256, 256), ...}`

**当前 tfs_net.py 中 Stage 3-5 的注释代码**（供你参考）：
```python
# ── Stage 3：SACE ──
# sace_out = self.sace(feats, tfsi_out)
# attn_maps = sace_out["attn_maps"]
# mu_t_clean = sace_out["mu_t_clean"]

# ── Stage 4：三源恢复分支 ──
# image_center = x[:, center_idx]
# image_down = F.interpolate(image_center, scale_factor=1/4, mode="bicubic")
# f_t_L = coarse_feats[:, center_idx]
# ifpn_out = self.ifpn(image_down, f_t_L, s_illum, feats, coarse_feats, center_idx)
# ndpn_out = self.ndpn(feats, sace_out["F_aligned_list"], sace_out["mu_t_clean"], tfsi_out["sigma_t"], s_noise, center_idx)
# mrpn_out = self.mrpn(feats, sace_out["F_aligned_list"], s_motion, center_idx)

# ── Stage 5：IGRF ──
# f_t_base = feats[:, center_idx]
# igrf_out = self.igrf(f_t_base, ifpn_out["f_illum_out"], ndpn_out["f_noise_out"],
#                       mrpn_out["f_motion_out"], s_illum, s_noise, s_motion, image_center)
```

请完成 B.3 修正 + B.4 + B.5 + B.6 + B.7 的全部代码，并给出完整参数量统计（目标 < 2M）。

---

## 七、现有可复用工具（blocks.py）

```python
ConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1, act=True)  # Conv2d + GELU
ResBlock(channels)
LayerNorm2d(channels, eps=1e-6)
safe_divide(x, y, eps=1e-6)
pairwise_cosine_logits(center, neighbors)  # 全局余弦相似度
```

## 八、约束条件

1. 不修改 `blocks.py`、`encoder.py`、`igrf.py`、`datasets/*`、`utils/*`
2. 不删除 v1 旧文件
3. 所有中间张量标注 shape 注释
4. 每模块实现后提供参数量统计
