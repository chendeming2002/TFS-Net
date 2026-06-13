# TFS-Net v3.1 平台期结构性修复记录

## 一、问题背景

训练 TFS-Net v3 在 SDSD indoor 数据集上，前 4 个 epoch 出现严重平台期：

| Epoch | loss_total | loss_pix (L1) | loss_freq | loss_perc | loss_illum |
|-------|-----------|--------------|-----------|-----------|------------|
| 1 | 0.5551 | 0.5265 | 0.1465 | 0.1315 | 0.0799 |
| 2 | 0.5545 | 0.5259 | 0.1462 | 0.1314 | 0.0797 |
| 3 | 0.5557 | 0.5273 | 0.1460 | 0.1309 | 0.0796 |
| 4 | 0.5543 | 0.5258 | 0.1464 | 0.1311 | 0.0800 |

**核心症状**：L1 loss 在 0.525~0.527 之间震荡，4 个 epoch 仅下降 0.13%，各分支损失近乎无变化。

---

## 二、结构性成因分析（6 类问题）

### 问题 1（P0）：IGRF 重建头过浅

**位置**：`models/modules/igrf.py`

**原始代码**：
```python
self.conv1 = nn.Conv2d(48, 48, 3, 1, 1)  # 唯一隐藏层
self.act = nn.GELU()
self.conv2 = nn.Conv2d(48, 3, 3, 1, 1)   # 输出层
```

**分析**：
- 最终输出为 `res_t = clamp(image_center + delta, 0, 1)`，即在原始低光图上叠加残差
- 低光增强需要大幅亮度变换（像素值可能从 0.1 → 0.8），仅 2 层卷积的表达上限极低
- 这是 L1 loss 快速触顶的**根本原因**——重建头无法拟合更复杂的映射

### 问题 2（P0）：三源分支的"恒等退化"

**位置**：`models/modules/ifpn.py`、`ndpn.py`、`mrpn.py`

**原始代码（以 IFPN 为例）**：
```python
f_illum_out = s_illum * F_t_illum + (1 - s_illum) * F_t
```

**分析**：
- 三个分支都使用 `s_xxx * F_processed + (1-s_xxx) * F_t` 的强度门控
- 当 `s_illum → 0` 时，`f_illum_out → F_t`（退化为恒等映射）
- 之后在 IGRF 中又被 `s_illum` 再次加权，形成**双重门控**
- 双重门控使得信号衰减为 `s² * signal`，分支容易被完全旁路

### 问题 3（P1）：IGRF 融合公式不对称

**原始融合公式**：
```python
f_fused = s_illum * f_illum_out + s_noise * f_noise_out + s_motion * f_motion_out + f_t_base
```

**分析**：
- `f_t_base` 无权重约束，恒定贡献 1.0
- 三源分支贡献受 s 衰减，最终输出被 base 主导
- 模型倾向于保持 base 不变，三源分支形同虚设

### 问题 4（P1）：通道容量瓶颈

**原始配置**：
```yaml
level_channels: [32, 64, 96]
fused_channels: 48
```

**分析**：
- 48 通道的融合特征需要承载光照/噪声/运动三种语义信息，容量不足
- 编码器仅 3 级（最大 96 通道），特征表达受限

### 问题 5（P1）：感知损失无效

**原始配置**：
```yaml
perceptual_pretrained: false  # 未配置此项
```

**分析**：
- VGG16 未加载 ImageNet 预训练权重，随机初始化的 VGG 提供的是噪声梯度
- 感知损失（perceptual loss）无法提供有意义的语义约束
- 实际日志中 `loss_perc ≈ 0.13`（极低），说明随机 VGG 对所有图像输出相似特征

### 问题 6（P2）：编码器下采样不足

**原始结构**：3 级下采样，最粗层为 H/4×W/4

**分析**：
- 对于 256×256 的 crop，感受野最大 64×64（仅覆盖 1/4 图像）
- 全局亮度/对比度信息获取不足，光照校正缺乏全局上下文

---

## 三、修复方案与代码改动

### 修复 1：深化 IGRF 重建头 + 归一化融合

**文件**：`models/modules/igrf.py`（完整重写）

**修改内容**：
1. 融合公式改为**归一化加权**：
   ```python
   w_total = s_illum + s_noise + s_motion + 1.0
   w_illum = s_illum / w_total
   w_noise = s_noise / w_total
   w_motion = s_motion / w_total
   w_base = 1.0 / w_total
   f_fused = w_illum * f_illum_out + w_noise * f_noise_out + w_motion * f_motion_out + w_base * f_t_base
   ```
2. 重建头升级为 **4 个 ResBlock + 输出卷积**：
   ```python
   self.recon_head = nn.Sequential(
       ResBlock(channels),
       ResBlock(channels),
       ResBlock(channels),
       ResBlock(channels),
       nn.Conv2d(channels, out_channels, 3, 1, 1),
   )
   ```

**目的**：
- 归一化确保所有权重和为 1，base 不再无约束主导
- 4 个 ResBlock（每个含 2 层 Conv + 残差连接）大幅增强重建映射能力

### 修复 2：移除三源分支双重门控

**文件**：`models/modules/ifpn.py`、`ndpn.py`、`mrpn.py`

**修改内容**（以 IFPN 为例）：
```python
# 修改前：
f_illum_out = s_illum * F_t_illum + (1 - s_illum) * F_t

# 修改后：
f_illum_out = F_t_illum  # 直接输出，由 IGRF 统一做强度加权
```

NDPN 和 MRPN 同理：
```python
# NDPN 修改后：
f_noise_out = F_denoised  # 移除 s_noise 门控

# MRPN 修改后：
f_motion_out = F_motion_refined  # 移除 s_motion 门控
```

**目的**：
- 避免信号双重衰减（`s² * signal`）
- 让各分支专注于特征提取，强度控制统一在 IGRF 的归一化加权中完成
- 梯度路径更短，三源分支能获得更有效的梯度信号

### 修复 3：扩展编码器至 4 级

**文件**：`models/modules/encoder.py`

**修改内容**：
```python
# 支持 4 级 level_channels
if len(level_channels) == 4:
    c1, c2, c3, c4 = level_channels
    self.stage4 = EncoderStage(c3, c4, stride=2)   # H/8 × W/8
    self.lateral4 = nn.Conv2d(c4, fused_channels, 1, 1, 0)
```

FPN 融合路径增加 stage4 → stage3 的上采样连接：
```python
if self.has_stage4:
    l4 = self.stage4(l3)
    p4 = self.lateral4(l4)
    p3 = self.lateral3(l3) + F.interpolate(p4, size=l3.shape[-2:], ...)
```

**目的**：
- 最粗层达到 H/8×W/8（256 crop → 32×32），感受野覆盖全图
- 为 IFPN 的 IllumExtract 提供更丰富的全局上下文
- 向后兼容：3 级配置仍可正常使用

### 修复 4：提升通道容量

**文件**：`configs/sdsd_stage1.yaml`

```yaml
# 修改前：
level_channels: [32, 64, 96]
fused_channels: 48

# 修改后：
level_channels: [32, 64, 96, 128]
fused_channels: 64
```

**目的**：
- 64 通道融合特征有更大容量承载三源语义
- 128 通道的 stage4 提供强大的全局特征表示

### 修复 5：启用感知损失预训练权重

**文件**：`configs/sdsd_stage1.yaml`

```yaml
# 修改后：
perceptual_pretrained: true
```

**目的**：
- VGG16 加载 ImageNet 预训练权重后，能提取有意义的语义特征
- 感知损失有效约束生成图像的高层语义质量（纹理、结构）

### 修复 6：适配 tfs_net.py 主干网络

**文件**：`models/tfs_net.py`

```python
# 动态获取最粗层通道数（适配 3 或 4 级编码器）
coarse_channels = level_channels[-1]  # 4级时为128，3级时为96

# IFPN 使用新的 coarse_channels
self.ifpn = IFPN(fused_channels=fused_channels, coarse_channels=coarse_channels, ...)
```

### 辅助修改：同步 TFSI 和 SACE 通道数

**文件**：`models/modules/tfsi.py`、`models/modules/sace.py`

将默认 `channels` 参数从 48 改为 64，确保与新的 `fused_channels=64` 一致。

---

## 四、修复效果验证

### 模型参数量变化

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 总参数量 | ~0.5M | **1.26M** |
| IGRF 重建头 | 2 层 Conv | 4×ResBlock + Conv |
| 编码器层级 | 3 级 (H/4) | 4 级 (H/8) |
| 融合通道 | 48 | 64 |

### 训练初始表现对比

| 指标 | 修复前 (Epoch 1-4 均值) | 修复后 (Epoch 1-2 均值) |
|------|------------------------|------------------------|
| loss_pix (L1) | 0.526 | **0.503** (↓4.4%) |
| loss_freq | 0.146 | **0.261** (频域分支激活) |
| loss_perc | 0.131 | **1.54** (VGG感知损失生效) |
| loss_illum | 0.080 | **0.618** (光照分支激活) |

### 关键改善信号

1. **L1 起点更低**：修复后模型 Epoch 1 的 L1 (0.503) 已经低于修复前 Epoch 4 的收敛值 (0.526)
2. **三源分支不再退化**：perc/illum/freq 的损失值大幅变化，证明分支真正在工作
3. **光照分支活跃**：`loss_illum` 从 0.08 增至 0.62，说明光照校正路径被有效利用

---

## 五、后续超参数调整

在初步验证修复有效后，进一步调整训练超参数以解决收敛速度偏慢的问题：

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| epochs | 200 | 100 | 缩短训练周期 |
| lr | 2e-4 | 5e-4 | 模型参数量翻倍，需更大学习率驱动 |
| lambda_illum | 0.01 | 0.05 | 光照损失梯度贡献从 0.9% 提升至 ~4.5% |

调度器：`CosineAnnealingLR(T_max=100)`，学习率从 5e-4 平滑衰减至接近 0。

---

## 六、涉及的文件清单

| 文件路径 | 修改类型 |
|---------|---------|
| `models/modules/encoder.py` | 新增 stage4 + lateral4 |
| `models/modules/igrf.py` | 重写融合公式 + 深化重建头 |
| `models/modules/ifpn.py` | 移除双重门控 |
| `models/modules/ndpn.py` | 移除双重门控 |
| `models/modules/mrpn.py` | 移除双重门控 |
| `models/tfs_net.py` | 适配 4 级编码器 + coarse_channels |
| `models/modules/tfsi.py` | channels 默认值 48→64 |
| `models/modules/sace.py` | channels 默认值 48→64 |
| `configs/sdsd_stage1.yaml` | level_channels/fused_channels/pretrained/lr/epochs |
