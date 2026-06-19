# TFS-Net v3 — B.1 修正 + B.2 实现指令

> **日期**：2026-06-12
> **前置**：v3answer3.md（你的 B.1 实现）、v3quest3.md（完整项目上下文）

---

## 一、B.1 修正（3 处，修正后继续）

### 修正 1: K 和 n_ang_freq 默认值

```python
# 你的代码（L72）
def __init__(self, K: int = 16, n_ang_freq: int = 0):

# 改为（与 FRBNet 论文一致）
def __init__(self, K: int = 10, n_ang_freq: int = 1):
```

### 修正 2: 角度调制公式

```python
# 你的代码（L118-120）
if self.n_ang_freq > 0:
    angular_mod = torch.cos(self.n_ang_freq * theta)     # 值域 [-1, 1]，会反转 basis 符号
    basis = basis * angular_mod.unsqueeze(0)

# 改为（FRBNet 原始方式：1 + λ*cos(n*θ)，始终为正的小扰动）
if self.n_ang_freq > 0:
    angular_mod = 1.0 + 0.1 * torch.cos(self.n_ang_freq * theta)  # 值域 [0.9, 1.1]
    basis = basis * angular_mod.unsqueeze(0)                       # (K, H, W)
```

**FRBNet 原始代码参考**（`frbnet_utils.py` L36-41）：
```python
angular_mod = torch.cos(self.n_ang_freq * theta)
angular_mod = 1 + 0.1 * angular_mod  # λ=0.1 硬编码
```

### 修正 3: log_bwh 初始值参考

FRBNet 的 `bwh = sigma * (mu[1]-mu[0]) * lambda`，其中 `sigma=0.2`, `mu[1]-mu[0]=1/10`, `lambda=0.1`，即 `bwh≈0.002`。你当前用 `math.log(0.05)` → `bwh≈0.05`（25倍宽）。建议改为更窄的初始带宽，如 `math.log(0.01)` → `bwh≈0.01`，使各基函数初始有区分度而非大面积重叠。

---

## 二、B.2: TFSI 频域分支接入 LFF

### 当前 tfsi.py 中 FrequencyBranch 代码（你需要替换的部分）

```python
# models/modules/tfsi.py L80-114（当前占位）
class FrequencyBranch(nn.Module):
    """
    TFSI 频域分支（LFF 可学习频率滤波）：占位骨架
    当前状态：未实现，forward() 返回零张量，使 TFSI 整体可独立运行。
    """

    def __init__(self, channels: int, fused_channels: int):
        super().__init__()
        self.channels = channels
        self.fused_channels = fused_channels
        self._placeholder = nn.Identity()

    def forward(self, feats: torch.Tensor, center_idx: int) -> torch.Tensor:
        """
        Args:
            feats      : (B, T, C, H, W) 多帧编码器特征
            center_idx : 中心帧索引
        Returns:
            F_f: (B, C_f, H, W) 频域特征（当前为零张量占位）
        """
        b, t, c, h, w = feats.shape
        f_center = feats[:, center_idx]  # (B, C, H, W)
        return torch.zeros(b, self.fused_channels, h, w,
                           device=f_center.device, dtype=f_center.dtype)
```

### TFSI 主类如何调用 FrequencyBranch（L246）

```python
# tfsi.py L246
f_f = self.freq_branch(feats_norm, center_idx)
```

### 要求

1. **重写 `FrequencyBranch`**，内部使用 `LFFFeatureAdapter`：
   - 输入：`feats (B, T, C, H, W)` + `center_idx`
   - 对中心帧特征应用 LFF：`f_center = feats[:, center_idx]` → `LFFFeatureAdapter(f_center)` → `(B, C, H, W)`
   - 如果 `C != fused_channels`，需要一个 1×1 Conv 投影
   - 输出：`(B, fused_channels, H, W)`

2. **不要修改** `SpatialBranch`、`GatedFusion`、`IntensityHead`、`TFSI.__init__` 中其他部分

3. **修改 `TFSI.__init__`** 中 `self.freq_branch` 的初始化，传入 `channels` 和 `fused_channels`

### 关键约束

- 当前 `channels=48, fused_channels=48`（相等），但代码要兼容 `channels != fused_channels` 的情况
- `LFFFeatureAdapter` 输入输出通道相同（C→C），如果 C≠fused_channels 需要额外投影
- LFF 的 K=10, n_ang_freq=1（修正后的默认值）

### 验证

```python
# 验证伪代码
from models.modules.tfsi import TFSI
import torch

tfsi = TFSI(channels=48, fused_channels=48)
feats = torch.randn(2, 5, 48, 64, 64)  # (B, T, C, H, W)
out = tfsi(feats)

assert out["F_f"].shape == (2, 48, 64, 64)      # 频域分支输出
assert out["F_s"].shape == (2, 48, 64, 64)      # 空间分支输出
assert out["F_fused"].shape == (2, 48, 64, 64)  # 门控融合输出
assert out["s_illum"].shape == (2, 1, 64, 64)   # 光照强度
assert out["s_noise"].shape == (2, 1, 64, 64)   # 噪声强度
assert out["s_motion"].shape == (2, 1, 64, 64)  # 运动强度

# F_f 不应再是零张量
assert out["F_f"].abs().mean() > 0.0
print("✅ B.2 验证通过")
```

---

## 三、B.2 完成后请直接继续 B.3

B.2 完成后，直接开始 **B.3: models/modules/sace.py**，不需要等确认。

B.3 需要实现的内容：
1. `DeformableCrossAttention`（N4 要求：提供简单版 + 优化版）
2. `SACE` 主类（N5 要求：完整 forward 数据流）

**SACE 接口约定**：
```python
class SACE(nn.Module):
    def __init__(self, channels=48, n_groups=4, kernel_size=3):
        ...

    def forward(self, feats, tfsi_out):
        """
        Args:
            feats   : (B, T, C, H, W) 多帧编码器特征
            tfsi_out: dict (TFSI 完整输出，含 mu_t, sigma_t, snr 等)

        Returns:
            dict:
                attn_maps      : list of (offset, mask) tuples，每帧一个
                mu_t_clean     : (B, C, H, W) 光照归一化后的时域中位值
                F_aligned_list : list of (B, C, H, W)，每帧对齐后的特征
        """
```

**DeformableCrossAttention 接口约定**：
```python
class DeformableCrossAttention(nn.Module):
    def __init__(self, channels, n_groups=4, kernel_size=3, use_optimized=False):
        ...

    def forward(self, query, key_value, offset, mask):
        """
        Args:
            query    : (B, C, H, W) 中心帧特征
            key_value: (B, C, H, W) 邻帧特征
            offset   : (B, n_groups*kernel_size*kernel_size*2, H, W) 可学习偏移
            mask     : (B, n_groups*kernel_size*kernel_size, H, W) 可学习注意力权重

        Returns:
            out: (B, C, H, W) 变形采样后的特征
        """
```

**现有 blocks.py 可复用工具**：
```python
LayerNorm2d(channels, eps=1e-6)  # 通道级 LayerNorm
ConvBlock(in_ch, out_ch, kernel_size, stride, padding, act=True)  # Conv2d + GELU
ResBlock(channels)               # 残差块
```

完成后请给出 B.2 + B.3 的完整代码和参数量统计。
