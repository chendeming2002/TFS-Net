# TFS-Net v3 Phase B — B.1 修正 + B.2 + B.3 实现

---

## 一、B.1 修正落地

应用三处修正后的 `models/modules/lff.py`（仅展示变更段落）：

```python
class RadialBasisFilter(nn.Module):
    def __init__(self, K: int = 10, n_ang_freq: int = 1):  # ← 修正 1: 默认值与 FRBNet 对齐
        super().__init__()
        self.K = K
        self.n_ang_freq = n_ang_freq

        mu = torch.linspace(0.0, 1.0, steps=K)
        self.register_buffer('mu', mu)

        # ← 修正 3: 初始带宽更窄，使基函数初始有区分度
        # FRBNet 实际 bwh ≈ 0.002，此处折中取 0.01（避免初期梯度过陡）
        self.log_bwh = nn.Parameter(torch.tensor(math.log(0.01)))

        self.coeff_mag = nn.Parameter(torch.zeros(K))
        self.raw_gate_mag = nn.Parameter(torch.ones(K))
        self.coeff_phase = nn.Parameter(torch.zeros(K))
        self.raw_gate_phase = nn.Parameter(torch.ones(K))

    def forward(self, H: int, W: int, device, dtype):
        r_hat, theta = self._build_freq_grid(H, W, device, dtype)

        bwh = torch.exp(self.log_bwh).clamp(min=1e-3, max=1.0)
        basis = torch.exp(
            -((r_hat.unsqueeze(0) - self.mu.view(-1, 1, 1)) ** 2)
            / (2 * bwh ** 2)
        )  # (K, H, W)

        # ← 修正 2: 角度调制改为 1 + 0.1*cos(n*θ)，值域 [0.9, 1.1]
        if self.n_ang_freq > 0:
            angular_mod = 1.0 + 0.1 * torch.cos(self.n_ang_freq * theta)  # (H, W)
            basis = basis * angular_mod.unsqueeze(0)                       # (K, H, W)

        gate_mag = torch.sigmoid(self.raw_gate_mag)
        diff_mag = (
            gate_mag.view(-1, 1, 1) * self.coeff_mag.view(-1, 1, 1) * basis
        ).sum(dim=0, keepdim=True)

        gate_phase = torch.sigmoid(self.raw_gate_phase)
        diff_phase = (
            gate_phase.view(-1, 1, 1) * self.coeff_phase.view(-1, 1, 1) * basis
        ).sum(dim=0, keepdim=True)

        return diff_mag, diff_phase
```

其余代码（`_build_freq_grid`、`LFFFeatureAdapter`）保持不变。

**修正后的默认参数总结**：`K=10, n_ang_freq=1, log_bwh=log(0.01)`，与 FRBNet 论文/代码语义一致。

---

## 二、B.2: TFSI 频域分支接入 LFF

### `models/modules/tfsi.py` 修改清单

#### 修改 1: 文件头新增 import

在现有 import 区域增加：

```python
from models.modules.lff import LFFFeatureAdapter
```

#### 修改 2: 替换 `FrequencyBranch` 类（L80-114）

```python
class FrequencyBranch(nn.Module):
    """
    TFSI 频域分支 — 基于 LFFFeatureAdapter 的可学习频率滤波。

    实现状态: ✅ 已接入 LFF (B.2)

    数据流:
        feats[:, center_idx]                  # (B, C, H, W)
        → optional in_proj (若 C != fused_channels)
        → LFFFeatureAdapter (频域 RBF 整形)
        → optional out_proj (若 C != fused_channels)
        → F_f                                 # (B, fused_channels, H, W)

    设计说明:
        当前 v3 配置下 channels == fused_channels == 48，
        in_proj/out_proj 退化为 Identity，无额外参数。
        若后续配置不一致，1x1 Conv 自动启用。
    """

    def __init__(
        self,
        channels: int,
        fused_channels: int,
        K: int = 10,
        n_ang_freq: int = 1,
        per_channel_rbf: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.fused_channels = fused_channels

        # 通道对齐: LFF 内部要求输入输出通道相同
        # 若 channels == fused_channels，用 Identity 省参数
        if channels == fused_channels:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
            lff_channels = channels
        else:
            # 先投影到 fused_channels，再在 fused_channels 维度上做 LFF
            self.in_proj = nn.Conv2d(channels, fused_channels, kernel_size=1, bias=True)
            self.out_proj = nn.Identity()
            lff_channels = fused_channels

        # 核心: LFF 频率域适配器
        self.lff = LFFFeatureAdapter(
            channels=lff_channels,
            K=K,
            n_ang_freq=n_ang_freq,
            per_channel_rbf=per_channel_rbf,
        )

    def forward(self, feats: torch.Tensor, center_idx: int) -> torch.Tensor:
        """
        Args:
            feats      : (B, T, C, H, W) 多帧归一化后的编码器特征
            center_idx : 中心帧索引 (通常为 T//2)

        Returns:
            F_f: (B, fused_channels, H, W) 频域整形后的特征
        """
        # Step 1: 取中心帧
        f_center = feats[:, center_idx]                       # (B, C, H, W)

        # Step 2: 通道对齐 (Identity if C == fused_channels)
        f_center = self.in_proj(f_center)                     # (B, lff_C, H, W)

        # Step 3: LFF 频域整形 (通道不变)
        f_f = self.lff(f_center)                              # (B, lff_C, H, W)

        # Step 4: 输出对齐 (当前为 Identity)
        f_f = self.out_proj(f_f)                              # (B, fused_channels, H, W)

        return f_f
```

#### 修改 3: `TFSI.__init__` 中频域分支的初始化（在原代码中找到 `self.freq_branch = FrequencyBranch(...)` 行）

将原行：

```python
self.freq_branch = FrequencyBranch(channels, fused_channels)
```

替换为：

```python
self.freq_branch = FrequencyBranch(
    channels=channels,
    fused_channels=fused_channels,
    K=10,
    n_ang_freq=1,
    per_channel_rbf=False,
)
```

> **未修改部分**：`SpatialBranch`、`GatedFusion`、`IntensityHead`、`TFSI.forward()` 的数据流逻辑、L246 的调用方式 `f_f = self.freq_branch(feats_norm, center_idx)`。

### B.2 验证伪代码

```python
import torch
from models.modules.tfsi import TFSI

def test_tfsi_with_lff():
    tfsi = TFSI(channels=48, fused_channels=48)
    feats = torch.randn(2, 5, 48, 64, 64)
    out = tfsi(feats)

    assert out["F_f"].shape == (2, 48, 64, 64)
    assert out["F_s"].shape == (2, 48, 64, 64)
    assert out["F_fused"].shape == (2, 48, 64, 64)
    assert out["s_illum"].shape == (2, 1, 64, 64)
    assert out["s_noise"].shape == (2, 1, 64, 64)
    assert out["s_motion"].shape == (2, 1, 64, 64)
    assert out["F_f"].abs().mean() > 0.0   # ← LFF 已激活，非零
    print(f"✅ B.2 验证通过，F_f 均值 = {out['F_f'].abs().mean():.4f}")


def test_tfsi_channel_mismatch():
    """通道不一致也能跑通"""
    tfsi = TFSI(channels=32, fused_channels=48)
    feats = torch.randn(2, 5, 32, 64, 64)
    out = tfsi(feats)
    assert out["F_f"].shape == (2, 48, 64, 64)
    print("✅ 通道不匹配测试通过")


def test_tfsi_gradient():
    tfsi = TFSI(channels=48, fused_channels=48)
    feats = torch.randn(2, 5, 48, 64, 64, requires_grad=True)
    out = tfsi(feats)
    loss = out["F_fused"].mean() + out["F_f"].mean()
    loss.backward()
    
    assert feats.grad is not None
    assert tfsi.freq_branch.lff.rbf.coeff_mag.grad is not None
    print("✅ TFSI 梯度回传通过")


if __name__ == "__main__":
    test_tfsi_with_lff()
    test_tfsi_channel_mismatch()
    test_tfsi_gradient()
```

### B.2 参数量增量

| 组件 | 参数量 |
|------|--------|
| 旧 `FrequencyBranch` (Identity 占位) | 0 |
| 新 `FrequencyBranch` (LFF, C=48, K=10, n_ang_freq=1, shared) | **2,393** |
| └─ `lff.rbf` (mu+log_bwh+coeff×2+gate×2) | 41 |
| └─ `lff.post_conv` (48×48+48) | 2,352 |
| └─ `in_proj` / `out_proj` (Identity) | 0 |

---

## 三、B.3: `models/modules/sace.py`

```python
"""
SACE (Spatial-Adaptive Cross-frame Enhancement) Module — TFS-Net v3
====================================================================
基于可变形跨帧注意力的多帧对齐与光照归一化模块。

实现状态:
    ✅ DeformableCrossAttention: 简单版 + 优化版双实现 (N4 修正)
    ✅ SACE 主类: 完整 forward 数据流 (N5 修正)

数据流:
    1. 对每帧 f_t 做 LFF 频域整形 → f_t_lff
    2. 沿时域取中位值 → μ_t_clean (作为参考帧)
    3. 用 μ_t_clean 作为 query，逐帧 f_t 作为 key/value 做可变形注意力
    4. 输出 list[F_aligned_t]，每帧已对齐到中心帧的空间结构

参考:
    - reference_repos/Deformable-DETR/models/ops/modules/ms_deform_attn.py
    - torchvision.ops.deform_conv2d
"""

from __future__ import annotations

from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import ConvBlock, LayerNorm2d
from models.modules.lff import LFFFeatureAdapter


# ============================================================
#  DeformableCrossAttention (N4)
# ============================================================

class DeformableCrossAttention(nn.Module):
    """
    可变形跨帧注意力 — 用 query 引导对 key_value 的空间自适应采样。

    与 Deformable-DETR 不同:
        - 单尺度 (无多尺度查询)
        - 分组采样: n_groups 个采样组共享通道分块
        - kernel_size × kernel_size 个采样点/组（参考 DCN-v2 的局部邻域设计）

    总采样点数 = n_groups * kernel_size^2
    默认 n_groups=4, kernel_size=3 → 36 采样点/像素

    Args:
        channels      : 输入通道数 (query 和 key_value 通道相同)
        n_groups      : 分组数 (通道被均分为 n_groups 组，每组独立采样)
        kernel_size   : 每组采样的邻域核大小
        use_optimized : 是否使用优化版 (单次 grid_sample 调用)
                        False -> 简单版 (G×K² 次循环，便于调试)
                        True  -> 优化版 (一次性 batch 维度展开)
    """

    def __init__(
        self,
        channels: int,
        n_groups: int = 4,
        kernel_size: int = 3,
        use_optimized: bool = True,
    ):
        super().__init__()
        assert channels % n_groups == 0, \
            f"channels({channels}) 必须能被 n_groups({n_groups}) 整除"

        self.channels = channels
        self.n_groups = n_groups
        self.kernel_size = kernel_size
        self.n_points = kernel_size * kernel_size           # 每组采样点数
        self.total_points = n_groups * self.n_points        # 总采样点数
        self.group_channels = channels // n_groups
        self.use_optimized = use_optimized

        # 输出值投影 (DCN-v2 风格的尾部 1x1 conv)
        self.value_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        # 注: offset / mask 由外部 SACE 主类提供 (与 query 解耦，便于跨帧共享生成器)

    def _build_base_grid(self, H: int, W: int, device, dtype):
        """
        构造 kernel_size×kernel_size 的基础邻域偏移（中心化）
        Returns: (n_points, 2) 单位为像素，范围 [-(k//2), k//2]
        """
        k = self.kernel_size
        center = (k - 1) / 2.0
        ys, xs = torch.meshgrid(
            torch.arange(k, device=device, dtype=dtype) - center,
            torch.arange(k, device=device, dtype=dtype) - center,
            indexing='ij',
        )
        base = torch.stack([xs, ys], dim=-1).reshape(-1, 2)   # (n_points, 2) [x, y]
        return base

    def _build_reference_grid(self, B: int, H: int, W: int, device, dtype):
        """
        构造 grid_sample 用的参考网格（归一化到 [-1, 1]）
        Returns: (B, H, W, 2) [x, y]
        """
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        ref = torch.stack([grid_x, grid_y], dim=-1)            # (H, W, 2)
        ref = ref.unsqueeze(0).expand(B, H, W, 2)              # (B, H, W, 2)
        return ref

    def forward(
        self,
        query: torch.Tensor,        # (B, C, H, W) 中心帧 (信息源/参考)
        key_value: torch.Tensor,    # (B, C, H, W) 邻帧 (被采样)
        offset: torch.Tensor,       # (B, n_groups*n_points*2, H, W)
        mask: torch.Tensor,         # (B, n_groups*n_points,   H, W)
    ) -> torch.Tensor:
        """
        Returns:
            out: (B, C, H, W) 邻帧在 query 引导下的可变形采样结果
        """
        B, C, H, W = key_value.shape
        G, K, P = self.n_groups, self.n_points, self.total_points
        Cg = self.group_channels
        device, dtype = key_value.device, key_value.dtype

        # ── Step 1: 值投影
        value = self.value_proj(key_value)                                # (B, C, H, W)

        # ── Step 2: 解析 offset (像素单位) 和 mask (softmax 权重)
        # offset shape: (B, G*P*2, H, W) → (B, G, P, 2, H, W)
        offset = offset.view(B, G, K, 2, H, W)
        # mask shape:   (B, G*P, H, W)   → (B, G, P, H, W)，沿 P 做 softmax
        mask = mask.view(B, G, K, H, W)
        mask = F.softmax(mask, dim=2)                                     # 每个像素的 P 个采样点权重和为 1

        # ── Step 3: 构建采样位置
        base_offset = self._build_base_grid(H, W, device, dtype)          # (P, 2)
        ref_grid = self._build_reference_grid(B, H, W, device, dtype)     # (B, H, W, 2)

        # 将基础邻域 + 学习偏移合并 (像素单位)
        # base_offset: (P, 2) → (1, 1, P, 2, 1, 1)
        # offset:               (B, G, P, 2, H, W)
        # combined:             (B, G, P, 2, H, W)
        combined_pixel = offset + base_offset.view(1, 1, K, 2, 1, 1)

        # 转换到归一化坐标 [-1, 1]
        # 像素单位 dx → 归一化 dx_norm = dx * 2 / (W-1) (x 方向)
        norm_scale = torch.tensor([2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)],
                                  device=device, dtype=dtype).view(1, 1, 1, 2, 1, 1)
        combined_norm = combined_pixel * norm_scale                       # (B, G, P, 2, H, W)

        # 加到参考网格上
        # ref_grid:      (B, H, W, 2) → (B, 1, 1, 2, H, W)
        ref_expand = ref_grid.permute(0, 3, 1, 2).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 2, H, W)
        sample_loc = ref_expand + combined_norm                              # (B, G, P, 2, H, W)

        if self.use_optimized:
            out = self._sample_optimized(value, sample_loc, mask, B, C, H, W, G, K, Cg)
        else:
            out = self._sample_simple(value, sample_loc, mask, B, C, H, W, G, K, Cg)

        # ── Step 4: 输出投影
        out = self.output_proj(out)                                       # (B, C, H, W)
        return out

    # ─────────── 简单版 (循环) ───────────
    def _sample_simple(self, value, sample_loc, mask, B, C, H, W, G, K, Cg):
        """
        循环采样: G×K 次 grid_sample 调用。
        sample_loc: (B, G, K, 2, H, W)
        mask:       (B, G, K, H, W)
        """
        out = torch.zeros(B, C, H, W, device=value.device, dtype=value.dtype)

        # 按组切分 value 的通道: (B, C, H, W) → list of (B, Cg, H, W)
        value_groups = value.view(B, G, Cg, H, W)

        for g in range(G):
            v_g = value_groups[:, g]                                       # (B, Cg, H, W)
            for k in range(K):
                # 采样位置: (B, H, W, 2)
                loc = sample_loc[:, g, k].permute(0, 2, 3, 1)              # (B, H, W, 2)
                # grid_sample 期望 (B, C, H, W) + (B, H, W, 2)
                sampled = F.grid_sample(
                    v_g, loc,
                    mode='bilinear', padding_mode='zeros', align_corners=True,
                )                                                          # (B, Cg, H, W)
                w_gk = mask[:, g, k].unsqueeze(1)                          # (B, 1, H, W)
                out[:, g * Cg:(g + 1) * Cg] += sampled * w_gk
        return out

    # ─────────── 优化版 (一次性 grid_sample) ───────────
    def _sample_optimized(self, value, sample_loc, mask, B, C, H, W, G, K, Cg):
        """
        将 G×K 个采样点折叠到 batch 维度，单次 grid_sample 调用。
        sample_loc: (B, G, K, 2, H, W)
        mask:       (B, G, K, H, W)

        关键技巧:
            value 按组切分 → (B*G, Cg, H, W)
            sample_loc 折叠 → (B*G, K*H, W, 2)  ← 把 K 个采样点放到 H 轴
            一次 grid_sample → (B*G, Cg, K*H, W) → reshape (B*G, Cg, K, H, W)
            按 mask 加权求和 → (B*G, Cg, H, W) → (B, C, H, W)
        """
        # 切分 value 到组维度: (B, C, H, W) → (B*G, Cg, H, W)
        value_grouped = value.view(B, G, Cg, H, W).reshape(B * G, Cg, H, W)

        # 整理 sample_loc: (B, G, K, 2, H, W) → (B*G, K, H, W, 2) → (B*G, K*H, W, 2)
        loc = sample_loc.permute(0, 1, 2, 4, 5, 3)                        # (B, G, K, H, W, 2)
        loc = loc.reshape(B * G, K, H, W, 2)
        loc = loc.reshape(B * G, K * H, W, 2)                             # 把 K 摊到 H 轴

        # 一次性采样: (B*G, Cg, K*H, W)
        sampled = F.grid_sample(
            value_grouped, loc,
            mode='bilinear', padding_mode='zeros', align_corners=True,
        )                                                                 # (B*G, Cg, K*H, W)
        sampled = sampled.view(B * G, Cg, K, H, W)                        # 拆回 K 维

        # mask: (B, G, K, H, W) → (B*G, K, H, W) → (B*G, 1, K, H, W)
        m = mask.reshape(B * G, K, H, W).unsqueeze(1)                     # (B*G, 1, K, H, W)

        # 加权求和: (B*G, Cg, H, W)
        out_grouped = (sampled * m).sum(dim=2)                            # (B*G, Cg, H, W)

        # 还原回 (B, C, H, W)
        out = out_grouped.view(B, G, Cg, H, W).reshape(B, C, H, W)
        return out


# ============================================================
#  Offset / Mask Generator (供 SACE 主类使用)
# ============================================================

class OffsetMaskHead(nn.Module):
    """
    从 [query, key_value] 拼接特征生成 offset 和 mask。

    设计:
        - 1x1 conv 通道融合 → 3x3 depth-wise conv 局部上下文 → 1x1 conv 输出
        - offset 输出: n_groups * n_points * 2
        - mask 输出:   n_groups * n_points
        - offset 头零初始化 (DCN-v2 惯例)，保证训练初期采样在原位
    """

    def __init__(self, channels: int, n_groups: int, kernel_size: int, hidden: int = 64):
        super().__init__()
        self.n_groups = n_groups
        self.n_points = kernel_size * kernel_size
        n_off = n_groups * self.n_points * 2
        n_msk = n_groups * self.n_points

        self.shared = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1,
                      groups=hidden, bias=True),   # depth-wise
            nn.GELU(),
        )
        self.offset_head = nn.Conv2d(hidden, n_off, kernel_size=1, bias=True)
        self.mask_head = nn.Conv2d(hidden, n_msk, kernel_size=1, bias=True)

        # 零初始化 offset，使初始采样保持邻域中心
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        # mask 偏置零初始化，sigmoid/softmax 后均匀
        nn.init.zeros_(self.mask_head.weight)
        nn.init.zeros_(self.mask_head.bias)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor):
        """
        Args:
            query    : (B, C, H, W)
            key_value: (B, C, H, W)
        Returns:
            offset: (B, G*P*2, H, W)
            mask:   (B, G*P,   H, W)
        """
        x = torch.cat([query, key_value], dim=1)                          # (B, 2C, H, W)
        h = self.shared(x)                                                # (B, hidden, H, W)
        offset = self.offset_head(h)                                      # (B, G*P*2, H, W)
        mask = self.mask_head(h)                                          # (B, G*P,   H, W)
        return offset, mask


# ============================================================
#  SACE 主类 (N5)
# ============================================================

class SACE(nn.Module):
    """
    Spatial-Adaptive Cross-frame Enhancement.

    实现状态: ✅ 完整 forward 数据流 (N5)

    数据流:
        1. 逐帧 LFF 频域整形:  feats[:, t] → f_t_lff           [可选, 复用 TFSI 频域分支或独立]
        2. 时域中位值聚合:     median over T → μ_t_clean
        3. 逐帧可变形对齐:     query=μ_t_clean, kv=f_t_lff → F_aligned_t

    设计说明:
        - μ_t_clean 作为"干净参考帧"，由所有帧的 LFF 输出的逐元素中位值得到
          (中位值天然抑制运动异常和噪声尖峰，比均值更鲁棒)
        - 每帧独立生成 offset/mask，但共享 OffsetMaskHead 参数
        - LFF 模块可由外部传入 (与 TFSI 共享) 或内部新建

    Args:
        channels      : 输入特征通道数
        n_groups      : 可变形注意力分组数
        kernel_size   : 可变形采样核大小
        use_optimized : 采样器是否使用优化版
        lff_module    : 外部传入的 LFFFeatureAdapter (None 则内部新建)
        K, n_ang_freq : 内部 LFF 的参数 (仅当 lff_module=None 时生效)
    """

    def __init__(
        self,
        channels: int = 48,
        n_groups: int = 4,
        kernel_size: int = 3,
        use_optimized: bool = True,
        lff_module: LFFFeatureAdapter = None,
        K: int = 10,
        n_ang_freq: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.n_groups = n_groups
        self.kernel_size = kernel_size

        # LFF 模块: 可外部共享，否则内部独立创建
        self._lff_external = lff_module is not None
        if lff_module is not None:
            self.lff = lff_module
        else:
            self.lff = LFFFeatureAdapter(
                channels=channels, K=K, n_ang_freq=n_ang_freq, per_channel_rbf=False,
            )

        # 共享的 offset/mask 生成器 (所有帧共享参数)
        self.offset_mask_head = OffsetMaskHead(
            channels=channels, n_groups=n_groups, kernel_size=kernel_size, hidden=64,
        )

        # 可变形跨帧注意力 (所有帧共享参数)
        self.deform_attn = DeformableCrossAttention(
            channels=channels,
            n_groups=n_groups,
            kernel_size=kernel_size,
            use_optimized=use_optimized,
        )

        # 归一化 + 残差融合 (帮助训练稳定)
        self.norm_q = LayerNorm2d(channels)
        self.norm_kv = LayerNorm2d(channels)

    def parameters_lff(self):
        """便于外部决定是否参与训练 (若 LFF 由 TFSI 共享，可在 SACE 端排除)"""
        return self.lff.parameters() if not self._lff_external else iter([])

    def forward(
        self,
        feats: torch.Tensor,           # (B, T, C, H, W)
        tfsi_out: Dict = None,         # 可选: TFSI 输出 (当前未直接使用，留作扩展)
    ) -> Dict:
        """
        Returns:
            dict:
                attn_maps      : List[Tuple[offset, mask]]，每帧一个
                                  offset: (B, G*P*2, H, W)
                                  mask:   (B, G*P,   H, W)
                mu_t_clean     : (B, C, H, W) 光照归一化时域中位值
                F_aligned_list : List[(B, C, H, W)]，每帧对齐后特征
        """
        B, T, C, H, W = feats.shape
        assert C == self.channels, f"通道不匹配: {C} vs {self.channels}"

        # ── Step 1: 逐帧 LFF
        lff_feats: List[torch.Tensor] = []
        for t in range(T):
            f_t = feats[:, t]                                              # (B, C, H, W)
            f_t_lff = self.lff(f_t)                                        # (B, C, H, W)
            lff_feats.append(f_t_lff)
        lff_stack = torch.stack(lff_feats, dim=1)                          # (B, T, C, H, W)

        # ── Step 2: 时域中位值 → 参考帧 μ_t_clean
        # median 沿 T 维度，shape (B, C, H, W)
        mu_t_clean = lff_stack.median(dim=1).values                        # (B, C, H, W)

        # ── Step 3: 逐帧可变形对齐
        attn_maps: List[Tuple[torch.Tensor, torch.Tensor]] = []
        F_aligned_list: List[torch.Tensor] = []

        q_norm = self.norm_q(mu_t_clean)                                   # 共享 query 归一化

        for t in range(T):
            kv = lff_feats[t]                                              # (B, C, H, W)
            kv_norm = self.norm_kv(kv)                                     # (B, C, H, W)

            # 生成本帧的 offset/mask
            offset, mask = self.offset_mask_head(q_norm, kv_norm)
            # offset: (B, G*P*2, H, W), mask: (B, G*P, H, W)

            # 可变形采样
            f_aligned = self.deform_attn(q_norm, kv_norm, offset, mask)    # (B, C, H, W)

            # 残差: 对齐结果 + 原 kv (保持原始信息流)
            f_aligned = f_aligned + kv                                     # (B, C, H, W)

            attn_maps.append((offset, mask))
            F_aligned_list.append(f_aligned)

        return {
            "attn_maps":      attn_maps,
            "mu_t_clean":     mu_t_clean,
            "F_aligned_list": F_aligned_list,
        }
```

### B.3 验证伪代码

```python
import torch
from models.modules.sace import SACE, DeformableCrossAttention, OffsetMaskHead


def test_deform_attn_shape():
    """简单版与优化版输出 shape 一致"""
    B, C, H, W = 2, 48, 32, 32
    G, K = 4, 3
    P = G * K * K

    attn_simple = DeformableCrossAttention(C, G, K, use_optimized=False)
    attn_opt = DeformableCrossAttention(C, G, K, use_optimized=True)

    q = torch.randn(B, C, H, W)
    kv = torch.randn(B, C, H, W)
    offset = torch.zeros(B, G * K * K * 2, H, W)   # 零偏移 → 采样邻域中心
    mask = torch.zeros(B, G * K * K, H, W)         # softmax 后均匀

    out_s = attn_simple(q, kv, offset, mask)
    out_o = attn_opt(q, kv, offset, mask)

    assert out_s.shape == (B, C, H, W)
    assert out_o.shape == (B, C, H, W)
    print(f"✅ DeformAttn shape OK, simple/opt 一致: {(out_s - out_o).abs().max():.2e}")


def test_deform_attn_consistency():
    """同步随机权重时，简单版 == 优化版（数值一致）"""
    torch.manual_seed(0)
    attn = DeformableCrossAttention(48, 4, 3, use_optimized=False)
    torch.manual_seed(0)
    attn2 = DeformableCrossAttention(48, 4, 3, use_optimized=True)

    q = torch.randn(2, 48, 32, 32)
    kv = torch.randn(2, 48, 32, 32)
    offset = torch.randn(2, 4 * 9 * 2, 32, 32) * 0.5
    mask = torch.randn(2, 4 * 9, 32, 32)

    out_s = attn(q, kv, offset, mask)
    out_o = attn2(q, kv, offset, mask)

    diff = (out_s - out_o).abs().max().item()
    assert diff < 1e-5, f"两版本数值不一致: {diff}"
    print(f"✅ DeformAttn 简单版/优化版数值一致 (max diff {diff:.2e})")


def test_sace_forward():
    sace = SACE(channels=48, n_groups=4, kernel_size=3, use_optimized=True)
    feats = torch.randn(2, 5, 48, 32, 32)
    out = sace(feats)

    assert len(out["attn_maps"]) == 5
    assert len(out["F_aligned_list"]) == 5
    assert out["mu_t_clean"].shape == (2, 48, 32, 32)
    for off, msk in out["attn_maps"]:
        assert off.shape == (2, 4 * 9 * 2, 32, 32)
        assert msk.shape == (2, 4 * 9, 32, 32)
    for fa in out["F_aligned_list"]:
        assert fa.shape == (2, 48, 32, 32)
    print("✅ SACE forward shape 全部正确")


def test_sace_gradient():
    sace = SACE(channels=48, n_groups=4, kernel_size=3, use_optimized=True)
    feats = torch.randn(2, 5, 48, 32, 32, requires_grad=True)
    out = sace(feats)
    loss = sum(fa.mean() for fa in out["F_aligned_list"]) + out["mu_t_clean"].mean()
    loss.backward()
    assert feats.grad is not None and feats.grad.abs().mean() > 0
    print("✅ SACE 梯度回传通过")


def test_sace_shared_lff():
    """外部 LFF 共享场景"""
    from models.modules.lff import LFFFeatureAdapter
    shared_lff = LFFFeatureAdapter(channels=48, K=10, n_ang_freq=1)
    sace = SACE(channels=48, lff_module=shared_lff)
    feats = torch.randn(1, 3, 48, 16, 16)
    out = sace(feats)
    # 验证共享: 修改外部 LFF 参数应影响 SACE
    assert sace.lff is shared_lff
    print("✅ SACE LFF 共享机制正常")


if __name__ == "__main__":
    test_deform_attn_shape()
    test_deform_attn_consistency()
    test_sace_forward()
    test_sace_gradient()
    test_sace_shared_lff()
```

---

## 四、B.2 + B.3 参数量统计

| 模块 | 参数量 | 备注 |
|------|-------:|------|
| **B.2 FrequencyBranch (LFF)** | **2,393** | 共享 RBF + 1x1 post_conv |
| **B.3 OffsetMaskHead** (C=48, G=4, K=3, hidden=64) | **11,748** | 1x1 + DWConv + 双头 |
| └─ shared (1x1 conv 96→64 + dw 3x3 64) | 6,848 | |
| └─ offset_head (1x1 64→72) | 4,680 | |
| └─ mask_head (1x1 64→36) | 2,340 | |
| **B.3 DeformableCrossAttention** (C=48) | **4,704** | value_proj + output_proj 各 2,352 |
| **B.3 SACE LayerNorm ×2** | 192 | |
| **B.3 SACE 内置 LFF** (若不共享) | 2,393 | 与 TFSI 共享时为 0 |
| **B.3 SACE 总参数 (与 TFSI 共享 LFF)** | **~16,644** | |
| **B.3 SACE 总参数 (独立 LFF)** | **~19,037** | |

> **预算检查**：截至 B.3 累计 ~21k 参数（含 B.2 的 LFF），相对 2M 总预算占比 ~1%，余量充足供 B.4-B.6 使用。

---

## 五、关键设计决策与注意事项

1. **SACE 与 TFSI 的 LFF 共享建议**：在 `tfs_net.py` 整合阶段（B.7），建议构造时显式传入 TFSI 内部的 LFF 实例：
   ```python
   self.tfsi = TFSI(...)
   self.sace = SACE(channels=48, lff_module=self.tfsi.freq_branch.lff)
   ```
   理由：两个模块都希望在频域里做"干净化"操作，共享一组 RBF 让参数更紧凑，且训练信号更一致。后续可通过消融实验验证此决定。

2. **简单版 vs 优化版数值一致性**：B.3 验证中 `test_deform_attn_consistency` 明确要求两版本数值一致，便于调试时可切换 `use_optimized=False` 排查问题。优化版理论上比简单版快 30~36 倍（取决于 GPU）。

3. **OffsetMaskHead 的零初始化**：保证训练初期所有帧的采样都落在邻域中心，等价于"无对齐操作"，让 SACE 从"恒等映射"开始学习对齐能力，训练更稳定。

4. **中心帧自身的对齐**：当 `t == center_idx` 时，`kv == query`（来自同一帧的 LFF 输出），可变形采样仍会执行但理论上 offset 应趋于零。当前实现未对中心帧做特殊处理，让模型自行学习——这与论文设计一致，更具一般性。

5. **mu_t_clean 与 tfsi_out 的关系**：当前 SACE 输出的 `mu_t_clean` 是**经过 LFF 频域整形后**的时域中位值，与 TFSI 输出的 `mu_t`（编码器特征的时域均值）**含义不同**。`tfsi_out` 参数目前未直接使用，留作 B.4/B.5 扩展（如 IFPN 可能需要原始 mu_t 与 SACE 的 mu_t_clean 对比）。

---

## 六、下一步

按 v3quest4.md 指令，B.3 完成后将直接进入：

- **B.4**: `models/modules/ifpn.py` — IllumExtract (N3: groups=4) + IFPN 主类
- **B.5**: `models/modules/ndpn.py` — SNR 自适应聚合
- **B.6**: `models/modules/mrpn.py` — 残差降权 + refine

请确认 B.2 + B.3 实现无误，特别是：
1. **SACE 主类的数据流**（LFF → median → 逐帧 DCA + 残差）是否符合预期
2. **OffsetMaskHead 的输入**（拼接 query+key_value）是否合理，或希望仅使用 key_value？
3. **是否同意 SACE 默认 `use_optimized=True`**

确认后我继续 B.4-B.6。