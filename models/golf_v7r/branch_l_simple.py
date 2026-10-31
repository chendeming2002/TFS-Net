#!/usr/bin/env python3
"""
Branch-L (简化版): 光照校正 (Retinex)

与 Golf R2 的差异:
  - 去掉 illum_refine 全分辨率修正 (光照图不再单独精化)
  - 2×NAFBlock (保持)
  - Upsample3x3 (仅 at_f1=False 时使用; v8 的 at_f1=True 下不建立)
  - 保留 Retinex 物理先验 (L_t, R_t, learnable gamma) ⭐
  - 保留 residual_conv 残差补偿

## 两种工作模式 (at_f1)

| | `at_f1=False` (历史) | `at_f1=True` (v8 默认) |
|---|---|---|
| 输入 F_L | F2 尺度 `[B, C, H/2, W/2]` | F1 全分辨率 `[B, C, H, W]` |
| 光照图 | 在 H/2 上预测, 再 bilinear 上采样到 H | **直接在 H×W 上预测** |
| 特征上采样 | `Upsample3x3` | 无 (`self.upsample = None`) |

v8 的三条支路都在 F1 直出, 走右列: **本模块内不存在任何降采样或上采样**。
`F.interpolate(..., size=(H, W))` 在 `at_f1=True` 时输入已是 H×W, 是恒等空操作 ——
保留它是为了让两条路径共用同一段代码。分辨率由调用方决定, 模块内不做假设。

Input:  F_L  [B, C, H, W]   (矩阵 RWKV 注入的光照上下文; C = channels, v8 用 32)
        X_t  [B, 3, H, W]   (中心帧原图 [0,1])
Output: Y_L, L_t, R_t       (分辨率同 F_L; at_f1=True 时即 F1 全分辨率)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import NAFBlock, LayerNorm2d
from models.golf_v7r.upsample import Upsample3x3


class LBranch(nn.Module):
    """Branch-L (简化): Illumination Correction via Retinex"""

    def __init__(self, channels: int = 128, num_blocks: int = 2,
                 gamma_init: float = 2.0, out_channels: int = 3,
                 at_f1: bool = False):
        super().__init__()
        self.channels = channels
        # at_f1=True: 输入已是全分辨率, 不再上采样; 光照图直接在 F1 预测。
        self.at_f1 = bool(at_f1)

        self.refine_blocks = nn.Sequential(*[
            NAFBlock(channels) for _ in range(num_blocks)
        ])
        self.refine_norm = LayerNorm2d(channels)

        # 光照图预测 (零初始化 → 初始 sigmoid(0)=0.5, 即初始 L_t ≡ 0.5)
        self.illum_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 3, 1, 1, bias=True),
        )
        nn.init.zeros_(self.illum_head[-1].weight)
        nn.init.zeros_(self.illum_head[-1].bias)

        # 上采样 (F1 模式下不需要)
        self.upsample = None if self.at_f1 else Upsample3x3(
            in_ch=channels, out_ch=64, scale=2)

        # 可学习 gamma
        self.log_gamma = nn.Parameter(torch.tensor(float(gamma_init)).log())

        # 残差补偿 (零初始化); F1 模式下输入通道就是 channels
        res_in = channels if self.at_f1 else 64
        self.residual_conv = nn.Conv2d(res_in, out_channels, 3, 1, 1, bias=True)
        nn.init.zeros_(self.residual_conv.weight)
        nn.init.zeros_(self.residual_conv.bias)

    def forward(self, F_L: torch.Tensor, X_t: torch.Tensor) -> dict:
        # 输出分辨率取中心帧原图; at_f1=True 时 F_L 的分辨率与它相同 (都是 F1)
        H, W = X_t.shape[-2:]

        x = self.refine_norm(self.refine_blocks(F_L))

        # 光照图: 对数域预测, 避免 sigmoid 在两端饱和
        # exp(0) = 1, 与 sigmoid 的初始 0.5 不同, 但 R_t = X_t / L_t 在 L_t=1 时
        # 给出 R_t = X_t, 是恒等起点。clamp 到 [-3, 3] 限制范围在 [0.05, 20]。
        L_map = torch.exp(torch.clamp(self.illum_head(x), -3.0, 3.0))

        # 特征上采样 (F1 模式直接用全分辨率特征)
        x_up = x if self.at_f1 else self.upsample(x)

        # 对齐到输出分辨率: at_f1=True 时输入已是 H×W, 此处是恒等空操作
        L_t = F.interpolate(L_map, size=(H, W), mode='bilinear', align_corners=False)

        # Retinex
        eps = 1e-4
        R_t = (X_t / L_t.clamp(min=eps)).clamp(max=10.0)
        gamma = self.log_gamma.exp()
        Y_L = R_t * L_t.pow(gamma)
        Y_L = Y_L + self.residual_conv(x_up)
        Y_L = Y_L.clamp(0.0, 1.0)

        return {"Y_L": Y_L, "L_t": L_t, "R_t": R_t}


if __name__ == '__main__':
    m = LBranch(channels=128, num_blocks=2)
    F_L = torch.randn(2, 128, 32, 32)
    X_t = torch.rand(2, 3, 64, 64)
    out = m(F_L, X_t)
    print(f"LBranch: F_L={F_L.shape}, X_t={X_t.shape}")
    print(f"  Y_L={out['Y_L'].shape}, L_t={out['L_t'].shape}, R_t={out['R_t'].shape}")
    print(f"  params={sum(p.numel() for p in m.parameters())/1e3:.1f}K")


# 旧名别名: 历史 run 与文档记为 BranchLSimple。
BranchLSimple = LBranch
