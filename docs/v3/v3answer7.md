# TFS-Net v3 — B.7 修正 + B.8/B.9 实现

---

## 一、B.7 修正：`models/tfs_net.py` 完整版

```python
"""
TFS-Net v3 — Three-source Fusion & Synthesis Network
======================================================
端到端多帧低光增强网络。

整体结构 (5 stages):
    Stage 0: PyramidEncoder        多帧 → 多尺度特征 + 全分辨率融合
    Stage 1: TFSI                  时序光照/噪声/运动强度场估计 + 频/空双分支
    Stage 2: SACE                  可变形跨帧对齐 (与 TFSI 共享 LFF)
    Stage 3: IFPN/NDPN/MRPN        三源恢复分支 (光照/噪声/运动)
    Stage 4: IGRF                  强度引导残差融合 → 输出
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.encoder import PyramidEncoder
from models.modules.tfsi import TFSI
from models.modules.sace import SACE
from models.modules.ifpn import IFPN
from models.modules.ndpn import NDPN
from models.modules.mrpn import MRPN
from models.modules.igrf import IGRF


class TFSNet(nn.Module):
    """
    Args:
        in_channels    : 输入图像通道 (默认 3)
        level_channels : 编码器三尺度通道 (默认 (32, 64, 96))
        fused_channels : 编码器融合输出通道，也是后续所有模块的特征通道 (默认 48)
        eps            : TFSI 的数值稳定项
        n_groups       : SACE 可变形分组数
        kernel_size    : SACE 可变形核大小
        share_lff      : SACE 是否与 TFSI 共享 LFF (默认 True)
    """

    def __init__(
        self,
        in_channels: int = 3,
        level_channels: Tuple[int, int, int] = (32, 64, 96),
        fused_channels: int = 48,
        eps: float = 1e-6,
        n_groups: int = 4,
        kernel_size: int = 3,
        share_lff: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.fused_channels = fused_channels
        self.share_lff = share_lff
        c1, c2, c3 = level_channels

        # ── Stage 0: PyramidEncoder（使用正确接口）──
        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            level_channels=level_channels,
            fused_channels=fused_channels,
        )

        # ── Stage 1: TFSI ──
        self.tfsi = TFSI(
            channels=fused_channels,
            fused_channels=fused_channels,
            eps=eps,
        )

        # ── Stage 2: SACE（与 TFSI 共享 LFF）──
        shared_lff = self.tfsi.freq_branch.lff if share_lff else None
        self.sace = SACE(
            channels=fused_channels,
            n_groups=n_groups,
            kernel_size=kernel_size,
            use_optimized=True,
            lff_module=shared_lff,
        )

        # ── Stage 3: 三源恢复分支 ──
        self.ifpn = IFPN(
            fused_channels=fused_channels,
            coarse_channels=c3,            # 粗尺度通道 = 96
            img_channels=in_channels,
        )
        self.ndpn = NDPN(channels=fused_channels)
        self.mrpn = MRPN(channels=fused_channels)

        # ── Stage 4: IGRF（使用正确接口）──
        self.igrf = IGRF(channels=fused_channels, out_channels=in_channels)

    # ─────────────────────────────────────────────────────────
    #  forward
    # ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, 3, H, W) 多帧低光输入

        Returns:
            dict: 见末尾
        """
        B, T, C_in, H, W = x.shape
        center_idx = T // 2

        # ── Stage 0: 编码（正确的 API：return_coarse=True 同时取两个尺度）──
        feats, coarse_feats = self.encoder(x, return_coarse=True)
        # feats        : (B, T, 48, H, W)        ← 全分辨率融合，无需上采样
        # coarse_feats : (B, T, 96, H/4, W/4)

        # ── Stage 1: TFSI ──
        tfsi_out = self.tfsi(feats)
        F_fused  = tfsi_out["F_fused"]        # (B, 48, H, W)
        s_illum  = tfsi_out["s_illum"]        # (B, 1, H, W)
        s_noise  = tfsi_out["s_noise"]        # (B, 1, H, W)
        s_motion = tfsi_out["s_motion"]       # (B, 1, H, W)
        sigma_t  = tfsi_out["sigma_t"]        # (B, 48, H, W)

        # ── Stage 2: SACE ──
        sace_out       = self.sace(feats, tfsi_out)
        mu_t_clean     = sace_out["mu_t_clean"]
        F_aligned_list = sace_out["F_aligned_list"]
        attn_maps      = sace_out["attn_maps"]

        # ── Stage 3: 三源恢复 ──
        image_center = x[:, center_idx]                       # (B, 3, H, W)

        # 下采样图像到 coarse 分辨率（H/4）
        _, _, _, hc, wc = coarse_feats.shape
        image_down = F.interpolate(
            image_center, size=(hc, wc), mode='bicubic', align_corners=False
        )                                                      # (B, 3, hc, wc)
        imgs_down = F.interpolate(
            x.view(B * T, C_in, H, W),
            size=(hc, wc), mode='bicubic', align_corners=False,
        ).view(B, T, C_in, hc, wc)                             # (B, T, 3, hc, wc)

        F_t_L = coarse_feats[:, center_idx]                    # (B, 96, hc, wc)

        ifpn_out = self.ifpn(
            I_t_down=image_down,
            F_t_L=F_t_L,
            s_illum=s_illum,
            feats=feats,
            coarse_feats=coarse_feats,
            center_idx=center_idx,
            imgs_down=imgs_down,
        )

        ndpn_out = self.ndpn(
            feats=feats,
            F_aligned_list=F_aligned_list,
            mu_t_clean=mu_t_clean,
            sigma_t=sigma_t,
            s_noise=s_noise,
            center_idx=center_idx,
        )

        mrpn_out = self.mrpn(
            feats=feats,
            F_aligned_list=F_aligned_list,
            s_motion=s_motion,
            center_idx=center_idx,
        )

        # ── Stage 4: IGRF ──
        igrf_out = self.igrf(
            f_t_base=F_fused,
            f_illum_out=ifpn_out["f_illum_out"],
            f_noise_out=ndpn_out["f_noise_out"],
            f_motion_out=mrpn_out["f_motion_out"],
            s_illum=s_illum,
            s_noise=s_noise,
            s_motion=s_motion,
            image_center=image_center,
        )

        return {
            # 主输出
            "res_t":          igrf_out["res_t"],
            "delta":          igrf_out["delta"],
            "f_fused_igrf":   igrf_out["f_fused"],
            # 强度场
            "s_illum":        s_illum,
            "s_noise":        s_noise,
            "s_motion":       s_motion,
            # 调试 / 监督辅助
            "L_t":            ifpn_out["L_t"],
            "L_ref":          ifpn_out["L_ref"],
            "L_ratio":        ifpn_out["L_ratio"],
            "attn_maps":      attn_maps,
            "mu_t_clean":     mu_t_clean,
            "s_snr":          ndpn_out["s_snr"],
            "motion_weights": mrpn_out["motion_weights"],
            "tfsi_out":       tfsi_out,
        }
```

### B.7 修正点总结

| Bug | 修正 |
|-----|------|
| P0-1 编码器构造参数 | `level_channels=(32,64,96), fused_channels=48`（删除 `base_channels/c1/c2/c3`） |
| P0-2 编码器返回值 | `feats, coarse_feats = encoder(x, return_coarse=True)`（无 dict/tuple 兼容） |
| P0-2-附属 | 删除 `feat_full` 上采样逻辑（feats 已是全分辨率） |
| P0-3 IGRF 构造参数 | `out_channels=in_channels`（替换 `img_channels`） |

---

## 二、B.8 实现：`losses/losses.py`（追加 TFSNetLoss）

> **重要约束**：仅在 `losses/losses.py` 末尾追加，不修改 `MINSLoss`/`PerceptualLoss`/`ssim_map` 等现有代码。

```python
# =================================================================
#  TFSNetLoss — TFS-Net v3 损失函数
#  追加在 losses.py 末尾，不修改既有代码
# =================================================================

class TFSNetLoss(nn.Module):
    """
    TFS-Net v3 损失函数（首版 use_temporal=False）

    L_total = L_recon + λ_perc * L_perc + λ_illum * L_illum
        L_recon = L_pix + λ_freq * L_freq
        L_pix   = L1(pred, target)
        L_freq  = L1(|FFT(pred)|, |FFT(target)|)
        L_perc  = PerceptualLoss(pred, target)         [复用现有 PerceptualLoss]
        L_illum = edge-aware smoothness on s_illum

    Args:
        use_temporal           : 是否启用时序一致性（首版置 False，预留）
        use_freq_loss          : 是否启用频域重建
        perceptual_pretrained  : PerceptualLoss 是否加载预训练权重
        lambda_perc            : 感知损失权重
        lambda_freq            : 频域损失权重
        lambda_illum           : 光照平滑权重
    """

    def __init__(
        self,
        use_temporal: bool = False,
        use_freq_loss: bool = True,
        perceptual_pretrained: bool = False,
        lambda_perc: float = 0.1,
        lambda_freq: float = 0.1,
        lambda_illum: float = 0.01,
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_freq_loss = use_freq_loss
        self.lambda_perc = lambda_perc
        self.lambda_freq = lambda_freq
        self.lambda_illum = lambda_illum

        # 复用现有 PerceptualLoss
        self.perceptual = PerceptualLoss(pretrained=perceptual_pretrained)

    # ----------------------------------------------------------
    #  辅助：边缘感知的光照平滑
    # ----------------------------------------------------------
    @staticmethod
    def _edge_aware_smooth(s: torch.Tensor, ref_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s       : (B, 1, H, W) 单通道强度图
            ref_img : (B, 3, H, W) RGB 参考图（用于计算图像梯度作权重）

        Returns:
            scalar loss
        """
        grad_s_x = (s[:, :, :, 1:] - s[:, :, :, :-1]).abs()                 # (B,1,H,W-1)
        grad_s_y = (s[:, :, 1:, :] - s[:, :, :-1, :]).abs()                 # (B,1,H-1,W)

        # 参考图梯度（通道平均）
        grad_i_x = (ref_img[:, :, :, 1:] - ref_img[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        grad_i_y = (ref_img[:, :, 1:, :] - ref_img[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)

        loss = (grad_s_x * torch.exp(-grad_i_x)).mean() + \
               (grad_s_y * torch.exp(-grad_i_y)).mean()
        return loss

    # ----------------------------------------------------------
    #  forward
    # ----------------------------------------------------------
    def forward(self, outputs: dict, target: torch.Tensor):
        """
        Args:
            outputs : dict from TFSNet.forward()
                需要的键: 'res_t' (B,3,H,W), 's_illum' (B,1,H,W)
            target  : (B, 3, H, W) GT

        Returns:
            (loss_total, loss_dict)
            loss_dict 键: loss_total / loss_pix / loss_freq / loss_perc / loss_illum
        """
        pred = outputs["res_t"]
        s_illum = outputs["s_illum"]

        # (1) 空间像素重建
        L_pix = F.l1_loss(pred, target)

        # (2) 频域重建（可选）
        if self.use_freq_loss:
            # rfft2 在最后两维做实数 FFT；用 norm='ortho' 保证尺度无关
            fft_pred = torch.fft.rfft2(pred, norm='ortho')
            fft_gt = torch.fft.rfft2(target, norm='ortho')
            L_freq = F.l1_loss(fft_pred.abs(), fft_gt.abs())
        else:
            L_freq = pred.new_tensor(0.0)

        L_recon = L_pix + self.lambda_freq * L_freq

        # (3) 感知损失（复用现有 PerceptualLoss）
        L_perc = self.perceptual(pred, target)

        # (4) 光照场边缘感知平滑（参考 GT 边缘）
        L_illum = self._edge_aware_smooth(s_illum, target)

        # 总损失
        L_total = L_recon + self.lambda_perc * L_perc + self.lambda_illum * L_illum

        loss_dict = {
            "loss_total": L_total.detach(),
            "loss_pix":   L_pix.detach(),
            "loss_freq":  L_freq.detach(),
            "loss_perc":  L_perc.detach(),
            "loss_illum": L_illum.detach(),
        }
        return L_total, loss_dict
```

### B.8 验证伪代码

```python
def test_tfsnetloss_basic():
    from models import TFSNet
    from losses.losses import TFSNetLoss

    net = TFSNet()
    criterion = TFSNetLoss(use_freq_loss=True, perceptual_pretrained=False)
    
    x = torch.randn(2, 5, 3, 128, 128)
    target = torch.randn(2, 3, 128, 128)
    
    out = net(x)
    loss, loss_dict = criterion(out, target)
    
    assert loss.dim() == 0, "loss 应该是标量"
    for k in ["loss_total", "loss_pix", "loss_freq", "loss_perc", "loss_illum"]:
        assert k in loss_dict, f"缺失 {k}"
    
    loss.backward()
    print(f"✅ TFSNetLoss 基础测试: total={loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"   {k}: {v.item():.4f}")


def test_tfsnetloss_no_freq():
    from losses.losses import TFSNetLoss
    criterion = TFSNetLoss(use_freq_loss=False)
    outputs = {
        "res_t": torch.randn(1, 3, 64, 64, requires_grad=True),
        "s_illum": torch.rand(1, 1, 64, 64, requires_grad=True),
    }
    loss, loss_dict = criterion(outputs, torch.randn(1, 3, 64, 64))
    assert loss_dict["loss_freq"].item() == 0.0
    print("✅ TFSNetLoss use_freq_loss=False 模式正确")
```

---

## 三、B.9 实现：配置与脚本适配

### 3.1 新版配置文件 `configs/sdsd_stage1.yaml`

```yaml
seed: 42
output_dir: outputs/sdsd_stage1

dataset:
  train_input_root: F:/DatasetDL/SDSD/indoor/input
  train_target_root: F:/DatasetDL/SDSD/indoor/GT
  val_input_root: F:/DatasetDL/SDSD/test/low-light
  val_target_root: F:/DatasetDL/SDSD/test/GT
  window_size: 5
  crop_size: 256
  num_workers: 4

model:
  type: TFSNet
  in_channels: 3
  level_channels: [32, 64, 96]
  fused_channels: 48

train:
  batch_size: 2
  epochs: 200
  lr: 0.0002
  weight_decay: 0.0001
  amp: false
  log_interval: 50
  val_interval: 5

loss:
  type: TFSNetLoss
  use_freq_loss: true
  perceptual_pretrained: false
  lambda_perc: 0.1
  lambda_freq: 0.1
  lambda_illum: 0.01

eval:
  tile_size: 256
  tile_overlap: 32
  amp: false
```

---

### 3.2 `models/modules/__init__.py` 完整版

```python
"""模块注册表 — 同时支持 v1 (MINSNet) 与 v3 (TFSNet)"""

# v1 既有模块（保留）
from .encoder import PyramidEncoder
from .mins import MINSBlock
from .ispn import ISPN
from .mspn import MSPN
from .reconstruction import FinalReconstruction

# v3 新增模块
from .tfsi import TFSI
from .igrf import IGRF
from .lff import RadialBasisFilter, LFFFeatureAdapter
from .sace import SACE, DeformableCrossAttention, OffsetMaskHead
from .ifpn import IFPN, IllumExtract
from .ndpn import NDPN
from .mrpn import MRPN

__all__ = [
    # v1
    "PyramidEncoder", "MINSBlock", "ISPN", "MSPN", "FinalReconstruction",
    # v3
    "TFSI", "IGRF",
    "RadialBasisFilter", "LFFFeatureAdapter",
    "SACE", "DeformableCrossAttention", "OffsetMaskHead",
    "IFPN", "IllumExtract",
    "NDPN", "MRPN",
]
```

---

### 3.3 `models/__init__.py` 完整版

```python
"""模型注册表"""

from .mins_net import MINSNet              # v1
from .tfs_net import TFSNet                # v3

__all__ = ["MINSNet", "TFSNet"]
```

> 假设原文件已有 `MINSNet`，仅需追加 TFSNet 导入。

---

### 3.4 `losses/__init__.py` 完整版

```python
"""损失函数注册表"""

from .losses import (
    MINSLoss,
    TFSNetLoss,
    PerceptualLoss,
    ssim_map,
)

__all__ = ["MINSLoss", "TFSNetLoss", "PerceptualLoss", "ssim_map"]
```

---

### 3.5 `train.py` 完整重写版

> **设计原则**：保留原有训练循环骨架，但根据 `cfg.model.type` 和 `cfg.loss.type` 动态选择模型/损失。这样既能跑 v1（MINSNet+MINSLoss）也能跑 v3（TFSNet+TFSNetLoss），由配置文件切换。

```python
"""
TFS-Net v3 / MINSNet v1 共用训练脚本
====================================

通过配置文件 model.type 与 loss.type 字段切换:
    - MINSNet  + MINSLoss   (v1)
    - TFSNet   + TFSNetLoss (v3)

用法:
    python train.py --config configs/sdsd_stage1.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from datasets.sdsd import SDSDDataset
from models import MINSNet, TFSNet
from losses.losses import MINSLoss, TFSNetLoss
from utils.metrics import psnr, ssim
from utils.misc import AverageMeter