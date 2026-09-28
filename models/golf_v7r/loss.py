#!/usr/bin/env python3
"""
Golf v7r Loss: 多分支监督 + 正交约束

Loss = L1(final, GT)
     + w_N * L1(Y_N, GT) + w_L * L1(Y_L, GT) + w_M * L1(Y_M, GT)
     + lambda_ortho * ortho_loss(ctx_N, ctx_L, ctx_M)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class GolfV7RLoss(nn.Module):
    """多分支 + 正交约束 loss"""

    def __init__(self,
                 w_branch_N: float = 0.1,
                 w_branch_L: float = 0.1,
                 w_branch_M: float = 0.1,
                 lambda_ortho: float = 0.01,
                 w_ssim: float = 0.0,
                 branch_warmup_epochs: int = 5):
        super().__init__()
        self.w_branch_N = w_branch_N
        self.w_branch_L = w_branch_L
        self.w_branch_M = w_branch_M
        self.lambda_ortho = lambda_ortho
        self.w_ssim = w_ssim
        self.branch_warmup_epochs = branch_warmup_epochs

        self.current_epoch = 0
        self._use_ssim = w_ssim > 0
        if self._use_ssim:
            try:
                from pytorch_msssim import ms_ssim
                self._ms_ssim = ms_ssim
            except ImportError:
                self._use_ssim = False

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _branch_weight(self, base: float) -> float:
        """分支监督权重 warmup: 前若 epoch 较少, 线性增长"""
        if base == 0:
            return 0.0
        if self.branch_warmup_epochs <= 0:
            return base
        frac = min(1.0, (self.current_epoch + 1) / self.branch_warmup_epochs)
        return base * frac

    def forward(self, output_dict: Dict[str, torch.Tensor],
                gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = output_dict['final']
        loss_l1 = F.l1_loss(pred, gt)
        loss = loss_l1

        loss_dict = {'loss': loss, 'l1': loss_l1}

        # 分支监督
        for key, base_w in [('branch_N', self.w_branch_N),
                            ('branch_L', self.w_branch_L),
                            ('branch_M', self.w_branch_M)]:
            if key in output_dict and output_dict[key] is not None:
                w = self._branch_weight(base_w)
                if w > 0:
                    bl = F.l1_loss(output_dict[key], gt)
                    loss = loss + w * bl
                    loss_dict[key] = bl

        # 正交约束 (由网络内部计算并传出)
        if 'ortho_loss' in output_dict and self.lambda_ortho > 0:
            ol = output_dict['ortho_loss']
            loss = loss + self.lambda_ortho * ol
            loss_dict['ortho'] = ol

        # MS-SSIM
        if self._use_ssim:
            ssim_val = self._ms_ssim(pred.clamp(0, 1), gt.clamp(0, 1),
                                     data_range=1.0, size_average=True)
            loss = loss + self.w_ssim * (1 - ssim_val)
            loss_dict['ms_ssim'] = ssim_val

        loss_dict['loss'] = loss
        return loss_dict


class SimpleLoss(nn.Module):
    """基线 loss (仅 L1), 用于对照"""

    def forward(self, output_dict, gt):
        pred = output_dict['final']
        return {'loss': F.l1_loss(pred, gt)}


if __name__ == '__main__':
    criterion = GolfV7RLoss()
    out = {
        'final': torch.rand(2, 3, 64, 64),
        'branch_N': torch.rand(2, 3, 64, 64),
        'branch_L': torch.rand(2, 3, 64, 64),
        'branch_M': torch.rand(2, 3, 64, 64),
        'ortho_loss': torch.tensor(0.1),
    }
    gt = torch.rand(2, 3, 64, 64)
    d = criterion(out, gt)
    print("Loss dict:", {k: f"{v.item():.4f}" for k, v in d.items()})
