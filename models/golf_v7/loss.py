#!/usr/bin/env python3
"""Simple Loss for GolfNet v7 (单分支架构)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SimpleLoss(nn.Module):
    """
    简化损失函数 (v7 单分支)
    
    L = lambda_base * (L1 + ssim_weight * (1 - SSIM))
    """
    def __init__(self, 
                 lambda_base: float = 1.0,
                 use_ssim: bool = True,
                 ssim_weight: float = 0.3):
        super().__init__()
        self.lambda_base = lambda_base
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight
        
        if use_ssim:
            try:
                from pytorch_msssim import SSIM
                self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
            except ImportError:
                print("Warning: pytorch_msssim not found, disabling SSIM loss")
                self.use_ssim = False
    
    def forward(self, pred: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        pred: dict with 'final' key
        target: [B, 3, H, W]
        """
        Y = pred['final']
        
        # L1 loss
        loss_l1 = F.l1_loss(Y, target)
        
        # SSIM loss
        if self.use_ssim:
            ssim_val = self.ssim(Y, target)
            loss_ssim = 1 - ssim_val
            loss_total = loss_l1 + self.ssim_weight * loss_ssim
        else:
            loss_ssim = torch.tensor(0.0)
            loss_total = loss_l1
        
        loss_total = self.lambda_base * loss_total
        
        return {
            'loss': loss_total,
            'total_loss': loss_total,
            'l1': loss_l1.detach(),
            'ssim': ssim_val.detach() if self.use_ssim else torch.tensor(0.0),
            'L_final': loss_l1.detach(),
            'L_N': loss_l1.detach(),
            'L_L': loss_l1.detach(),
            'L_M': loss_l1.detach(),
            'L_ortho': torch.tensor(0.0),
            'L_temp': torch.tensor(0.0),
            'L_div': torch.tensor(0.0),
        }


if __name__ == '__main__':
    loss_fn = SimpleLoss(lambda_base=1.0, use_ssim=True, ssim_weight=0.3)
    
    pred = {'final': torch.randn(2, 3, 256, 256)}
    target = torch.randn(2, 3, 256, 256)
    
    out = loss_fn(pred, target)
    print(f"Total loss: {out['loss'].item():.4f}")
    print(f"L1: {out['l1'].item():.4f}")
    print(f"SSIM: {out['ssim'].item():.4f}")
