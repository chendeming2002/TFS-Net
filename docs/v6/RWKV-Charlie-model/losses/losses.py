"""
损失函数 — RWKV-Charlie
=======================
Charlie 使用 Bravo2 P0 权重:
   lambda_perc = 0.04
   lambda_pix  = 1.0
   lambda_ssim = 0.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff ** 2 + self.eps ** 2).mean()


class PerceptualLoss(nn.Module):
    def __init__(self, layers: list[int] | None = None):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        if layers is None:
            layers = [3, 8, 15, 22]
        self.layers = layers
        self.blocks = nn.ModuleList()
        prev = 0
        for layer in layers:
            block = nn.Sequential(*list(vgg[prev:layer + 1]))
            self.blocks.append(block)
            prev = layer + 1
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] == 3:
            pred_rgb = pred
            target_rgb = target
        else:
            pred_rgb = pred[:, :3, :, :]
            target_rgb = target[:, :3, :, :]
        pred_feats = self._extract(pred_rgb)
        target_feats = self._extract(target_rgb)
        loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            loss += F.l1_loss(pf, tf)
        return loss / len(pred_feats)

    def _extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.window = self._create_window(window_size)

    def _create_window(self, window_size: int) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        window = gauss[:, None] * gauss[None, :]
        window /= window.sum()
        return window.view(1, 1, window_size, window_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = pred.shape[1]
        window = self.window.to(pred.device, dtype=pred.dtype)
        window = window.expand(C, 1, self.window_size, self.window_size)
        mu1 = F.conv2d(pred, window, groups=C)
        mu2 = F.conv2d(target, window, groups=C)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(pred ** 2, window, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, groups=C) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, groups=C) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if self.size_average:
            return 1 - ssim.mean()
        return 1 - ssim.mean(dim=(1, 2, 3))


class TFSNetLoss(nn.Module):
    """Charlie: Bravo2 P0 权重 lambda_perc=0.04, lambda_pix=1.0, lambda_ssim=0.2"""

    def __init__(self, lambda_pix: float = 1.0, lambda_perc: float = 0.04, lambda_ssim: float = 0.2):
        super().__init__()
        self.lambda_pix = lambda_pix
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.pix_loss = CharbonnierLoss()
        self.perc_loss = PerceptualLoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, deep_supervision: list | None = None) -> torch.Tensor:
        loss = self.lambda_pix * self.pix_loss(pred, target)
        loss += self.lambda_perc * self.perc_loss(pred, target)
        loss += self.lambda_ssim * self.ssim_loss(pred, target)
        if deep_supervision is not None:
            for ds_pred in deep_supervision:
                loss += 0.5 * self.lambda_pix * self.pix_loss(ds_pred, target)
        return loss
