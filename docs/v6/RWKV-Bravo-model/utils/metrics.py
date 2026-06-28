import math

import torch

from losses.losses import ssim_map


def tensor_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def tensor_ssim(pred, target):
    return ssim_map(pred, target).mean().item()
