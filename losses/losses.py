import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import VGG16_Weights, vgg16
except Exception:
    VGG16_Weights = None
    vgg16 = None


def gaussian_window(window_size=11, sigma=1.5, channels=3, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window_2d = torch.outer(gauss, gauss)
    window_2d = window_2d / window_2d.sum()
    return window_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def ssim_map(x, y, window_size=11, sigma=1.5):
    channels = x.shape[1]
    window = gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channels)
    sigma_x = F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu_x * mu_x
    sigma_y = F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return numerator / (denominator + 1e-6)


class PerceptualLoss(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.enabled = vgg16 is not None
        if not self.enabled:
            self.features = None
            warnings.warn("torchvision is not available, perceptual loss will be zero.")
            return
        weights = None
        if pretrained and VGG16_Weights is not None:
            try:
                weights = VGG16_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        try:
            backbone = vgg16(weights=weights)
        except Exception:
            backbone = vgg16(weights=None)
            warnings.warn("Unable to load pretrained VGG16 weights, using randomly initialized VGG16.")
        self.features = backbone.features[:16].eval()
        for param in self.features.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if self.features is None:
            return pred.new_tensor(0.0)
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        feat_pred = self.features(pred_norm)
        feat_target = self.features(target_norm)
        return F.l1_loss(feat_pred, feat_target)


class TotalVariationLoss(nn.Module):
    def forward(self, x):
        loss_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        loss_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return loss_h + loss_w


class MINSLoss(nn.Module):
    def __init__(self, lambda_pix=1.0, lambda_ssim=0.2, lambda_perc=0.05, lambda_tv=0.01, perceptual_pretrained=False):
        super().__init__()
        self.lambda_pix = lambda_pix
        self.lambda_ssim = lambda_ssim
        self.lambda_perc = lambda_perc
        self.lambda_tv = lambda_tv
        self.perceptual = PerceptualLoss(pretrained=perceptual_pretrained)
        self.tv = TotalVariationLoss()

    def forward(self, outputs, target):
        pred = outputs["res_t"]
        priors = outputs["priors"]

        loss_pix = F.l1_loss(pred, target)
        loss_ssim = 1.0 - ssim_map(pred, target).mean()
        loss_perc = self.perceptual(pred, target)
        loss_prior = self.tv(priors["P_t_m"]) + self.tv(priors["P_t_i"])

        total = (
            self.lambda_pix * loss_pix
            + self.lambda_ssim * loss_ssim
            + self.lambda_perc * loss_perc
            + self.lambda_tv * loss_prior
        )

        return total, {
            "loss_total": total.detach(),
            "loss_pix": loss_pix.detach(),
            "loss_ssim": loss_ssim.detach(),
            "loss_perc": loss_perc.detach(),
            "loss_prior": loss_prior.detach(),
        }

