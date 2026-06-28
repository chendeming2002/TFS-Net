"""
TFSNetLoss — RMS-RWKV Bravo 损失函数
======================================
Bravo P2: BVI-Lowlight (arXiv 2024) 风格重新加权
  - L1:       λ=1.0 → 0.3 (L1 对未对齐最敏感, 降权)
  - VGG Perceptual: λ=0.2 → 0.8 (占主导, 容错对齐偏差)
  - SSIM:    λ=0.1 → 0.5 (最佳容错)
  - Focal Frequency: λ=0.2 (FAN IET IP 2025 新增)
"""

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
    def __init__(self, pretrained=False, multilayer=True, layer_weights=None):
        super().__init__()
        self.enabled = vgg16 is not None
        self.multilayer = multilayer
        if not self.enabled:
            self.features = None
            warnings.warn("torchvision not available, perceptual loss will be zero.")
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
            warnings.warn("Unable to load pretrained VGG16, using random init.")

        if multilayer:
            self.layer1 = backbone.features[:4]
            self.layer2 = backbone.features[4:9]
            self.layer3 = backbone.features[9:16]
            self.weights = layer_weights or [0.1, 0.2, 0.5]
            self.features = None
            for mod in [self.layer1, self.layer2, self.layer3]:
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
        else:
            self.features = backbone.features[:16].eval()
            for param in self.features.parameters():
                param.requires_grad = False
            self.layer1 = self.layer2 = self.layer3 = None

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if not self.enabled:
            return pred.new_tensor(0.0)
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        if self.multilayer and self.layer1 is not None:
            f1_pred = self.layer1(pred_norm)
            f1_tgt = self.layer1(target_norm)
            f2_pred = self.layer2(f1_pred)
            f2_tgt = self.layer2(f1_tgt)
            f3_pred = self.layer3(f2_pred)
            f3_tgt = self.layer3(f2_tgt)
            loss = (self.weights[0] * F.l1_loss(f1_pred, f1_tgt)
                    + self.weights[1] * F.l1_loss(f2_pred, f2_tgt)
                    + self.weights[2] * F.l1_loss(f3_pred, f3_tgt))
            return loss
        else:
            feat_pred = self.features(pred_norm)
            feat_target = self.features(target_norm)
            return F.l1_loss(feat_pred, feat_target)


def charbonnier_loss(pred, target, eps=1e-6):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps))


class TFSNetLoss(nn.Module):
    """
    Bravo P2 默认权重 (BVI-Lowlight arXiv 2024 + FAN 2025):
      lambda_pix=0.3, lambda_perc=0.8, lambda_ssim=0.5, lambda_focal=0.2
    """

    def __init__(
        self,
        use_freq_loss: bool = True,
        perceptual_pretrained: bool = False,
        # Bravo P2 默认值:
        lambda_pix: float = 1.0,
        lambda_perc: float = 0.1,
        lambda_freq: float = 0.1,
        lambda_illum: float = 0.001,
        lambda_ssim: float = 0.2,
        lambda_inter: float = 0.3,
        lambda_illum_sup: float = 0.1,
        lambda_noise_sup: float = 0.05,
        lambda_recon: float = 0.5,
        noise_tau_high: float = 5.0,
        perc_multilayer: bool = True,
        freq_with_phase: bool = True,
        freq_phase_weight: float = 0.5,
        lambda_ifpn_sup: float = 0.0,
        # Bravo P2 新增
        lambda_focal: float = 0.2,
    ):
        super().__init__()
        self.use_freq_loss = use_freq_loss
        self.lambda_pix = lambda_pix
        self.lambda_perc = lambda_perc
        self.lambda_freq = lambda_freq
        self.lambda_illum = lambda_illum
        self.lambda_ssim = lambda_ssim
        self.lambda_inter = lambda_inter
        self.lambda_illum_sup = lambda_illum_sup
        self.lambda_noise_sup = lambda_noise_sup
        self.noise_tau_high = noise_tau_high
        self.freq_with_phase = freq_with_phase
        self.freq_phase_weight = freq_phase_weight
        self.lambda_ifpn_sup = lambda_ifpn_sup
        self.lambda_focal = lambda_focal

        self.perceptual = PerceptualLoss(pretrained=perceptual_pretrained, multilayer=perc_multilayer)

    @staticmethod
    def _edge_aware_smooth(s, ref_img):
        grad_s_x = (s[:, :, :, 1:] - s[:, :, :, :-1]).abs()
        grad_s_y = (s[:, :, 1:, :] - s[:, :, :-1, :]).abs()
        grad_i_x = (ref_img[:, :, :, 1:] - ref_img[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        grad_i_y = (ref_img[:, :, 1:, :] - ref_img[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
        loss = (grad_s_x * torch.exp(-grad_i_x)).mean() + (grad_s_y * torch.exp(-grad_i_y)).mean()
        return loss

    @staticmethod
    def _focal_freq_loss(pred, target, sigma=0.1):
        """
        Bravo P2: Focal Frequency Loss (FAN IET IP 2025)
        对频率域中差异大的区域 (边缘/纹理) 施加更大权重.
        """
        fft_pred = torch.fft.rfft2(pred, norm='ortho')
        fft_tgt = torch.fft.rfft2(target, norm='ortho')
        freq_diff = (fft_pred.abs() - fft_tgt.abs()).abs()
        weight = (1.0 - torch.exp(-sigma * freq_diff.detach())) + 1.0
        return (weight * freq_diff).mean()

    def forward(self, outputs, target):
        pred = outputs["res_t"]
        s_illum = outputs["s_illum"]
        s_noise = outputs["s_noise"]

        # (1) 空间像素重建
        L_pix = charbonnier_loss(pred, target)

        # (2) 频域重建
        if self.use_freq_loss:
            fft_pred = torch.fft.rfft2(pred, norm='ortho')
            fft_gt = torch.fft.rfft2(target, norm='ortho')
            if self.freq_with_phase:
                L_freq = F.l1_loss(fft_pred.abs(), fft_gt.abs()) + \
                         self.freq_phase_weight * F.l1_loss(fft_pred.angle(), fft_gt.angle())
            else:
                L_freq = F.l1_loss(fft_pred.abs(), fft_gt.abs())
        else:
            L_freq = pred.new_tensor(0.0)

        # (2b) Bravo P2: Focal Frequency Loss
        L_focal = self._focal_freq_loss(pred, target)

        L_recon_base = self.lambda_pix * L_pix + self.lambda_freq * L_freq + self.lambda_focal * L_focal

        # (3) SSIM
        L_ssim = 1.0 - ssim_map(pred, target).mean()

        # (4) Perceptual
        L_perc = self.perceptual(pred, target)

        # (5) 光照平滑
        L_illum_smooth = self._edge_aware_smooth(s_illum, target)

        # (5b) s_illum 显式监督
        L_illum_sup = pred.new_tensor(0.0)
        if self.lambda_illum_sup > 0 and "image_center" in outputs:
            img_center = outputs["image_center"]
            eps = 1e-3
            img_gray = img_center.mean(dim=1, keepdim=True)
            gt_gray = target.mean(dim=1, keepdim=True)
            s_illum_target = torch.clamp(1.0 - img_gray / (gt_gray + eps), 0.0, 1.0).detach()
            L_illum_sup = F.l1_loss(s_illum, s_illum_target)

        # (5c) s_noise 显式监督
        L_noise_sup = pred.new_tensor(0.0)
        if self.lambda_noise_sup > 0 and "tfsi_out" in outputs:
            snr = outputs["tfsi_out"].get("snr")
            if snr is not None:
                snr_scalar = snr.mean(dim=1, keepdim=True)
                s_noise_target = torch.clamp(1.0 - snr_scalar / self.noise_tau_high, 0.0, 1.0).detach()
                L_noise_sup = F.l1_loss(s_noise, s_noise_target)

        # (6) 中间监督
        L_inter = pred.new_tensor(0.0)
        if self.lambda_inter > 0 and "img_s2" in outputs and "lit_up_map" in outputs:
            img_s2_lit = torch.clamp(outputs["img_s2"] * outputs["lit_up_map"], 0.0, 1.0)
            L_inter = charbonnier_loss(img_s2_lit, target)

        # (7) IFPN 侧监督
        L_ifpn_sup = pred.new_tensor(0.0)
        if self.lambda_ifpn_sup > 0 and "ifpn_side" in outputs:
            ifpn_side = outputs["ifpn_side"]
            tgt_down = F.interpolate(target, size=ifpn_side.shape[-2:], mode='bilinear', align_corners=False)
            L_ifpn_sup = charbonnier_loss(ifpn_side, tgt_down)

        L_total = (
            L_recon_base
            + self.lambda_ssim * L_ssim
            + self.lambda_perc * L_perc
            + self.lambda_illum * L_illum_smooth
            + self.lambda_illum_sup * L_illum_sup
            + self.lambda_noise_sup * L_noise_sup
            + self.lambda_inter * L_inter
            + self.lambda_ifpn_sup * L_ifpn_sup
        )

        loss_dict = {
            "loss_total": L_total.detach(),
            "loss_pix": L_pix.detach(),
            "loss_freq": L_freq.detach(),
            "loss_focal": L_focal.detach(),
            "loss_ssim": L_ssim.detach(),
            "loss_perc": L_perc.detach(),
            "loss_illum": L_illum_smooth.detach(),
            "loss_illum_sup": L_illum_sup.detach(),
            "loss_noise_sup": L_noise_sup.detach(),
            "loss_inter": L_inter.detach(),
            "loss_ifpn_sup": L_ifpn_sup.detach(),
        }
        return L_total, loss_dict
