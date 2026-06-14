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


# =================================================================
#  TFSNetLoss — TFS-Net v3.2 损失函数
# =================================================================


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Charbonnier loss (smooth L1 alternative, standard in image restoration)."""
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps))


class BranchReconHead(nn.Module):
    """分支独立重建头：将分支特征投影到图像空间进行辅助监督。"""

    def __init__(self, channels: int, out_channels: int = 3):
        super().__init__()
        mid = max(channels // 2, 16)
        self.head = nn.Sequential(
            nn.Conv2d(channels, mid, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class TFSNetLoss(nn.Module):
    """
    TFS-Net v4.1 损失函数

    L_total = L_recon + λ_ssim * L_ssim + λ_perc * L_perc
              + λ_illum * L_illum + λ_inter * L_inter

        L_recon  = L_pix + λ_freq * L_freq
        L_pix    = Charbonnier(pred, target)
        L_freq   = L1(|FFT(pred)|, |FFT(target)|)
        L_ssim   = 1 - SSIM(pred, target)
        L_perc   = PerceptualLoss(pred, target)
        L_illum  = edge-aware smoothness on s_illum
        L_inter  = (Charbonnier(img_s1, target) + Charbonnier(img_s2, target)) / 2
                   # IGRF 中间阶段图像直接监督，无需额外 head
    """

    def __init__(
        self,
        use_temporal: bool = False,
        use_freq_loss: bool = True,
        perceptual_pretrained: bool = False,
        lambda_perc: float = 0.1,
        lambda_freq: float = 0.1,
        lambda_illum: float = 0.01,
        lambda_ssim: float = 0.2,
        lambda_inter: float = 0.3,
        # 保留旧参数兼容性
        lambda_aux: float = 0.0,
        fused_channels: int = 64,
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_freq_loss = use_freq_loss
        self.lambda_perc = lambda_perc
        self.lambda_freq = lambda_freq
        self.lambda_illum = lambda_illum
        self.lambda_ssim = lambda_ssim
        self.lambda_inter = lambda_inter

        self.perceptual = PerceptualLoss(pretrained=perceptual_pretrained)

    @staticmethod
    def _edge_aware_smooth(s: torch.Tensor, ref_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s       : (B, 1, H, W) 单通道强度图
            ref_img : (B, 3, H, W) RGB 参考图
        Returns:
            scalar loss
        """
        grad_s_x = (s[:, :, :, 1:] - s[:, :, :, :-1]).abs()
        grad_s_y = (s[:, :, 1:, :] - s[:, :, :-1, :]).abs()

        grad_i_x = (ref_img[:, :, :, 1:] - ref_img[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        grad_i_y = (ref_img[:, :, 1:, :] - ref_img[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)

        loss = (grad_s_x * torch.exp(-grad_i_x)).mean() + \
               (grad_s_y * torch.exp(-grad_i_y)).mean()
        return loss

    def forward(self, outputs: dict, target: torch.Tensor):
        """
        Args:
            outputs : dict from TFSNet.forward()
            target  : (B, 3, H, W) GT
        Returns:
            (loss_total, loss_dict)
        """
        pred = outputs["res_t"]
        s_illum = outputs["s_illum"]

        # (1) 空间像素重建 (Charbonnier)
        L_pix = charbonnier_loss(pred, target)

        # (2) 频域重建
        if self.use_freq_loss:
            fft_pred = torch.fft.rfft2(pred, norm='ortho')
            fft_gt = torch.fft.rfft2(target, norm='ortho')
            L_freq = F.l1_loss(fft_pred.abs(), fft_gt.abs())
        else:
            L_freq = pred.new_tensor(0.0)

        L_recon = L_pix + self.lambda_freq * L_freq

        # (3) SSIM 损失
        L_ssim = 1.0 - ssim_map(pred, target).mean()

        # (4) 感知损失
        L_perc = self.perceptual(pred, target)

        # (5) 光照场边缘感知平滑
        L_illum = self._edge_aware_smooth(s_illum, target)

        # (6) v4.2: intermediate supervision - brightened intermediate images vs GT
        # img_s1/img_s2 are dark (denoised/deblurred but not brightened)
        # lit_up_map brightens them: img_s1 * lit_up_map should approximate GT
        L_inter = pred.new_tensor(0.0)
        if self.lambda_inter > 0 and "img_s1" in outputs and "lit_up_map" in outputs:
            lit_up_map = outputs["lit_up_map"]
            res_s1 = torch.clamp(outputs["img_s1"] * lit_up_map, 0.0, 1.0)
            res_s2 = torch.clamp(outputs["img_s2"] * lit_up_map, 0.0, 1.0)
            L_s1 = charbonnier_loss(res_s1, target)
            L_s2 = charbonnier_loss(res_s2, target)
            L_inter = (L_s1 + L_s2) / 2.0

        # 总损失
        L_total = (
            L_recon
            + self.lambda_ssim * L_ssim
            + self.lambda_perc * L_perc
            + self.lambda_illum * L_illum
            + self.lambda_inter * L_inter
        )

        loss_dict = {
            "loss_total": L_total.detach(),
            "loss_pix":   L_pix.detach(),
            "loss_freq":  L_freq.detach(),
            "loss_ssim":  L_ssim.detach(),
            "loss_perc":  L_perc.detach(),
            "loss_illum": L_illum.detach(),
            "loss_inter": L_inter.detach(),
        }
        return L_total, loss_dict

