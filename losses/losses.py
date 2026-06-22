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
    """v5.6 P1-3: 多层 VGG 感知损失 (relu1_2, relu2_2, relu3_3 加权 L1).

    旧版只用单层 relu3_3，对纹理细节不敏感。新版提取 3 层特征加权 L1，
    高层权重更大（捕获语义），低层捕获纹理细节。
    """

    def __init__(self, pretrained=False, multilayer=True, layer_weights=None):
        super().__init__()
        self.enabled = vgg16 is not None
        self.multilayer = multilayer
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

        if multilayer:
            # v5.6 P1-3: 多层特征提取
            self.layer1 = backbone.features[:4]    # relu1_2
            self.layer2 = backbone.features[4:9]   # relu2_2
            self.layer3 = backbone.features[9:16]  # relu3_3
            self.weights = layer_weights or [0.1, 0.2, 0.5]
            self.features = None
            for mod in [self.layer1, self.layer2, self.layer3]:
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
        else:
            # 旧版单层 (向后兼容)
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
            # v5.6 P1-3: 多层加权 L1
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
    TFS-Net v5.6 loss function

    L_total = L_pix + lambda_freq * L_freq
              + lambda_ssim * L_ssim + lambda_perc * L_perc
              + lambda_illum * L_illum_smooth
              + lambda_illum_sup * L_illum_sup        # v5.6 P0-2: s_illum 显式监督
              + lambda_noise_sup * L_noise_sup        # v5.6 P0-2: s_noise 显式监督
              + lambda_inter * L_inter                # 中间监督: img_s2 * lit_up_map (乘法路径)
              + lambda_recon * L_recon                # v5.6 P0-3: 最终输出监督 (含 s_illum 加法路径)

        L_pix          = Charbonnier(pred, target)
        L_freq         = L1(|FFT(pred)|, |FFT(target)|)
        L_ssim         = 1 - SSIM(pred, target)
        L_perc         = PerceptualLoss(pred, target)
        L_illum_smooth = edge-aware smoothness on s_illum  (正则, 降权到 0.001)
        L_illum_sup    = L1(s_illum, clamp(1 - L_t/(L_ref+eps), 0, 1).detach())   # P0-2
        L_noise_sup    = L1(s_noise, clamp(1 - SNR/tau_high, 0, 1).detach())      # P0-2
        L_inter        = Charbonnier(clamp(img_s2 * lit_up_map, 0, 1), target)    # 乘法路径
        L_recon        = Charbonnier(res_t, target)                                # 含加法路径
    """

    def __init__(
        self,
        use_temporal: bool = False,
        use_freq_loss: bool = True,
        perceptual_pretrained: bool = False,
        lambda_perc: float = 0.1,
        lambda_freq: float = 0.1,
        lambda_illum: float = 0.001,
        lambda_ssim: float = 0.2,
        lambda_inter: float = 0.3,
        # 保留旧参数兼容性
        lambda_aux: float = 0.0,
        fused_channels: int = 64,
        # v5.6 P0-2/P0-3 新增
        lambda_illum_sup: float = 0.1,
        lambda_noise_sup: float = 0.05,
        lambda_recon: float = 0.5,
        noise_tau_high: float = 5.0,
        # v5.6 P1-3/P1-4 新增
        perc_multilayer: bool = True,
        freq_with_phase: bool = True,
        freq_phase_weight: float = 0.5,
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_freq_loss = use_freq_loss
        self.lambda_perc = lambda_perc
        self.lambda_freq = lambda_freq
        self.lambda_illum = lambda_illum
        self.lambda_ssim = lambda_ssim
        self.lambda_inter = lambda_inter
        # v5.6
        self.lambda_illum_sup = lambda_illum_sup
        self.lambda_noise_sup = lambda_noise_sup
        self.lambda_recon = lambda_recon
        self.noise_tau_high = noise_tau_high
        self.freq_with_phase = freq_with_phase
        self.freq_phase_weight = freq_phase_weight

        self.perceptual = PerceptualLoss(pretrained=perceptual_pretrained, multilayer=perc_multilayer)

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
        s_noise = outputs["s_noise"]

        # (1) 空间像素重建 (Charbonnier) — P0-3: 降权，因新增 L_recon 重复监督
        L_pix = charbonnier_loss(pred, target)

        # (2) 频域重建 — v5.6 P1-4: 加相位项
        if self.use_freq_loss:
            fft_pred = torch.fft.rfft2(pred, norm='ortho')
            fft_gt = torch.fft.rfft2(target, norm='ortho')
            if self.freq_with_phase:
                # v5.6 P1-4: 幅度 + 相位
                L_freq = F.l1_loss(fft_pred.abs(), fft_gt.abs()) + \
                         self.freq_phase_weight * F.l1_loss(fft_pred.angle(), fft_gt.angle())
            else:
                L_freq = F.l1_loss(fft_pred.abs(), fft_gt.abs())
        else:
            L_freq = pred.new_tensor(0.0)

        L_recon_base = L_pix + self.lambda_freq * L_freq

        # (3) SSIM 损失
        L_ssim = 1.0 - ssim_map(pred, target).mean()

        # (4) 感知损失
        L_perc = self.perceptual(pred, target)

        # (5) 光照场边缘感知平滑 (正则, v5.6 降权)
        L_illum_smooth = self._edge_aware_smooth(s_illum, target)

        # (5b) v5.6 P0-2: s_illum 显式监督 — 基于输入暗度（非 L_ratio）
        # 物理含义: 输入比 GT 暗多少 → s_illum 越大
        # 旧版用 L_ratio (1 - L_t/L_ref) 在 SDSD indoor (均匀光照) 上恒为 0，导致塌缩。
        # 新版直接用输入与 GT 的亮度比，对任何数据集都有效。
        L_illum_sup = pred.new_tensor(0.0)
        if self.lambda_illum_sup > 0 and "image_center" in outputs:
            img_center = outputs["image_center"]  # (B,3,H,W) 低光输入
            eps = 1e-3
            # 逐像素暗度: 输入越暗于 GT → s_illum 越大
            img_gray = img_center.mean(dim=1, keepdim=True)    # (B,1,H,W)
            gt_gray = target.mean(dim=1, keepdim=True)          # (B,1,H,W)
            s_illum_target = torch.clamp(1.0 - img_gray / (gt_gray + eps), 0.0, 1.0).detach()
            L_illum_sup = F.l1_loss(s_illum, s_illum_target)

        # (5c) v5.6 P0-2: s_noise 显式监督 — 基于 SNR
        # 物理含义: SNR 越低 (噪声越大) → s_noise 越大
        L_noise_sup = pred.new_tensor(0.0)
        if self.lambda_noise_sup > 0 and "tfsi_out" in outputs:
            snr = outputs["tfsi_out"].get("snr")
            if snr is not None:
                snr_scalar = snr.mean(dim=1, keepdim=True)
                s_noise_target = torch.clamp(1.0 - snr_scalar / self.noise_tau_high, 0.0, 1.0).detach()
                L_noise_sup = F.l1_loss(s_noise, s_noise_target)

        # (6) v4.3: intermediate supervision on img_s2 (乘法路径)
        L_inter = pred.new_tensor(0.0)
        if self.lambda_inter > 0 and "img_s2" in outputs and "lit_up_map" in outputs:
            img_s2_lit = torch.clamp(outputs["img_s2"] * outputs["lit_up_map"], 0.0, 1.0)
            L_inter = charbonnier_loss(img_s2_lit, target)

        # (7) v5.6 P0-3: L_recon 已移除 — 与 L_pix 完全相同 (都是 Charb(res_t, GT))，冗余。
        # L_pix 已监督 res_t (含 s_illum 加法路径), 无需额外 L_recon。

        # 总损失
        L_total = (
            L_recon_base
            + self.lambda_ssim * L_ssim
            + self.lambda_perc * L_perc
            + self.lambda_illum * L_illum_smooth
            + self.lambda_illum_sup * L_illum_sup
            + self.lambda_noise_sup * L_noise_sup
            + self.lambda_inter * L_inter
        )

        loss_dict = {
            "loss_total":      L_total.detach(),
            "loss_pix":        L_pix.detach(),
            "loss_freq":       L_freq.detach(),
            "loss_ssim":       L_ssim.detach(),
            "loss_perc":       L_perc.detach(),
            "loss_illum":      L_illum_smooth.detach(),
            "loss_illum_sup":  L_illum_sup.detach(),
            "loss_noise_sup":  L_noise_sup.detach(),
            "loss_inter":      L_inter.detach(),
        }
        return L_total, loss_dict

