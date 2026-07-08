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
    TFS-Net v6 Delta Mark2 loss function — Kendall uncertainty weighting + PE Loss

    L_total = Σ 1/(2σ²_i) * L_i + 1/2 * log(σ²_i)  (Kendall CVPR 2018)

    9项损失分为语义关联组:
      组1 输出质量: pix, freq
      组2 结构感知: ssim, perc
      组3 中间监督: illum, illum_sup, inter, ifpn_sup
      (noise_sup 权重=0 已关闭)
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
        lambda_aux: float = 0.0,
        fused_channels: int = 64,
        lambda_illum_sup: float = 0.1,
        lambda_noise_sup: float = 0.05,
        lambda_recon: float = 0.5,
        noise_tau_high: float = 5.0,
        perc_multilayer: bool = True,
        freq_with_phase: bool = True,
        freq_phase_weight: float = 0.5,
        lambda_ifpn_sup: float = 0.0,
        lambda_pix: float = 1.0,
        # Mark2: uncertainty weighting
        uncertainty_weighting: bool = True,
        use_pe_charbonnier: bool = False,
        pe_edge_weight: float = 2.0,
        perceptual_decoupling: bool = False,
        warmup_loss_only_pix_ssim: bool = True,
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_freq_loss = use_freq_loss
        self.lambda_perc = lambda_perc
        self.lambda_freq = lambda_freq
        self.lambda_illum = lambda_illum
        self.lambda_ssim = lambda_ssim
        self.lambda_inter = lambda_inter
        self.lambda_recon = lambda_recon
        self.lambda_illum_sup = lambda_illum_sup
        self.lambda_noise_sup = lambda_noise_sup
        self.noise_tau_high = noise_tau_high
        self.freq_with_phase = freq_with_phase
        self.freq_phase_weight = freq_phase_weight
        self.lambda_ifpn_sup = lambda_ifpn_sup
        self.lambda_pix = lambda_pix

        # Mark2 flags
        self.uncertainty_weighting = uncertainty_weighting
        self.use_pe_charbonnier = use_pe_charbonnier
        self.perceptual_decoupling = perceptual_decoupling
        self.warmup_loss_only_pix_ssim = warmup_loss_only_pix_ssim

        # Mark2: Kendall log-variance parameters
        if uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                'pix':   nn.Parameter(torch.tensor(0.0)),
                'freq':  nn.Parameter(torch.tensor(0.0)),
                'ssim':  nn.Parameter(torch.tensor(-1.0)),
                'perc':  nn.Parameter(torch.tensor(2.0)),  # warmup: very low weight
                'illum': nn.Parameter(torch.tensor(0.0)),
                'inter': nn.Parameter(torch.tensor(0.0)),
                'ifpn':  nn.Parameter(torch.tensor(0.0)),
            })

        # Mark2: PE Charbonnier loss
        if use_pe_charbonnier:
            self.sob_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
            self.sob_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
            self.sob_x.weight.data = torch.tensor(
                [[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32
            ).reshape(1,1,3,3)
            self.sob_y.weight.data = torch.tensor(
                [[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32
            ).reshape(1,1,3,3)
            self.sob_x.weight.requires_grad = False
            self.sob_y.weight.requires_grad = False
            self.pe_edge_weight = pe_edge_weight

        self.perceptual = PerceptualLoss(pretrained=perceptual_pretrained, multilayer=perc_multilayer)

    def _pe_charbonnier(self, pred, gt, eps=1e-6):
        """PE Charbonnier: edge-weighted smooth L1"""
        gt_gray = gt[:, :1] if gt.shape[1] == 3 else gt[:, 0:1]
        edge_x = self.sob_x(gt_gray)
        edge_y = self.sob_y(gt_gray)
        edge_map = torch.sqrt(edge_x**2 + edge_y**2 + eps)
        edge_norm = 1.0 + (self.pe_edge_weight - 1.0) * (edge_map / (edge_map.max() + eps))
        diff = pred - gt
        return (torch.sqrt(diff**2 + eps) * edge_norm).mean()

    def _uw(self, loss, key):
        """Kendall uncertainty weight"""
        if not self.uncertainty_weighting:
            return loss
        log_var = self.log_vars[key]
        precision = torch.exp(-log_var)
        return 0.5 * precision * loss + 0.5 * log_var

    @staticmethod
    def _edge_aware_smooth(s: torch.Tensor, ref_img: torch.Tensor) -> torch.Tensor:
        grad_s_x = (s[:, :, :, 1:] - s[:, :, :, :-1]).abs()
        grad_s_y = (s[:, :, 1:, :] - s[:, :, :-1, :]).abs()
        grad_i_x = (ref_img[:, :, :, 1:] - ref_img[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        grad_i_y = (ref_img[:, :, 1:, :] - ref_img[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
        return (grad_s_x * torch.exp(-grad_i_x)).mean() + (grad_s_y * torch.exp(-grad_i_y)).mean()

    def schedule_loss_phase(self, epoch: int):
        """Mark2 训练阶段调度 (epoch 边界调用)"""
        if epoch == 15 and self.uncertainty_weighting:
            # Phase2: 解锁感知损失 log_var 2.0→-1.0
            with torch.no_grad():
                self.log_vars['perc'].fill_(-1.0)
                self.log_vars['freq'].fill_(-0.5)
            return {"action": "phase2_unlock_perc", "perc_log_var": -1.0, "freq_log_var": -0.5}
        return None

    def forward(self, outputs: dict, target: torch.Tensor, epoch: int = 0,
                phase: str = 'phase2', unlock_ratio: float = 1.0, model=None):
        pred = outputs["res_t"]
        s_illum = outputs["s_illum"]
        s_noise = outputs["s_noise"]

        # --- Mark3 Phase 1 alignment losses ---
        losses = {}
        L_align_warp = pred.new_tensor(0.0)
        L_diag_prior = pred.new_tensor(0.0)
        if "C_omega" in outputs and outputs["C_omega"] and "F_out_list" in outputs:
            C_omega = torch.stack(outputs["C_omega"], dim=1)  # (B, T-1, N, N)
            F_out = outputs["F_out_list"]
            center = len(F_out) // 2
            F_center = F_out[center]  # (B, C, H, W)
            Bv, Cv, _, _ = F_center.shape
            # warp consistency (downsample features to match C_omega N)
            ds = int(C_omega.shape[-1] ** 0.5)
            F_center_ds = F.adaptive_avg_pool2d(F_center, (ds, ds)).reshape(Bv, Cv, -1)
            for k in range(len(F_out) - 1):
                tp = k if k < center else k + 1
                F_tp_ds = F.adaptive_avg_pool2d(F_out[tp], (ds, ds)).reshape(Bv, Cv, -1)
                C_k = C_omega[:, k]
                w = torch.bmm(F_tp_ds, C_k.transpose(-1, -2)).reshape(Bv, Cv, ds, ds)
                L_align_warp = L_align_warp + F.l1_loss(w, F_center_ds.reshape(Bv, Cv, ds, ds))
            L_align_warp = L_align_warp / max(len(F_out) - 1, 1)
            # diag prior: encourage diag(C_omega)→1 (self-supervised, no GT needed)
            # C_omega: (B, T-1, N, N) softmax-normalized per row
            N = C_omega.shape[-1]
            diag_C = torch.diagonal(C_omega, dim1=-2, dim2=-1)  # (B, T-1, N)
            L_diag_prior = -torch.log(diag_C.clamp(min=1e-6)).mean()
        losses['align_warp'] = L_align_warp
        losses['diag_prior'] = L_diag_prior

        # --- Phase 1 / Warmup: limited losses ---
        if phase in ('phase1_warmup', 'phase1'):
            L_pix = self._pe_charbonnier(pred, target) if self.use_pe_charbonnier else charbonnier_loss(pred, target)
            L_ssim = 1.0 - ssim_map(pred, target).mean()
            L_illum_smooth = self._edge_aware_smooth(s_illum, target)

            # gain_map supervision (Mark4: replaces lit_up_map)
            L_gain_sup = pred.new_tensor(0.0)
            if "gain_map" in outputs and "image_center" in outputs:
                img_c = outputs["image_center"]
                gt_g = target.mean(dim=1, keepdim=True)
                ic_g = img_c.mean(dim=1, keepdim=True)
                gain_target = (gt_g / (ic_g + 1e-6)).clamp(1.0, 8.0)
                gain_pred = outputs["gain_map"]
                L_gain_sup = F.l1_loss(gain_pred, gain_target.expand_as(gain_pred))

            # WFR reg: keep alpha/gate near 0.5
            L_wfr_reg = pred.new_tensor(0.0)
            if model is not None and hasattr(model, 'wfr'):
                wsd = model.wfr
                # alpha_net: DWConv→GELU→Conv1x1→Sigmoid, check Conv1x1 bias
                try:
                    alpha_bias = wfr.alpha_net[2].bias.mean()
                    wfr._alpha_mean = alpha_bias.sigmoid()
                    L_wfr_reg = (wfr._alpha_mean - 0.5)**2
                except: pass

            if self.uncertainty_weighting:
                L = (self._uw(L_pix, 'pix') + self._uw(L_ssim, 'ssim')
                   + self._uw(L_illum_smooth, 'illum')
                   + self._uw(L_align_warp + L_diag_prior, 'ifpn')
                   + 0.5 * L_gain_sup + 0.001 * L_wfr_reg)
            else:
                L = self.lambda_pix * L_pix + self.lambda_ssim * L_ssim + self.lambda_illum * L_illum_smooth
            losses.update({'loss_pix': L_pix.detach(), 'loss_ssim': L_ssim.detach(),
                          'loss_illum': L_illum_smooth.detach(), 'loss_perc': pred.new_tensor(0.0),
                          'loss_freq': pred.new_tensor(0.0), 'loss_illum_sup': L_gain_sup.detach(),
                          'loss_noise_sup': pred.new_tensor(0.0), 'loss_inter': pred.new_tensor(0.0),
                          'loss_ifpn_sup': L_diag_prior.detach(), 'loss_total': L.detach()})
            return L, losses

        # --- Phase 1.5 / Phase2: full loss (existing logic below) ---

        # Pix loss (PE or standard Charbonnier)
        L_pix = self._pe_charbonnier(pred, target) if self.use_pe_charbonnier \
                else charbonnier_loss(pred, target)

        # Freq loss
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

        # Mark2: perceptual decoupling — SSIM on img_s1, VGG on img_s2
        img_s1 = outputs.get("img_s1", pred)
        img_s2 = outputs.get("img_s2", pred)
        if self.perceptual_decoupling:
            L_ssim = 1.0 - ssim_map(img_s1, target).mean()
            L_perc = self.perceptual(img_s2, target)
        else:
            L_ssim = 1.0 - ssim_map(pred, target).mean()
            L_perc = self.perceptual(pred, target)

        # Illumination smoothness
        L_illum_smooth = self._edge_aware_smooth(s_illum, target)

        # s_noise supervision (disabled)
        L_noise_sup = pred.new_tensor(0.0)

        # gain_map supervision (Mark4: shared between Phase 1 and Phase 2)
        L_gain_sup = pred.new_tensor(0.0)
        if "gain_map" in outputs and "image_center" in outputs:
            img_c = outputs["image_center"]
            gt_g = target.mean(dim=1, keepdim=True)
            ic_g = img_c.mean(dim=1, keepdim=True)
            gain_target = (gt_g / (ic_g + 1e-6)).clamp(1.0, 8.0)
            gain_pred = outputs["gain_map"]
            L_gain_sup = F.l1_loss(gain_pred, gain_target.expand_as(gain_pred))

        # Intermediate supervision (Mark4: img_s2 × gain_map)
        L_inter = pred.new_tensor(0.0)
        if self.lambda_inter > 0 and "img_s2" in outputs and "gain_map" in outputs:
            gain = F.interpolate(outputs["gain_map"], size=outputs["img_s2"].shape[-2:],
                                  mode='bilinear', align_corners=False)
            img_s2_lit = torch.clamp(outputs["img_s2"] * gain, 0.0, 1.0)
            L_inter = charbonnier_loss(img_s2_lit, target)

        # WFR reg (Mark4: shared)
        L_wfr_reg = pred.new_tensor(0.0)
        if model is not None and hasattr(model, 'wfr'):
            wfr = model.wfr
            try:
                alpha_bias = wfr.alpha_net[2].bias.mean()
                wfr._alpha_mean = alpha_bias.sigmoid()
                L_wfr_reg = (wfr._alpha_mean - 0.5)**2
            except: pass

        # align_warp + diag_prior (already computed above, shared)
        # Mark4: these C_omega losses remain active in all phases

        # Mark2 warmup: pix+ssim only for first 5 epochs
        if self.warmup_loss_only_pix_ssim and epoch < 5:
            if self.uncertainty_weighting:
                L_total = self._uw(L_pix, 'pix') + self._uw(L_ssim, 'ssim')
            else:
                L_total = self.lambda_pix * L_pix + self.lambda_ssim * L_ssim
            L_total = L_total + self.lambda_illum * L_illum_smooth
        elif self.uncertainty_weighting:
            L_total = (
                self._uw(L_pix, 'pix') + self._uw(L_freq, 'freq')
                + self._uw(L_ssim, 'ssim') + self._uw(L_perc, 'perc')
                + self._uw(L_illum_smooth, 'illum')
                + self._uw(L_inter, 'inter')
                + self._uw(L_align_warp + L_diag_prior, 'ifpn')
                + 0.5 * L_gain_sup + 0.001 * L_wfr_reg
            )
        else:
            L_total = (
                self.lambda_pix * L_pix + self.lambda_freq * L_freq
                + self.lambda_ssim * L_ssim + self.lambda_perc * L_perc
                + self.lambda_illum * L_illum_smooth
                + self.lambda_inter * L_inter
            )

        loss_dict = {
            "loss_total":      L_total.detach(),
            "loss_pix":        L_pix.detach(),
            "loss_freq":       L_freq.detach(),
            "loss_ssim":       L_ssim.detach(),
            "loss_perc":       L_perc.detach(),
            "loss_illum":      L_illum_smooth.detach(),
            "loss_illum_sup":  L_gain_sup.detach(),
            "loss_noise_sup":  L_noise_sup.detach(),
            "loss_inter":      L_inter.detach(),
            "loss_ifpn_sup":   L_diag_prior.detach(),
        }
        return L_total, loss_dict

