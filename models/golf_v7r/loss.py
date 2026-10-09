#!/usr/bin/env python3
"""
Golf v7r Loss: 多分支监督 + 正交约束

Loss = L1(final, GT)
     + w_N * L1(Y_N, GT) + w_L * L1(Y_L, GT) + w_M * L1(Y_M, GT)
     + lambda_ortho * ortho_loss(ctx_N, ctx_L, ctx_M)

Phase B 追加项 (§6.6-B / §6.9, 全部默认 0 即关闭):
     + dark_weight_alpha * 加权L1(final, GT)   (§6.6-B 暗区加权)
     + w_perceptual * VGG 多层感知损失          (§6.9)
     + w_freq * FFT 幅值(+相位) L1              (§6.9)
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class GolfV7RLoss(nn.Module):
    """多分支 + 正交约束 loss

    Phase A.3 开关 (§6.2-B, 按退化类型分配监督空间):
        use_lowfreq_L=False (默认/历史行为):
            Branch-L 与 N/M 一样对整幅 GT 做 L1 → 被强信号拉向同一解 (RGB 余弦 0.99).
        use_lowfreq_L=True:
            Branch-L 只对【低频分量】做 L1 (先 avg_pool 降采样再上采样),
            强制它学光照/亮度, 而非高频纹理 → 从标签层面制造分支分工.
    """

    def __init__(self,
                 w_branch_N: float = 0.1,
                 w_branch_L: float = 0.1,
                 w_branch_M: float = 0.1,
                 lambda_ortho: float = 0.01,
                 w_ssim: float = 0.0,
                 branch_warmup_epochs: int = 5,
                 use_lowfreq_L: bool = False,
                 lowfreq_kernel: int = 16,
                 w_perceptual: float = 0.0,
                 perceptual_multilayer: bool = True,
                 w_freq: float = 0.0,
                 freq_phase_weight: float = 1.0,
                 dark_weight_alpha: float = 0.0,
                 dark_weight_gamma: float = 1.0):
        """Phase B 开关 (§6.6-B 暗区加权 + §6.9 感知/频率损失)。

        w_perceptual / w_freq / dark_weight_alpha:
          - 默认全 0.0 → 历史行为完全不变 (可逐项消融)。
        dark_weight_alpha (§6.6-B):
          - >0 时对 L1 做亮度倒数加权: w(x) = (1/(lum(x)+eps))^gamma 归一化到均值 1。
            动机: L1 在亮区数值大, 暗区梯度被亮区主导 → 极暗序列学不动。
            归一化保证总 loss 尺度不变, 不因加权而实质改变有效学习率。
        """
        super().__init__()
        self.w_branch_N = w_branch_N
        self.w_branch_L = w_branch_L
        self.w_branch_M = w_branch_M
        self.lambda_ortho = lambda_ortho
        self.w_ssim = w_ssim
        self.branch_warmup_epochs = branch_warmup_epochs
        self.use_lowfreq_L = use_lowfreq_L
        self.lowfreq_kernel = int(lowfreq_kernel) if lowfreq_kernel else 16

        self.current_epoch = 0
        self._use_ssim = w_ssim > 0
        if self._use_ssim:
            try:
                from pytorch_msssim import ms_ssim
                self._ms_ssim = ms_ssim
            except ImportError:
                self._use_ssim = False

        # §6.9 感知/频率项 (默认关闭)
        self.w_perceptual = float(w_perceptual or 0.0)
        self.w_freq = float(w_freq or 0.0)
        self.freq_phase_weight = float(freq_phase_weight)
        self._perceptual = None
        if self.w_perceptual > 0:
            try:
                from losses.losses import PerceptualLoss
                self._perceptual = PerceptualLoss(multilayer=perceptual_multilayer)
            except Exception as exc:  # torchvision 缺失等
                warnings.warn(f"PerceptualLoss unavailable, w_perceptual ignored: {exc}")
                self.w_perceptual = 0.0

        # §6.6-B 暗区加权 (默认关闭)
        self.dark_weight_alpha = float(dark_weight_alpha or 0.0)
        self.dark_weight_gamma = float(dark_weight_gamma or 1.0)

    def _dark_weight_map(self, gt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """§6.6-B 亮度倒数权重图, 缩放使加权 L1 的期望量级与裸 L1 一致。

        权重主体 w = (1/(lum+eps))^gamma 做均值归一; 再整体乘 `ref` (= 裸 L1 的标量值),
        使 Σ w*|pred-gt| 的均值与裸 L1 同量级 —— 即本项只做【误差在空间上的重分配】
        (暗区权重 >1, 亮区 <1), 而非额外叠加一份 L1。由此不改变有效学习率, alpha 可直接
        作为"混合比例"解读 (0=纯 L1, 1=完全按暗度重新加权)。
        """
        lum = gt.mean(dim=1, keepdim=True)                     # [B,1,H,W]
        w = (1.0 / (lum + 1e-2)) ** self.dark_weight_gamma
        w = w / (w.mean() + 1e-8)
        return self.dark_weight_alpha * w * ref

    def _freq_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """§6.9 FFT 幅值 (+相位) L1。"""
        fft_pred = torch.fft.rfft2(pred.float(), norm='ortho')
        fft_gt = torch.fft.rfft2(gt.float(), norm='ortho')
        loss = F.l1_loss(fft_pred.abs(), fft_gt.abs())
        if self.freq_phase_weight > 0:
            loss = loss + self.freq_phase_weight * F.l1_loss(fft_pred.angle(), fft_gt.angle())
        return loss

    def _lowfreq(self, x: torch.Tensor) -> torch.Tensor:
        """低通: avg_pool 下采样 → 上采样回原分辨率 (保留低频亮度结构)"""
        k = self.lowfreq_kernel
        h, w = x.shape[-2:]
        pooled = F.avg_pool2d(x, kernel_size=k, stride=k,
                              ceil_mode=True, count_include_pad=False)
        return F.interpolate(pooled, size=(h, w), mode='bilinear',
                             align_corners=False)

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
                    pred_b = output_dict[key]
                    # Phase A.3 开关: Branch-L 只监督低频分量 (从标签层面强制分工)
                    if key == 'branch_L' and self.use_lowfreq_L:
                        pred_b = self._lowfreq(pred_b)
                        gt_b = self._lowfreq(gt)
                    else:
                        gt_b = gt
                    bl = F.l1_loss(pred_b, gt_b)
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

        # §6.6-B 暗区加权 L1 (与裸 L1 同量级, 只重分配空间权重, 不改变有效学习率)
        if self.dark_weight_alpha > 0:
            wmap = self._dark_weight_map(gt, loss_l1.detach())
            l_dark = (wmap * (pred - gt).abs()).mean()
            loss = loss + l_dark
            loss_dict['dark_l1'] = l_dark.detach()

        # §6.9 VGG 多层感知损失
        if self.w_perceptual > 0 and self._perceptual is not None:
            l_perc = self._perceptual(pred.clamp(0, 1), gt.clamp(0, 1))
            loss = loss + self.w_perceptual * l_perc
            loss_dict['perceptual'] = l_perc.detach()

        # §6.9 FFT 频率损失
        if self.w_freq > 0:
            l_freq = self._freq_loss(pred, gt)
            loss = loss + self.w_freq * l_freq
            loss_dict['freq'] = l_freq.detach()

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
