"""
Golf Loss Function
===================
在 FoxtrotLoss 基础上新增两项:

1. [棋盘格抑制] L_chess — 频域棋盘格惩罚
   棋盘格在 FFT 频谱的 Nyquist 频率 (周期2px) 有能量尖峰。
   直接惩罚该频段能量, 主动抑制残余格纹:
     L_chess = |FFT_shift(O_t)|[Nyquist 十字带] 的均值
   虽然 Golf 的 resize-conv 已从结构上消除主要来源, 此损失作为兜底
   并防止 fusion 的 3×3 卷积重新引入周期伪影。

2. [时序一致性] L_temp — 多帧输出一致性
   Foxtrot 的 L_temp 未实现 (lambda_temp=0)。
   Golf 实现: 利用相邻帧预测的一致性约束。
   由于网络只输出中心帧, 采用"相邻窗口重叠预测"策略:
   用相邻的 GT 帧与当前输出做时序平滑 (需要多帧输出, 成本高)
   → Golf 改用轻量方案: 对 O_t 与低通(X_t) 的高频残差做正则,
     保证输出不引入不自然的帧间高频跳变 (单帧可算)。

3. [分支差异化弱监督] L_branch_div
   Foxtrot 三分支都回归同一 GT → 功能冗余。
   Golf 给分支加差异化引导:
     - Branch-N (噪声): 侧重暗区/平坦区域的平滑
     - Branch-L (光照): 侧重全局亮度一致性
     - Branch-M (运动): 侧重边缘/结构保持
   通过空间加权 mask 实现 (不需要额外 GT)

损失结构:
  L_total = L_final + λ_N·L_N + λ_L·L_L + λ_M·L_M
            + λ_ortho·L_ortho + λ_chess·L_chess + λ_temp·L_temp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SSIMLoss(nn.Module):
    """SSIM Loss (1 - SSIM)"""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 3
        self.window = self._create_window(window_size, sigma, 3)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, sigma, channel):
        _1d = self._gaussian(window_size, sigma).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
        return _2d.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        img1 = img1.float()
        img2 = img2.float()
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        mu1 = F.conv2d(img1, self.window, padding=pad, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=pad, groups=self.channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        s1 = F.conv2d(img1 * img1, self.window, padding=pad, groups=self.channel) - mu1_sq
        s2 = F.conv2d(img2 * img2, self.window, padding=pad, groups=self.channel) - mu2_sq
        s12 = F.conv2d(img1 * img2, self.window, padding=pad, groups=self.channel) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
        return 1 - ssim_map.mean()


class GolfLoss(nn.Module):
    """Golf 多任务损失

    Args:
        lambda_N/L/M:    分支监督权重
        lambda_ortho:    正交约束权重
        lambda_chess:    棋盘格抑制权重 (Golf 新增)
        lambda_temp:     时序/高频稳定权重 (Golf 新增)
        lambda_div:      分支差异化权重 (Golf 新增)
        use_ssim:        是否用 SSIM
        ssim_weight:     SSIM 权重
    """

    def __init__(self, lambda_N: float = 0.3, lambda_L: float = 0.3,
                 lambda_M: float = 0.3, lambda_ortho: float = 0.01,
                 lambda_chess: float = 0.05, lambda_temp: float = 0.02,
                 lambda_div: float = 0.05,
                 use_ssim: bool = True, ssim_weight: float = 0.3):
        super().__init__()
        self.lambda_N = lambda_N
        self.lambda_L = lambda_L
        self.lambda_M = lambda_M
        self.lambda_ortho = lambda_ortho
        self.lambda_chess = lambda_chess
        self.lambda_temp = lambda_temp
        self.lambda_div = lambda_div
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight

        self.l1_loss = nn.L1Loss()
        if use_ssim:
            self.ssim_loss = SSIMLoss()

    def _recon_loss(self, pred, gt):
        loss = self.l1_loss(pred, gt)
        if self.use_ssim:
            loss = loss + self.ssim_weight * self.ssim_loss(pred, gt)
        return loss

    def _retinex_regularization(self, L_t):
        gx = torch.abs(L_t[:, :, :, :-1] - L_t[:, :, :, 1:])
        gy = torch.abs(L_t[:, :, :-1, :] - L_t[:, :, 1:, :])
        return gx.mean() + gy.mean()

    def _flow_smoothness(self, flow):
        gx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        gy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        return gx.mean() + gy.mean()

    def _checkerboard_loss(self, img: torch.Tensor) -> torch.Tensor:
        """棋盘格抑制: 惩罚 FFT 频谱中周期 2px 的 Nyquist 频段能量.

        棋盘格在频域的表现为 (H/2, 0) 与 (0, W/2) 附近的能量尖峰。
        通过遮蔽低频/中频, 只统计 Nyquist 十字带, 惩罚其能量。
        """
        gray = img.float().mean(dim=1)              # (B,H,W)
        fft = torch.fft.fft2(gray, norm='ortho')
        mag = torch.abs(torch.fft.fftshift(fft, dim=(-2, -1)))  # (B,H,W)
        H, W = gray.shape[-2:]

        # Nyquist 十字带: 中心 ± 1 像素的行/列
        mask = torch.zeros(1, H, W, device=img.device, dtype=mag.dtype)
        cy, cx = H // 2, W // 2
        # 垂直 Nyquist 线 (列 cx)
        mask[:, cy, :] = 1.0
        mask[:, :, cx] = 1.0
        # 只保留端点附近的 Nyquist 频率 (周期≈2px)
        # 用带阻: 去掉中心低频
        band = torch.zeros(1, H, W, device=img.device, dtype=mag.dtype)
        k = max(3, min(H, W) // 16)
        band[:, cy - k:cy + k + 1, :] = 1.0
        band[:, :, cx - k:cx + k + 1] = 1.0
        mask = mask * (1.0 - band)   # 十字带 减去 中心低频区

        # 归一化: 除以总能量避免尺度依赖
        total = mag.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        loss = (mag * mask).sum(dim=(-2, -1)) / total.squeeze(-1).squeeze(-1) * 1e3
        return loss.mean()

    def _temporal_highfreq_stability(self, O_t: torch.Tensor,
                                     X_t: torch.Tensor) -> torch.Tensor:
        """时序高频稳定: 输出相对中心帧的高频残差不应有过大的孤立跳变.

        低光增强中, 帧间高频差异主要来自噪声。约束增强后的高频细节
        不产生超过输入高频的额外能量, 抑制帧间闪烁的源头。
        """
        # 高频 = 原图 - 低通
        k = 7
        pad = k // 2
        X_low = F.avg_pool2d(X_t, k, 1, pad)
        O_low = F.avg_pool2d(O_t, k, 1, pad)
        X_high = X_t - X_low
        O_high = O_t - O_low
        # 增强后高频能量不应显著超过输入高频 (噪声被放大)
        return F.relu(O_high.abs().mean() - 1.5 * X_high.abs().mean())

    def _branch_divergence(self, Y_N, Y_L, Y_M, X_t) -> torch.Tensor:
        """分支差异化弱监督: 用输入亮度/梯度构造三个空间 mask,
        引导不同分支关注不同区域 (不需要额外 GT).

        暗区 mask    → Branch-N (噪声主导)
        亮区 mask    → Branch-L (光照主导)
        高梯度 mask  → Branch-M (结构/运动主导)
        """
        gray = X_t.mean(dim=1, keepdim=True)               # (B,1,H,W)
        # 暗区 mask
        dark = (1.0 - gray).clamp(0, 1)
        # 梯度 mask
        gx = torch.abs(X_t[:, :, :, :-1] - X_t[:, :, :, 1:])
        gy = torch.abs(X_t[:, :, :-1, :] - X_t[:, :, 1:, :])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        grad = (gx + gy).mean(dim=1, keepdim=True)
        bright = gray

        # 各分支输出与输入的差异应该在对应区域更大 (分支专注)
        d_N = (Y_N - X_t).abs().mean(dim=1, keepdim=True)
        d_L = (Y_L - X_t).abs().mean(dim=1, keepdim=True)
        d_M = (Y_M - X_t).abs().mean(dim=1, keepdim=True)

        # 期望: 暗区 N 变化大, 亮区 L 变化大, 边缘区 M 变化大
        L_div = -(dark * d_N).mean() - (bright * d_L).mean() - (grad * d_M).mean()
        return -L_div  # 取负 → 最大化差异 (loss 下降)

    def forward(self, outputs: Dict[str, torch.Tensor],
                gt: torch.Tensor, gt_seq: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        O_t = outputs["O_t"]
        Y_N, Y_L, Y_M = outputs["Y_N"], outputs["Y_L"], outputs["Y_M"]
        L_t = outputs["L_t"]
        flow_vis = outputs["flow_vis"]
        X_t = outputs.get("image_center", None)
        ortho_loss = outputs.get("ortho_loss", torch.tensor(0.0, device=O_t.device))

        # 1. 最终监督
        L_final = self._recon_loss(O_t, gt)

        # 2. 分支监督
        L_N = self._recon_loss(Y_N, gt)
        L_L = self._recon_loss(Y_L, gt) + 0.1 * self._retinex_regularization(L_t)
        L_M = self._recon_loss(Y_M, gt) + 0.01 * self._flow_smoothness(flow_vis)

        # 3. 正交约束
        L_ortho = ortho_loss

        # 4. [Golf] 棋盘格抑制
        L_chess = self._checkerboard_loss(O_t)

        # 5. [Golf] 高频稳定
        if X_t is not None:
            L_temp = self._temporal_highfreq_stability(O_t, X_t)
        else:
            L_temp = torch.tensor(0.0, device=O_t.device)

        # 6. [Golf] 分支差异化
        if X_t is not None:
            L_div = self._branch_divergence(Y_N, Y_L, Y_M, X_t)
        else:
            L_div = torch.tensor(0.0, device=O_t.device)

        total_loss = (L_final +
                      self.lambda_N * L_N +
                      self.lambda_L * L_L +
                      self.lambda_M * L_M +
                      self.lambda_ortho * L_ortho +
                      self.lambda_chess * L_chess +
                      self.lambda_temp * L_temp +
                      self.lambda_div * L_div)

        return {
            "total_loss": total_loss,
            "L_final": L_final.item(),
            "L_N": L_N.item(),
            "L_L": L_L.item(),
            "L_M": L_M.item(),
            "L_ortho": L_ortho.item() if isinstance(L_ortho, torch.Tensor) else L_ortho,
            "L_chess": L_chess.item() if isinstance(L_chess, torch.Tensor) else L_chess,
            "L_temp": L_temp.item() if isinstance(L_temp, torch.Tensor) else L_temp,
            "L_div": L_div.item() if isinstance(L_div, torch.Tensor) else L_div,
        }
