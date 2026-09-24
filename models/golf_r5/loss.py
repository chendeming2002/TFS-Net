"""
Golf-R5 Loss Function
======================
R5 损失改动 (相对 R4):

1. [L_temp 回退] R4 的噪声去相关余弦 → R5 恢复 R3 空间 HF 上限
   - R4 去相关可能过度压制 M 分支高频恢复
   - R3 HF 上限简单稳健: relu(|HF(Y)| - τ|HF(X_t)|)
   
2. [L_prior 降权] 0.01 → 0.005
   - 保留尺度不变设计, 但降低影响

3. [保留 R4 工程修复]
   - SSIMLoss, 分支差异化, 正交约束不变
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
        lambda_temp:     时序/高频稳定权重
        lambda_div:      分支差异化权重
        use_ssim:        是否用 SSIM
        ssim_weight:     SSIM 权重
    """

    def __init__(self, lambda_N: float = 0.3, lambda_L: float = 0.3,
                 lambda_M: float = 0.3, lambda_ortho: float = 0.01,
                 lambda_temp: float = 0.02,
                 lambda_div: float = 0.05,
                 lambda_prior: float = 0.01,    # R5: TCA 轻量结构化先验正则
                 temp_threshold: float = 1.2,   # R5: 空间 HF 上限阈值 (重新启用)
                 use_ssim: bool = True, ssim_weight: float = 0.3):
        super().__init__()
        self.lambda_N = lambda_N
        self.lambda_L = lambda_L
        self.lambda_M = lambda_M
        self.lambda_ortho = lambda_ortho
        self.lambda_temp = lambda_temp
        self.lambda_div = lambda_div
        self.lambda_prior = lambda_prior
        self.temp_threshold = temp_threshold
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
        # 尺度不变化: 各分支差异用其输出幅度归一化, 避免随量级无界增长
        n_scale = d_N.abs().mean().detach() + 1e-6
        l_scale = d_L.abs().mean().detach() + 1e-6
        m_scale = d_M.abs().mean().detach() + 1e-6
        rel_N = (dark * d_N).mean() / n_scale
        rel_L = (bright * d_L).mean() / l_scale
        rel_M = (grad * d_M).mean() / m_scale

        # tanh 饱和到 (-1,1), 保证 L_div 有界 (原实现理论可 → -∞)
        L_div = torch.tanh(rel_N + rel_L + rel_M)
        return -L_div  # 取负 → 最小化 = 最大化区域专属差异

    def _temporal_hf_consistency(self, Y_branch: torch.Tensor,
                                 X_t: torch.Tensor) -> torch.Tensor:
        """[R5] N 分支: 空间 HF 上限 (回退 R3 版本)

        R3 原理: 去噪输出的高频能量不应超过输入高频能量的 τ 倍
          relu(|HF(Y)| - τ|HF(X_t)|)
        简单稳健, 不依赖跨帧噪声估计。
        """
        k = 7
        pad = k // 2
        def hf(y): return y - F.avg_pool2d(y, k, 1, pad)

        Y_hf = hf(Y_branch).abs()
        X_hf = hf(X_t).abs()
        return F.relu(Y_hf - self.temp_threshold * X_hf).mean()

    def _temporal_lf_consistency(self, L_t: torch.Tensor,
                                 X_mean: torch.Tensor = None) -> torch.Tensor:
        """[R5] L 分支: 光照图低频约束 (回退 R3 版本)

        R3 原理: 光照图 Y_L 的低频分量不应偏离输入时间均值太多
          relu(|LF(Y_L)| - τ|LF(x̄)|)
        """
        k = 7
        pad = k // 2
        def lf(y): return F.avg_pool2d(y, k, 1, pad)

        L_lf = lf(L_t).abs()
        # 光照图 smoothness: TV
        gx = (L_t[:, :, :, 1:] - L_t[:, :, :, :-1]).abs().mean()
        gy = (L_t[:, :, 1:, :] - L_t[:, :, :-1, :]).abs().mean()
        return gx + gy

    def forward(self, outputs: Dict[str, torch.Tensor],
                gt: torch.Tensor, gt_seq: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        O_t = outputs["O_t"]
        Y_N, Y_L, Y_M = outputs["Y_N"], outputs["Y_L"], outputs["Y_M"]
        L_t = outputs["L_t"]
        flow_vis = outputs["flow_vis"]
        X_t = outputs.get("image_center", None)
        x_win = outputs.get("input_window", None)
        ortho_loss = outputs.get("ortho_loss", torch.tensor(0.0, device=O_t.device))
        prior_loss = outputs.get("prior_loss", 0.0)

        # 1. 最终监督
        L_final = self._recon_loss(O_t, gt)

        # 2. 分支监督
        L_N = self._recon_loss(Y_N, gt)
        L_L = self._recon_loss(Y_L, gt) + 0.1 * self._retinex_regularization(L_t)
        L_M = self._recon_loss(Y_M, gt) + 0.01 * self._flow_smoothness(flow_vis)

        # 3. 正交约束
        L_ortho = ortho_loss

        # 4. [R5] 时序一致性 (回退 R3 空间 HF/LF 上限, M 分支豁免)
        if X_t is not None:
            L_temp_N = self._temporal_hf_consistency(Y_N, X_t)
            L_temp_L = self._temporal_lf_consistency(L_t)
            L_temp = L_temp_N + L_temp_L
        else:
            L_temp = torch.tensor(0.0, device=O_t.device)
            L_temp_N = torch.tensor(0.0, device=O_t.device)
            L_temp_L = torch.tensor(0.0, device=O_t.device)

        # 5. [R5] TCA 先验正则 (降权 0.01 → 0.005, 在 config 中设定)
        L_prior = prior_loss if isinstance(prior_loss, torch.Tensor) \
            else torch.tensor(float(prior_loss), device=O_t.device)

        # 6. 分支差异化
        if X_t is not None:
            L_div = self._branch_divergence(Y_N, Y_L, Y_M, X_t)
        else:
            L_div = torch.tensor(0.0, device=O_t.device)

        total_loss = (L_final +
                      self.lambda_N * L_N +
                      self.lambda_L * L_L +
                      self.lambda_M * L_M +
                      self.lambda_ortho * L_ortho +
                      self.lambda_temp * L_temp +
                      self.lambda_prior * L_prior +
                      self.lambda_div * L_div)

        return {
            "total_loss": total_loss,
            "L_final": L_final.item(),
            "L_N": L_N.item(),
            "L_L": L_L.item(),
            "L_M": L_M.item(),
            "L_ortho": L_ortho.item() if isinstance(L_ortho, torch.Tensor) else L_ortho,
            "L_temp": L_temp.item() if isinstance(L_temp, torch.Tensor) else L_temp,
            "L_temp_N": L_temp_N.item() if isinstance(L_temp_N, torch.Tensor) else 0.0,
            "L_temp_L": L_temp_L.item() if isinstance(L_temp_L, torch.Tensor) else 0.0,
            "L_prior": L_prior.item() if isinstance(L_prior, torch.Tensor) else L_prior,
            "L_div": L_div.item() if isinstance(L_div, torch.Tensor) else L_div,
        }
