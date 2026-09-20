"""
Golf Loss Function
===================
在 FoxtrotLoss 基础上新增两项:

1. [时序一致性] L_temp — 分支选择性高频稳定
   仅约束 Branch-N/L 的高频能量，保证去噪和光照分支不引入噪声放大。
   Branch-M（运动对齐）豁免约束，允许细节恢复产生的合理高频增强。
   设计理由：M 分支职责是边缘/结构保持，需要高频增强自由。

2. [分支差异化弱监督] L_branch_div
   Foxtrot 三分支都回归同一 GT → 功能冗余。
   Golf 给分支加差异化引导:
     - Branch-N (噪声): 侧重暗区/平坦区域
     - Branch-L (光照): 侧重全局亮度一致性
     - Branch-M (运动): 侧重边缘/结构保持
   通过空间加权 mask 实现 (不需要额外 GT), 最小化 = 最大化区域专属性。

[已于 2026-09-17 删除] L_chess — 频域棋盘格惩罚
   删除理由:
   (1) 棋盘格是推理期结构问题 (PixelShuffle 子像素隔离 + tiled_forward
       均匀平均), 已由 resize-conv 上采样 + 余弦窗口缝合从结构上修复;
       用训练损失去补救推理期伪影属于原理错位。
   (2) 实测该损失项在 Golf 训练中恒为 0 (mask 逻辑错误致死代码),
       从未产生任何贡献; 且其绝对能量形式会无差别压制正常高频细节。
   (3) 标定困难: 需要同时调对 mask 位置与半径, 收益为负。

损失结构:
  L_total = L_final + λ_N·L_N + λ_L·L_L + λ_M·L_M
            + λ_ortho·L_ortho + λ_temp·(L_temp_N + L_temp_L) + λ_div·L_div
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
                 lambda_prior: float = 0.01,    # R4: TCA 结构化先验正则
                 temp_threshold: float = 1.2,   # 保留兼容 (R4 不再使用)
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
                                 X_mean: torch.Tensor) -> torch.Tensor:
        """[R4] N 分支: 噪声的帧间 i.i.d. 一致性 (覆盖 Read+Shot 两类噪声).

        物理依据:
          成像噪声 = Read Noise (信号无关) + Shot Noise (信号相关, Poisson)
          两者共享关键性质: 【跨帧 i.i.d.】→ 时间平均是最优估计器 (Foi 2008)
          故去噪输出 Y_N 的【高频分量】应逼近【输入窗口时间均值】的高频分量:
            - Read noise:  高斯, 时间平均后方差 /N  → HF(x̄) ≈ 干净HF
            - Shot noise:  泊松, 时间平均后方差 /N  → HF(x̄) ≈ 干净HF
          两者都被同一策略覆盖 (这正是 TSDR 合并 I+II 为"成像噪声"的依据)。

        与旧实现 (Golf-R3) 的区别:
          旧: relu(|HF(Y_N)| - 1.2|HF(X_t)|) — 纯空间 HF 上限, 无任何跨帧信息
          新: |HF(Y_N) - HF(x̄)|                — 真正的时序参考 (用 5 帧窗口均值)
        """
        k = 7
        pad = k // 2
        Y_hf = Y_branch - F.avg_pool2d(Y_branch, k, 1, pad)
        M_hf = X_mean - F.avg_pool2d(X_mean, k, 1, pad)
        return F.l1_loss(Y_hf, M_hf.detach())

    def _temporal_lf_consistency(self, Y_branch: torch.Tensor,
                                 X_mean: torch.Tensor) -> torch.Tensor:
        """[R4] L 分支: 光照的帧间缓变一致性 (邻帧不突变).

        物理依据:
          光照扰动是【帧间强相关、慢变】→ 相邻帧照度几乎相同
          → 时间均值 x̄ 的低频 ≈ 每一帧的低频 (因为光照本就慢变)
          → 要求光照分支输出的低频逼近 HF(x̄) 的低频, 即抑制帧间亮度突变。
        """
        k = 15   # 光照是低频, 用更大核 (对应更低的频带)
        pad = k // 2
        Y_lf = F.avg_pool2d(Y_branch, k, 1, pad)
        M_lf = F.avg_pool2d(X_mean, k, 1, pad)
        return F.l1_loss(Y_lf, M_lf.detach())

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

        # 4. [R4] 真·时序一致性 (N: 噪声i.i.d. / L: 光照缓变), M 分支豁免
        if x_win is not None:
            X_mean = x_win.mean(dim=1)   # (B,3,H,W) 时间均值 — 物理上的"去噪参考"
            L_temp_N = self._temporal_hf_consistency(Y_N, X_mean)
            L_temp_L = self._temporal_lf_consistency(Y_L, X_mean)
            L_temp = L_temp_N + L_temp_L
        else:
            L_temp = torch.tensor(0.0, device=O_t.device)
            L_temp_N = torch.tensor(0.0, device=O_t.device)
            L_temp_L = torch.tensor(0.0, device=O_t.device)

        # 5. [R4] TCA 三路结构化先验正则 (N: 均值一致 / L: 空间TV / M: L1稀疏)
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
