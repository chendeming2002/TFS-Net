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
        self._noise_ref = None   # R4-fix: 帧间噪声参考 (供 L_t 去相关)
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
                                 X_mean: torch.Tensor,
                                 X_center: torch.Tensor = None) -> torch.Tensor:
        """[R4-fix] N 分支: 输出不应保留输入噪声 (亮度不变形式).

        物理依据 (覆盖 Read + Shot 两类噪声):
          成像噪声 = Read Noise (信号无关, 高斯) + Shot Noise (信号相关, 泊松)
          两者共享【跨帧 i.i.d.】→ 时间平均是最优估计器 (Foi 2008)
          帧间差异 (x_c - x̄) 即噪声分量估计; 去噪输出 Y_N 的 HF 应【与噪声不相关】。

        为何用相关而非 L1 差 (前两版都错):
          旧版A: relu(|HF(Y)| - 1.2|HF(X_t)|)     — 纯空间上限, 无跨帧
          旧版B: |HF(Y) - HF(x̄)|                  — 提亮按比例放大 HF 幅度 →
                 实测 "不处理"(0.0000) < "理想GT"(0.4109) → 惩罚正确增强 ✗
          新版:  |cos(HF(Y), HF(x_c - x̄))|        — 余弦对亮度缩放不变 (k^0),
                 实测判别力 14x (原图0.857 vs 理想GT0.062) ✓
        """
        k = 7
        pad = k // 2
        def hf(y): return y - F.avg_pool2d(y, k, 1, pad)

        Y_hf = hf(Y_branch).flatten(1)
        if X_center is not None:
            noise_hf = hf(X_center - X_mean).detach().flatten(1)  # 噪声分量 HF
        else:
            noise_hf = hf(X_mean).detach().flatten(1)
        cos = F.cosine_similarity(Y_hf, noise_hf, dim=1, eps=1e-6)
        return cos.abs().mean()

    def _temporal_lf_consistency(self, L_t: torch.Tensor) -> torch.Tensor:
        """[R4-fix] L 分支: 光照图时序稳定性 (防帧间闪烁).

        物理依据: 光照扰动帧间强相关、慢变 → 光照图 L_t 应平滑, 不得携带
        帧间噪声 (噪声是帧间 i.i.d., 是闪烁的直接来源)。

        ⚠ 为何前两版都错 (用 Y_L 的低频去匹配输入时间均值):
          Y_L = X_t · L_t^(γ-1) — 光照校正【本就改变低频】(这是它的职责),
          γ-1>0 时提亮增大输出幅值 → |LF(Y_L)-LF(x̄)| 被亮度差主导,
          实测 "不处理"(0.0000) < "理想GT"(0.4109) → 惩罚正确校正 ✗
        正确对象是【光照图 L_t】而非 Y_L: L_t 必须慢变, 与帧间噪声无关。

        实现: L_t 的 HF 与帧间噪声去相关 (亮度不变, 与 N 分支同一原则)。
        """
        k = 7
        pad = k // 2
        L_hf = (L_t - F.avg_pool2d(L_t, k, 1, pad)).flatten(1)
        if self._noise_ref is not None:
            # L_t 单通道, 噪声参考多通道 → 取通道均值匹配
            nref = self._noise_ref.mean(dim=1, keepdim=True) if self._noise_ref.shape[1] != 1 else self._noise_ref
            n_hf = (nref - F.avg_pool2d(nref, k, 1, pad)).detach().flatten(1)
            cos = F.cosine_similarity(L_hf, n_hf, dim=1, eps=1e-6)
            return cos.abs().mean()
        # 无噪声参考时退回空间 TV (保证平滑)
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

        # 4. [R4-fix] 真·时序一致性 (N: 噪声去相关 / L: 光照图防闪烁), M 分支豁免
        if x_win is not None:
            X_mean = x_win.mean(dim=1)       # 时间均值 — 去噪参考
            self._noise_ref = (X_t - X_mean) if X_t is not None else None  # 帧间噪声分量
            L_temp_N = self._temporal_hf_consistency(Y_N, X_mean, X_center=X_t)
            L_temp_L = self._temporal_lf_consistency(L_t)
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
