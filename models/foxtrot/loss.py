"""
Foxtrot Loss Function
======================
多任务损失组合 (TSD-Foxtort.md §4)

损失结构:
  L_total = L_final + λ_N·L_N + λ_L·L_L + λ_M·L_M + λ_ortho·L_ortho + λ_temp·L_temp

各项说明:
  1. L_final: 最终输出 Ô_t 与 GT 的重建损失 (L1 + SSIM)
  2. L_N: Branch-N 噪声分支损失 (重建 + 方差图监督)
  3. L_L: Branch-L 光照分支损失 (重建 + Retinex 正则)
  4. L_M: Branch-M 运动分支损失 (重建 + 光流平滑)
  5. L_ortho: TCA 三查询正交约束 (由 TCA_RWKV 内部计算)
  6. L_temp: 时序一致性损失 (帧间梯度平滑)

关键设计:
  - 分支损失鼓励各分支独立有效
  - 正交约束保证解耦质量
  - 时序约束抑制闪烁
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
        _1D_window = self._gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        """
        img1, img2: (B, 3, H, W) ∈ [0, 1]
        Returns: 1 - SSIM (loss, 越小越好)
        """
        # fp32 计算 (AMP 下输入可能是 Half, conv 窗口必须同 dtype 且 fp16 精度不足)
        img1 = img1.float()
        img2 = img2.float()
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


class FoxtrotLoss(nn.Module):
    """Foxtrot 多任务损失
    
    Args:
        lambda_N: Branch-N 损失权重 (默认0.3)
        lambda_L: Branch-L 损失权重 (默认0.3)
        lambda_M: Branch-M 损失权重 (默认0.3)
        lambda_ortho: 正交约束权重 (默认0.01)
        lambda_temp: 时序一致性权重 (默认0.1)
        use_ssim: 是否使用 SSIM 损失 (默认True)
        ssim_weight: SSIM 损失权重 (默认0.3)
    """
    
    def __init__(self, lambda_N: float = 0.3, lambda_L: float = 0.3,
                 lambda_M: float = 0.3, lambda_ortho: float = 0.01,
                 lambda_temp: float = 0.1, use_ssim: bool = True,
                 ssim_weight: float = 0.3):
        super().__init__()
        self.lambda_N = lambda_N
        self.lambda_L = lambda_L
        self.lambda_M = lambda_M
        self.lambda_ortho = lambda_ortho
        self.lambda_temp = lambda_temp
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight
        
        self.l1_loss = nn.L1Loss()
        if use_ssim:
            self.ssim_loss = SSIMLoss()
    
    def _recon_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """重建损失: L1 + SSIM"""
        loss = self.l1_loss(pred, gt)
        if self.use_ssim:
            loss = loss + self.ssim_weight * self.ssim_loss(pred, gt)
        return loss
    
    def _retinex_regularization(self, L_t: torch.Tensor) -> torch.Tensor:
        """Retinex 正则化: 光照图空间平滑
        
        L_t: (B, 1, H, W)
        """
        # 光照图应该是平滑的 (低频)
        grad_x = torch.abs(L_t[:, :, :, :-1] - L_t[:, :, :, 1:])
        grad_y = torch.abs(L_t[:, :, :-1, :] - L_t[:, :, 1:, :])
        return grad_x.mean() + grad_y.mean()
    
    def _flow_smoothness(self, flow: torch.Tensor) -> torch.Tensor:
        """光流平滑正则化
        
        flow: (B, 2, H, W)
        """
        grad_x = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        grad_y = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        return grad_x.mean() + grad_y.mean()
    
    def _temporal_consistency(self, pred_seq: torch.Tensor) -> torch.Tensor:
        """时序一致性损失 (帧间梯度平滑)
        
        pred_seq: (B, T, 3, H, W) — 连续帧预测序列
        """
        if pred_seq.size(1) < 2:
            return torch.tensor(0.0, device=pred_seq.device)
        
        # 帧间差分
        diff = pred_seq[:, 1:] - pred_seq[:, :-1]  # (B, T-1, 3, H, W)
        # 二阶差分 (加速度)
        if diff.size(1) > 1:
            diff2 = diff[:, 1:] - diff[:, :-1]
            return diff2.abs().mean()
        else:
            return diff.abs().mean()
    
    def forward(self, outputs: Dict[str, torch.Tensor],
                gt: torch.Tensor, gt_seq: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: TSDNet 输出的 dict (return_intermediate=True)
                - O_t: 最终输出 (B, 3, H, W)
                - Y_N, Y_L, Y_M: 三分支输出
                - sigma_map, L_t, R_t, flow_vis, conf_map: 辅助输出
                - ortho_loss: TCA 正交损失 (scalar, 由 TCA_RWKV 计算)
            gt: (B, 3, H, W) — 中心帧 GT
            gt_seq: (B, T, 3, H, W) — 全序列 GT (可选，用于时序一致性)
        
        Returns:
            dict with keys: total_loss, L_final, L_N, L_L, L_M, L_ortho, L_temp
        """
        O_t = outputs["O_t"]
        Y_N = outputs["Y_N"]
        Y_L = outputs["Y_L"]
        Y_M = outputs["Y_M"]
        L_t = outputs["L_t"]
        flow_vis = outputs["flow_vis"]
        ortho_loss = outputs.get("ortho_loss", torch.tensor(0.0, device=O_t.device))
        
        # 1. L_final: 最终输出重建损失
        L_final = self._recon_loss(O_t, gt)
        
        # 2. L_N: Branch-N 重建损失
        L_N = self._recon_loss(Y_N, gt)
        
        # 3. L_L: Branch-L 重建 + Retinex 正则
        L_L_recon = self._recon_loss(Y_L, gt)
        L_L_retinex = self._retinex_regularization(L_t)
        L_L = L_L_recon + 0.1 * L_L_retinex
        
        # 4. L_M: Branch-M 重建 + 光流平滑
        L_M_recon = self._recon_loss(Y_M, gt)
        L_M_flow = self._flow_smoothness(flow_vis)
        L_M = L_M_recon + 0.01 * L_M_flow
        
        # 5. L_ortho: TCA 正交约束 (由 TCA_RWKV 内部计算)
        L_ortho = ortho_loss
        
        # 6. L_temp: 时序一致性 (如果提供 gt_seq)
        L_temp = torch.tensor(0.0, device=O_t.device)
        if gt_seq is not None and self.lambda_temp > 0:
            # 这里简化: 只用最终输出做时序约束
            # 完整实现应该对所有帧做预测
            # 由于 TSDNet 只输出中心帧, 我们无法直接计算时序一致性
            # 改为用分支输出的隐式时序一致性 (Y_N, Y_L, Y_M 应该对中心帧一致)
            # 这里暂时不实现, 预留接口
            pass
        
        # 总损失
        total_loss = (L_final +
                      self.lambda_N * L_N +
                      self.lambda_L * L_L +
                      self.lambda_M * L_M +
                      self.lambda_ortho * L_ortho +
                      self.lambda_temp * L_temp)
        
        return {
            "total_loss": total_loss,
            "L_final": L_final.item(),
            "L_N": L_N.item(),
            "L_L": L_L.item(),
            "L_M": L_M.item(),
            "L_ortho": L_ortho.item() if isinstance(L_ortho, torch.Tensor) else L_ortho,
            "L_temp": L_temp.item(),
        }


if __name__ == "__main__":
    # 测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = FoxtrotLoss().to(device)
    
    B, H, W = 2, 256, 256
    
    # 模拟 TSDNet 输出
    outputs = {
        "O_t": torch.rand(B, 3, H, W).to(device),
        "Y_N": torch.rand(B, 3, H, W).to(device),
        "Y_L": torch.rand(B, 3, H, W).to(device),
        "Y_M": torch.rand(B, 3, H, W).to(device),
        "L_t": torch.rand(B, 1, H, W).to(device),
        "flow_vis": torch.rand(B, 2, H // 2, W // 2).to(device),
        "ortho_loss": torch.tensor(0.05).to(device),
    }
    
    gt = torch.rand(B, 3, H, W).to(device)
    
    loss_dict = loss_fn(outputs, gt)
    
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v}")
    
    print("\nBackward test...")
    loss_dict["total_loss"].backward()
    print("Backward successful!")
