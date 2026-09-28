import math

import torch

from losses.losses import ssim_map


def tensor_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def tensor_ssim(pred, target):
    return ssim_map(pred, target).mean().item()


class LPIPSMetric:
    """LPIPS 感知距离 (懒加载, 失败时静默降级为 None)

    用法:
        lp = LPIPSMetric()
        d = lp(pred, target)   # pred/target ∈ [0,1]; 返回 float 或 None
    """

    def __init__(self, net='vgg', device='cuda'):
        self.fn = None
        self.device = device
        try:
            import lpips
            self.fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()
            self.net = net
        except Exception:
            self.fn = None

    @property
    def available(self):
        return self.fn is not None

    @torch.no_grad()
    def __call__(self, pred, target):
        if self.fn is None:
            return None
        # LPIPS 期望输入 [-1, 1]
        pred_n = pred.clamp(0, 1) * 2 - 1
        target_n = target.clamp(0, 1) * 2 - 1
        return self.fn(pred_n.to(self.device), target_n.to(self.device)).item()

