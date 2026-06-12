import torch
import torch.nn as nn

from .blocks import ResBlock, pairwise_cosine_logits, safe_divide


class ISPN(nn.Module):
    def __init__(self, channels=48, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.refine = ResBlock(channels)

    def forward(self, f_t_i, f_omega_i):
        logits = pairwise_cosine_logits(f_t_i, f_omega_i)
        alpha = torch.softmax(logits, dim=1)
        l_shared = (alpha[:, :, None, None, None] * f_omega_i).sum(dim=1)
        r_t = safe_divide(f_t_i, l_shared, self.eps)
        hat_f_t_i = self.refine(r_t) + f_t_i
        return {
            "alpha": alpha,
            "L_shared": l_shared,
            "R_t": r_t,
            "hat_f_t_i": hat_f_t_i,
        }

