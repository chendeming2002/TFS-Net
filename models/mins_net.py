import torch
import torch.nn as nn

from .modules.encoder import PyramidEncoder
from .modules.ispn import ISPN
from .modules.mins import MINSBlock
from .modules.mspn import MSPN
from .modules.reconstruction import FinalReconstruction


class MINSNet(nn.Module):
    def __init__(self, in_channels=3, level_channels=(32, 64, 96), fused_channels=48, window_size=8):
        super().__init__()
        self.encoder = PyramidEncoder(in_channels=in_channels, level_channels=level_channels, fused_channels=fused_channels)
        self.mins = MINSBlock(channels=fused_channels, window_size=window_size)
        self.ispn = ISPN(channels=fused_channels)
        self.mspn = MSPN(channels=fused_channels, window_size=window_size)
        self.reconstruction = FinalReconstruction(channels=fused_channels, out_channels=in_channels)

    def forward(self, x):
        b, t, c, h, w = x.shape
        assert t >= 3 and t % 2 == 1, "Temporal window must be odd and >= 3."
        feats = self.encoder(x)
        center_idx = t // 2
        f_t = feats[:, center_idx]
        f_omega = torch.cat([feats[:, :center_idx], feats[:, center_idx + 1 :]], dim=1)

        mins_out = self.mins(f_t, f_omega)
        ispn_out = self.ispn(mins_out["f_t_i"], mins_out["f_omega_i"])
        mspn_out = self.mspn(mins_out["f_t_m"], mins_out["f_omega_m"], mins_out["C_t_omega"])
        recon_out = self.reconstruction(
            image_center=x[:, center_idx],
            hat_f_t_i=ispn_out["hat_f_t_i"],
            hat_f_t_m=mspn_out["hat_f_t_m"],
            p_t_i=mins_out["P_t_i"],
            p_t_m=mins_out["P_t_m"],
        )

        return {
            "res_t": recon_out["res_t"],
            "delta": recon_out["delta"],
            "priors": {
                "P_t_m": mins_out["P_t_m"],
                "P_t_i": mins_out["P_t_i"],
                "P_omega_m": mins_out["P_omega_m"],
                "P_omega_i": mins_out["P_omega_i"],
            },
            "features": {
                "f_t_m": mins_out["f_t_m"],
                "f_t_i": mins_out["f_t_i"],
                "f_omega_m": mins_out["f_omega_m"],
                "f_omega_i": mins_out["f_omega_i"],
                "hat_f_t_m": mspn_out["hat_f_t_m"],
                "hat_f_t_i": ispn_out["hat_f_t_i"],
                "f_omega_aligned": mspn_out["f_omega_aligned"],
                "f_t_rec": recon_out["f_t_rec"],
            },
            "correspondence": mins_out["C_t_omega"],
            "aux": {
                "H_t": mins_out["H_t"],
                "L_shared": ispn_out["L_shared"],
                "alpha": ispn_out["alpha"],
                "G_t_m": mspn_out["G_t_m"],
            },
        }

