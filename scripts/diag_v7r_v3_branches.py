#!/usr/bin/env python3
"""v7r-v3 分支解耦诊断 (对应 docs/v7/03-v7r-v3-design.md §5.5.6 / §6.2)。

检查三路特征是否正交、三路 RGB 输出是否分化、融合权重分布、中心帧残差门。
结论 (best.pth = ep55): 特征正交 (ortho≈0.003–0.014) 但输出余弦 0.99–0.9995 →
解耦未转化为输出多样性。

用法:
    PYTHONPATH=/home/a1005/25/TFS-Net python scripts/diag_v7r_v3_branches.py \
        --ckpt outputs/golf_v7r_v3_pospair/best.pth
"""
import argparse
import glob

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

from models.golf_v7r import GolfNet_v7r_v3


def read_image(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def cosine(a, b):
    a = Fn.normalize(a.flatten(1), dim=1)
    b = Fn.normalize(b.flatten(1), dim=1)
    return (a * b).sum(1).abs().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/golf_v7r_v3_pospair/best.pth")
    p.add_argument("--seq_root", default="/home/a1005/yzy/dataset/SDSD/indoor/input")
    p.add_argument("--seqs", nargs="+", default=["pair50", "pair45", "pair19"])
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = GolfNet_v7r_v3().to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    print("RESULT loaded epoch", ck.get("epoch"), "from", args.ckpt)

    for seq in args.seqs:
        lq = sorted(glob.glob(f"{args.seq_root}/{seq}/*"))
        ci = len(lq) // 2
        inds = [min(max(ci + o, 0), len(lq) - 1) for o in (-2, -1, 0, 1, 2)]
        clip = torch.stack([read_image(lq[j]) for j in inds], 0).unsqueeze(0).to(dev)
        with torch.no_grad():
            out = model(clip)
        w = out["fusion_weights"][0]
        print(f"RESULT [{seq}] branch mean N/L/M =",
              [round(out[k].mean().item(), 4) for k in ["branch_N", "branch_L", "branch_M"]])
        print(f"RESULT [{seq}] fusion w mean N/L/M =", [round(w[i].mean().item(), 3) for i in range(3)])
        print(f"RESULT [{seq}] fusion w std  N/L/M =", [round(w[i].std().item(), 3) for i in range(3)])
        print(f"RESULT [{seq}] out cos NL/NM/LM =",
              round(cosine(out["branch_N"], out["branch_L"]), 4),
              round(cosine(out["branch_N"], out["branch_M"]), 4),
              round(cosine(out["branch_L"], out["branch_M"]), 4))
        print(f"RESULT [{seq}] ortho={out['ortho_loss'].item():.5f}")

    tca = model.triple_query_tca
    print("RESULT scale_N/L/M (mean abs) =",
          [round(getattr(tca, n).abs().mean().item(), 4) for n in ["scale_N", "scale_L", "scale_M"]])
    print("RESULT fusion gamma=tanh(residual_gamma) =",
          round(torch.tanh(model.fusion.residual_gamma).item(), 5))


if __name__ == "__main__":
    main()
