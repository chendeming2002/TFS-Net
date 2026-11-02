#!/usr/bin/env python3
"""OmniShift 分支置零消融 (推理期干预, 不重训练)。

对 v8.8-B 的每个 OmniShift, 分别把 1×1 / 3×3 / 5×5 卷积分支的 alpha 置零,
记录最终 PSNR / SSIM / LPIPS 与三路分支输出的变化。

覆盖范围:
  - Encoder 三个尺度的 Spatial Shift (omni_shift) 与 Channel Shift (cm_shift);
  - TCA 三个 QueryRWKVBlock 的共享 shift_r/g、共享 shift_k/v、cm_shift。

注意: 置零只衡量训练完成后模型对该分支的即时依赖, 不等价于重训练时删除该分支。

用法:
    python scripts/diag_shift_ablation.py --ckpt <best.pth> --config <cfg> [--n_seqs 5]
"""
import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_image(path, crop):
    from PIL import Image
    import torchvision.transforms.functional as TF
    img = TF.to_tensor(Image.open(path).convert("RGB"))
    if crop:
        _, h, w = img.shape
        img = img[:, (h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seq_root", default="/home/a1005/yzy/dataset/SDSD/test/low-light")
    p.add_argument("--gt_root", default="/home/a1005/yzy/dataset/SDSD/test/GT")
    p.add_argument("--n_seqs", type=int, default=5)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import yaml
    from models.golf_v7r import GolfNet_v8
    from utils.io import load_state_dict_compat
    from utils.metrics import tensor_psnr, tensor_ssim

    cfg = yaml.safe_load(open(args.config))
    mk = dict(cfg.get("model", {}) or {})
    mk.pop("arch", None)
    dev = args.device if torch.cuda.is_available() else "cpu"
    model = GolfNet_v8(**mk).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    load_state_dict_compat(model, ck["model_state_dict"])

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    except Exception:
        lpips_fn = None

    # ---------- 收集所有 Shift 位置 ----------
    # (名称, 模块, 该模型的父引用列表)
    shifts = []
    for sname in ("scale1", "scale2", "scale3"):
        h = getattr(model.encoder, f"{sname}_blocks")[0].head
        if h.omni_shift is not None:
            shifts.append((f"enc.{sname}.spatial", h.omni_shift))
        if getattr(h, "cm_shift", None) is not None:
            shifts.append((f"enc.{sname}.cm", h.cm_shift))
    for bname in ("attn_N", "attn_L", "attn_M"):
        h = getattr(model.tca, bname)
        if getattr(h, "shift_r", None) is not None:
            shifts.append((f"tca.{bname}.rg", h.shift_r))
        if getattr(h, "shift_k", None) is not None:
            shifts.append((f"tca.{bname}.kv", h.shift_k))
        if getattr(h, "cm_shift", None) is not None:
            shifts.append((f"tca.{bname}.cm", h.cm_shift))

    # ---------- 数据 ----------
    names = sorted(os.listdir(args.seq_root),
                   key=lambda x: int(x.replace("pair", "")) if x.replace("pair", "").isdigit() else 0)
    clips, gts = [], []
    for seq in names[:args.n_seqs]:
        low_dir = os.path.join(args.seq_root, seq)
        gt_dir = os.path.join(args.gt_root, seq)
        lows = sorted(os.listdir(low_dir))
        gts_f = sorted(os.listdir(gt_dir))
        if not lows or not gts_f:
            continue
        c = len(lows) // 2
        idxs = [min(max(c + o, 0), len(lows) - 1) for o in (-2, -1, 0, 1, 2)]
        clip = torch.stack([_read_image(os.path.join(low_dir, lows[i]), args.crop)
                            for i in idxs], 0).to(dev).unsqueeze(0)
        gt = _read_image(os.path.join(gt_dir, gts_f[min(c, len(gts_f) - 1)]),
                         args.crop).to(dev).unsqueeze(0)
        clips.append(clip)
        gts.append(gt)

    # ---------- 评估函数 ----------
    def evaluate():
        ps, ss, lp, bn, bl, bm = [], [], [], [], [], []
        with torch.no_grad():
            for clip, gt in zip(clips, gts):
                o = model(clip)
                ps.append(tensor_psnr(o["final"], gt))
                ss.append(tensor_ssim(o["final"], gt))
                if lpips_fn is not None:
                    lp.append(lpips_fn(o["final"] * 2 - 1, gt * 2 - 1).item())
                bn.append(o["branch_N"].std().item())
                bl.append(o["branch_L"].std().item())
                bm.append(o["branch_M"].std().item())
        n = len(ps)
        return {"psnr": sum(ps) / n, "ssim": sum(ss) / n,
                "lpips": (sum(lp) / n) if lp else float("nan"),
                "bstd_N": sum(bn) / n, "bstd_L": sum(bl) / n, "bstd_M": sum(bm) / n}

    base = evaluate()
    print(f"基线: PSNR {base['psnr']:.3f} SSIM {base['ssim']:.4f} "
          f"LPIPS {base['lpips']:.4f} | bstd N/L/M "
          f"{base['bstd_N']:.4f}/{base['bstd_L']:.4f}/{base['bstd_M']:.4f}", flush=True)

    # ---------- 置零扫描 ----------
    results = {"baseline": base, "interventions": {}}
    header = (f"{'干预':<28}{'PSNR':>8}{'ΔPSNR':>8}{'SSIM':>8}{'LPIPS':>8}"
              f"{'bstd N/L/M':>22}")
    print("\n" + header, flush=True)

    for name, shift in shifts:
        alpha_orig = shift.alpha.detach().clone()
        for branch in (1, 2, 3):            # 1=1x1, 2=3x3, 3=5x5
            bname = {1: "1x1", 2: "3x3", 3: "5x5"}[branch]
            # OmniShift 推理用合并卷积; 需清缓存让新 alpha 生效
            shift._merged_cache[0] = None
            with torch.no_grad():
                shift.alpha[branch] = 0.0
            r = evaluate()
            tag = f"{name}.{bname}"
            results["interventions"][tag] = r
            print(f"{tag:<28}{r['psnr']:>8.3f}{r['psnr']-base['psnr']:>+8.3f}"
                  f"{r['ssim']:>8.4f}{r['lpips']:>8.4f}"
                  f"{r['bstd_N']:.3f}/{r['bstd_L']:.3f}/{r['bstd_M']:.3f}   ",
                  flush=True)
            # 恢复
            with torch.no_grad():
                shift.alpha.copy_(alpha_orig)
            shift._merged_cache[0] = None

    out_json = os.path.join(os.path.dirname(args.ckpt), "shift_ablation.json")
    with open(out_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_json}", flush=True)
    print("\n注意: 置零只衡量训练完成后的即时依赖, 不等价于重训练时删除该分支。",
          flush=True)


if __name__ == "__main__":
    main()
