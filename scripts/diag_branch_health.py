#!/usr/bin/env python3
"""分支健康度诊断: 判断 N/L/M 三路是在输出内容, 还是在输出常数图。

为什么需要这个脚本:
    单看分支 PSNR 无法区分"学会了重建"和"塌成常数图" —— 一张接近均值的灰色图
    在暗场景里也能拿到不低的 PSNR。真正能区分两者的是输出的【空间方差】:
    同一张特征图上逐通道求方差再平均, 约 1e-4 量级说明图内有结构,
    1e-9 及以下说明整张图几乎是同一个值 (常数图)。设计文档
    docs/v8/05-UNet重构设计.md 的判据 1 就是这条指标。

为什么固定 256 裁剪:
    全分辨率 (如 1920x1080) 会把大量均匀暗区算进方差, 稀释局部结构;
    256x256 的窗口同时保证足够的分辨率和可重复性, 所有实验用同一位置。

用法:
    PYTHONPATH=. python scripts/diag_branch_health.py \
        --ckpt outputs/v8_unet_refactor_15ep/checkpoints/epoch_15.pth \
        --config configs/v8/unet_refactor_15ep.yaml \
        --n_seqs 24 --crop 256
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image


def read_image(path, crop=None):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1)
    if crop:
        h, w = img.shape[-2:]
        top, left = (h - crop) // 2, (w - crop) // 2
        img = img[:, top:top + crop, left:left + crop]
    return img


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))


def space_var(x):
    """逐通道空间方差再对通道求平均, 单位沿用输入的数值尺度。"""
    return x.var(dim=(2, 3), unbiased=False).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seq_root", default="/home/a1005/yzy/dataset/SDSD/indoor/input")
    p.add_argument("--gt_root", default="/home/a1005/yzy/dataset/SDSD/indoor/GT")
    p.add_argument("--n_seqs", type=int, default=24)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import yaml
    from models.golf_v7r import GolfNet_v7r_v3, GolfNet_v8

    with open(args.config) as f:
        model_kwargs = dict((yaml.safe_load(f) or {}).get("model", {}) or {})
    arch = model_kwargs.pop("arch", "v7r_v3")
    model_cls = GolfNet_v8 if arch == "v8" else GolfNet_v7r_v3

    dev = args.device if torch.cuda.is_available() else "cpu"
    model = model_cls(**model_kwargs).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    print(f"RESULT loaded epoch {ck.get('epoch')} arch={arch}", flush=True)

    names = sorted(os.listdir(args.seq_root),
                   key=lambda x: int(x.replace("pair", "")) if x.replace("pair", "").isdigit() else 0)
    seqs = names[:args.n_seqs]

    acc = {k: [] for k in ("varN", "varL", "varM", "varF",
                           "psnrN", "psnrL", "psnrM", "psnrF",
                           "fmean", "fstd", "wN", "wL", "wM", "wT")}

    with torch.no_grad():
        for seq in seqs:
            low_dir = os.path.join(args.seq_root, seq)
            gt_dir = os.path.join(args.gt_root, seq)
            low_files = sorted(os.listdir(low_dir))
            gt_files = sorted(os.listdir(gt_dir))
            if not low_files or not gt_files:
                continue
            center = len(low_files) // 2
            idxs = [min(max(center + off, 0), len(low_files) - 1)
                    for off in (-2, -1, 0, 1, 2)]
            clip = torch.stack([read_image(os.path.join(low_dir, low_files[i]), args.crop)
                                for i in idxs], 0).unsqueeze(0).to(dev)
            gt = read_image(os.path.join(gt_dir, gt_files[min(center, len(gt_files) - 1)]),
                            args.crop).unsqueeze(0).to(dev)

            out = model(clip)
            y_n, y_l, y_m, final = (out["branch_N"], out["branch_L"],
                                    out["branch_M"], out["final"])

            acc["varN"].append(space_var(y_n))
            acc["varL"].append(space_var(y_l))
            acc["varM"].append(space_var(y_m))
            acc["varF"].append(space_var(final))
            acc["psnrN"].append(psnr(y_n, gt))
            acc["psnrL"].append(psnr(y_l, gt))
            acc["psnrM"].append(psnr(y_m, gt))
            acc["psnrF"].append(psnr(final, gt))
            acc["fmean"].append(final.mean().item())
            acc["fstd"].append(final.std().item())

            w = out.get("fusion_weights")
            if w is not None:
                # additive 模式是三路独立权重; gated 模式只有一个逐通道门, 记为 wT
                if getattr(model.fusion, "mode", "") == "gated":
                    acc["wT"].append(w.mean().item())
                else:
                    acc["wN"].append(w[:, 0].mean().item())
                    acc["wL"].append(w[:, 1].mean().item())
                    acc["wM"].append(w[:, 2].mean().item())

    def mean(key):
        return float(np.mean(acc[key])) if acc[key] else float("nan")

    print(f"RESULT {len(acc['varN'])} sequences, crop={args.crop}", flush=True)
    print(f"空间方差   N={mean('varN'):.3e}  L={mean('varL'):.3e}  "
          f"M={mean('varM'):.3e}  Final={mean('varF'):.3e}", flush=True)
    print(f"单独 PSNR  N={mean('psnrN'):.2f}  L={mean('psnrL'):.2f}  "
          f"M={mean('psnrM'):.2f} dB", flush=True)
    print(f"最终 PSNR={mean('psnrF'):.2f} dB  mean={mean('fmean'):.3f}  "
          f"std={mean('fstd'):.3f}", flush=True)
    print(f"GT   mean={gt.mean().item():.3f}  std={gt.std().item():.3f}", flush=True)
    if acc["wT"]:
        print(f"门 g mean={mean('wT'):.3f}", flush=True)
    elif acc["wN"]:
        print(f"融合权重 N={mean('wN'):.3f}  L={mean('wL'):.3f}  "
              f"M={mean('wM'):.3f}", flush=True)

    # 判读提示: 方差低于 1e-7 基本可认定为常数图
    for name in ("N", "L", "M"):
        v = mean(f"var{name}")
        state = "常数图" if v < 1e-7 else ("弱结构" if v < 1e-5 else "有结构")
        print(f"RESULT branch_{name}: {v:.3e} -> {state}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
