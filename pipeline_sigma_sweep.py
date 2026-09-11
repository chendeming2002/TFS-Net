#!/usr/bin/env python3
"""
Sigma sensitivity sweep for the denoising stage (FastDVDNet) in the serial pipeline.

Question from the three-branch ablation: FastDVDNet degrades the dark SDSD input
(checkerboard artifacts) and hurts the final brightened result. Does a different
assumed noise level (sigma) fix this?

For each sigma in {5, 10, 15, 25, 50} (0-255 scale):
  - D(sigma)        : FastDVDNet on raw input
  - D(sigma) -> B   : FastDVDNet then StableLLVE

Metrics: LPIPS / PSNR / SSIM / NIQE vs GT.

Usage:
  python pipeline_sigma_sweep.py --input ... --gt_dir ... --output outputs/pipeline_ablation
"""

import os
import sys
import csv
import argparse
import numpy as np
import cv2
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from pipeline_full import (
    FASTDVDNET_DIR, STABLELLVE_DIR,
    load_frames, build_fastdvdnet, build_stablellve,
    stage_denoise, stage_brighten,
    compute_lpips, compute_psnr, compute_ssim,
)
from pipeline_ablation import compute_niqe, load_niqe_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--output', default='outputs/pipeline_ablation')
    parser.add_argument('--num_frames', type=int, default=11)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--sigmas', type=str, default='5,10,15,25,50')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)
    sigmas = [float(s) for s in args.sigmas.split(',')]

    frames, filenames = load_frames(args.input, args.num_frames)
    gt_frames, _ = load_frames(args.gt_dir, args.num_frames)
    n = min(len(frames), len(gt_frames))
    frames, gt_frames, filenames = frames[:n], gt_frames[:n], filenames[:n]
    if args.scale < 1.0:
        H, W = frames.shape[1:3]
        nh, nw = int(H * args.scale), int(W * args.scale)
        nh, nw = nh - nh % 4, nw - nw % 4
        frames = np.stack([cv2.resize(f, (nw, nh)) for f in frames])
        gt_frames = np.stack([cv2.resize(g, (nw, nh)) for g in gt_frames])

    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)
    niqe_ctx = load_niqe_context()

    print('[Sweep] Loading FastDVDNet + StableLLVE ...')
    fd_model = build_fastdvdnet(os.path.join(FASTDVDNET_DIR, 'model.pth'), device)
    st_model = build_stablellve(os.path.join(STABLELLVE_DIR, 'checkpoint.pth'), device)

    rows = []
    for s in sigmas:
        print(f'\n[Sweep] sigma = {s}')
        d = stage_denoise(frames, fd_model, s, device)
        d_b = stage_brighten(d, st_model, device)
        for name, arr in [(f'D(sigma={s:g})', d), (f'D(sigma={s:g})+B', d_b)]:
            for i in range(len(arr)):
                row = {
                    'config': name,
                    'sigma': s,
                    'frame': i,
                    'lpips': compute_lpips(arr[i], gt_frames[i], lpips_model, device),
                    'psnr': float(compute_psnr(arr[i], gt_frames[i])),
                    'ssim': float(compute_ssim(arr[i], gt_frames[i])),
                    'niqe': compute_niqe(arr[i], niqe_ctx),
                }
                rows.append(row)
            vals = [r for r in rows if r['config'] == name]
            print(f"  {name:18s} LPIPS={np.mean([v['lpips'] for v in vals]):.4f} "
                  f"PSNR={np.mean([v['psnr'] for v in vals]):.2f} "
                  f"SSIM={np.mean([v['ssim'] for v in vals]):.4f} "
                  f"NIQE={np.mean([v['niqe'] for v in vals]):.3f}")

    with open(os.path.join(args.output, 'sigma_sweep.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['config', 'sigma', 'frame',
                                               'lpips', 'psnr', 'ssim', 'niqe'])
        writer.writeheader()
        writer.writerows(rows)

    # Figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (m, title, better) in zip(axes.flat, [
            ('lpips', 'LPIPS (lower better)', 'lower'),
            ('psnr', 'PSNR / dB (higher better)', 'higher'),
            ('ssim', 'SSIM (higher better)', 'higher'),
            ('niqe', 'NIQE (lower better)', 'lower')]):
        for name, marker in [(f'D', 'o'), (f'D+B', 's')]:
            xs, ys = [], []
            for s in sigmas:
                key = f'D(sigma={s:g})' if name == 'D' else f'D(sigma={s:g})+B'
                vals = [r[m] for r in rows if r['config'] == key]
                xs.append(s)
                ys.append(np.mean(vals))
            ax.plot(xs, ys, marker=marker, label=name)
        ax.set_xlabel('assumed sigma (0-255)')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(args.output, 'sigma_sweep.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'\n[Sweep] Saved: {fig_path}')
    print(f'[Sweep] CSV: {os.path.join(args.output, "sigma_sweep.csv")}')


if __name__ == '__main__':
    main()
