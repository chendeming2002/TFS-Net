#!/usr/bin/env python3
"""
Three-degradation-branch ablation for the serial video enhancement pipeline
(FastDVDNet -> CDVD-TSP -> StableLLVE), matching TSD-Net's degradation model:
  D  = FastDVDNet  (noise denoising)
  DB = CDVD-TSP    (motion deblurring)
  B  = StableLLVE  (illumination enhancement)

Ablation matrix:
  Input      : no processing
  D          : denoise only
  DB         : deblur only
  B          : brighten only
  D+DB       : denoise -> deblur
  D+B        : denoise -> brighten
  DB+B       : deblur -> brighten
  D+DB+B     : full pipeline
  DB+D+B     : order variant (deblur -> denoise -> brighten)

Metrics: LPIPS (AlexNet, lower better), PSNR / SSIM (higher better),
NIQE (no-reference, lower better).

Usage:
  python pipeline_ablation.py \
    --input /home/a1005/yzy/dataset/SDSD/test/low-light/pair19 \
    --gt_dir /home/a1005/yzy/dataset/SDSD/test/GT/pair19 \
    --output outputs/pipeline_ablation \
    --num_frames 11 --scale 0.5 --noise_sigma 25 --gpu 0
"""

import os
import sys
import csv
import json
import time
import argparse
import importlib.util
import numpy as np
import cv2
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, 'reference_repos')

from pipeline_full import (
    FASTDVDNET_DIR, CDVD_TSP_DIR, STABLELLVE_DIR,
    load_frames, save_frames,
    build_fastdvdnet, build_cdvd_tsp, build_stablellve,
    stage_denoise, stage_deblur, stage_brighten,
    compute_lpips, compute_psnr, compute_ssim,
)

HVI_DIR = os.path.join(REPO_DIR, 'HVI-CIDNet')

CONFIGS = ['Input', 'D', 'DB', 'B', 'D+DB', 'D+B', 'DB+B', 'D+DB+B', 'DB+D+B']
CONFIG_LABELS = {
    'Input':   'Input (no proc.)',
    'D':       'D only',
    'DB':      'DB only',
    'B':       'B only',
    'D+DB':    'D -> DB',
    'D+B':     'D -> B',
    'DB+B':    'DB -> B',
    'D+DB+B':  'Full (D->DB->B)',
    'DB+D+B':  'Variant (DB->D->B)',
}
METRICS = ['lpips', 'psnr', 'ssim', 'niqe']
# sign: +1 = higher is better, -1 = lower is better
METRIC_SIGN = {'lpips': -1, 'psnr': +1, 'ssim': +1, 'niqe': -1}


# ---------------------------------------------------------------------------
# Proper NIQE (BasicSR/HVI-CIDNet implementation + official pristine params)
# ---------------------------------------------------------------------------

def load_niqe_context():
    spec = importlib.util.spec_from_file_location(
        'niqe_utils', os.path.join(HVI_DIR, 'loss', 'niqe_utils.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params = np.load(os.path.join(HVI_DIR, 'loss', 'niqe_pris_params.npz'))
    return mod, params['mu_pris_param'], params['cov_pris_param'], params['gaussian_window']


def compute_niqe(img_bgr, ctx):
    mod, mu, cov, win = ctx
    y = mod.to_y_channel(img_bgr).squeeze(-1)
    return float(mod.niqe(np.float32(y), mu, cov, win))


# ---------------------------------------------------------------------------
# Stage cache (compute each model pass exactly once)
# ---------------------------------------------------------------------------

def compute_stage_outputs(frames, args, device):
    """Compute all intermediates needed for the ablation matrix.

    Returns dict config -> np.ndarray [N,H,W,3] BGR uint8.
    """
    outputs = {'Input': frames.copy()}

    # D = FastDVDNet(frames)
    need_den = any(cfg in ('D', 'D+DB', 'D+B', 'D+DB+B', 'DB+D+B') for cfg in CONFIGS)
    denoised = None
    if need_den:
        print('\n[Stage D] FastDVDNet denoising ...')
        t0 = time.time()
        m = build_fastdvdnet(os.path.join(FASTDVDNET_DIR, 'model.pth'), device)
        denoised = stage_denoise(frames, m, args.noise_sigma, device)
        del m; torch.cuda.empty_cache()
        print(f'  done in {time.time()-t0:.1f}s')
        outputs['D'] = denoised

    # DB = CDVD-TSP(frames)  (deblur raw input)
    need_db = any(cfg in ('DB', 'DB+B', 'DB+D+B') for cfg in CONFIGS)
    deblur_input = None
    if need_db:
        print('\n[Stage DB] CDVD-TSP deblurring (on input) ...')
        t0 = time.time()
        m = build_cdvd_tsp(os.path.join(CDVD_TSP_DIR, 'pretrain_models',
                                        'CDVD_TSP_DVD_Convergent.pt'), device)
        deblur_input = stage_deblur(frames, m, device)
        del m; torch.cuda.empty_cache()
        print(f'  done in {time.time()-t0:.1f}s')
        outputs['DB'] = deblur_input

    # B = StableLLVE(frames)  (brighten raw input)
    if 'B' in CONFIGS:
        print('\n[Stage B] StableLLVE brightening (on input) ...')
        t0 = time.time()
        m = build_stablellve(os.path.join(STABLELLVE_DIR, 'checkpoint.pth'), device)
        outputs['B'] = stage_brighten(frames, m, device)
        del m; torch.cuda.empty_cache()
        print(f'  done in {time.time()-t0:.1f}s')

    # D -> DB
    if 'D+DB' in CONFIGS:
        print('\n[Stage D->DB] CDVD-TSP deblurring (on denoised) ...')
        t0 = time.time()
        m = build_cdvd_tsp(os.path.join(CDVD_TSP_DIR, 'pretrain_models',
                                        'CDVD_TSP_DVD_Convergent.pt'), device)
        deblur_denoised = stage_deblur(denoised, m, device)
        del m; torch.cuda.empty_cache()
        print(f'  done in {time.time()-t0:.1f}s')
        outputs['D+DB'] = deblur_denoised
    else:
        deblur_denoised = None

    # D -> B
    if 'D+B' in CONFIGS:
        print('\n[Stage D->B] StableLLVE brightening (on denoised) ...')
        t0 = time.time()
        m = build_stablellve(os.path.join(STABLELLVE_DIR, 'checkpoint.pth'), device)
        outputs['D+B'] = stage_brighten(denoised, m, device)
        del m; torch.cuda.empty_cache()
        print(f'  done in {time.time()-t0:.1f}s')

    # DB -> B
    if 'DB+B' in CONFIGS:
        print('\n[Stage DB->B] StableLLVE brightening (on deblurred input) ...')
        t0 = time.time()
        m = build_stablellve(os.path.join(STABLELLVE_DIR, 'checkpoint.pth'), device)
        outputs['DB+B'] = stage_brighten(deblur_input, m, device)
        del m; torch.cuda.empty_cache()
        print(f'  done in {time.time()-t0:.1f}s')

    # D -> DB -> B (full)
    if 'D+DB+B' in CONFIGS:
        print('\n[Stage D->DB->B] StableLLVE brightening (on denoised+deblurred) ...')
        t0 = time.time()
        m = build_stablellve(os.path.join(STABLELLVE_DIR, 'checkpoint.pth'), device)
        outputs['D+DB+B'] = stage_brighten(deblur_denoised, m, device)
        del m; torch.cuda.empty_cache()
        print(f'  done in {time.time()-t0:.1f}s')

    # DB -> D -> B (order variant)
    if 'DB+D+B' in CONFIGS:
        print('\n[Stage DB->D->B] FastDVDNet denoising (on deblurred input) ...')
        t0 = time.time()
        m = build_fastdvdnet(os.path.join(FASTDVDNET_DIR, 'model.pth'), device)
        denoised_deblur = stage_denoise(deblur_input, m, args.noise_sigma, device)
        del m; torch.cuda.empty_cache()
        print(f'  done in {time.time()-t0:.1f}s')
        print('[Stage DB->D->B] StableLLVE brightening ...')
        m = build_stablellve(os.path.join(STABLELLVE_DIR, 'checkpoint.pth'), device)
        outputs['DB+D+B'] = stage_brighten(denoised_deblur, m, device)
        del m; torch.cuda.empty_cache()

    return outputs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_configs(outputs, gt_frames, device):
    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)
    niqe_ctx = load_niqe_context()

    rows = []
    for cfg in CONFIGS:
        if cfg not in outputs:
            continue
        pred = outputs[cfg]
        for i in range(len(pred)):
            row = {
                'config': cfg,
                'frame': i,
                'lpips': compute_lpips(pred[i], gt_frames[i], lpips_model, device),
                'psnr': float(compute_psnr(pred[i], gt_frames[i])),
                'ssim': float(compute_ssim(pred[i], gt_frames[i])),
                'niqe': compute_niqe(pred[i], niqe_ctx),
            }
            rows.append(row)
            print(f"  [{cfg:>8s}] frame {i:3d}: LPIPS={row['lpips']:.4f} "
                  f"PSNR={row['psnr']:.2f} SSIM={row['ssim']:.4f} NIQE={row['niqe']:.3f}")
    return rows


def summarize(rows):
    summary = {}
    for cfg in CONFIGS:
        cfg_rows = [r for r in rows if r['config'] == cfg]
        if not cfg_rows:
            continue
        summary[cfg] = {}
        for m in METRICS:
            vals = [r[m] for r in cfg_rows]
            summary[cfg][m] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_bar_charts(summary, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfgs = [c for c in CONFIGS if c in summary]
    x = np.arange(len(cfgs))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    metric_info = [
        ('lpips', 'LPIPS (lower better)'),
        ('psnr', 'PSNR / dB (higher better)'),
        ('ssim', 'SSIM (higher better)'),
        ('niqe', 'NIQE (lower better)'),
    ]
    for ax, (m, title) in zip(axes.flat, metric_info):
        means = [summary[c][m]['mean'] for c in cfgs]
        stds = [summary[c][m]['std'] for c in cfgs]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color='steelblue', alpha=0.85)
        # Highlight full pipeline
        for b, c in zip(bars, cfgs):
            if c == 'D+DB+B':
                b.set_color('crimson')
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in cfgs], rotation=25, ha='right')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        for xi, v in zip(x, means):
            ax.text(xi, v, f'{v:.3f}' if m in ('lpips', 'ssim') else f'{v:.2f}',
                    ha='center', va='bottom' if m in ('psnr', 'ssim') else 'top', fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'ablation_bars.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[Figure] {path}')


def save_comparison_grid(frames_by_cfg, gt_frame, best_idx, out_dir):
    """2-row montage: row 1 = Input, D, DB, B, D+DB; row 2 = D+B, DB+B, Full, Variant, GT."""
    order = [('Input', 'Input'), ('D', 'D only'), ('DB', 'DB only'), ('B', 'B only'),
             ('D+DB', 'D -> DB'), ('D+B', 'D -> B'), ('DB+B', 'DB -> B'),
             ('D+DB+B', 'Full (D->DB->B)'), ('DB+D+B', 'Variant (DB->D->B)'), ('GT', 'Ground Truth')]

    max_h = 320
    imgs = {}
    for key, _ in order:
        img = gt_frame if key == 'GT' else frames_by_cfg[key][best_idx]
        h, w = img.shape[:2]
        scale = max_h / h
        imgs[key] = cv2.resize(img, (int(w * scale), max_h))

    tile_w = imgs['Input'].shape[1]
    tile_h = imgs['Input'].shape[0]
    pad, title_h = 6, 34
    cols = 5
    rows = 2
    canvas = np.ones(((tile_h + title_h + pad) * rows + pad, (tile_w + pad) * cols + pad, 3), np.uint8) * 255

    for idx, (key, label) in enumerate(order):
        r, c = divmod(idx, cols)
        y0 = pad + r * (tile_h + title_h + pad)
        x0 = pad + c * (tile_w + pad)
        color = (0, 0, 0)
        if key == 'D+DB+B':
            color = (0, 0, 200)
        cv2.putText(canvas, label, (x0, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        canvas[y0 + title_h:y0 + title_h + tile_h, x0:x0 + tile_w] = imgs[key]

    path = os.path.join(out_dir, f'ablation_grid_best_frame{best_idx:03d}.png')
    cv2.imwrite(path, canvas)
    print(f'[Figure] {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--output', default='outputs/pipeline_ablation')
    parser.add_argument('--num_frames', type=int, default=11)
    parser.add_argument('--noise_sigma', type=float, default=25)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--only_configs', type=str, default='',
                        help='comma-separated subset of configs to run (default: all)')
    args = parser.parse_args()
    args.only_configs = [s for s in args.only_configs.split(',') if s]

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    # Load frames (aligned by index; SDSD LL/GT filename offsets differ)
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
    H, W = frames.shape[1:3]
    if H % 4 or W % 4:
        nh, nw = H - H % 4, W - W % 4
        frames, gt_frames = frames[:, :nh, :nw], gt_frames[:, :nh, :nw]
    print(f'[Ablation] {n} frames @ {frames.shape[2]}x{frames.shape[1]}, '
          f'noise_sigma={args.noise_sigma}, scale={args.scale}')

    t_start = time.time()
    outputs = compute_stage_outputs(frames, args, device)
    print(f'\n[Ablation] Stage computation done in {(time.time()-t_start)/60:.1f} min')

    print('\n[Ablation] Evaluating metrics ...')
    rows = evaluate_configs(outputs, gt_frames, device)
    summary = summarize(rows)

    # Save frames per config
    for cfg, arr in outputs.items():
        save_frames(arr, os.path.join(args.output, 'frames', cfg.replace('+', '_')), filenames)

    # Save per-frame CSV
    csv_path = os.path.join(args.output, 'metrics_per_frame.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['config', 'frame', 'filename',
                                               'lpips', 'psnr', 'ssim', 'niqe'])
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, 'filename': os.path.basename(filenames[r['frame']])})
    print(f'[Ablation] {csv_path}')

    # Save summary CSV + JSON
    with open(os.path.join(args.output, 'metrics_summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'lpips_mean', 'lpips_std', 'psnr_mean', 'psnr_std',
                         'ssim_mean', 'ssim_std', 'niqe_mean', 'niqe_std'])
        for cfg in CONFIGS:
            if cfg not in summary:
                continue
            s = summary[cfg]
            writer.writerow([cfg,
                             s['lpips']['mean'], s['lpips']['std'],
                             s['psnr']['mean'], s['psnr']['std'],
                             s['ssim']['mean'], s['ssim']['std'],
                             s['niqe']['mean'], s['niqe']['std']])
    with open(os.path.join(args.output, 'metrics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[Ablation] summary saved')

    # Print summary table
    print('\n' + '=' * 86)
    print(f"{'Config':<18s} {'LPIPS':>14s} {'PSNR':>14s} {'SSIM':>14s} {'NIQE':>14s}")
    print('-' * 86)
    for cfg in CONFIGS:
        if cfg not in summary:
            continue
        s = summary[cfg]
        print(f"{CONFIG_LABELS[cfg]:<18s} "
              f"{s['lpips']['mean']:>7.4f}±{s['lpips']['std']:.3f} "
              f"{s['psnr']['mean']:>7.2f}±{s['psnr']['std']:.2f} "
              f"{s['ssim']['mean']:>7.4f}±{s['ssim']['std']:.3f} "
              f"{s['niqe']['mean']:>7.3f}±{s['niqe']['std']:.3f}")
    print('=' * 86)

    # Figures
    save_bar_charts(summary, args.output)
    if 'D+DB+B' in outputs:
        best_idx = min((r for r in rows if r['config'] == 'D+DB+B'),
                       key=lambda r: r['lpips'])['frame']
        save_comparison_grid(outputs, gt_frames[best_idx], best_idx, args.output)
        print(f'[Ablation] Best full-pipeline frame (LPIPS): #{best_idx} ({os.path.basename(filenames[best_idx])})')

    print(f'\n[Ablation] Done. Results in: {args.output}')


if __name__ == '__main__':
    main()
