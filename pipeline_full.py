#!/usr/bin/env python3
"""
Full serial pipeline: FastDVDNet -> CDVD-TSP -> StableLLVE
with LPIPS/PSNR/SSIM metrics and best-frame visualization.

Usage:
  python pipeline_full.py --input ... --gt_dir ... --output ...
"""

import os
import sys
import argparse
import glob
import importlib.util
import numpy as np
import cv2
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, 'reference_repos')

FASTDVDNET_DIR = os.path.join(REPO_DIR, 'fastdvdnet')
CDVD_TSP_DIR = os.path.join(REPO_DIR, 'CDVD-TSP')
STABLELLVE_DIR = os.path.join(REPO_DIR, 'StableLLVE')

# Import StableLLVE UNet via file path to avoid 'model' name conflict
_unet_spec = importlib.util.spec_from_file_location('_unet_module', os.path.join(STABLELLVE_DIR, 'model.py'))
_unet_module = importlib.util.module_from_spec(_unet_spec)
_unet_spec.loader.exec_module(_unet_module)


# ============================================================================
# Build models
# ============================================================================

def build_fastdvdnet(ckpt_path, device):
    _orig_sys_path = list(sys.path)
    sys.path.insert(0, FASTDVDNET_DIR)
    for mod in list(sys.modules.keys()):
        if mod.startswith('utils') or mod.startswith('models') or mod.startswith('fastdvdnet'):
            del sys.modules[mod]
    try:
        from models import FastDVDnet
        from utils import remove_dataparallel_wrapper
    finally:
        sys.path[:] = _orig_sys_path
    model = FastDVDnet(num_input_frames=5)
    state_dict = torch.load(ckpt_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = remove_dataparallel_wrapper(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_cdvd_tsp(ckpt_path, device):
    _orig_sys_path = list(sys.path)
    # Remove potentially conflicting cached modules
    sys.path.insert(0, os.path.join(CDVD_TSP_DIR, 'code'))
    for mod in list(sys.modules.keys()):
        if mod in ('utils', 'model', 'model.blocks', 'model.cdvd_tsp',
                    'model.flow_pwc', 'model.recons_video', 'model.correlation',
                    'utils.utils'):
            del sys.modules[mod]
    try:
        from model.cdvd_tsp import CDVD_TSP
    finally:
        sys.path[:] = _orig_sys_path
    model = CDVD_TSP(in_channels=3, n_sequence=5, out_channels=3, n_resblock=3, n_feat=32,
                     is_mask_filter=True, device=device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def build_stablellve(ckpt_path, device):
    model = _unet_module.UNet(n_channels=3, bilinear=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ============================================================================
# Load / Save
# ============================================================================

def load_frames(input_dir, max_frames=None):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files = sorted(files, key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))) if any(
        c.isdigit() for c in os.path.basename(f)) else f)
    if max_frames:
        files = files[:max_frames]
    frames = [cv2.imread(f) for f in files]
    return np.stack(frames, axis=0), files


def save_frames(frames_np, output_dir, basenames=None):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames_np):
        name = os.path.basename(basenames[i]) if basenames else f'{i:06d}.png'
        cv2.imwrite(os.path.join(output_dir, name), frame)


# ============================================================================
# Stage functions
# ============================================================================

def stage_denoise(frames_bgr, model, noise_sigma, device):
    _orig_sys_path = list(sys.path)
    sys.path.insert(0, FASTDVDNET_DIR)
    try:
        from fastdvdnet import denoise_seq_fastdvdnet
    finally:
        sys.path[:] = _orig_sys_path
    frames_rgb = frames_bgr[:, :, :, ::-1]
    seq = torch.from_numpy(frames_rgb.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    noise_std = torch.FloatTensor([noise_sigma / 255.0]).to(device)
    with torch.no_grad():
        denoised_t = denoise_seq_fastdvdnet(seq=seq, noise_std=noise_std, temp_psz=5, model_temporal=model)
    denoised_rgb = (denoised_t.detach().permute(0, 2, 3, 1).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return denoised_rgb[:, :, :, ::-1]


def stage_deblur(frames_bgr, model, device):
    n_seq = 5; half = n_seq // 2; N = len(frames_bgr)
    frames_rgb = frames_bgr[:, :, :, ::-1]
    padded = np.concatenate([frames_rgb[half - 1::-1], frames_rgb, frames_rgb[:-(half + 1):-1]], axis=0)
    results = np.zeros_like(frames_rgb)
    for i in range(N):
        seq = padded[i:i + n_seq]
        in_tensor = torch.from_numpy(seq.astype(np.float32) / 255.0).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, _, out, _ = model(in_tensor)
        results[i] = (out[0].detach().permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return results[:, :, :, ::-1]


def stage_brighten(frames_bgr, model, device):
    results = []
    for i in range(len(frames_bgr)):
        frame = frames_bgr[i]
        in_tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(in_tensor)
        results.append((out[0].detach().permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
    return np.stack(results, axis=0)


# ============================================================================
# Metrics
# ============================================================================

def compute_lpips(img1_bgr, img2_bgr, lpips_model, device):
    import lpips
    img1_rgb = torch.from_numpy(img1_bgr[:, :, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    img2_rgb = torch.from_numpy(img2_bgr[:, :, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return lpips_model(img1_rgb, img2_rgb).item()


def compute_psnr(img1_bgr, img2_bgr):
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    return psnr_func(img1_bgr, img2_bgr, data_range=255)


def compute_ssim(img1_bgr, img2_bgr):
    from skimage.metrics import structural_similarity as ssim_func
    return ssim_func(img1_bgr, img2_bgr, channel_axis=2, data_range=255)


def compute_niqe(img_bgr):
    """Approximate NIQE using MSCN + GGD kurtosis. Lower = better."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    kernel = cv2.getGaussianKernel(7, 7 / 6)
    window = kernel @ kernel.T
    mu = cv2.filter2D(gray, -1, window, borderType=cv2.BORDER_REPLICATE)
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(cv2.filter2D(gray * gray, -1, window, borderType=cv2.BORDER_REPLICATE) - mu_sq))
    mscn = (gray - mu) / (sigma + 1.0)
    from scipy.stats import kurtosis
    return abs(float(kurtosis(mscn.flatten(), fisher=True)))


# ============================================================================
# Visualization
# ============================================================================

def make_comparison_figure(inputs, output_path, metrics, frame_idx):
    """Create a 1x5 row comparison figure.
    inputs = [input_bgr, denoised_bgr, deblurred_bgr, brightened_bgr, gt_bgr]
    """
    titles = [
        'Low-Light Input',
        'Stage 1: Denoised\n(FastDVDNet)',
        'Stage 2: Deblurred\n(CDVD-TSP)',
        'Stage 3: Brightened\n(StableLLVE)',
        'Ground Truth'
    ]
    h, w = inputs[0].shape[:2]
    max_h = 400
    scale = min(1.0, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = []
    for img in inputs:
        if scale < 1.0:
            resized.append(cv2.resize(img, (new_w, new_h)))
        else:
            resized.append(img.copy())

    pad = 8
    title_h = 60
    total_w = new_w * 5 + pad * 6
    canvas_h = new_h + title_h + 30
    canvas = np.ones((canvas_h, total_w, 3), dtype=np.uint8) * 255

    for idx, (img, title) in enumerate(zip(resized, titles)):
        x_start = pad + idx * (new_w + pad)
        y = 5
        for line in title.split('\n'):
            cv2.putText(canvas, line, (x_start, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
            y += 20
        canvas[title_h:title_h + new_h, x_start:x_start + new_w] = img

    metric_text = (
        f"Frame #{frame_idx} | "
        f"PSNR: {metrics.get('psnr', 0):.2f}dB | "
        f"SSIM: {metrics.get('ssim', 0):.4f} | "
        f"LPIPS: {metrics.get('lpips', 0):.4f} | "
        f"NIQE: {metrics.get('niqe_in', 0):.2f} -> {metrics.get('niqe_out', 0):.2f}"
    )
    cv2.putText(canvas, metric_text, (pad, canvas_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(output_path, canvas)
    print(f"[Visualize] Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--output', default='outputs/pipeline_result')
    parser.add_argument('--num_frames', type=int, default=20)
    parser.add_argument('--noise_sigma', type=float, default=25)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip_denoise', action='store_true')
    parser.add_argument('--skip_deblur', action='store_true')
    parser.add_argument('--skip_brighten', action='store_true')
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    # -------------------------------------------------------------------
    # Load frames
    # -------------------------------------------------------------------
    frames, filenames = load_frames(args.input, args.num_frames)
    gt_frames, _ = load_frames(args.gt_dir, args.num_frames)
    min_len = min(len(frames), len(gt_frames))
    frames, gt_frames, filenames = frames[:min_len], gt_frames[:min_len], filenames[:min_len]
    print(f"[Pipeline] {min_len} frames, {frames.shape[2]}x{frames.shape[1]}")

    # Scale down
    if args.scale < 1.0:
        N, H, W, C = frames.shape
        new_h, new_w = int(H * args.scale), int(W * args.scale)
        new_h, new_w = new_h - new_h % 4, new_w - new_w % 4
        frames = np.stack([cv2.resize(f, (new_w, new_h)) for f in frames])
        gt_frames = np.stack([cv2.resize(g, (new_w, new_h)) for g in gt_frames])

    # Crop to multiple of 4
    N, H, W, C = frames.shape
    new_h, new_w = H - H % 4, W - W % 4
    if new_h != H or new_w != W:
        frames = frames[:, :new_h, :new_w, :]
        gt_frames = gt_frames[:, :new_h, :new_w, :]

    current = frames.copy()
    denoised = deblurred = brightened = None

    # -------------------------------------------------------------------
    # Stage 1: Denoise
    # -------------------------------------------------------------------
    if not args.skip_denoise:
        print(f"\n{'='*60}")
        print(f"  Stage 1/3: FastDVDNet Denoising (sigma={args.noise_sigma})")
        print(f"{'='*60}")
        ckpt = os.path.join(FASTDVDNET_DIR, 'model.pth')
        m = build_fastdvdnet(ckpt, device)
        denoised = stage_denoise(current, m, args.noise_sigma, device)
        del m; torch.cuda.empty_cache()
        current = denoised

    # -------------------------------------------------------------------
    # Stage 2: Deblur
    # -------------------------------------------------------------------
    if not args.skip_deblur:
        print(f"\n{'='*60}")
        print(f"  Stage 2/3: CDVD-TSP Deblurring")
        print(f"{'='*60}")
        ckpt = os.path.join(CDVD_TSP_DIR, 'pretrain_models', 'CDVD_TSP_DVD_Convergent.pt')
        m = build_cdvd_tsp(ckpt, device)
        deblurred = stage_deblur(current, m, device)
        del m; torch.cuda.empty_cache()
        current = deblurred

    # -------------------------------------------------------------------
    # Stage 3: Brighten
    # -------------------------------------------------------------------
    if not args.skip_brighten:
        print(f"\n{'='*60}")
        print(f"  Stage 3/3: StableLLVE Illumination Enhancement")
        print(f"{'='*60}")
        ckpt = os.path.join(STABLELLVE_DIR, 'checkpoint.pth')
        m = build_stablellve(ckpt, device)
        brightened = stage_brighten(current, m, device)
        del m; torch.cuda.empty_cache()
        current = brightened

    # -------------------------------------------------------------------
    # Fill missing stages with best available
    # -------------------------------------------------------------------
    if denoised is None:
        denoised = frames.copy()
    if deblurred is None:
        deblurred = denoised.copy()
    if brightened is None:
        brightened = deblurred.copy()

    # -------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Computing LPIPS / PSNR / SSIM / NIQE")
    print(f"{'='*60}")

    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)

    all_metrics = []
    best_lpips, best_psnr = float('inf'), -float('inf')
    best_idx_lpips, best_idx_psnr = 0, 0

    for i in range(len(frames)):
        lpips_v = compute_lpips(brightened[i], gt_frames[i], lpips_model, device)
        psnr_v = compute_psnr(brightened[i], gt_frames[i])
        ssim_v = compute_ssim(brightened[i], gt_frames[i])
        niqe_in = compute_niqe(frames[i])
        niqe_out = compute_niqe(brightened[i])

        all_metrics.append({
            'idx': i, 'lpips': lpips_v, 'psnr': psnr_v,
            'ssim': ssim_v, 'niqe_in': niqe_in, 'niqe_out': niqe_out
        })
        if lpips_v < best_lpips:
            best_lpips, best_idx_lpips = lpips_v, i
        if psnr_v > best_psnr:
            best_psnr, best_idx_psnr = psnr_v, i

    # Summary
    for k in ['lpips', 'psnr', 'ssim', 'niqe_in', 'niqe_out']:
        vals = [m[k] for m in all_metrics]
        print(f"  {k:>10s}: avg={np.mean(vals):.4f}  min={np.min(vals):.4f}  max={np.max(vals):.4f}")
    print(f"  Best LPIPS: frame #{best_idx_lpips} = {best_lpips:.4f}")
    print(f"  Best PSNR:  frame #{best_idx_psnr} = {best_psnr:.2f} dB")

    # -------------------------------------------------------------------
    # Save frames
    # -------------------------------------------------------------------
    save_frames(denoised, os.path.join(args.output, '01_denoised'), filenames)
    save_frames(deblurred, os.path.join(args.output, '02_deblurred'), filenames)
    save_frames(brightened, os.path.join(args.output, '03_brightened'), filenames)

    # -------------------------------------------------------------------
    # Comparison figures
    # -------------------------------------------------------------------
    best_idx = best_idx_lpips
    make_comparison_figure(
        [frames[best_idx], denoised[best_idx], deblurred[best_idx],
         brightened[best_idx], gt_frames[best_idx]],
        os.path.join(args.output, f'comparison_LPIPS_best_frame{best_idx:03d}.png'),
        all_metrics[best_idx], best_idx
    )
    if best_idx_psnr != best_idx_lpips:
        make_comparison_figure(
            [frames[best_idx_psnr], denoised[best_idx_psnr], deblurred[best_idx_psnr],
             brightened[best_idx_psnr], gt_frames[best_idx_psnr]],
            os.path.join(args.output, f'comparison_PSNR_best_frame{best_idx_psnr:03d}.png'),
            all_metrics[best_idx_psnr], best_idx_psnr
        )

    # -------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------
    with open(os.path.join(args.output, 'metrics.csv'), 'w') as f:
        f.write('frame,filename,lpips,psnr,ssim,niqe_in,niqe_out\n')
        for m in all_metrics:
            fn = os.path.basename(filenames[m['idx']])
            f.write(f"{m['idx']},{fn},{m['lpips']:.6f},{m['psnr']:.4f},{m['ssim']:.6f},{m['niqe_in']:.4f},{m['niqe_out']:.4f}\n")

    print(f"\n[Pipeline] Done. Results in: {args.output}")


if __name__ == '__main__':
    main()
