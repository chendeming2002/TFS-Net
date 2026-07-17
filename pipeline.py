#!/usr/bin/env python3
"""
Serial Video Enhancement Pipeline: FastDVDNet -> CDVD-TSP -> StableLLVE

Corresponds to TSD-Net's three degradation types:
  Stage 1: FastDVDNet  -> Noise Denoising       (TIP 2020, CVPR 2020 oral)
  Stage 2: CDVD-TSP    -> Motion Deblurring       (CVPR 2020)
  Stage 3: StableLLVE  -> Illumination Enhancement (CVPR 2021)

Usage:
  # Full pipeline (recommended order: denoise -> deblur -> brighten)
  python pipeline.py --input /path/to/frames/ --output /path/to/results/ \
      --noise_sigma 20 --gpu 0

  # Single stage only
  python pipeline.py --input /path/to/frames/ --output /path/to/results/ \
      --skip_denoise --skip_deblur

  # Video file input
  python pipeline.py --input video.mp4 --output /path/to/results/ --video_input

Dependencies:
  - FastDVDNet: model.pth (already in reference_repos/fastdvdnet/)
  - CDVD-TSP:   needs pretrained weights (see --deblur_ckpt)
  - StableLLVE:  checkpoint.pth (already in reference_repos/StableLLVE/)
"""

import os
import sys
import argparse
import glob
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, 'reference_repos')

FASTDVDNET_DIR = os.path.join(REPO_DIR, 'fastdvdnet')
CDVD_TSP_DIR = os.path.join(REPO_DIR, 'CDVD-TSP')
STABLELLVE_DIR = os.path.join(REPO_DIR, 'StableLLVE')

sys.path.insert(0, os.path.join(CDVD_TSP_DIR, 'code'))
sys.path.insert(0, FASTDVDNET_DIR)
sys.path.insert(0, STABLELLVE_DIR)


# ============================================================================
# Utils
# ============================================================================

def load_image_frames(input_path, is_video=False, max_frames=None):
    """Load frames from either a directory of images or a video file.

    Returns:
        frames_np: np.ndarray [N, H, W, C] uint8 BGR
        fps: float (only meaningful for video input)
    """
    if is_video:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
        if len(frames) == 0:
            raise ValueError(f"No frames read from video: {input_path}")
        print(f"[Pipeline] Loaded {len(frames)} frames from video @ {fps:.2f} fps")
        return np.stack(frames, axis=0), fps
    else:
        # Image sequence directory
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(input_path, ext)))
        files = sorted(files, key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))) if any(
            c.isdigit() for c in os.path.basename(f)) else f)
        if not files:
            raise FileNotFoundError(f"No image files found in {input_path}")
        if max_frames:
            files = files[:max_frames]
        frames = [cv2.imread(f) for f in files]
        print(f"[Pipeline] Loaded {len(frames)} frames from directory: {input_path}")
        return np.stack(frames, axis=0), 0.0


def save_image_frames(frames_np, output_dir, start_idx=0):
    """Save frames as numbered PNG images."""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames_np):
        out_path = os.path.join(output_dir, f'{start_idx + i:06d}.png')
        cv2.imwrite(out_path, frame)
    print(f"[Pipeline] Saved {len(frames_np)} frames to {output_dir}")


def ensure_size_multiple(frames_np, divisor=4):
    """Crop spatial dims to be multiples of `divisor` (required by both denoising/deblurring)."""
    N, H, W, C = frames_np.shape
    new_h = H - H % divisor
    new_w = W - W % divisor
    if new_h != H or new_w != W:
        print(f"[Pipeline] Cropping frames from {H}x{W} to {new_h}x{new_w} (multiple of {divisor})")
        frames_np = frames_np[:, :new_h, :new_w, :]
    return frames_np


# ============================================================================
# Stage 1: FastDVDNet (Denoising)
# ============================================================================

def build_fastdvdnet(ckpt_path, device):
    from models import FastDVDnet
    from utils import remove_dataparallel_wrapper

    model = FastDVDnet(num_input_frames=5)
    state_dict = torch.load(ckpt_path, map_location=device)
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = remove_dataparallel_wrapper(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def stage_denoise(frames_bgr, model, noise_sigma, device):
    """FastDVDNet: denoise all frames.

    Args:
        frames_bgr: np.ndarray [N, H, W, 3] uint8 BGR
    Returns:
        denoised: np.ndarray [N, H, W, 3] uint8 BGR
    """
    from fastdvdnet import denoise_seq_fastdvdnet

    # FastDVDNet internally converts BGR->RGB in open_image, but since we feed
    # pre-loaded tensors we do: BGR -> RGB for the model, then RGB -> BGR back.
    frames_rgb = frames_bgr[:, :, :, ::-1]  # BGR -> RGB

    # Normalize to [0, 1], shape [N, C, H, W]
    seq = torch.from_numpy(frames_rgb.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    noise_std = torch.FloatTensor([noise_sigma / 255.0]).to(device)

    denoised_t = denoise_seq_fastdvdnet(seq=seq, noise_std=noise_std,
                                        temp_psz=5, model_temporal=model)
    # Back to numpy uint8 BGR
    denoised_rgb = (denoised_t.detach().permute(0, 2, 3, 1).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    denoised_bgr = denoised_rgb[:, :, :, ::-1]  # RGB -> BGR
    return denoised_bgr


# ============================================================================
# Stage 2: CDVD-TSP (Deblurring)
# ============================================================================

def build_cdvd_tsp(ckpt_path, device):
    try:
        from model.cdvd_tsp import CDVD_TSP
    except ImportError as e:
        raise ImportError(
            f"Failed to import CDVD-TSP: {e}\n"
            "CDVD-TSP requires cupy and CUDA toolkit.\n"
            "Install: pip install cupy-cuda11x (adjust for your CUDA version)\n"
            "Or skip deblurring with: --skip_deblur"
        )

    model = CDVD_TSP(in_channels=3, n_sequence=5, out_channels=3, n_resblock=3, n_feat=32,
                     is_mask_filter=True, device=device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def stage_deblur(frames_bgr, model, device):
    """CDVD-TSP: deblur video.

    Args:
        frames_bgr: np.ndarray [N, H, W, 3] uint8 BGR
    Returns:
        deblurred: np.ndarray [N, H, W, 3] uint8 BGR
    """
    n_seq = 5
    half = n_seq // 2
    N = len(frames_bgr)

    # CDVD-TSP uses RGB
    frames_rgb = frames_bgr[:, :, :, ::-1]

    # Handle border condition: reflect border frames
    padded = np.concatenate([
        frames_rgb[half - 1::-1],   # left reflect
        frames_rgb,
        frames_rgb[:-(half + 1):-1]  # right reflect
    ], axis=0)

    results = np.zeros_like(frames_rgb)

    for i in range(N):
        seq = padded[i:i + n_seq]  # [5, H, W, 3]
        # Normalize to [0, 1], shape [1, 5, 3, H, W]
        in_tensor = torch.from_numpy(seq.astype(np.float32) / 255.0)
        in_tensor = in_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, _, out, _ = model(in_tensor)

        # out: [1, 3, H, W]
        result = (out[0].detach().permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        results[i] = result

    # RGB -> BGR
    deblurred_bgr = results[:, :, :, ::-1]
    return deblurred_bgr


# ============================================================================
# Stage 3: StableLLVE (Low-Light Enhancement)
# ============================================================================

def build_stablellve(ckpt_path, device):
    from model import UNet
    model = UNet(n_channels=3, bilinear=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def stage_brighten(frames_bgr, model, device):
    """StableLLVE: enhance low-light frames one by one.

    Args:
        frames_bgr: np.ndarray [N, H, W, 3] uint8 BGR
    Returns:
        enhanced: np.ndarray [N, H, W, 3] uint8 BGR
    """
    results = []
    for i in range(len(frames_bgr)):
        frame = frames_bgr[i]  # [H, W, 3] uint8 BGR

        # Normalize to [0, 1], shape [1, 3, H, W]
        in_tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(in_tensor)

        # out: [1, 3, H, W]
        result = (out[0].detach().permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        results.append(result)

    return np.stack(results, axis=0)


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Serial Video Enhancement: FastDVDNet -> CDVD-TSP -> StableLLVE')
    parser.add_argument('--input', required=True,
                        help='Path to input directory of frames, or video file (with --video_input)')
    parser.add_argument('--output', required=True,
                        help='Path to output directory for enhanced frames')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id (-1 for CPU)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max number of frames to process (for quick testing)')
    parser.add_argument('--video_input', action='store_true',
                        help='Treat --input as a video file instead of image directory')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save intermediate outputs (denoised, deblurred)')

    # Stage control
    parser.add_argument('--skip_denoise', action='store_true',
                        help='Skip Stage 1 (FastDVDNet denoising)')
    parser.add_argument('--skip_deblur', action='store_true',
                        help='Skip Stage 2 (CDVD-TSP deblurring)')
    parser.add_argument('--skip_brighten', action='store_true',
                        help='Skip Stage 3 (StableLLVE illumination enhancement)')

    # Stage 1: Denoise parameters
    parser.add_argument('--noise_sigma', type=float, default=20,
                        help='Estimated noise standard deviation [0-255] for FastDVDNet')
    parser.add_argument('--denoise_ckpt', type=str,
                        default=os.path.join(FASTDVDNET_DIR, 'model.pth'),
                        help='Path to FastDVDNet checkpoint')

    # Stage 2: Deblur parameters
    parser.add_argument('--deblur_ckpt', type=str,
                        default=os.path.join(CDVD_TSP_DIR, 'pretrain_models', 'CDVD_TSP_DVD_Convergent.pt'),
                        help='Path to CDVD-TSP checkpoint')

    # Stage 3: Brighten parameters
    parser.add_argument('--brighten_ckpt', type=str,
                        default=os.path.join(STABLELLVE_DIR, 'checkpoint.pth'),
                        help='Path to StableLLVE checkpoint')

    args = parser.parse_args()

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
        if args.gpu >= 0:
            print("[Pipeline] CUDA not available, falling back to CPU")
        # CDVD-TSP requires CUDA (cupy correlation kernel)
        if not args.skip_deblur:
            print("[Pipeline] WARNING: CDVD-TSP requires CUDA/cupy. Use --skip_deblur or run on GPU.")

    # -----------------------------------------------------------------------
    # Load frames
    # -----------------------------------------------------------------------
    frames, _ = load_image_frames(args.input, is_video=args.video_input,
                                  max_frames=args.max_frames)
    print(f"[Pipeline] Input: {frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]}")

    # Crop to multiples of 4 (required by both FastDVDNet and CDVD-TSP)
    frames = ensure_size_multiple(frames, divisor=4)
    current = frames

    # -----------------------------------------------------------------------
    # Stage 1: Denoise
    # -----------------------------------------------------------------------
    if not args.skip_denoise:
        if not os.path.exists(args.denoise_ckpt):
            raise FileNotFoundError(
                f"FastDVDNet checkpoint not found: {args.denoise_ckpt}")

        print(f"\n{'='*60}\n  Stage 1/3: FastDVDNet Denoising (sigma={args.noise_sigma})\n{'='*60}")
        denoise_model = build_fastdvdnet(args.denoise_ckpt, device)
        current = stage_denoise(current, denoise_model, args.noise_sigma, device)
        print(f"[Pipeline] Denoised. Output shape: {current.shape}")

        if args.save_intermediate:
            out_dir = os.path.join(args.output, 'intermediate', '01_denoised')
            save_image_frames(current, out_dir)

        del denoise_model
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Stage 2: Deblur
    # -----------------------------------------------------------------------
    if not args.skip_deblur:
        if not os.path.exists(args.deblur_ckpt):
            print(f"[Pipeline] CDVD-TSP checkpoint not found: {args.deblur_ckpt}")
            print(f"[Pipeline] Please download from: "
                  f"https://drive.google.com/drive/folders/1lw_1jITafEQ9DvMys_S6aYwtNApYKWsz")
            print(f"[Pipeline] Place it at: {args.deblur_ckpt}")
            print(f"[Pipeline] Skipping deblur stage.")
        else:
            try:
                print(f"\n{'='*60}\n  Stage 2/3: CDVD-TSP Deblurring\n{'='*60}")
                deblur_model = build_cdvd_tsp(args.deblur_ckpt, device)
                current = stage_deblur(current, deblur_model, device)
                print(f"[Pipeline] Deblurred. Output shape: {current.shape}")

                if args.save_intermediate:
                    out_dir = os.path.join(args.output, 'intermediate', '02_deblurred')
                    save_image_frames(current, out_dir)

                del deblur_model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[Pipeline] CDVD-TSP failed: {e}")
                print("[Pipeline] Skipping deblur stage.")
                import traceback
                traceback.print_exc()

    # -----------------------------------------------------------------------
    # Stage 3: Brighten
    # -----------------------------------------------------------------------
    if not args.skip_brighten:
        if not os.path.exists(args.brighten_ckpt):
            raise FileNotFoundError(
                f"StableLLVE checkpoint not found: {args.brighten_ckpt}")

        print(f"\n{'='*60}\n  Stage 3/3: StableLLVE Illumination Enhancement\n{'='*60}")
        brighten_model = build_stablellve(args.brighten_ckpt, device)
        current = stage_brighten(current, brighten_model, device)
        print(f"[Pipeline] Brightened. Output shape: {current.shape}")

        del brighten_model
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Save final output
    # -----------------------------------------------------------------------
    final_dir = os.path.join(args.output, 'final')
    save_image_frames(current, final_dir)
    print(f"\n[Pipeline] Done. Final output saved to: {final_dir}")


if __name__ == '__main__':
    main()
