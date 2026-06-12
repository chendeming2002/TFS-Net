# MINS-Net

PyTorch implementation scaffold for the first runnable version of MINS-Net on SDSD.

## Features
- SDSD-first video low-light enhancement training and inference
- Lightweight pyramid encoder
- MINS / ISPN / MSPN / Final Reconstruction pipeline
- Sliding-window center-frame supervision
- Pixel + SSIM + perceptual + prior smoothness losses

## Quick Start
1. Install dependencies from `requirements.txt`.
2. Edit `configs/sdsd_stage1.yaml` if your paths differ.
3. Train:
   `python train.py --config configs/sdsd_stage1.yaml`
4. Inference:
   `python infer.py --config configs/sdsd_stage1.yaml --checkpoint path/to/best.pth --input_root F:/DatasetDL/SDSD/test/low-light --output_root outputs/infer`

## Notes
- `loss.perceptual_pretrained` defaults to `false` to avoid offline weight download failures.
- Validation uses `SDSD/test/low-light` and aligned `GT`.
- DID is not enabled in this first version, but the project layout leaves room for extension.
