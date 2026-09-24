#!/usr/bin/env python3
"""Golf-R4 推理脚本（pair45 全分辨率输出）

用法:
    python infer_golf_r4.py --ckpt outputs/golf_r4/best.pth --output outputs/golf_r4_inference/pair45
"""
import torch
import yaml
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from models.golf_r4 import GolfNet_R4
from utils.inference import tiled_forward
from tqdm import tqdm
import torchvision.transforms.functional as TF


def load_model(config_path, ckpt_path, device='cuda'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = {k: v for k, v in cfg['model'].items() if k != 'type'}
    model = GolfNet_R4(**model_cfg).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded from {ckpt_path}, epoch {ckpt.get('epoch', '?')}")
    return model, cfg


def process_video(model, input_dir, output_dir, max_frames=131, window=5, 
                 tile_size=256, tile_overlap=32, use_amp=True, amp_dtype=torch.float16, 
                 device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    
    frames = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])[:max_frames]
    print(f"Processing {len(frames)} frames from {input_dir}")
    print(f"  tile_size={tile_size}, overlap={tile_overlap}, amp={use_amp}, dtype={amp_dtype}")
    
    # Load all frames
    imgs = []
    for fname in frames:
        img = Image.open(os.path.join(input_dir, fname)).convert('RGB')
        img = TF.to_tensor(img).unsqueeze(0).to(device)  # 1,3,H,W
        imgs.append(img)
    
    print(f"  Frame resolution: {imgs[0].shape[2:]} (HxW)")
    
    # Pad for window
    pad = window // 2
    imgs_padded = [imgs[0]] * pad + imgs + [imgs[-1]] * pad
    
    with torch.no_grad():
        for i in tqdm(range(len(frames)), desc="Inference"):
            # Get window
            window_imgs = imgs_padded[i:i+window]
            clip = torch.cat(window_imgs, dim=0).unsqueeze(0)  # 1,T,C,H,W
            
            # tiled_forward（与训练验证相同）
            out = tiled_forward(
                model=model,
                clip=clip,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            
            # Save
            out_img = out.squeeze(0).clamp(0, 1).cpu()
            out_pil = TF.to_pil_image(out_img)
            out_pil.save(os.path.join(output_dir, frames[i]))
    
    print(f"\nDone! {len(frames)} frames saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='outputs/golf_r4/best.pth', help='checkpoint path')
    parser.add_argument('--config', default='configs/golf_r4.yaml', help='config path')
    parser.add_argument('--input', default='/home/a1005/yzy/dataset/SDSD/test/low-light/pair45', 
                       help='input directory')
    parser.add_argument('--output', default='outputs/golf_r4_inference/pair45', help='output directory')
    parser.add_argument('--max-frames', type=int, default=131, help='max frames to process')
    parser.add_argument('--tile-size', type=int, default=256, help='tile size for inference')
    parser.add_argument('--tile-overlap', type=int, default=32, help='tile overlap')
    parser.add_argument('--no-amp', action='store_true', help='disable AMP')
    parser.add_argument('--device', default='cuda', help='device')
    args = parser.parse_args()
    
    model, cfg = load_model(args.config, args.ckpt, device=args.device)
    
    # 从 config 读取默认参数
    tile_size = args.tile_size if args.tile_size != 256 else cfg.get('eval', {}).get('tile_size', 256)
    tile_overlap = args.tile_overlap if args.tile_overlap != 32 else cfg.get('eval', {}).get('tile_overlap', 32)
    use_amp = not args.no_amp and cfg.get('eval', {}).get('amp', True)
    amp_dtype_str = cfg.get('train', {}).get('amp_dtype', 'fp16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bf16' else torch.float16
    
    process_video(
        model=model,
        input_dir=args.input,
        output_dir=args.output,
        max_frames=args.max_frames,
        window=cfg['model']['num_frames'],
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        device=args.device,
    )
