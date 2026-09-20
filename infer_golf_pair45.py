import torch
import yaml
import os
import numpy as np
from pathlib import Path
from PIL import Image
from models.golf import GolfNet
from tqdm import tqdm
import torchvision.transforms.functional as TF

def load_model(config_path, ckpt_path, device='cuda'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = {k:v for k,v in cfg['model'].items() if k != 'type'}
    model = GolfNet(**model_cfg).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded from {ckpt_path}, epoch {ckpt.get('epoch', '?')}")
    return model

def process_video(model, input_dir, output_dir, max_frames=30, window=5, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    
    frames = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])[:max_frames]
    print(f"Processing {len(frames)} frames from {input_dir}")
    
    # Load all frames
    imgs = []
    for fname in frames:
        img = Image.open(os.path.join(input_dir, fname)).convert('RGB')
        img = TF.to_tensor(img).unsqueeze(0).to(device)  # 1,3,H,W
        imgs.append(img)
    
    # Pad for window
    pad = window // 2
    imgs_padded = [imgs[0]] * pad + imgs + [imgs[-1]] * pad
    
    with torch.no_grad():
        for i in tqdm(range(len(frames))):
            # Get window
            window_imgs = imgs_padded[i:i+window]
            x = torch.stack(window_imgs, dim=1).squeeze(2)  # 1,T,C,H,W
            
            # Inference
            out = model(x)
            
            # Save
            out_img = out['res_t'].squeeze(0).clamp(0, 1).cpu()
            out_pil = TF.to_pil_image(out_img)
            out_pil.save(os.path.join(output_dir, frames[i]))

if __name__ == '__main__':
    model = load_model('configs/golf_r2.yaml', 'outputs/golf_r2/best.pth')
    
    input_dir = '/home/a1005/yzy/dataset/SDSD/test/low-light/pair45'
    output_dir = 'outputs/golf_r2/pair45_infer'
    
    process_video(model, input_dir, output_dir, max_frames=30, window=5)
    print(f"\nDone! Results saved to {output_dir}")
