#!/usr/bin/env python3
"""独立评估脚本: 加载 checkpoint 计算 val PSNR/SSIM"""
import argparse, torch
from torch.utils.data import DataLoader
from datasets import SDSDDataset
from utils.metrics import tensor_psnr, tensor_ssim

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--model', choices=['v7', 'v7r'], required=True)
args = p.parse_args()

if args.model == 'v7':
    from models.golf_v7 import GolfNet_v7 as M
else:
    from models.golf_v7r import GolfNet_v7r as M

ds = SDSDDataset(input_root='/home/a1005/yzy/dataset/SDSD/test/low-light',
                 target_root='/home/a1005/yzy/dataset/SDSD/test/GT',
                 window_size=5, mode='val', crop_size=None)
loader = DataLoader(ds, batch_size=1, shuffle=False)

model = M().cuda().eval()
ck = torch.load(args.ckpt, map_location='cuda', weights_only=False)
model.load_state_dict(ck['model_state_dict'])
print(f'Loaded {args.model} epoch {ck.get("epoch")}', flush=True)

ps, ss = [], []
with torch.no_grad():
    for i, (lq, gt, meta) in enumerate(loader):
        pred = model(lq.cuda())['final']
        ps.append(tensor_psnr(pred, gt.cuda()))
        ss.append(tensor_ssim(pred, gt.cuda()))
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(loader)}: PSNR={sum(ps)/len(ps):.2f}', flush=True)

print(f'RESULT {args.model} epoch={ck.get("epoch")} '
      f'PSNR={sum(ps)/len(ps):.3f} SSIM={sum(ss)/len(ss):.4f}', flush=True)
