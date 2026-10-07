#!/usr/bin/env python3
"""v7r-v3 推理可视化脚本

对 val 序列逐帧滑窗推理 (window=5), 保存预测图, 并生成对比图
(Pred | GT | Low-light 横向拼接), 用于定性评估。

用法:
    python infer_golf_v7r_v3.py --ckpt outputs/golf_v7r_v3/best.pth \
        --seqs pair19 pair45 --out_dir outputs/golf_v7r_v3_inference
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from models.golf_v7r import GolfNet_v7r_v3
from datasets import SDSDDataset


def load_model(ckpt_path, device='cuda'):
    model = GolfNet_v7r_v3().to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    print(f"Loaded {ckpt_path} | epoch {ck.get('epoch', '?')}", flush=True)
    return model


def _read(path):
    return TF.to_tensor(Image.open(path).convert('RGB'))


def build_compare_grid(pred, gt, lq, out_path):
    """横向拼接: Low-light | Prediction | GT, 中间加白边"""
    imgs = []
    for t in (lq, pred, gt):
        arr = (t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        imgs.append(arr)
    sep = np.full((imgs[0].shape[0], 4, 3), 255, dtype=np.uint8)
    grid = np.concatenate([imgs[0], sep, imgs[1], sep, imgs[2]], axis=1)
    Image.fromarray(grid).save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--seqs', nargs='+', default=['pair19', 'pair45'])
    p.add_argument('--out_dir', default='outputs/golf_v7r_v3_inference')
    p.add_argument('--data_root', default='/home/a1005/yzy/dataset/SDSD/test')
    p.add_argument('--max_frames', type=int, default=None)
    p.add_argument('--pairing', default='name', choices=['name', 'position'],
                   help='LQ/GT 配对方式: name=文件名交集(历史), position=sorted 位置(官方约定)')
    p.add_argument('--amp', action='store_true',
                   help='fp16 autocast 推理 (全 1080p 显存 ~7.5GB, 可与训练进程共存)')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.ckpt, device)

    window = 5
    half = window // 2

    for seq in args.seqs:
        in_dir = os.path.join(args.data_root, 'low-light', seq)
        gt_dir = os.path.join(args.data_root, 'GT', seq)
        out_seq = os.path.join(args.out_dir, seq)
        out_cmp = os.path.join(args.out_dir, seq + '_compare')
        out_gt = os.path.join(args.out_dir, seq + '_gt')
        os.makedirs(out_seq, exist_ok=True)
        os.makedirs(out_cmp, exist_ok=True)
        os.makedirs(out_gt, exist_ok=True)

        lq_names = sorted([f for f in os.listdir(in_dir) if f.endswith('.png')])
        gt_names = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
        if args.pairing == 'position':
            # 官方 SDSD 约定: 按 sorted 位置配对 (LQ/GT 帧数相等, 文件名区间可能偏移)
            n = min(len(lq_names), len(gt_names))
            lq_names, gt_names = lq_names[:n], gt_names[:n]
            paired = list(zip(lq_names, gt_names))
            print(f"[{seq}] {len(paired)} position-paired frames (lq={len(lq_names)}, gt={len(gt_names)})", flush=True)
        else:
            # 按 LQ/GT 文件名交集配对
            gt_set = set(gt_names)
            names = [n for n in lq_names if n in gt_set]
            if not names:
                print(f"[{seq}] 无 LQ/GT 交集帧, 跳过", flush=True)
                continue
            paired = [(n, n) for n in names]
            print(f"[{seq}] {len(paired)} common frames (lq={len(lq_names)}, gt={len(gt_names)})", flush=True)

        if args.max_frames:
            paired = paired[:args.max_frames]
        if not paired:
            print(f"[{seq}] 无可推理帧, 跳过", flush=True)
            continue
        names = [p[0] for p in paired]

        # 读入 LQ 帧 (RGB float); 窗口仅在配对序列上滑动
        lq_map = {n: _read(os.path.join(in_dir, n)) for n in names}

        with torch.no_grad():
            for i, (lq_name, gt_name) in enumerate(paired):
                idxs = [min(max(i + o, 0), len(names) - 1) for o in range(-half, half + 1)]
                clip = torch.stack([lq_map[names[j]] for j in idxs], dim=0).unsqueeze(0).to(device)  # [1,T,3,H,W]
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=args.amp):
                    pred = model(clip)['final'][0].float().clamp(0, 1)

                TF.to_pil_image(pred.cpu()).save(os.path.join(out_seq, lq_name))

                gt = _read(os.path.join(gt_dir, gt_name))
                build_compare_grid(pred, gt, lq_map[lq_name], os.path.join(out_cmp, lq_name))
                TF.to_pil_image(gt).save(os.path.join(out_gt, lq_name))

                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/{len(paired)}", flush=True)

        print(f"[{seq}] done -> {out_seq}, {out_cmp}", flush=True)


if __name__ == '__main__':
    main()
