#!/usr/bin/env python3
"""独立评估脚本: 加载 checkpoint 计算 val PSNR / SSIM / LPIPS

用法:
    python eval_checkpoint.py --ckpt outputs/golf_v7/latest.pth --model v7 --tag "v7-ep10"
    python eval_checkpoint.py --ckpt outputs/golf_v7r/latest.pth --model v7r --tag "v7r-ep5"

指标 (均在全分辨率 1080p 上计算, 越低/越高越好的方向见表头):
    PSNR  ↑   dB
    SSIM  ↑   [0,1]
    LPIPS ↓   [0,~1)  VGG backbone, 越小越感知相似
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from datasets import SDSDDataset
from utils.metrics import tensor_psnr, tensor_ssim

try:
    import lpips
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False


def build_model(name):
    if name == 'v7':
        from models.golf_v7 import GolfNet_v7 as M
        return M()
    elif name == 'v7r':
        from models.golf_v7r import GolfNet_v7r as M
        return M()
    elif name == 'v7r_v3':
        from models.golf_v7r import GolfNet_v7r_v3 as M
        return M()
    elif name == 'r2':
        from models.golf import GolfNet as M
        return M()
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model', choices=['v7', 'v7r', 'v7r_v3', 'r2'], required=True)
    p.add_argument('--tag', default=None, help='结果标签, 默认用 model+epoch')
    p.add_argument('--lpips_net', default='vgg', choices=['vgg', 'alex', 'squeeze'])
    p.add_argument('--out_json', default='outputs/eval_metrics.json',
                   help='结果追加写入的 JSON 文件 (方便综合比较)')
    p.add_argument('--pairing', default='name', choices=['name', 'position'],
                   help='LQ/GT 配对方式: name=文件名交集(历史), position=sorted 位置(官方约定)')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- dataset ----
    ds = SDSDDataset(input_root='/home/a1005/yzy/dataset/SDSD/test/low-light',
                     target_root='/home/a1005/yzy/dataset/SDSD/test/GT',
                     window_size=5, mode='val', crop_size=None,
                     pairing=args.pairing)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # ---- model ----
    model = build_model(args.model).to(device).eval()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ck.get('model_state_dict', ck.get('model'))
    model.load_state_dict(state)
    epoch = ck.get('epoch', -1)
    n_params = sum(p.numel() for p in model.parameters())
    tag = args.tag or f'{args.model}-ep{epoch}'
    print(f'Loaded {args.model} epoch {epoch} | params {n_params/1e6:.2f}M | tag={tag}', flush=True)

    # ---- LPIPS ----
    lpips_fn = None
    if _HAS_LPIPS:
        lpips_fn = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()
    else:
        print('WARNING: lpips 未安装, 跳过 LPIPS', flush=True)

    ps, ss, lp = [], [], []
    # 按序列 (视频) 聚合: 每帧指标 → 每视频均值
    per_seq = {}   # seq_name -> {'psnr': [], 'ssim': [], 'lpips': []}
    t0 = time.time()
    with torch.no_grad():
        for i, (lq, gt, meta) in enumerate(loader):
            lq = lq.to(device)
            gt = gt.to(device)
            out = model(lq)
            pred = out['final'] if 'final' in out else out['res_t']

            seq = meta['sequence'][0] if isinstance(meta['sequence'], (list, tuple)) else meta['sequence']
            psnr_i = tensor_psnr(pred, gt)
            ssim_i = tensor_ssim(pred, gt)
            ps.append(psnr_i)
            ss.append(ssim_i)

            s = per_seq.setdefault(seq, {'psnr': [], 'ssim': [], 'lpips': [], 'n_frames': 0})
            s['psnr'].append(psnr_i)
            s['ssim'].append(ssim_i)
            s['n_frames'] += 1

            if lpips_fn is not None:
                # LPIPS 输入范围 [-1, 1]
                lpips_i = lpips_fn(pred * 2 - 1, gt * 2 - 1).item()
                lp.append(lpips_i)
                s['lpips'].append(lpips_i)

            if (i + 1) % 200 == 0:
                msg = (f'  {i+1}/{len(loader)}  PSNR={sum(ps)/len(ps):.2f}  '
                       f'SSIM={sum(ss)/len(ss):.4f}')
                if lp:
                    msg += f'  LPIPS={sum(lp)/len(lp):.4f}'
                print(msg, flush=True)

    # 按视频聚合 (每序列均值)
    seq_metrics = []
    for seq in sorted(per_seq.keys()):
        s = per_seq[seq]
        seq_metrics.append({
            'sequence': seq,
            'n_frames': s['n_frames'],
            'psnr': round(sum(s['psnr']) / len(s['psnr']), 4),
            'ssim': round(sum(s['ssim']) / len(s['ssim']), 5),
            'lpips': round(sum(s['lpips']) / len(s['lpips']), 5) if s['lpips'] else None,
        })

    # 视频级宏观平均: 序列间平均 (每个视频等权, 而非每帧等权)
    def _macro(key):
        vals = [m[key] for m in seq_metrics if m[key] is not None]
        return round(sum(vals) / len(vals), 5) if vals else None

    result = {
        'tag': tag,
        'model': args.model,
        'epoch': epoch,
        'params_M': round(n_params / 1e6, 3),
        'pairing': args.pairing,
        'n_samples': len(ps),
        'n_sequences': len(seq_metrics),
        # 微平均 (每帧等权)
        'psnr': round(sum(ps) / len(ps), 4),
        'ssim': round(sum(ss) / len(ss), 5),
        'lpips': round(sum(lp) / len(lp), 5) if lp else None,
        'lpips_net': args.lpips_net if lp else None,
        # 宏平均 (每视频等权, 即"各帧 LPIPS 平均作为该视频指标"再对视频平均)
        'lpips_video': _macro('lpips'),
        'psnr_video': _macro('psnr'),
        'ssim_video': _macro('ssim'),
        'per_sequence': seq_metrics,
        'ckpt': args.ckpt,
        'eval_time_s': round(time.time() - t0, 1),
    }

    lp_micro = f"{result['lpips']:.4f}" if result['lpips'] is not None else 'N/A'
    lp_macro = f"{result['lpips_video']:.4f}" if result['lpips_video'] is not None else 'N/A'
    print(f"\nRESULT {tag} ({result['n_sequences']} videos, {result['n_samples']} frames):", flush=True)
    print(f"  Micro (per-frame avg): PSNR={result['psnr']:.4f} dB | "
          f"SSIM={result['ssim']:.4f} | LPIPS={lp_micro}", flush=True)
    print(f"  Macro (per-video avg):  PSNR={result['psnr_video']:.4f} dB | "
          f"SSIM={result['ssim_video']:.4f} | LPIPS={lp_macro}", flush=True)

    # 追加写入 JSON (便于综合比较)
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    all_results = []
    if os.path.exists(args.out_json):
        try:
            with open(args.out_json) as f:
                all_results = json.load(f)
        except Exception:
            all_results = []
    # 同 tag 覆盖
    all_results = [r for r in all_results if r.get('tag') != tag]
    all_results.append(result)
    with open(args.out_json, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'Saved to {args.out_json}', flush=True)


if __name__ == '__main__':
    main()
