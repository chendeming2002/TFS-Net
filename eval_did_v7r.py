#!/usr/bin/env python3
"""DID 跨数据集评测 (GolfNet_v7r_v3)。

在 SDSD 上训练的 v7r-v3 模型直接跑 DID test (跨数据集泛化/阶段测试)。
- 帧配对: 按 sorted 位置 (DID 的 LQ/GT 文件名一致, 001.jpg <-> 001.jpg)
- 全分辨率 1080p 计算 PSNR / SSIM / LPIPS(VGG)
- fp16 autocast: 峰值显存 ~7.5GB, 可与训练进程共存
- 结果追加写入 JSON, 便于 ep45 vs ep50 对比

用法:
    python eval_did_v7r.py --ckpt outputs/golf_v7r_v3_pospair/epoch_045.pth \
        --tag v7r_v3_pospair-ep45-DID --out_json outputs/did_metrics.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from models.golf_v7r import GolfNet_v7r_v3
from utils.metrics import tensor_psnr, tensor_ssim, LPIPSMetric


def read_image(path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tag', default=None)
    p.add_argument('--did_root', default='/home/a1005/yzy/dataset/DID/test')
    p.add_argument('--out_json', default='outputs/did_metrics.json')
    p.add_argument('--lpips_net', default='vgg', choices=['vgg', 'alex', 'squeeze'])
    p.add_argument('--max_frames', type=int, default=0, help='0=全部; N=每视频只跑前 N 帧')
    p.add_argument('--frac', type=float, default=0.0,
                   help='每视频均匀抽样比例 (如 0.33≈1/3); 0=全帧。与 --max_frames 互斥')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    half = 2  # window = 5

    model = GolfNet_v7r_v3().to(device).eval()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ck.get('model_state_dict', ck.get('model'))
    model.load_state_dict(state)
    epoch = ck.get('epoch', -1)
    tag = args.tag or 'v7r_v3-ep{}-DID'.format(epoch)
    print('Loaded {} | epoch {} | tag={}'.format(args.ckpt, epoch, tag), flush=True)

    lpips_fn = LPIPSMetric(net=args.lpips_net, device=device)
    print('LPIPS available: {}'.format(lpips_fn.available), flush=True)

    ll_root = os.path.join(args.did_root, 'low-light')
    gt_root = os.path.join(args.did_root, 'GT')
    videos = sorted([d for d in glob.glob(os.path.join(ll_root, '*')) if os.path.isdir(d)])

    all_ps, all_ss, all_lp = [], [], []
    per_seq = []
    t0 = time.time()

    with torch.no_grad():
        for vpath in videos:
            vname = os.path.basename(vpath)
            gtdir = os.path.join(gt_root, vname)
            if not os.path.isdir(gtdir):
                print('  skip {}: no GT'.format(vname), flush=True)
                continue

            frames = sorted(glob.glob(os.path.join(vpath, '*')))
            gt_frames = sorted(glob.glob(os.path.join(gtdir, '*')))
            n = min(len(frames), len(gt_frames))
            frames, gt_frames = frames[:n], gt_frames[:n]

            # 选择推理中心帧: --max_frames 取前 N; --frac 均匀抽样约 frac 比例; 否则全部
            if args.max_frames > 0:
                centers = list(range(min(args.max_frames, n)))
            elif args.frac > 0:
                k = max(1, int(round(args.frac * n)))
                centers = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, k)))
            else:
                centers = list(range(n))

            # 仅按需解码 (窗口相邻帧从磁盘读, 不整段驻留内存)
            def _frame(i):
                return read_image(frames[i])

            ps, ss, lps = [], [], []
            for ci, idx in enumerate(centers):
                mx = n - 1
                inds = [min(max(idx + o, 0), mx) for o in range(-half, half + 1)]
                clip = torch.stack([_frame(j) for j in inds], 0).unsqueeze(0).to(device)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = model(clip)['final'][0].float().clamp(0, 1)
                gt = read_image(gt_frames[idx]).to(device)
                ps.append(tensor_psnr(pred.unsqueeze(0), gt.unsqueeze(0)))
                ss.append(tensor_ssim(pred.unsqueeze(0), gt.unsqueeze(0)))
                if lpips_fn.available:
                    lps.append(lpips_fn.fn(pred.unsqueeze(0) * 2 - 1,
                                          gt.unsqueeze(0) * 2 - 1).item())
                del clip, pred, gt
                if (ci + 1) % 20 == 0:
                    print('  {}: {}/{} PSNR={:.2f}'.format(vname, ci + 1, len(centers),
                                                           np.mean(ps)), flush=True)

            all_ps.extend(ps); all_ss.extend(ss); all_lp.extend(lps)
            rec = {'sequence': vname, 'n_frames': len(ps),
                   'psnr': round(float(np.mean(ps)), 4),
                   'ssim': round(float(np.mean(ss)), 5),
                   'lpips': round(float(np.mean(lps)), 5) if lps else None}
            per_seq.append(rec)
            print('{}: frames={} PSNR={:.3f} SSIM={:.4f}{}'.format(
                vname, len(ps), rec['psnr'], rec['ssim'],
                ' LPIPS={:.4f}'.format(rec['lpips']) if rec['lpips'] is not None else ''),
                flush=True)

    dt = time.time() - t0

    def _macro(key):
        vals = [m[key] for m in per_seq if m[key] is not None]
        return round(sum(vals) / len(vals), 5) if vals else None

    result = {
        'tag': tag,
        'model': 'v7r_v3',
        'epoch': epoch,
        'dataset': 'DID/test',
        'protocol': ('frac={:.2f}'.format(args.frac) if args.frac > 0
                     else ('first{}'.format(args.max_frames) if args.max_frames > 0 else 'full')),
        'n_samples': len(all_ps),
        'n_sequences': len(per_seq),
        'psnr': round(float(np.mean(all_ps)), 4),
        'ssim': round(float(np.mean(all_ss)), 5),
        'lpips': round(float(np.mean(all_lp)), 5) if all_lp else None,
        'lpips_net': args.lpips_net if all_lp else None,
        'psnr_video': _macro('psnr'),
        'ssim_video': _macro('ssim'),
        'lpips_video': _macro('lpips'),
        'per_sequence': per_seq,
        'ckpt': args.ckpt,
        'eval_time_s': round(dt, 1),
    }

    print('\nRESULT {} ({} videos, {} frames, {:.0f}s):'.format(
        tag, result['n_sequences'], result['n_samples'], dt), flush=True)
    print('  Micro (per-frame avg): PSNR={:.4f} dB | SSIM={:.4f} | LPIPS={}'.format(
        result['psnr'], result['ssim'],
        '{:.4f}'.format(result['lpips']) if result['lpips'] is not None else 'N/A'), flush=True)
    print('  Macro (per-video avg):  PSNR={:.4f} dB | SSIM={:.4f} | LPIPS={}'.format(
        result['psnr_video'], result['ssim_video'],
        '{:.4f}'.format(result['lpips_video']) if result['lpips_video'] is not None else 'N/A'), flush=True)

    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    all_results = []
    if os.path.exists(args.out_json):
        try:
            all_results = json.load(open(args.out_json))
        except Exception:
            all_results = []
    all_results = [r for r in all_results if r.get('tag') != tag]
    all_results.append(result)
    json.dump(all_results, open(args.out_json, 'w'), indent=2, ensure_ascii=False)
    print('Saved to {}'.format(args.out_json), flush=True)


if __name__ == '__main__':
    main()
