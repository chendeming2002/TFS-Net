#!/usr/bin/env bash
# 顺序评估 v7 与 v7r 的 checkpoint (PSNR + SSIM + LPIPS)
# LPIPS 以「各帧 LPIPS 平均」作为该视频整体的指标, 再在视频间聚合
set -u
cd /home/a1005/25/TFS-Net

PY=/home/a1005/anaconda3/envs/ptorch/bin/python
export PYTHONPATH=/home/a1005/25/TFS-Net

echo "===== [1/2] Golf v7 (baseline) ep10 ====="
$PY -u eval_checkpoint.py \
    --ckpt outputs/golf_v7/latest.pth \
    --model v7 \
    --tag "v7-ep10"

echo ""
echo "===== [2/2] Golf v7r-v3 (pospair) best=ep55 ====="
# 注: 原 v7r (outputs/golf_v7r/) 目录已清理; 当前对照改用 v7r-v3 pospair best。
$PY -u eval_checkpoint.py \
    --ckpt outputs/golf_v7r_v3_pospair/best.pth \
    --model v7r_v3 \
    --pairing position \
    --tag "v7r_v3-pospair-ep55"

echo ""
echo "===== 汇总 (微平均 / 宏平均) ====="
$PY -c "
import json
r = json.load(open('outputs/eval_metrics.json'))
hdr = f\"{'tag':<10} {'PSNR↑':>8} {'SSIM↑':>8} {'LPIPS↓':>8} {'LPIPS_vid↓':>11} {'params':>8} {'videos':>7} {'frames':>7}\"
print(hdr)
print('-'*len(hdr))
for x in sorted(r, key=lambda z: z['tag']):
    lpm = x.get('lpips')
    lpv = x.get('lpips_video')
    lpm_s = f'{lpm:.4f}' if lpm is not None else 'N/A'
    lpv_s = f'{lpv:.4f}' if lpv is not None else 'N/A'
    print(f\"{x['tag']:<10} {x['psnr']:>8.3f} {x['ssim']:>8.4f} {lpm_s:>8} {lpv_s:>11} {x['params_M']:>7.2f}M {x.get('n_sequences',0):>7} {x['n_samples']:>7}\")
print()
print('注: LPIPS = 全部帧平均; LPIPS_vid = 各视频帧平均后再对视频平均')
"
