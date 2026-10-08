#!/usr/bin/env python3
"""数据集亮度分布统计 (支持 docs/v7/03-v7r-v3-design.md §6.6)。

对每个序列采样若干帧，计算平均亮度 (0-1)，用于判断训练集是否覆盖测试集的暗度分布。
结论：DID video19/video20 平均亮度 0.019/0.023，比 SDSD 训练集最暗序列 (0.043) 还暗 ~2x，
属分布外 → 解释为何这两段 PSNR 仅 ~12。

用法:
    PYTHONPATH=/home/a1005/25/TFS-Net python scripts/dataset_luminance.py \
        --roots /home/a1005/yzy/dataset/DID/test/low-light \
                /home/a1005/yzy/dataset/SDSD/indoor/input \
                /home/a1005/yzy/dataset/SDSD/test/low-light
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image


def seq_mean_luminance(d, n_samples=8):
    fs = sorted(glob.glob(os.path.join(d, "*")))
    if not fs:
        return None
    idx = np.linspace(0, len(fs) - 1, min(n_samples, len(fs))).astype(int)
    vals = []
    for i in idx:
        a = np.asarray(Image.open(fs[i]).convert("RGB"), dtype=np.float32) / 255.0
        vals.append(float(a.mean()))
    return float(np.mean(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--n_samples", type=int, default=8)
    args = p.parse_args()

    for root in args.roots:
        print(f"== {root} ==")
        rows = []
        for s in sorted(os.listdir(root)):
            d = os.path.join(root, s)
            if os.path.isdir(d):
                m = seq_mean_luminance(d, args.n_samples)
                if m is not None:
                    rows.append((m, s))
        rows.sort()
        for m, s in rows:
            print(f"  {s:<12} {m:.4f}")
        if rows:
            med = rows[len(rows) // 2][0]
            print(f"  --- darkest={rows[0][0]:.4f} ({rows[0][1]}), median={med:.4f}, "
                  f"brightest={rows[-1][0]:.4f} ({rows[-1][1]})")
        print()


if __name__ == "__main__":
    main()
