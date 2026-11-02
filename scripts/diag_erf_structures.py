#!/usr/bin/env python3
"""随机初始化 ERF 的结构对比: 三个 RWKV 块变体 + VRT 的注意力块。

用随机初始化 (同 seed) 隔离"结构先验"—— 没有训练痕迹, ERF 差异完全来自
模块的数据流拓扑。对比对象:

  1. RWKV Encoder (v8.6+ 结构): OmniShift KV + 4 方向 BiWKV + sigmoid(q) + 残差 + CM
  2. RWKV Query omni (v8.8 结构): SPAB R 路 + 共享 OmniShift + 残差 + cm_shift
  3. RWKV Query RLK-TS (v8.8N 结构): 同上, Shift 换 RLK-TS
  4. VRT TMSA (CVPR 2022): 窗口 (2×8×8) 时空互注意 + GEGLU FFN, 经适配器
     接入 (center, context) 双输入 —— center 为第 0 帧, context 为第 1 帧,
     Q 路梯度 = 中心帧 (自注意 q + 残差), KV 路梯度 = 邻帧 (互注意 k/v)。
     源码: reference_repos/VRT/models/network_vrt.py (官方仓库, BSD)。

用法 (CPU):
    PYTHONPATH=. python scripts/diag_erf_structures.py --size 64 --n 16
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402

from diag_erf import erf_maps, _force_cpu_wkv, locality_metrics  # noqa: E402

PATHS = ("q", "kv", "spatial")
PATH_NAMES = {"q": "Q path", "kv": "KV path", "spatial": "Block total"}


class VRTAttentionAdapter(nn.Module):
    """把 VRT 的 TMSA 块适配到 (center_feat, context_feat) 双输入接口。

    center 作为第 0 帧、context 作为第 1 帧堆成 D=2 的视频张量, 跑一个
    TMSA 块 (窗口 2×8×8, 不 shift, 互注意开), 返回中心帧的输出。
    VRT 的互注意正是为两帧设计: 各帧一半头与对方做交叉检索。
    """

    def __init__(self, dim=32, size=64, num_heads=4):
        super().__init__()
        self.channels = dim
        spec = importlib.util.spec_from_file_location(
            "network_vrt",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "reference_repos/VRT/models/network_vrt.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.block = mod.TMSA(dim=dim, input_resolution=(2, size, size),
                              num_heads=num_heads, window_size=(2, 8, 8),
                              shift_size=(0, 0, 0), mut_attn=True)

    def forward(self, center_feat, context_feat):
        x = torch.stack([center_feat, context_feat], dim=1)   # B,2,C,H,W
        x = x.permute(0, 1, 3, 4, 2).contiguous()             # B,D,H,W,C
        y = self.block(x, None)
        return y[:, 0].permute(0, 3, 1, 2).contiguous()       # B,C,H,W


def build_heads(size, device):
    from models.golf_v7r.spatial_rwkv_head_v8 import SpatialRWKVHeadV8
    from models.golf_v7r.cross_rwkv_head_v8 import CrossRWKVHeadV8
    torch.manual_seed(0)
    heads = [
        ("RWKV Encoder (v8.6+)",
         SpatialRWKVHeadV8(32, wkv_mode="single_norm")),
        ("RWKV Query omni (v8.8)",
         CrossRWKVHeadV8(32, wkv_mode="single_norm",
                         shift_share="by_source", cm_shift=True)),
        ("RWKV Query RLK-TS (v8.8N)",
         CrossRWKVHeadV8(32, wkv_mode="single_norm",
                         shift_share="by_source", cm_shift=True, shift_kind="rlk")),
        ("VRT TMSA (window 8x8)",
         VRTAttentionAdapter(32, size)),
    ]
    out = []
    for name, h in heads:
        h = h.to(device).eval()
        if name.startswith("RWKV"):
            _force_cpu_wkv(h)
        out.append((name, h))
    return out


def dir_energy(m):
    h, w = m.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    total = m.sum() + 1e-20
    return {"row": float(m[cy, :].sum() / total),
            "col": float(m[:, cx].sum() / total),
            "diag": float(m[(xx - yy) == 0].sum() / total),
            "adiag": float(m[(xx + yy) == (h - 1)].sum() / total)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--out_dir", default="outputs/erf_structures")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(16)

    rows = []
    for name, h in build_heads(args.size, args.device):
        maps = erf_maps(h, args.size, args.n, args.device)
        rows.append((name, maps))

    print(f"RESULT random init, size={args.size}, n={args.n}", flush=True)
    print(f"{'block':<28}{'path':<10}{'E(r<=4)':>10}{'R90(px)':>9}", flush=True)
    for name, maps in rows:
        for pn in PATHS:
            f4, r90 = locality_metrics(maps[pn])
            print(f"{name:<28}{pn:<10}{f4:>10.1%}{r90:>9.0f}", flush=True)

    print(f"\nKV path directional energy (uniform 1.56%/line):", flush=True)
    print(f"{'block':<28}{'row':>7}{'col':>7}{'diag':>7}{'adiag':>7}", flush=True)
    for name, maps in rows:
        d = dir_energy(maps["kv"])
        print(f"{name:<28}{d['row']:>7.1%}{d['col']:>7.1%}"
              f"{d['diag']:>7.1%}{d['adiag']:>7.1%}", flush=True)

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.7 * n_rows))
    for i, (name, maps) in enumerate(rows):
        for j, pn in enumerate(PATHS):
            m = maps[pn]
            disp = np.log10(m / (m.max() + 1e-20) + 1e-6)
            ax = axes[i, j]
            im = ax.imshow(disp, cmap="magma", vmin=-6, vmax=0)
            f4, r90 = locality_metrics(m)
            ax.set_title(f"{name}\n{PATH_NAMES[pn]}  E(r<=4) {f4:.1%} / "
                         f"R90 {r90:.0f}px", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Structural ERF at random init: RWKV variants vs VRT TMSA",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.out_dir, "erf_structures.png")
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
