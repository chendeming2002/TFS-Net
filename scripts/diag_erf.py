#!/usr/bin/env python3
"""ERF (有效感受野) 图: 对两种 RWKV 块的 Spatial Mix 做梯度分解。

方法 (Luo et al. 2016; Vision-RWKV 论文同款): 取输出的中心位置, 对输入求梯度,
把每个输入位置的梯度幅值画成热图。多组随机输入取平均。

本脚本利用两个头的签名 `forward(center_feat, context_feat)` —— Q 路与 KV 路是两个
独立张量参数 —— 不改模型代码即可分解:

  ERF_Q      = |d(final_center) / d(center_feat)|      (Q 路: R/G/残差/OmniShift/SPAB)
  ERF_KV     = |d(final_center) / d(context_feat)|     (KV 路: proj_k/v + 双向 WKV 扫描)
  ERF_Spatial= |d(spatial_mix_center) / d(x)|, x 同送两路 (两偏导之和 = 全导数)

探针位置:
  final        —— 头的最终输出 (Spatial Mix + 残差 + Channel Mix 之后)
  spatial mix  —— Channel Mix 之前的中间量 (encoder 头: channel_mix 的输入;
                  query 头: post_norm 的输入)

用法:
    python scripts/diag_erf.py --size 128 --n 24 --out_dir outputs/erf_random
    python scripts/diag_erf.py --ckpt outputs/v8_6_30ep/best.pth \
        --config configs/v8_5/v8_6_30ep.yaml --out_dir outputs/erf_v86
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402


def _spatial_probe_module(head):
    """返回接收 Spatial Mix 输出的子模块, 用 forward pre-hook 截取其输入。"""
    if hasattr(head, "cm_r"):            # CrossRWKVHeadV8: post_norm 的输入
        return head.post_norm
    if head.channel_mix is not None:      # SpatialRWKVHeadV8: channel_mix 的输入
        return head.channel_mix
    return None


def _force_cpu_wkv(head):
    """CPU 运行时把 CUDA WKV 换成纯 PyTorch 版 (参数名一致, 直接搬)。

    GPU 被训练占用时本脚本在 CPU 跑; O(L²) 的稠密实现限制了尺寸 (64×64)。
    """
    from models.modules.cuda_wkv import CudaBiWKV, TorchBiWKVSingleNorm
    import torch.nn as nn
    new_list = nn.ModuleList()
    for m in head.bi_wkv_list:
        if isinstance(m, CudaBiWKV):
            t = TorchBiWKVSingleNorm(m.channels)
            t.load_state_dict(m.state_dict())
            new_list.append(t)
        else:
            new_list.append(m)
    head.bi_wkv_list = new_list


def erf_maps(head, size, n_inputs, device, seed=0):
    """返回 (q, kv, spatial) 三张平均后的 ERF 图, 已按最大值归一。"""
    C = head.channels
    torch.manual_seed(seed)
    acc = {"q": None, "kv": None, "spatial": None}

    probe_mod = _spatial_probe_module(head)
    H = W = size
    center_tok = (H // 2) * W + (W // 2)

    for _ in range(n_inputs):
        x = torch.randn(1, C, H, W, device=device)

        # ---- 探针 A: 最终输出的中心 → Q / KV 两张偏导图 ----
        xq = x.clone().requires_grad_(True)
        xv = x.clone().requires_grad_(True)
        out = head(xq, xv)
        out[0, :, H // 2, W // 2].sum().backward()
        m_q = xq.grad.abs().mean(dim=1)[0]        # [H, W]
        m_kv = xv.grad.abs().mean(dim=1)[0]

        # ---- 探针 B: Spatial Mix 输出的中心 → 总 ERF (两偏导之和) ----
        if probe_mod is not None:
            captured = {}
            handle = probe_mod.register_forward_pre_hook(
                lambda mod, args: captured.__setitem__("t", args[0]))
            xq2 = x.clone().requires_grad_(True)
            xv2 = x.clone().requires_grad_(True)
            head(xq2, xv2)
            handle.remove()
            captured["t"][0, center_tok].sum().backward()
            m_sp = (xq2.grad.abs() + xv2.grad.abs()).mean(dim=1)[0]
        else:
            m_sp = m_q + m_kv

        for key, m in (("q", m_q), ("kv", m_kv), ("spatial", m_sp)):
            m = m / (m.sum() + 1e-20)             # 每张图先按总能量归一再平均
            acc[key] = m if acc[key] is None else acc[key] + m

    return {k: (v / n_inputs).cpu().numpy() for k, v in acc.items()}


def locality_metrics(m):
    """中心能量占比 (半径 4 内) 与覆盖 90% 能量的最小半径。"""
    h, w = m.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    total = m.sum() + 1e-20
    frac4 = m[dist <= 4].sum() / total
    order = np.argsort(m.ravel())[::-1]
    csum = np.cumsum(m.ravel()[order]) / total
    r90 = dist.ravel()[order][np.searchsorted(csum, 0.9)]
    return float(frac4), float(r90)


def plot_grid(maps_by_head, out_path, title_prefix):
    """2 行 (encoder/query) × 3 列 (Q/KV/Spatial) 的热图, log 尺度着色。"""
    heads = list(maps_by_head.keys())
    paths = ["q", "kv", "spatial"]
    path_names = {"q": "Q path", "kv": "KV path", "spatial": "Spatial Mix (total)"}
    fig, axes = plt.subplots(len(heads), 3, figsize=(13, 4.6 * len(heads)))
    if len(heads) == 1:
        axes = axes[None, :]
    for i, hname in enumerate(heads):
        for j, p in enumerate(paths):
            m = maps_by_head[hname][p]
            disp = np.log10(m / (m.max() + 1e-20) + 1e-6)
            ax = axes[i, j]
            im = ax.imshow(disp, cmap="magma",
                           vmin=-6, vmax=0)
            f4, r90 = locality_metrics(m)
            ax.set_title(f"{hname} / {path_names[p]}\n"
                         f"E(r<=4) {f4:.1%} / R90 {r90:.0f}px", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{title_prefix} — ERF (log scale, |grad|)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="训练好的 checkpoint; 缺省用随机初始化")
    p.add_argument("--config", default=None, help="与 ckpt 配套的训练配置")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--n", type=int, default=24, help="随机输入组数")
    p.add_argument("--out_dir", default="outputs/erf")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # GPU 可能被训练占用; CUDA WKV 无法在 CPU 张量上运行
        args.device = "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    from models.golf_v7r.spatial_rwkv_head_v8 import SpatialRWKVHeadV8
    from models.golf_v7r.cross_rwkv_head_v8 import CrossRWKVHeadV8

    if args.ckpt:
        import yaml
        from models.golf_v7r import GolfNet_v8
        from utils.io import load_state_dict_compat
        cfg = yaml.safe_load(open(args.config))
        mk = dict(cfg.get("model", {}) or {})
        mk.pop("arch", None)
        model = GolfNet_v8(**mk).to(device).eval()
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        try:
            load_state_dict_compat(model, ck["model_state_dict"])
        except RuntimeError:
            # ckpt 早于后续版本新增的分支键 (如 v8.7 的 ms_conv*):
            # ERF 只用 encoder 头与 tca.attn_N, 校验这两个头无缺失即可
            r = model.load_state_dict(ck["model_state_dict"], strict=False)
            need = [k for k in r.missing_keys
                    if k.startswith("encoder.") or k.startswith("tca.attn_N.")]
            assert not need, f"ERF 所需的头缺键: {need[:4]}"
        enc_head = model.encoder.scale1_blocks[0].head
        q_head = model.tca.attn_N
        tag = f"trained (epoch {ck.get('epoch')})"
    else:
        torch.manual_seed(0)
        enc_head = SpatialRWKVHeadV8(32, wkv_mode="single_norm").to(device).eval()
        q_head = CrossRWKVHeadV8(32, wkv_mode="single_norm").to(device).eval()
        tag = "random init"

    heads = {"EncoderRWKVBlock": enc_head, "QueryRWKVBlock": q_head}
    if device == "cpu":
        for h in heads.values():
            _force_cpu_wkv(h)
        torch.set_num_threads(16)
    maps_by_head = {name: erf_maps(h, args.size, args.n, device)
                    for name, h in heads.items()}

    plot_grid(maps_by_head, os.path.join(args.out_dir, "erf.png"), tag)

    print(f"RESULT {tag}, size={args.size}, n={args.n}", flush=True)
    print(f"{'block':<20}{'path':<10}{'E(r<=4)':>12}{'R90(px)':>10}", flush=True)
    for name, maps in maps_by_head.items():
        for pname in ("q", "kv", "spatial"):
            f4, r90 = locality_metrics(maps[pname])
            print(f"{name:<20}{pname:<10}{f4:>11.1%}{r90:>10.0f}", flush=True)
    print(f"saved -> {args.out_dir}/erf.png", flush=True)


if __name__ == "__main__":
    main()
