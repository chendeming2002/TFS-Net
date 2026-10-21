#!/usr/bin/env python3
"""v8 模型复杂度与推理性能测量。

固定口径 (便于横向比较):
  - 输入 256×256, batch_size=1, 5 帧, FP32 (关闭 TF32)
  - 参数量: 全部可训练参数
  - FLOPs: thop 统计卷积/线性层的乘加; WKV 扫描是自定义 CUDA 算子, thop 无法追踪,
    按线性复杂度单独折算 (每个元素 8 次浮点运算), 两部分相加
  - 显存: torch.cuda.max_memory_allocated, 单次前向的峰值
  - 延迟: 预热 20 次后取 100 次的均值与中位数, CUDA event 计时
  - FPS: 1000 / 平均延迟 (batch 1 的单帧吞吐)

用法:
    python scripts/bench_v8.py --config configs/v8/fullkv_15ep.yaml
    python scripts/bench_v8.py --config configs/v8/unet_refactor_15ep.yaml --tag baseline
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _count_macs(model, x):
    """统计一次前向中卷积与线性层的乘加次数 (与 thop 的口径相同)。"""
    total = [0]

    def conv_hook(m, inp, out):
        out_elems = out.numel()
        k = m.kernel_size[0] * m.kernel_size[1]
        total[0] += out_elems * (m.in_channels // m.groups) * k

    def linear_hook(m, inp, out):
        total[0] += m.in_features * m.out_features * (out.numel() // m.out_features)

    handles = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return float(total[0])


def wkv_flops(model, H, W):
    """按线性复杂度折算 WKV 扫描的浮点运算量。

    每个空间头有 num_directions 个扫描, 每个扫描在 L=H'*W' 个 token 上处理 head_dim 通道。
    每个元素约 8 次浮点运算 (衰减乘、加权累加、分母累加、除法各约 2 次)。
    """
    total = 0
    heads = []

    def add(head_mod, L, loops):
        nonlocal total
        c = head_mod.channels
        total += loops * head_mod.num_dirs * L * c * 8
        heads.append((c, L, loops))

    # 编码器与聚合器内的 EncoderRWKVBlock
    enc = model.encoder
    for scale, (L, loops) in enumerate([
            (H * W, enc.scale1_blocks[0].num_loops),
            ((H // 2) * (W // 2), enc.scale2_blocks[0].num_loops),
            ((H // 4) * (W // 4), enc.scale3_blocks[0].num_loops)], start=1):
        add(getattr(enc, f"scale{scale}_blocks")[0].head, L, loops)

    # 聚合器内的 frame_encoder, 对 5 帧各跑一次
    T = model.num_frames
    add(model.summary.frame_encoder.head, (H // 4) * (W // 4),
        model.summary.frame_encoder.num_loops * T)

    # 三路任务查询头, 均在 F1 尺度
    for name in ("attn_N", "attn_L", "attn_M"):
        add(getattr(model.tca, name), H * W, 1)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/v8/fullkv_15ep.yaml')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--tag', default=None)
    p.add_argument('--json', default=None)
    args = p.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    mcfg = dict(cfg.get('model') or {})
    arch = mcfg.pop('arch', 'v7r_v3')
    if arch != 'v8':
        raise SystemExit(f"本脚本只测 v8 架构, 收到 {arch}")

    from models.golf_v7r import GolfNet_v8
    from models.modules.cuda_wkv import backend_report

    dev = 'cuda'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False     # 严格 FP32
    torch.backends.cudnn.allow_tf32 = False

    model = GolfNet_v8(**mcfg).to(dev).eval()
    print(backend_report(require_cuda=True), flush=True)

    H = W = args.size
    x = torch.randn(1, model.num_frames, 3, H, W, device=dev)

    n_params = sum(q.numel() for q in model.parameters())

    # ---------- FLOPs ----------
    # 用 hook 统计卷积与线性层的乘加次数。thop 对 PixelShuffle 这类无参数重排操作
    # 会残留错误的属性, 在 v8.5 的 lite 聚合器上直接报错, 因此这里自己统计。
    # 训练形态 (OmniShift 四分支) 与推理形态 (重参数化后单卷积) 的乘加次数不同,
    # 这里都报告: 前者用于与训练成本比较, 后者用于推理部署。
    macs_train_form = _count_macs(model, x)
    model.eval()                     # 触发 OmniShift 自动重参数化
    with torch.no_grad():
        model(x)
    macs = _count_macs(model, x)
    # thop 返回的是乘加次数 (MAC)。一次乘加计两次浮点运算。
    conv_macs = macs
    conv_flops = 2.0 * conv_macs
    wkv = wkv_flops(model, H, W)
    total_flops = conv_flops + wkv

    # ---------- 显存与延迟 ----------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(x)
        torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated() / 1024 ** 3

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    with torch.no_grad():
        for i in range(args.iters):
            starts[i].record()
            model(x)
            ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    mean_ms = sum(times) / len(times)
    med_ms = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]

    result = {
        "tag": args.tag or os.path.basename(os.path.dirname(args.config)),
        "config": args.config,
        "input": f"{H}x{W}, batch_size=1, FP32",
        "params_M": round(n_params / 1e6, 3),
        "flops_G": round(total_flops / 1e9, 2),
        "flops_conv_G": round(conv_flops / 1e9, 2),
        "macs_G": round(conv_macs / 1e9, 2),
        "flops_train_form_G": round((2 * macs_train_form + wkv) / 1e9, 2),
        "flops_wkv_G": round(wkv / 1e9, 3),
        "memory_GB": round(mem_peak, 3),
        "latency_mean_ms": round(mean_ms, 2),
        "latency_median_ms": round(med_ms, 2),
        "latency_p95_ms": round(p95, 2),
        "fps": round(1000.0 / mean_ms, 1),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        old = []
        if os.path.exists(args.json):
            old = json.load(open(args.json))
        old.append(result)
        json.dump(old, open(args.json, 'w'), ensure_ascii=False, indent=2)
        print(f"saved -> {args.json}")


if __name__ == '__main__':
    main()
