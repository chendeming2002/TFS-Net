#!/usr/bin/env python3
"""多 seed 基线: 量化训练的 run-to-run 随机性, 为判据提供 σ 估计。

动机 (docs/v7/03 §6.14, 2026-10-10):
  实测同配置两次独立运行 (`pospair` vs `pospair_quick`, 仅 output_dir 不同、同 seed)
  的 ep5 val PSNR 相差 **2.13 dB**。该方差远超任何 toggle 效应量 (≤0.3 dB),
  使所有单 run 对比失效。
  ⇒ 判据不能用单 run 的绝对值, 必须用 **多 seed 重复的均值 ± σ**, 容差取 ≥2σ。

关于"逐位复现"的结论 (2026-10-10 实测, 已放弃):
  已依次排除 cuDNN (关 autotune)、CuBLAS (CUBLAS_WORKSPACE_CONFIG)、
  CUDA cumsum (CPU 绕行), 残余不可复现量级 ~1e-5 (相对) 仍存在, 指向更底层的
  atomicAdd 归约。**追到逐位复现的成本高、收益低** —— 因为多 seed 重复
  **不依赖**逐位复现, 直接量化随机性, 更直接也更稳健。

用法:
    PYTHONPATH=. python scripts/multiseed_baseline.py --config configs/golf_v7r_v3_phaseB_floored.yaml \
        --seeds 0 1 2 --stop_epoch 5 --out outputs/multiseed_floored.json
    # 后台跑 (推荐, 配合 run_in_background):
    nohup ... > outputs/multiseed_floored.log 2>&1 &

产物: JSON 含每 seed 的 PSNR/SSIM/LPIPS + 均值/标准差 + 95% 区间, 供后续判据直接引用。
"""
import argparse
import json
import os
import re
import subprocess
import sys
from statistics import mean, stdev


def run_one(config, seed, stop_epoch, out_root, extra_env=None):
    """跑一个 seed: 复制配置改 seed + output_dir, 训练, 返回 val 指标。"""
    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg['seed'] = seed
    cfg['output_dir'] = f"{out_root}/seed{seed}"
    tag = os.path.basename(out_root.rstrip('/'))
    # 临时配置写到 out_root 下 (不污染 configs/)
    os.makedirs(out_root, exist_ok=True)
    cfg_path = f"{out_root}/_cfg_seed{seed}.yaml"
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    log = f"{cfg['output_dir']}/train.log"
    os.makedirs(cfg['output_dir'], exist_ok=True)
    # 磁盘保护: 每个 run 约 130MB (best+epoch_005+latest), 留 2GB 余量
    st = os.statvfs(cfg['output_dir'])
    free_gb = st.f_bavail * st.f_frsize / 1e9
    if free_gb < 2.0:
        raise RuntimeError(f"磁盘余量不足 ({free_gb:.1f} GB < 2 GB), 中止以免写坏检查点")
    env = dict(os.environ, PYTHONPATH='.')
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, 'train_golf_v7r_v3.py', '--config', cfg_path,
           '--stop_epoch', str(stop_epoch)]
    print(f"[multiseed] seed={seed} → {cfg['output_dir']}  (磁盘余量 {free_gb:.1f} GB)", flush=True)
    with open(log, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)

    # 解析最后一条 Validation
    txt = open(log, errors='ignore').read()
    m = re.findall(r'Validation - PSNR: ([\d.]+) dB, SSIM: ([\d.]+), LPIPS: ([\d.]+)', txt)
    if not m:
        raise RuntimeError(f"seed={seed}: 日志中未找到 Validation 行 ({log})")
    psnr, ssim, lpips = (float(x) for x in m[-1])
    return {'seed': seed, 'psnr': psnr, 'ssim': ssim, 'lpips': lpips,
            'ckpt': f"{cfg['output_dir']}/best.pth", 'log': log}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--stop_epoch', type=int, default=5)
    p.add_argument('--out', required=True, help='结果 JSON 路径')
    p.add_argument('--out_root', default=None,
                   help='各 seed 的 output_dir 父目录; 默认由 --out 推导')
    args = p.parse_args()

    out_root = args.out_root or os.path.splitext(args.out)[0] + '_runs'
    os.makedirs(out_root, exist_ok=True)

    results = []
    for s in args.seeds:
        results.append(run_one(args.config, s, args.stop_epoch, out_root))

    psnrs = [r['psnr'] for r in results]
    ssims = [r['ssim'] for r in results]
    lpips = [r['lpips'] for r in results]
    summary = {
        'config': args.config,
        'seeds': args.seeds,
        'stop_epoch': args.stop_epoch,
        'per_seed': results,
        'psnr': {'mean': mean(psnrs), 'std': stdev(psnrs) if len(psnrs) > 1 else 0.0,
                 'min': min(psnrs), 'max': max(psnrs)},
        'ssim': {'mean': mean(ssims), 'std': stdev(ssims) if len(ssims) > 1 else 0.0},
        'lpips': {'mean': mean(lpips), 'std': stdev(lpips) if len(lpips) > 1 else 0.0},
    }
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"多 seed 基线: {os.path.basename(args.config)}")
    print("=" * 60)
    for r in results:
        print(f"  seed {r['seed']}: PSNR={r['psnr']:.4f}  SSIM={r['ssim']:.4f}  LPIPS={r['lpips']:.4f}")
    ps = summary['psnr']
    print(f"  ── PSNR 均值 {ps['mean']:.4f} ± {ps['std']:.4f} (min {ps['min']:.2f}, max {ps['max']:.2f})")
    print(f"  ── 2σ 容差门槛: ±{2*ps['std']:.4f} dB")
    print(f"\n  判据建议: 两配置差异 > {2*ps['std']:.2f} dB 才视为真实效应")
    print(f"  结果已写入 {args.out}")


if __name__ == '__main__':
    main()
