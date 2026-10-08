#!/usr/bin/env python3
"""Golf v7r-v3 训练脚本: Matrix RWKV + Triple Query TCA + 多元噪声分割"""
import argparse
import os
import sys
import time
import yaml
import torch
from torch.utils.data import DataLoader

from datasets import SDSDDataset
from models.golf_v7r import GolfNet_v7r_v3, GolfV7RLoss
from utils.io import save_checkpoint
from utils.misc import seed_everything
from utils.metrics import tensor_psnr, tensor_ssim, LPIPSMetric
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stop_epoch', type=int, default=None,
                        help='仅用于快速验证: 训练到该 epoch 即停, 不改变 cosine 调度 (T_max 仍为配置值)')
    parser.add_argument('--resume', default=None, help='覆盖 config 中的 resume 路径')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Seed (含 Python random / numpy: 数据增强用 random 模块, 必须一起播种才能复现)
    seed = cfg.get('seed', 42)
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataset
    pairing = cfg['dataset'].get('pairing', 'name')
    exclude_seqs = cfg['dataset'].get('exclude_seqs') or None
    val_holdout_seqs = cfg['dataset'].get('val_holdout_seqs')
    logger.info(f"Dataset pairing mode: {pairing}")
    if exclude_seqs:
        logger.info(f"Train exclude_seqs (hold-out protocol): {exclude_seqs}")
    train_ds = SDSDDataset(
        input_root=cfg['dataset']['train_input_root'],
        target_root=cfg['dataset']['train_target_root'],
        window_size=cfg['dataset']['window_size'],
        mode="train",
        crop_size=cfg['dataset']['crop_size'],
        pairing=pairing,
        exclude_seqs=exclude_seqs,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg['train']['batch_size'], shuffle=True,
        num_workers=cfg['dataset'].get('num_workers', 0),
        pin_memory=False, drop_last=True,
    )
    logger.info(f"Train dataset: {len(train_ds)} samples, {len(train_loader)} batches")

    # 验证集: 默认用 test/low-light (与训练帧级重叠 -> 泄漏, §6.4);
    # 若配置 val_holdout_seqs, 则改用被排除的序列作为【干净留出集】(帧级不相交)。
    if val_holdout_seqs:
        logger.info(f"Val = hold-out sequences from train_root: {val_holdout_seqs}")
        val_ds = SDSDDataset(
            input_root=cfg['dataset']['train_input_root'],
            target_root=cfg['dataset']['train_target_root'],
            window_size=cfg['dataset']['window_size'],
            mode="val", crop_size=None,
            pairing=pairing,
            max_seqs=None,
        )
        # 仅保留留出序列的样本 (帧级与训练不相交)
        val_ds.samples = [s for s in val_ds.samples if s['sequence'] in set(val_holdout_seqs)]
        if not val_ds.samples:
            raise RuntimeError(f"val_holdout_seqs {val_holdout_seqs} matched no samples")
        logger.info(f"Val(hold-out) samples: {len(val_ds.samples)} "
                    f"(seqs: {sorted(set(s['sequence'] for s in val_ds.samples))})")
    else:
        val_ds = SDSDDataset(
            input_root=cfg['dataset']['val_input_root'],
            target_root=cfg['dataset']['val_target_root'],
            window_size=cfg['dataset']['window_size'],
            mode="val", crop_size=None,
            pairing=pairing,
        )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=cfg['dataset'].get('num_workers', 0), pin_memory=False,
    )
    logger.info(f"Val dataset: {len(val_ds)} samples")

    # Model
    model = GolfNet_v7r_v3(**cfg['model']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"GolfNet_v7r_v3 params: {n_params/1e6:.2f}M")

    # Loss & Optimizer
    criterion = GolfV7RLoss(**cfg.get('loss', {})).to(device)
    lr = cfg['train']['lr']
    wd = cfg['train'].get('weight_decay', 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    num_epochs = cfg['train']['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    out_dir = cfg['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    log_interval = cfg['train'].get('log_interval', 100)
    val_interval = cfg['train'].get('val_interval', 5)
    grad_clip = cfg['train'].get('grad_clip', 0.0)

    # LPIPS 感知指标 (评估时使用)
    lpips_metric = LPIPSMetric(net=cfg['train'].get('lpips_net', 'vgg'),
                               device=device)
    logger.info(f"LPIPS available: {lpips_metric.available}")

    # 断点续训
    start_epoch = 1
    best_psnr = 0.0
    resume = args.resume if args.resume is not None else cfg['train'].get('resume')
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        if 'optimizer_state_dict' in ck:
            optimizer.load_state_dict(ck['optimizer_state_dict'])
        if 'scheduler_state_dict' in ck:
            scheduler.load_state_dict(ck['scheduler_state_dict'])
        start_epoch = ck.get('epoch', 0) + 1
        best_psnr = ck.get('metrics', {}).get('psnr', 0.0) or 0.0
        # latest.pth 无 metrics 字段: 允许从 config 显式传入历史 best, 避免误覆盖
        if best_psnr == 0.0:
            best_psnr = cfg['train'].get('resume_best_psnr', 0.0) or 0.0
        logger.info(f"Resumed from {resume} (epoch {ck.get('epoch')}), "
                    f"start at epoch {start_epoch}, best_psnr={best_psnr:.2f}")

    for epoch in range(start_epoch, num_epochs + 1):
        if args.stop_epoch is not None and epoch > args.stop_epoch:
            logger.info(f"Reached --stop_epoch {args.stop_epoch}, stopping early.")
            break
        criterion.set_epoch(epoch - 1)
        logger.info("=" * 60)
        logger.info(f"Epoch {epoch}/{num_epochs} - lr={optimizer.param_groups[0]['lr']:.2e}")
        logger.info("=" * 60)

        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_steps = 0
        for step, batch in enumerate(train_loader):
            lq, gt, meta = batch
            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            output_dict = model(lq)
            loss_dict = criterion(output_dict, gt)
            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

            if (step + 1) % log_interval == 0:
                msg = f"  Step {step+1}/{len(train_loader)} - Loss: {loss.item():.4f}"
                if 'inject_stat' in output_dict:
                    gk = output_dict['inject_stat']['gate_k']
                    msg += (f" | gate={output_dict['inject_stat']['gate'].item():.4f}"
                            f" (N/L/M={gk[0].item():.3f}/"
                            f"{gk[1].item():.3f}/{gk[2].item():.3f})")
                logger.info(msg)

        scheduler.step()
        dt = time.time() - t0
        logger.info(f"Epoch {epoch} done in {dt/60:.1f} min, avg loss: {epoch_loss/max(n_steps,1):.4f}")

        # Save latest
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        save_checkpoint(state, os.path.join(out_dir, 'latest.pth'))

        # Validate
        if epoch % val_interval == 0:
            model.eval()
            psnr_list, ssim_list, lpips_list = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    lq, gt, meta = batch
                    lq = lq.to(device, non_blocking=True)
                    gt = gt.to(device, non_blocking=True)
                    pred = model(lq)['final']
                    psnr_list.append(tensor_psnr(pred, gt))
                    ssim_list.append(tensor_ssim(pred, gt))
                    lp = lpips_metric(pred, gt)
                    if lp is not None:
                        lpips_list.append(lp)

            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)
            avg_lpips = (sum(lpips_list) / len(lpips_list)) if lpips_list else None
            msg = f"Validation - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}"
            if avg_lpips is not None:
                msg += f", LPIPS: {avg_lpips:.4f}"
            logger.info(msg)

            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': {'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips},
            }
            save_checkpoint(state, os.path.join(out_dir, f'epoch_{epoch:03d}.pth'))
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                save_checkpoint(state, os.path.join(out_dir, 'best.pth'))
                logger.info(f"  New best PSNR: {best_psnr:.2f} dB")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
