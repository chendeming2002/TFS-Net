"""
训练+推理脚本 — RWKV-Charlie
============================
合并自原始 train.py + inference.py + utils.py

Charlie 特有:
  - sigma_t 传入 MRPN
  - warmup_epochs=5
  - max_lr=2e-4
"""

import os
import sys
import math
import time
import glob
import random
import logging
from typing import Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import TFSNet
from losses import TFSNetLoss
from datasets import create_sdsd_dataloader, SDSDDataset, VideoToTensor, VideoRandomCrop, VideoRandomFlip, TestTimeAug

logger = logging.getLogger("charlie")


def warp(img, flow):
    B, _, H, W = flow.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=flow.device), torch.arange(W, device=flow.device), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).float().unsqueeze(0).repeat(B, 1, 1, 1)
    grid = grid + flow.permute(0, 2, 3, 1)
    grid = grid * 2 / torch.tensor([W - 1, H - 1], device=flow.device) - 1
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


def create_optimizer(model, lr=2e-4, weight_decay=1e-4):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'norm' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=lr)


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class Config:
    num_frames = 5
    width = 32
    batch_size = 4
    num_workers = 4
    max_epochs = 300
    warmup_epochs = 5
    max_lr = 2e-4
    min_lr = 1e-6
    weight_decay = 1e-4
    patch_size = 128
    lambda_pix = 1.0
    lambda_perc = 0.04
    lambda_ssim = 0.2
    use_rwkv = True
    deep_supervision = False
    use_temporal_fusion = True


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, config, writer=None):
    model.train()
    loss_meter = AverageMeter()
    pix_meter = AverageMeter()
    perc_meter = AverageMeter()
    ssim_meter = AverageMeter()
    t0 = time.time()
    for step, batch in enumerate(dataloader):
        noisy = batch["noisy"].cuda()
        clean = batch["clean"].cuda()
        B, T, C, H, W = noisy.shape
        optimizer.zero_grad()
        pred = model(noisy)
        if isinstance(pred, list):
            ds_preds = pred[1:]
            pred = pred[0]
        else:
            ds_preds = None
        loss = criterion(pred, clean, ds_preds)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_meter.update(loss.item(), B)
        if step % 50 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch}/{config.max_epochs} | Step {step}/{len(dataloader)} | "
                f"Loss {loss_meter.avg:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
            )
            t0 = time.time()
    if writer:
        writer.add_scalar("train/loss", loss_meter.avg, epoch)


@torch.no_grad()
def validate(model, dataloader, epoch, config, writer=None):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    for batch in dataloader:
        noisy = batch["noisy"].cuda()
        clean = batch["clean"].cuda()
        B, T, C, H, W = noisy.shape
        pred = model(noisy)
        if isinstance(pred, list):
            pred = pred[0]
        pred_clip = pred.clamp(0, 1)
        for b in range(B):
            for t in range(T):
                p = pred_clip[b, t].unsqueeze(0)
                c = clean[b, t].unsqueeze(0)
                mse = F.mse_loss(p, c).item()
                psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
                psnr_meter.update(psnr)
    logger.info(f"Validation Epoch {epoch} | PSNR {psnr_meter.avg:.2f}")
    if writer:
        writer.add_scalar("val/psnr", psnr_meter.avg, epoch)


def test(model, test_loader, output_dir=None):
    model.eval()
    psnr_meter = AverageMeter()
    for batch in test_loader:
        noisy = batch["noisy"].cuda()
        clean = batch["clean"].cuda()
        pred = model(noisy)
        if isinstance(pred, list):
            pred = pred[0]
        pred_clip = pred.clamp(0, 1)
        B, T, C, H, W = pred_clip.shape
        for b in range(B):
            p = pred_clip[b].mean(dim=0, keepdim=True)
            c = clean[b].mean(dim=0, keepdim=True)
            mse = F.mse_loss(p, c).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
            psnr_meter.update(psnr)
    logger.info(f"Test PSNR {psnr_meter.avg:.2f}")
    return psnr_meter.avg


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    config = Config()
    num_gpus = torch.cuda.device_count()
    logger.info(f"Using {num_gpus} GPU(s)")

    model = TFSNet(
        in_channels=3,
        num_frames=config.num_frames,
        width=config.width,
        use_rwkv=config.use_rwkv,
        deep_supervision=config.deep_supervision,
        use_temporal_fusion=config.use_temporal_fusion,
    )
    model = model.cuda()
    if num_gpus > 1:
        model = nn.DataParallel(model)

    transform_train = [
        VideoToTensor(),
        VideoRandomCrop(config.patch_size),
        VideoRandomFlip(),
    ]
    transform_test = [VideoToTensor()]

    def compose(transforms):
        class Compose:
            def __call__(self, sample):
                for t in transforms:
                    sample = t(sample)
                return sample
        return Compose()

    train_loader = create_sdsd_dataloader(
        root_dir="/data/sdsd",
        split="train",
        num_frames=config.num_frames,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        transform=compose(transform_train),
    )

    val_loader = create_sdsd_dataloader(
        root_dir="/data/sdsd",
        split="val",
        num_frames=config.num_frames,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        transform=compose(transform_test),
    )

    criterion = TFSNetLoss(
        lambda_pix=config.lambda_pix,
        lambda_perc=config.lambda_perc,
        lambda_ssim=config.lambda_ssim,
    )
    optimizer = create_optimizer(model, config.max_lr, config.weight_decay)
    total_steps = len(train_loader) * config.max_epochs
    warmup_steps = len(train_loader) * config.warmup_epochs

    def warmup_cosine_lr(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lr)
    writer = SummaryWriter(log_dir="logs/charlie")

    best_psnr = 0
    for epoch in range(1, config.max_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, config, writer)
        if epoch % 5 == 0:
            validate(model, val_loader, epoch, config, writer)
            torch.save(model.state_dict(), f"checkpoints/charlie_epoch_{epoch}.pth")
        scheduler.step()

    logger.info("Training complete")


if __name__ == "__main__":
    main()
