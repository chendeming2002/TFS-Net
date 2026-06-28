#!/usr/bin/env python3
"""
RWKV-Bravo 训练/推理脚本
========================
整合: 训练循环 + 推理 (tiled_forward) + 指标 (PSNR, SSIM) + IO + 日志
"""

import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

try:
    from tqdm import tqdm
except Exception:
    class _TqdmFallback:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix(self, **kwargs):
            pass
    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable, *args, **kwargs)

from datasets import SDSDDataset
from losses import TFSNetLoss, ssim_map
from models import TFSNet


# =====================================================================
# Utils: AverageMeter, 日志, 随机种子, IO
# =====================================================================

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("tfs_net")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(save_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state, path):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def save_image_tensor(tensor, path):
    from PIL import Image
    ensure_dir(os.path.dirname(path))
    image = tensor.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(path)


# =====================================================================
# Metrics: PSNR / SSIM
# =====================================================================

def tensor_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def tensor_ssim(pred, target):
    return ssim_map(pred, target).mean().item()


# =====================================================================
# Inference: tiled_forward
# =====================================================================

def _compute_starts(length, tile_size, overlap):
    if tile_size >= length:
        return [0]
    stride = max(tile_size - overlap, 1)
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _pad_clip_for_tiling(clip, tile_size):
    _, _, _, h, w = clip.shape
    pad_h = max(tile_size - h, 0)
    pad_w = max(tile_size - w, 0)
    if pad_h == 0 and pad_w == 0:
        return clip, (0, 0), (h, w)
    clip = F.pad(clip, (0, pad_w, 0, pad_h), mode="reflect")
    return clip, (pad_h, pad_w), (h, w)


@torch.no_grad()
def tiled_forward(model, clip, tile_size=256, tile_overlap=32, use_amp=False, frame_indices=None):
    if tile_size is None or tile_size <= 0:
        with autocast(enabled=use_amp and clip.is_cuda):
            return model(clip, frame_indices=frame_indices)["res_t"]

    clip, pad_hw, original_hw = _pad_clip_for_tiling(clip, tile_size)
    b, t, c, h, w = clip.shape
    if b != 1:
        raise ValueError("tiled_forward expects batch size 1.")

    h_starts = _compute_starts(h, tile_size, tile_overlap)
    w_starts = _compute_starts(w, tile_size, tile_overlap)

    use_cache = (len(h_starts) == 1 and len(w_starts) == 1)
    cache_indices = frame_indices if use_cache else None

    output = clip.new_zeros((b, c, h, w))
    weight = clip.new_zeros((b, 1, h, w))

    for top in h_starts:
        for left in w_starts:
            tile = clip[:, :, :, top: top + tile_size, left: left + tile_size]
            with autocast(enabled=use_amp and clip.is_cuda):
                tile_pred = model(tile, frame_indices=cache_indices)["res_t"]
            output[:, :, top: top + tile_size, left: left + tile_size] += tile_pred
            weight[:, :, top: top + tile_size, left: left + tile_size] += 1.0

    output = output / weight.clamp_min(1.0)

    pad_h, pad_w = pad_hw
    orig_h, orig_w = original_hw
    if pad_h > 0:
        output = output[:, :, :orig_h, :]
    if pad_w > 0:
        output = output[:, :, :, :orig_w]
    return output


# =====================================================================
# Data
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg, smoke=False):
    ds_cfg = cfg["dataset"]
    max_train = ds_cfg.get("max_train_seqs", None)
    max_val = ds_cfg.get("max_val_seqs", None)
    train_set = SDSDDataset(
        input_root=ds_cfg["train_input_root"], target_root=ds_cfg["train_target_root"],
        window_size=ds_cfg["window_size"], mode="train",
        crop_size=ds_cfg["crop_size"], max_seqs=max_train,
    )
    val_set = SDSDDataset(
        input_root=ds_cfg["val_input_root"], target_root=ds_cfg["val_target_root"],
        window_size=ds_cfg["window_size"], mode="val",
        crop_size=ds_cfg["crop_size"], max_seqs=max_val,
    )
    if smoke:
        train_set = Subset(train_set, list(range(min(8, len(train_set)))))
        val_set = Subset(val_set, list(range(min(4, len(val_set)))))
    else:
        max_val_samples = ds_cfg.get("max_val_samples", None)
        if max_val_samples is not None and len(val_set) > max_val_samples:
            val_set = Subset(val_set, list(range(max_val_samples)))

    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=ds_cfg["num_workers"], pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=ds_cfg["num_workers"], pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader


def build_model(cfg, device):
    mcfg = cfg["model"]
    model = TFSNet(
        in_channels=mcfg["in_channels"],
        level_channels=tuple(mcfg["level_channels"]),
        fused_channels=mcfg["fused_channels"],
        use_soft_clamp=mcfg.get("use_soft_clamp", True),
        use_soft_median=mcfg.get("use_soft_median", True),
        use_nafblock=mcfg.get("use_nafblock", False),
        num_bottleneck_blocks=mcfg.get("num_bottleneck_blocks", 0),
        num_igrf_res_blocks=mcfg.get("num_igrf_res_blocks", 2),
    )
    return model.to(device)


def build_loss(cfg, device):
    lcfg = cfg["loss"]
    criterion = TFSNetLoss(
        use_freq_loss=lcfg.get("use_freq_loss", True),
        perceptual_pretrained=lcfg.get("perceptual_pretrained", False),
        lambda_pix=lcfg.get("lambda_pix", 1.0),
        lambda_perc=lcfg.get("lambda_perc", 0.1),
        lambda_freq=lcfg.get("lambda_freq", 0.1),
        lambda_illum=lcfg.get("lambda_illum", 0.001),
        lambda_ssim=lcfg.get("lambda_ssim", 0.2),
        lambda_inter=lcfg.get("lambda_inter", 0.3),
        lambda_illum_sup=lcfg.get("lambda_illum_sup", 0.1),
        lambda_noise_sup=lcfg.get("lambda_noise_sup", 0.05),
        noise_tau_high=lcfg.get("noise_tau_high", 5.0),
        perc_multilayer=lcfg.get("perc_multilayer", True),
        freq_with_phase=lcfg.get("freq_with_phase", True),
        freq_phase_weight=lcfg.get("freq_phase_weight", 0.5),
        lambda_ifpn_sup=lcfg.get("lambda_ifpn_sup", 0.0),
    )
    return criterion.to(device)


# =====================================================================
# Training / Validation
# =====================================================================

def train_one_epoch(model, criterion, optimizer, scaler, loader, device, use_amp, logger, log_interval, grad_clip=1.0):
    model.train()
    meters = {
        k: AverageMeter() for k in [
            "loss_total", "loss_pix", "loss_freq",
            "loss_ssim", "loss_perc", "loss_illum", "loss_illum_sup",
            "loss_noise_sup", "loss_inter", "loss_ifpn_sup",
        ]
    }

    progress = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
    for step, (clip, target, _) in progress:
        clip = clip.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(clip)
        loss, loss_dict = criterion(outputs, target)

        if not torch.isfinite(loss):
            logger.warning("Non-finite loss at step %d, skipping", step + 1)
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        for k, m in meters.items():
            m.update(loss_dict.get(k, loss.detach()).item(), clip.size(0))

        progress.set_postfix(loss=meters["loss_total"].avg, pix=meters["loss_pix"].avg,
                             ssim=meters["loss_ssim"].avg,
                             i_sup=meters["loss_illum_sup"].avg, n_sup=meters["loss_noise_sup"].avg)

        if (step + 1) % log_interval == 0:
            log_parts = [f"step {step+1}/{len(loader)}"]
            for k, m in meters.items():
                log_parts.append(f"{k}={m.avg:.4f}")
            logger.info(" ".join(log_parts))

    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def validate(model, loader, device, tile_size, tile_overlap, use_amp, val_crop_size=None):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    loss_meter = AverageMeter()
    for clip, target, _ in tqdm(loader, total=len(loader), desc="val", leave=False):
        clip = clip.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if val_crop_size is not None:
            _, _, _, h, w = clip.shape
            cs = min(val_crop_size, h, w)
            top = (h - cs) // 2
            left = (w - cs) // 2
            clip = clip[:, :, :, top:top + cs, left:left + cs]
            target = target[:, :, top:top + cs, left:left + cs]
        pred = tiled_forward(model=model, clip=clip, tile_size=tile_size,
                             tile_overlap=tile_overlap, use_amp=use_amp)
        loss = torch.mean(torch.abs(pred - target))
        psnr_meter.update(tensor_psnr(pred, target), clip.size(0))
        ssim_meter.update(tensor_ssim(pred, target), clip.size(0))
        loss_meter.update(loss.item(), clip.size(0))
    return {"val_l1": loss_meter.avg, "psnr": psnr_meter.avg, "ssim": ssim_meter.avg}


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir)
    logger.info("Loading config from %s", args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_loader, val_loader = build_dataloaders(cfg, smoke=args.smoke)
    model = build_model(cfg, device)
    criterion = build_loss(cfg, device)

    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    warmup_epochs = cfg["train"].get("warmup_epochs", 5)
    total_epochs = cfg["train"]["epochs"] if not args.smoke else 1
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1))
    if warmup_epochs > 0 and total_epochs > warmup_epochs:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = cosine_scheduler

    scaler = GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")
    grad_clip = cfg["train"].get("grad_clip", 1.0)

    best_psnr = -1.0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt["model"]
        remapped = {}
        for k, v in sd.items():
            parts = k.split(".")
            remapped_key = k
            for i in range(len(parts) - 2):
                if parts[i] == "fuse" and parts[i + 2] == "0":
                    remapped_key = ".".join(parts[:i + 2] + parts[i + 3:])
                    break
            remapped[remapped_key] = v
        filtered = {}
        for k, v in remapped.items():
            if k in model.state_dict():
                if model.state_dict()[k].shape != v.shape:
                    logger.warning("Size-mismatch %s: ckpt %s vs model %s, skipping",
                                   k, list(v.shape), list(model.state_dict()[k].shape))
                    continue
            filtered[k] = v
        model.load_state_dict(filtered, strict=False)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                logger.warning("Optimizer load failed")
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                logger.warning("Scheduler load failed")
        start_epoch = ckpt["epoch"]
        best_psnr = ckpt.get("best_psnr", -1.0)
        logger.info("Resumed from %s (epoch %d, best_psnr=%.4f)", args.resume, start_epoch, best_psnr)

    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        sd = ckpt["model"]
        remapped = {}
        for k, v in sd.items():
            parts = k.split(".")
            remapped_key = k
            for i in range(len(parts) - 2):
                if parts[i] == "fuse" and parts[i + 2] == "0":
                    remapped_key = ".".join(parts[:i + 2] + parts[i + 3:])
                    break
            remapped[remapped_key] = v
        filtered = {}
        for k, v in remapped.items():
            if k in model.state_dict():
                if model.state_dict()[k].shape != v.shape:
                    logger.warning("Size-mismatch %s, skipping", k)
                    continue
            filtered[k] = v
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        logger.info("Pretrained from %s (missing=%d, unexpected=%d)", args.pretrained, len(missing), len(unexpected))

    for epoch in range(start_epoch, total_epochs):
        logger.info("Epoch %d / %d", epoch + 1, total_epochs)
        train_stats = train_one_epoch(
            model, criterion, optimizer, scaler, train_loader, device,
            use_amp=cfg["train"]["amp"] and device.type == "cuda",
            logger=logger, log_interval=cfg["train"]["log_interval"], grad_clip=grad_clip,
        )
        scheduler.step()
        logger.info("Train: %s", train_stats)

        if (epoch + 1) % cfg["train"]["val_interval"] == 0:
            val_stats = validate(
                model, val_loader, device,
                tile_size=cfg["eval"]["tile_size"],
                tile_overlap=cfg["eval"]["tile_overlap"],
                use_amp=cfg["eval"]["amp"] and device.type == "cuda",
                val_crop_size=cfg["dataset"].get("val_crop_size", None),
            )
            logger.info("Val: %s", val_stats)
            save_checkpoint(
                {"epoch": epoch + 1, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "config": cfg},
                os.path.join(output_dir, "latest.pth"),
            )
            if val_stats["psnr"] > best_psnr:
                best_psnr = val_stats["psnr"]
                save_checkpoint(
                    {"epoch": epoch + 1, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "config": cfg, "best_psnr": best_psnr},
                    os.path.join(output_dir, "best.pth"),
                )
        if args.smoke:
            break


if __name__ == "__main__":
    main()
