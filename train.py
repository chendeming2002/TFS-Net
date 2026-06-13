import argparse
import os

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

try:
    from tqdm import tqdm
except Exception:
    class _TqdmFallback(object):
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable, *args, **kwargs)

from datasets import SDSDDataset
from losses.losses import TFSNetLoss
from models import TFSNet
from utils.io import save_checkpoint
from utils.inference import tiled_forward
from utils.metrics import tensor_psnr, tensor_ssim
from utils.misc import AverageMeter, create_logger, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg, smoke=False):
    train_set = SDSDDataset(
        input_root=cfg["dataset"]["train_input_root"],
        target_root=cfg["dataset"]["train_target_root"],
        window_size=cfg["dataset"]["window_size"],
        mode="train",
        crop_size=cfg["dataset"]["crop_size"],
    )
    val_set = SDSDDataset(
        input_root=cfg["dataset"]["val_input_root"],
        target_root=cfg["dataset"]["val_target_root"],
        window_size=cfg["dataset"]["window_size"],
        mode="val",
        crop_size=cfg["dataset"]["crop_size"],
    )

    if smoke:
        train_set = Subset(train_set, list(range(min(8, len(train_set)))))
        val_set = Subset(val_set, list(range(min(4, len(val_set)))))

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def build_model(cfg, device):
    model = TFSNet(
        in_channels=cfg["model"]["in_channels"],
        level_channels=tuple(cfg["model"]["level_channels"]),
        fused_channels=cfg["model"]["fused_channels"],
    )
    return model.to(device)


def build_loss(cfg, device):
    loss_cfg = cfg["loss"]
    criterion = TFSNetLoss(
        use_freq_loss=loss_cfg.get("use_freq_loss", True),
        perceptual_pretrained=loss_cfg.get("perceptual_pretrained", False),
        lambda_perc=loss_cfg.get("lambda_perc", 0.1),
        lambda_freq=loss_cfg.get("lambda_freq", 0.1),
        lambda_illum=loss_cfg.get("lambda_illum", 0.01),
        lambda_ssim=loss_cfg.get("lambda_ssim", 0.2),
        lambda_aux=loss_cfg.get("lambda_aux", 0.2),
        fused_channels=loss_cfg.get("fused_channels", 64),
    )
    return criterion.to(device)


def train_one_epoch(model, criterion, optimizer, scaler, loader, device, use_amp, logger, log_interval, grad_clip=1.0):
    model.train()
    meter_total = AverageMeter()
    meter_pix = AverageMeter()
    meter_freq = AverageMeter()
    meter_ssim = AverageMeter()
    meter_perc = AverageMeter()
    meter_illum = AverageMeter()
    meter_aux = AverageMeter()

    progress = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
    for step, (clip, target, _) in progress:
        clip = clip.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(clip)
            loss, loss_dict = criterion(outputs, target)

        if not torch.isfinite(loss):
            logger.warning("Skipping non-finite loss at step %d", step + 1)
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        meter_total.update(loss_dict["loss_total"].item(), clip.size(0))
        meter_pix.update(loss_dict["loss_pix"].item(), clip.size(0))
        meter_freq.update(loss_dict["loss_freq"].item(), clip.size(0))
        meter_ssim.update(loss_dict["loss_ssim"].item(), clip.size(0))
        meter_perc.update(loss_dict["loss_perc"].item(), clip.size(0))
        meter_illum.update(loss_dict["loss_illum"].item(), clip.size(0))
        meter_aux.update(loss_dict["loss_aux"].item(), clip.size(0))

        progress.set_postfix(loss=meter_total.avg, pix=meter_pix.avg, ssim=meter_ssim.avg)
        if (step + 1) % log_interval == 0:
            logger.info(
                "step %d/%d loss=%.4f pix=%.4f freq=%.4f ssim=%.4f perc=%.4f illum=%.4f aux=%.4f",
                step + 1,
                len(loader),
                meter_total.avg,
                meter_pix.avg,
                meter_freq.avg,
                meter_ssim.avg,
                meter_perc.avg,
                meter_illum.avg,
                meter_aux.avg,
            )

    return {
        "loss_total": meter_total.avg,
        "loss_pix": meter_pix.avg,
        "loss_freq": meter_freq.avg,
        "loss_ssim": meter_ssim.avg,
        "loss_perc": meter_perc.avg,
        "loss_illum": meter_illum.avg,
        "loss_aux": meter_aux.avg,
    }


@torch.no_grad()
def validate(model, loader, device, tile_size, tile_overlap, use_amp):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    loss_meter = AverageMeter()
    for clip, target, _ in tqdm(loader, total=len(loader), desc="val", leave=False):
        clip = clip.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = tiled_forward(
            model=model,
            clip=clip,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            use_amp=use_amp,
        )
        loss = torch.mean(torch.abs(pred - target))
        psnr_meter.update(tensor_psnr(pred, target), clip.size(0))
        ssim_meter.update(tensor_ssim(pred, target), clip.size(0))
        loss_meter.update(loss.item(), clip.size(0))
        del clip, target, pred, loss
    return {"val_l1": loss_meter.avg, "psnr": psnr_meter.avg, "ssim": ssim_meter.avg}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir)
    logger.info("Loading config from %s", args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

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
    for epoch in range(total_epochs):
        logger.info("Epoch %d / %d", epoch + 1, total_epochs)
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            loader=train_loader,
            device=device,
            use_amp=cfg["train"]["amp"] and device.type == "cuda",
            logger=logger,
            log_interval=cfg["train"]["log_interval"],
            grad_clip=grad_clip,
        )
        scheduler.step()
        logger.info("Train stats: %s", train_stats)

        if (epoch + 1) % cfg["train"]["val_interval"] == 0:
            val_stats = validate(
                model,
                val_loader,
                device,
                tile_size=cfg["eval"]["tile_size"],
                tile_overlap=cfg["eval"]["tile_overlap"],
                use_amp=cfg["eval"]["amp"] and device.type == "cuda",
            )
            logger.info("Val stats: %s", val_stats)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": cfg,
                },
                os.path.join(output_dir, "latest.pth"),
            )
            if val_stats["psnr"] > best_psnr:
                best_psnr = val_stats["psnr"]
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": cfg,
                        "best_psnr": best_psnr,
                    },
                    os.path.join(output_dir, "best.pth"),
                )
        if args.smoke:
            break


if __name__ == "__main__":
    main()
