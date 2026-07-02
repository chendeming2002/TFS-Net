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
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    parser.add_argument("--pretrained", type=str, default=None, help="pretrained weights to init model only")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg, smoke=False):
    ds_cfg = cfg["dataset"]
    max_train = ds_cfg.get("max_train_seqs", None)
    max_val = ds_cfg.get("max_val_seqs", None)
    train_set = SDSDDataset(
        input_root=ds_cfg["train_input_root"],
        target_root=ds_cfg["train_target_root"],
        window_size=ds_cfg["window_size"],
        mode="train",
        crop_size=ds_cfg["crop_size"],
        max_seqs=max_train,
    )
    val_set = SDSDDataset(
        input_root=ds_cfg["val_input_root"],
        target_root=ds_cfg["val_target_root"],
        window_size=ds_cfg["window_size"],
        mode="val",
        crop_size=ds_cfg["crop_size"],
        max_seqs=max_val,
    )

    if smoke:
        train_set = Subset(train_set, list(range(min(8, len(train_set)))))
        val_set = Subset(val_set, list(range(min(4, len(val_set)))))
    else:
        # 简短训练: 限制 validation 帧数避免 CPU 推理耗时过长
        max_val_samples = ds_cfg.get("max_val_samples", None)
        if max_val_samples is not None and len(val_set) > max_val_samples:
            val_set = Subset(val_set, list(range(max_val_samples)))

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
    model_cfg = cfg["model"]
    model = TFSNet(
        in_channels=model_cfg["in_channels"],
        level_channels=tuple(model_cfg["level_channels"]),
        fused_channels=model_cfg["fused_channels"],
        share_lff=model_cfg.get("share_lff", True),
        sace_phase_preserving=model_cfg.get("sace_phase_preserving", True),
        use_soft_clamp=model_cfg.get("use_soft_clamp", True),
        sace_offset_use_norm=model_cfg.get("sace_offset_use_norm", False),
        sace_offset_kaiming_init=model_cfg.get("sace_offset_kaiming_init", False),
        use_soft_median=model_cfg.get("use_soft_median", True),
        use_cross_rwkv=model_cfg.get("use_cross_rwkv", False),
        use_dwt_lff=model_cfg.get("use_dwt_lff", False),
        use_pure_rwkv=model_cfg.get("use_pure_rwkv", False),
        use_nafblock=model_cfg.get("use_nafblock", False),
        num_bottleneck_blocks=model_cfg.get("num_bottleneck_blocks", 0),
        num_igrf_res_blocks=model_cfg.get("num_igrf_res_blocks", 2),
        use_amp_enhance=model_cfg.get("use_amp_enhance", False),
        charlie_mode=model_cfg.get("charlie_mode", False),
    )
    return model.to(device)


def build_loss(cfg, device):
    loss_cfg = cfg["loss"]
    criterion = TFSNetLoss(
        use_freq_loss=loss_cfg.get("use_freq_loss", True),
        perceptual_pretrained=loss_cfg.get("perceptual_pretrained", False),
        lambda_perc=loss_cfg.get("lambda_perc", 0.1),
        lambda_freq=loss_cfg.get("lambda_freq", 0.1),
        lambda_illum=loss_cfg.get("lambda_illum", 0.001),
        lambda_ssim=loss_cfg.get("lambda_ssim", 0.2),
        lambda_inter=loss_cfg.get("lambda_inter", 0.3),
        fused_channels=loss_cfg.get("fused_channels", 64),
        # v5.6 P0-2/P0-3
        lambda_illum_sup=loss_cfg.get("lambda_illum_sup", 0.1),
        lambda_noise_sup=loss_cfg.get("lambda_noise_sup", 0.05),
        lambda_recon=loss_cfg.get("lambda_recon", 0.5),
        noise_tau_high=loss_cfg.get("noise_tau_high", 5.0),
        # v5.6 P1-3/P1-4
        perc_multilayer=loss_cfg.get("perc_multilayer", True),
        freq_with_phase=loss_cfg.get("freq_with_phase", True),
        freq_phase_weight=loss_cfg.get("freq_phase_weight", 0.5),
        # v5.9.2
        lambda_ifpn_sup=loss_cfg.get("lambda_ifpn_sup", 0.0),
        # v6 Bravo P0-1
        lambda_pix=loss_cfg.get("lambda_pix", 1.0),
    )
    return criterion.to(device)


def train_one_epoch(model, criterion, optimizer, scaler, loader, device, use_amp, logger, log_interval, epoch=0, grad_clip=1.0):
    model.train()
    meter_total = AverageMeter()
    meter_pix = AverageMeter()
    meter_freq = AverageMeter()
    meter_ssim = AverageMeter()
    meter_perc = AverageMeter()
    meter_illum = AverageMeter()
    meter_illum_sup = AverageMeter()
    meter_noise_sup = AverageMeter()
    meter_inter = AverageMeter()
    meter_ifpn = AverageMeter()

    progress = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
    for step, (clip, target, _) in progress:
        clip = clip.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(clip)
        # 损失计算在 fp32（autocast 外），避免 SSIM/VGG 的 fp16 精度问题（§8.4 修复 A）
        loss, loss_dict = criterion(outputs, target, epoch=epoch)

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
        meter_illum_sup.update(loss_dict["loss_illum_sup"].item(), clip.size(0))
        meter_noise_sup.update(loss_dict["loss_noise_sup"].item(), clip.size(0))
        meter_inter.update(loss_dict["loss_inter"].item(), clip.size(0))
        meter_ifpn.update(loss_dict["loss_ifpn_sup"].item(), clip.size(0))

        progress.set_postfix(loss=meter_total.avg, pix=meter_pix.avg, ssim=meter_ssim.avg,
                             i_sup=meter_illum_sup.avg, n_sup=meter_noise_sup.avg)
        if (step + 1) % log_interval == 0:
            logger.info(
                "step %d/%d loss=%.4f pix=%.4f freq=%.4f ssim=%.4f perc=%.4f illum=%.4f i_sup=%.4f n_sup=%.4f inter=%.4f ifpn=%.4f",
                step + 1,
                len(loader),
                meter_total.avg,
                meter_pix.avg,
                meter_freq.avg,
                meter_ssim.avg,
                meter_perc.avg,
                meter_illum.avg,
                meter_illum_sup.avg,
                meter_noise_sup.avg,
                meter_inter.avg,
                meter_ifpn.avg,
            )

    return {
        "loss_total": meter_total.avg,
        "loss_pix": meter_pix.avg,
        "loss_freq": meter_freq.avg,
        "loss_ssim": meter_ssim.avg,
        "loss_perc": meter_perc.avg,
        "loss_illum": meter_illum.avg,
        "loss_illum_sup": meter_illum_sup.avg,
        "loss_noise_sup": meter_noise_sup.avg,
        "loss_inter": meter_inter.avg,
        "loss_ifpn_sup": meter_ifpn.avg,
    }


@torch.no_grad()
def validate(model, loader, device, tile_size, tile_overlap, use_amp, val_crop_size=None):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    loss_meter = AverageMeter()
    for clip, target, _ in tqdm(loader, total=len(loader), desc="val", leave=False):
        clip = clip.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # 简短训练/消融: 中心裁剪到小尺寸避免 CPU 全分辨率推理耗时过长
        if val_crop_size is not None:
            _, _, _, h, w = clip.shape
            cs = min(val_crop_size, h, w)
            top = (h - cs) // 2
            left = (w - cs) // 2
            clip = clip[:, :, :, top:top + cs, left:left + cs]
            target = target[:, :, top:top + cs, left:left + cs]
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
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt["model"]
        # Remap: 旧 checkpoint 用 Sequential 包裹 ResBlock (fuse.N.0.conv1 → fuse.N.conv1)
        remapped = {}
        for k, v in sd.items():
            parts = k.split(".")
            remapped_key = k
            for i in range(len(parts) - 2):
                if parts[i] == "fuse" and parts[i + 2] == "0":
                    remapped_key = ".".join(parts[:i + 2] + parts[i + 3:])
                    break
            remapped[remapped_key] = v
        # Filter size-mismatched keys (for architecture changes like phase_conf)
        filtered = {}
        for k, v in remapped.items():
            if k in model.state_dict():
                if model.state_dict()[k].shape != v.shape:
                    logger.warning("Skipping size-mismatched key: %s (ckpt %s != model %s)", 
                                   k, list(v.shape), list(model.state_dict()[k].shape))
                    continue
            filtered[k] = v
        model.load_state_dict(filtered, strict=False)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                logger.warning("Optimizer state load failed, starting fresh optimizer")
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                logger.warning("Scheduler state load failed, starting fresh scheduler")
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
        # Filter size-mismatched keys
        filtered = {}
        for k, v in remapped.items():
            if k in model.state_dict():
                if model.state_dict()[k].shape != v.shape:
                    logger.warning("Skipping size-mismatched key: %s", k)
                    continue
            filtered[k] = v
        missing_keys, unexpected = model.load_state_dict(filtered, strict=False)
        logger.info("Loaded pretrained weights from %s (missing=%d, unexpected=%d)", args.pretrained, len(missing_keys), len(unexpected))

    for epoch in range(start_epoch, total_epochs):
        logger.info("Epoch %d / %d", epoch + 1, total_epochs)
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            loader=train_loader,
            device=device,
            use_amp=cfg["train"]["amp"] and device.type == "cuda",
            epoch=epoch,
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
                val_crop_size=cfg["dataset"].get("val_crop_size", None),
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
