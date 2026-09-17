"""Foxtrot (TSD-Net) 训练脚本 — 与 train.py 同构 (keepalive/monitor 兼容)

接口约定 (与既有基础设施对齐):
  - 日志格式: "step %d/%d loss=..." / "Epoch %d / %d [%s] lr=..." / "Val stats: {...}"
    → scripts/monitor.sh 直接可读
  - checkpoint: latest.pth / best.pth, 键 "epoch"(int) → scripts/keepalive_train.sh 的 DONE 判据可用
  - resume: --resume <latest.pth> (keepalive 自动传入)
  - val: tiled_forward + tensor_psnr/tensor_ssim + LPIPS (与历史 run 同协议)
"""
import argparse
import os

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
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
from models.golf import GolfNet
from models.golf.loss import GolfLoss
from utils.io import save_checkpoint
from utils.inference import tiled_forward
from utils.metrics import tensor_psnr, tensor_ssim
from utils.misc import AverageMeter, create_logger, seed_everything

_LPIPS_AVAILABLE = None
_LPIPS_FN = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
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
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "type"}
    model = GolfNet(**model_cfg)
    return model.to(device)


def build_loss(cfg, device):
    loss_cfg = {k: v for k, v in cfg["loss"].items() if k != "type"}
    return GolfLoss(**loss_cfg).to(device)


def train_one_epoch(model, criterion, optimizer, scaler, loader, device, use_amp,
                    logger, log_interval, epoch=0, grad_clip=1.0, grad_accum_steps=1):
    model.train()
    meter_total = AverageMeter()
    meter_final = AverageMeter()
    meter_bN = AverageMeter()
    meter_bL = AverageMeter()
    meter_bM = AverageMeter()
    meter_ortho = AverageMeter()
    meter_temp = AverageMeter()
    meter_div = AverageMeter()

    progress = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
    for step, (clip, target, _) in progress:
        clip = clip.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(clip)
        # 损失在 autocast 外计算 (与 train.py 模式一致, fp32 语义稳定)
        loss_dict = criterion(outputs, target)
        loss = loss_dict["total_loss"]

        if not torch.isfinite(loss):
            logger.warning("Skipping non-finite loss at step %d", step + 1)
            optimizer.zero_grad(set_to_none=True)
            del outputs, loss, loss_dict, clip, target
            continue

        loss_scaled = loss / grad_accum_steps
        scaler.scale(loss_scaled).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        meter_total.update(loss_dict["total_loss"].item(), clip.size(0))
        meter_final.update(loss_dict["L_final"], clip.size(0))
        meter_bN.update(loss_dict["L_N"], clip.size(0))
        meter_bL.update(loss_dict["L_L"], clip.size(0))
        meter_bM.update(loss_dict["L_M"], clip.size(0))
        meter_ortho.update(loss_dict["L_ortho"], clip.size(0))
        meter_temp.update(loss_dict["L_temp"], clip.size(0))
        meter_div.update(loss_dict["L_div"], clip.size(0))

        progress.set_postfix(loss=meter_total.avg, final=meter_final.avg,
                             div=meter_div.avg, ortho=meter_ortho.avg)
        if (step + 1) % log_interval == 0:
            logger.info(
                "step %d/%d loss=%.4f final=%.4f bN=%.4f bL=%.4f bM=%.4f ortho=%.4f temp=%.4f div=%.4f",
                step + 1, len(loader),
                meter_total.avg, meter_final.avg,
                meter_bN.avg, meter_bL.avg, meter_bM.avg, meter_ortho.avg,
                meter_temp.avg, meter_div.avg,
            )
            with torch.no_grad():
                w = outputs.get("fusion_weights")
                lt = outputs.get("L_t")
                cv = outputs.get("conf_map")
                w_std = w.std(dim=1).mean().item() if w is not None else 0.0
                lt_m = lt.mean().item() if lt is not None else 0.0
                lt_s = lt.std().item() if lt is not None else 0.0
                cv_m = cv.mean().item() if cv is not None else 0.0
                logger.info("diag: wstd=%.3f Lt=%.3f/%.3f conf=%.3f", w_std, lt_m, lt_s, cv_m)

        del outputs, loss, loss_dict, clip, target
        if (step + 1) % (grad_accum_steps * 50) == 0:
            torch.cuda.empty_cache()
    return {
        "loss_total": meter_total.avg,
        "loss_final": meter_final.avg,
        "loss_bN": meter_bN.avg,
        "loss_bL": meter_bL.avg,
        "loss_bM": meter_bM.avg,
        "loss_ortho": meter_ortho.avg,
        "loss_temp": meter_temp.avg,
        "loss_div": meter_div.avg,
    }


@torch.no_grad()
def validate(model, loader, device, tile_size, tile_overlap, use_amp, val_crop_size=None):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    loss_meter = AverageMeter()
    lpips_meter = AverageMeter()
    global _LPIPS_AVAILABLE, _LPIPS_FN
    if _LPIPS_AVAILABLE is None:
        try:
            import lpips
            _lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
            _LPIPS_AVAILABLE = True
            _LPIPS_FN = _lpips_fn
        except Exception:
            _LPIPS_AVAILABLE = False
            _LPIPS_FN = None
    _lpips_fn = _LPIPS_FN
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
        if _lpips_fn is not None:
            lpips_meter.update(_lpips_fn(pred, target).mean().item(), clip.size(0))
        del clip, target, pred, loss
    result = {"val_l1": loss_meter.avg, "psnr": psnr_meter.avg, "ssim": ssim_meter.avg}
    if _lpips_fn is not None:
        result["lpips"] = lpips_meter.avg
    return result


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

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("GolfNet params: %.2fM", n_params / 1e6)

    criterion = build_loss(cfg, device)
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    total_epochs = cfg["train"]["epochs"] if not args.smoke else 1
    warmup_epochs = cfg["train"].get("warmup_epochs", 5)

    def get_phase(epoch):
        if epoch < warmup_epochs:
            return 'phase1_warmup'
        return 'phase2'

    def get_lr(epoch, base=cfg["train"]["lr"]):
        if epoch < warmup_epochs:
            return base * (0.01 + 0.99 * epoch / warmup_epochs)
        elif epoch < 11:
            return base * 0.75
        elif epoch < 26:
            return base * 0.75 * (1 - (epoch - 11) / 15 * 0.33)
        elif epoch < 51:
            return base * 0.5
        else:
            return base * 0.125

    scaler = GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    grad_accum_steps = cfg["train"].get("grad_accum_steps", 1)

    best_psnr = -1.0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                logger.warning("Optimizer state load failed, starting fresh optimizer")
        start_epoch = ckpt["epoch"]
        best_psnr = ckpt.get("best_psnr", -1.0)
        logger.info("Resumed from %s (epoch %d, best_psnr=%.4f)", args.resume, start_epoch, best_psnr)

    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        logger.info("Loaded pretrained from %s (missing=%d, unexpected=%d)",
                    args.pretrained, len(missing), len(unexpected))

    for epoch in range(start_epoch, total_epochs):
        phase = get_phase(epoch)
        lr = get_lr(epoch)
        logger.info("Epoch %d / %d [%s] lr=%.2e unlock=1.00", epoch + 1, total_epochs, phase, lr)

        for pg in optimizer.param_groups:
            pg['lr'] = lr

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
            grad_accum_steps=grad_accum_steps,
        )
        logger.info("Train stats: %s", train_stats)

        val_stats = None
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
                "config": cfg,
                "best_psnr": best_psnr,   # fix: resume 时正确恢复 best_psnr
            },
            os.path.join(output_dir, "latest.pth"),
        )
        if val_stats is not None:
            if val_stats["psnr"] > best_psnr:
                best_psnr = val_stats["psnr"]
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": cfg,
                        "best_psnr": best_psnr,
                    },
                    os.path.join(output_dir, "best.pth"),
                )
                logger.info("Saved best.pth (psnr=%.4f)", best_psnr)
        if args.smoke:
            break


if __name__ == "__main__":
    main()
