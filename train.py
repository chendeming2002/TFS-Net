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
from losses.losses import TFSNetLoss
from models import TFSNet
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
        # Flight9: DPE anti-collapse + edge-aware illumination
        lambda_illum_spatial=loss_cfg.get("lambda_illum_spatial", 0.1),
        lambda_illum_tv=loss_cfg.get("lambda_illum_tv", 0.05),
    )
    return criterion.to(device)


def train_one_epoch(model, criterion, optimizer, scaler, loader, device, use_amp, logger, log_interval, epoch=0, grad_clip=1.0, grad_accum_steps=1, phase='phase2', unlock_ratio=1.0):
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
    meter_gamma_reg = AverageMeter()

    progress = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
    diag_cache = {"s_illum": None, "gain": None, "g_ndpn": 0.0, "g_mcpn": 0.0, "ca": None}
    for step, (clip, target, _) in progress:
        clip = clip.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(clip, phase=phase)
        model._unlock_ratio = unlock_ratio
        loss, loss_dict = criterion(outputs, target, epoch=epoch, phase=phase, unlock_ratio=unlock_ratio, model=model)

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
        meter_gamma_reg.update(loss_dict.get("loss_gamma_reg", 0.0), clip.size(0))

        progress.set_postfix(loss=meter_total.avg, pix=meter_pix.avg, ssim=meter_ssim.avg,
                              perc=meter_perc.avg, i_sup=meter_illum_sup.avg)
        if (step + 1) % log_interval == 0:
            logger.info(
                "step %d/%d loss=%.4f pix=%.4f freq=%.4f ssim=%.4f perc=%.4f illum=%.4f i_sup=%.4f inter=%.4f ifpn=%.4f",
                step + 1,
                len(loader),
                meter_total.avg,
                meter_pix.avg,
                meter_freq.avg,
                meter_ssim.avg,
                meter_perc.avg,
                meter_illum.avg,
                meter_illum_sup.avg,
                meter_inter.avg,
                meter_ifpn.avg,
            )
            with torch.no_grad():
                s_ill = outputs.get("s_illum")
                gain = outputs.get("gain_map")
                g_ndpn = model.ndpn.gamma.abs().mean().item() if hasattr(model, 'ndpn') and hasattr(model.ndpn, 'gamma') else 0
                g_mcpn = model.mcpn.gamma.abs().mean().item() if hasattr(model, 'mcpn') and hasattr(model.mcpn, 'gamma') else 0
                ca = outputs.get("curve_alpha")
                if s_ill is not None:
                    logger.info(
                        "diag: dpe_si=%.3f/%.4f g=%.2f/%.3f gn=%.4f gm=%.4f",
                        s_ill.mean().item(), s_ill.std().item(),
                        gain.mean().item() if gain is not None else 0,
                        gain.std().item() if gain is not None else 0,
                        g_ndpn, g_mcpn,
                    )
                    if ca is not None:
                        ca_mean = ca.abs().mean().item()
                        logger.info("      curve_α|mean|=%.4f", ca_mean)

        # Release intermediate tensors after each step
        del outputs, loss, loss_dict, clip, target
        if (step + 1) % (grad_accum_steps * 50) == 0:
            torch.cuda.empty_cache()
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
def validate(model, loader, device, tile_size, tile_overlap, use_amp, val_crop_size=None, phase='phase2'):
    model.eval()
    if hasattr(model, 'clear_frame_cache'):
        model.clear_frame_cache()
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
            phase=phase,
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
    criterion = build_loss(cfg, device)
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    total_epochs = cfg["train"]["epochs"] if not args.smoke else 1

    # Mark3: Phase-based lr schedule (replaces CosineAnnealing + Linear warmup)
    def get_phase(epoch):
        if epoch < 5:   return 'phase1_warmup'
        elif epoch < 11: return 'phase1'
        elif epoch < 26: return 'phase1_5'
        else:            return 'phase2'

    def get_unlock_ratio(epoch):
        if epoch < 11: return 0.0
        elif epoch < 26: return (epoch - 11) / 15.0   # linear 0→1 over 15 epochs
        else: return 1.0

    def get_lr(epoch, base=cfg["train"]["lr"]):
        if epoch < 5: return base * (0.01 + 0.99 * epoch / 5)
        elif epoch < 11: return base * 0.75
        elif epoch < 26: return base * 0.75 * (1 - (epoch - 11) / 15 * 0.33)
        elif epoch < 51: return base * 0.5
        elif epoch < 65: return base * 0.125
        elif epoch < 73: return base * 0.05
        else: return base * 0.02
    scaler = GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    grad_accum_steps = cfg["train"].get("grad_accum_steps", 1)

    best_psnr = -1.0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt["model"]
        # Remap: die→dpe, wsd→wfr (Flight3 rename)
        sd = {k.replace('die.', 'dpe.').replace('wsd.', 'wfr.'): v for k, v in sd.items()}
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
        # MCPN startup reset: only for pre-Flight3 checkpoints (gamma==0 and startup_gate==1 in ckpt)
        ckpt_gamma_near_zero = True
        for k, v in sd.items():
            if 'mcpn.gamma' in k and v.abs().max() > 0.001:
                ckpt_gamma_near_zero = False
                break
        if ckpt_gamma_near_zero and hasattr(model, 'mcpn') and hasattr(model.mcpn, 'reset_startup'):
            model.mcpn.reset_startup()
            logger.info("MCPN startup params reset (pre-Flight3 checkpoint detected)")
        else:
            logger.info("Skipped MCPN reset (Flight3+ checkpoint, gamma already learned)")
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
        phase = get_phase(epoch)
        lr = get_lr(epoch)
        unlock = get_unlock_ratio(epoch)
        logger.info("Epoch %d / %d [%s] lr=%.2e unlock=%.2f", epoch + 1, total_epochs, phase, lr, unlock)

        # Clear frame cache at epoch boundary to prevent VRAM creep
        if hasattr(model, 'clear_frame_cache'):
            model.clear_frame_cache()
            torch.cuda.empty_cache()

        # Set lr
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Flight3 I: dynamic max_gain scheduling (4→16 over training)
        if hasattr(model, 'ispn') and hasattr(model.ispn, 'set_max_gain'):
            max_g = 4.0 + (16.0 - 4.0) * min(epoch / 30.0, 1.0)
            model.ispn.set_max_gain(max_g)
            if epoch % 10 == 0:
                logger.info("max_gain updated to %.1f", max_g)

        # Phase transition actions
        if epoch == 10:
            logger.info("Phase 1.5: Unlocking NDPN/MCPN/CXG")
        if epoch == 25:
            logger.info("Phase 2: Full tri-source restoration")

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
            phase=phase,
            unlock_ratio=unlock,
        )
        # Mark3: lr managed by get_lr(), scheduler disabled
        # scheduler.step()
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
                phase=phase,
            )
            logger.info("Val stats: %s", val_stats)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            },
            os.path.join(output_dir, "latest.pth"),
        )
        # Flight3 Mod3: per-phase checkpoint saving
        phase_ckpt_map = {5: "phase1_warmup.pth", 10: "phase1.pth", 25: "phase15.pth"}
        if epoch + 1 in phase_ckpt_map:
            phase_path = os.path.join(output_dir, phase_ckpt_map[epoch + 1])
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "phase": phase,
                },
                phase_path,
            )
            logger.info("Saved phase checkpoint: %s", phase_path)
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
        if args.smoke:
            break


if __name__ == "__main__":
    main()
