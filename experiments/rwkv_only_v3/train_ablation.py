import argparse, os, sys
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

import torch, yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from datasets import SDSDDataset
from datasets.samplers import TemporalBucketSampler
from losses.losses import ssim_map
from utils.metrics import tensor_psnr, tensor_ssim
from utils.misc import AverageMeter, create_logger, seed_everything
from utils.io import save_checkpoint
from model_ablation import RWKVOnlyAblation

import torch.nn.functional as F

try:
    from tqdm import tqdm
except:
    def tqdm(x, **kw): return x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--ablation", type=str, default="none",
                   choices=["none", "A", "B", "C", "BC", "ABC"])
    return p.parse_args()


def train_one_epoch(model, opt, scaler, loader, device, use_amp, logger,
                    log_interval, grad_clip, grad_accum):
    model.train()
    meter = AverageMeter()
    progress = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
    for step, (clip, target, _) in progress:
        clip = clip.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            out = model(clip)
        pred = out["res_t"].float()
        loss = F.l1_loss(pred, target) + 0.2 * (1 - ssim_map(pred, target).mean())
        loss = loss / grad_accum
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        meter.update(loss.item() * grad_accum, clip.size(0))
        progress.set_postfix(loss=meter.avg)
        if (step + 1) % log_interval == 0:
            logger.info("step %d/%d loss=%.4f", step + 1, len(loader), meter.avg)
    return {"loss": meter.avg}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_m = AverageMeter()
    ssim_m = AverageMeter()
    for clip, target, _ in tqdm(loader, total=len(loader), desc="val", leave=False):
        clip = clip.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        out = model(clip)
        pred = out["res_t"]
        psnr_m.update(tensor_psnr(pred, target), clip.size(0))
        ssim_m.update(tensor_ssim(pred, target), clip.size(0))
    return {"psnr": psnr_m.avg, "ssim": ssim_m.avg}


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed_everything(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logger = create_logger(cfg["output_dir"])
    device = torch.device("cuda")

    flags = {
        "A":   {"remove_sigmoid": True,  "use_hf_residual": False, "use_conf_scale": False},
        "B":   {"remove_sigmoid": False, "use_hf_residual": True,  "use_conf_scale": False},
        "C":   {"remove_sigmoid": False, "use_hf_residual": False, "use_conf_scale": True},
        "BC":  {"remove_sigmoid": False, "use_hf_residual": True,  "use_conf_scale": True},
        "ABC": {"remove_sigmoid": True,  "use_hf_residual": True,  "use_conf_scale": True},
        "none":{"remove_sigmoid": False, "use_hf_residual": False, "use_conf_scale": False},
    }[args.ablation]

    ds_cfg = cfg["dataset"]
    train_ds = SDSDDataset(ds_cfg["train_input_root"], ds_cfg["train_target_root"],
                           ds_cfg["window_size"], "train", ds_cfg["crop_size"])
    val_ds = SDSDDataset(ds_cfg["val_input_root"], ds_cfg["val_target_root"],
                         ds_cfg["window_size"], "val", ds_cfg["crop_size"])
    if args.smoke:
        train_ds = Subset(train_ds, range(8))
        val_ds = Subset(val_ds, range(4))

    nw = 0 if args.smoke else cfg["dataset"].get("num_workers", 2)
    if args.smoke:
        train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                                  shuffle=True, num_workers=0, pin_memory=True)
    else:
        train_sampler = TemporalBucketSampler(train_ds.samples,
                                              cfg["train"]["batch_size"],
                                              seed=cfg["seed"])
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                                  num_workers=nw, pin_memory=True,
                                  persistent_workers=nw > 0,
                                  prefetch_factor=2 if nw > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg["dataset"].get("num_workers", 2),
                            pin_memory=True)

    model = RWKVOnlyAblation(num_frames=ds_cfg["window_size"], **flags).to(device)
    opt = AdamW(model.parameters(), lr=cfg["train"]["lr"],
                 weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=cfg["train"].get("amp", False))
    total_epochs = 1 if args.smoke else cfg["train"]["epochs"]
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        logger.info("Resumed from epoch %d", start_epoch)

    logger.info("Ablation=%s flags=%s Params: %.2fM",
                args.ablation, flags, sum(p.numel() for p in model.parameters()) / 1e6)

    for epoch in range(start_epoch, total_epochs):
        t = train_one_epoch(model, opt, scaler, train_loader, device,
                            cfg["train"].get("amp", False), logger,
                            cfg["train"]["log_interval"],
                            cfg["train"]["grad_clip"],
                            cfg["train"]["grad_accum_steps"])
        logger.info("Epoch %d/%d train: %s", epoch + 1, total_epochs, t)

        if (epoch + 1) % cfg["train"]["val_interval"] == 0:
            v = validate(model, val_loader, device)
            logger.info("Val: %s", v)

        save_checkpoint({"epoch": epoch + 1, "model": model.state_dict(),
                         "optimizer": opt.state_dict()},
                        os.path.join(cfg["output_dir"], "latest.pth"))


if __name__ == "__main__":
    main()
