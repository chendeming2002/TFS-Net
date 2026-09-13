"""
Foxtrot (TSD-Net) 训练脚本
支持混合精度、梯度累积、warmup、val tile推理
"""
import os
import sys
import yaml
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.foxtrot import TSDNet
from models.foxtrot.loss import FoxtrotLoss
from data.video_loader import SDSDDataset
from utils.metrics import calculate_psnr, calculate_ssim


def setup_logger(log_path: Path):
    """设置日志"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 文件 handler
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def tile_inference(model: nn.Module, x: torch.Tensor, tile_size: int = 256, tile_overlap: int = 32) -> torch.Tensor:
    """
    Tile-based inference for large images
    x: (B, T, C, H, W)
    return: (B, C, H, W)
    """
    B, T, C, H, W = x.shape
    device = x.device
    
    # 如果图像小于 tile_size，直接推理
    if H <= tile_size and W <= tile_size:
        with torch.no_grad():
            out = model(x, return_intermediate=False)
        return out['O_t']
    
    # Tile 推理
    stride = tile_size - tile_overlap
    output = torch.zeros(B, 3, H, W, device=device)
    weight_map = torch.zeros(B, 1, H, W, device=device)
    
    # 分块推理
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            h_end = min(h + tile_size, H)
            w_end = min(w + tile_size, W)
            h_start = h_end - tile_size if h_end == H else h
            w_start = w_end - tile_size if w_end == W else w
            
            tile_x = x[:, :, :, h_start:h_end, w_start:w_end]
            
            with torch.no_grad():
                tile_out = model(tile_x, return_intermediate=False)['O_t']
            
            output[:, :, h_start:h_end, w_start:w_end] += tile_out
            weight_map[:, :, h_start:h_end, w_start:w_end] += 1.0
    
    output = output / weight_map.clamp(min=1.0)
    return output


def validate(model: nn.Module, val_loader: DataLoader, config: dict, logger: logging.Logger) -> Dict[str, float]:
    """验证"""
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    tile_size = config['eval'].get('tile_size', 256)
    tile_overlap = config['eval'].get('tile_overlap', 32)
    use_amp = config['eval'].get('amp', False)
    
    for batch in val_loader:
        x = batch['input'].cuda()  # (B, T, C, H, W)
        gt = batch['target'].cuda()  # (B, C, H, W)
        B = x.shape[0]
        
        # Tile推理
        with autocast(enabled=use_amp):
            pred = tile_inference(model, x, tile_size, tile_overlap)
        
        # 计算指标
        for i in range(B):
            psnr = calculate_psnr(pred[i], gt[i])
            ssim = calculate_ssim(pred[i].unsqueeze(0), gt[i].unsqueeze(0))
            total_psnr += psnr
            total_ssim += ssim
            count += 1
    
    avg_psnr = total_psnr / count if count > 0 else 0.0
    avg_ssim = total_ssim / count if count > 0 else 0.0
    
    logger.info(f"Val: PSNR={avg_psnr:.3f} dB, SSIM={avg_ssim:.4f}")
    
    return {"psnr": avg_psnr, "ssim": avg_ssim}


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    config: dict,
    writer: SummaryWriter,
    logger: logging.Logger,
    global_step: int,
) -> int:
    """训练一个 epoch"""
    model.train()
    
    log_interval = config['train']['log_interval']
    grad_accum_steps = config['train'].get('grad_accum_steps', 1)
    grad_clip = config['train'].get('grad_clip', None)
    use_amp = config['train'].get('amp', True)
    
    epoch_loss = 0.0
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        x = batch['input'].cuda()  # (B, T, C, H, W)
        gt = batch['target'].cuda()  # (B, C, H, W)
        
        # Forward
        with autocast(enabled=use_amp):
            out = model(x, return_intermediate=True)
            losses = loss_fn(out, gt)
            loss = losses['total_loss'] / grad_accum_steps
        
        # Backward
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            global_step += 1
        
        epoch_loss += losses['total_loss'].item()
        
        # 日志
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = epoch_loss / (batch_idx + 1)
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                f"Loss={avg_loss:.4f} "
                f"L_final={losses['L_final'].item():.4f} "
                f"L_N={losses['L_N'].item():.4f} "
                f"L_L={losses['L_L'].item():.4f} "
                f"L_M={losses['L_M'].item():.4f} "
                f"L_ortho={losses['L_ortho'].item():.4f}"
            )
            
            # TensorBoard
            writer.add_scalar('train/total_loss', losses['total_loss'].item(), global_step)
            writer.add_scalar('train/L_final', losses['L_final'].item(), global_step)
            writer.add_scalar('train/L_N', losses['L_N'].item(), global_step)
            writer.add_scalar('train/L_L', losses['L_L'].item(), global_step)
            writer.add_scalar('train/L_M', losses['L_M'].item(), global_step)
            writer.add_scalar('train/L_ortho', losses['L_ortho'].item(), global_step)
    
    epoch_time = time.time() - epoch_start
    avg_loss = epoch_loss / len(train_loader)
    logger.info(f"Epoch {epoch} finished in {epoch_time/60:.1f} min, avg_loss={avg_loss:.4f}")
    
    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/foxtrot.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的 checkpoint 路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    
    # 设置日志
    logger = setup_logger(output_dir / 'train.log')
    logger.info(f"Config: {args.config}")
    logger.info(f"Output dir: {output_dir}")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    
    # 设置随机种子
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 构建数据集
    logger.info("Building datasets...")
    train_dataset = SDSDDataset(
        input_root=config['dataset']['train_input_root'],
        target_root=config['dataset']['train_target_root'],
        window_size=config['dataset']['window_size'],
        crop_size=config['dataset']['crop_size'],
        mode='train',
    )
    
    val_dataset = SDSDDataset(
        input_root=config['dataset']['val_input_root'],
        target_root=config['dataset']['val_target_root'],
        window_size=config['dataset']['window_size'],
        crop_size=config['dataset'].get('val_crop_size', None),
        mode='val',
        max_seqs=config['dataset'].get('max_val_seqs', None),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 构建模型
    logger.info("Building model...")
    model = TSDNet(**config['model']).cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params / 1e6:.2f}M")
    
    # 损失函数
    loss_fn = FoxtrotLoss(**config['loss']).cuda()
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
    )
    
    # 学习率调度器 (cosine with warmup)
    warmup_epochs = config['train'].get('warmup_epochs', 0)
    total_epochs = config['train']['epochs']
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP scaler
    scaler = GradScaler(enabled=config['train'].get('amp', True))
    
    # 恢复训练
    start_epoch = 1
    global_step = 0
    best_psnr = 0.0
    
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_psnr = ckpt.get('best_psnr', 0.0)
        logger.info(f"Resumed at epoch {start_epoch}, best_psnr={best_psnr:.3f}")
    
    # 训练循环
    logger.info("Start training...")
    for epoch in range(start_epoch, total_epochs + 1):
        logger.info(f"Epoch {epoch}/{total_epochs}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练
        global_step = train_epoch(
            model, train_loader, loss_fn, optimizer, scaler,
            epoch, config, writer, logger, global_step
        )
        
        # 学习率更新
        scheduler.step()
        
        # 验证
        if epoch % config['train']['val_interval'] == 0 or epoch == total_epochs:
            val_metrics = validate(model, val_loader, config, logger)
            writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
            writer.add_scalar('val/ssim', val_metrics['ssim'], epoch)
            
            # 保存最优模型
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_psnr': best_psnr,
                }, ckpt_dir / 'best.pth')
                logger.info(f"Best model saved: {best_psnr:.3f} dB")
        
        # 定期保存 checkpoint
        if epoch % 10 == 0 or epoch == total_epochs:
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_psnr': best_psnr,
            }, ckpt_dir / f'epoch_{epoch}.pth')
            logger.info(f"Checkpoint saved: epoch_{epoch}.pth")
    
    logger.info(f"Training finished! Best PSNR: {best_psnr:.3f} dB")
    writer.close()


if __name__ == '__main__':
    main()
