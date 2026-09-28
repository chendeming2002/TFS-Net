#!/usr/bin/env python3
"""Golf v7 简化训练脚本 - 去掉所有非必要诊断代码"""
import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from datasets import SDSDDataset
from models.golf_v7 import GolfNet_v7
from models.golf_v7.loss import SimpleLoss
from utils.io import save_checkpoint
from utils.metrics import tensor_psnr, tensor_ssim
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Device
    device = torch.device("cuda")
    logger.info("Using device: cuda")
    
    # Dataset
    train_ds = SDSDDataset(
        input_root=cfg['dataset']['train_input_root'],
        target_root=cfg['dataset']['train_target_root'],
        window_size=cfg['dataset']['window_size'],
        mode="train",
        crop_size=cfg['dataset']['crop_size']
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    logger.info(f"Train dataset: {len(train_ds)} samples, {len(train_loader)} batches")
    
    # Val dataset
    val_ds = SDSDDataset(
        input_root=cfg['dataset']['val_input_root'],
        target_root=cfg['dataset']['val_target_root'],
        window_size=cfg['dataset']['window_size'],
        mode="val",
        crop_size=None
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    logger.info(f"Val dataset: {len(val_ds)} samples")
    
    # Model
    model = GolfNet_v7().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"GolfNet_v7 params: {n_params/1e6:.2f}M")
    
    # Loss & Optimizer
    criterion = SimpleLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'])
    
    # Output dir
    out_dir = cfg['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    # Training loop
    num_epochs = cfg['train']['epochs']
    log_interval = cfg['train']['log_interval']
    val_interval = cfg['train']['val_interval']
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"=" * 60)
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"=" * 60)
        
        # Train
        model.train()
        for step, batch in enumerate(train_loader):
            lq, gt, meta = batch
            lq = lq.to(device)
            gt = gt.to(device)
            
            # Forward
            output_dict = model(lq)
            loss_dict = criterion(output_dict, gt)
            loss = loss_dict['loss']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log
            if (step + 1) % log_interval == 0:
                logger.info(f"  Step {step+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # Save checkpoint
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        save_checkpoint(state, os.path.join(out_dir, 'latest.pth'))
        logger.info(f"Saved checkpoint: latest.pth (epoch {epoch})")
        
        # Validate
        if epoch % val_interval == 0:
            model.eval()
            psnr_list = []
            ssim_list = []
            
            with torch.no_grad():
                for batch in val_loader:
                    lq, gt, meta = batch
                    lq = lq.to(device)
                    gt = gt.to(device)
                    
                    output_dict = model(lq)
                    pred = output_dict['final']
                    
                    psnr = tensor_psnr(pred, gt)
                    ssim = tensor_ssim(pred, gt)
                    
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
            
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)
            
            logger.info(f"Validation - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
            
            # Save epoch checkpoint
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {'psnr': avg_psnr, 'ssim': avg_ssim}
            }
            save_checkpoint(state, os.path.join(out_dir, f'epoch_{epoch:03d}.pth'))
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
