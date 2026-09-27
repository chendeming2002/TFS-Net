#!/usr/bin/env python3
"""最小化训练测试 - 只运行100步"""
import sys
import torch
from torch.utils.data import DataLoader
from datasets import SDSDDataset
from models.golf_v7 import GolfNet_v7
from models.golf_v7.loss import SimpleLoss
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Minimal Golf v7 Test ===")
    
    # 1. Model
    device = torch.device("cuda")
    model = GolfNet_v7().to(device)
    logger.info(f"Model created, params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 2. Dataset (极小batch)
    train_ds = SDSDDataset(
        input_root="/home/a1005/yzy/dataset/SDSD/indoor/input",
        target_root="/home/a1005/yzy/dataset/SDSD/indoor/GT",
        window_size=5,
        mode="train",
        crop_size=256
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    logger.info(f"Dataset ready, {len(train_ds)} samples")
    
    # 3. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = SimpleLoss()
    logger.info("Optimizer & loss ready")
    
    # 4. Train loop - 只100步
    model.train()
    logger.info("Starting training loop...")
    
    for step, batch in enumerate(train_loader):
        if step >= 100:
            break
        
        try:
            # SDSDDataset 返回 (lq, gt, meta) tuple
            lq, gt, meta = batch
            lq = lq.to(device)
            gt = gt.to(device)
            
            if step < 3:
                logger.info(f"Step {step+1}: lq shape={lq.shape}, gt shape={gt.shape}")
            
            # Forward
            output_dict = model(lq)
            
            if step < 3:
                logger.info(f"Step {step+1}: output shape={output_dict['final'].shape}")
            
            # Loss 函数期望 (pred_dict, target)
            loss_dict = criterion(output_dict, gt)
            loss = loss_dict['loss']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 10 == 0:
                logger.info(f"✓ Step {step+1}/100 - Loss: {loss.item():.4f}")
            
            # 每20步检查内存
            if (step + 1) % 20 == 0:
                mem_alloc = torch.cuda.memory_allocated()/1e9
                mem_max = torch.cuda.max_memory_allocated()/1e9
                logger.info(f"  GPU memory: {mem_alloc:.2f}GB / {mem_max:.2f}GB peak")
        
        except Exception as e:
            logger.error(f"✗ Error at step {step}: {e}", exc_info=True)
            raise
    
    logger.info("=" * 60)
    logger.info("🎉 Test COMPLETED Successfully - All 100 steps ran without crash!")
    logger.info("=" * 60)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)
