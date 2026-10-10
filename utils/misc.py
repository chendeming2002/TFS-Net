import logging
import os
import random

import numpy as np
import torch


class AverageMeter(object):
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


def seed_everything(seed, deterministic=True):
    """固定随机种子。

    deterministic=True (默认, 2026-10-10 新增): 同时开启 cuDNN 确定性。
    **为什么必要**: 此前只设四个种子而未设 cuDNN 确定性, 导致**同 seed 同配置
    的两次独立训练, ep5 val PSNR 相差 2.13 dB**
    (`pospair` 24.05 vs `pospair_quick` 21.92, 配置仅差 output_dir)。
    该方差远超任何 toggle 效应量 (≤0.3 dB), 使所有单 run 对比失效
    (见 docs/v7/03 §6.14)。开启后同 seed 应可复现, 多 seed 重复才有意义。

    代价: 关闭 cuDNN autotune, 卷积可能变慢。本任务实测影响见 §6.14;
    若需追求速度可显式传 deterministic=False, 但**跨 run 比较必须用 True**。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 个别 CUDA 算子 (如某些 atomic 归约) 需显式开启确定性实现
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def create_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("mins_net")
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

