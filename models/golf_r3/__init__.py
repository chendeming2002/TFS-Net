"""Golf-R3: Confidence fix + Hi-res warp + F3 FiLM"""
from .golfnet import GolfNet as GolfNet_R3
from .loss import GolfLoss as GolfLoss_R3

# 向后兼容别名
GolfNet = GolfNet_R3
GolfLoss = GolfLoss_R3

__all__ = ["GolfNet_R3", "GolfLoss_R3", "GolfNet", "GolfLoss"]
