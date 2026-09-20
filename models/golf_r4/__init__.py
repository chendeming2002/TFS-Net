"""Golf-R4: full fix — double-zero deadlock, FiLM, prior_L, K/V shared, true temporal loss"""
from .golfnet import GolfNet as GolfNet_R4
from .loss import GolfLoss as GolfLoss_R4

GolfNet = GolfNet_R4
GolfLoss = GolfLoss_R4

__all__ = ["GolfNet_R4", "GolfLoss_R4", "GolfNet", "GolfLoss"]
