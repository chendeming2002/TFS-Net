"""Golf-R5: conservative fix — dynamic temporal gating, R2 KV rollback, no FiLM"""
from .golfnet import GolfNet as GolfNet_R5
from .loss import GolfLoss as GolfLoss_R5

GolfNet = GolfNet_R5
GolfLoss = GolfLoss_R5

__all__ = ["GolfNet_R5", "GolfLoss_R5", "GolfNet", "GolfLoss"]
