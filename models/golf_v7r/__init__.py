#!/usr/bin/env python3
"""
Golf v7r: Matrix RWKV + 多元噪声分割

模块导出
"""
from .golfnet_v7r import GolfNet_v7r
from .golfnet_v7r_v3 import GolfNet_v7r_v3, count_params as count_params_v3
from .golfnet_v8 import GolfNet_v8
from .matrix_rwkv import MatrixRWKVTimeMix, ReLUSquaredMLP, MatrixRWKVBlock
from .spatial_summary import SpatialSummary
from .context_decomp import ContextDecomposition, BranchFiLM
from .triple_query_tca import TripleQueryTCA
from .triple_query_tca_v9 import TaskQueryAttention, TripleQueryTCAv9
from .spatial_summary_unet import TemporalAggregator
from .cross_rwkv_head_v8 import CrossRWKVHeadV8
from .branch_n_simple import NBranch, BranchNSimple
from .branch_l_simple import LBranch, BranchLSimple
from .branch_m_simple import MBranch, BranchMSimple
from .fusion_v7r import V7RFusion
from .loss import GolfV7RLoss, SimpleLoss

__all__ = [
    'GolfNet_v7r',
    'GolfNet_v7r_v3',
    'GolfNet_v8',
    'count_params_v3',
    'MatrixRWKVTimeMix',
    'ReLUSquaredMLP',
    'MatrixRWKVBlock',
    'SpatialSummary',
    'ContextDecomposition',
    'BranchFiLM',
    'TripleQueryTCA',
    'TaskQueryAttention',
    'TemporalAggregator',
    'CrossRWKVHeadV8',
    'BranchNSimple',
    'NBranch',
    'BranchLSimple',
    'LBranch',
    'BranchMSimple',
    'MBranch',
    'V7RFusion',
    'GolfV7RLoss',
    'SimpleLoss',
]
