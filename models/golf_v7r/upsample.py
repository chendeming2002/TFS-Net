#!/usr/bin/env python3
"""
golf_v7r 上采样模块: 复用 golf_v7 的 Upsample3x3 (修复棋盘伪影)
"""
from models.golf_v7.upsample import Upsample3x3, Upsample1x1

__all__ = ['Upsample3x3', 'Upsample1x1']
