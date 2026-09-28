#!/usr/bin/env python3
"""
golf_v7r MatrixRWKVTimeMix 单元测试

验证矩阵状态递推与 RWKV-6 朴素参考实现一致:
    S_t = S_{t-1}·diag(w_t) + v_t^T·k_t
    o_t = r_t · S_t
其中 state[b,h,i,j]: i = value 通道, j = key 通道。

运行:
    PYTHONPATH=. python tests/test_matrix_rwkv.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.matrix_rwkv import MatrixRWKVTimeMix, MatrixRWKVBlock


def reference_rwkv(r, k, v, w):
    """
    朴素参考实现。输入均为 [B, T, H, d], 返回 [B, T, H, d]。
    state[i, j] = v[i] * k[j]; 衰减作用在 key 维 j; 读出 r 与 j 收缩。
    """
    B, T, H, d = r.shape
    state = torch.zeros(B, H, d, d, dtype=r.dtype)
    out = torch.zeros(B, T, H, d, dtype=r.dtype)
    for t in range(T):
        state = state * w[:, t, :, None, :]
        state = state + torch.einsum('bhi,bhj->bhij', v[:, t], k[:, t])
        out[:, t] = torch.einsum('bhj,bhij->bhi', r[:, t], state)
    return out


def test_recurrence_matches_reference():
    torch.manual_seed(0)
    B, T, H, d = 2, 5, 4, 8
    m = MatrixRWKVTimeMix(dim=H * d, num_heads=H, head_size=d).eval()

    # 直接喂入随机 r/k/v/w, 比较 recurrence 与参考实现
    r = torch.randn(B, T, H, d)
    k = torch.randn(B, T, H, d)
    v = torch.randn(B, T, H, d)
    w = torch.rand(B, T, H, d)  # ∈ (0,1)

    ref = reference_rwkv(r, k, v, w)
    actual = MatrixRWKVTimeMix.recurrence(r, k, v, w)

    diff = (ref - actual).abs().max().item()
    assert diff < 1e-5, f"recurrence mismatch: max diff = {diff}"
    print(f"  [OK] recurrence == reference (max diff {diff:.2e})")


def test_decay_masking():
    """w=0 (完全遗忘) 时, 输出不应受更早帧影响 (除当前帧外)。"""
    torch.manual_seed(1)
    B, T, H, d = 1, 4, 2, 4
    r = torch.randn(B, T, H, d)
    k = torch.randn(B, T, H, d)
    v = torch.randn(B, T, H, d)
    w = torch.zeros(B, T, H, d)  # 每步衰减到 0

    out = MatrixRWKVTimeMix.recurrence(r, k, v, w)
    # 每帧输出应只依赖当前帧: o_t = (r_t·v_t) * k_t
    for t in range(T):
        expect = torch.einsum('bhj,bhi,bhj->bhi', r[:, t], v[:, t], k[:, t])
        # 注意: o[i] = sum_j r[j]*v[i]*k[j] = v[i] * sum_j r[j]k[j]
        expect2 = v[:, t] * (r[:, t] * k[:, t]).sum(-1, keepdim=True)
        diff = (out[:, t] - expect2).abs().max().item()
        assert diff < 1e-5, f"t={t}: decay masking failed, diff={diff}"
    print("  [OK] w=0 时输出仅依赖当前帧 (无跨帧泄漏)")


def test_output_shape_and_grad():
    torch.manual_seed(2)
    B, T, D = 2, 5, 192
    block = MatrixRWKVBlock(dim=D, num_heads=6, head_size=32)
    x = torch.randn(B, T, D, requires_grad=True)
    out = block(x)
    assert out.shape == (B, T, D), f"unexpected shape {out.shape}"
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"  [OK] MatrixRWKVBlock shape {tuple(out.shape)} + 梯度有限")


if __name__ == '__main__':
    print("test_matrix_rwkv:")
    test_recurrence_matches_reference()
    test_decay_masking()
    test_output_shape_and_grad()
    print("All tests passed.")
