#!/usr/bin/env python3
"""Structural and gradient tests for the v9 Core-QRB redesign."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.core_query_rwkv import CoreQueryRWKVHead, WTConvLite
from models.golf_v7r.coreg_query_rwkv import CoRegisteredQueryRWKVHead
from models.golf_v7r.triple_query_tca_v9 import TaskQueryAttention


def _head(mixer="current"):
    return CoreQueryRWKVHead(
        32, wkv_mode="legacy", shift_share="by_source",
        cm_shift=True, query_mixer=mixer)


def test_spatial_g_path_is_absent():
    head = _head()
    names = set(dict(head.named_parameters()))
    assert not hasattr(head, "proj_g")
    assert not hasattr(head, "shift_g")
    assert not any("pair_to_g" in name or "gate_scale" in name for name in names)
    # Channel Mix receptance remains intentionally.
    assert hasattr(head, "cm_r") and head.cm_r is not None


def test_initial_kq_and_signed_alignment_are_zero():
    head = _head().eval()
    assert torch.count_nonzero(head.proj_kq.weight) == 0
    assert torch.count_nonzero(head.proj_align.weight) == 0
    assert torch.count_nonzero(head.proj_align.bias) == 0
    with torch.no_grad():
        y = head(torch.randn(1, 32, 7, 9), torch.randn(1, 32, 7, 9))
    assert y.shape == (1, 32, 7, 9) and torch.isfinite(y).all()
    assert head._last_core["align_abs_mean"].item() == 0.0
    assert head._last_core["kq_rms"].item() == 0.0


def test_kq_and_alignment_unlock_on_first_step():
    torch.manual_seed(7)
    head = _head()
    q = torch.randn(2, 32, 8, 8, requires_grad=True)
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    head(q, x).square().mean().backward()
    for name in ("proj_kq.weight", "proj_align.weight", "proj_align.bias"):
        grad = dict(head.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, f"{name} has no gradient"
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_alignment_is_directly_signed():
    head = _head().eval()
    with torch.no_grad():
        head.proj_align.weight.zero_()
        head.proj_align.bias.copy_(torch.tensor([-0.8, -0.2, 0.3, 0.9]))
        head(torch.randn(1, 32, 6, 6), torch.randn(1, 32, 6, 6))
    stats = head._last_core
    assert stats["align_min"] < 0
    assert stats["align_max"] > 0
    assert 0 < stats["align_positive_fraction"] < 1


def test_all_query_mixers_forward_backward():
    for mixer in CoreQueryRWKVHead._QUERY_MIXERS:
        torch.manual_seed(11)
        head = _head(mixer)
        q = torch.randn(1, 32, 7, 9, requires_grad=True)
        x = torch.randn(1, 32, 7, 9, requires_grad=True)
        y = head(q, x)
        assert y.shape == q.shape and torch.isfinite(y).all(), mixer
        y.mean().backward()
        assert q.grad is not None and q.grad.abs().sum() > 0, mixer
        assert x.grad is not None and x.grad.abs().sum() > 0, mixer


def test_query_mixer_topologies_are_exclusive():
    current, omni, span, wt = (_head(name) for name in ("current", "omni", "span", "wt"))
    assert current.query_shift is not None and current.query_spab is not None
    assert omni.query_shift is not None and omni.query_spab is None
    assert span.query_shift is None and span.query_spab is not None
    assert wt.query_shift is None and wt.query_spab is None
    assert isinstance(wt.query_wt, WTConvLite)


def test_wtconv_identity_initialization_and_odd_shape():
    mixer = WTConvLite(5)
    x = torch.randn(2, 5, 7, 9, requires_grad=True)
    y = mixer(x)
    assert y.shape == x.shape
    assert torch.equal(y, x)
    y.square().mean().backward()
    assert mixer.gamma.grad is not None and mixer.gamma.grad.abs() > 0


def test_core_is_smaller_than_coreg():
    core = _head()
    coreg = CoRegisteredQueryRWKVHead(
        32, wkv_mode="legacy", shift_share="by_source", cm_shift=True)
    assert sum(p.numel() for p in core.parameters()) < sum(
        p.numel() for p in coreg.parameters())


def test_tca_core_mode_and_stats():
    tca = TaskQueryAttention(
        32, spatial_head="v8", wkv_mode="legacy", q_mode="core",
        q_shift_share="by_source", cm_shift=True, q_mixer="span")
    multi = torch.randn(1, 32, 8, 8, requires_grad=True)
    single = torch.randn(1, 32, 8, 8, requires_grad=True)
    outputs = tca(multi, single)
    assert isinstance(tca.attn_N, CoreQueryRWKVHead)
    assert set(tca._last_core_stats) == {"N", "L", "M"}
    loss = sum(feat.mean() for feat in outputs[:3]) + outputs[3]
    loss.backward()
    assert multi.grad is not None and multi.grad.abs().sum() > 0
    assert single.grad is not None and single.grad.abs().sum() > 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\nAll {len(tests)} tests passed")
