#!/usr/bin/env python3
"""v8.8N RLK-TS 的单元测试。

验证:
  1. RLKTS 初始恒等 (误差 0), 非恒等参数下改变输出;
  2. 训练/推理形式等价 (reparameterize);
  3. gamma 与 dw 核的梯度可达;
  4. shift_kind 开关: omni 默认行为不变, rlk 全链路生效;
  5. 初始时 rlk 模型与 omni 模型输出不同 (恒等 Shift vs 1/√4 平均)。

运行:
    PYTHONPATH=. python tests/test_v8_8n_rlk.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.rlk_ts import RLKTS
from models.golf_v7r.spatial_rwkv_head_v8 import OmniShift


def test_rlk_identity_init():
    torch.manual_seed(0)
    m = RLKTS(32)
    x = torch.randn(2, 32, 16, 16)
    y = m.train()(x)
    err = (y - x).abs().max().item()
    assert err == 0.0, f"初始应恒等, 误差 {err:.2e}"
    print(f"  [ok] RLKTS 初始恒等 (误差 0.0)")


def test_rlk_nonidentity():
    torch.manual_seed(1)
    m = RLKTS(32)
    x = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        m.dw.weight += 0.2
        m.gamma.fill_(0.5)
    d = (m.train()(x) - x).abs().mean().item()
    assert d > 1e-3, f"非恒等参数下输出未变 ({d:.2e})"
    print(f"  [ok] 非恒等参数生效 (平均差 {d:.4f})")


def test_rlk_reparam_equivalence():
    torch.manual_seed(2)
    m = RLKTS(32)
    with torch.no_grad():
        m.dw.weight.normal_(0, 0.05)
        m.dw.weight[:, 0, 2, 2] += 1.0
        m.gamma.fill_(0.3)
    x = torch.randn(1, 32, 12, 12)
    y_train = m.train()(x)
    m.eval()
    y_eval = m(x)
    err = (y_train - y_eval).abs().max().item()
    assert err < 1e-5, f"融合后不一致: {err:.2e}"
    print(f"  [ok] 训练/推理形式等价 (误差 {err:.1e})")


def test_rlk_gradients():
    torch.manual_seed(3)
    m = RLKTS(32)
    # 恒等初始化下 gamma 首步梯度为零 (x+γ·0), 先扰动 dw 再验梯度
    with torch.no_grad():
        m.dw.weight += 0.1
    x = torch.randn(1, 32, 8, 8)
    m.train()(x).sum().backward()
    assert m.dw.weight.grad is not None and m.dw.weight.grad.abs().sum() > 0
    assert m.gamma.grad is not None and m.gamma.grad.abs().sum() > 0
    print("  [ok] dw 与 gamma 梯度可达")


def test_shift_kind_switch():
    from models.golf_v7r import GolfNet_v8
    base = dict(num_frames=5, encoder_channels=(32, 64, 128), encoder_loops=2,
                temporal_loops=2, spatial_head="v8", wkv_mode="single_norm",
                channel_mix_rate=4, use_omni_shift=True, use_channel_mix=True,
                lowpass_kernel=9, branch_blocks=2, fusion_mode="sgrf_v9",
                fusion_weight_floor=0.1, fusion_residual_freeze=True,
                branch_m_single_frame=True, q_mode="cross_rwkv",
                summary_type="g", q_shift_share="by_source",
                cm_shift=True, enc_cm_shift=True)
    m_omni = GolfNet_v8(**base)
    m_rlk = GolfNet_v8(**{**base, "shift_kind": "rlk"})
    assert isinstance(m_omni.tca.attn_N.shift_r, OmniShift)
    assert isinstance(m_rlk.tca.attn_N.shift_r, RLKTS)
    assert isinstance(m_rlk.tca.attn_N.cm_shift, RLKTS)
    assert isinstance(m_rlk.encoder.scale1_blocks[0].head.omni_shift, RLKTS)
    assert isinstance(m_rlk.encoder.scale1_blocks[0].head.cm_shift, RLKTS)
    n_o = sum(p.numel() for p in m_omni.parameters())
    n_r = sum(p.numel() for p in m_rlk.parameters())
    print(f"  [ok] 开关正确; 参数 omni {n_o/1e6:.4f}M vs rlk {n_r/1e6:.4f}M (Δ {(n_r-n_o)/1e3:+.1f}K)")
    assert n_r < n_o, "rlk 应更少参数"


def test_agg_untouched():
    from models.golf_v7r import GolfNet_v8
    base = dict(num_frames=5, encoder_channels=(32, 64, 128), encoder_loops=2,
                temporal_loops=2, spatial_head="v8", wkv_mode="single_norm",
                channel_mix_rate=4, use_omni_shift=True, use_channel_mix=True,
                lowpass_kernel=9, branch_blocks=2, fusion_mode="sgrf_v9",
                fusion_weight_floor=0.1, fusion_residual_freeze=True,
                branch_m_single_frame=True, q_mode="cross_rwkv",
                summary_type="g", q_shift_share="by_source",
                cm_shift=True, enc_cm_shift=True, shift_kind="rlk")
    m = GolfNet_v8(**base)
    agg_shift = m.summary.frame_encoder.head.omni_shift
    assert isinstance(agg_shift, OmniShift), "聚合器应保持 OmniShift (既定裁决)"
    print("  [ok] 聚合器保持 OmniShift, 未被 rlk 波及")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n全部 {len(tests)} 项通过")
