#!/usr/bin/env python3
"""Contract tests for the actual v9 SGRFv10 S0/S1 experiments.

These tests intentionally separate the state updater from the expensive RWKV
backbone.  They verify the experiment-defining properties: X_t initialization,
fixed N->L->M order, signed/unclamped corrections, the 449-parameter shared
gate, identity initialization, and strict S0/S1 configuration parity.
"""
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.branch_l_simple import LBranch
from models.golf_v7r.branch_m_simple import MBranch
from models.golf_v7r.branch_n_simple import NBranch
from models.golf_v7r.golfnet_v8 import GolfNet_v8
from models.golf_v7r.sgrf_v10 import SGRFv10, SharedCorrectionGate


ROOT = Path(__file__).resolve().parents[1]


class _LearnedDelta(nn.Module):
    """Minimal branch exposing the SGRFv10 ``predict_delta`` contract."""

    def __init__(self, value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))
        self.seen_states = []

    def predict_delta(self, feat, state, *args):
        self.seen_states.append(state.detach().clone())
        delta = torch.ones_like(state) * self.value
        return {"delta": delta, "candidate": state + delta}


def _run(gate_mode="fixed", values=(0.1, 0.2, 0.3), zero_stage=None):
    sgrf = SGRFv10(gate_mode=gate_mode)
    branches = [_LearnedDelta(v) for v in values]
    x = torch.full((1, 3, 5, 7), 0.2)
    feats = {name: torch.zeros(1, 4, 5, 7) for name in "NLM"}
    out = sgrf(feats, x, *branches, torch.zeros(1, 4, 3, 4),
               zero_stage=zero_stage)
    return sgrf, branches, x, out


def _model_kwargs(gate):
    return dict(
        num_frames=3,
        encoder_channels=(8, 16, 32),
        encoder_loops=1,
        temporal_loops=1,
        wkv_mode="legacy",
        branch_blocks=1,
        fusion_mode="sgrf_v10",
        sgrf_v10_gate=gate,
        branch_m_single_frame=True,
        q_mode="core",
        q_mixer="omni",
        q_shift_share="by_source",
        cm_shift=True,
        enc_cm_shift=True,
        summary_type="g",
    )


def test_shared_gate_is_exactly_449_params_and_starts_at_half():
    gate = SharedCorrectionGate()
    assert sum(p.numel() for p in gate.parameters()) == 449
    with torch.no_grad():
        g = gate(torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8))
    assert torch.equal(g, torch.full_like(g, 0.5))


def test_s0_is_fixed_nlm_state_update_without_clamp():
    sgrf, branches, x, out = _run("fixed")
    assert sgrf.gate is None
    assert sum(p.numel() for p in sgrf.parameters()) == 0
    assert torch.allclose(out["O_t"], x + 0.6)
    assert out["O_t"].detach().max().item() > 0.5
    assert torch.equal(branches[0].seen_states[0], x)
    assert torch.allclose(branches[1].seen_states[0], x + 0.1)
    assert torch.allclose(branches[2].seen_states[0], x + 0.3)
    assert all(torch.equal(out["gates"][name],
                           torch.ones_like(out["gates"][name]))
               for name in "NLM")

    _, _, _, raw = _run("fixed", values=(0.5, 0.5, 0.5))
    assert raw["O_t"].detach().max().item() > 1.0, \
        "intermediate/final training state was clamped"


def test_s1_uses_one_shared_gate_and_zero_stage_is_effective():
    sgrf, _, x, out = _run("shared")
    assert sum(p.numel() for p in sgrf.parameters()) == 449
    assert torch.allclose(out["O_t"], x + 0.5 * 0.6)
    assert all(torch.equal(out["gates"][name],
                           torch.full_like(out["gates"][name], 0.5))
               for name in "NLM")
    _, _, _, no_n = _run("shared", zero_stage="N")
    assert torch.allclose(no_n["O_t"], x + 0.5 * (0.2 + 0.3))


def test_s1_gradients_reach_gate_and_all_three_deltas():
    sgrf, branches, _, out = _run("shared")
    out["O_t"].square().mean().backward()
    for branch in branches:
        assert branch.value.grad is not None and branch.value.grad.abs() > 0
    last = sgrf.gate.net[-1]
    assert last.weight.grad is not None and last.weight.grad.abs().sum() > 0
    assert last.bias.grad is not None and last.bias.grad.abs().sum() > 0


def test_real_branches_make_sgrfv10_an_exact_identity_at_initialization():
    torch.manual_seed(3)
    n = NBranch(8, num_blocks=1, at_f1=True, state_conditioned=True)
    l = LBranch(8, num_blocks=1, at_f1=True)
    m = MBranch(8, enc_channels=16, num_frames=3, num_blocks=1,
                single_frame=True, at_f1=True, state_conditioned=True)
    feats = {name: torch.randn(1, 8, 8, 8) for name in "NLM"}
    x = torch.rand(1, 3, 8, 8)
    for mode in ("fixed", "shared"):
        out = SGRFv10(mode)(feats, x, n, l, m, torch.randn(1, 16, 4, 4))
        assert torch.equal(out["O_t"], x), mode
        assert all(torch.count_nonzero(out["deltas"][name]) == 0
                   for name in "NLM")


def test_full_s0_s1_parameter_delta_and_forward_contract():
    fixed = GolfNet_v8(**_model_kwargs("fixed"))
    shared = GolfNet_v8(**_model_kwargs("shared"))
    n_fixed = sum(p.numel() for p in fixed.parameters())
    n_shared = sum(p.numel() for p in shared.parameters())
    assert n_shared - n_fixed == 449
    assert fixed.branch_N.state_proj is not None
    assert fixed.branch_M.state_proj is not None

    x = torch.rand(1, 3, 3, 8, 8)
    with torch.no_grad():
        out = shared(x)
    assert torch.equal(out["final"], x[:, 1])
    assert set(out["stage_stats"]) >= {
        *(f"gate_{name}" for name in "NLM"),
        *(f"delta_{name}" for name in "NLM"),
        *(f"inc_{name}" for name in "NLM"),
    }


def test_s0_s1_configs_are_a_strict_single_variable_pair():
    s0 = yaml.safe_load((ROOT / "configs/v9/v9_sgrfv10_s0_fixed_30ep.yaml")
                        .read_text(encoding="utf-8"))
    s1 = yaml.safe_load((ROOT / "configs/v9/v9_sgrfv10_s1_shared_30ep.yaml")
                        .read_text(encoding="utf-8"))
    assert s0["model"]["sgrf_v10_gate"] == "fixed"
    assert s1["model"]["sgrf_v10_gate"] == "shared"
    s0_cmp, s1_cmp = copy.deepcopy(s0), copy.deepcopy(s1)
    s0_cmp["output_dir"] = s1_cmp["output_dir"] = "<ignored>"
    s0_cmp["model"]["sgrf_v10_gate"] = "<ignored>"
    s1_cmp["model"]["sgrf_v10_gate"] = "<ignored>"
    assert s0_cmp == s1_cmp
    assert all(s0["loss"][name] == 0 for name in
               ("w_branch_N", "w_branch_L", "w_branch_M",
                "lambda_ortho", "w_gate_reg"))
    assert s0["loss"]["branch_specific_targets"] is False
    assert s0["loss"]["w_ssim"] == 0.1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\nAll {len(tests)} tests passed")
