#!/usr/bin/env python3
"""SGRFv10: minimal fixed-order image-state restoration.

The module starts from the observed center frame and applies three signed RGB
corrections in the fixed TSDR order N -> L -> M.  It does not extract another
set of degradation features, bound correction magnitudes, learn stage scales,
or impose a gate regularizer.

S0 uses ``gate_mode='fixed'`` (all corrections are fully accepted).  S1 uses
one shared single-channel gate conditioned only on ``[state, delta]``.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SharedCorrectionGate(nn.Module):
    """Shared spatial acceptance gate: [RGB state, RGB delta] -> one channel."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor,
                delta: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat((state, delta), dim=1)))


class SGRFv10(nn.Module):
    """Apply N/L/M corrections to an image state without hand-built constraints.

    Args:
        gate_mode: ``fixed`` for S0 (G=1) or ``shared`` for S1.
        gate_hidden: hidden width of the shared S1 gate.  The default gives
            exactly 449 trainable gate parameters.
    """

    stage_order = "NLM"

    def __init__(self, gate_mode: str = "shared", gate_hidden: int = 8):
        super().__init__()
        if gate_mode not in ("fixed", "shared"):
            raise ValueError(
                f"gate_mode must be 'fixed' or 'shared', got {gate_mode!r}")
        self.gate_mode = gate_mode
        self.gate = (SharedCorrectionGate(gate_hidden)
                     if gate_mode == "shared" else None)

    @staticmethod
    def _check_delta(name: str, state: torch.Tensor,
                     delta: torch.Tensor) -> None:
        if delta.shape != state.shape:
            raise ValueError(
                f"SGRFv10 stage {name} requires an RGB delta shaped like the "
                f"state, got {tuple(delta.shape)} vs {tuple(state.shape)}")

    def _acceptance(self, state: torch.Tensor,
                    delta: torch.Tensor) -> torch.Tensor:
        if self.gate is None:
            return state.new_ones((state.shape[0], 1, *state.shape[-2:]))
        return self.gate(state, delta)

    def forward(self, feats: Dict[str, torch.Tensor], X_t: torch.Tensor,
                branch_N: nn.Module, branch_L: nn.Module, branch_M: nn.Module,
                F2_single: torch.Tensor,
                zero_stage: str | None = None) -> Dict[str, torch.Tensor]:
        if zero_stage is not None and zero_stage not in self.stage_order:
            raise ValueError(f"zero_stage must be N/L/M or None, got {zero_stage!r}")
        missing = set(self.stage_order) - set(feats)
        if missing:
            raise ValueError(f"SGRFv10 missing task features: {sorted(missing)}")

        state = X_t
        states: Dict[str, torch.Tensor] = {}
        deltas: Dict[str, torch.Tensor] = {}
        gates: Dict[str, torch.Tensor] = {}
        stats: Dict[str, torch.Tensor] = {}

        for name in self.stage_order:
            if name == "N":
                result = branch_N.predict_delta(feats[name], state)
            elif name == "L":
                result = branch_L.predict_delta(feats[name], state)
            else:
                result = branch_M.predict_delta(
                    feats[name], state, F2_single.unsqueeze(1))

            delta = result["delta"]
            self._check_delta(name, state, delta)
            gate = self._acceptance(state, delta)
            step = gate * delta
            if zero_stage == name:
                step = step * 0.0
            state = state + step

            states[name] = state
            deltas[name] = delta
            gates[name] = gate
            detached_gate = gate.detach()
            stats[f"gate_{name}"] = detached_gate.mean()
            stats[f"gate_{name}_std"] = detached_gate.std(unbiased=False)
            stats[f"gate_{name}_sat"] = (
                (detached_gate < 0.1) | (detached_gate > 0.9)).float().mean()
            stats[f"delta_{name}"] = delta.detach().abs().mean()
            stats[f"inc_{name}"] = step.detach().abs().mean()

        # Keep training states raw.  Evaluation/serialization code is
        # responsible for clamping to the legal image range.
        return {
            "O_t": state,
            "stages": states,
            "deltas": deltas,
            "gates": gates,
            "Y_N": states["N"],
            "Y_L": states["L"],
            "Y_M": states["M"],
            "stage_stats": stats,
            "gate_reg": None,
        }


if __name__ == "__main__":
    gate = SharedCorrectionGate()
    print(f"shared gate params={sum(p.numel() for p in gate.parameters())}")
