#!/usr/bin/env python3
"""Core-QRB: a minimal query-conditioned RWKV block for co-registered images.

The block keeps only the three query roles supported by the v9 CoReg experiment:

* R reads the recurrent state;
* Kq lets the query condition what is written into the state;
* A applies a signed, per-direction correction between the recurrent read and
  the co-located value.

There is deliberately no Spatial-Mix G path.  The sigmoid receptance inside
the unchanged Channel Mix is a different mechanism and is retained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.golf_v7r.cross_rwkv_head_v8 import CrossRWKVHeadV8, SPAB
from models.modules.batched_wkv import batched_scan


class WTConvLite(nn.Module):
    """One-level Haar wavelet token mixer used only on the query stream.

    A fixed orthonormal DWT/IWT surrounds one depthwise 3x3 convolution over
    the four sub-bands.  No hand-designed sub-band weights or frequency loss
    are introduced.  ``gamma=0`` makes the mixer an exact identity at
    initialization while allowing it to unlock on the first optimization step.
    """

    def __init__(self, channels: int):
        super().__init__()
        haar = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ).unsqueeze(1) / 2.0
        self.channels = int(channels)
        self.register_buffer("haar", haar, persistent=False)
        self.subband_conv = nn.Conv2d(
            channels * 4, channels * 4, 3, 1, 1,
            groups=channels * 4, bias=False,
        )
        nn.init.kaiming_normal_(
            self.subband_conv.weight, mode="fan_out", nonlinearity="linear")
        self.gamma = nn.Parameter(torch.zeros(()))

    def _filters(self, x: torch.Tensor) -> torch.Tensor:
        return self.haar.to(dtype=x.dtype).repeat(self.channels, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h, pad_w = h % 2, w % 2
        padded = F.pad(x, (0, pad_w, 0, pad_h)) if pad_h or pad_w else x
        filters = self._filters(padded)
        bands = F.conv2d(padded, filters, stride=2, groups=self.channels)
        bands = self.subband_conv(bands)
        restored = F.conv_transpose2d(
            bands, filters, stride=2, groups=self.channels)
        restored = restored[..., :h, :w]
        return x + self.gamma * restored


class CoreQueryRWKVHead(CrossRWKVHeadV8):
    """Minimal co-registered Query-RWKV head without a Spatial-Mix G path.

    Args:
        query_mixer: local mixer shared by the query-side R/Kq/A roles.
            ``current`` is the v9 bridge (OmniShift -> SPAB), ``omni`` keeps
            only OmniShift, ``span`` keeps only SPAB, and ``wt`` uses
            :class:`WTConvLite`.

    ``center_feat`` (Q) and ``context_feat`` (X) must share the same image grid.
    The context-side local mixer remains the existing OmniShift/RLK-TS path and
    is shared by K and V.
    """

    _QUERY_MIXERS = ("current", "omni", "span", "wt")

    def __init__(self, channels: int, query_mixer: str = "current", **kwargs):
        if query_mixer not in self._QUERY_MIXERS:
            raise ValueError(
                f"query_mixer must be one of {self._QUERY_MIXERS}, "
                f"got {query_mixer!r}")
        super().__init__(channels, **kwargs)
        if query_mixer in ("current", "omni") and not self.use_omni_shift:
            raise ValueError(
                f"query_mixer={query_mixer!r} requires use_omni_shift=True")
        self.query_mixer_kind = query_mixer

        # Reuse the already initialized v9 local modules, but register each
        # source exactly once.  K/V share context_shift; R/Kq/A share one query
        # mixer.  The old G modules are not retained as dormant parameters.
        query_shift = self.shift_r if query_mixer in ("current", "omni") else None
        query_spab = self.spab_r if query_mixer in ("current", "span") else None
        if query_spab is None and query_mixer in ("current", "span"):
            query_spab = SPAB(channels)
        context_shift = self.shift_k
        for name in ("shift_r", "shift_g", "shift_k", "shift_v", "spab_r"):
            delattr(self, name)
        del self.proj_g

        self.query_shift = query_shift
        self.query_spab = query_spab
        self.query_wt = WTConvLite(channels) if query_mixer == "wt" else None
        self.context_shift = context_shift

        # K = Kx(X) + Kq(Q).  The zero-initialized Kq branch makes the initial
        # write context-only, while its weights receive gradients immediately.
        self.proj_kq = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.proj_kq.weight)

        # One signed coefficient per pixel and scan direction.  Unlike the old
        # sigmoid rho multiplied by a second scale, this directly represents
        # both attraction to and subtraction from the co-located value.
        self.proj_align = nn.Linear(channels * 2, self.num_dirs, bias=True)
        nn.init.zeros_(self.proj_align.weight)
        nn.init.zeros_(self.proj_align.bias)
        self._last_core = None

    def _mix_query(self, x: torch.Tensor) -> torch.Tensor:
        if self.query_shift is not None:
            x = self.query_shift(x)
        if self.query_spab is not None:
            x = self.query_spab(x)[0]
        if self.query_wt is not None:
            x = self.query_wt(x)
        return x

    def _encode_inputs(self, query: torch.Tensor, context: torch.Tensor):
        if query.shape != context.shape:
            raise ValueError(
                "Core-QRB requires co-registered Query/Context tensors with "
                f"the same shape, got {tuple(query.shape)} and {tuple(context.shape)}")
        q_2d = self._mix_query(query)
        x_2d = self.context_shift(context) if self.context_shift is not None else context
        q = self.pre_norm(q_2d.flatten(2).transpose(1, 2))
        x = self.pre_norm(x_2d.flatten(2).transpose(1, 2))
        return q, x

    def forward(self, center_feat: torch.Tensor,
                context_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = center_feat.shape
        d, head_dim = self.num_dirs, self.head_dim
        q_tokens, x_tokens = self._encode_inputs(center_feat, context_feat)

        r = self.proj_r(q_tokens).view(b, h * w, d, head_dim)
        kx = self.proj_k(x_tokens)
        kq = self.proj_kq(q_tokens)
        k = kx + kq
        v = self.proj_v(x_tokens)
        align = torch.tanh(self.proj_align(torch.cat((q_tokens, x_tokens), dim=-1)))

        scan_fns = self._scan_fns()
        k_list, v_list = [], []
        for i, scan_fn in enumerate(scan_fns):
            c0, c1 = i * head_dim, (i + 1) * head_dim
            k_2d = k[:, :, c0:c1].transpose(1, 2).reshape(b, head_dim, h, w)
            v_2d = v[:, :, c0:c1].transpose(1, 2).reshape(b, head_dim, h, w)
            k_list.append(scan_fn(k_2d))
            v_list.append(scan_fn(v_2d))
        wkv_list = batched_scan(
            self.bi_wkv_list, k_list, v_list, total_tokens=h * w)

        heads = []
        for i, scan_fn in enumerate(scan_fns):
            recurrent = wkv_list[i][:, self._inv_index(
                scan_fn, b, head_dim, h, w, center_feat.device)]
            value = v[:, :, i * head_dim:(i + 1) * head_dim]
            a_i = align[:, :, i:i + 1]
            corrected = recurrent + a_i * (value - recurrent)
            heads.append(self.head_norm(r[:, :, i] * corrected))
        out = torch.cat(heads, dim=-1)
        out = self.proj_out(out).transpose(1, 2).reshape(b, c, h, w)
        out = out + center_feat

        # Keep the validated v9 Channel Mix unchanged.  Its sigmoid(r2) is a
        # channel-mixing receptance, not the removed Spatial-Mix G branch.
        if self.use_channel_mix:
            cm_in = self.cm_shift(out) if self.cm_shift is not None else out
            tokens = self.post_norm(cm_in.flatten(2).transpose(1, 2))
            r2 = self.cm_r(tokens)
            k2 = self.cm_k(tokens)
            if self.cm_act == "relu_sq":
                v2 = F.relu(k2).pow(2)
            else:
                v2 = F.leaky_relu(k2, 0.01).pow(2)
            mixed = torch.sigmoid(r2) * self.cm_v(v2)
            out = out + self.cm_out(mixed).transpose(1, 2).reshape(b, c, h, w)

        with torch.no_grad():
            kx_rms = kx.detach().square().mean().sqrt()
            kq_rms = kq.detach().square().mean().sqrt()
            self._last_core = {
                "align_mean": align.detach().mean(),
                "align_abs_mean": align.detach().abs().mean(),
                "align_min": align.detach().amin(),
                "align_max": align.detach().amax(),
                "align_positive_fraction": (align.detach() > 0).float().mean(),
                "kq_rms": kq_rms,
                "kx_rms": kx_rms,
                "kq_to_kx": kq_rms / kx_rms.clamp_min(1e-8),
            }
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    head = CoreQueryRWKVHead(32, wkv_mode="legacy")
    q = torch.randn(1, 32, 16, 16, requires_grad=True)
    x = torch.randn(1, 32, 16, 16, requires_grad=True)
    y = head(q, x)
    y.mean().backward()
    print(f"Core-QRB: {tuple(y.shape)}, params={sum(p.numel() for p in head.parameters()):,}")
