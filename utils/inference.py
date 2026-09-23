import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import math


def _compute_starts(length, tile_size, overlap):
    if tile_size >= length:
        return [0]
    stride = max(tile_size - overlap, 1)
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _pad_clip_for_tiling(clip, tile_size):
    _, _, _, h, w = clip.shape
    pad_h = max(tile_size - h, 0)
    pad_w = max(tile_size - w, 0)
    if pad_h == 0 and pad_w == 0:
        return clip, (0, 0), (h, w)
    clip = F.pad(clip, (0, pad_w, 0, pad_h), mode="reflect")
    return clip, (pad_h, pad_w), (h, w)


def _cosine_window(size: int, device, dtype) -> torch.Tensor:
    """1-D raised-cosine window, floored at 0.1.

    A pure Hann window drops to exactly 0 at both ends.  At the *image*
    border (covered by a single tile) that would zero the contribution and
    produce a black frame edge.  Flooring at 0.1 keeps the weighted average
    exact at borders (0.1*pred / 0.1 == pred) while remaining perfectly
    smooth inside overlap regions.
    """
    t = torch.arange(size, device=device, dtype=dtype)
    w = 0.5 * (1.0 - torch.cos(2.0 * math.pi * t / (size - 1)))
    return 0.1 + 0.9 * w


def _make_tile_weight(tile_size: int, device, dtype) -> torch.Tensor:
    """2-D raised-cosine weight map for a square tile of shape (1, 1, H, W).

    Centre pixels get weight ≈ 1.0; edges taper smoothly to 0.
    This ensures that the weighted average across overlapping tiles is
    seamless — no step discontinuity at tile boundaries.
    """
    w1d = _cosine_window(tile_size, device=device, dtype=dtype)  # (T,)
    w2d = w1d.unsqueeze(0) * w1d.unsqueeze(1)  # (T, T)
    return w2d.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


@torch.no_grad()
def tiled_forward(model, clip, tile_size=256, tile_overlap=32, use_amp=False,
                  amp_dtype=torch.float16, frame_indices=None, phase='phase2'):
    """Run model on a large clip with raised-cosine-blended tile stitching.

    Replaces the previous uniform-average strategy that caused visible
    brightness step-discontinuities at tile boundaries (the 'checkerboard'
    / 'grid' artifact seen in outputs).

    Each tile is multiplied by a 2-D raised-cosine (Hann) weight before
    accumulation; the final output is the normalised weighted sum.  Because
    the cosine window tapers smoothly to zero at edges, contributions from
    adjacent tiles blend without seams.
    
    Args:
        amp_dtype: torch.float16 or torch.bfloat16 (default: fp16)
    """
    if tile_size is None or tile_size <= 0:
        with autocast(enabled=use_amp and clip.is_cuda, dtype=amp_dtype if use_amp else torch.float32):
            return model(clip, frame_indices=frame_indices, phase=phase)["res_t"]

    clip, pad_hw, original_hw = _pad_clip_for_tiling(clip, tile_size)
    b, t, c, h, w = clip.shape
    if b != 1:
        raise ValueError("tiled_forward currently expects batch size 1.")

    h_starts = _compute_starts(h, tile_size, tile_overlap)
    w_starts = _compute_starts(w, tile_size, tile_overlap)

    # Cache is only valid when a single tile covers the whole frame
    use_cache = (len(h_starts) == 1 and len(w_starts) == 1)
    cache_indices = frame_indices if use_cache else None

    output = clip.new_zeros((b, c, h, w))
    weight = clip.new_zeros((b, 1, h, w))

    # Pre-compute the raised-cosine weight map once (reused for all tiles)
    tile_w = _make_tile_weight(tile_size, device=clip.device, dtype=clip.dtype)

    for top in h_starts:
        for left in w_starts:
            tile = clip[:, :, :, top: top + tile_size, left: left + tile_size]

            with autocast(enabled=use_amp and clip.is_cuda, dtype=amp_dtype if use_amp else torch.float32):
                tile_pred = model(tile, frame_indices=cache_indices, phase=phase)["res_t"]

            output[:, :, top: top + tile_size, left: left + tile_size] += tile_pred * tile_w
            weight[:, :, top: top + tile_size, left: left + tile_size] += tile_w

    output = output / weight.clamp_min(1e-8)

    pad_h, pad_w = pad_hw
    orig_h, orig_w = original_hw
    if pad_h > 0:
        output = output[:, :, :orig_h, :]
    if pad_w > 0:
        output = output[:, :, :, :orig_w]
    return output
