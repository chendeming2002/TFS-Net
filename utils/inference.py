import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


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


@torch.no_grad()
def tiled_forward(model, clip, tile_size=256, tile_overlap=32, use_amp=False,
                  frame_indices=None):
    if tile_size is None or tile_size <= 0:
        with autocast(enabled=use_amp and clip.is_cuda):
            return model(clip, frame_indices=frame_indices)["res_t"]

    clip, pad_hw, original_hw = _pad_clip_for_tiling(clip, tile_size)
    b, t, c, h, w = clip.shape
    if b != 1:
        raise ValueError("tiled_forward currently expects batch size 1.")

    h_starts = _compute_starts(h, tile_size, tile_overlap)
    w_starts = _compute_starts(w, tile_size, tile_overlap)

    # 帧缓存仅在单 tile 时有效（多 tile 的空间裁剪不同，缓存特征不可复用）
    use_cache = (len(h_starts) == 1 and len(w_starts) == 1)
    cache_indices = frame_indices if use_cache else None

    output = clip.new_zeros((b, c, h, w))
    weight = clip.new_zeros((b, 1, h, w))

    for top in h_starts:
        for left in w_starts:
            tile = clip[:, :, :, top : top + tile_size, left : left + tile_size]
            with autocast(enabled=use_amp and clip.is_cuda):
                tile_pred = model(tile, frame_indices=cache_indices)["res_t"]
            output[:, :, top : top + tile_size, left : left + tile_size] += tile_pred
            weight[:, :, top : top + tile_size, left : left + tile_size] += 1.0

    output = output / weight.clamp_min(1.0)

    pad_h, pad_w = pad_hw
    orig_h, orig_w = original_hw
    if pad_h > 0:
        output = output[:, :, :orig_h, :]
    if pad_w > 0:
        output = output[:, :, :, :orig_w]
    return output

