import os

import numpy as np
import torch
from PIL import Image


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state, path):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def save_image_tensor(tensor, path):
    ensure_dir(os.path.dirname(path))
    image = tensor.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(path)



def load_state_dict_compat(model, state):
    """加载 state_dict, 忽略旧 checkpoint 里已删除的 omni_shift 键。

    cross_rwkv 查询头早期版本继承了一个从未参与前向的 `omni_shift`, 现已删除。
    旧 checkpoint 仍带有这些键, 直接加载会报 "unexpected keys", 因此在这里过滤掉。

    判断依据是同一个头是否存在 shift_r 键: cross_rwkv 头用 shift_r/g/k/v 四份
    独立 Q-Shift, 没有基类那份; receptance 头则相反, 它的 omni_shift 是必需参数。
    编码器与聚合器的空间头不属于查询头, 它们的 omni_shift 必须保留。
    其余键一律严格匹配, 不做其他宽松处理。
    """
    stale = []
    for k in state:
        if ".omni_shift." not in k:
            continue
        prefix = k.split(".omni_shift.")[0]          # 例如 tca.attn_N
        if not prefix.startswith("tca.attn_"):
            continue
        if any(key.startswith(prefix + ".shift_r.") for key in state):
            stale.append(k)
    # 早期版本的推理合并卷积被注册成了子模块, 它的权重可能被存进 checkpoint。
    # 那是推理期生成的等价核, 不应参与加载。
    stale += [k for k in state if ".conv_merged." in k]
    if stale:
        state = {k: v for k, v in state.items() if k not in stale}
    return model.load_state_dict(state)
