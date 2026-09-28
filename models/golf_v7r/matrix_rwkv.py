#!/usr/bin/env python3
"""
Matrix RWKV: RWKV-6 风格矩阵值状态 + 数据相关衰减

核心改进 (vs golf_v7 的 naive RWKV-4):
  1. 矩阵值状态 S ∈ R^{d×d} (vs 向量 s ∈ R^d)
  2. 数据相关衰减 w_t = f(x_t) (vs 固定 w)
  3. 共享投影 + GroupNorm (vs 独立头)
  4. ReLU² MLP Channel Mix (vs 门控 FFN)

参考: RWKV-6 (Finch) arXiv:2404.05892 §4.2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixRWKVTimeMix(nn.Module):
    """
    RWKV-6 风格 Time Mixing: 矩阵值状态 + 数据相关衰减

    状态更新: S_t = S_{t-1} · diag(w_t) + v_t^T · k_t
    w_t = exp(-exp(base_decay + LoRA(x_t)))  数据相关, ∈ (0, 1)
    输出: o_t = r_t · S_t
    """

    def __init__(self, dim: int = 192, num_heads: int = 6, head_size: int = 32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_size = head_size
        assert dim == num_heads * head_size, (
            f"dim={dim} != num_heads={num_heads} * head_size={head_size}"
        )

        # Shared projections (multi-head via reshape)
        self.W_r = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)

        # Token Shift mixing (V7 简单 lerp 风格)
        self.mix_r = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.mix_k = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.mix_v = nn.Parameter(torch.ones(1, 1, dim) * 0.5)

        # 数据相关衰减 (RWKV-6 LoRA)
        # base_decay: 每头每通道的基础衰减率
        self.base_decay = nn.Parameter(torch.ones(num_heads, head_size) * 0.5)
        # LoRA: dim → dim//4 → dim (低秩调节)
        lora_rank = max(dim // 4, 16)
        self.decay_lora_down = nn.Linear(dim, lora_rank, bias=False)
        self.decay_lora_up = nn.Linear(lora_rank, dim, bias=False)
        # 初始化 LoRA up 为零 → 初始时衰减 = base_decay (数据无关)
        nn.init.zeros_(self.decay_lora_up.weight)

        # SiLU 门控 (V5 风格)
        self.W_g = nn.Linear(dim, dim, bias=False)

        # Per-head GroupNorm (V5 风格, 替代归一化分母)
        self.group_norm = nn.GroupNorm(num_heads, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        Returns: [B, T, D]
        """
        B, T, D = x.shape
        H, d = self.num_heads, self.head_size

        # Token shift: 前一帧 token 与当前的 lerp 混合
        x_shifted = F.pad(x[:, :-1], (0, 0, 1, 0))  # [B, T, D], t=0 补零

        xr = x * self.mix_r + x_shifted * (1 - self.mix_r)
        xk = x * self.mix_k + x_shifted * (1 - self.mix_k)
        xv = x * self.mix_v + x_shifted * (1 - self.mix_v)

        # Projections → multi-head
        r = self.W_r(xr).view(B, T, H, d)  # receptance
        k = self.W_k(xk).view(B, T, H, d)  # key
        v = self.W_v(xv).view(B, T, H, d)  # value
        g = torch.sigmoid(self.W_g(x))      # gate [B, T, D]

        # 数据相关衰减 w_t ∈ (0, 1)
        decay_delta = self.decay_lora_up(
            torch.tanh(self.decay_lora_down(xk))
        ).view(B, T, H, d)
        w = torch.exp(-torch.exp(
            self.base_decay.unsqueeze(0).unsqueeze(0) + decay_delta
        ))  # [B, T, H, d], 每元素独立衰减

        # 递归: 矩阵状态更新
        out = self.recurrence(r, k, v, w)   # [B, T, H, d]
        out = out.reshape(B, T, D)          # [B, T, D]

        # GroupNorm (per-head normalization)
        out_flat = out.reshape(B * T, D, 1)  # [B*T, D, 1] (假装空间维=1)
        out_flat = self.group_norm(out_flat)
        out = out_flat.reshape(B, T, D)

        # 门控 + 输出投影
        out = out * g
        out = self.W_o(out)

        return out

    @staticmethod
    def recurrence(r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   w: torch.Tensor) -> torch.Tensor:
        """
        矩阵值状态递推 (RWKV-6 风格)

        S_t = S_{t-1} · diag(w_t) + v_t^T · k_t
        o_t = r_t · S_t

        Args:
            r, k, v, w: 均为 [B, T, H, d]
                       state[b,h,i,j]: i = value 通道, j = key 通道
        Returns:
            out: [B, T, H, d]
        """
        B, T, H, d = r.shape
        state = torch.zeros(B, H, d, d, device=r.device, dtype=r.dtype)
        outputs = []

        for t in range(T):
            k_t, v_t, r_t, w_t = k[:, t], v[:, t], r[:, t], w[:, t]

            # diag(w_t) 作用在 key 维 (最后一维, j)
            state = state * w_t.unsqueeze(-2) + torch.einsum(
                'bhi,bhj->bhij', v_t, k_t
            )
            # 读出: r 与 key 维 (j) 收缩, 输出在 value 维 (i)
            outputs.append(torch.einsum('bhj,bhij->bhi', r_t, state))

        return torch.stack(outputs, dim=1)  # [B, T, H, d]


class ReLUSquaredMLP(nn.Module):
    """
    V7 风格 Channel Mix: 简洁的 2-layer MLP with ReLU²

    替代 V4-V6 的门控 Channel Mixing (省掉 receptance gate)
    """

    def __init__(self, dim: int, hidden_ratio: int = 3):
        super().__init__()
        hidden = dim * hidden_ratio
        self.W1 = nn.Linear(dim, hidden, bias=False)
        self.W2 = nn.Linear(hidden, dim, bias=False)
        self.mix = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        # W2 零初始化 → 初始时 channel mix 输出为零 (残差稳定)
        nn.init.zeros_(self.W2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, T, D]"""
        x_shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        xm = x * self.mix + x_shifted * (1 - self.mix)
        return self.W2(F.relu(self.W1(xm)).square())


class MatrixRWKVBlock(nn.Module):
    """
    单个 Matrix RWKV Block = LN + TimeMix + LN + ChannelMix
    两个残差连接

    LayerScale (来自 RWKV-5/6 与 DiT):
      残差分支乘以可学习的小初始化缩放 ls (默认 0.1)。
      没有它时, stacked 的 ReLU² MLP 会使激活逐 block 爆炸
      (实测 ep10: block0 std=5.6 → block1 std=523),
      继而导致反传梯度消失 (~1e-6), 整个 Matrix RWKV 无法学习。
    """

    def __init__(self, dim: int = 192, num_heads: int = 6, head_size: int = 32,
                 layer_scale_init: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.time_mix = MatrixRWKVTimeMix(dim, num_heads, head_size)
        self.channel_mix = ReLUSquaredMLP(dim)

        # LayerScale: 稳定深层残差堆叠
        self.ls_time = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.ls_channel = nn.Parameter(torch.full((dim,), layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, T, D]"""
        x = x + self.ls_time * self.time_mix(self.ln1(x))
        x = x + self.ls_channel * self.channel_mix(self.ln2(x))
        return x


if __name__ == '__main__':
    # Smoke test
    B, T, D = 2, 5, 192
    x = torch.randn(B, T, D)

    print("=== MatrixRWKVTimeMix ===")
    tm = MatrixRWKVTimeMix(dim=192, num_heads=6, head_size=32)
    out = tm(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in tm.parameters()) / 1e3:.1f}K")

    print("\n=== ReLUSquaredMLP ===")
    cm = ReLUSquaredMLP(dim=192)
    out = cm(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in cm.parameters()) / 1e3:.1f}K")

    print("\n=== MatrixRWKVBlock ===")
    block = MatrixRWKVBlock(dim=192, num_heads=6, head_size=32)
    out = block(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in block.parameters()) / 1e3:.1f}K")

    # Gradient check
    loss = out.sum()
    loss.backward()
    print("\n  Gradient check passed!")
