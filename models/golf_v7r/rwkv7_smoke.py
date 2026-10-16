#!/usr/bin/env python3
"""v8 C: RWKV-7 与 Value Residual 的搬运与冒烟测试 (docs/v8/01-设计.md §3.4)。

## 目的

先期只确认"能跑通、参数正常、反传正常", 不参与正式对照训练。
搬运自本地参考实现, 不做任何"顺手修正", 以便后续若采纳时结果可复现。

来源 (逐行对照):
    reference_repos/RWKV-LM/RWKV-v7/rwkv_v7_demo.py
      - `RWKV7_OP`            第 168–203 行 (矩阵状态 + delta 规则)
      - `RWKV_Tmix_x070`      第 210–288 行 (含 Value Residual, 见第 274–277 行)
      - `RWKV_CMix_x070`      第 292 行起 (Channel Mix)

## 相对原实现的改动 (只做序列→图像的必要适配, 不改算法)

原实现是语言模型, 沿时间轴做 token shift (`nn.ZeroPad2d((0,0,1,-1))`)。
图像没有时间轴, 时序 RWKV 用 "前一时刻" 混入当前输入, 这里改用**沿扫描序列的
前一位**做同样的 shift (即把一维扫描序列当作序列, shift 即取前一个 token)。
这与 OmniShift 补局部邻域是两回事: token shift 是序列内的, OmniShift 是空间邻域。

除此之外, 衰减、delta 规则、a/b 门控、值残差、低秩分解 (LoRA) 全部按原式搬运。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 与参考实现一致的默认低秩分解维度 (rwkv_v7_demo.py 的 D_DECAY_LORA 等)
D_DECAY_LORA = 64
D_AAA_LORA = 64
D_MV_LORA = 32
D_GATE_LORA = 128
HEAD_SIZE = 64


def rwkv7_op(r: torch.Tensor, w: torch.Tensor, k: torch.Tensor,
             v: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
             head_size: int) -> torch.Tensor:
    """RWKV-7 的矩阵状态递推 (对照 rwkv_v7_demo.py 第 168–203 行的纯 PyTorch 分支)。

        state_t = state_{t-1} ⊙ w_t + state_{t-1} @ a_t @ b_t + v_t ⊗ k_t
        out_t   = state_t @ r_t

    r/w/k/v/a/b: [B, T, C], C = H · N
    返回: [B, T, C]
    """
    B, T, C = r.shape
    H = C // head_size
    N = head_size
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

    for t in range(T):
        kk = k[:, t].view(B, H, 1, N)
        rr = r[:, t].view(B, H, N, 1)
        vv = v[:, t].view(B, H, N, 1)
        aa = a[:, t].view(B, H, N, 1)
        bb = b[:, t].view(B, H, 1, N)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t] = (state @ rr).view(B, H, N)

    return out.reshape(B, T, C)


class RWKV7TimeMixSmoke(nn.Module):
    """RWKV-7 的 Time Mix 单元 (含 Value Residual)。

    与参考实现的差别仅在于 token shift 的实现: 参考沿时间轴取前一 token,
    这里沿传入的序列轴取前一个位置 (图像扫描序列)。
    """

    def __init__(self, dim: int, head_size: int = HEAD_SIZE, layer_id: int = 0):
        super().__init__()
        assert dim % head_size == 0, f"dim({dim}) 必须能被 head_size({head_size}) 整除"
        self.dim = dim
        self.head_size = head_size
        self.n_head = dim // head_size
        self.layer_id = layer_id
        H, C = self.n_head, dim

        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        # Value Residual 的门控参数 (来自 FEAT 的机制, 在 RWKV-7 里原生存在)
        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, self.head_size))

        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        self._reset_parameters()

    def _reset_parameters(self):
        """按参考实现 `RWKV_x070` 的 `_init_weights` 惯例做小初始化。

        参考里各 LoRA 矩阵用 0.01 量级的均匀初始化, 门控偏置为 0, 线性层用默认初始化。
        这里保持同样的量级, 避免矩阵状态在早期发散。
        """
        for name, p in self.named_parameters():
            if name.endswith(("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")):
                with torch.no_grad():
                    p.uniform_(-1.0, 1.0)
            elif name.endswith(("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2")):
                with torch.no_grad():
                    p.uniform_(-0.01, 0.01)
            elif name.endswith(("w0", "a0", "v0")):
                with torch.no_grad():
                    p.zero_()
            elif name.endswith("k_k"):
                with torch.no_grad():
                    p.fill_(1.0)
            elif name.endswith("k_a"):
                with torch.no_grad():
                    p.fill_(1.0)
            elif name.endswith("r_k"):
                with torch.no_grad():
                    p.uniform_(-1.0, 1.0)
        for lin in (self.receptance, self.key, self.value, self.output):
            nn.init.uniform_(lin.weight, -0.1, 0.1)

    @staticmethod
    def _token_shift(x: torch.Tensor) -> torch.Tensor:
        """序列内前一位 shift: 参考用 ZeroPad2d((0,0,1,-1)), 这里沿 T 轴等价实现。"""
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        return shifted

    def forward(self, x: torch.Tensor, v_first: torch.Tensor | None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, C] (T 为扫描序列长度); v_first: 第一层保存的 value, 或 None。

        返回 (输出 [B,T,C], 本层更新后的 v_first)。
        """
        B, T, C = x.shape
        xx = self._token_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            # Value Residual: 首层的 value 经门控混入后续每一层。
            # 注意 layer_id == 0 的 v0/v1/v2 不参与计算, 因此它们不会有梯度 ——
            # 参考实现同样如此 (layer 0 只负责保存 v_first), 这不是缺陷。
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, self.n_head, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        out = rwkv7_op(r, w, k, v, -kk, kk * a, self.head_size)
        out = self.ln_x(out.reshape(B * T, C)).reshape(B, T, C)
        out = out + ((r.view(B, T, self.n_head, -1)
                      * k.view(B, T, self.n_head, -1)
                      * self.r_k).sum(dim=-1, keepdim=True)
                     * v.view(B, T, self.n_head, -1)).reshape(B, T, C)
        out = self.output(out * g)
        return out, v_first


class RWKV7ChannelMixSmoke(nn.Module):
    """RWKV-7 的 Channel Mix (对照 rwkv_v7_demo.py 的 `RWKV_CMix_x070`)。"""

    def __init__(self, dim: int, hidden_rate: int = 4):
        super().__init__()
        hidden = int(dim * hidden_rate)
        self.x_k = nn.Parameter(torch.zeros(1, 1, dim))
        self.key = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(hidden, dim, bias=False)

    @staticmethod
    def _token_shift(x: torch.Tensor) -> torch.Tensor:
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        return shifted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self._token_shift(x) - x
        k = self.key(x + xx * self.x_k)
        return self.value(torch.relu(k) ** 2)


if __name__ == '__main__':
    torch.manual_seed(0)
    B, T, C = 2, 64, 128
    x = torch.randn(B, T, C)

    print("=== RWKV-7 Time Mix (含 Value Residual) 冒烟 ===")
    tr = torch.nn.ModuleList([
        RWKV7TimeMixSmoke(C, head_size=64, layer_id=0),
        RWKV7TimeMixSmoke(C, head_size=64, layer_id=1),
    ])
    h = x
    v_first = None
    for i, blk in enumerate(tr):
        h, v_first = blk(h, v_first)
        print(f"  layer{i}: {tuple(x.shape)} -> {tuple(h.shape)}"
              + (" , 已保存 v_first" if i == 0 else " , 值残差生效"))
    h.sum().backward()
    dead = [n for n, p in tr.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    print(f"  首步无梯度参数: {dead if dead else '无'}")
    gate = torch.sigmoid(tr[1].v0)
    print(f"  Value Residual 初始门控 (v0=0 时应为 0.5): {gate.mean().item():.4f}")

    print("=== RWKV-7 Channel Mix 冒烟 ===")
    cm = RWKV7ChannelMixSmoke(C)
    yc = cm(x)
    print(f"  {tuple(x.shape)} -> {tuple(yc.shape)}")
    yc.sum().backward()
    dead_c = [n for n, p in cm.named_parameters()
              if p.grad is None or p.grad.abs().sum() == 0]
    print(f"  首步无梯度参数: {dead_c if dead_c else '无'}")
