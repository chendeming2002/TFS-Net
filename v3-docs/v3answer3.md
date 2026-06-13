# TFS-Net v3 Phase B — 启动确认与 B.1 实现

## 〇、N1-N3 修正理解确认

在开始编码前，我先复述对三个 P0 修正的理解，请确认无误：

### N1 理解 ✅
**问题**：`r_grid = sqrt(fx² + fy²)` 在 `fftfreq` 网格上的最大值约 `0.5*sqrt(2) ≈ 0.707`（Nyquist 频率），而我之前的代码让 `mu ∈ [0, 1]` 直接与未归一化的 r_grid 比较，导致 `mu > 0.707` 的基函数永远无法激活。
**修正**：在生成 r_grid 后立即归一化 `r_hat = r_grid / r_grid.max()`，使值域严格落入 [0, 1]。

### N2 理解 ✅
**问题**：我之前用两个独立 `RadialBasisFilter` 实例分别处理 mag 和 phase，导致 mu、log_bwh、basis 计算被重复且参数化不一致。FRBNet 原意是**同一组径向基**对幅度/相位采用**不同系数**进行频域整形。
**修正**：单个 RBF 实例内部维护 `coeff_mag`/`coeff_phase`/`raw_gate_mag`/`raw_gate_phase`，前向时一次性返回 `(diff_mag, diff_phase)`，共享 `mu/log_bwh/angular_mod/basis` 计算。

### N3 理解 ✅
**问题**：Retinexformer 的 depth_conv 使用 `groups=n_fea_in`（默认 4）是有意的分组卷积设计——31 通道分 4 组（约 8 通道/组），组间有信息交流。我之前误改为 `groups=n_fea_middle` 会变成纯逐通道卷积，破坏原始设计。
**修正**：IFPN 的 IllumExtract 保留 `groups=4`。

---

下面开始 **B.1: LFF 模块**。

---

## B.1: `models/modules/lff.py`

```python
"""
LFF (Local Frequency Feature) Module — TFS-Net v3
====================================================
基于 FRBNet (Frequency-domain Radial Basis Network) 的频率域特征适配器。

实现状态:
    ✅ RadialBasisFilter: 共享 basis 的单实例双输出 RBF (修正 N2)
    ✅ LFFFeatureAdapter: 频率域特征整形 (修正 N1: r_hat 归一化)

设计原则:
    1. 一个 RBF 实例同时输出 mag/phase 响应，共享 mu/log_bwh/basis (N2)
    2. r_hat 归一化到 [0, 1]，与 mu ∈ [0, 1] 对齐 (N1)
    3. 角度调制通过 cos(n*theta) 实现方向选择性
    4. 频率域整形不改变特征维度: (B, C, H, W) → (B, C, H, W)

参考: reference_repos/FRBNet/frbnet_utils.py L7-47
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.fft as fft


class RadialBasisFilter(nn.Module):
    """
    径向基频率滤波器 — 一个实例同时输出幅度响应和相位响应 (N2 修正)。

    所有基函数 (mu, bwh, angular_mod, basis) 在 mag/phase 之间共享，
    仅 coeff 和 gate 独立，使物理含义为:
        "同一组频带，不同的幅度/相位整形系数"

    Args:
        K (int): 径向基函数个数 (沿径向频率轴均匀分布)
        n_ang_freq (int): 角度调制频率，0 表示各向同性

    Forward:
        Input:  H (int), W (int), device, dtype
        Output: (diff_mag, diff_phase), 各 shape (1, H, W)
                可直接广播乘到傅里叶系数上
    """

    def __init__(self, K: int = 16, n_ang_freq: int = 0):
        super().__init__()
        self.K = K
        self.n_ang_freq = n_ang_freq

        # 共享: 径向基中心 (固定，均匀分布在归一化频率轴上)
        mu = torch.linspace(0.0, 1.0, steps=K)
        self.register_buffer('mu', mu)

        # 共享: 带宽 (可学习，全部基函数共享同一带宽)
        # 初始化为 log(0.05) 使初始 bwh ≈ 0.05，即每个基函数覆盖 ~5% 频率范围
        self.log_bwh = nn.Parameter(torch.tensor(math.log(0.05)))

        # 独立: 幅度通路的系数和门控
        self.coeff_mag = nn.Parameter(torch.zeros(K))      # 初始为 0，使整形从恒等开始
        self.raw_gate_mag = nn.Parameter(torch.ones(K))    # sigmoid 前的 logit，初始 ≈ 0.73 激活

        # 独立: 相位通路的系数和门控
        self.coeff_phase = nn.Parameter(torch.zeros(K))
        self.raw_gate_phase = nn.Parameter(torch.ones(K))

    @torch.no_grad()
    def _build_freq_grid(self, H: int, W: int, device, dtype):
        """构造归一化频率网格 r_hat 和角度 theta"""
        fy = fft.fftfreq(H, device=device, dtype=dtype).view(H, 1).expand(H, W)  # (H, W)
        fx = fft.fftfreq(W, device=device, dtype=dtype).view(1, W).expand(H, W)  # (H, W)
        r_grid = torch.sqrt(fx ** 2 + fy ** 2)                                    # (H, W)
        # ─── N1 修正: 归一化到 [0, 1] ───
        r_hat = r_grid / (r_grid.max() + 1e-8)                                    # (H, W) ∈ [0, 1]
        theta = torch.atan2(fy, fx)                                               # (H, W) ∈ [-π, π]
        return r_hat, theta

    def forward(self, H: int, W: int, device, dtype):
        # Step 1: 构造频率网格 (共享给 mag/phase)
        r_hat, theta = self._build_freq_grid(H, W, device, dtype)
        # r_hat: (H, W), theta: (H, W)

        # Step 2: 计算径向基函数 (共享)
        bwh = torch.exp(self.log_bwh).clamp(min=1e-3, max=1.0)                    # scalar
        # basis: (K, H, W)
        basis = torch.exp(
            -((r_hat.unsqueeze(0) - self.mu.view(-1, 1, 1)) ** 2)
            / (2 * bwh ** 2)
        )

        # Step 3: 角度调制 (共享)
        if self.n_ang_freq > 0:
            angular_mod = torch.cos(self.n_ang_freq * theta)                      # (H, W)
            basis = basis * angular_mod.unsqueeze(0)                              # (K, H, W)

        # Step 4: 幅度通路加权求和
        gate_mag = torch.sigmoid(self.raw_gate_mag)                               # (K,)
        diff_mag = (
            gate_mag.view(-1, 1, 1) * self.coeff_mag.view(-1, 1, 1) * basis
        ).sum(dim=0, keepdim=True)                                                # (1, H, W)

        # Step 5: 相位通路加权求和
        gate_phase = torch.sigmoid(self.raw_gate_phase)                           # (K,)
        diff_phase = (
            gate_phase.view(-1, 1, 1) * self.coeff_phase.view(-1, 1, 1) * basis
        ).sum(dim=0, keepdim=True)                                                # (1, H, W)

        return diff_mag, diff_phase


class LFFFeatureAdapter(nn.Module):
    """
    Local Frequency Feature 特征适配器 — TFSI 频域分支的核心。

    将输入特征通过 2D FFT 转到频率域，
    用 RBF 对幅度和相位做加性整形（残差形式，保留原始信息），
    再 IFFT 回空间域。

    数学形式:
        F = FFT2(x)                       # 复数频谱
        mag, phase = |F|, ∠F
        mag'   = mag * (1 + diff_mag)     # 幅度残差整形
        phase' = phase + diff_phase       # 相位残差整形
        F' = mag' * exp(i * phase')
        x' = Real(IFFT2(F'))

    Args:
        channels (int): 输入特征通道数 (不变)
        K (int): RBF 基函数个数
        n_ang_freq (int): 角度调制频率
        per_channel_rbf (bool): 是否每通道独立 RBF（默认 False，全通道共享一组）

    Forward:
        Input:  x (B, C, H, W)
        Output: x' (B, C, H, W)，shape 完全一致
    """

    def __init__(
        self,
        channels: int,
        K: int = 16,
        n_ang_freq: int = 0,
        per_channel_rbf: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.per_channel_rbf = per_channel_rbf

        if per_channel_rbf:
            # 每通道一个 RBF (参数量 = C * K * 4)
            self.rbf_bank = nn.ModuleList([
                RadialBasisFilter(K=K, n_ang_freq=n_ang_freq)
                for _ in range(channels)
            ])
        else:
            # 全通道共享一个 RBF (参数量 = K * 4)，更经济
            self.rbf = RadialBasisFilter(K=K, n_ang_freq=n_ang_freq)

        # 频域整形后的细微通道融合 (1x1 conv)，便于模型微调通道间相对响应
        self.post_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # 初始化为恒等映射 + 零偏置，使初始整体行为接近 identity
        nn.init.eye_(self.post_conv.weight.squeeze(-1).squeeze(-1))
        nn.init.zeros_(self.post_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.channels, f"通道数不匹配: {C} vs {self.channels}"

        device, dtype = x.device, x.dtype

        # Step 1: 2D FFT (在 H, W 维度)
        F = fft.fft2(x, dim=(-2, -1), norm='ortho')                               # (B, C, H, W) complex
        mag = torch.abs(F)                                                         # (B, C, H, W)
        phase = torch.angle(F)                                                     # (B, C, H, W)

        # Step 2: 计算频域响应
        if self.per_channel_rbf:
            # 逐通道独立 RBF
            diff_mag_list, diff_phase_list = [], []
            for c in range(C):
                dm, dp = self.rbf_bank[c](H, W, device, dtype)                     # (1, H, W) each
                diff_mag_list.append(dm)
                diff_phase_list.append(dp)
            diff_mag = torch.stack(diff_mag_list, dim=0).unsqueeze(0)              # (1, C, H, W)
            diff_phase = torch.stack(diff_phase_list, dim=0).unsqueeze(0)          # (1, C, H, W)
        else:
            # 共享 RBF：广播到所有通道
            diff_mag, diff_phase = self.rbf(H, W, device, dtype)                   # (1, H, W) each
            diff_mag = diff_mag.unsqueeze(0)                                       # (1, 1, H, W)
            diff_phase = diff_phase.unsqueeze(0)                                   # (1, 1, H, W)

        # Step 3: 残差整形 (加性残差，避免破坏 identity)
        mag_new = mag * (1.0 + diff_mag)                                           # (B, C, H, W)
        phase_new = phase + diff_phase                                             # (B, C, H, W)

        # Step 4: 重组复数频谱并 IFFT
        F_new = torch.polar(mag_new, phase_new)                                    # (B, C, H, W) complex
        x_freq = fft.ifft2(F_new, dim=(-2, -1), norm='ortho').real                 # (B, C, H, W)

        # Step 5: 后处理融合
        x_out = self.post_conv(x_freq)                                             # (B, C, H, W)

        return x_out
```

---

## B.1 验证伪代码

```python
"""
验证脚本 (建议放在 tests/test_lff.py 或临时测试)
"""

import torch
from models.modules.lff import RadialBasisFilter, LFFFeatureAdapter


def test_rbf_basic():
    """RBF 基础功能：共享 basis 双输出"""
    rbf = RadialBasisFilter(K=16, n_ang_freq=0)
    diff_mag, diff_phase = rbf(H=64, W=64, device='cpu', dtype=torch.float32)
    
    assert diff_mag.shape == (1, 64, 64), f"diff_mag shape 错误: {diff_mag.shape}"
    assert diff_phase.shape == (1, 64, 64), f"diff_phase shape 错误: {diff_phase.shape}"
    
    # N1 验证：r_hat 已归一化，K=16 时 mu[-1]=1.0 应能在 r_hat 最大处激活
    # 由于 coeff 初始为 0，输出应全 0（恒等行为）
    assert torch.allclose(diff_mag, torch.zeros_like(diff_mag), atol=1e-7)
    assert torch.allclose(diff_phase, torch.zeros_like(diff_phase), atol=1e-7)
    
    # N2 验证：共享参数确认
    assert rbf.mu.shape == (16,)
    assert rbf.log_bwh.shape == ()  # scalar
    assert rbf.coeff_mag.shape == (16,) and rbf.coeff_phase.shape == (16,)  # 独立
    
    print("✅ test_rbf_basic 通过")


def test_rbf_angular():
    """带角度调制的 RBF"""
    rbf = RadialBasisFilter(K=8, n_ang_freq=2)
    diff_mag, diff_phase = rbf(H=32, W=32, device='cpu', dtype=torch.float32)
    assert diff_mag.shape == (1, 32, 32)
    # 手动激活 coeff_mag 看是否产生方向性响应
    with torch.no_grad():
        rbf.coeff_mag.fill_(1.0)
    diff_mag2, _ = rbf(H=32, W=32, device='cpu', dtype=torch.float32)
    # 应有非零输出且呈现角度调制
    assert diff_mag2.abs().max() > 0.0
    print("✅ test_rbf_angular 通过")


def test_lff_shape():
    """LFFFeatureAdapter shape 一致性"""
    adapter = LFFFeatureAdapter(channels=48, K=16, n_ang_freq=0)
    x = torch.randn(2, 48, 64, 64)
    y = adapter(x)
    assert y.shape == x.shape, f"shape 不一致: {x.shape} vs {y.shape}"
    print(f"✅ test_lff_shape 通过: {x.shape} → {y.shape}")


def test_lff_identity_init():
    """初始化时应近似恒等映射（coeff=0 + post_conv=eye）"""
    adapter = LFFFeatureAdapter(channels=48, K=16, n_ang_freq=0)
    x = torch.randn(2, 48, 64, 64)
    y = adapter(x)
    # 因 IFFT 数值误差，允许小偏差
    diff = (y - x).abs().mean()
    assert diff < 1e-4, f"初始化非恒等: 平均差异 {diff:.6f}"
    print(f"✅ test_lff_identity_init 通过: 平均差异 {diff:.2e}")


def test_lff_gradient():
    """梯度可回传"""
    adapter = LFFFeatureAdapter(channels=48, K=16, n_ang_freq=0)
    x = torch.randn(2, 48, 64, 64, requires_grad=True)
    y = adapter(x)
    loss = y.mean()
    loss.backward()
    
    assert x.grad is not None and x.grad.abs().mean() > 0
    assert adapter.rbf.coeff_mag.grad is not None
    assert adapter.rbf.coeff_phase.grad is not None
    assert adapter.rbf.log_bwh.grad is not None
    print("✅ test_lff_gradient 通过：所有可学参数都有梯度")


def test_lff_param_count():
    """参数量统计"""
    adapter_shared = LFFFeatureAdapter(channels=48, K=16, n_ang_freq=2, per_channel_rbf=False)
    adapter_perch = LFFFeatureAdapter(channels=48, K=16, n_ang_freq=2, per_channel_rbf=True)
    
    n_shared = sum(p.numel() for p in adapter_shared.parameters())
    n_perch = sum(p.numel() for p in adapter_perch.parameters())
    
    print(f"📊 LFF 参数量 (channels=48, K=16):")
    print(f"   共享 RBF:  {n_shared:,} 参数")
    print(f"   逐通道 RBF: {n_perch:,} 参数")
    # 共享版本 ≈ post_conv(48*48+48) + RBF(16*4+1) = 2,352 + 65 = 2,417
    # 逐通道版本 ≈ post_conv + 48*RBF = 2,352 + 48*65 = 5,472
    assert n_shared < 3000, "共享 RBF 参数量异常"
    assert n_perch < 6000, "逐通道 RBF 参数量异常"


if __name__ == '__main__':
    test_rbf_basic()
    test_rbf_angular()
    test_lff_shape()
    test_lff_identity_init()
    test_lff_gradient()
    test_lff_param_count()
    print("\n🎉 所有 LFF 测试通过")
```

---

## B.1 参数量统计与设计说明

| 配置 | 参数量 | 说明 |
|------|--------|------|
| `LFFFeatureAdapter(C=48, K=16, n_ang_freq=2, shared)` | **~2,417** | TFSI 推荐默认 |
| `LFFFeatureAdapter(C=48, K=16, n_ang_freq=2, per_channel)` | **~5,472** | 表达力更强，可选 |
| `RadialBasisFilter(K=16, n_ang_freq=2)` 单独 | **65** | mu(16) + log_bwh(1) + 4×coeff/gate(64) |

**关键设计决策回顾**：

1. **N1 修正落点**：`_build_freq_grid()` 中显式 `r_hat = r_grid / (r_grid.max() + 1e-8)`，从根源保证 mu 与 r_hat 在同一值域。
2. **N2 修正落点**：单个 `RadialBasisFilter` 一次 `forward()` 返回 `(diff_mag, diff_phase)`，basis/mu/log_bwh/angular_mod 全部共享计算，仅最后的 sum 使用独立的 coeff/gate。
3. **加性残差形式**（`1 + diff_mag` 而非 `diff_mag`）：保证 `coeff=0` 时输出 = 输入，使训练初期稳定。
4. **post_conv 初始化为恒等**（`eye_(weight)` + `zeros_(bias)`）：进一步保证初始恒等行为，让 LFF 的"扰动效应"在训练中逐步学习而非破坏初始特征。
5. **FFT norm='ortho'**：保证 FFT 和 IFFT 严格保模，避免量级漂移。

---

## 下一步

请确认：

- ✅ 是否同意 N1-N3 的修正落地方式
- ✅ 是否同意 LFF 默认采用 `per_channel_rbf=False`（共享 RBF）
- ✅ 是否同意 `K=16, n_ang_freq=2` 作为 TFSI 默认配置

确认后我将开始 **B.2: TFSI 频域分支接入 LFF**，将现有 `FrequencyBranch` 替换为 `LFFFeatureAdapter`。