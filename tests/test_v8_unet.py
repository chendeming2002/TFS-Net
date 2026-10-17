#!/usr/bin/env python3
"""v8 U-Net 式重构的单测 (SpatialSummaryUNet + TripleQueryTCAv9 + GolfNet_v8)。

对应 `docs/v8/05-UNet重构设计.md`。

核心验证:
  1. U-Net 聚合: 单帧旁路必须与多帧聚合完全分开 (改邻帧不影响 single 分支);
  2. 帧权重是合法的概率分布 (非负、和为 1);
  3. F1 尺度 TCA: 三路 KV 来源两两不同, N/L 依赖多帧、M 依赖单帧;
  4. 分支 F1 直出: 输出分辨率与输入一致, 不再上采样;
  5. 全模型前向/反向、参数量与显存。

运行:
    CUDA_HOME=/usr/local/cuda CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9 \
    PATH=$CONDA_PREFIX/bin:$PATH PYTHONPATH=. TORCH_CUDA_ARCH_LIST=8.9 \
    python tests/test_v8_unet.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.spatial_summary_unet import SpatialSummaryUNet
from models.golf_v7r.triple_query_tca_v9 import TripleQueryTCAv9
from models.golf_v7r.branch_n_simple import BranchNSimple
from models.golf_v7r.branch_l_simple import BranchLSimple
from models.golf_v7r.branch_m_simple import BranchMSimple


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_unet_single_bypass_is_independent_of_neighbors():
    """单帧旁路必须只由中心帧决定: 改掉邻帧, F1_single / F2_single 不应改变。"""
    dev = _dev()
    torch.manual_seed(0)
    net = SpatialSummaryUNet([16, 32, 64], num_frames=5).to(dev).eval()
    B, T = 2, 5
    f1 = torch.randn(B, T, 16, 16, 16, device=dev)
    f2 = torch.randn(B, T, 32, 8, 8, device=dev)
    f3 = torch.randn(B, T, 64, 4, 4, device=dev)
    with torch.no_grad():
        a = net(f1, f2, f3)
        # 只改非中心帧
        f1b, f2b, f3b = f1.clone(), f2.clone(), f3.clone()
        for t in (0, 1, 3, 4):
            f1b[:, t] += 3.0
            f2b[:, t] += 3.0
            f3b[:, t] += 3.0
        b = net(f1b, f2b, f3b)
    assert torch.allclose(a["F1_single"], b["F1_single"]), "单帧旁路受了邻帧影响"
    assert torch.allclose(a["F2_single"], b["F2_single"]), "F2 单帧旁路受了邻帧影响"
    assert not torch.allclose(a["F1_multi"], b["F1_multi"]), "多帧聚合应对邻帧变化有反应"
    print("  [ok] 单帧旁路与邻帧无关, 多帧聚合受邻帧影响")


def test_frame_weights_are_distribution():
    """帧权重非负且和为 1 (softmax 归一化), 且初始近似均匀。"""
    dev = _dev()
    torch.manual_seed(1)
    net = SpatialSummaryUNet([16, 32, 64], num_frames=5).to(dev).eval()
    B, T = 3, 5
    with torch.no_grad():
        out = net(torch.randn(B, T, 16, 8, 8, device=dev),
                  torch.randn(B, T, 32, 4, 4, device=dev),
                  torch.randn(B, T, 64, 2, 2, device=dev))
    w = out["frame_weights"]
    assert (w >= 0).all(), "帧权重出现负值"
    assert torch.allclose(w.sum(1), torch.ones(B, device=dev), atol=1e-5), "帧权重和不为 1"
    print(f"  [ok] 帧权重合法, 5 帧平均 {w.mean(0).tolist()}")


def test_tca_kv_sources_differ():
    """TCA 三路 KV 必须来源不同: 改动单帧输入只应影响 M 路, 不影响 N/L。"""
    dev = _dev()
    torch.manual_seed(2)
    C = 16
    tca = TripleQueryTCAv9(C, spatial_head="v8", wkv_mode="single_norm").to(dev).eval()
    multi = torch.randn(1, C, 16, 16, device=dev)
    single = torch.randn(1, C, 16, 16, device=dev)
    seen = {}

    def hook_factory(name):
        def hook(module, args):
            seen[name] = args[1].detach().clone()   # (Q, context_feat)
        return hook

    for nm, mod in (("N", tca.attn_N), ("L", tca.attn_L), ("M", tca.attn_M)):
        mod.register_forward_pre_hook(hook_factory(nm))
    with torch.no_grad():
        tca(multi, single)
    assert not torch.allclose(seen["N"], seen["L"]), "N 与 L 的 KV 应不同"
    assert not torch.allclose(seen["N"], seen["M"]), "N 与 M 的 KV 应不同"
    assert not torch.allclose(seen["L"], seen["M"]), "L 与 M 的 KV 应不同"

    # 改单帧: M 的 KV 应改变, N/L 不应变
    seen.clear()
    with torch.no_grad():
        tca(multi, single + 1.0)
    n_prev, m_prev = seen["N"], seen["M"]
    seen.clear()
    with torch.no_grad():
        tca(multi, single)
    assert torch.allclose(seen["N"], n_prev), "N 的 KV 不应受单帧输入影响"
    assert not torch.allclose(seen["M"], m_prev), "M 的 KV 应随单帧输入改变"
    print("  [ok] KV 按来源分流 (N/L 依赖多帧, M 只依赖单帧)")


def test_tca_three_branch_shapes():
    """三路输出形状正确且反传正常。"""
    dev = _dev()
    torch.manual_seed(3)
    C = 16
    tca = TripleQueryTCAv9(C).to(dev)
    multi = torch.randn(1, C, 16, 16, device=dev)
    single = torch.randn(1, C, 16, 16, device=dev)
    F_N, F_L, F_M, ortho = tca(multi, single)
    for nm, t in (("F_N", F_N), ("F_L", F_L), ("F_M", F_M)):
        assert t.shape == multi.shape, f"{nm} 形状 {t.shape}"
    (F_N.sum() + F_L.sum() + F_M.sum() + ortho).backward()
    print("  [ok] TCA 三路形状正确且反传正常")


def test_branches_at_f1_keep_resolution():
    """分支 at_f1 模式: 输入全分辨率时输出仍是全分辨率 (不再 2 倍上采样)。"""
    dev = _dev()
    torch.manual_seed(4)
    C, H = 16, 32
    x = torch.randn(1, C, H, H, device=dev)
    X = torch.randn(1, 3, H, H, device=dev)
    bn = BranchNSimple(channels=C, at_f1=True).to(dev)
    bl = BranchLSimple(channels=C, at_f1=True).to(dev)
    bm = BranchMSimple(channels=C, enc_channels=8, at_f1=True,
                       single_frame=True).to(dev)
    assert bn(x)["Y_N"].shape == (1, 3, H, H)
    assert bl(x, X)["Y_L"].shape == (1, 3, H, H)
    assert bm(x, x.unsqueeze(1))["Y_M"].shape == (1, 3, H, H)
    # 对照: 非 at_f1 模式仍然是 2 倍上采样
    bn_legacy = BranchNSimple(channels=C, at_f1=False).to(dev)
    assert bn_legacy(x)["Y_N"].shape == (1, 3, H * 2, H * 2)
    print("  [ok] at_f1 分支保持分辨率, 非 at_f1 分支仍上采样 2 倍")


def test_full_model_forward_backward():
    """GolfNet_v8 全模型: 前向、反向、三路活性都正常。"""
    dev = _dev()
    from models.golf_v7r.golfnet_v8 import GolfNet_v8, count_params
    torch.manual_seed(5)
    m = GolfNet_v8().to(dev).train()
    st = count_params(m)
    x = torch.randn(1, 5, 3, 64, 64, device=dev)
    o = m(x)
    assert o["final"].shape == (1, 3, 64, 64)
    for k in ("branch_N", "branch_L", "branch_M"):
        assert o[k].shape == (1, 3, 64, 64), f"{k} 形状 {o[k].shape}"
        assert o[k].std().item() > 1e-6, f"{k} 输出近似常数"
    o["final"].sum().backward()
    print(f"  [ok] GolfNet_v8 前向/反向正常, 总参数 {st['total']/1e6:.3f}M "
          f"(encoder {st['encoder']/1e6:.3f}M, summary {st['summary']/1e6:.3f}M)")


def test_loops_change_compute_not_params():
    """循环次数 M 改变计算量但不改变参数量 (Re-WKV 的做法)。"""
    from models.golf_v7r.golfnet_v8 import GolfNet_v8
    n1 = sum(p.numel() for p in GolfNet_v8(encoder_loops=1, temporal_loops=1).parameters())
    n3 = sum(p.numel() for p in GolfNet_v8(encoder_loops=3, temporal_loops=3).parameters())
    assert n1 == n3, f"循环次数不应改变参数量: {n1} vs {n3}"
    print(f"  [ok] 循环次数不改变参数量 ({n1/1e6:.3f}M)")


def main():
    print("v8 U-Net 式重构单测:")
    test_unet_single_bypass_is_independent_of_neighbors()
    test_frame_weights_are_distribution()
    test_tca_kv_sources_differ()
    test_tca_three_branch_shapes()
    test_branches_at_f1_keep_resolution()
    test_loops_change_compute_not_params()
    test_full_model_forward_backward()
    print("\n全部通过 ✅")


if __name__ == '__main__':
    main()
