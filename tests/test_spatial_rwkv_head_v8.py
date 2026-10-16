#!/usr/bin/env python3
"""v8 空间 RWKV 头 (OmniShift + Channel Mix) 的单元测试。

对应 `models/golf_v7r/spatial_rwkv_head_v8.py` 与 `docs/v8/01-设计.md` §三。

设计要点与对应的验证:
  1. **OmniShift 四分支可重参数化** —— 合并为单个 5×5 depthwise 后输出等价;
  2. **Channel Mix 末层零初始化不构成双零死锁** —— 首步梯度即可达上游权重;
  3. **输出形状与输入一致** —— 可直接替换旧空间头;
  4. **各扫描方向确实不同** —— 保证四个方向的 index 映射正确;
  5. **三路 Q 门控生效** —— Q 不同的两路输出不同, 这是三路分化的前提。

运行:
    PYTHONPATH=. python tests/test_spatial_rwkv_head_v8.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.golf_v7r.spatial_rwkv_head_v8 import OmniShift, SpatialRWKVHeadV8


def test_omni_shift_output_shape():
    """OmniShift 保持形状与通道数。"""
    torch.manual_seed(0)
    omni = OmniShift(16)
    x = torch.randn(2, 16, 8, 8)
    y = omni(x)
    assert y.shape == x.shape, f"形状不符: {y.shape} vs {x.shape}"
    assert torch.isfinite(y).all(), "输出出现 NaN/Inf"
    print(f"  [ok] OmniShift 形状保持: {tuple(y.shape)}")


def test_omni_shift_reparameterization_equivalent():
    """四分支合并为单个 5×5 卷积后, 输出必须等价。

    这是写进论文的加速声明, 出错会导致推理结果与训练不一致。
    """
    torch.manual_seed(1)
    omni = OmniShift(8).eval()
    x = torch.randn(1, 8, 12, 12)
    with torch.no_grad():
        y_multi = omni(x)
        conv = omni.reparameterize()
        y_single = conv(x)
    err = (y_multi - y_single).abs().max().item()
    assert err < 1e-5, f"重参数化后输出不一致, 最大误差 {err:.2e}"
    print(f"  [ok] OmniShift 重参数化等价 (最大误差 {err:.2e})")


def test_omni_shift_initial_scale_bounded():
    """初始化时输出尺度与输入同量级, 不因四路叠加而放大。"""
    torch.manual_seed(2)
    omni = OmniShift(32)
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        y = omni(x)
    ratio = y.std().item() / x.std().item()
    assert 0.2 < ratio < 5.0, f"初始尺度比异常: {ratio:.3f}"
    print(f"  [ok] OmniShift 初始尺度比 = {ratio:.3f} (应接近 1)")


def test_channel_mix_zero_init_is_identity():
    """末层零初始化 → 初始时 Channel Mix 输出为 0, 整头等价于纯 WKV 路径。"""
    torch.manual_seed(3)
    head = SpatialRWKVHeadV8(32).eval()
    # 手动把 channel_mix 末层设为 0 (构造函数已保证)
    assert head.channel_mix[-1].weight.abs().sum().item() == 0.0, \
        "channel_mix 末层未零初始化"
    print("  [ok] Channel Mix 末层零初始化")


def test_channel_mix_gradient_reaches_upstream():
    """零初始化末层后, 首步只有末层有梯度; 末层更新一步后上游即获得梯度。

    末层零初始化时, channel_mix 输出恒为 0 (恒等残差起点), 上游首层的梯度需要乘以
    末层权重, 而末层权重此刻为 0, 所以上游在**第 1 步**为零。这不是永久死锁:
    末层在第 1 步就拿到非零梯度, 更新后不再是 0, 上游从第 2 步起可达。
    本测试把这个两步过程固化下来, 与「LayerScale×0 且上游也零初始化」的双零死锁区分开
    (那种写法两边永远都是 0)。
    """
    torch.manual_seed(4)
    head = SpatialRWKVHeadV8(32)
    x_c = torch.randn(1, 32, 8, 8)
    x_kv = torch.randn(1, 32, 8, 8)
    y = head(x_c, x_kv)
    y.sum().backward()

    g_last = head.channel_mix[-1].weight.grad
    assert g_last is not None and g_last.abs().sum() > 0, "末层首步无梯度"
    print(f"  [ok] Channel Mix 末层首步梯度 = {g_last.abs().mean():.2e}")

    # 手动让末层更新一步 (模拟优化器), 再查上游是否可达
    with torch.no_grad():
        head.channel_mix[2].weight -= 0.1 * head.channel_mix[2].weight.grad
    for p in head.parameters():
        p.grad = None
    head(x_c, x_kv).sum().backward()
    g_first = head.channel_mix[0].weight.grad
    assert g_first is not None and g_first.abs().sum() > 0, "末层更新后上游仍无梯度 (双零死锁)"
    print(f"  [ok] 末层更新后上游层梯度可达 = {g_first.abs().mean():.2e}")


def test_all_parameters_receive_gradient():
    """末层更新一步后, 整头所有可训练参数都应有非零梯度。

    首步 channel_mix 输出恒为 0, 因此它内部的 post_norm 与首层权重 (只在 channel_mix
    内部使用) 梯度为 0 —— 这是零初始化残差分支的正常起点, 不是死锁。
    末层更新一步之后, channel_mix 通路打开, 这些参数应全部变为可达。
    """
    torch.manual_seed(5)
    head = SpatialRWKVHeadV8(32)
    x_c = torch.randn(1, 32, 8, 8)
    x_kv = torch.randn(1, 32, 8, 8)
    head(x_c, x_kv).sum().backward()
    with torch.no_grad():   # 让末层离开 0, 打开 channel_mix 通路
        head.channel_mix[2].weight -= 0.1 * head.channel_mix[2].weight.grad
    for p in head.parameters():
        p.grad = None
    head(x_c, x_kv).sum().backward()
    dead = [n for n, p in head.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"以下参数在末层更新后仍无梯度: {dead}"
    print("  [ok] 末层更新后全部参数均有非零梯度")


def test_output_shape_matches_input():
    """输出形状与输入一致, 可直接替换旧空间头。"""
    torch.manual_seed(6)
    head = SpatialRWKVHeadV8(64, num_directions=4)
    x_c = torch.randn(2, 64, 10, 12)
    x_kv = torch.randn(2, 64, 10, 12)
    y = head(x_c, x_kv)
    assert y.shape == x_c.shape, f"形状不符: {y.shape} vs {x_c.shape}"
    print(f"  [ok] 输出形状与输入一致: {tuple(y.shape)}")


def test_scan_directions_are_distinct():
    """四个扫描方向的完整索引序列必须两两不同, 否则等于只有一个方向。

    用单位索引图 (而非随机特征) 提取每个方向的访问顺序; 取完整长度比较,
    不能只看前缀 —— 不同的扫描在开头可能恰好重合。
    """
    head = SpatialRWKVHeadV8(32)
    H, W = 6, 7
    ident = torch.arange(H * W, dtype=torch.float).view(1, 1, H, W)
    orders = []
    for scan_fn in head._scan_fns():
        orders.append(tuple(scan_fn(ident).flatten().long().tolist()))
    assert all(len(o) == H * W for o in orders), "扫描序列长度不等于像素数"
    assert len(set(orders)) == len(orders), "存在方向索引映射重复"
    print("  [ok] 四个扫描方向的完整索引序列两两不同")


def test_scan_inverse_index_is_consistent():
    """扫描 → 逆索引 必须还原成原始位置顺序。"""
    torch.manual_seed(8)
    head = SpatialRWKVHeadV8(32)
    B, C, H, W = 1, 32, 5, 6
    feat = torch.randn(B, C, H, W)
    for scan_fn in head._scan_fns():
        scanned = scan_fn(feat)
        inv = head._inv_index(scan_fn, B, C, H, W, feat.device)
        restored = scanned[:, inv]
        err = (restored - feat.flatten(2).transpose(1, 2)).abs().max().item()
        assert err < 1e-6, f"{scan_fn.__name__} 逆索引还原失败, 误差 {err:.2e}"
    print("  [ok] 扫描逆索引一致")


def test_query_gating_differentiates_outputs():
    """Q 不同的两路 (共享同一 KV) 输出必须不同 —— 三路分化的前提。"""
    torch.manual_seed(9)
    head = SpatialRWKVHeadV8(32).eval()
    x_kv = torch.randn(1, 32, 8, 8)
    q1 = torch.randn(1, 32, 8, 8)
    q2 = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        y1 = head(q1, x_kv)
        y2 = head(q2, x_kv)
    diff = (y1 - y2).abs().mean().item()
    assert diff > 1e-4, f"不同 Q 的输出几乎相同 (差 {diff:.2e}), Q 门控未生效"
    print(f"  [ok] 不同 Q 输出分化 (平均差 {diff:.4f})")


def test_gradients_flow_to_both_inputs():
    """中心帧与上下文两条输入路径都应获得梯度。"""
    torch.manual_seed(10)
    head = SpatialRWKVHeadV8(32)
    x_c = torch.randn(1, 32, 8, 8, requires_grad=True)
    x_kv = torch.randn(1, 32, 8, 8, requires_grad=True)
    head(x_c, x_kv).sum().backward()
    assert x_c.grad is not None and x_c.grad.abs().sum() > 0, "中心帧路径无梯度"
    assert x_kv.grad is not None and x_kv.grad.abs().sum() > 0, "上下文路径无梯度"
    print("  [ok] 两条输入路径均有梯度 (含 OmniShift)")


def test_bad_directions_rejected():
    """通道数不能被方向数整除时应报错。"""
    try:
        SpatialRWKVHeadV8(30, num_directions=4)
    except AssertionError:
        print("  [ok] 非法方向数被拒绝")
        return
    raise AssertionError("通道数不能整除方向数时未报错")


def main():
    print("v8 空间 RWKV 头单测 (OmniShift + Channel Mix):")
    test_omni_shift_output_shape()
    test_omni_shift_reparameterization_equivalent()
    test_omni_shift_initial_scale_bounded()
    test_channel_mix_zero_init_is_identity()
    test_channel_mix_gradient_reaches_upstream()
    test_all_parameters_receive_gradient()
    test_output_shape_matches_input()
    test_scan_directions_are_distinct()
    test_scan_inverse_index_is_consistent()
    test_query_gating_differentiates_outputs()
    test_gradients_flow_to_both_inputs()
    test_bad_directions_rejected()
    print("\n全部通过 ✅")


if __name__ == '__main__':
    main()
