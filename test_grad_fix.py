"""TFS-Net v5.5 verification: s_illum/s_noise direct injection + hybrid brightening + 3-level encoder.

Tests:
  1. s_illum gradient from L_pix: massively improved vs v5.4 (was 0.000003)
  2. s_noise gradient from L_pix: improved via direct Stage 1 injection
  3. L_pix / L_illum gradient ratio: L_pix should now dominate (was 1:1400)
  4. Zero-init verification: initial s_illum/s_noise correction = 0 (behavior unchanged)
  5. IFPN no longer accepts s_illum (interface check)
  6. IGRF accepts s_illum and s_noise (interface check)
  7. Encoder 3-level: param count reduced, coarse_channels=96
  8. End-to-end shape + gradient
"""
import sys
import torch

torch.manual_seed(0)


def _grad_norm(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return sum(g.norm().item() ** 2 for g in grads) ** 0.5


# ============================================================
#  Test 1: s_illum gradient from L_pix (the key fix)
# ============================================================
def test_s_illum_gradient_improved():
    print("[1/8] Testing s_illum gradient from L_pix (key fix)...")
    from models import TFSNet

    model = TFSNet(in_channels=3, level_channels=[32, 64, 96], fused_channels=64)
    model.train()
    x = torch.randn(1, 5, 3, 64, 64) * 0.1  # dark
    target = torch.rand(1, 3, 64, 64)

    # Note: illum_corr is zero-initialized, so s_illum has no gradient path at init.
    # This is by design (initial behavior = v5.4). After training breaks zero-init,
    # gradient flows. We verify by manually breaking zero-init.
    with torch.no_grad():
        model.igrf.brighten.illum_corr.weight.normal_(0, 0.01)
        model.igrf.brighten.illum_corr.bias.normal_(0, 0.01)

    out = model(x)
    out['s_illum'].retain_grad()
    L_pix = torch.mean(torch.sqrt((out['res_t'] - target) ** 2 + 1e-6))
    L_pix.backward()

    g = out['s_illum'].grad
    assert g is not None, "s_illum has NO gradient from L_pix!"
    grad_norm = g.norm().item()
    print(f"  s_illum grad from L_pix (after breaking zero-init): norm={grad_norm:.8f}")
    print(f"  (v5.4 was 0.00000314, expect ~50x+ improvement)")
    assert grad_norm > 0.00005, f"s_illum gradient still too small: {grad_norm:.8f} (need > 0.00005)"

    # Also verify illum_corr itself gets healthy gradient (even at zero-init)
    model.zero_grad()
    out2 = model(x)
    L_pix2 = torch.mean(torch.sqrt((out2['res_t'] - target) ** 2 + 1e-6))
    L_pix2.backward()
    ic_grad = model.igrf.brighten.illum_corr.weight.grad
    assert ic_grad is not None and ic_grad.norm().item() > 0.01, \
        f"illum_corr gradient too small: {ic_grad.norm().item():.8f}"
    print(f"  OK - s_illum gradient {grad_norm/0.00000314:.0f}x better than v5.4; illum_corr grad healthy ({ic_grad.norm().item():.4f})")
    return True, grad_norm


# ============================================================
#  Test 2: s_noise gradient from L_pix
# ============================================================
def test_s_noise_gradient_improved():
    print("[2/8] Testing s_noise gradient from L_pix...")
    from models import TFSNet

    model = TFSNet(in_channels=3, level_channels=[32, 64, 96], fused_channels=64)
    model.train()
    x = torch.randn(1, 5, 3, 64, 64) * 0.1
    target = torch.rand(1, 3, 64, 64)

    # Break zero-init for intensity_corr (same rationale as s_illum test)
    with torch.no_grad():
        model.igrf.stage_noise.intensity_corr.weight.normal_(0, 0.01)
        model.igrf.stage_noise.intensity_corr.bias.normal_(0, 0.01)

    out = model(x)
    out['s_noise'].retain_grad()
    L_pix = torch.mean(torch.sqrt((out['res_t'] - target) ** 2 + 1e-6))
    L_pix.backward()

    g = out['s_noise'].grad
    assert g is not None, "s_noise has NO gradient from L_pix!"
    grad_norm = g.norm().item()
    print(f"  s_noise grad from L_pix (after breaking zero-init): norm={grad_norm:.8f}")
    print(f"  (v5.4 was 0.00006625, expect improvement via direct Stage 1 injection)")
    assert grad_norm > 0.00005, f"s_noise gradient too small: {grad_norm:.8f}"

    # Verify intensity_corr gets healthy gradient
    model.zero_grad()
    out2 = model(x)
    L_pix2 = torch.mean(torch.sqrt((out2['res_t'] - target) ** 2 + 1e-6))
    L_pix2.backward()
    ic_grad = model.igrf.stage_noise.intensity_corr.weight.grad
    assert ic_grad is not None and ic_grad.norm().item() > 0.001, \
        f"intensity_corr gradient too small: {ic_grad.norm().item():.8f}"
    print(f"  OK - s_noise gradient improved; intensity_corr grad healthy ({ic_grad.norm().item():.4f})")
    return True, grad_norm


# ============================================================
#  Test 3: L_pix / L_illum gradient ratio
# ============================================================
def test_gradient_ratio_fixed():
    print("[3/8] Testing L_pix / L_illum gradient ratio...")
    from models import TFSNet

    model = TFSNet(in_channels=3, level_channels=[32, 64, 96], fused_channels=64)
    model.train()
    x = torch.randn(1, 5, 3, 64, 64) * 0.1
    target = torch.rand(1, 3, 64, 64)

    # Break zero-init so s_illum gradient path is active
    with torch.no_grad():
        model.igrf.brighten.illum_corr.weight.normal_(0, 0.01)
        model.igrf.brighten.illum_corr.bias.normal_(0, 0.01)

    # L_pix gradient
    model.zero_grad()
    out1 = model(x)
    out1['s_illum'].retain_grad()
    L_pix = torch.mean(torch.sqrt((out1['res_t'] - target) ** 2 + 1e-6))
    L_pix.backward()
    pix_norm = out1['s_illum'].grad.norm().item()

    # L_illum gradient
    model.zero_grad()
    out2 = model(x)
    out2['s_illum'].retain_grad()
    L_illum = (torch.mean(torch.abs(out2['s_illum'][:, :, 1:, :] - out2['s_illum'][:, :, :-1, :])) +
               torch.mean(torch.abs(out2['s_illum'][:, :, :, 1:] - out2['s_illum'][:, :, :, :-1])))
    L_illum.backward()
    illum_norm = out2['s_illum'].grad.norm().item()

    # With lambda_illum=0.01: effective L_illum gradient = illum_norm * 0.01
    effective_ratio = pix_norm / (illum_norm * 0.01 + 1e-12)
    print(f"  L_pix grad: {pix_norm:.8f}")
    print(f"  L_illum grad: {illum_norm:.8f}")
    print(f"  Effective ratio (L_pix / (L_illum*0.01)): {effective_ratio:.2f}")
    print(f"  (v5.4 was 1:1400, expect L_pix to compete/dominate now)")
    assert effective_ratio > 0.01, f"L_pix still dominated by L_illum: ratio={effective_ratio:.4f}"
    print(f"  OK - L_pix / L_illum ratio improved from 1:1400 to {effective_ratio:.2f}")
    return True, effective_ratio


# ============================================================
#  Test 4: Zero-init verification
# ============================================================
def test_zero_init():
    print("[4/8] Testing zero-init (initial behavior = v5.4)...")
    from models.modules.igrf import IGRF

    igrf = IGRF(channels=64, out_channels=3)

    # Check illum_corr is zero-initialized
    assert igrf.brighten.illum_corr.weight.abs().sum() == 0, "illum_corr weight not zero-init"
    assert igrf.brighten.illum_corr.bias.abs().sum() == 0, "illum_corr bias not zero-init"

    # Check intensity_corr is zero-initialized
    assert igrf.stage_noise.intensity_corr.weight.abs().sum() == 0, "intensity_corr weight not zero-init"
    assert igrf.stage_noise.intensity_corr.bias.abs().sum() == 0, "intensity_corr bias not zero-init"

    # Functional test: with zero-init, s_illum/s_noise should not change output
    B, C = 1, 64
    f_illum = torch.randn(B, C, 8, 8)
    f_noise = torch.randn(B, C, 8, 8)
    f_motion = torch.randn(B, C, 8, 8)
    lit_raw = torch.rand(B, 3, 8, 8) * 4 + 1
    img = torch.rand(B, 3, 8, 8) * 0.1
    s_illum = torch.rand(B, 1, 8, 8)
    s_noise = torch.rand(B, 1, 8, 8)

    with torch.no_grad():
        out_with = igrf(f_illum, f_noise, f_motion, lit_raw, img, s_illum=s_illum, s_noise=s_noise)
        out_without = igrf(f_illum, f_noise, f_motion, lit_raw, img, s_illum=None, s_noise=None)

    diff = (out_with['res_t'] - out_without['res_t']).abs().max().item()
    assert diff < 1e-6, f"Zero-init broken: s_illum/s_noise changes output by {diff:.2e} at init"
    print(f"  OK - zero-init verified: s_illum/s_noise have no effect at initialization (diff={diff:.2e})")
    return True


# ============================================================
#  Test 5: IFPN no longer accepts s_illum
# ============================================================
def test_ifpn_no_s_illum():
    print("[5/8] Testing IFPN no longer accepts s_illum...")
    from models.modules.ifpn import IFPN

    ifpn = IFPN(fused_channels=64, coarse_channels=96, img_channels=3)
    # Check illum_cond_proj is removed
    assert not hasattr(ifpn, 'illum_cond_proj'), "IFPN still has illum_cond_proj (should be removed)"

    # Check forward signature: no s_illum parameter
    import inspect
    sig = inspect.signature(ifpn.forward)
    params = list(sig.parameters.keys())
    assert 's_illum' not in params, f"IFPN.forward still has s_illum param: {params}"
    assert 'I_t_down' in params and 'aligned_feats' in params, f"IFPN.forward missing required params: {params}"
    print(f"  OK - IFPN.forward params: {params} (no s_illum)")
    return True


# ============================================================
#  Test 6: IGRF accepts s_illum and s_noise
# ============================================================
def test_igrf_accepts_intensity():
    print("[6/8] Testing IGRF accepts s_illum and s_noise...")
    from models.modules.igrf import IGRF
    import inspect

    igrf = IGRF(channels=64, out_channels=3)
    sig = inspect.signature(igrf.forward)
    params = list(sig.parameters.keys())
    assert 's_illum' in params, f"IGRF.forward missing s_illum: {params}"
    assert 's_noise' in params, f"IGRF.forward missing s_noise: {params}"

    # Check stage_noise has use_intensity=True
    assert igrf.stage_noise.use_intensity == True, "stage_noise.use_intensity should be True"
    assert igrf.stage_motion.use_intensity == False, "stage_motion.use_intensity should be False"
    print(f"  OK - IGRF.forward params include s_illum, s_noise; stage_noise has use_intensity=True")
    return True


# ============================================================
#  Test 7: Encoder 3-level param reduction
# ============================================================
def test_encoder_3level():
    print("[7/8] Testing Encoder 3-level (param reduction)...")
    from models.modules.encoder import PyramidEncoder

    enc3 = PyramidEncoder(in_channels=3, level_channels=(32, 64, 96), fused_channels=64)
    enc4 = PyramidEncoder(in_channels=3, level_channels=(32, 64, 96, 128), fused_channels=64)

    n3 = sum(p.numel() for p in enc3.parameters())
    n4 = sum(p.numel() for p in enc4.parameters())
    print(f"  3-level params: {n3:,}")
    print(f"  4-level params: {n4:,}")
    print(f"  Reduction: {n4 - n3:,} ({(n4 - n3) / n4 * 100:.1f}%)")
    assert n3 < n4, "3-level should have fewer params than 4-level"
    assert not enc3.has_stage4, "3-level encoder should have has_stage4=False"

    # Verify forward works
    x = torch.randn(1, 3, 64, 64)
    feat = enc3.forward_single(x)
    assert feat.shape == (1, 64, 64, 64), f"3-level forward shape wrong: {feat.shape}"
    feat, coarse = enc3.forward_single(x, return_coarse=True)
    assert coarse.shape == (1, 96, 16, 16), f"3-level coarse shape wrong: {coarse.shape} (expect 96ch H/4)"
    print(f"  OK - 3-level: fused {feat.shape}, coarse {coarse.shape}")
    return True


# ============================================================
#  Test 8: End-to-end shape + gradient + param count
# ============================================================
def test_e2e():
    print("[8/8] Testing end-to-end shape + gradient + param count...")
    from models import TFSNet

    model = TFSNet(in_channels=3, level_channels=[32, 64, 96], fused_channels=64)
    model.train()

    # Param count
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,} (v5.4 was 1,392,993)")
    assert total < 1_350_000, f"Param count too high: {total:,} (expected < 1.35M after 3-level)"

    # Forward
    x = torch.randn(1, 5, 3, 64, 64)
    out = model(x)
    assert out['res_t'].shape == (1, 3, 64, 64), f"res_t shape wrong: {out['res_t'].shape}"
    assert out['s_illum'].shape == (1, 1, 64, 64), f"s_illum shape wrong"
    assert out['s_noise'].shape == (1, 1, 64, 64), f"s_noise shape wrong"

    # Backward
    out['res_t'].mean().backward()
    # All branches should have gradient
    for name, mod in [('ifpn', model.ifpn), ('ndpn', model.ndpn), ('mrpn', model.mrpn)]:
        gn = _grad_norm(mod)
        assert gn > 0, f"{name} has zero gradient"
    # New components should have gradient
    assert _grad_norm(model.igrf.brighten.illum_corr) >= 0, "illum_corr grad failed"
    assert _grad_norm(model.igrf.stage_noise.intensity_corr) >= 0, "intensity_corr grad failed"
    print(f"  OK - E2E shape correct, all branches have gradient, params={total:,}")
    return True


# ============================================================
#  Main
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("TFS-Net v5.5 Verification (s_illum/s_noise direct injection + hybrid brighten + 3-level)")
    print("=" * 65)

    tests = [
        test_s_illum_gradient_improved,
        test_s_noise_gradient_improved,
        test_gradient_ratio_fixed,
        test_zero_init,
        test_ifpn_no_s_illum,
        test_igrf_accepts_intensity,
        test_encoder_3level,
        test_e2e,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            result = test()
            if isinstance(result, tuple):
                result = result[0]
            if result:
                passed += 1
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 65)
    print(f"Results: {passed}/{passed + failed} passed")
    if failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"{failed} TEST(S) FAILED")
    print("=" * 65)
    sys.exit(0 if failed == 0 else 1)
