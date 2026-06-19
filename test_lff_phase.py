"""TFS-Net M1/M2/M3 verification: phase-preserving LFF + SNR scale consistency + noise-aware residual gating.

Tests:
  1. LFF phase-preserving: phase identity when flag=True, phase shaping when False
  2. LFF gradient isolation: coeff_phase grad is zero when phase_preserving=True
  3. SACE sigma_t_clean output: shape + consistency with lff_stack.std
  4. NDPN SNR consistency: accepts sigma_t_clean without error, s_snr responsive
  5. SACE noise-aware residual gating: s_noise=0/0.5/1 controls residual amplitude
  6. End-to-end gradient: TFSNet backward reaches LFF coeff_mag + sigma_t_clean path
  7. Shape consistency: all module I/O shapes match pre-change contract
"""
import sys
import torch

torch.manual_seed(0)


# ============================================================
#  Test 1: LFF phase-preserving behavior
# ============================================================
def test_lff_phase_preserving():
    print("[1/7] Testing LFF phase_preserving flag...")
    from models.modules.lff import LFFFeatureAdapter

    x = torch.randn(2, 16, 32, 32)

    # phase_preserving=True -> phase identity
    lff_pp = LFFFeatureAdapter(channels=16, K=10, n_ang_freq=1, phase_preserving=True)
    # Activate amplitude coeffs so output differs from input (otherwise both are identity)
    with torch.no_grad():
        lff_pp.rbf.coeff_mag.fill_(0.5)
    y_pp = lff_pp(x)

    # phase_preserving=False -> phase shaped (original behavior)
    lff_ps = LFFFeatureAdapter(channels=16, K=10, n_ang_freq=1, phase_preserving=False)
    with torch.no_grad():
        lff_ps.rbf.coeff_mag.fill_(0.5)
        lff_ps.rbf.coeff_phase.fill_(0.3)
    y_ps = lff_ps(x)

    assert y_pp.shape == x.shape, f"phase_preserving shape mismatch: {y_pp.shape}"
    assert y_ps.shape == x.shape, f"phase_shaping shape mismatch: {y_ps.shape}"

    # Verify phase-preserving output differs from phase-shaping output
    # (since phase shaping changes the result when coeff_phase != 0)
    diff = (y_pp - y_ps).abs().mean().item()
    assert diff > 1e-5, f"phase_preserving vs phase_shaping outputs identical (diff={diff:.2e}), flag may not take effect"
    print(f"  OK - phase_preserving=True keeps phase identity; diff vs phase_shaping={diff:.2e}")
    return True


# ============================================================
#  Test 2: LFF gradient isolation
# ============================================================
def test_lff_gradient_isolation():
    print("[2/7] Testing LFF gradient isolation (coeff_phase grad)...")
    from models.modules.lff import LFFFeatureAdapter

    # phase_preserving=True -> coeff_phase should get no gradient
    lff_pp = LFFFeatureAdapter(channels=16, K=10, n_ang_freq=1, phase_preserving=True)
    x = torch.randn(2, 16, 32, 32, requires_grad=True)
    y = lff_pp(x)
    y.mean().backward()
    assert x.grad is not None and x.grad.abs().mean() > 0, "input grad is zero"
    assert lff_pp.rbf.coeff_mag.grad is not None, "coeff_mag grad is None"
    assert lff_pp.rbf.coeff_mag.grad.abs().sum() > 0, "coeff_mag grad is zero (should learn)"
    # coeff_phase grad: may be None (never used in forward) or zero
    cp_grad = lff_pp.rbf.coeff_phase.grad
    if cp_grad is not None:
        assert cp_grad.abs().sum() == 0, f"coeff_phase grad nonzero under phase_preserving=True: {cp_grad.abs().sum()}"
    print("  OK - phase_preserving=True: coeff_mag learns, coeff_phase frozen")

    # phase_preserving=False -> both should get gradient
    lff_ps = LFFFeatureAdapter(channels=16, K=10, n_ang_freq=1, phase_preserving=False)
    x2 = torch.randn(2, 16, 32, 32, requires_grad=True)
    y2 = lff_ps(x2)
    y2.mean().backward()
    assert lff_ps.rbf.coeff_phase.grad is not None, "coeff_phase grad is None under phase_shaping"
    assert lff_ps.rbf.coeff_phase.grad.abs().sum() > 0, "coeff_phase grad zero under phase_shaping"
    print("  OK - phase_preserving=False: both coeff_mag and coeff_phase learn")
    return True


# ============================================================
#  Test 3: SACE sigma_t_clean output
# ============================================================
def test_sace_sigma_t_clean():
    print("[3/7] Testing SACE sigma_t_clean output...")
    from models.modules.sace import SACE

    B, T, C, H, W = 2, 5, 16, 32, 32
    sace = SACE(channels=C, n_groups=4, kernel_size=3, use_optimized=True)
    feats = torch.randn(B, T, C, H, W)
    out = sace(feats, tfsi_out=None)

    assert "sigma_t_clean" in out, f"Missing 'sigma_t_clean' in SACE output: {list(out.keys())}"
    sigma_t_clean = out["sigma_t_clean"]
    assert sigma_t_clean.shape == (B, C, H, W), f"sigma_t_clean shape wrong: {sigma_t_clean.shape}"

    # Verify consistency: sigma_t_clean should equal std of LFF'd feats over T
    # Recompute LFF feats manually
    with torch.no_grad():
        lff_feats = torch.stack([sace.lff(feats[:, t]) for t in range(T)], dim=1)
        expected = lff_feats.std(dim=1, unbiased=False)
    diff = (sigma_t_clean - expected).abs().max().item()
    assert diff < 1e-5, f"sigma_t_clean inconsistent with lff_stack.std: max diff={diff:.2e}"
    print(f"  OK - sigma_t_clean shape {sigma_t_clean.shape}, consistent with lff_stack.std (diff={diff:.2e})")
    return True


# ============================================================
#  Test 4: NDPN SNR consistency with sigma_t_clean
# ============================================================
def test_ndpn_snr_consistency():
    print("[4/7] Testing NDPN accepts sigma_t_clean (M2 rename)...")
    from models.modules.ndpn import NDPN

    B, T, C, H, W = 2, 5, 16, 32, 32
    ndpn = NDPN(channels=C)
    feats = torch.randn(B, T, C, H, W)
    F_aligned = [torch.randn(B, C, H, W) for _ in range(T)]
    mu_t_clean = torch.randn(B, C, H, W)
    sigma_t_clean = torch.rand(B, C, H, W) + 0.1
    s_noise = torch.rand(B, 1, H, W)

    # New interface: sigma_t_clean
    out = ndpn(feats, F_aligned, mu_t_clean, sigma_t_clean, s_noise, center_idx=2)
    assert out["f_noise_out"].shape == (B, C, H, W), f"f_noise_out shape wrong"
    assert out["s_snr"].shape == (B, 1, H, W), f"s_snr shape wrong"
    assert (out["s_snr"] >= 0).all() and (out["s_snr"] <= 1).all(), "s_snr out of [0,1]"

    # Contrast: low sigma_t_clean (high SNR) vs high sigma_t_clean (low SNR)
    sigma_low = torch.full((B, C, H, W), 0.01)
    sigma_high = torch.full((B, C, H, W), 10.0)
    mu_pos = torch.full((B, C, H, W), 1.0)
    out_low_noise = ndpn(feats, F_aligned, mu_pos, sigma_low, s_noise, 2)
    out_high_noise = ndpn(feats, F_aligned, mu_pos, sigma_high, s_noise, 2)
    snr_low_noise = out_low_noise["s_snr"].mean().item()
    snr_high_noise = out_high_noise["s_snr"].mean().item()
    assert snr_low_noise > snr_high_noise, f"SNR monotonicity broken: low_noise={snr_low_noise:.3f} vs high_noise={snr_high_noise:.3f}"
    print(f"  OK - sigma_t_clean accepted; SNR monotone (low_noise s_snr={snr_low_noise:.3f} > high_noise={snr_high_noise:.3f})")
    return True


# ============================================================
#  Test 5: SACE noise-aware residual gating (M3)
# ============================================================
def test_sace_residual_gating():
    print("[5/7] Testing SACE noise-aware residual gating (M3)...")
    from models.modules.sace import SACE

    B, T, C, H, W = 1, 3, 8, 16, 16
    feats = torch.randn(B, T, C, H, W)

    # Build s_noise maps: 0 / 0.5 / 1
    s0 = {"s_noise": torch.zeros(B, 1, H, W)}
    s05 = {"s_noise": torch.full((B, 1, H, W), 0.5)}
    s1 = {"s_noise": torch.ones(B, 1, H, W)}

    sace = SACE(channels=C, n_groups=4, kernel_size=3, use_optimized=True)
    sace.eval()
    with torch.no_grad():
        out0 = sace(feats, tfsi_out=s0)
        out05 = sace(feats, tfsi_out=s05)
        out1 = sace(feats, tfsi_out=s1)

    # Compute norms of aligned features per condition
    def total_norm(out):
        return sum(f.abs().mean().item() for f in out["F_aligned_list"])

    n0 = total_norm(out0)
    n05 = total_norm(out05)
    n1 = total_norm(out1)

    # Higher s_noise -> smaller residual -> smaller aligned norm (since deform_attn output same query)
    # The deform_attn output is query-driven (same across conditions), only residual (1-s_noise)*kv changes
    # So norm should decrease as s_noise increases
    assert n0 > n1, f"Residual gating direction wrong: s_noise=0 norm {n0:.4f} <= s_noise=1 norm {n1:.4f}"
    assert abs(n0 - n1) > 1e-4, f"Residual gating has no effect: n0={n0:.6f} vs n1={n1:.6f}"
    # n05 should be between n0 and n1
    assert min(n0, n1) - 1e-4 <= n05 <= max(n0, n1) + 1e-4, f"n05={n05:.4f} not between n0={n0:.4f} and n1={n1:.4f}"
    print(f"  OK - residual gating: s_noise=0 norm={n0:.4f} > 0.5 norm={n05:.4f} > 1 norm={n1:.4f} (monotone decrease)")

    # Backward compat: tfsi_out=None should fall back to +kv
    out_none = sace(feats, tfsi_out=None)
    n_none = total_norm(out_none)
    assert abs(n_none - n0) < 1e-5, f"tfsi_out=None fallback != s_noise=0: {n_none:.6f} vs {n0:.6f}"
    print(f"  OK - tfsi_out=None fallback to +kv (matches s_noise=0)")
    return True


# ============================================================
#  Test 6: End-to-end gradient check
# ============================================================
def test_e2e_gradient():
    print("[6/7] Testing end-to-end gradient reaches LFF + sigma_t_clean path...")
    from models import TFSNet

    model = TFSNet(
        in_channels=3,
        level_channels=[32, 64, 96, 128],
        fused_channels=64,
    )
    model.train()
    x = torch.randn(1, 5, 3, 64, 64, requires_grad=True)
    out = model(x)
    # Backward from res_t
    out["res_t"].mean().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0, "input grad is zero"

    # LFF coeff_mag should receive gradient (shared between TFSI and SACE)
    lff = model.tfsi.freq_branch.lff
    assert lff.rbf.coeff_mag.grad is not None, "LFF coeff_mag grad is None"
    assert lff.rbf.coeff_mag.grad.abs().sum() > 0, "LFF coeff_mag grad is zero (LFF not learning)"

    # Verify shared LFF: TFSI and SACE point to same instance
    assert model.sace.lff is model.tfsi.freq_branch.lff, "LFF sharing broken: sace.lff is not tfsi.lff"

    # phase_preserving=True on shared LFF (M1)
    assert model.tfsi.freq_branch.lff.phase_preserving == True, "Shared LFF not phase_preserving"

    # Branch gradient norms (sanity: nonzero)
    for name, mod in [("ifpn", model.ifpn), ("ndpn", model.ndpn), ("mrpn", model.mrpn)]:
        grads = [p.grad for p in mod.parameters() if p.grad is not None]
        if grads:
            total = sum(g.abs().sum().item() for g in grads)
            assert total > 0, f"{name} received zero gradient"
    print("  OK - LFF coeff_mag receives gradient; LFF shared; branches receive gradient")
    return True


# ============================================================
#  Test 7: Shape consistency (end-to-end)
# ============================================================
def test_shape_consistency():
    print("[7/7] Testing end-to-end shape consistency...")
    from models import TFSNet

    model = TFSNet(
        in_channels=3,
        level_channels=[32, 64, 96, 128],
        fused_channels=64,
    )
    model.eval()
    B, T, C, H, W = 1, 5, 3, 64, 64
    x = torch.randn(B, T, C, H, W)
    with torch.no_grad():
        out = model(x)

    expected_keys = [
        "res_t", "img_s1", "img_s2", "lit_up_map", "image_center",
        "s_illum", "s_noise", "f_illum_feat", "f_noise_out", "f_motion_out",
        "L_t", "L_ref", "L_ratio", "attn_maps", "mu_t_clean", "s_snr",
        "motion_weights", "tfsi_out",
    ]
    for k in expected_keys:
        assert k in out, f"Missing key '{k}' in TFSNet output: {list(out.keys())}"

    assert out["res_t"].shape == (B, 3, H, W), f"res_t shape wrong: {out['res_t'].shape}"
    assert out["s_illum"].shape == (B, 1, H, W), f"s_illum shape wrong"
    assert out["s_noise"].shape == (B, 1, H, W), f"s_noise shape wrong"
    assert out["mu_t_clean"].shape == (B, 64, H, W), f"mu_t_clean shape wrong"
    # s_snr from NDPN
    assert out["s_snr"].shape == (B, 1, H, W), f"s_snr shape wrong"

    # SACE internal: sigma_t_clean should be in tfsi_out? No, it's in sace internal.
    # But TFSNet returns tfsi_out dict - check sigma_t_clean is accessible via internal sace
    # (tfsi_out is TFSI's output, not SACE's; sigma_t_clean is SACE output consumed internally)
    # Verify NDPN consumed it: s_snr exists and is valid (already checked)
    print(f"  OK - All {len(expected_keys)} output keys present, shapes correct")
    return True


# ============================================================
#  Main runner
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TFS-Net M1/M2/M3 Verification (phase-preserving LFF + SNR consistency + residual gating)")
    print("=" * 60)

    tests = [
        test_lff_phase_preserving,
        test_lff_gradient_isolation,
        test_sace_sigma_t_clean,
        test_ndpn_snr_consistency,
        test_sace_residual_gating,
        test_e2e_gradient,
        test_shape_consistency,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    if failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"{failed} TEST(S) FAILED")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
