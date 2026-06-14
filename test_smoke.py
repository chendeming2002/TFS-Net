"""Smoke test for TFS-Net v4.1 end-to-end verification.
Tests: import, model construction, forward pass, loss computation, param count, gradient flow.
"""
import sys
import torch

def test_imports():
    print("[1/5] Testing imports...")
    from models import TFSNet
    from losses.losses import TFSNetLoss
    from models.modules import (
        LFFFeatureAdapter, RadialBasisFilter,
        SACE, TFSI, IFPN, NDPN, MRPN
    )
    print("  OK - All imports successful")
    return True

def test_model_construction():
    print("[2/5] Testing model construction...")
    from models import TFSNet
    model = TFSNet(
        in_channels=3,
        level_channels=[32, 64, 96, 128],
        fused_channels=64
    )
    print(f"  OK - TFSNet created")
    return model

def test_forward_pass(model):
    print("[3/5] Testing forward pass...")
    device = 'cpu'
    model = model.to(device).eval()
    # B=1, T=5, C=3, H=64, W=64
    x = torch.randn(1, 5, 3, 64, 64, device=device)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, dict), f"Expected dict, got {type(out)}"
    assert 'res_t' in out, f"Missing 'res_t' in output keys: {list(out.keys())}"
    assert 'img_s1' in out, f"Missing 'img_s1' (IGRF stage 1 output)"
    assert 'img_s2' in out, f"Missing 'img_s2' (IGRF stage 2 output)"
    assert 'image_center' in out, f"Missing 'image_center'"
    res_t = out['res_t']
    print(f"  OK - Output shape: {res_t.shape}")
    assert res_t.shape == (1, 3, 64, 64), f"Unexpected shape: {res_t.shape}"
    assert out['img_s1'].shape == (1, 3, 64, 64), f"img_s1 shape mismatch"
    assert out['img_s2'].shape == (1, 3, 64, 64), f"img_s2 shape mismatch"
    print("  OK - All output shapes verified")
    return True

def test_loss():
    print("[4/5] Testing loss computation...")
    from losses.losses import TFSNetLoss
    criterion = TFSNetLoss(
        use_freq_loss=True,
        lambda_perc=0.1,
        lambda_freq=0.1,
        lambda_illum=0.05,
        lambda_ssim=0.2,
        lambda_inter=0.3,
        fused_channels=64,
        perceptual_pretrained=False,
    )
    # Simulate model output (v4.1: img_s1, img_s2 from IGRF)
    pred = torch.randn(1, 3, 64, 64, requires_grad=True)
    target = torch.randn(1, 3, 64, 64)
    s_illum = torch.sigmoid(torch.randn(1, 1, 64, 64))
    img_s1 = torch.sigmoid(torch.randn(1, 3, 64, 64))
    img_s2 = torch.sigmoid(torch.randn(1, 3, 64, 64))
    image_center = torch.randn(1, 3, 64, 64)
    
    model_output = {
        'res_t': pred,
        's_illum': s_illum,
        'img_s1': img_s1,
        'img_s2': img_s2,
        'image_center': image_center,
    }
    
    result = criterion(model_output, target)
    # TFSNetLoss returns (L_total, loss_dict) tuple
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    total_loss, loss_dict = result
    assert isinstance(loss_dict, dict), f"Expected dict, got {type(loss_dict)}"
    print(f"  Loss keys: {list(loss_dict.keys())}")
    assert 'loss_ssim' in loss_dict, "Missing loss_ssim"
    assert 'loss_inter' in loss_dict, "Missing loss_inter"
    
    total_loss.backward()
    print(f"  OK - Loss value: {total_loss.item():.4f}")
    print(f"  OK - Backward pass successful")
    return True

def test_param_count(model):
    print("[5/5] Testing parameter count...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    # Warn if exceeds target
    if total_params > 5_000_000:
        print(f"  WARNING: Params exceed 5M target!")
    else:
        print(f"  OK - Under 5M target")
    return True

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TFS-Net v4.1 Smoke Test")
    print("=" * 60)
    
    try:
        test_imports()
        model = test_model_construction()
        test_forward_pass(model)
        test_loss()
        test_param_count(model)
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
