# -*- coding: utf-8 -*-
"""
Model validation and parameter verification script for TriDiffNet.
Focuses strictly on the two lightweight proposed variants:
- TriDiffNet-compact (~1.48M params)
- TriDiffNet-ultra   (~1.05M params)
"""
import torch
from models.TriDiffNet import build_TriDiffNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Test TriDiffNet-compact (Stanford Dogs, 120 classes, 224x224)
    print("\n" + "="*50)
    print("Testing TriDiffNet-compact (Stanford Dogs, 120 classes, 224x224)...")
    model_compact = build_TriDiffNet(num_classes=120, typesize='compact', cifar=False)
    model_compact.to(device).eval()
    params_compact = sum(p.numel() for p in model_compact.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params_compact:,} ({params_compact / 1e6:.2f} M)")
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out_compact = model_compact(dummy_input)
    assert out_compact.shape == (2, 120)
    print(f"[PASS] TriDiffNet-compact (120 classes) forward pass successful! Output: {out_compact.shape}")

    # 2. Test TriDiffNet-compact on CIFAR-100 (32x32)
    print("\n" + "="*50)
    print("Testing TriDiffNet-compact (CIFAR-100, 100 classes, 32x32)...")
    model_compact_cifar = build_TriDiffNet(num_classes=100, typesize='compact', cifar=True)
    model_compact_cifar.to(device).eval()
    params_cifar = sum(p.numel() for p in model_compact_cifar.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params_cifar:,} ({params_cifar / 1e6:.2f} M)")
    dummy_cifar = torch.randn(2, 3, 32, 32, device=device)
    with torch.no_grad():
        out_cifar = model_compact_cifar(dummy_cifar)
    assert out_cifar.shape == (2, 100)
    print(f"[PASS] TriDiffNet-compact (CIFAR-100) forward pass successful! Output: {out_cifar.shape}")

    # 3. Test TriDiffNet-compact on ImageNet (1000 classes, 224x224)
    print("\n" + "="*50)
    print("Testing TriDiffNet-compact (ImageNet, 1000 classes, 224x224)...")
    model_compact_in = build_TriDiffNet(num_classes=1000, typesize='compact', cifar=False)
    model_compact_in.to(device).eval()
    params_compact_in = sum(p.numel() for p in model_compact_in.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params_compact_in:,} ({params_compact_in / 1e6:.2f} M)")
    with torch.no_grad():
        out_compact_in = model_compact_in(dummy_input)
    assert out_compact_in.shape == (2, 1000)
    print(f"[PASS] TriDiffNet-compact (ImageNet) forward pass successful! Output: {out_compact_in.shape}")

    # 4. Test Resolution-Agnostic capability (e.g. 256x256)
    print("\n" + "="*50)
    print("Testing Resolution-Agnostic forward pass (256x256)...")
    dummy_arbitrary = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        out_arbitrary = model_compact(dummy_arbitrary)
    assert out_arbitrary.shape == (2, 120)
    print(f"[PASS] Dynamic resolution forward pass (256x256) successful! Output: {out_arbitrary.shape}")

    # 5. Test TriDiffNet-ultra (120 classes & 1000 classes)
    print("\n" + "="*50)
    print("Testing TriDiffNet-ultra (Stanford Dogs, 120 classes, 224x224)...")
    model_ultra = build_TriDiffNet(num_classes=120, typesize='ultra', cifar=False)
    model_ultra.to(device).eval()
    params_ultra = sum(p.numel() for p in model_ultra.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params_ultra:,} ({params_ultra / 1e6:.2f} M)")
    with torch.no_grad():
        out_ultra = model_ultra(dummy_input)
    assert out_ultra.shape == (2, 120)
    print(f"[PASS] TriDiffNet-ultra (120 classes) forward pass successful! Output: {out_ultra.shape}")

    print("\n" + "="*50)
    print("Testing TriDiffNet-ultra (ImageNet, 1000 classes, 224x224)...")
    model_ultra_in = build_TriDiffNet(num_classes=1000, typesize='ultra', cifar=False)
    model_ultra_in.to(device).eval()
    params_ultra_in = sum(p.numel() for p in model_ultra_in.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params_ultra_in:,} ({params_ultra_in / 1e6:.2f} M)")
    with torch.no_grad():
        out_ultra_in = model_ultra_in(dummy_input)
    assert out_ultra_in.shape == (2, 1000)
    print(f"[PASS] TriDiffNet-ultra (ImageNet) forward pass successful! Output: {out_ultra_in.shape}")

    # FLOPs profiling using thop if available
    try:
        from thop import profile, clever_format
        print("\n" + "="*50)
        print("FLOPs & Parameters Summary (thop):")
        x_224 = torch.randn(1, 3, 224, 224, device=device)
        flops_c, params_c = profile(model_compact, inputs=(x_224,), verbose=False)
        flops_u, params_u = profile(model_ultra, inputs=(x_224,), verbose=False)
        flops_c_str, params_c_str = clever_format([flops_c, params_c], "%.2f")
        flops_u_str, params_u_str = clever_format([flops_u, params_u], "%.2f")
        print(f"| TriDiffNet-compact (224x224) | Params: {params_c_str} | FLOPs: {flops_c_str} |")
        print(f"| TriDiffNet-ultra   (224x224) | Params: {params_u_str} | FLOPs: {flops_u_str} |")
    except ImportError:
        pass

    print("\n" + "="*50)
    print("ALL TESTS PASSED SUCCESSFULLY! Only 'compact' and 'ultra' are active.")

if __name__ == '__main__':
    main()
