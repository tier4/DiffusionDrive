#!/usr/bin/env python3
"""
Validate that all NaN fixes for Bench2Drive training are properly applied.
"""

import os
import sys
import numpy as np
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def check_fixes():
    """Check if all fixes are properly applied."""
    
    print("="*60)
    print("BENCH2DRIVE NAN FIX VALIDATION")
    print("="*60)
    
    all_good = True
    
    # 1. Check if Bench2Drive anchors exist
    print("\n1. Checking Bench2Drive plan anchors...")
    b2d_anchor_path = "./download/kmeans_bench2drive_traj_20.npy"
    if os.path.exists(b2d_anchor_path):
        anchors = np.load(b2d_anchor_path)
        print(f"   ✓ Found Bench2Drive anchors: {anchors.shape}")
        print(f"     X range: [{anchors[:, :, 0].min():.3f}, {anchors[:, :, 0].max():.3f}]")
        print(f"     Y range: [{anchors[:, :, 1].min():.3f}, {anchors[:, :, 1].max():.3f}]")
    else:
        print(f"   ✗ Bench2Drive anchors NOT FOUND at {b2d_anchor_path}")
        print("     Run: python3 scripts/generate_b2d_plan_anchors.py")
        all_good = False
    
    # 2. Check transfuser_config.py for dataset-specific normalization
    print("\n2. Checking transfuser_config.py...")
    config_path = "navsim/agents/diffusiondrive/transfuser_config.py"
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    if "b2d_traj_norm_x_offset" in config_content:
        print("   ✓ Found Bench2Drive normalization parameters in config")
    else:
        print("   ✗ Bench2Drive normalization parameters NOT FOUND in config")
        all_good = False
        
    if "dataset_type: str" in config_content:
        print("   ✓ Found dataset_type flag in config")
    else:
        print("   ✗ dataset_type flag NOT FOUND in config")
        all_good = False
    
    # 3. Check transfuser_model_v2.py for configurable normalization
    print("\n3. Checking transfuser_model_v2.py...")
    model_path = "navsim/agents/diffusiondrive/transfuser_model_v2.py"
    with open(model_path, 'r') as f:
        model_content = f.read()
    
    if "if config.dataset_type == \"bench2drive\":" in model_content:
        print("   ✓ Found dataset-specific normalization in model")
    else:
        print("   ✗ Dataset-specific normalization NOT FOUND in model")
        all_good = False
        
    if "torch.clamp(odo_info_fut_x, -1, 1)" in model_content:
        print("   ✓ Found clamping in norm_odo to prevent extreme values")
    else:
        print("   ✗ Clamping NOT FOUND in norm_odo")
        all_good = False
    
    # 4. Check multimodal_loss.py for NaN detection
    print("\n4. Checking multimodal_loss.py...")
    loss_path = "navsim/agents/diffusiondrive/modules/multimodal_loss.py"
    with open(loss_path, 'r') as f:
        loss_content = f.read()
    
    if "if torch.isnan(loss_cls) or torch.isinf(loss_cls):" in loss_content:
        print("   ✓ Found NaN/Inf detection in loss computation")
    else:
        print("   ✗ NaN/Inf detection NOT FOUND in loss computation")
        all_good = False
    
    # 5. Check for other fixes
    print("\n5. Checking other fixes...")
    
    # Camera normalization
    features_path = "navsim/agents/diffusiondrive/transfuser_features_b2d.py"
    with open(features_path, 'r') as f:
        features_content = f.read()
    
    if "stitched = stitched / 255.0" in features_content:
        print("   ✓ Found camera normalization to [0,1]")
    else:
        print("   ✗ Camera normalization NOT FOUND")
        all_good = False
    
    # Gradient clipping
    lightning_path = "navsim/planning/training/agent_lightning_module.py"
    with open(lightning_path, 'r') as f:
        lightning_content = f.read()
    
    if "clip_grad_norm_" in lightning_content:
        print("   ✓ Found gradient clipping")
    else:
        print("   ✗ Gradient clipping NOT FOUND")
        all_good = False
    
    # Summary
    print("\n" + "="*60)
    if all_good:
        print("✓ ALL FIXES PROPERLY APPLIED!")
        print("\nTo train with fixes, run:")
        print("  ./scripts/train_bench2drive_fixed.sh")
    else:
        print("✗ Some fixes are missing. Please check the items marked with ✗ above.")
    print("="*60)
    
    return all_good


if __name__ == "__main__":
    check_fixes()