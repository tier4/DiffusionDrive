#!/usr/bin/env python3
"""
Check if all NaN fixes have been properly applied to the codebase.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))


def check_file_contains(filepath, search_strings, description):
    """Check if file contains expected strings."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        found_all = all(s in content for s in search_strings)
        
        if found_all:
            print(f"✅ {description}")
            return True
        else:
            print(f"❌ {description}")
            for s in search_strings:
                if s not in content:
                    print(f"   Missing: {s}")
            return False
    except FileNotFoundError:
        print(f"❌ {description} - File not found: {filepath}")
        return False
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return False


def main():
    print("=== Checking NaN Fix Status ===\n")
    
    all_good = True
    
    # Check 1: Camera normalization
    all_good &= check_file_contains(
        project_root / "navsim/agents/diffusiondrive/transfuser_features_b2d.py",
        ["stitched = stitched / 255.0", "Normalize to [0, 1] range"],
        "Camera normalization fix"
    )
    
    # Check 2: LiDAR normalization
    all_good &= check_file_contains(
        project_root / "navsim/agents/diffusiondrive/transfuser_features_b2d.py",
        ["hist = np.clip(hist, 0, hist_max_per_pixel)", "hist = hist / hist_max_per_pixel"],
        "LiDAR normalization fix"
    )
    
    # Check 3: BEV semantic map data type
    all_good &= check_file_contains(
        project_root / "navsim/common/bev_semantic_utils.py",
        ["dtype=np.uint8", "return bev_map.astype(np.uint8)"],
        "BEV semantic map integer type fix"
    )
    
    # Check 4: Autograd anomaly detection
    all_good &= check_file_contains(
        project_root / "navsim/planning/script/run_bench2drive_training.py",
        ["torch.autograd.set_detect_anomaly(True)"],
        "Autograd anomaly detection"
    )
    
    # Check 5: Gradient clipping
    all_good &= check_file_contains(
        project_root / "navsim/planning/training/agent_lightning_module.py",
        ["clip_grad_norm_", "on_before_optimizer_step"],
        "Gradient clipping"
    )
    
    # Check 6: Feature validation
    all_good &= check_file_contains(
        project_root / "navsim/agents/diffusiondrive/transfuser_features_b2d.py",
        ["_validate_features", "torch.isnan", "torch.isinf"],
        "Feature validation checks"
    )
    
    print("\n" + "="*50 + "\n")
    
    if all_good:
        print("✅ All fixes are properly applied!")
        print("\nNext steps:")
        print("1. Clear your old cache: rm -rf /path/to/cache")
        print("2. Regenerate cache with fixed code")
        print("3. Run test_nan_fixes.py to verify")
        print("4. Start training with lower learning rate (1e-5)")
    else:
        print("❌ Some fixes are missing!")
        print("\nPlease ensure all fixes are applied before training.")
        print("The missing fixes are likely causing the NaN issues.")
    
    # Additional recommendations
    print("\n" + "="*50)
    print("\n📝 Additional Recommendations:")
    print("- Always regenerate cache after applying fixes")
    print("- Start with very low learning rate (1e-5 or 2e-5)")
    print("- Monitor gradient norms during training")
    print("- Use the validation script to check data ranges")
    print("- Keep autograd anomaly detection on for first few iterations")
    
    # Check for debug scripts
    print("\n🛠️  Debug Tools Available:")
    debug_scripts = [
        "validate_b2d_data.py",
        "test_nan_fixes.py",
        "regenerate_b2d_cache_safe.py",
        "monitor_training_nan.py"
    ]
    
    debug_dir = Path(__file__).parent
    for script in debug_scripts:
        if (debug_dir / script).exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} (missing)")


if __name__ == "__main__":
    main()