#!/usr/bin/env python3
"""
Safe cache regeneration script for Bench2Drive with fixed normalizations.
This script helps regenerate the cache with proper data validation.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder
)
from navsim.common.bench2drive_dataloader import Bench2DriveConfig, Bench2DriveSceneLoader
from navsim.planning.training.bench2drive_dataset import Bench2DriveDataset


def validate_cache_entry(features, targets, token):
    """Validate a single cache entry for common issues."""
    issues = []
    
    # Check features
    for name, tensor in features.items():
        if torch.isnan(tensor).any():
            issues.append(f"{name} contains NaN")
        if torch.isinf(tensor).any():
            issues.append(f"{name} contains Inf")
        
        # Check expected ranges
        if name == "camera_feature":
            if tensor.min() < -0.01 or tensor.max() > 1.01:
                issues.append(f"Camera values out of [0,1]: [{tensor.min():.3f}, {tensor.max():.3f}]")
        elif name == "lidar_feature":
            if tensor.min() < -0.01 or tensor.max() > 1.01:
                issues.append(f"LiDAR values out of [0,1]: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # Check targets
    for name, tensor in targets.items():
        if tensor.dtype in [torch.float32, torch.float64]:
            if torch.isnan(tensor).any():
                issues.append(f"{name} target contains NaN")
        
        if name == "bev_semantic_map":
            unique_vals = torch.unique(tensor)
            if unique_vals.max() > 6 or unique_vals.min() < 0:
                issues.append(f"BEV map has invalid values: {unique_vals.tolist()}")
    
    if issues:
        print(f"\nValidation failed for {token}:")
        for issue in issues:
            print(f"  - {issue}")
    
    return len(issues) == 0


def regenerate_cache_with_validation(args):
    """Regenerate cache with validation at each step."""
    
    # Initialize config
    config = TransfuserConfig()
    
    # Create dataset configuration
    b2d_config = Bench2DriveConfig(
        data_root=Path(args.data_root),
        scenarios=args.scenarios,
        bev_cache_dir=Path(args.bev_cache_dir) if args.bev_cache_dir else None
    )
    
    # Create scene loader
    scene_loader = Bench2DriveSceneLoader(b2d_config)
    print(f"Loaded {len(scene_loader)} scenes")
    
    # Backup old cache if requested
    if args.backup_old and Path(args.cache_path).exists():
        backup_path = Path(args.cache_path).parent / f"{Path(args.cache_path).name}_backup"
        print(f"Backing up old cache to {backup_path}")
        shutil.copytree(args.cache_path, backup_path, dirs_exist_ok=True)
    
    # Create dataset with new feature builders
    dataset = Bench2DriveDataset(
        scene_loader=scene_loader,
        feature_builders=[Bench2DriveFeatureBuilder(config)],
        target_builders=[Bench2DriveTargetBuilder(config)],
        cache_path=args.cache_path,
        force_cache_computation=args.force  # This will regenerate all
    )
    
    # If not forcing, validate existing cache entries
    if not args.force and args.validate_existing:
        print("\nValidating existing cache entries...")
        valid_count = 0
        invalid_tokens = []
        
        for i in tqdm(range(len(dataset)), desc="Validating"):
            try:
                features, targets = dataset[i]
                token = scene_loader.scene_tokens[i]
                
                if validate_cache_entry(features, targets, token):
                    valid_count += 1
                else:
                    invalid_tokens.append(token)
            except Exception as e:
                print(f"\nError loading index {i}: {e}")
                invalid_tokens.append(scene_loader.scene_tokens[i])
        
        print(f"\nValidation complete: {valid_count}/{len(dataset)} entries are valid")
        
        if invalid_tokens:
            print(f"\nFound {len(invalid_tokens)} invalid entries")
            if args.fix_invalid:
                print("Regenerating invalid entries...")
                # Force regeneration of invalid entries
                # This would require modifying the dataset class
                # For now, recommend full regeneration
                print("Note: Full regeneration recommended. Use --force to regenerate all.")
    
    print("\nCache generation/validation complete!")
    
    # Final statistics
    if args.show_stats:
        print("\n=== Cache Statistics ===")
        # Sample a few entries to show typical ranges
        sample_indices = np.random.choice(len(dataset), min(5, len(dataset)), replace=False)
        
        for idx in sample_indices:
            features, targets = dataset[idx]
            print(f"\nSample {idx}:")
            for name, tensor in features.items():
                print(f"  {name}: shape={tensor.shape}, range=[{tensor.min():.3f}, {tensor.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description='Regenerate B2D cache with fixed normalizations')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of Bench2Drive data')
    parser.add_argument('--cache-path', type=str, required=True,
                        help='Path to cache directory')
    parser.add_argument('--scenarios', nargs='+', default=['ConstructionObstacle'],
                        help='Scenarios to process')
    parser.add_argument('--bev-cache-dir', type=str,
                        help='Directory containing pre-generated BEV maps')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration of all cache entries')
    parser.add_argument('--backup-old', action='store_true',
                        help='Backup old cache before regenerating')
    parser.add_argument('--validate-existing', action='store_true',
                        help='Validate existing cache entries')
    parser.add_argument('--fix-invalid', action='store_true',
                        help='Regenerate only invalid entries (not fully implemented)')
    parser.add_argument('--show-stats', action='store_true',
                        help='Show statistics of cached data')
    
    args = parser.parse_args()
    
    print("=== Bench2Drive Cache Regeneration ===")
    print(f"Data root: {args.data_root}")
    print(f"Cache path: {args.cache_path}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Force regeneration: {args.force}")
    print()
    
    # Check if fixes are applied
    print("Checking if normalization fixes are applied...")
    with open(__file__.replace('regenerate_b2d_cache_safe.py', '../../navsim/agents/diffusiondrive/transfuser_features_b2d.py'), 'r') as f:
        content = f.read()
        if '/ 255.0' in content:
            print("✅ Camera normalization fix is applied")
        else:
            print("❌ WARNING: Camera normalization fix not found!")
            if not input("Continue anyway? (y/n): ").lower().startswith('y'):
                return
    
    try:
        regenerate_cache_with_validation(args)
    except KeyboardInterrupt:
        print("\n\nCache generation interrupted by user")
    except Exception as e:
        print(f"\n\nError during cache generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()