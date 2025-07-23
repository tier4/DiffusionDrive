#!/usr/bin/env python3
"""
Script to validate Bench2Drive data and compare with NavSim data ranges.
This helps identify data normalization issues that cause NaN losses.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder
)
from navsim.common.bench2drive_dataloader import Bench2DriveConfig, Bench2DriveSceneLoader
from navsim.planning.training.bench2drive_dataset import Bench2DriveDataset
from navsim.planning.training.dataset import CacheOnlyDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_tensor_stats(tensor, name, check_nan=True, check_inf=True):
    """Check tensor statistics and anomalies."""
    stats = {
        'name': name,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'min': float(tensor.min()),
        'max': float(tensor.max()),
        'mean': float(tensor.mean()),
        'std': float(tensor.std()),
        'has_nan': bool(torch.isnan(tensor).any()) if check_nan else None,
        'has_inf': bool(torch.isinf(tensor).any()) if check_inf else None,
        'num_zeros': int((tensor == 0).sum()),
        'zero_ratio': float((tensor == 0).sum() / tensor.numel())
    }
    
    # Check for specific value ranges that might cause issues
    if stats['max'] > 100:
        stats['warning'] = f"Large values detected (max={stats['max']})"
    elif stats['std'] < 1e-6:
        stats['warning'] = f"Very small variance (std={stats['std']})"
    
    return stats


def validate_batch(batch, batch_idx=0):
    """Validate a single batch of data."""
    features, targets = batch
    
    results = {
        'batch_idx': batch_idx,
        'features': {},
        'targets': {},
        'issues': []
    }
    
    # Check features
    logger.info(f"\n=== Batch {batch_idx} Feature Statistics ===")
    for key, tensor in features.items():
        stats = check_tensor_stats(tensor, key)
        results['features'][key] = stats
        
        logger.info(f"\n{key}:")
        logger.info(f"  Shape: {stats['shape']}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        logger.info(f"  Mean/Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if stats.get('has_nan'):
            logger.error(f"  ⚠️  Contains NaN values!")
            results['issues'].append(f"{key} contains NaN")
        
        if stats.get('has_inf'):
            logger.error(f"  ⚠️  Contains Inf values!")
            results['issues'].append(f"{key} contains Inf")
            
        if stats.get('warning'):
            logger.warning(f"  ⚠️  {stats['warning']}")
            results['issues'].append(stats['warning'])
        
        # Special checks for specific features
        if key == 'camera_feature':
            if stats['max'] > 1.1:
                logger.warning(f"  ⚠️  Camera values exceed [0,1] range! Likely not normalized.")
                results['issues'].append("Camera feature not normalized")
        
        elif key == 'lidar_feature':
            if stats['zero_ratio'] > 0.9:
                logger.warning(f"  ⚠️  LiDAR feature is {stats['zero_ratio']*100:.1f}% zeros!")
                results['issues'].append("LiDAR feature mostly zeros")
    
    # Check targets
    logger.info(f"\n=== Batch {batch_idx} Target Statistics ===")
    for key, tensor in targets.items():
        stats = check_tensor_stats(tensor, key)
        results['targets'][key] = stats
        
        logger.info(f"\n{key}:")
        logger.info(f"  Shape: {stats['shape']}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        logger.info(f"  Mean/Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if stats.get('has_nan'):
            logger.error(f"  ⚠️  Contains NaN values!")
            results['issues'].append(f"{key} contains NaN")
        
        # Special checks for BEV semantic map
        if key == 'bev_semantic_map':
            unique_values = torch.unique(tensor)
            logger.info(f"  Unique values: {unique_values.tolist()}")
            if len(unique_values) > 7:
                logger.warning(f"  ⚠️  BEV map has unexpected values!")
                results['issues'].append("BEV map has unexpected values")
            if stats['max'] > 6:
                logger.warning(f"  ⚠️  BEV map values exceed expected range [0,6]!")
                results['issues'].append("BEV map values out of range")
    
    return results


def compare_with_navsim_ranges():
    """Print expected data ranges from NavSim for comparison."""
    logger.info("\n=== Expected NavSim Data Ranges ===")
    logger.info("Camera features:")
    logger.info("  - Range: [0.0, 1.0] (normalized by ToTensor)")
    logger.info("  - Shape: [3, 256, 1024]")
    logger.info("\nLiDAR features:")
    logger.info("  - Range: [0.0, 1.0] (histogram normalized by max_per_pixel)")
    logger.info("  - Shape: [1, 256, 256] or [2, 256, 256] with ground plane")
    logger.info("\nBEV semantic map:")
    logger.info("  - Range: [0, 6] (integer class labels)")
    logger.info("  - Shape: [128, 256]")
    logger.info("\nTrajectory:")
    logger.info("  - Typical range: [-50, 50] meters")
    logger.info("  - Shape: [8, 3] (8 timesteps, x/y/heading)")


def main():
    parser = argparse.ArgumentParser(description='Validate Bench2Drive data')
    parser.add_argument('--cache-path', type=str, required=True,
                        help='Path to B2D cache directory')
    parser.add_argument('--scenarios', nargs='+', default=['ConstructionObstacle'],
                        help='Scenarios to validate')
    parser.add_argument('--num-batches', type=int, default=5,
                        help='Number of batches to validate')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for validation')
    parser.add_argument('--use-cache-only', action='store_true',
                        help='Use CacheOnlyDataset instead of full dataset')
    
    args = parser.parse_args()
    
    # Initialize config
    config = TransfuserConfig()
    
    if args.use_cache_only:
        # Use cached data directly
        logger.info("Using CacheOnlyDataset for validation")
        
        # Get all cached scenarios
        cache_path = Path(args.cache_path)
        available_scenarios = [d.name for d in cache_path.iterdir() if d.is_dir()]
        logger.info(f"Available scenarios in cache: {available_scenarios}")
        
        # Filter by requested scenarios
        test_scenarios = [s for s in available_scenarios 
                         if any(req in s for req in args.scenarios)]
        
        if not test_scenarios:
            logger.error(f"No scenarios found matching {args.scenarios}")
            return
        
        logger.info(f"Testing scenarios: {test_scenarios[:5]}...")  # Show first 5
        
        # Create dataset
        dataset = CacheOnlyDataset(
            cache_path=str(cache_path),
            feature_builders=[Bench2DriveFeatureBuilder(config)],
            target_builders=[Bench2DriveTargetBuilder(config)],
            log_names=test_scenarios[:10]  # Limit to 10 scenarios for testing
        )
    else:
        # Build full dataset
        b2d_config = Bench2DriveConfig(
            data_root=Path(args.cache_path).parent / "data",  # Assuming standard structure
            scenarios=args.scenarios,
            bev_cache_dir=Path(args.cache_path).parent / "bev_cache"
        )
        
        scene_loader = Bench2DriveSceneLoader(b2d_config)
        
        dataset = Bench2DriveDataset(
            scene_loader=scene_loader,
            feature_builders=[Bench2DriveFeatureBuilder(config)],
            target_builders=[Bench2DriveTargetBuilder(config)],
            cache_path=args.cache_path,
            force_cache_computation=False
        )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Use 0 for debugging
    )
    
    # Compare with expected ranges
    compare_with_navsim_ranges()
    
    # Validate batches
    all_issues = []
    logger.info(f"\n=== Validating {args.num_batches} batches ===")
    
    for i, batch in enumerate(dataloader):
        if i >= args.num_batches:
            break
        
        try:
            results = validate_batch(batch, i)
            if results['issues']:
                all_issues.extend(results['issues'])
        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            all_issues.append(f"Batch {i} processing error: {e}")
    
    # Summary
    logger.info("\n=== Validation Summary ===")
    if all_issues:
        logger.error(f"Found {len(all_issues)} issues:")
        for issue in set(all_issues):  # Use set to remove duplicates
            count = all_issues.count(issue)
            logger.error(f"  - {issue} (occurred {count} times)")
    else:
        logger.info("✅ No issues found in validated batches!")
    
    # Recommendations
    if any("Camera feature not normalized" in issue for issue in all_issues):
        logger.info("\n📌 Recommendation: Add normalization to camera features in B2D feature builder")
        logger.info("   Fix: Divide image values by 255.0 after loading")
    
    if any("LiDAR feature mostly zeros" in issue for issue in all_issues):
        logger.info("\n📌 Recommendation: Check LiDAR data loading and histogram computation")
        logger.info("   Possible causes: Wrong coordinate system, clipping issues, or empty point clouds")
    
    if any("BEV map" in issue for issue in all_issues):
        logger.info("\n📌 Recommendation: Review BEV semantic map generation")
        logger.info("   Ensure values are in [0,6] range and properly normalized if needed")


if __name__ == "__main__":
    main()