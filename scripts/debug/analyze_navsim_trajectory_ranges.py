#!/usr/bin/env python3
"""
Analyze trajectory ranges in NavSim dataset to understand coordinate scales.
This helps compare with Bench2Drive and understand the normalization parameters.
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features import (
    TransfuserFeatureBuilder,
    TransfuserTargetBuilder
)


def analyze_trajectory_ranges(cache_path: str, num_samples: int = 1000):
    """
    Analyze trajectory ranges from cached NavSim data.
    
    Args:
        cache_path: Path to the cached dataset
        num_samples: Number of samples to analyze
    """
    print(f"Analyzing trajectory ranges from: {cache_path}")
    
    # Create feature/target builders
    config = TransfuserConfig()
    feature_builder = TransfuserFeatureBuilder(config)
    target_builder = TransfuserTargetBuilder(config)
    
    # Create dataset
    dataset = CacheOnlyDataset(
        cache_path=cache_path,
        feature_builders=[feature_builder],
        target_builders=[target_builder],
    )
    
    print(f"Dataset size: {len(dataset)}")
    num_samples = min(num_samples, len(dataset))
    
    # Collect trajectory statistics
    all_x = []
    all_y = []
    all_heading = []
    
    # Also collect trajectory deltas (consecutive differences)
    all_dx = []
    all_dy = []
    all_dheading = []
    
    print(f"\nAnalyzing {num_samples} samples...")
    for i in tqdm(range(num_samples)):
        try:
            features, targets = dataset[i]
            trajectory = targets["trajectory"]  # Shape: [8, 3] - (x, y, heading)
            
            # Extract x, y, heading
            x_vals = trajectory[:, 0].numpy()
            y_vals = trajectory[:, 1].numpy()
            heading_vals = trajectory[:, 2].numpy()
            
            all_x.extend(x_vals)
            all_y.extend(y_vals)
            all_heading.extend(heading_vals)
            
            # Calculate deltas between consecutive waypoints
            if len(x_vals) > 1:
                dx = np.diff(x_vals)
                dy = np.diff(y_vals)
                dheading = np.diff(heading_vals)
                
                all_dx.extend(dx)
                all_dy.extend(dy)
                all_dheading.extend(dheading)
                
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue
    
    # Convert to numpy arrays
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_heading = np.array(all_heading)
    all_dx = np.array(all_dx) if all_dx else np.array([0])
    all_dy = np.array(all_dy) if all_dy else np.array([0])
    all_dheading = np.array(all_dheading) if all_dheading else np.array([0])
    
    # Print statistics
    print("\n" + "="*60)
    print("NAVSIM TRAJECTORY RANGE ANALYSIS")
    print("="*60)
    
    print("\nAbsolute Position Statistics:")
    print(f"X range: [{all_x.min():.3f}, {all_x.max():.3f}] (range: {all_x.max() - all_x.min():.3f})")
    print(f"Y range: [{all_y.min():.3f}, {all_y.max():.3f}] (range: {all_y.max() - all_y.min():.3f})")
    print(f"Heading range: [{all_heading.min():.3f}, {all_heading.max():.3f}] (range: {all_heading.max() - all_heading.min():.3f})")
    
    print(f"\nX stats: mean={all_x.mean():.3f}, std={all_x.std():.3f}")
    print(f"Y stats: mean={all_y.mean():.3f}, std={all_y.std():.3f}")
    print(f"Heading stats: mean={all_heading.mean():.3f}, std={all_heading.std():.3f}")
    
    print("\nConsecutive Delta Statistics:")
    print(f"ΔX range: [{all_dx.min():.3f}, {all_dx.max():.3f}]")
    print(f"ΔY range: [{all_dy.min():.3f}, {all_dy.max():.3f}]")
    print(f"ΔHeading range: [{all_dheading.min():.3f}, {all_dheading.max():.3f}]")
    
    print("\n" + "="*60)
    print("COMPARISON WITH HARDCODED NORMALIZATION")
    print("="*60)
    
    print("\nHardcoded ranges in model:")
    print("X: [-1.2, 55.7] (range: 56.9)")
    print("Y: [-20, 26] (range: 46)")
    print("Heading: [-2, 1.9] (range: 3.9)")
    
    print("\nActual NavSim ranges:")
    print(f"X: [{all_x.min():.3f}, {all_x.max():.3f}] (range: {all_x.max() - all_x.min():.3f})")
    print(f"Y: [{all_y.min():.3f}, {all_y.max():.3f}] (range: {all_y.max() - all_y.min():.3f})")
    print(f"Heading: [{all_heading.min():.3f}, {all_heading.max():.3f}] (range: {all_heading.max() - all_heading.min():.3f})")
    
    # Check if values are outside hardcoded ranges
    x_outside = np.logical_or(all_x < -1.2, all_x > 55.7)
    y_outside = np.logical_or(all_y < -20, all_y > 26)
    heading_outside = np.logical_or(all_heading < -2, all_heading > 1.9)
    
    print(f"\nPercentage of values outside hardcoded ranges:")
    print(f"X: {100 * x_outside.sum() / len(all_x):.1f}%")
    print(f"Y: {100 * y_outside.sum() / len(all_y):.1f}%")
    print(f"Heading: {100 * heading_outside.sum() / len(all_heading):.1f}%")
    
    # Distribution analysis
    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Check if trajectories are forward-biased
    x_positive_ratio = (all_x > 0).sum() / len(all_x)
    y_centered_ratio = np.abs(all_y.mean()) / all_y.std()
    
    print(f"\nX positive ratio: {100 * x_positive_ratio:.1f}% (forward bias)")
    print(f"Y centering: mean/std = {y_centered_ratio:.3f} (lower is more centered)")
    
    # Percentile analysis
    x_percentiles = np.percentile(all_x, [5, 25, 50, 75, 95])
    y_percentiles = np.percentile(all_y, [5, 25, 50, 75, 95])
    heading_percentiles = np.percentile(all_heading, [5, 25, 50, 75, 95])
    
    print("\nPercentile distribution:")
    print(f"X: 5%={x_percentiles[0]:.1f}, 25%={x_percentiles[1]:.1f}, 50%={x_percentiles[2]:.1f}, 75%={x_percentiles[3]:.1f}, 95%={x_percentiles[4]:.1f}")
    print(f"Y: 5%={y_percentiles[0]:.1f}, 25%={y_percentiles[1]:.1f}, 50%={y_percentiles[2]:.1f}, 75%={y_percentiles[3]:.1f}, 95%={y_percentiles[4]:.1f}")
    
    # Save statistics to file
    stats = {
        "num_samples": num_samples,
        "x_range": [float(all_x.min()), float(all_x.max())],
        "y_range": [float(all_y.min()), float(all_y.max())],
        "heading_range": [float(all_heading.min()), float(all_heading.max())],
        "x_mean": float(all_x.mean()),
        "y_mean": float(all_y.mean()),
        "heading_mean": float(all_heading.mean()),
        "x_std": float(all_x.std()),
        "y_std": float(all_y.std()),
        "heading_std": float(all_heading.std()),
        "x_positive_ratio": float(x_positive_ratio),
        "percent_outside_hardcoded": {
            "x": float(100 * x_outside.sum() / len(all_x)),
            "y": float(100 * y_outside.sum() / len(all_y)),
            "heading": float(100 * heading_outside.sum() / len(all_heading))
        },
        "percentiles": {
            "x": [float(p) for p in x_percentiles],
            "y": [float(p) for p in y_percentiles],
            "heading": [float(p) for p in heading_percentiles]
        }
    }
    
    stats_file = Path(cache_path).parent / "navsim_trajectory_range_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze NavSim trajectory ranges")
    parser.add_argument(
        "--cache-path",
        type=str,
        help="Path to the cached NavSim dataset (will search for it if not provided)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to analyze (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # If no cache path provided, try to find NavSim cache
    if not args.cache_path:
        cache_root = os.environ.get("NAVSIM_EXP_ROOT", "/workspace/navsim_workspace/exp")
        possible_paths = [
            f"{cache_root}/training_cache",
            f"{cache_root}/training_diffusiondrive_agent_cache",
            "/workspace/navsim_workspace/cache/trainval_cache"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                args.cache_path = path
                print(f"Found NavSim cache at: {path}")
                break
        else:
            print("Error: Could not find NavSim cache. Please specify --cache-path")
            sys.exit(1)
    
    if not os.path.exists(args.cache_path):
        print(f"Error: Cache path does not exist: {args.cache_path}")
        sys.exit(1)
    
    analyze_trajectory_ranges(args.cache_path, args.num_samples)


if __name__ == "__main__":
    main()