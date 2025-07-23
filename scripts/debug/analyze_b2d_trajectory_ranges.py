#!/usr/bin/env python3
"""
Analyze trajectory ranges in Bench2Drive dataset to understand coordinate scales.
This helps identify the mismatch with NavSim normalization ranges.
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
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder
)


def analyze_trajectory_ranges(cache_path: str, num_samples: int = 1000):
    """
    Analyze trajectory ranges from cached Bench2Drive data.
    
    Args:
        cache_path: Path to the cached dataset
        num_samples: Number of samples to analyze
    """
    print(f"Analyzing trajectory ranges from: {cache_path}")
    
    # Create feature/target builders
    config = TransfuserConfig()
    feature_builder = Bench2DriveFeatureBuilder(config)
    target_builder = Bench2DriveTargetBuilder(config)
    
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
    print("TRAJECTORY RANGE ANALYSIS")
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
    print("COMPARISON WITH NAVSIM NORMALIZATION")
    print("="*60)
    
    print("\nNavSim expected ranges (hardcoded in norm_odo):")
    print("X: [-1.2, 55.7] (range: 56.9)")
    print("Y: [-20, 26] (range: 46)")
    print("Heading: [-2, 1.9] (range: 3.9)")
    
    print("\nBench2Drive actual ranges:")
    print(f"X: [{all_x.min():.3f}, {all_x.max():.3f}] (range: {all_x.max() - all_x.min():.3f})")
    print(f"Y: [{all_y.min():.3f}, {all_y.max():.3f}] (range: {all_y.max() - all_y.min():.3f})")
    print(f"Heading: [{all_heading.min():.3f}, {all_heading.max():.3f}] (range: {all_heading.max() - all_heading.min():.3f})")
    
    # Check if values are outside NavSim ranges
    x_outside = np.logical_or(all_x < -1.2, all_x > 55.7)
    y_outside = np.logical_or(all_y < -20, all_y > 26)
    heading_outside = np.logical_or(all_heading < -2, all_heading > 1.9)
    
    print(f"\nPercentage of values outside NavSim ranges:")
    print(f"X: {100 * x_outside.sum() / len(all_x):.1f}%")
    print(f"Y: {100 * y_outside.sum() / len(all_y):.1f}%")
    print(f"Heading: {100 * heading_outside.sum() / len(all_heading):.1f}%")
    
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
        "percent_outside_navsim": {
            "x": float(100 * x_outside.sum() / len(all_x)),
            "y": float(100 * y_outside.sum() / len(all_y)),
            "heading": float(100 * heading_outside.sum() / len(all_heading))
        }
    }
    
    stats_file = Path(cache_path).parent / "trajectory_range_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")
    
    # Suggest normalization parameters
    print("\n" + "="*60)
    print("SUGGESTED NORMALIZATION PARAMETERS")
    print("="*60)
    
    # Use percentiles to avoid outliers
    x_min_p5, x_max_p95 = np.percentile(all_x, [5, 95])
    y_min_p5, y_max_p95 = np.percentile(all_y, [5, 95])
    heading_min_p5, heading_max_p95 = np.percentile(all_heading, [5, 95])
    
    x_center = (x_min_p5 + x_max_p95) / 2
    y_center = (y_min_p5 + y_max_p95) / 2
    heading_center = (heading_min_p5 + heading_max_p95) / 2
    
    x_range = x_max_p95 - x_min_p5
    y_range = y_max_p95 - y_min_p5
    heading_range = heading_max_p95 - heading_min_p5
    
    print(f"\nBased on 5th-95th percentiles:")
    print(f"X: center={x_center:.3f}, range={x_range:.3f}")
    print(f"Y: center={y_center:.3f}, range={y_range:.3f}")
    print(f"Heading: center={heading_center:.3f}, range={heading_range:.3f}")
    
    print(f"\nSuggested norm_odo parameters:")
    print(f"x_offset = {-x_min_p5:.3f}")
    print(f"x_scale = {x_range:.3f}")
    print(f"y_offset = {-y_min_p5:.3f}")
    print(f"y_scale = {y_range:.3f}")
    print(f"heading_offset = {-heading_min_p5:.3f}")
    print(f"heading_scale = {heading_range:.3f}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze Bench2Drive trajectory ranges")
    parser.add_argument(
        "--cache-path",
        type=str,
        required=True,
        help="Path to the cached dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to analyze (default: 1000)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cache_path):
        print(f"Error: Cache path does not exist: {args.cache_path}")
        sys.exit(1)
    
    analyze_trajectory_ranges(args.cache_path, args.num_samples)


if __name__ == "__main__":
    main()