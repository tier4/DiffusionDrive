#!/usr/bin/env python3
"""
Analyze the generated trajectory anchors to check for validity.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def analyze_anchors(anchor_path: str):
    """Analyze anchor trajectories."""
    print(f"Analyzing anchors from: {anchor_path}")
    
    # Load anchors
    anchors = np.load(anchor_path)
    print(f"Anchors shape: {anchors.shape}")  # Expected: (20, 8, 2) or (20, 8, 3)
    
    # Extract dimensions
    num_anchors, num_timesteps, dims = anchors.shape
    
    # Analyze ranges
    print("\n" + "="*60)
    print("ANCHOR TRAJECTORY ANALYSIS")
    print("="*60)
    
    if dims >= 2:
        x_vals = anchors[..., 0].flatten()
        y_vals = anchors[..., 1].flatten()
        
        print(f"\nX range: [{x_vals.min():.3f}, {x_vals.max():.3f}] (range: {x_vals.max() - x_vals.min():.3f})")
        print(f"Y range: [{y_vals.min():.3f}, {y_vals.max():.3f}] (range: {y_vals.max() - y_vals.min():.3f})")
        
        print(f"\nX stats: mean={x_vals.mean():.3f}, std={x_vals.std():.3f}")
        print(f"Y stats: mean={y_vals.mean():.3f}, std={y_vals.std():.3f}")
        
        # Check for invalid values
        print(f"\nChecking for invalid values:")
        print(f"  NaN values: {np.isnan(anchors).any()}")
        print(f"  Inf values: {np.isinf(anchors).any()}")
        
        # Analyze individual anchors
        print(f"\nIndividual anchor statistics:")
        for i in range(num_anchors):
            anchor = anchors[i]
            x_range = [anchor[:, 0].min(), anchor[:, 0].max()]
            y_range = [anchor[:, 1].min(), anchor[:, 1].max()]
            
            # Calculate trajectory length
            diffs = np.diff(anchor[:, :2], axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            total_length = distances.sum()
            
            print(f"  Anchor {i:2d}: X[{x_range[0]:6.2f}, {x_range[1]:6.2f}], "
                  f"Y[{y_range[0]:6.2f}, {y_range[1]:6.2f}], Length: {total_length:6.2f}")
        
        # Compare with expected Bench2Drive ranges
        print("\n" + "="*60)
        print("COMPARISON WITH BENCH2DRIVE DATA")
        print("="*60)
        
        print("\nExpected Bench2Drive ranges (from analysis):")
        print("  Full range: X:[-35.58, 33.66], Y:[-36.84, 34.61]")
        print("  5-95 percentile: X:[-11.00, 11.21], Y:[-8.13, 10.63]")
        
        print("\nActual anchor ranges:")
        print(f"  X: [{x_vals.min():.3f}, {x_vals.max():.3f}]")
        print(f"  Y: [{y_vals.min():.3f}, {y_vals.max():.3f}]")
        
        # Check coverage
        x_coverage = (x_vals.max() - x_vals.min()) / (33.66 - (-35.58)) * 100
        y_coverage = (y_vals.max() - y_vals.min()) / (34.61 - (-36.84)) * 100
        
        print(f"\nCoverage of full data range:")
        print(f"  X coverage: {x_coverage:.1f}%")
        print(f"  Y coverage: {y_coverage:.1f}%")
        
        # Plot anchors
        plt.figure(figsize=(10, 8))
        for i in range(num_anchors):
            anchor = anchors[i]
            plt.plot(anchor[:, 0], anchor[:, 1], 'o-', alpha=0.6, label=f'Anchor {i}' if i < 5 else '')
            plt.plot(anchor[0, 0], anchor[0, 1], 'go', markersize=8)  # Start
            plt.plot(anchor[-1, 0], anchor[-1, 1], 'ro', markersize=8)  # End
        
        # Add expected range box
        plt.axhline(-35.58, color='k', linestyle='--', alpha=0.3)
        plt.axhline(33.66, color='k', linestyle='--', alpha=0.3)
        plt.axvline(-36.84, color='k', linestyle='--', alpha=0.3)
        plt.axvline(34.61, color='k', linestyle='--', alpha=0.3)
        
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title(f'Trajectory Anchors ({num_anchors} clusters)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        if num_anchors <= 5:
            plt.legend()
        
        # Save plot
        plot_path = anchor_path.replace('.npy', '_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
    
    if dims >= 3:
        heading_vals = anchors[..., 2].flatten()
        print(f"\nHeading range: [{heading_vals.min():.3f}, {heading_vals.max():.3f}]")
        print(f"Heading stats: mean={heading_vals.mean():.3f}, std={heading_vals.std():.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze trajectory anchors")
    parser.add_argument("anchor_path", help="Path to anchor .npy file")
    args = parser.parse_args()
    
    if not Path(args.anchor_path).exists():
        print(f"Error: File not found: {args.anchor_path}")
        sys.exit(1)
    
    analyze_anchors(args.anchor_path)


if __name__ == "__main__":
    main()