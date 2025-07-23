#!/usr/bin/env python3
"""
Simple test of normalization functions without loading the full model.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from navsim.agents.diffusiondrive.extended_transfuser_config import ExtendedTransfuserConfig


def test_normalization_formulas():
    """Test normalization formulas directly."""
    
    print("Testing normalization formulas...")
    
    # Test configurations
    configs = [
        ("NavSim", "navsim"),
        ("Bench2Drive", "bench2drive")
    ]
    
    for config_name, dataset_type in configs:
        print(f"\n{'-'*60}")
        print(f"Testing {config_name} normalization")
        print(f"{'-'*60}")
        
        # Create config
        config = ExtendedTransfuserConfig(dataset_type=dataset_type)
        norm_params = config.get_normalization_params()
        
        print(f"Normalization parameters:")
        print(f"  X: offset={norm_params['x_offset']}, scale={norm_params['x_scale']}")
        print(f"  Y: offset={norm_params['y_offset']}, scale={norm_params['y_scale']}")
        
        # Test values
        if dataset_type == "navsim":
            # NavSim expected range
            test_values = [
                ("Min X", -1.2, "x"),
                ("Max X", 55.7, "x"),
                ("Min Y", -20.0, "y"),
                ("Max Y", 26.0, "y"),
                ("Center", 0.0, "x"),
            ]
        else:
            # Bench2Drive expected range
            test_values = [
                ("Min X", -35.0, "x"),
                ("Max X", 35.0, "x"),
                ("Min Y", -35.0, "y"),
                ("Max Y", 35.0, "y"),
                ("Center", 0.0, "x"),
            ]
        
        print("\nTesting normalization:")
        for name, value, coord in test_values:
            offset = norm_params[f"{coord}_offset"]
            scale = norm_params[f"{coord}_scale"]
            
            # Normalize
            normalized = 2 * (value + offset) / scale - 1
            
            # Denormalize
            denormalized = (normalized + 1) / 2 * scale - offset
            
            print(f"  {name}: {value:.1f} -> {normalized:.3f} -> {denormalized:.1f}")
            
            # Check if normalized is in [-1, 1]
            if normalized < -1.0 or normalized > 1.0:
                print(f"    ⚠️  WARNING: Normalized value {normalized:.3f} outside [-1, 1]!")
            
            # Check reconstruction
            error = abs(denormalized - value)
            if error > 1e-4:
                print(f"    ⚠️  WARNING: Reconstruction error: {error:.6f}")


def test_trajectory_batch():
    """Test normalization on trajectory batches."""
    print(f"\n{'='*60}")
    print("Testing batch trajectory normalization")
    print(f"{'='*60}")
    
    # Create Bench2Drive config
    config = ExtendedTransfuserConfig(dataset_type="bench2drive")
    norm_params = config.get_normalization_params()
    
    # Create test batch similar to training data
    batch_size = 64
    num_modes = 20
    num_timesteps = 8
    
    # Random trajectories in Bench2Drive range
    trajectories = torch.randn(batch_size, num_modes, num_timesteps, 2) * 25.0  # [-25, 25] range
    
    print(f"Test batch shape: {trajectories.shape}")
    print(f"Original range: X:[{trajectories[..., 0].min():.1f}, {trajectories[..., 0].max():.1f}], "
          f"Y:[{trajectories[..., 1].min():.1f}, {trajectories[..., 1].max():.1f}]")
    
    # Normalize
    traj_x = trajectories[..., 0:1]
    traj_y = trajectories[..., 1:2]
    
    norm_x = 2 * (traj_x + norm_params["x_offset"]) / norm_params["x_scale"] - 1
    norm_y = 2 * (traj_y + norm_params["y_offset"]) / norm_params["y_scale"] - 1
    
    # Clamp
    norm_x = torch.clamp(norm_x, min=-1.0, max=1.0)
    norm_y = torch.clamp(norm_y, min=-1.0, max=1.0)
    
    normalized = torch.cat([norm_x, norm_y], dim=-1)
    
    print(f"Normalized range: X:[{normalized[..., 0].min():.3f}, {normalized[..., 0].max():.3f}], "
          f"Y:[{normalized[..., 1].min():.3f}, {normalized[..., 1].max():.3f}]")
    
    # Check for NaN
    has_nan = torch.isnan(normalized).any()
    print(f"Contains NaN: {has_nan}")
    
    # Check how many values got clamped
    clamped_x = ((norm_x == -1.0) | (norm_x == 1.0)).sum()
    clamped_y = ((norm_y == -1.0) | (norm_y == 1.0)).sum()
    total_values = norm_x.numel()
    
    print(f"Values clamped: X: {clamped_x}/{total_values} ({100*clamped_x/total_values:.1f}%), "
          f"Y: {clamped_y}/{total_values} ({100*clamped_y/total_values:.1f}%)")
    
    if has_nan:
        print("\n⚠️  WARNING: NaN values detected!")
    else:
        print("\n✅ No NaN values in batch normalization!")


if __name__ == "__main__":
    test_normalization_formulas()
    test_trajectory_batch()
    
    print("\n" + "="*60)
    print("Normalization test complete!")
    print("="*60)