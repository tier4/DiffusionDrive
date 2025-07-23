#!/usr/bin/env python3
"""
Verify that the normalization fixes are working correctly for Bench2Drive.
Tests the normalization pipeline to ensure no NaN values are produced.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from navsim.agents.diffusiondrive.extended_transfuser_config import ExtendedTransfuserConfig
from navsim.agents.diffusiondrive.transfuser_model_wrapper import V2TransfuserModelWrapper


def test_normalization():
    """Test normalization with both NavSim and Bench2Drive configs."""
    
    print("Testing trajectory normalization fixes...")
    
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
        print(f"  Heading: offset={norm_params['heading_offset']}, scale={norm_params['heading_scale']}")
        
        # Create test trajectories based on expected ranges
        if dataset_type == "navsim":
            # NavSim range: X:[-1.2, 55.7], Y:[-20, 26]
            test_trajectories = torch.tensor([
                [[0, 0, 0], [10, 5, 0.1], [20, 10, 0.2], [30, 15, 0.3]],  # Forward trajectory
                [[-1, -10, -0.5], [5, -5, -0.2], [15, 0, 0], [25, 5, 0.2]],  # Mixed trajectory
                [[50, 20, 1.5], [45, 15, 1.2], [40, 10, 0.9], [35, 5, 0.6]],  # Edge case
            ], dtype=torch.float32)
        else:
            # Bench2Drive range: X:[-35, 35], Y:[-35, 35]
            test_trajectories = torch.tensor([
                [[0, 0, 0], [10, 10, 0.05], [20, 20, 0.1], [30, 30, 0.15]],  # Diagonal
                [[-30, -30, -0.1], [-20, -20, -0.05], [-10, -10, 0], [0, 0, 0.05]],  # Negative
                [[30, -30, 0.2], [20, -20, 0.15], [10, -10, 0.1], [0, 0, 0.05]],  # Mixed
            ], dtype=torch.float32)
        
        print(f"\nTest trajectories shape: {test_trajectories.shape}")
        print(f"Test trajectories range: X:[{test_trajectories[..., 0].min():.1f}, {test_trajectories[..., 0].max():.1f}], "
              f"Y:[{test_trajectories[..., 1].min():.1f}, {test_trajectories[..., 1].max():.1f}]")
        
        # Create model wrapper to get normalized methods
        try:
            model = V2TransfuserModelWrapper(config)
            
            # Test normalization
            normalized = model._trajectory_head.norm_odo(test_trajectories)
            print(f"\nNormalized range: X:[{normalized[..., 0].min():.3f}, {normalized[..., 0].max():.3f}], "
                  f"Y:[{normalized[..., 1].min():.3f}, {normalized[..., 1].max():.3f}]")
            
            # Check for NaN
            has_nan = torch.isnan(normalized).any()
            print(f"Contains NaN: {has_nan}")
            
            # Check if in [-1, 1] range
            in_range = (normalized >= -1.0).all() and (normalized <= 1.0).all()
            print(f"All values in [-1, 1]: {in_range}")
            
            # Test denormalization
            denormalized = model._trajectory_head.denorm_odo(normalized)
            print(f"\nDenormalized range: X:[{denormalized[..., 0].min():.1f}, {denormalized[..., 0].max():.1f}], "
                  f"Y:[{denormalized[..., 1].min():.1f}, {denormalized[..., 1].max():.1f}]")
            
            # Check reconstruction error
            reconstruction_error = torch.abs(denormalized - test_trajectories).max()
            print(f"Max reconstruction error: {reconstruction_error:.6f}")
            
            if has_nan:
                print("\n⚠️  WARNING: NaN values detected in normalization!")
            elif not in_range:
                print("\n⚠️  WARNING: Normalized values outside [-1, 1] range!")
            elif reconstruction_error > 1e-3:
                print(f"\n⚠️  WARNING: High reconstruction error: {reconstruction_error}")
            else:
                print("\n✅ Normalization working correctly!")
                
        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            import traceback
            traceback.print_exc()


def test_edge_cases():
    """Test edge cases that might cause NaN."""
    print(f"\n{'='*60}")
    print("Testing edge cases for NaN prevention")
    print(f"{'='*60}")
    
    config = ExtendedTransfuserConfig(dataset_type="bench2drive")
    model = V2TransfuserModelWrapper(config)
    
    # Test cases that might cause NaN
    edge_cases = [
        ("Zero trajectory", torch.zeros((1, 4, 3))),
        ("Very large values", torch.tensor([[[100, 100, 3.14]]])),
        ("Very small values", torch.tensor([[[1e-6, 1e-6, 1e-6]]])),
        ("Mixed scales", torch.tensor([[[35, 0.001, 0.5]]])),
    ]
    
    for case_name, test_data in edge_cases:
        print(f"\nTesting: {case_name}")
        normalized = model._trajectory_head.norm_odo(test_data)
        has_nan = torch.isnan(normalized).any()
        in_range = (normalized >= -1.0).all() and (normalized <= 1.0).all()
        
        print(f"  Input: {test_data.squeeze()}")
        print(f"  Normalized: {normalized.squeeze()}")
        print(f"  Has NaN: {has_nan}")
        print(f"  In range [-1, 1]: {in_range}")


if __name__ == "__main__":
    test_normalization()
    test_edge_cases()
    
    print("\n" + "="*60)
    print("Normalization verification complete!")
    print("="*60)