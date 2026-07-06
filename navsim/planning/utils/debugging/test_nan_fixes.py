#!/usr/bin/env python3
"""
Quick test script to verify NaN fixes for Bench2Drive training.
Tests the data normalization fixes before running full training.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.common.bench2drive_dataloader import Bench2DriveDataConfig, Bench2DriveSceneLoader
from navsim.planning.training.bench2drive_dataset import Bench2DriveDataset
from torch.utils.data import DataLoader

def test_single_batch():
    """Test a single batch through the model to check for NaN."""
    print("=== Testing Single Batch with Fixes ===\n")
    
    # Initialize config and agent
    config = TransfuserConfig()
    agent = TransfuserAgent(config)
    
    # Create minimal dataset for testing
    cache_path = "/workspace/DiffusionDrive/cache/bench2drive_mini"  # Adjust as needed
    
    # Create dataset using cache
    from navsim.planning.training.dataset import CacheOnlyDataset
    from navsim.agents.diffusiondrive.transfuser_features_b2d import (
        Bench2DriveFeatureBuilder,
        Bench2DriveTargetBuilder
    )
    
    # Get first available scenario
    cache_dir = Path(cache_path)
    if not cache_dir.exists():
        print(f"Error: Cache directory {cache_dir} does not exist!")
        print("Please run caching first or adjust the path.")
        return False
    
    scenarios = [d.name for d in cache_dir.iterdir() if d.is_dir()][:1]  # Just one scenario
    if not scenarios:
        print("Error: No cached scenarios found!")
        return False
    
    print(f"Testing with scenario: {scenarios[0]}")
    
    dataset = CacheOnlyDataset(
        cache_path=str(cache_path),
        feature_builders=[Bench2DriveFeatureBuilder(config)],
        target_builders=[Bench2DriveTargetBuilder(config)],
        log_names=scenarios
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Get one batch
    features, targets = next(iter(dataloader))
    
    print("\n--- Feature Statistics ---")
    for name, tensor in features.items():
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
        print(f"  Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")
        print(f"  Has NaN: {torch.isnan(tensor).any()}")
        print(f"  Has Inf: {torch.isinf(tensor).any()}")
    
    print("\n--- Target Statistics ---")
    for name, tensor in targets.items():
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        if tensor.dtype in [torch.float32, torch.float64]:
            print(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
            print(f"  Has NaN: {torch.isnan(tensor).any()}")
        else:
            print(f"  Unique values: {torch.unique(tensor).tolist()}")
    
    # Test forward pass
    print("\n--- Testing Forward Pass ---")
    try:
        with torch.no_grad():
            # Move to GPU if available
            if torch.cuda.is_available():
                agent = agent.cuda()
                features = {k: v.cuda() for k, v in features.items()}
                targets = {k: v.cuda() for k, v in targets.items()}
            
            # Forward pass
            predictions = agent.forward(features, targets)
            
            print("Forward pass successful!")
            
            # Check predictions
            for name, tensor in predictions.items():
                has_nan = torch.isnan(tensor).any()
                has_inf = torch.isinf(tensor).any()
                print(f"{name}: shape={tensor.shape}, has_nan={has_nan}, has_inf={has_inf}")
                
                if has_nan or has_inf:
                    print(f"  WARNING: {name} contains invalid values!")
                    return False
        
        # Test loss computation
        print("\n--- Testing Loss Computation ---")
        loss_dict = agent.compute_loss(features, targets, predictions)
        
        for name, loss in loss_dict.items():
            if loss is not None:
                has_nan = torch.isnan(loss).any()
                print(f"{name}: {loss.item():.4f}, has_nan={has_nan}")
                
                if has_nan:
                    print(f"  ERROR: {name} is NaN!")
                    return False
        
        print("\n✅ All tests passed! No NaN detected.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_recommendations():
    """Print recommendations based on test results."""
    print("\n=== Recommendations ===")
    print("1. If the test passed:")
    print("   - Clear your cache and regenerate with the fixed feature builder")
    print("   - Start training with a lower learning rate (e.g., 1e-5)")
    print("   - Monitor gradient norms in tensorboard")
    print()
    print("2. If the test failed:")
    print("   - Check the error message and stack trace")
    print("   - Run the validation script to check data ranges")
    print("   - Consider adding more debugging to isolate the issue")
    print()
    print("3. To clear and regenerate cache:")
    print("   rm -rf /path/to/cache/bench2drive_mini")
    print("   python3 navsim/planning/script/run_dataset_caching.py ...")


if __name__ == "__main__":
    print("Testing NaN fixes for Bench2Drive training...\n")
    
    success = test_single_batch()
    
    if not success:
        print("\n⚠️  Test failed! See errors above.")
    
    print_recommendations()