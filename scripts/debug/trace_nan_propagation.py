#!/usr/bin/env python3
"""
Debug script to trace NaN propagation in Bench2Drive training.
Monitors values at each step of the loss computation.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from navsim.agents.diffusiondrive.extended_transfuser_config import ExtendedTransfuserConfig
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer


def check_tensor(tensor, name, verbose=True):
    """Check tensor for NaN/Inf and print statistics."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf or verbose:
        print(f"\n{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        
        if not has_nan and not has_inf:
            print(f"  Range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
            print(f"  Mean: {tensor.mean().item():.3f}, Std: {tensor.std().item():.3f}")
    
    return has_nan or has_inf


def test_normalization_extremes():
    """Test normalization with extreme values from Bench2Drive."""
    print("="*60)
    print("TESTING NORMALIZATION WITH EXTREME VALUES")
    print("="*60)
    
    # Create config with current (incorrect) normalization
    config = ExtendedTransfuserConfig(dataset_type="bench2drive")
    norm_params = config.get_normalization_params()
    
    print(f"\nCurrent normalization parameters:")
    print(f"  X: offset={norm_params['x_offset']}, scale={norm_params['x_scale']}")
    print(f"  Y: offset={norm_params['y_offset']}, scale={norm_params['y_scale']}")
    
    # Test extreme values from the dataset
    extreme_values = torch.tensor([
        [-35.58, -36.84],  # Min values
        [33.66, 34.61],    # Max values
        [0.0, 0.0],        # Center
        [-11.0, -8.13],    # 5th percentile
        [11.21, 10.63],    # 95th percentile
    ])
    
    print(f"\nTesting extreme values:")
    for i, val in enumerate(extreme_values):
        x, y = val
        # Normalize
        norm_x = 2 * (x + norm_params['x_offset']) / norm_params['x_scale'] - 1
        norm_y = 2 * (y + norm_params['y_offset']) / norm_params['y_scale'] - 1
        
        print(f"  [{x:6.2f}, {y:6.2f}] -> [{norm_x:6.3f}, {norm_y:6.3f}]")
        
        if abs(norm_x) > 1.0 or abs(norm_y) > 1.0:
            print(f"    ⚠️  WARNING: Normalized value outside [-1, 1]!")


def simulate_loss_computation():
    """Simulate the loss computation to find where NaN occurs."""
    print("\n" + "="*60)
    print("SIMULATING LOSS COMPUTATION")
    print("="*60)
    
    # Setup
    batch_size = 2
    num_modes = 20
    num_timesteps = 8
    dims = 3
    
    # Create test data
    config = ExtendedTransfuserConfig(dataset_type="bench2drive")
    loss_computer = LossComputer(config)
    
    # Simulate predicted trajectories (after diffusion)
    poses_reg = torch.randn(batch_size, num_modes, num_timesteps, dims) * 10.0
    poses_cls = torch.randn(batch_size, num_modes)  # Logits for classification
    
    # Simulate target trajectory from Bench2Drive
    # Using realistic values from the dataset
    target_traj = torch.tensor([
        [[0.0, 0.0, 0.0], [2.0, 1.0, 0.1], [4.0, 2.0, 0.1], [6.0, 3.0, 0.1],
         [8.0, 4.0, 0.1], [10.0, 5.0, 0.1], [12.0, 6.0, 0.1], [14.0, 7.0, 0.1]],
        [[-5.0, -5.0, -0.1], [-3.0, -4.0, -0.05], [-1.0, -3.0, 0.0], [1.0, -2.0, 0.05],
         [3.0, -1.0, 0.05], [5.0, 0.0, 0.05], [7.0, 1.0, 0.05], [9.0, 2.0, 0.05]]
    ])
    
    # Load actual anchors
    anchor_path = Path("/workspace/DiffusionDrive/download/kmeans_bench2drive_traj_20.npy")
    if anchor_path.exists():
        plan_anchor_np = np.load(anchor_path)
        plan_anchor = torch.tensor(plan_anchor_np, dtype=torch.float32)
        plan_anchor = plan_anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        print(f"\nLoaded anchors: {plan_anchor.shape}")
    else:
        print("\n⚠️  Warning: Anchor file not found, using random anchors")
        plan_anchor = torch.randn(batch_size, num_modes, num_timesteps, 2) * 15.0
    
    targets = {"trajectory": target_traj}
    
    # Check inputs
    print("\nChecking inputs:")
    check_tensor(poses_reg, "poses_reg", verbose=False)
    check_tensor(poses_cls, "poses_cls", verbose=False)
    check_tensor(target_traj, "target_traj", verbose=False)
    check_tensor(plan_anchor, "plan_anchor", verbose=False)
    
    # Step through loss computation
    print("\nStep-by-step loss computation:")
    
    # 1. Distance computation
    dist = torch.linalg.norm(target_traj.unsqueeze(1)[...,:2] - plan_anchor, dim=-1)
    check_tensor(dist, "Distance between target and anchors")
    
    # 2. Mean distance
    dist_mean = dist.mean(dim=-1)
    check_tensor(dist_mean, "Mean distance")
    
    # 3. Find closest anchor
    mode_idx = torch.argmin(dist_mean, dim=-1)
    print(f"\nClosest anchor indices: {mode_idx}")
    
    # 4. Gather best predictions
    mode_idx_expanded = mode_idx[...,None,None,None].repeat(1,1,num_timesteps,dims)
    best_reg = torch.gather(poses_reg, 1, mode_idx_expanded).squeeze(1)
    check_tensor(best_reg, "Best regression predictions")
    
    # 5. Classification target
    cls_target = mode_idx
    target_classes_onehot = torch.zeros([batch_size, num_modes],
                                      dtype=poses_cls.dtype,
                                      device=poses_cls.device)
    target_classes_onehot.scatter_(1, cls_target.unsqueeze(1), 1)
    check_tensor(target_classes_onehot, "One-hot classification target")
    
    # 6. Focal loss computation
    print("\nFocal loss computation:")
    
    # Sigmoid of predictions
    pred_sigmoid = poses_cls.sigmoid()
    check_tensor(pred_sigmoid, "Sigmoid of predictions")
    
    # Focal loss terms
    target = target_classes_onehot.type_as(poses_cls)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    check_tensor(pt, "pt term")
    
    focal_weight = (0.25 * target + 0.75 * (1 - target)) * pt.pow(2.0)
    check_tensor(focal_weight, "Focal weight")
    
    # Binary cross entropy
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        poses_cls, target, reduction='none')
    check_tensor(bce_loss, "BCE loss")
    
    weighted_loss = bce_loss * focal_weight
    check_tensor(weighted_loss, "Weighted loss")
    
    loss_cls = weighted_loss.mean()
    check_tensor(loss_cls, "Final classification loss")
    
    # 7. Regression loss
    reg_loss = torch.nn.functional.l1_loss(best_reg, target_traj)
    check_tensor(reg_loss, "Regression loss")
    
    # 8. Total loss
    total_loss = loss_cls + reg_loss
    check_tensor(total_loss, "Total loss")
    
    print(f"\nFinal losses:")
    print(f"  Classification loss: {loss_cls.item():.3f}")
    print(f"  Regression loss: {reg_loss.item():.3f}")
    print(f"  Total loss: {total_loss.item():.3f}")


def test_with_actual_normalization():
    """Test with the suggested normalization parameters."""
    print("\n" + "="*60)
    print("TESTING WITH SUGGESTED NORMALIZATION")
    print("="*60)
    
    # Suggested parameters from analysis
    suggested_params = {
        "x_offset": 11.053,
        "x_scale": 22.208,
        "y_offset": 8.285,
        "y_scale": 18.766,
    }
    
    print(f"\nSuggested normalization parameters:")
    print(f"  X: offset={suggested_params['x_offset']:.3f}, scale={suggested_params['x_scale']:.3f}")
    print(f"  Y: offset={suggested_params['y_offset']:.3f}, scale={suggested_params['y_scale']:.3f}")
    
    # Test extreme values
    extreme_values = torch.tensor([
        [-35.58, -36.84],  # Min values
        [33.66, 34.61],    # Max values
        [-11.0, -8.13],    # 5th percentile
        [11.21, 10.63],    # 95th percentile
    ])
    
    print(f"\nTesting extreme values with suggested normalization:")
    for i, val in enumerate(extreme_values):
        x, y = val
        # Normalize with suggested params
        norm_x = 2 * (x + suggested_params['x_offset']) / suggested_params['x_scale'] - 1
        norm_y = 2 * (y + suggested_params['y_offset']) / suggested_params['y_scale'] - 1
        
        print(f"  [{x:6.2f}, {y:6.2f}] -> [{norm_x:6.3f}, {norm_y:6.3f}]")
        
        if abs(norm_x) > 1.0 or abs(norm_y) > 1.0:
            print(f"    ⚠️  WARNING: Normalized value outside [-1, 1]!")


if __name__ == "__main__":
    test_normalization_extremes()
    simulate_loss_computation()
    test_with_actual_normalization()
    
    print("\n" + "="*60)
    print("NaN propagation analysis complete!")
    print("="*60)