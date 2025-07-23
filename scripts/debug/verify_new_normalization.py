#!/usr/bin/env python3
"""
Verify the new normalization parameters work correctly.
"""

import torch
import numpy as np


def test_new_normalization():
    """Test the new normalization parameters."""
    
    # New normalization parameters
    params = {
        "x_offset": 0.955,
        "x_scale": 75.0,
        "y_offset": 1.115,
        "y_scale": 75.0,
        "heading_offset": 0.0,
        "heading_scale": 0.25
    }
    
    print("New Bench2Drive normalization parameters:")
    print(f"  X: offset={params['x_offset']}, scale={params['x_scale']}")
    print(f"  Y: offset={params['y_offset']}, scale={params['y_scale']}")
    print(f"  Heading: offset={params['heading_offset']}, scale={params['heading_scale']}")
    
    # Test with extreme values from the dataset
    test_cases = [
        ("X min", -35.58, "x"),
        ("X max", 33.66, "x"),
        ("Y min", -36.84, "y"),
        ("Y max", 34.61, "y"),
        ("X center", 0.0, "x"),
        ("Y center", 0.0, "y"),
        ("X 5th percentile", -11.0, "x"),
        ("X 95th percentile", 11.21, "x"),
        ("Y 5th percentile", -8.13, "y"),
        ("Y 95th percentile", 10.63, "y"),
        ("Heading min", -0.11, "heading"),
        ("Heading max", 0.11, "heading"),
    ]
    
    print("\nTesting normalization:")
    all_valid = True
    
    for name, value, coord in test_cases:
        offset = params[f"{coord}_offset"]
        scale = params[f"{coord}_scale"]
        
        # Normalize
        normalized = 2 * (value + offset) / scale - 1
        
        # Denormalize
        denormalized = (normalized + 1) / 2 * scale - offset
        
        # Check if in range
        in_range = -1.0 <= normalized <= 1.0
        error = abs(denormalized - value)
        
        status = "✓" if in_range else "✗"
        print(f"  {status} {name:20s}: {value:7.2f} -> {normalized:7.3f} -> {denormalized:7.2f} (error: {error:.6f})")
        
        if not in_range:
            all_valid = False
    
    print(f"\nAll values in [-1, 1] range: {'✓ YES' if all_valid else '✗ NO'}")
    
    # Test with anchor ranges
    print("\n" + "="*60)
    print("Testing with anchor ranges:")
    anchor_x_range = [-19.003, 19.287]
    anchor_y_range = [-19.578, 19.404]
    
    for coord, range_vals, param_prefix in [("X", anchor_x_range, "x"), ("Y", anchor_y_range, "y")]:
        offset = params[f"{param_prefix}_offset"]
        scale = params[f"{param_prefix}_scale"]
        
        min_norm = 2 * (range_vals[0] + offset) / scale - 1
        max_norm = 2 * (range_vals[1] + offset) / scale - 1
        
        print(f"  Anchor {coord} range: [{range_vals[0]:.3f}, {range_vals[1]:.3f}] -> [{min_norm:.3f}, {max_norm:.3f}]")
    
    # Calculate margins
    print("\n" + "="*60)
    print("Safety margins:")
    
    # X margins
    x_min_norm = 2 * (-35.58 + params["x_offset"]) / params["x_scale"] - 1
    x_max_norm = 2 * (33.66 + params["x_offset"]) / params["x_scale"] - 1
    print(f"  X: [{x_min_norm:.3f}, {x_max_norm:.3f}] (margins: {1.0 + x_min_norm:.3f}, {1.0 - x_max_norm:.3f})")
    
    # Y margins  
    y_min_norm = 2 * (-36.84 + params["y_offset"]) / params["y_scale"] - 1
    y_max_norm = 2 * (34.61 + params["y_offset"]) / params["y_scale"] - 1
    print(f"  Y: [{y_min_norm:.3f}, {y_max_norm:.3f}] (margins: {1.0 + y_min_norm:.3f}, {1.0 - y_max_norm:.3f})")


if __name__ == "__main__":
    test_new_normalization()