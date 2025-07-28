#!/usr/bin/env python3
"""Verify Bench2Drive trajectory ego-relative coordinates and heading values."""

import os
import pickle
import gzip
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import random
import json

def load_cached_sample(cache_dir, token):
    """Load a cached sample's features and targets."""
    # Find the cache file
    log_name = "_".join(token.split("_")[:-1])
    token_dir = Path(cache_dir) / log_name / token
    
    if not token_dir.exists():
        return None, None
        
    # Load features
    feature_file = token_dir / "transfuser_feature.gz"
    target_file = token_dir / "transfuser_target.gz"
    
    features = None
    targets = None
    
    if feature_file.exists():
        with gzip.open(feature_file, 'rb') as f:
            features = pickle.load(f)
            
    if target_file.exists():
        with gzip.open(target_file, 'rb') as f:
            targets = pickle.load(f)
            
    return features, targets

def load_raw_annotation(bench2drive_dir, token):
    """Load raw annotation to compare with cached data."""
    # Parse token to get scenario and frame
    parts = token.split("_")
    frame_num = parts[-1]
    scenario = "_".join(parts[:-1])
    
    anno_file = Path(bench2drive_dir) / scenario / "anno" / f"{frame_num}.json.gz"
    
    if anno_file.exists():
        with gzip.open(anno_file, 'rt') as f:
            return json.load(f)
    return None

def verify_trajectory_coordinates(cache_dir, bench2drive_dir, num_samples=1000):
    """Verify trajectory coordinates are ego-relative and heading values."""
    
    results = {
        'total_samples': 0,
        'trajectories_found': 0,
        'correct_ego_relative': 0,
        'incorrect_start_position': 0,
        'heading_statistics': {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0,
            'std': 0,
            'all_headings': []
        },
        'position_statistics': {
            'first_waypoint_distances': [],
            'x_range': {'min': float('inf'), 'max': float('-inf')},
            'y_range': {'min': float('inf'), 'max': float('-inf')}
        },
        'examples': [],
        'heading_issues': []
    }
    
    # Get all available tokens
    all_tokens = []
    for log_dir in Path(cache_dir).iterdir():
        if log_dir.is_dir():
            for token_dir in log_dir.iterdir():
                if token_dir.is_dir():
                    all_tokens.append(token_dir.name)
    
    # Random sample
    sampled_tokens = random.sample(all_tokens, min(num_samples, len(all_tokens)))
    
    print(f"Checking {len(sampled_tokens)} cached samples...")
    
    for token in sampled_tokens:
        features, targets = load_cached_sample(cache_dir, token)
        
        if targets is None:
            continue
            
        results['total_samples'] += 1
        
        # Check trajectory
        if 'trajectory' in targets:
            trajectory = targets['trajectory']
            results['trajectories_found'] += 1
            
            # Convert to numpy if tensor
            if isinstance(trajectory, torch.Tensor):
                trajectory = trajectory.cpu().numpy()
            
            # Check shape (should be [8, 3] for 8 waypoints with x, y, heading)
            if trajectory.shape == (8, 3):
                # First waypoint should be close to ego (0, 0) if ego-relative
                first_x, first_y, first_heading = trajectory[0]
                distance_from_ego = np.sqrt(first_x**2 + first_y**2)
                
                results['position_statistics']['first_waypoint_distances'].append(distance_from_ego)
                
                # Check if reasonably close to ego (within 1 meter)
                if distance_from_ego < 1.0:
                    results['correct_ego_relative'] += 1
                else:
                    results['incorrect_start_position'] += 1
                
                # Collect all heading values
                all_headings = trajectory[:, 2]
                results['heading_statistics']['all_headings'].extend(all_headings.tolist())
                
                # Update position ranges
                x_vals = trajectory[:, 0]
                y_vals = trajectory[:, 1]
                results['position_statistics']['x_range']['min'] = min(results['position_statistics']['x_range']['min'], x_vals.min())
                results['position_statistics']['x_range']['max'] = max(results['position_statistics']['x_range']['max'], x_vals.max())
                results['position_statistics']['y_range']['min'] = min(results['position_statistics']['y_range']['min'], y_vals.min())
                results['position_statistics']['y_range']['max'] = max(results['position_statistics']['y_range']['max'], y_vals.max())
                
                # Check for heading issues (very small range indicates theta bug)
                heading_range = all_headings.max() - all_headings.min()
                if heading_range < 0.1:  # Very small heading variation
                    if len(results['heading_issues']) < 10:
                        # Try to load raw annotation to compare
                        raw_anno = load_raw_annotation(bench2drive_dir, token)
                        ego_yaw = None
                        if raw_anno and 'bounding_boxes' in raw_anno:
                            for bbox in raw_anno['bounding_boxes']:
                                if bbox.get('class') == 'ego_vehicle':
                                    if 'rotation' in bbox and isinstance(bbox['rotation'], list):
                                        ego_yaw = bbox['rotation'][2]  # yaw component
                                    break
                        
                        results['heading_issues'].append({
                            'token': token,
                            'heading_range': heading_range,
                            'headings': all_headings.tolist(),
                            'raw_theta': raw_anno.get('theta') if raw_anno else None,
                            'ego_yaw': ego_yaw
                        })
                
                # Store examples
                if len(results['examples']) < 5:
                    results['examples'].append({
                        'token': token,
                        'trajectory_shape': trajectory.shape,
                        'first_waypoint': trajectory[0].tolist(),
                        'last_waypoint': trajectory[-1].tolist(),
                        'heading_range': heading_range
                    })
            else:
                print(f"Unexpected trajectory shape: {trajectory.shape} for token {token}")
        else:
            print(f"No trajectory found for token {token}")
    
    # Calculate heading statistics
    if results['heading_statistics']['all_headings']:
        headings = np.array(results['heading_statistics']['all_headings'])
        results['heading_statistics']['min'] = headings.min()
        results['heading_statistics']['max'] = headings.max()
        results['heading_statistics']['mean'] = headings.mean()
        results['heading_statistics']['std'] = headings.std()
    
    return results

def main():
    cache_dir = "/workspace/navsim_workspace/cache/bench2drive_Base_cache"
    bench2drive_dir = "/workspace/Bench2Drive-Base"
    
    print(f"Verifying Bench2Drive trajectory coordinates...")
    print(f"Cache dir: {cache_dir}")
    print(f"Raw data dir: {bench2drive_dir}")
    
    results = verify_trajectory_coordinates(cache_dir, bench2drive_dir)
    
    print(f"\n=== Trajectory Verification Results ===")
    print(f"Total samples analyzed: {results['total_samples']}")
    print(f"Trajectories found: {results['trajectories_found']}")
    print(f"Correct ego-relative start: {results['correct_ego_relative']}")
    print(f"Incorrect start position: {results['incorrect_start_position']}")
    
    if results['trajectories_found'] > 0:
        ego_relative_rate = results['correct_ego_relative'] / results['trajectories_found'] * 100
        print(f"Ego-relative rate: {ego_relative_rate:.1f}%")
    
    print(f"\n=== Position Statistics ===")
    if results['position_statistics']['first_waypoint_distances']:
        distances = np.array(results['position_statistics']['first_waypoint_distances'])
        print(f"First waypoint distance from ego:")
        print(f"  Mean: {distances.mean():.3f} m")
        print(f"  Max: {distances.max():.3f} m")
        print(f"  Min: {distances.min():.3f} m")
    
    print(f"X coordinate range: [{results['position_statistics']['x_range']['min']:.2f}, {results['position_statistics']['x_range']['max']:.2f}] m")
    print(f"Y coordinate range: [{results['position_statistics']['y_range']['min']:.2f}, {results['position_statistics']['y_range']['max']:.2f}] m")
    
    print(f"\n=== Heading Statistics ===")
    print(f"Min heading: {results['heading_statistics']['min']:.3f} rad")
    print(f"Max heading: {results['heading_statistics']['max']:.3f} rad")
    print(f"Mean heading: {results['heading_statistics']['mean']:.3f} rad")
    print(f"Std heading: {results['heading_statistics']['std']:.3f} rad")
    print(f"Heading range: {results['heading_statistics']['max'] - results['heading_statistics']['min']:.3f} rad")
    
    if results['examples']:
        print(f"\n=== Example Trajectories ===")
        for i, example in enumerate(results['examples']):
            print(f"\nExample {i+1} (token: {example['token']}):")
            print(f"  Shape: {example['trajectory_shape']}")
            print(f"  First waypoint (x, y, heading): {example['first_waypoint']}")
            print(f"  Last waypoint (x, y, heading): {example['last_waypoint']}")
            print(f"  Heading range: {example['heading_range']:.3f} rad")
    
    if results['heading_issues']:
        print(f"\n=== Heading Issues (Low Variation) ===")
        print(f"Found {len(results['heading_issues'])} trajectories with heading range < 0.1 rad")
        for i, issue in enumerate(results['heading_issues'][:3]):
            print(f"\nIssue {i+1} (token: {issue['token']}):")
            print(f"  Heading range: {issue['heading_range']:.6f} rad")
            print(f"  Heading values: {issue['headings'][:3]} ... (first 3)")
            print(f"  Raw anno['theta']: {issue['raw_theta']}°")
            print(f"  Ego vehicle yaw: {issue['ego_yaw']}°")
    
    # Summary
    if results['heading_statistics']['std'] < 0.02:
        print(f"\n✗ WARNING: Very low heading variation detected (std={results['heading_statistics']['std']:.3f})")
        print(f"This confirms the heading bug - using anno['theta'] instead of ego yaw")
    else:
        print(f"\n✓ Heading values show reasonable variation")

if __name__ == "__main__":
    main()