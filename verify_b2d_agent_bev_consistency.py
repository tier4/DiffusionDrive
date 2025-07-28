#!/usr/bin/env python3
"""Verify Bench2Drive agent-BEV category consistency."""

import os
import pickle
import gzip
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import random

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

def verify_agent_bev_consistency(cache_dir, num_samples=500):
    """Verify consistency between agent states and BEV semantic map."""
    
    results = {
        'total_samples': 0,
        'bev_maps_found': 0,
        'bev_class_distribution': defaultdict(int),
        'bev_shape_correct': 0,
        'bev_shape_incorrect': 0,
        'vehicle_pixels_found': 0,
        'pedestrian_pixels_found': 0,
        'examples': [],
        'bev_statistics': {
            'vehicle_pixel_counts': [],
            'pedestrian_pixel_counts': [],
            'road_pixel_counts': []
        }
    }
    
    # BEV semantic classes from NavSim
    # 0: background
    # 1: road (lanes, intersections)
    # 2: walkways
    # 3: lane centerlines
    # 4: static objects (barriers, cones)
    # 5: vehicles
    # 6: pedestrians
    
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
        
        # Check BEV semantic map
        if 'bev_semantic_map' in targets:
            bev_map = targets['bev_semantic_map']
            results['bev_maps_found'] += 1
            
            # Convert to numpy if tensor
            if isinstance(bev_map, torch.Tensor):
                bev_map = bev_map.cpu().numpy()
            
            # Check shape (should be [128, 256] for NavSim format)
            if bev_map.shape == (128, 256):
                results['bev_shape_correct'] += 1
                
                # Count each class
                unique_classes, counts = np.unique(bev_map, return_counts=True)
                for cls, count in zip(unique_classes, counts):
                    results['bev_class_distribution'][int(cls)] += count
                
                # Count specific classes
                vehicle_pixels = np.sum(bev_map == 5)
                pedestrian_pixels = np.sum(bev_map == 6)
                road_pixels = np.sum(bev_map == 1)
                
                results['bev_statistics']['vehicle_pixel_counts'].append(vehicle_pixels)
                results['bev_statistics']['pedestrian_pixel_counts'].append(pedestrian_pixels)
                results['bev_statistics']['road_pixel_counts'].append(road_pixels)
                
                if vehicle_pixels > 0:
                    results['vehicle_pixels_found'] += 1
                if pedestrian_pixels > 0:
                    results['pedestrian_pixels_found'] += 1
                
                # Store examples
                if len(results['examples']) < 5:
                    results['examples'].append({
                        'token': token,
                        'shape': bev_map.shape,
                        'unique_classes': unique_classes.tolist(),
                        'class_counts': {int(c): int(n) for c, n in zip(unique_classes, counts)},
                        'vehicle_pixels': vehicle_pixels,
                        'pedestrian_pixels': pedestrian_pixels
                    })
            else:
                results['bev_shape_incorrect'] += 1
                print(f"Incorrect BEV shape: {bev_map.shape} for token {token}")
        else:
            print(f"No bev_semantic_map found for token {token}")
        
        # Also check agent states for comparison
        if 'agent_states' in targets and 'agent_labels' in targets:
            agent_states = targets['agent_states']
            agent_labels = targets['agent_labels']
            
            if isinstance(agent_labels, torch.Tensor):
                agent_labels = agent_labels.cpu().numpy()
            
            # Count valid agents (should correlate with vehicle pixels in BEV)
            num_valid_agents = np.sum(agent_labels)
            
            # Note: Since all agent states are zero, we expect no correlation
    
    return results

def main():
    cache_dir = "/workspace/navsim_workspace/cache/bench2drive_Base_cache"
    
    print(f"Verifying Bench2Drive agent-BEV consistency...")
    print(f"Cache dir: {cache_dir}")
    
    results = verify_agent_bev_consistency(cache_dir)
    
    print(f"\n=== BEV Semantic Map Verification Results ===")
    print(f"Total samples analyzed: {results['total_samples']}")
    print(f"BEV maps found: {results['bev_maps_found']}")
    print(f"Correct shape [128, 256]: {results['bev_shape_correct']}")
    print(f"Incorrect shape: {results['bev_shape_incorrect']}")
    
    if results['bev_maps_found'] > 0:
        print(f"\nMaps with vehicle pixels: {results['vehicle_pixels_found']} ({results['vehicle_pixels_found']/results['bev_maps_found']*100:.1f}%)")
        print(f"Maps with pedestrian pixels: {results['pedestrian_pixels_found']} ({results['pedestrian_pixels_found']/results['bev_maps_found']*100:.1f}%)")
    
    print(f"\n=== BEV Class Distribution (Total Pixels) ===")
    total_pixels = sum(results['bev_class_distribution'].values())
    class_names = {
        0: "background",
        1: "road",
        2: "walkways", 
        3: "centerlines",
        4: "static objects",
        5: "vehicles",
        6: "pedestrians"
    }
    
    for cls in sorted(results['bev_class_distribution'].keys()):
        count = results['bev_class_distribution'][cls]
        pct = count / total_pixels * 100 if total_pixels > 0 else 0
        name = class_names.get(cls, f"unknown({cls})")
        print(f"Class {cls} ({name}): {count:,} pixels ({pct:.2f}%)")
    
    if results['bev_statistics']['vehicle_pixel_counts']:
        print(f"\n=== Per-Sample Statistics ===")
        vehicle_counts = np.array(results['bev_statistics']['vehicle_pixel_counts'])
        pedestrian_counts = np.array(results['bev_statistics']['pedestrian_pixel_counts'])
        road_counts = np.array(results['bev_statistics']['road_pixel_counts'])
        
        print(f"Vehicle pixels per map:")
        print(f"  Mean: {vehicle_counts.mean():.1f}")
        print(f"  Max: {vehicle_counts.max()}")
        print(f"  Min: {vehicle_counts.min()}")
        
        print(f"\nPedestrian pixels per map:")
        print(f"  Mean: {pedestrian_counts.mean():.1f}")
        print(f"  Max: {pedestrian_counts.max()}")
        print(f"  Min: {pedestrian_counts.min()}")
        
        print(f"\nRoad pixels per map:")
        print(f"  Mean: {road_counts.mean():.1f}")
        print(f"  Max: {road_counts.max()}")
        print(f"  Min: {road_counts.min()}")
    
    if results['examples']:
        print(f"\n=== Example BEV Maps ===")
        for i, example in enumerate(results['examples'][:3]):
            print(f"\nExample {i+1} (token: {example['token']}):")
            print(f"  Shape: {example['shape']}")
            print(f"  Unique classes: {example['unique_classes']}")
            print(f"  Vehicle pixels: {example['vehicle_pixels']}")
            print(f"  Pedestrian pixels: {example['pedestrian_pixels']}")
            print(f"  Class distribution: {example['class_counts']}")
    
    # Summary
    print(f"\n=== Summary ===")
    if results['vehicle_pixels_found'] == 0:
        print(f"✗ WARNING: No vehicle pixels found in any BEV maps!")
        print(f"This is inconsistent since Bench2Drive scenarios should contain vehicles")
    else:
        print(f"✓ Vehicle pixels found in {results['vehicle_pixels_found']} maps")
    
    print(f"\nNote: Since agent_states are all zeros (as verified), we cannot check")
    print(f"agent-BEV position consistency. The BEV generation appears to be")
    print(f"using a simplified method without proper agent integration.")

if __name__ == "__main__":
    main()