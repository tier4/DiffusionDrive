#!/usr/bin/env python3
"""Check if anno['speed'] matches ego vehicle speed in bounding boxes."""

import os
import json
import gzip
import numpy as np
from pathlib import Path
from collections import defaultdict

def check_speed_consistency(bench2drive_dir):
    """Check if anno['speed'] matches ego vehicle speed across files."""
    
    results = {
        'total_files': 0,
        'files_with_speed': 0,
        'files_with_ego_speed': 0,
        'matching_speed': 0,
        'mismatched_speed': 0,
        'speed_differences': [],
        'examples': []
    }
    
    # Iterate through scenario folders directly
    base_dir = Path(bench2drive_dir)
    
    for scenario_dir in base_dir.iterdir():
        if not scenario_dir.is_dir():
            continue
            
        # Check annotation files in anno subfolder
        anno_dir = scenario_dir / 'anno'
        if not anno_dir.exists():
            continue
            
        for anno_file in anno_dir.glob('*.json.gz'):
            if results['total_files'] >= 5000:  # Check more files
                break
                    
            results['total_files'] += 1
            
            try:
                with gzip.open(anno_file, 'rt') as f:
                    anno = json.load(f)
                    
                    # Check if anno has speed
                    anno_speed = anno.get('speed', None)
                    if anno_speed is not None:
                        results['files_with_speed'] += 1
                    
                    # Find ego vehicle in bounding boxes
                    ego_index = None
                    ego_speed = None
                    
                    if 'bounding_boxes' in anno:
                        for idx, bbox in enumerate(anno['bounding_boxes']):
                            if bbox.get('class') == 'ego_vehicle':
                                ego_index = idx
                                # Check if bbox has speed/velocity
                                if 'speed' in bbox:
                                    ego_speed = bbox['speed']
                                    results['files_with_ego_speed'] += 1
                                elif 'velocity' in bbox:
                                    # Velocity might be [vx, vy, vz]
                                    vel = bbox['velocity']
                                    if isinstance(vel, list) and len(vel) >= 2:
                                        ego_speed = np.sqrt(vel[0]**2 + vel[1]**2)
                                        results['files_with_ego_speed'] += 1
                                break
                    
                    # Compare speeds
                    if anno_speed is not None and ego_speed is not None:
                        if isinstance(anno_speed, (int, float)) and isinstance(ego_speed, (int, float)):
                            diff = abs(anno_speed - ego_speed)
                            results['speed_differences'].append(diff)
                            
                            if diff < 0.01:  # Allow small tolerance
                                results['matching_speed'] += 1
                            else:
                                results['mismatched_speed'] += 1
                                
                                # Store examples of mismatches (or matches for inspection)
                                if len(results['examples']) < 5 or (diff < 0.01 and len(results['examples']) < 2):
                                    results['examples'].append({
                                        'file': str(anno_file),
                                        'anno_speed': anno_speed,
                                        'ego_speed': ego_speed,
                                        'difference': diff,
                                        'ego_bbox': anno['bounding_boxes'][ego_index] if ego_index is not None else None
                                    })
                    
            except Exception as e:
                print(f"Error processing {anno_file}: {e}")
                continue
    
    return results

def main():
    bench2drive_dir = "/workspace/Bench2Drive-Base"
    
    print("Checking speed consistency between anno['speed'] and ego vehicle speed...")
    results = check_speed_consistency(bench2drive_dir)
    
    print(f"\n=== Speed Consistency Analysis ===")
    print(f"Total annotation files: {results['total_files']}")
    print(f"Files with anno['speed']: {results['files_with_speed']}")
    print(f"Files with ego vehicle speed: {results['files_with_ego_speed']}")
    print(f"Matching speed values: {results['matching_speed']}")
    print(f"Mismatched speed values: {results['mismatched_speed']}")
    
    if results['speed_differences']:
        print(f"\nSpeed difference statistics:")
        print(f"  Mean: {np.mean(results['speed_differences']):.6f}")
        print(f"  Max: {np.max(results['speed_differences']):.6f}")
        print(f"  Min: {np.min(results['speed_differences']):.6f}")
        print(f"  Std: {np.std(results['speed_differences']):.6f}")
    
    if results['examples']:
        print(f"\n=== Example Mismatches ===")
        for i, example in enumerate(results['examples']):
            print(f"\nExample {i+1}:")
            print(f"  File: {example['file']}")
            print(f"  anno['speed']: {example['anno_speed']}")
            print(f"  ego vehicle speed: {example['ego_speed']}")
            print(f"  Difference: {example['difference']:.6f}")
            
            # Check what fields are in the ego bbox
            if example['ego_bbox']:
                print(f"  Ego bbox fields: {list(example['ego_bbox'].keys())}")
                if 'velocity' in example['ego_bbox']:
                    print(f"  Ego velocity: {example['ego_bbox']['velocity']}")
    
    # Final summary
    if results['mismatched_speed'] == 0 and results['matching_speed'] > 0:
        print(f"\n✓ All speed values match!")
    elif results['files_with_ego_speed'] == 0:
        print(f"\n✗ No ego vehicle speed found in bounding boxes!")
    else:
        match_rate = results['matching_speed'] / (results['matching_speed'] + results['mismatched_speed']) * 100 if (results['matching_speed'] + results['mismatched_speed']) > 0 else 0
        print(f"\n✗ Speed match rate: {match_rate:.1f}%")

if __name__ == "__main__":
    main()