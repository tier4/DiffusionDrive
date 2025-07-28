#!/usr/bin/env python3
"""Investigate Bench2Drive world2ego matrix and bounding box structure."""

import json
import gzip
import numpy as np
from pathlib import Path
import random

def analyze_bounding_box_structure(bench2drive_dir, num_samples=10):
    """Analyze the structure of bounding boxes in Bench2Drive annotations."""
    
    # Get sample annotation files
    anno_files = []
    base_path = Path(bench2drive_dir)
    
    for scenario_dir in base_path.iterdir():
        if scenario_dir.is_dir():
            anno_dir = scenario_dir / "anno"
            if anno_dir.exists():
                for anno_file in anno_dir.glob("*.json.gz"):
                    anno_files.append(anno_file)
                    if len(anno_files) >= num_samples:
                        break
        if len(anno_files) >= num_samples:
            break
    
    print(f"Analyzing {len(anno_files)} annotation files...\n")
    
    # Analyze each file
    ego_fields = set()
    other_fields = set()
    class_types = set()
    has_world2ego = 0
    world2ego_examples = []
    
    for i, anno_file in enumerate(anno_files[:num_samples]):
        print(f"\n{'='*60}")
        print(f"File {i+1}: {anno_file.parent.parent.name}/{anno_file.name}")
        
        with gzip.open(anno_file, 'rt') as f:
            anno = json.load(f)
        
        if 'bounding_boxes' not in anno:
            print("No bounding_boxes found!")
            continue
            
        bboxes = anno['bounding_boxes']
        ego_found = False
        
        print(f"\nTotal bounding boxes: {len(bboxes)}")
        
        for j, bbox in enumerate(bboxes):
            obj_class = bbox.get('class', 'unknown')
            class_types.add(obj_class)
            
            if obj_class == 'ego_vehicle':
                ego_found = True
                ego_fields.update(bbox.keys())
                
                print(f"\nEgo vehicle found at index {j}")
                print(f"Ego fields: {sorted(bbox.keys())}")
                
                # Check for world2ego
                if 'world2ego' in bbox:
                    has_world2ego += 1
                    world2ego = bbox['world2ego']
                    print(f"\nworld2ego matrix found!")
                    print(f"Type: {type(world2ego)}")
                    if isinstance(world2ego, list):
                        print(f"Shape: {len(world2ego)}x{len(world2ego[0]) if world2ego and isinstance(world2ego[0], list) else '?'}")
                        if len(world2ego) == 4 and all(len(row) == 4 for row in world2ego):
                            print("Matrix (4x4):")
                            for row in world2ego:
                                print(f"  {[f'{val:8.3f}' for val in row]}")
                            world2ego_examples.append(np.array(world2ego))
                    
                # Show rotation values
                if 'rotation' in bbox:
                    print(f"Rotation: {bbox['rotation']} (roll, pitch, yaw in degrees)")
                
                # Show location
                if 'location' in bbox:
                    print(f"Location: {bbox['location']}")
                    
                # Show other key fields
                if 'speed' in bbox:
                    print(f"Speed: {bbox['speed']}")
            else:
                other_fields.update(bbox.keys())
        
        if not ego_found:
            print("\nWARNING: No ego vehicle found!")
        
        # Show first few non-ego objects
        print(f"\nOther object classes in this file:")
        class_counts = {}
        for bbox in bboxes:
            cls = bbox.get('class', 'unknown')
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nFiles with world2ego matrix: {has_world2ego}/{len(anno_files)}")
    print(f"\nAll ego vehicle fields: {sorted(ego_fields)}")
    print(f"\nAll other object fields: {sorted(other_fields)}")
    print(f"\nAll object classes found: {sorted(class_types)}")
    
    # Test coordinate transformation if we have world2ego
    if world2ego_examples:
        print(f"\n{'='*60}")
        print("TESTING COORDINATE TRANSFORMATION")
        print(f"{'='*60}")
        
        # Use first world2ego matrix
        W2E = world2ego_examples[0]
        print(f"\nUsing world2ego matrix:")
        print(W2E)
        
        # Test point
        world_point = np.array([10, 5, 0, 1])  # homogeneous coordinates
        ego_point = W2E @ world_point
        
        print(f"\nTest transformation:")
        print(f"World point: {world_point[:3]}")
        print(f"Ego point: {ego_point[:3]}")
        
        # Extract ego position from inverse of world2ego
        try:
            E2W = np.linalg.inv(W2E)
            ego_world_pos = E2W[:3, 3]
            print(f"\nEgo position in world (from inverse): {ego_world_pos}")
            
            # Extract rotation
            rotation_matrix = E2W[:3, :3]
            # Compute yaw from rotation matrix (assuming Z-up)
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            print(f"Ego yaw from matrix: {np.degrees(yaw):.2f} degrees")
        except:
            print("\nFailed to compute inverse of world2ego")

def main():
    bench2drive_dir = "/workspace/Bench2Drive-Base"
    analyze_bounding_box_structure(bench2drive_dir, num_samples=10)

if __name__ == "__main__":
    main()