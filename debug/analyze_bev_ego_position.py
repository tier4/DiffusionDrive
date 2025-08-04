#!/usr/bin/env python3
"""
Analyze where ego vehicle actually is in BEV semantic maps.
"""

import numpy as np
import gzip
import pickle
from pathlib import Path

def analyze_bev_ego_position():
    """Check where roads typically appear in BEV semantic."""
    
    cache_dir = Path("/workspace/navsim_workspace/cache/Bench2Drive-Base-training_cache")
    
    # Collect statistics
    road_row_positions = []
    
    # Sample multiple scenarios
    count = 0
    for scenario_dir in list(cache_dir.iterdir())[:5]:  # Check 5 scenarios
        if scenario_dir.is_dir():
            for sample_dir in list(scenario_dir.iterdir())[:10]:  # 10 samples each
                if sample_dir.is_dir() and count < 50:
                    target_file = sample_dir / "transfuser_target.gz"
                    if target_file.exists():
                        with gzip.open(target_file, 'rb') as f:
                            targets = pickle.load(f)
                        
                        bev_sem = targets["bev_semantic_map"].numpy()
                        
                        # Find road pixels (class 1)
                        road_pixels = np.where(bev_sem == 1)
                        if len(road_pixels[0]) > 0:
                            # Get average row position of road
                            avg_road_row = np.mean(road_pixels[0])
                            road_row_positions.append(avg_road_row)
                            
                            # Also check trajectory start position
                            traj = targets["trajectory"].numpy()
                            if len(traj) > 0 and abs(traj[0, 0]) < 1.0:  # Near ego
                                print(f"Sample {count}: Road avg row={avg_road_row:.1f}, "
                                      f"BEV shape={bev_sem.shape}, "
                                      f"First traj point: x={traj[0,0]:.2f}, y={traj[0,1]:.2f}")
                        
                        count += 1
    
    if road_row_positions:
        avg_position = np.mean(road_row_positions)
        print(f"\n=== BEV Semantic Analysis ===")
        print(f"Average road row position: {avg_position:.1f}")
        print(f"BEV height is typically 128")
        print(f"If ego at bottom: expect road near row 120-127")
        print(f"If ego at top: expect road near row 0-20")
        print(f"If ego at center: expect road near row 64")
        
        if avg_position > 100:
            print("\n=> EGO IS AT BOTTOM (high row number)")
        elif avg_position < 30:
            print("\n=> EGO IS AT TOP (low row number)")
        else:
            print("\n=> EGO IS AT CENTER")

if __name__ == "__main__":
    analyze_bev_ego_position()