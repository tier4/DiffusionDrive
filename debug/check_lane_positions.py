#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from navsim.common.bev_map_utils import load_map_data, extract_lane_points

# Load the Town02 map
map_path = Path('/workspace/DiffusionDrive/tests/test_data/bench2drive_sample/maps/Town02_HD_map.npz')
map_data = load_map_data(map_path)

# Find lanes near ego position (34.55, 109.44)
ego_x, ego_y = 34.55, 109.44
nearby_lanes = []

for road_id, road_data in map_data.items():
    for lane_id, lane_segments in road_data.items():
        if lane_id != 'Trigger_Volumes' and isinstance(lane_segments, list):
            for segment in lane_segments:
                points, lane_type = extract_lane_points(segment)
                if len(points) > 0:
                    # Check if any point is within 50m of ego
                    distances = np.sqrt((points[:, 0] - ego_x)**2 + (points[:, 1] - ego_y)**2)
                    min_dist = distances.min()
                    if min_dist < 50:
                        nearby_lanes.append({
                            'road': road_id,
                            'lane': lane_id, 
                            'type': lane_type,
                            'min_dist': min_dist,
                            'closest_point': points[distances.argmin()]
                        })

print(f'Ego position: ({ego_x:.2f}, {ego_y:.2f})')
print(f'Found {len(nearby_lanes)} lanes within 50m of ego')

# Show closest lanes
nearby_lanes.sort(key=lambda x: x['min_dist'])
for i, lane in enumerate(nearby_lanes[:5]):
    p = lane['closest_point']
    print(f"  {i+1}. Road {lane['road']}, Lane {lane['lane']}, Type {lane['type']}")
    print(f"     Closest point: ({p[0]:.2f}, {p[1]:.2f}), Distance: {lane['min_dist']:.2f}m")