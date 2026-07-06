# NavSim Data Format Documentation

This document provides comprehensive documentation of the NavSim data format used in DiffusionDrive, including data structures, coordinate systems, and feature extraction details.

## Table of Contents

- [Ego Pose Structure](#ego-pose-structure)
- [Compute Targets Output](#compute-targets-output)
- [Coordinate Systems](#coordinate-systems)
- [Data Flow and Transformations](#data-flow-and-transformations)

## Ego Pose Structure

### StateSE2 Format

The ego vehicle's pose in NavSim is represented using the `StateSE2` class from NuPlan:

```python
ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)
```

**Content**: `scene.frames[frame_idx].ego_status.ego_pose` contains 3 values:

- **Index 0**: X coordinate (meters)
- **Index 1**: Y coordinate (meters)  
- **Index 2**: Heading angle (radians)

**Coordinate System**: Global/World coordinates (NOT ego-relative)

## Compute Targets Output

The `compute_targets` function in `transfuser_features.py` returns a dictionary with four key components:

### 1. trajectory

- **Description**: Future trajectory of the ego vehicle
- **Shape**: `[8, 3]`
  - 8 timesteps (4 seconds / 0.5s interval)
  - 3 values per pose: [x, y, heading]
- **Units**:
  - x, y: meters
  - heading: radians
- **Coordinate System**: **Ego-relative** (relative to current ego rear-axle position)
- **Data Type**: `torch.float32`
- **Time Horizon**: 4 seconds into the future
- **Sampling Rate**: 0.5 seconds (2 Hz)

### 2. agent_states

- **Description**: Bounding box information for nearby vehicles
- **Shape**: `[30, 5]`
  - 30 agents maximum (`num_bounding_boxes=30`)
  - 5 attributes per agent: [x, y, heading, length, width]
- **Units**:
  - x, y, length, width: meters
  - heading: radians
- **Coordinate System**: **Ego-relative**
- **Data Type**: `torch.float32`
- **Selection Criteria**:
  - Only vehicles within lidar range (-32m to 32m in both x and y)
  - Sorted by distance, closest 30 are kept
  - Padded with zeros if fewer than 30 agents

### 3. agent_labels

- **Description**: Boolean mask indicating valid agent entries
- **Shape**: `[30]`
- **Values**:
  - `True`: Valid agent at this index
  - `False`: Padding (no agent)
- **Data Type**: `torch.bool`

### 4. bev_semantic_map

- **Description**: Bird's eye view semantic segmentation map
- **Shape**: `[128, 256]` (height × width in pixels)
- **Pixel Resolution**: 0.25 meters/pixel
- **Coverage Area**: 32m × 64m (centered on ego)
- **Coordinate System**: **Ego-centered** (ego at center of map)
- **Data Type**: `torch.int64`
- **Semantic Classes**:
  - 0: Background
  - 1: Road (lanes, intersections)
  - 2: Walkways
  - 3: Lane centerlines
  - 4: Static objects (barriers, cones, signs)
  - 5: Vehicles
  - 6: Pedestrians

## Coordinate Systems

### Global/World Coordinates

- Used for: Raw ego pose, map queries
- Origin: Fixed world origin for entire scene
- Example: `ego_pose` from scene data

### Ego-Relative Coordinates

- Used for: Trajectories, agent states
- Origin: Current ego vehicle position
- Transformation: Applied during feature extraction
- Forward direction: Positive X axis
- Left direction: Positive Y axis

### BEV Map Coordinates

- Pixel coordinates with ego at center
- Size: 256×128 pixels (width×height)
- Coverage: -32m to +32m in X, -16m to +16m in Y
- Resolution: 0.25m per pixel

## Data Flow and Transformations

### 1. Map Query Process

```python
# Query global map database around ego position
map_objects = map_api.get_proximal_map_objects(
    point=ego_pose.point,  # Global coordinates
    radius=32.0,           # Meters
    layers=[...]
)
```

### 2. Global to Ego-Relative Transformation

The `_geometry_local_coords` function transforms map objects from global to ego-relative:

```python
# Translation
xoff = -origin.x
yoff = -origin.y

# Rotation matrix
a = np.cos(origin.heading)
b = np.sin(origin.heading)
d = -np.sin(origin.heading)
e = np.cos(origin.heading)
```

### 3. Trajectory Extraction

Future trajectories are extracted in global coordinates then converted to ego-relative:

```python
# Get global poses
global_ego_poses = [frame.ego_status.ego_pose for frame in frames]

# Convert to ego-relative
local_ego_poses = convert_absolute_to_relative_se2_array(
    origin=StateSE2(*global_ego_poses[0]),
    positions=global_ego_poses[1:]
)
```

## Key Configuration Parameters

### Lidar Range

- X range: -32m to 32m
- Y range: -32m to 32m
- Used for agent filtering and BEV map bounds

### BEV Map Settings

- Width: 256 pixels
- Height: 128 pixels (half of width)
- Pixel size: 0.25m
- Total coverage: 64m × 32m

### Trajectory Sampling

- Time horizon: 4 seconds
- Interval: 0.5 seconds
- Number of points: 8

## Important Notes

1. **Coordinate Consistency**: All features returned by `compute_targets` are in ego-relative coordinates for neural network processing

2. **Agent Filtering**: Only vehicles within the lidar range are included in agent_states

3. **BEV Generation**: The semantic map is generated by querying the global map and transforming to ego-relative coordinates

4. **Padding Strategy**: Fixed-size arrays (30 agents) with boolean masks for variable-length data

5. **Units**: All distances in meters, all angles in radians throughout the pipeline
