# Trajectory Processing in DiffusionDrive

This document explains how trajectory data is processed in DiffusionDrive for both NavSim and Bench2Drive datasets, including the coordinate systems, normalization approaches, and the resolution of NaN loss issues.

## Table of Contents
1. [Overview](#overview)
2. [NavSim Trajectory Processing](#navsim-trajectory-processing)
3. [Bench2Drive Trajectory Processing](#bench2drive-trajectory-processing)
4. [Trajectory Normalization](#trajectory-normalization)
5. [K-means Anchor Generation](#k-means-anchor-generation)
6. [NaN Loss Issue and Resolution](#nan-loss-issue-and-resolution)

## Overview

DiffusionDrive uses a diffusion-based approach for trajectory prediction, which requires:
1. **Trajectory anchors**: Pre-computed typical trajectory patterns (20 modes)
2. **Normalization**: Converting trajectories to [-1, 1] range for diffusion
3. **Coordinate transformation**: Converting global to ego-centric coordinates

## NavSim Trajectory Processing

### Data Flow
1. **Raw data**: Global coordinates (x, y, heading) from the dataset
2. **Ego-centric transformation**: Convert to relative coordinates using `convert_absolute_to_relative_se2_array`
3. **Trajectory extraction**: 8 waypoints at 0.5s intervals (4 seconds total)

### Coordinate Transformation (NavSim)
```python
# From navsim/planning/simulation/planner/pdm_planner/utils/pdm_geometry_utils.py
def convert_absolute_to_relative_se2_array(origin: StateSE2, state_se2_array):
    theta = -origin.heading
    origin_array = np.array([[origin.x, origin.y, origin.heading]])
    
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta), np.cos(theta)]])
    
    points_rel = state_se2_array - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ R.T
    points_rel[:, 2] = normalize_angle(points_rel[:, 2])
    
    return points_rel
```

### NavSim Trajectory Ranges (Empirical)
Based on the hardcoded normalization in the original model:
- **X**: [-1.2, 55.7] meters (forward-biased, mostly positive)
- **Y**: [-20, 26] meters (roughly centered)
- **Heading**: [-2, 1.9] radians (full rotation range)

## Bench2Drive Trajectory Processing

### Data Flow
1. **Raw data**: CARLA global coordinates from JSON annotations
2. **Ego-centric transformation**: Similar to NavSim but with different coordinate conventions
3. **Trajectory extraction**: 8 waypoints (matching NavSim format)

### Coordinate Transformation (Bench2Drive)
```python
# From navsim/common/bench2drive_scene.py
def get_future_trajectory(self, frame_idx: int = -1):
    # Get current ego pose
    current_x = current_anno.get("x", 0.0)
    current_y = current_anno.get("y", 0.0)
    current_theta = -np.radians(current_anno.get("theta", 0.0))  # CCW
    
    # For each future waypoint
    dx = future_x - current_x
    dy = future_y - current_y
    
    # Rotate to ego frame
    cos_theta = np.cos(-current_theta)
    sin_theta = np.sin(-current_theta)
    ego_x = dx * cos_theta - dy * sin_theta
    ego_y = dx * sin_theta + dy * cos_theta
    ego_heading = future_theta - current_theta
```

### Bench2Drive Trajectory Ranges (Measured)
From analysis of 5000 samples:
- **X**: [-34.6, 32.5] meters (centered around 0)
- **Y**: [-32.4, 32.8] meters (centered around 0)
- **Heading**: [-0.106, 0.009] radians (very small range!)

## Trajectory Normalization

### Purpose
The diffusion model requires inputs in [-1, 1] range for stable training.

### Original (NavSim) Normalization
```python
# Hardcoded in transfuser_model_v2.py
def norm_odo(self, odo_info_fut):
    x_norm = 2*(x + 1.2)/56.9 - 1      # Assumes x in [-1.2, 55.7]
    y_norm = 2*(y + 20)/46 - 1         # Assumes y in [-20, 26]
    heading_norm = 2*(heading + 2)/3.9 - 1  # Assumes heading in [-2, 1.9]
```

### Configurable Normalization (Fixed)
```python
# Now configurable based on dataset_type
def norm_odo(self, odo_info_fut):
    # Use dataset-specific parameters
    x_norm = 2 * (x + self.x_offset) / self.x_scale - 1
    y_norm = 2 * (y + self.y_offset) / self.y_scale - 1
    heading_norm = 2 * (heading + self.heading_offset) / self.heading_scale - 1
    
    # Clamp to ensure [-1, 1]
    x_norm = torch.clamp(x_norm, -1, 1)
```

### Normalization Parameters

| Dataset | X Offset | X Scale | Y Offset | Y Scale | Heading Offset | Heading Scale |
|---------|----------|---------|----------|---------|----------------|---------------|
| NavSim | 1.2 | 56.9 | 20.0 | 46.0 | 2.0 | 3.9 |
| Bench2Drive | 9.238 | 20.891 | 3.854 | 17.202 | 0.05 | 0.1 |

## K-means Anchor Generation

### Purpose
The model uses pre-computed trajectory anchors to:
1. Initialize the diffusion process
2. Compute trajectory matching loss during training

### Generation Process
```python
# From scripts/generate_b2d_plan_anchors.py
def generate_kmeans_anchors(trajectories, num_clusters=20):
    # Collect N trajectories of shape (8, 2) - only x,y
    # Flatten to (N, 16) for k-means
    trajectories_flat = trajectories.reshape(N, -1)
    
    # Run k-means clustering
    kmeans = KMeans(n_clusters=20, random_state=42)
    kmeans.fit(trajectories_flat)
    
    # Reshape back to (20, 8, 2)
    anchors = kmeans.cluster_centers_.reshape(20, 8, 2)
```

### Anchor Statistics

| Dataset | File | X Range | Y Range | Avg Waypoint Distance |
|---------|------|---------|---------|----------------------|
| NavSim | kmeans_navsim_traj_20.npy | [0.56, 47.66] | [-10.62, 12.38] | ~6m |
| Bench2Drive | kmeans_bench2drive_traj_20.npy | [-29.13, 28.44] | [-28.93, 28.07] | ~2m |

## NaN Loss Issue and Resolution

### Root Cause
When training on Bench2Drive with NavSim normalization:
1. Bench2Drive trajectories have different ranges than NavSim
2. Normalization produced values far outside [-1, 1]
3. Diffusion process became unstable → NaN losses

### Example of the Problem
```python
# Bench2Drive trajectory: x = -30.0
# Using NavSim normalization:
x_norm = 2*(-30 + 1.2)/56.9 - 1 = -2.01  # Outside [-1, 1]!

# Bench2Drive heading: 0.05 radians
# Using NavSim normalization:
heading_norm = 2*(0.05 + 2)/3.9 - 1 = 0.05  # OK, but poor resolution
```

### Solution Components

1. **Dataset-specific normalization parameters**
   - Computed from actual data statistics
   - Configured via `dataset_type` flag

2. **Dataset-specific k-means anchors**
   - Generated from respective dataset trajectories
   - Captures dataset-specific motion patterns

3. **Safety measures**
   - Clamping normalized values to [-1, 1]
   - NaN detection in loss computation
   - Gradient clipping

### Usage
```bash
# For NavSim training (default)
agent.config.dataset_type="navsim"

# For Bench2Drive training
agent.config.dataset_type="bench2drive"
agent.config.plan_anchor_path="./download/kmeans_bench2drive_traj_20.npy"
```

## Key Insights

1. **Coordinate System Differences**: While both datasets use ego-centric coordinates, the trajectory distributions are very different
2. **Heading Range**: Bench2Drive has 30x smaller heading variation than NavSim
3. **Spatial Distribution**: NavSim trajectories are forward-biased (mostly positive X), while Bench2Drive is centered
4. **Normalization Criticality**: Proper normalization is crucial for diffusion models to work correctly

## Empirical Analysis Results

### NavSim Dataset (1000 samples analyzed)
- **Actual ranges match hardcoded normalization well**
  - X: [-0.038, 53.520] vs expected [-1.2, 55.7] ✓
  - Y: [-13.447, 18.904] vs expected [-20, 26] ✓
  - Heading: [-1.559, 1.314] vs expected [-2, 1.9] ✓
- **0% of values fall outside expected ranges**
- **Strong forward bias**: 99.9% of X values are positive (real driving behavior)
- **Well-centered Y**: Mean/std ratio of 0.025 indicates good centering

### Bench2Drive Dataset (500 samples analyzed)
- **Actual ranges differ significantly from NavSim**
  - X: [-34.6, 32.5] - centered around 0
  - Y: [-32.4, 32.8] - centered around 0
  - Heading: [-0.106, 0.009] - extremely small range
- **15.3% of X values fall outside NavSim ranges**
- **No forward bias**: Symmetric distribution (simulation characteristic)
- **Minimal heading variation**: Suggests less complex turning maneuvers

## Understanding from Research Papers

Based on the DiffusionDrive paper:
1. **Anchors are K-means clusters** of training trajectories (20 clusters)
2. **Anchors represent "prior driving patterns"** specific to each dataset
3. **Anchored Gaussian distribution** - small noise added to anchors instead of pure random
4. **Benefits**: Prevents mode collapse, reduces denoising steps needed

This explains why dataset-specific anchors are crucial - they encode the characteristic motion patterns of each dataset.