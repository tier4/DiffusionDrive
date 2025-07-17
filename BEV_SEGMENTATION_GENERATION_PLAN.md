# BEV Segmentation Generation Plan for Bench2Drive

## Overview

This document outlines the plan for generating Bird's Eye View (BEV) semantic segmentation maps for the Bench2Drive dataset, which are currently missing but required for DiffusionDrive integration.

## Current State Analysis

### Available Resources in Bench2Drive

1. **Top-Down Camera** (`tools/data_collect.py`):
   - Position: 50m above vehicle (z=50.0)
   - Rotation: -90° pitch (pointing down)
   - Resolution: 1600x900
   - FOV: 110°
   - Provides RGB images but NO semantic segmentation

2. **LiDAR BEV Visualization** (`tools/visualize.py`):
   - Projects LiDAR points to 2D plane
   - 85m x 85m range
   - Only for visualization, not semantic segmentation

3. **Perspective Semantic Views**:
   - `semantic_front`, `semantic_back`, `semantic_left`, `semantic_right`
   - Contains full semantic segmentation from CARLA
   - 23 semantic classes that need mapping to 7 classes

4. **HD Maps** (`tools/gen_hdmap.py`):
   - Road geometry and lane information
   - Could supplement BEV generation

### Critical Missing Components

- **NO semantic top-down camera output** - only RGB top-down exists
- **NO BEV semantic segmentation generation pipeline**
- **NO perspective-to-BEV projection code**

### DiffusionDrive Integration Issue

- DiffusionDrive expects BEV semantic maps (loss weight: 14.0)
- Current implementation returns **placeholder zeros** in `bench2drive_scene.py:get_bev_semantic_map()`
- This severely limits training effectiveness

### Semantic Analysis Results (Bench2Drive Base Dataset)

**Analysis of 13,770 semantic images from Bench2Drive base dataset:**
- **Total unique pixel values found**: 27 values
- **Actual values present**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
- **Missing values**: [16, 17] (Bus=16, Train=17)
- **Key finding**: Not all semantic categories defined in CARLA are present in the base dataset

### Required Category Mapping

23 CARLA semantic classes → 7 BEV classes:
- 0: Background (buildings, vegetation, sky, etc.)
- 1: Road (roads, road lines, ground)
- 2: Walkways (sidewalks)
- 3: Lane centerlines
- 4: Static objects (signs, traffic lights, barriers)
- 5: Vehicles (cars, trucks, buses, motorcycles)
- 6: Pedestrians (walkers, riders)

## ✅ SOLUTION FOUND: Bench2DriveZoo Implementation

**Discovery**: Extensive BEV segmentation implementations already exist in Bench2DriveZoo repository that can be directly adapted for DiffusionDrive!

### Key Implementation Components in Bench2DriveZoo

#### 1. Core BEV Transformation Pipeline

**File**: `/workspace/Bench2DriveZoo/mmcv/models/modules/transformerV2.py`
- `PerceptionTransformerBEVEncoder` class
- Multi-level feature processing for BEV transformation
- Temporal BEV features with rotation and scaling transformations

**File**: `/workspace/Bench2DriveZoo/mmcv/models/modules/spatial_cross_attention.py`
- `SpatialCrossAttention` and `MSDeformableAttention3D` classes
- Perspective-to-BEV mapping using deformable attention
- Camera-to-BEV coordinate transformation with reference points

#### 2. BEV Segmentation Head

**File**: `/workspace/Bench2DriveZoo/mmcv/models/dense_heads/panseg_head.py`
- `PansegformerHead` class for panoptic segmentation
- Generates semantic maps including:
  - **Drivable areas** (`drivable_pred`)
  - **Lane segmentation** (`lane_pred`) with classes:
    - Divider lanes (class 0)
    - Crossing lanes (class 1)
    - Contour lanes (class 2)
- Outputs BEV semantic maps with 7 classes: road, walkway, vehicles, etc.

#### 3. Agent Implementation with BEV Generation

**File**: `/workspace/Bench2DriveZoo/team_code/vad_b2d_agent_visualize.py`
- `VadAgent` class with actual BEV map generation
- Coordinate transformation matrices (`coor2topdown`)
- Methods for BEV visualization and perspective-to-BEV conversion

#### 4. UniAD BEV Visualization Framework

**File**: `/workspace/Bench2DriveZoo/adzoo/uniad/analysis_tools/visualize/render/bev_render.py`
- `BEVRender` class for BEV visualization
- Methods for rendering predicted semantic maps, occupancy maps, and planning trajectories
- Handles map color coding and BEV coordinate transformations

### Recommended Implementation Strategy

#### Phase 1: Extract Core Components from Bench2DriveZoo

1. **Spatial Cross-Attention Module**:
   - Copy and adapt `spatial_cross_attention.py` components
   - Focus on `MSDeformableAttention3D` for perspective-to-BEV mapping

2. **BEV Transformer Encoder**:
   - Extract `PerceptionTransformerBEVEncoder` from `transformerV2.py`
   - Adapt for DiffusionDrive's multi-camera setup

3. **Panoptic Segmentation Head**:
   - Adapt `PansegformerHead` from `panseg_head.py`
   - Configure for DiffusionDrive's 7-class BEV semantic map

#### Phase 2: Create Integration Module

Create new file: `navsim/agents/diffusiondrive/bev_semantic_generator.py`

```python
class BEVSemanticGenerator:
    """
    Generate BEV semantic maps from multi-camera perspective views
    Adapted from Bench2DriveZoo implementations
    """
    
    def __init__(self, camera_configs, bev_size=(256, 128)):
        self.spatial_attention = MSDeformableAttention3D(...)
        self.bev_encoder = PerceptionTransformerBEVEncoder(...)
        self.segmentation_head = PansegformerHead(...)
        self.coordinate_transform = CoordinateTransform(...)
    
    def generate_bev_semantic(self, camera_features, ego_transform):
        """
        Generate BEV semantic map from camera features
        """
        # 1. Multi-camera feature extraction
        # 2. Spatial cross-attention for BEV mapping
        # 3. BEV feature aggregation
        # 4. Semantic segmentation head
        # 5. Category mapping to DiffusionDrive classes
        pass
```

#### Phase 3: Category Mapping Adaptation

Map Bench2DriveZoo's semantic classes to DiffusionDrive's 7 BEV classes:

```python
BENCH2DRIVE_TO_DIFFUSION_MAPPING = {
    # Drivable areas → Road (1)
    'drivable_pred': 1,
    
    # Lane types → Lane centerlines (3)
    'lane_divider': 3,
    'lane_crossing': 3,
    'lane_contour': 3,
    
    # Vehicle occupancy → Vehicles (5)
    'vehicle_occupancy': 5,
    
    # Background → Background (0)
    'background': 0,
    
    # Static objects → Static objects (4)
    'static_objects': 4,
    
    # Pedestrians → Pedestrians (6)
    'pedestrians': 6,
    
    # Walkways → Walkways (2)
    'walkways': 2,
}
```

### Phase 4: Configuration and Integration

Configure the BEV generation pipeline for DiffusionDrive:

```python
# In navsim/agents/diffusiondrive/transfuser_config.py
@dataclass
class BEVSemanticConfig:
    enable_bev_generation: bool = True
    bev_size: Tuple[int, int] = (256, 128)
    bev_resolution: float = 0.5  # meters per pixel
    num_bev_classes: int = 7
    use_temporal_features: bool = True
    attention_levels: int = 4
    
    # Camera configuration for BEV
    camera_names: List[str] = field(default_factory=lambda: [
        'rgb_front_left', 'rgb_front', 'rgb_front_right'
    ])
    
    # Coordinate transformation
    coordinate_transform_matrix: str = 'coor2topdown'
    
    # Integration with existing pipeline
    replace_placeholder_bev: bool = True
    validate_bev_output: bool = True
```

### Phase 5: Updated Category Mapping

Based on semantic analysis from Bench2Drive base dataset (27 unique values found):

```python
BENCH2DRIVE_TO_BEV_MAPPING = {
    # Background (0)
    0: 0,   # Unlabeled
    3: 0,   # Building
    9: 0,   # Vegetation
    10: 0,  # Terrain
    11: 0,  # Sky
    22: 0,  # Other
    23: 0,  # Water
    26: 0,  # Bridge
    27: 0,  # RailTrack
    
    # Road (1)
    1: 1,   # Roads
    24: 1,  # RoadLine
    25: 1,  # Ground
    
    # Walkways (2)
    2: 2,   # SideWalks
    
    # Lane centerlines (3) - custom detection needed
    24: 3,  # RoadLine (when centerline)
    
    # Static objects (4)
    4: 4,   # Wall
    5: 4,   # Fence
    6: 4,   # Pole
    7: 4,   # TrafficLight
    8: 4,   # TrafficSign
    20: 4,  # Static
    21: 4,  # Dynamic
    28: 4,  # GuardRail
    
    # Vehicles (5)
    14: 5,  # Car
    15: 5,  # Truck
    # 16: 5,  # Bus (missing in dataset)
    # 17: 5,  # Train (missing in dataset)
    18: 5,  # Motorcycle
    19: 5,  # Bicycle
    
    # Pedestrians (6)
    12: 6,  # Pedestrian
    13: 6,  # Rider
}
```

### Phase 6: Integration with DiffusionDrive Pipeline

1. **Update Feature Builder** (`transfuser_features_b2d.py`):
   - Replace placeholder BEV semantic map generation
   - Integrate `BEVSemanticGenerator` into feature pipeline

2. **Update Target Builder**:
   - Use real BEV semantic maps for training targets
   - Validate semantic map quality and coverage

3. **Update Model Configuration**:
   - Enable BEV semantic auxiliary loss (weight: 14.0)
   - Configure BEV head for proper semantic classes

### Phase 7: Validation and Testing

Create validation script: `tools/validate_bev_semantic.py`

```python
def validate_bev_semantic():
    # 1. Load generated BEV semantic map
    # 2. Compare with ground truth (vehicles, pedestrians from annotations)
    # 3. Visualize overlaid with RGB top-down
    # 4. Check consistency with perspective views
    # 5. Compute coverage statistics
```

## Implementation Priority

### For Existing Data (No Re-collection)

1. **High Priority**: Implement Option B (Perspective-to-BEV Projection)
2. **Medium Priority**: Implement Option C (Hybrid approach) as fallback
3. **Low Priority**: Visualization and validation tools

### For New Data Collection

1. **High Priority**: Modify data collection to include semantic top-down
2. **High Priority**: Implement Option A (Direct top-down processing)
3. **Medium Priority**: Category mapping validation

## File Structure

```
Bench2Drive/
├── tools/
│   ├── generate_bev_semantic.py      # New: BEV generation module
│   ├── validate_bev_semantic.py      # New: Validation script
│   └── data_collect.py               # Modified: Add semantic top-down
├── leaderboard/
│   └── envs/
│       └── sensor_interface.py       # May need updates for semantic top-down
└── BEV_SEGMENTATION_GENERATION_PLAN.md  # This file
```

## Testing Approach

1. Start with a single scene from Dev10 validation set
2. Generate BEV using each method
3. Visually compare results
4. Validate against known ground truth (vehicle positions)
5. Test integration with DiffusionDrive

## Expected Challenges

1. **Camera Calibration**: Need accurate intrinsics/extrinsics for projection
2. **Occlusion Handling**: Multiple views may have conflicting semantics
3. **Height Filtering**: Determining ground plane for projection
4. **Performance**: Real-time generation for online evaluation

## Success Metrics

1. BEV semantic maps correctly identify drivable areas (>90% accuracy)
2. Dynamic objects (vehicles/pedestrians) placed accurately (IoU > 0.7)
3. Generation time < 100ms per frame
4. Successful integration with DiffusionDrive (non-zero BEV loss)

## Next Steps (Updated)

1. **Study Bench2DriveZoo implementation details** - understand spatial cross-attention mechanism
2. **Adapt core components for DiffusionDrive** - create `BEVSemanticGenerator` module
3. **Implement semantic category mapping** - handle 27 unique pixel values to 7 BEV classes
4. **Validate with Bench2Drive base dataset** - test on 13,770 semantic images
5. **Integrate with DiffusionDrive training** - replace placeholder BEV maps
6. **Enable BEV auxiliary loss** - improve training effectiveness with proper semantic maps

## Key Technical Insight

**Coordinate Transformation**: Bench2DriveZoo uses projection matrices (`coor2topdown`) to convert 3D world coordinates to BEV pixel coordinates, supporting multiple resolutions (200x200, 512x512, 1024x1024). This solves the perspective-to-BEV projection challenge that was previously a major implementation barrier.

## Expected Impact

- Enable proper BEV semantic auxiliary loss in DiffusionDrive
- Improve autonomous driving model training on Bench2Drive
- Provide foundation for future BEV-based research

## References

- CARLA Semantic Segmentation: <https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera>
- Bench2Drive Dataset: <https://github.com/Thinklab-SJTU/Bench2Drive>
- DiffusionDrive Requirements: /workspace/DiffusionDrive/BENCH2DRIVE_INTEGRATION.md
