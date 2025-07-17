# BEV Segmentation Generation for Bench2Drive Integration

## Executive Summary

This document provides a detailed analysis of BEV segmentation generation for the Bench2Drive dataset integration with DiffusionDrive. The implementation includes **HD map-based BEV generation** using vectorized map data from CARLA towns, providing rich semantic information including lanes, traffic lights, and stop signs. The system supports both online (runtime) and offline (pre-cached) BEV generation.

## Background

The Bench2Drive dataset, unlike nuScenes, does not provide ground truth BEV segmentation maps. The original Bench2Drive dataset contains:

- Multi-view camera images
- LiDAR point clouds
- 3D bounding boxes for objects
- Vectorized map information from CARLA towns
- Ego vehicle trajectories and planning data

## Dataset Architecture

### Core Dataset Classes

1. **B2D_Dataset** (`/workspace/Bench2DriveZoo/mmcv/datasets/B2D_dataset.py`)
   - Base detection dataset for Bench2Drive
   - Handles basic object detection tasks

2. **B2D_E2E_Dataset** (`/workspace/Bench2DriveZoo/mmcv/datasets/B2D_e2e_dataset.py`)
   - End-to-end dataset for UniAD stage 2 training
   - Integrates detection, tracking, mapping, and planning
   - **Key class for BEV segmentation generation**

3. **B2D_VAD_Dataset** (`/workspace/Bench2DriveZoo/mmcv/datasets/B2D_vad_dataset.py`)
   - VAD-specific dataset implementation

### Data Preprocessing Pipeline

The preprocessing is handled by `/workspace/Bench2DriveZoo/mmcv/datasets/prepare_B2D.py`:

```python
# Key preprocessing outputs
- b2d_infos_train.pkl    # Training sample metadata
- b2d_infos_val.pkl      # Validation sample metadata  
- b2d_map_infos.pkl      # Map information for all towns
```

## BEV Segmentation Generation Process

### 1. Map Information Extraction

The `get_map_info()` method in `B2D_E2E_Dataset` handles map data extraction:

```python
def get_map_info(self, index):
    # Load town-specific map information
    town_name = ann_info['town_name']
    map_info = self.map_infos[town_name]
    
    # Extract vectorized map elements
    lane_points = map_info['lane_points']
    lane_types = map_info['lane_types']
    trigger_volumes_points = map_info['trigger_volumes_points']
    trigger_volumes_types = map_info['trigger_volumes_types']
```

### 2. Coordinate System Transformations

**Critical Challenge**: Bench2Drive uses CARLA's left-handed coordinate system, while UniAD expects right-handed coordinates (nuScenes standard).

```python
# Left-hand to right-hand coordinate transformation
left2right = np.eye(4)
left2right[1,1] = -1

# LiDAR coordinate transformations
lidar_to_righthand_ego = np.array([[ 0, 1, 0, 0],
                                   [-1, 0, 0, 0],
                                   [ 0, 0, 1, 0],
                                   [ 0, 0, 0, 1]])
```

**Transformation Pipeline**:

1. **World → LiDAR**: Transform from world coordinates to LiDAR frame
2. **Coordinate System Conversion**: Left-handed → Right-handed
3. **Range Filtering**: Keep only points within BEV range `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]`
4. **Normalization**: Map to 200×200 BEV grid coordinates

### 3. BEV Rasterization Process

**Map Element Classes**:

```python
self.map_element_class = {
    'Broken': 0,      # Broken lane lines
    'Solid': 1,       # Solid lane lines  
    'SolidSolid': 2,  # Double solid lines
    'Center': 3,      # Center lines
    'TrafficLight': 4, # Traffic light trigger volumes
    'StopSign': 5     # Stop sign trigger volumes
}
```

**Rasterization Methods**:

- **Lane Lines**: `cv2.polylines()` with thickness=2
- **Trigger Volumes**: `cv2.fillConvexPoly()` for filled polygons

**Output Specification**:

- **Resolution**: 200×200 pixels
- **Spatial Coverage**: 102.4m × 102.4m
- **Pixel Resolution**: ~0.512m per pixel

### 4. Runtime Generation Pipeline

```python
# Pseudo-code for BEV generation
def generate_bev_segmentation(self, sample_info):
    # 1. Load town-specific map data
    town_name = sample_info['town_name']
    map_data = self.map_infos[town_name]
    
    # 2. Extract ego pose and transformation matrices
    ego_pose = sample_info['ego2global_transform']
    world2lidar = sample_info['world2lidar_transform']
    
    # 3. Transform map elements to LiDAR coordinates
    transformed_lanes = transform_coordinates(map_data['lane_points'], world2lidar)
    transformed_triggers = transform_coordinates(map_data['trigger_volumes'], world2lidar)
    
    # 4. Filter points within BEV range
    filtered_lanes = filter_bev_range(transformed_lanes)
    filtered_triggers = filter_bev_range(transformed_triggers)
    
    # 5. Rasterize to BEV grid
    bev_mask = np.zeros((200, 200), dtype=np.uint8)
    bev_mask = rasterize_lanes(bev_mask, filtered_lanes)
    bev_mask = rasterize_triggers(bev_mask, filtered_triggers)
    
    return bev_mask
```

## Key Differences from nuScenes Implementation

### Dataset Format Differences

| Aspect | nuScenes | Bench2Drive |
|--------|----------|-------------|
| **Coordinate System** | Right-handed | Left-handed (CARLA) |
| **Frame Rate** | 2Hz | 10Hz |
| **Map Format** | HD Map API | Vectorized CARLA maps |
| **BEV Ground Truth** | None (generated from HD maps) | None (generated from vectors) |
| **Spatial Range** | Variable | Fixed CARLA town layout |

### Processing Differences

1. **Map Source**:
   - **nuScenes**: Uses nuScenes HD Map API with real-world map data
   - **Bench2Drive**: Uses CARLA town vectorized map data

2. **Coordinate Handling**:
   - **nuScenes**: Direct coordinate transformations
   - **Bench2Drive**: Requires left-to-right handed coordinate conversion

3. **Map Elements**:
   - **nuScenes**: Road segments, lane dividers, pedestrian crossings
   - **Bench2Drive**: Lane lines, trigger volumes, traffic elements

## Configuration and Integration

### UniAD Configuration Files

**Stage 1 Configuration** (`configs/stage1_track_map/base_track_map_b2d.py`):

```python
dataset_type = 'B2D_E2E_Dataset'
data_root = 'data/Bench2Drive/'
ann_file = 'data/Bench2Drive/b2d_infos_train.pkl'
map_ann_file = 'data/Bench2Drive/b2d_map_infos.pkl'
```

**Stage 2 Configuration** (`configs/stage2_e2e/base_e2e_b2d.py`):

```python
# BEV parameters
bev_size = (200, 200)
patch_size = (102.4, 102.4)
canvas_size = (200, 200)
map_num_classes = 6  # Including background
```

### Key Configuration Parameters

- **BEV Grid Size**: 200×200 pixels
- **Spatial Coverage**: 102.4m × 102.4m
- **Map Classes**: 6 classes (background + 5 map elements)
- **Thickness**: 2 pixels for lane lines
- **Queue Length**: 3-5 frames for temporal sequences

## Technical Implementation Details

### Memory and Performance Considerations

1. **Runtime Generation**: BEV maps are generated on-the-fly during data loading
2. **Caching**: Map information is cached per town to avoid repeated loading
3. **Vectorization**: Uses NumPy vectorized operations for coordinate transformations
4. **GPU Acceleration**: Rasterization uses OpenCV optimized functions

### Error Handling and Edge Cases

1. **Missing Map Data**: Graceful handling of towns without complete map information
2. **Coordinate Overflow**: Clipping of coordinates outside BEV range
3. **Empty Maps**: Handling of areas with no lane information
4. **Temporal Consistency**: Ensuring consistent BEV generation across temporal sequences

## Validation and Quality Assurance

### Verification Methods

1. **Visual Inspection**: Generated BEV maps are visualized against CARLA town layouts
2. **Coordinate Validation**: Transformation matrices are verified against known landmarks
3. **Consistency Checks**: Temporal sequences maintain spatial consistency
4. **Performance Metrics**: BEV generation speed and memory usage monitoring

### Known Limitations

1. **Map Completeness**: Not all CARLA towns have complete vectorized map data
2. **Coordinate Precision**: Floating-point precision limitations in transformations
3. **Rasterization Artifacts**: Potential aliasing in line rasterization
4. **Performance**: Runtime generation adds computational overhead

## Conclusion

UniAD successfully adapts to the Bench2Drive dataset by implementing a sophisticated BEV segmentation generation pipeline that:

1. **Converts Vectorized Maps**: Transforms CARLA's vectorized map data into pixel-based BEV segmentation
2. **Handles Coordinate Systems**: Properly converts between left-handed and right-handed coordinate systems
3. **Maintains Compatibility**: Preserves UniAD's original architecture while adapting to new data format
4. **Ensures Quality**: Provides consistent and accurate BEV segmentation for training and evaluation

This implementation demonstrates how deep learning models can be adapted to work with different dataset formats while maintaining their core functionality and performance characteristics.

---

## DiffusionDrive Implementation Updates

### Enhanced BEV Generation System

The DiffusionDrive implementation has been enhanced with a comprehensive BEV generation system that matches the quality of the original Bench2Drive approach:

#### 1. **HD Map Integration**

The system now uses HD map data from `/workspace/Bench2Drive-Map/` containing:
- Vectorized lane data (broken, solid, double solid, center lines)
- Trigger volumes (traffic lights, stop signs)
- Per-town map files in NPZ format

#### 2. **Coordinate System Fixes**

Fixed coordinate transformations in `navsim/common/bev_semantic_utils.py`:
```python
# Corrected ego-to-BEV transformation
# Ego vehicle at bottom-center (row=127, col=128)
rows = (bev_height - 1) - points_ego[:, 0] / resolution
cols = bev_width / 2 + points_ego[:, 1] / resolution
```

#### 3. **Map-Based BEV Generation**

New utilities in `navsim/common/bev_map_utils.py`:
- `load_map_data()`: Load Bench2Drive map files
- `generate_bev_from_map()`: Generate BEV from vectorized map
- `generate_full_bev_from_map()`: Generate 360° BEV
- `extract_front_half_bev()`: Extract front view for NavSim

#### 4. **Offline BEV Cache Generation**

Script `scripts/generate_bev_cache.py` for pre-generating BEV maps:
```bash
python3 scripts/generate_bev_cache.py \
    --data-root /workspace/Bench2Drive-mini \
    --map-dir /workspace/Bench2Drive-Map \
    --output-dir data/bev_cache \
    --full-bev \
    --workers 4
```

#### 5. **Semantic Class Mapping**

Bench2Drive to NavSim BEV class mapping:
```python
LANE_TYPE_TO_BEV_CLASS = {
    'Broken': 1,      # Road
    'Solid': 1,       # Road  
    'SolidSolid': 1,  # Road
    'Center': 3,      # Lane centerline
}

TRIGGER_TYPE_TO_BEV_CLASS = {
    'TrafficLight': 4,  # Static object
    'StopSign': 4,      # Static object
}
```

### Key Improvements

1. **Accuracy**: BEV maps now accurately reflect road geometry from HD maps
2. **Rich Semantics**: Includes lane markings and traffic elements
3. **Performance**: Supports offline pre-generation for faster training
4. **Flexibility**: Can generate full 360° or front-only views
5. **Integration**: Seamlessly integrates with existing DiffusionDrive pipeline

### Usage Notes

- **Full BEV**: 256×256 pixels covering 64×64m (360° view)
- **Front BEV**: 128×256 pixels covering 32×64m (NavSim format)
- **Resolution**: 0.25m per pixel
- **Cache Structure**: `cache_dir/scenario_name/frame_number.npz`

**Document Version**: 2.0  
**Last Updated**: 2025-07-18  
**Author**: DiffusionDrive Development Team
