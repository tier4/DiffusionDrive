# HD Map-Based BEV Generation for Bench2Drive

## Overview

This document describes the enhanced BEV semantic map generation system that uses HD map data from Bench2Drive. The implementation provides high-quality BEV maps with accurate road geometry, lane markings, and traffic elements.

## Features

### 1. HD Map Integration

- Uses vectorized map data from `/workspace/Bench2Drive-Map/`
- Supports all CARLA towns (Town01-Town15)
- Includes lane lines (broken, solid, double solid, center)
- Includes traffic elements (traffic lights, stop signs)

### 2. BEV Generation Modes

#### Online Generation (Runtime)

- Generates BEV maps during data loading
- Automatically loads appropriate town map
- Transforms map elements to ego-centric coordinates
- Rasterizes lanes and traffic elements

#### Offline Generation (Pre-cached)

- Pre-generate BEV maps using `scripts/generate_bev_cache.py`
- Supports parallel processing for efficiency
- Stores as compressed NPZ files
- Includes both full (360°) and front-only views

### 3. Coordinate System Handling

- Properly converts CARLA left-handed to right-handed coordinates
- Fixed ego-to-BEV transformation (ego at bottom-center)
- Accurate world-to-ego transformations using annotation data

## Usage

### Basic Usage (Online Generation)

```python
from navsim.common.bench2drive_dataloader import Bench2DriveConfig, Bench2DriveSceneLoader

config = Bench2DriveConfig(
    data_root=Path("/workspace/Bench2Drive-mini"),
    scenarios=["ConstructionObstacle_Town05_Route68_Weather8"],
    map_dir=Path("/workspace/Bench2Drive-Map"),  # HD map directory
)

loader = Bench2DriveSceneLoader(config)
scene = loader.get_scene(loader.scene_tokens[0])

# BEV will be generated from HD map automatically
bev_map = scene.get_bev_semantic_map(frame_idx=0)
```

### Offline Cache Generation

```bash
# Generate BEV cache for all scenarios
python3 scripts/generate_bev_cache.py \
    --data-root /workspace/Bench2Drive-mini \
    --map-dir /workspace/Bench2Drive-Map \
    --output-dir data/bev_cache \
    --full-bev \
    --workers 4

# Generate for specific scenarios
python3 scripts/generate_bev_cache.py \
    --scenarios "ConstructionObstacle_Town05_Route68_Weather8" \
    --map-dir /workspace/Bench2Drive-Map \
    --output-dir data/bev_cache \
    --max-frames 100
```

### Using Cached BEV

```python
config = Bench2DriveConfig(
    data_root=Path("/workspace/Bench2Drive-mini"),
    scenarios=["ConstructionObstacle_Town05_Route68_Weather8"],
    bev_cache_dir=Path("data/bev_cache"),  # Use pre-generated cache
)

# BEV will be loaded from cache automatically
```

## Technical Details

### BEV Specifications

- **Resolution**: 0.25 meters per pixel
- **Front BEV**: 128×256 pixels (32×64 meters) - NavSim format
- **Full BEV**: 256×256 pixels (64×64 meters) - 360° coverage
- **Semantic Classes**:
  - 0: Background
  - 1: Road
  - 2: Walkways (not used in current implementation)
  - 3: Lane centerlines
  - 4: Static objects (traffic lights, stop signs)
  - 5: Vehicles (from dynamic object detection)
  - 6: Pedestrians (not used in current implementation)

### Map Element Processing

1. **Lane Lines**:
   - Rendered using `cv2.polylines()` with configurable thickness
   - Broken, Solid, SolidSolid → Class 1 (Road)
   - Center → Class 3 (Lane centerline)

2. **Trigger Volumes**:
   - Rendered using `cv2.fillPoly()` for filled regions
   - TrafficLight, StopSign → Class 4 (Static objects)

3. **Dynamic Objects**:
   - Vehicles added from agent detection
   - Rendered as rotated rectangles
   - Override static map elements

### Performance Considerations

- Online generation adds ~0.5-1.0s per scene
- Offline cache reduces loading time to ~0.01s per frame
- Full BEV generation recommended for training stability
- Cache size: ~1MB per scenario (compressed)

## Testing

Run comprehensive tests:

```bash
# Test BEV generation with maps
python3 -m pytest tests/test_bev_generation_with_maps.py -v

# Test coordinate transformations
python3 -m pytest tests/test_bev_generation_dataset.py::TestBEVSemanticMapping -v

# Integration tests
python3 -m pytest tests/test_bev_integration_bench2drive.py -v
```

## Visualization

Use the updated notebook for visualization:

```bash
jupyter notebook notebooks/bench2drive_cache_visualization.ipynb
```

The notebook includes:

- HD map BEV visualization
- Comparison with simple trajectory-based BEV
- Full vs front BEV views
- Class distribution analysis

## Key Improvements

1. **Accuracy**: BEV maps now accurately reflect road geometry from HD maps
2. **Rich Semantics**: Includes lane markings and traffic elements
3. **Performance**: Supports offline pre-generation for faster training
4. **Flexibility**: Can generate full 360° or front-only views
5. **Integration**: Seamlessly integrates with existing DiffusionDrive pipeline

## Notes

- HD maps are required for best quality BEV generation
- Falls back to trajectory-based generation if maps unavailable
- Cache invalidation: Delete cache directory to regenerate
- Memory usage: ~500MB for all town maps in memory

## Future Enhancements

- Add walkway/sidewalk detection from map data
- Include more detailed lane type differentiation
- Support for dynamic map updates
- Integration with real-time map services
