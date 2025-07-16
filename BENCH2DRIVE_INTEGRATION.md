# Bench2Drive Integration for DiffusionDrive

This document describes the implementation of Bench2Drive dataset support for DiffusionDrive training.

## Implementation Overview

We implemented **Method 3 (CARLA-Native Pipeline)** from the integration strategy, which:

- Keeps everything in CARLA coordinates (no transformation needed)
- Adapts data structures to match NAVSIM format
- Simplifies CARLA commands to discrete values
- Provides seamless integration with the existing training pipeline

## Key Components

### 1. Data Loader (`navsim/common/bench2drive_dataloader.py`)

- `Bench2DriveSceneLoader`: Loads Bench2Drive dataset structure
- Handles temporal downsampling (10Hz → 2Hz)
- Creates sliding windows for scene generation
- Maps CARLA commands to discrete values (0=LEFT, 1=STRAIGHT, 2=RIGHT, 3=UNKNOWN)

### 2. Scene Representation (`navsim/common/bench2drive_scene.py`)

- `Bench2DriveScene`: Represents a sequence of frames
- Loads sensor data (cameras, LiDAR)
- Extracts ego status and agent information
- Generates future trajectories for training

### 3. Feature Builder (`navsim/agents/diffusiondrive/transfuser_features_b2d.py`)

- `Bench2DriveFeatureBuilder`: Processes raw data into model features
  - Camera: Stitches 3 front cameras into [3, 256, 1024] tensor
  - LiDAR: Creates BEV histogram [1, 256, 256]
  - Status: One-hot command + velocity + acceleration [8]
- `Bench2DriveTargetBuilder`: Creates training targets
  - Future trajectory waypoints
  - Agent bounding boxes
  - BEV semantic map

### 4. Dataset Wrapper (`navsim/planning/training/bench2drive_dataset.py`)

- `Bench2DriveDataset`: PyTorch dataset compatible with training pipeline
- Supports caching for efficient training
- Integrates with existing training infrastructure

### 5. Configuration Files

- `config/common/train_test_split/bench2drive.yaml`: Dataset configuration
- `config/common/agent/diffusiondrive_agent_b2d.yaml`: Model configuration

### 6. Testing Framework (`tests/test_bench2drive_integration.py`)

- Pytest-based test suite for integration verification
- Tests command mapping logic
- Validates scene loader functionality
- Checks feature and target building
- Includes visualization generation for debugging
- Gracefully skips tests when data is not available

## Usage

### 1. Setup Environment

```bash
# Set environment variables
export NAVSIM_DEVKIT_ROOT=/workspace/DiffusionDrive
export NAVSIM_EXP_ROOT=/workspace/navsim_workspace/exp
export BENCH2DRIVE_ROOT=/workspace/Bench2Drive-Base
```

### 2. Test Integration

```bash
# Run pytest tests to verify implementation
pytest tests/test_bench2drive_integration.py -v

# Or run all tests
pytest tests/ -v

# Test specific components
pytest tests/test_bench2drive_integration.py::TestBench2DriveIntegration::test_command_mapping -v
```

**What the tests validate:**
- Command mapping from CARLA to NavSim discrete values
- Scene loader creation and dataset indexing
- Scene loading with proper data structures
- Feature extraction (camera stitching, LiDAR BEV, status encoding)
- Target building (trajectories, agents, BEV maps)
- End-to-end integration with visualization output

**Test requirements:**
- Tests will skip gracefully if `/workspace/Bench2Drive-Base/` doesn't exist
- Requires extracted Bench2Drive data for full functionality tests
- Command mapping tests run independently without data requirements

### 3. Cache Dataset

```bash
python navsim/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_agent_b2d \
    train_test_split=bench2drive \
    experiment_name=bench2drive_caching \
    split=train
```

### 4. Train Model

```bash
# Use the provided training script
./scripts/train_bench2drive.sh

# Or run manually
python navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent_b2d \
    train_test_split=bench2drive \
    experiment_name=bench2drive_training \
    split=train \
    trainer.params.max_epochs=100
```

## Key Design Decisions

### 1. Coordinate System

- **No transformation**: Keep CARLA's left-handed coordinate system
- Only convert angles from degrees to radians with sign flip (CW → CCW)
- Simplifies implementation and avoids transformation errors

### 2. Command Mapping

Based on analysis from Bench2DriveZoo, we map CARLA commands as follows:

```python
CARLA Command → Processing → NavSim Command
LEFT (1)      → 0         → LEFT (0)
RIGHT (2)     → 1         → RIGHT (2)
STRAIGHT (3)  → 2         → STRAIGHT (1)
```

### 3. Temporal Downsampling

- Bench2Drive: 10Hz recording
- DiffusionDrive: 2Hz training
- Solution: Take every 5th frame (sampling_rate=5)

### 4. Camera Configuration

- Bench2Drive: 6 cameras
- DiffusionDrive: Uses 3 front cameras
- Selected: `rgb_front_left`, `rgb_front`, `rgb_front_right`
- Stitched horizontally to [3, 256, 1024]

## Data Flow

1. **Raw Data** (Bench2Drive format)

   ```
   scenario/town_weather_route/
   ├── anno/00000.json.gz    # Annotations
   ├── camera/rgb_*/         # Images
   └── lidar/00000.laz      # Point clouds
   ```

2. **Scene Creation**
   - Load frames with sliding window
   - Apply temporal downsampling
   - Group into scenes of 30 frames (15 seconds)

3. **Feature Extraction**
   - Stitch camera images
   - Create LiDAR BEV histogram
   - Extract ego status with discrete command

4. **Training**
   - Standard DiffusionDrive training pipeline
   - Compatible with existing loss functions
   - Uses same model architecture

## Limitations and Future Work

1. **BEV Semantic Map**: Currently returns placeholder. Full implementation would require:
   - Processing semantic segmentation images
   - Projecting to BEV space
   - Integrating HD map information

2. **Caching Performance**: Initial caching can be slow due to:
   - Decompressing tar.gz files
   - Loading LAZ point clouds
   - Consider pre-extracting data for production

3. **Scenario Selection**: Currently uses hardcoded scenario lists. Could be improved with:
   - Dynamic scenario discovery
   - Configuration-based filtering
   - Cross-validation splits

## High Priority TODOs

### 1. **BEV Semantic Map Generation** (Priority: HIGH)

**Problem**: Bench2Drive dataset has `rgb_top_down` views but **no `semantic_top_down`** views. The current implementation uses placeholder BEV semantic maps, which limits the effectiveness of the BEV semantic auxiliary task that DiffusionDrive uses for training (weight: 14.0).

**Available Data**:
- Camera semantic views: `semantic_front`, `semantic_back`, `semantic_left`, `semantic_right`
- Top-down RGB view: `rgb_top_down`
- Instance segmentation: `instance_*` for all camera views
- LiDAR point clouds and bounding box annotations

**Potential Solutions**:
1. **Project perspective semantic views to BEV space** using camera intrinsics/extrinsics
2. **Apply semantic segmentation model to top-down RGB views** to generate semantic labels
3. **Generate basic semantic maps from LiDAR + bounding box data** (vehicles, static objects, road areas)
4. **Use instance segmentation + object annotations** to create semantic categories

**Impact**: This would significantly improve the training effectiveness since BEV semantic segmentation is a key auxiliary task in DiffusionDrive's multi-task learning approach.

### 2. **Semantic Category Mapping** (Priority: MEDIUM)

**Problem**: Need to verify that Bench2Drive's semantic categories map correctly to DiffusionDrive's expected BEV semantic classes. The current mapping strategy needs validation and potential updates.

**Required Actions**:
- Review `becn2drive_category_mapping_strategy.md` for current mapping
- Validate semantic tag mappings against DiffusionDrive's expected classes
- Ensure consistency between perspective semantic views and BEV semantic generation
- Test category swapping issues similar to the command mapping problems

**Reference**: See `becn2drive_category_mapping_strategy.md` for current category definitions (0=Background, 1=Road, 2=Walkways, 3=Lane centerlines, 4=Static objects, 5=Vehicles, 6=Pedestrians).

## Troubleshooting

### No scenes found

- Check if tar.gz files are extracted in `BENCH2DRIVE_ROOT`
- Verify scenario names match directory structure
- Ensure annotation files exist (`.json.gz` in `anno/` folders)

### Memory issues

- Reduce batch size in training config
- Enable caching to avoid repeated data loading
- Limit number of workers in dataloader

### Command mapping confusion

- Remember the coordinate system flip causes LEFT/RIGHT inversion
- Check `COMMAND_MAPPING_ANALYSIS.md` in Bench2DriveZoo for details
- Verify with visualization to ensure correct mapping
