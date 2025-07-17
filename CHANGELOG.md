# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Note: Changes marked as "Community Contribution" were contributed by community members and are not from the original DiffusionDrive authors.**

## [Unreleased] - 2025-07-18 - Community Contribution

### Added - HD Map-Based BEV Generation

#### Enhanced BEV Generation with Bench2Drive HD Maps

- **HD Map Integration**: Implemented comprehensive BEV generation using vectorized HD map data
  - `navsim/common/bev_map_utils.py` - New utilities for map-based BEV generation
  - Supports vectorized lane data (broken, solid, center lines)
  - Includes traffic elements (traffic lights, stop signs) from trigger volumes
  - Matches original Bench2Drive quality with rich semantic information

- **Coordinate System Fixes**:
  - Fixed ego-to-BEV coordinate transformation in `bev_semantic_utils.py`
  - Ego vehicle now correctly positioned at bottom-center (row=127, col=128)
  - Proper handling of CARLA left-handed to right-handed coordinate conversion

- **Offline BEV Cache Generation**:
  - New script `scripts/generate_bev_cache.py` for pre-generating BEV maps
  - Supports full 360° BEV (256×256) and front-only NavSim format (128×256)
  - Multi-process generation for performance
  - Cache structure: `cache_dir/scenario_name/frame_number.npz`

- **Test-Driven Implementation**:
  - Comprehensive test suite in `tests/test_bev_generation_with_maps.py`
  - 17 unit tests covering map loading, coordinate transforms, and generation
  - Tests for both online and offline BEV generation approaches

#### Updated Visualization and Documentation

- **Visualization Improvements**:
  - Updated `notebooks/bench2drive_cache_visualization.ipynb` with HD map visualization
  - Side-by-side comparison of simple vs HD map-based BEV
  - Visual validation of lane markings and traffic elements

- **Documentation Updates**:
  - Enhanced `BEV_SEGMENTATION_ANALYSIS.md` with HD map implementation details
  - Added semantic class mapping documentation
  - Usage instructions for offline BEV generation

## [Unreleased] - 2025-07-17 - Community Contribution

### Added - BEV Semantic Map Generation

#### Dataset-Level BEV Generation (Correct Approach)

- **BEV Semantic Map Generation**: Implemented real BEV semantic maps at the dataset level
  - `navsim/common/bev_semantic_utils.py` - Utilities for BEV map generation
  - Updated `get_bev_semantic_map()` in `bench2drive_scene.py` to generate real maps
  - Replaces placeholder zeros with actual semantic data
  - Enables BEV auxiliary loss (weight: 14.0) to provide meaningful training signal

#### Implementation Details

- **Simple BEV Generation**:
  - Generates road mask from ego trajectory (4m width)
  - Places detected vehicles on the map using rotated bounding boxes
  - Uses native NavSim format (128×256 pixels, 0.25m/pixel resolution)
  - 7 semantic classes: background, road, walkway, lane, static, vehicle, pedestrian

- **Test-Driven Development**:
  - Comprehensive test suite in `tests/test_bev_generation_dataset.py`
  - Integration tests in `tests/test_bev_integration_bench2drive.py`
  - 15 unit tests covering all BEV generation functions
  - Validates coordinate transformations, rasterization, and integration

- **Key Design Decisions**:
  - No model modifications - all changes at dataset/dataloader level
  - Geometric projection instead of neural network components
  - Runtime generation during data loading
  - Fallback approach when vectorized map data not available

#### Documentation

- **Analysis Documents**:
  - `BEV_SEGMENTATION_ANALYSIS.md` - Correct approach documentation
  - Updated `BEV_SEGMENTATION_GENERATION_PLAN.md` with critical warnings
  - Archived incorrect model-based approach in branch `incorrect/bev-model-based-implementation`

### Fixed

- **Agent Detection**: Fixed get_agents() to handle list-format bounding boxes in Bench2Drive mini dataset
- **Trajectory Loading**: Fixed annotation loading in get_bev_semantic_map()

## [Unreleased] - 2025-07-16 - Community Contribution

### Added - Bench2Drive Dataset Integration

#### Core Integration Components

- **Bench2Drive Dataset Support**: Complete integration of Bench2Drive dataset for training
  - `navsim/common/bench2drive_dataloader.py` - Scene loader implementing Method 3 (CARLA-native pipeline)
  - `navsim/common/bench2drive_scene.py` - Scene representation maintaining CARLA coordinates
  - `navsim/agents/diffusiondrive/transfuser_features_b2d.py` - Feature/target builders for Bench2Drive data
  - Temporal downsampling from 10Hz to 2Hz for training compatibility
  - Command mapping from CARLA navigation commands to discrete values
  - Camera stitching (3 front cameras: left, front, right) to 1024x256 resolution
  - LiDAR BEV histogram generation (256x256) from LAZ point clouds

#### Training Pipeline

- **Custom Training Scripts**:
  - `navsim/planning/script/run_bench2drive_caching.py` - Custom caching for Bench2Drive structure
  - `navsim/planning/script/run_bench2drive_training.py` - Training script with Bench2Drive dataset
  - `navsim/planning/training/bench2drive_dataset.py` - Dataset wrapper for PyTorch compatibility
  - `scripts/train_bench2drive.sh` - Shell script for end-to-end training

#### Configuration

- **Hydra Configuration Files**:
  - `navsim/planning/script/config/common/train_test_split/bench2drive.yaml` - Dataset configuration
  - `navsim/planning/script/config/common/agent/diffusiondrive_agent_b2d.yaml` - Agent configuration
  - Support for scenario-based train/val/test splits
  - Configurable sampling rates and frame counts

#### Testing and Validation

- **Comprehensive Test Suite**:
  - `tests/test_bench2drive_integration.py` - Integration tests for full pipeline
  - `tests/test_bench2drive_minimal.py` - Minimal pytest tests with fixtures
  - Scene loading, feature extraction, and target generation validation
  - Tensor shape and value validation for all outputs

#### Documentation and Analysis

- **Documentation**:
  - `BENCH2DRIVE_INTEGRATION.md` - Complete integration documentation
  - `becn2drive_category_mapping_strategy.md` - Category mapping strategy
  - `docs/bench2drive_visualization_testing.md` - Visualization testing procedures
- **Analysis Tools**:
  - `notebooks/bench2drive_cache_visualization.ipynb` - Data visualization and validation notebook

#### Dependencies and Environment

- **New Dependencies**:
  - Added `laspy` for LAZ LiDAR file processing
  - Updated Docker configuration and requirements.txt
- **Testing Framework**:
  - Migrated to pytest with proper fixtures and assertions
  - Comprehensive test coverage for all integration components

### Changed

#### Development Guidelines

- **CLAUDE.md Updates**:
  - Added testing guidelines: Use pytest, put tests under ./tests folder
  - Added code style guidelines: Module imports at top, use black formatter
  - Updated all Python command references from `python` to `python3`

#### Code Quality Improvements

- **Import Organization**: Standardized import ordering across all files
- **Error Handling**: Added comprehensive error handling for missing/invalid data
- **Type Hints**: Improved type hints and documentation throughout codebase

### Technical Details

#### Dataset Structure Handling

- **CARLA-Native Approach**: Maintains original CARLA coordinate system without transformation
- **Flexible Scenario Loading**: Supports both full scenario paths and pattern matching
- **Robust Data Validation**: Handles missing cameras, empty LiDAR data, and invalid annotations
- **Memory Efficient**: Caches annotations and sensor data to minimize I/O

#### Feature Processing

- **Camera Processing**:
  - Horizontal stitching of 3 front cameras (left, front, right)
  - Efficient single-resize approach instead of per-camera resizing
  - Proper handling of missing or invalid camera data
- **LiDAR Processing**:
  - BEV histogram generation from LAZ point clouds
  - Configurable range and resolution parameters
  - Intensity normalization and clipping
- **Status Features**:
  - One-hot encoding of driving commands (4 classes)
  - Velocity and acceleration extraction from annotations
  - Proper tensor type handling for PyTorch compatibility

#### Training Compatibility

- **Multi-Task Learning**: Supports trajectory prediction, agent detection, and BEV segmentation
- **Batch Processing**: Efficient batching for training with proper tensor shapes
- **Temporal Consistency**: Maintains temporal relationships in downsampled data

## [Unreleased] - 2025-07-08 - Community Contribution

### Docker Updates

#### PyTorch and Dependencies Update

- **PyTorch**: 2.4.1 → 2.7.1
  - Significant performance improvements
  - Better GPU memory management
  - Enhanced CUDA 12.x support
  - **Important**: Fixed compatibility issue with WarmupCosLR scheduler
  
- **TorchVision**: 0.19.1 → 0.22.1
  - Aligned with PyTorch 2.7.1
  - New transforms and models support
  
- **PyTorch Lightning**: 1.9.4 → 2.4.0
  - Breaking API changes - check your training scripts
  - Improved distributed training support
  - Better checkpoint handling

#### Python Version Constraints

- Changed from Python 3.9 to Python >=3.9,<=3.10
- Docker image uses Python 3.10 as default

#### Security Updates

- **Pillow**: Updated to >=10.4.0 (security fixes)
- **notebook**: Updated to >=7.3.2 (security fixes)
- **tornado**: Updated to >=6.4.2 (security fixes)
- **setuptools**: Updated to 78.1.1 (security fixes)

### Docker Configuration Changes

#### Genericization for Public Release

- Removed internal/proprietary configurations
- Made build arguments configurable
- Added environment variable support in run script

#### New Features

- Integrated requirements.txt directly into Dockerfile
  - Better Docker layer caching
  - Reduced build times for dependency updates
- Added diffusers and einops packages for diffusion model support
- Improved user/group ID handling

#### Build Script Updates

- Added explicit build arguments
- Better error messages
- Support for custom CUDA versions (minimum 11.8 for H100)

#### Run Script Updates

- Added environment variable support:
  - `WORKSPACE_DIR`: Custom workspace directory
  - `DATA_DIR`: Custom data directory
  - `CONTAINER_NAME`: Custom container name
- Improved documentation in script comments

### Compatibility Fixes

#### WarmupCosLR Scheduler

- Fixed compatibility with PyTorch 2.7.1
- The scheduler now properly handles the new PyTorch optimizer API

### Installation Process Changes

- Requirements are now installed during Docker build
- Only navsim package needs to be installed after container start
- Reduced setup time inside container

## Migration Guide

### From Previous Docker Version

1. **Rebuild the image** - Required due to PyTorch version change:

   ```bash
   cd docker
   ./build.sh
   ```

2. **Update your code** for PyTorch 2.7.1:
   - Check for deprecated APIs
   - Update any custom schedulers/optimizers
   - Review PyTorch Lightning code for API changes

3. **Modify build/run scripts** for your environment:
   - Set correct user/group IDs in build.sh
   - Set correct data directory in run.sh

### Known Issues and Workarounds

1. **PyTorch Lightning API Changes**:
   - Some callbacks may need updates
   - Trainer arguments might have changed

2. **GPU Memory**:
   - PyTorch 2.7.1 may have different memory patterns
   - Monitor and adjust batch sizes if needed

## Rollback Instructions

If you need to rollback to previous versions:

1. Checkout previous commit:

   ```bash
   git checkout <previous-commit-hash>
   ```

2. Rebuild Docker image with old configuration

3. Note: You may need to downgrade PyTorch in your local environment as well

## [Unreleased] - 2025-06-26 - Community Contribution

### Added

- Organized script structure following software engineering best practices
- Reusable shell script components in `scripts/utils/common.sh`
- Parameterized training script with CLI arguments
- Batch experiment runner for automated hyperparameter sweeps
- Comprehensive evaluation scripts with automatic checkpoint discovery
- Script documentation in `scripts/README.md`
- Comprehensive test suite for all shell scripts in `tests/scripts/`
  - Test framework with assertions and utilities
  - Unit tests for training and evaluation scripts
  - Integration tests for script workflows
  - 2-GPU constraint enforcement in all tests
  - Mock environment for isolated testing
  - Example GitHub Actions workflow for CI/CD

### Changed

- **BREAKING**: Reorganized all scripts into structured directories:
  - Training scripts moved to `scripts/training/`
  - Evaluation scripts moved to `scripts/evaluation/`
  - Shared utilities in `scripts/utils/`
- Replaced date-based script naming with parameterized scripts
- Unified logging and error handling across all scripts

### Deprecated

- Date-based script naming convention (e.g., `train_bs32_20250617.sh`)
- Hardcoded parameters in individual script files

### Removed

- Root directory script clutter (8 training scripts)
- Duplicate code across multiple scripts

### Fixed

- Carriage return errors in shell scripts for Ubuntu compatibility

### Reorganized

- Moved notebooks from root directory to `notebooks/evaluation/`
  - `compare_eval.ipynb` - Model comparison and analysis
  - `visualization_eval.ipynb` - Evaluation visualization utilities

### Migration Guide

Old scripts have been moved to `archive/legacy/`. To migrate:

**Before:**

```bash
./train_bs32_20250617.sh
```

**After:**

```bash
./scripts/training/train.sh --name my_experiment --batch-size 32 --epochs 1000
```

## [0.1.0] - 2025-06-09

### Added

- Initial release of DiffusionDrive
- TransFuser backbone with diffusion-based trajectory planning
- NAVSIM integration for autonomous driving simulation
- Basic training and evaluation scripts
