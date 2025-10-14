# Bench2Drive Integration for DiffusionDrive

This document describes the integration of the Bench2Drive dataset for training the DiffusionDrive model. The implementation uses a CARLA-native pipeline, ensuring simplicity and direct compatibility for live evaluation within the CARLA simulator.

-----

## 1\. Overview

The core of this integration is the **CARLA-Native Pipeline** strategy, which was chosen for its simplicity and robustness. This approach keeps all data, including sensor readings and BEV maps, in **CARLA's native left-handed coordinate system**. This avoids complex transformation errors and allows the trained model to be directly evaluated in the CARLA environment.

The implementation provides a complete data processing pipeline that relies on a mandatory caching step to prepare data for training. This includes an HD map-based system for generating the Bird's Eye View (BEV) semantic maps.

-----

## 2\. Key Components

The integration is modular, consisting of several key Python scripts and configuration files.

* **Data Loader** (`navsim/common/bench2drive_dataloader.py`): Loads the Bench2Drive dataset structure, handles temporal downsampling from 10Hz to 2Hz, and creates sliding windows to generate scenes.
* **Scene Representation** (`navsim/common/bench2drive_scene.py`): Represents a sequence of frames, loads sensor data, and extracts ego status and agent information.
* **Feature Builder** (`navsim/agents/diffusiondrive/transfuser_features_b2d.py`): Processes raw data from the scene into feature tensors for the model (camera, LiDAR, status).
* **Dataset Wrapper** (`navsim/planning/training/bench2drive_dataset.py`): A PyTorch-compatible dataset class that wraps the data loader and feature builder, which relies on a pre-cached dataset for training.
* **BEV Static Map Generation** (`navsim/common/bev_map_utils.py`): A utility module to generate the static BEV semantic map layer (lanes, stop lines) from vectorized CARLA HD map data.
* **BEV Dynamic Object Rendering** (`navsim/common/bev_semantic_utils.py`): Renders dynamic objects, including the ego vehicle and other agents, onto the BEV map.
* **Configuration Files**: The training/validation scenario split is defined in `navsim/planning/script/config/common/train_test_split/bench2drive_base_train_val_split.json`. Agent-specific parameters are located in `config/diffusiondrive_agent_b2d.yaml`.
* **Testing Framework** (`tests/test_bench2drive_integration.py`): A Pytest suite to verify the entire integration, from data loading to feature and target building.

-----

## 3\. Data Processing and Feature Extraction

The pipeline processes raw Bench2Drive data and converts it into features and targets suitable for training the DiffusionDrive model. This is done as a pre-processing step via caching.

### 3.1 Data Flow

1. **Raw Data**: The data loader accesses the raw Bench2Drive data, including camera images, LiDAR point clouds, and annotation files.
2. **Scene Creation**: The `Bench2DriveSceneLoader` creates scenes by applying a sliding window over the raw data, downsampling it from 10Hz to 2Hz.
3. **Feature Extraction**: The `Bench2DriveFeatureBuilder` processes each scene to generate model inputs:
      * **Camera**: Stitches the three front cameras (`rgb_front_left`, `rgb_front`, `rgb_front_right`) into a single `[3, 256, 1024]` tensor.
      * **LiDAR**: Creates a BEV histogram of size `[1, 256, 256]`.
      * **Status**: Encodes the vehicle's state into an `[8]` tensor containing a one-hot command, velocity, and acceleration.
4. **Training**: The cached features and targets are fed into the standard DiffusionDrive training pipeline.

### 3.2 Command Mapping

CARLA high-level commands are mapped to discrete integer values to match the model's expected input.

| CARLA Command       | Processing | NavSim Command |
| :------------------ | :--------- | :------------- |
| `VOID` (-1)         | → 3        | `UNKNOWN` (3)  |
| `LEFT` (1)          | → 0        | `LEFT` (0)     |
| `RIGHT` (2)         | → 2        | `RIGHT` (2)    |
| `STRAIGHT` (3)      | → 1        | `STRAIGHT` (1) |
| `LANEFOLLOW` (4)    | → 1        | `STRAIGHT` (1) |
| `CHANGELANELEFT` (5)  | → 0        | `LEFT` (0)     |
| `CHANGELANERIGHT` (6) | → 2        | `RIGHT` (2)    |

### 3.3 LiDAR and BEV Processing

* **LiDAR Coverage Adaptation**: The native LiDAR data from Bench2Drive covers an 85m x 85m area. To match DiffusionDrive's expected input, this data is processed at its full resolution and then resized to a 64m x 64m coverage (256x256 pixels) using LANCZOS downsampling to preserve maximum detail.
* **BEV Dimensions**: The system generates a full 360° BEV map from the map data at a 256x256 resolution (covering 64x64m). This full BEV is then cropped to a **128x256 (H x W) frontal view**. This rectangular shape is the native format for BEV semantic maps in NavSim, designed to align with the coverage of the frontal cameras.

### 3.4 Key Differences from NAVSIM Implementation

The Bench2Drive integration includes several important architectural differences from the original NAVSIM implementation:

#### Agent Filtering Strategy
* **NAVSIM**: Uses rectangular/square filtering with bounds of -32m to +32m in both X and Y axes, creating a 64m × 64m square region centered on the ego vehicle. This naturally emphasizes frontal and side views.
* **Bench2Drive**: Uses circular/radial filtering with a 42.5m radius (half of the 85m LiDAR diameter), providing uniform 360° coverage around the ego vehicle. This approach better handles complex driving scenarios that require full situational awareness.

#### Practical Impact
The circular filtering in B2D ensures consistent detection range in all directions, which is particularly important for:
* Intersection scenarios where threats can approach from any angle
* Highway merging situations requiring awareness of vehicles approaching from behind
* Complex urban environments with pedestrians and vehicles in all directions

This design choice aligns with Bench2Drive's focus on diverse, challenging driving scenarios that go beyond highway-style forward driving.

-----

## 4\. BEV Map Generation

Because the Bench2Drive dataset does not include ground-truth BEV semantic maps, a system was implemented to generate them. The process is split into generating a static map layer from HD map data and rendering dynamic objects on top. This is done offline when generating the cache.

### 4.1 Static Map Layer Generation

The static background of the BEV is handled by utilities in `navsim/common/bev_map_utils.py`. The system leverages pre-processed `.npz` files for each CARLA town, which contain vectorized data for lane markings and trigger volumes (e.g., traffic lights, stop signs). These map elements are rasterized onto a blank BEV grid. To maintain consistency with the CARLA-native pipeline, all map coordinates are processed directly in the ego-vehicle frame without any coordinate system conversions.

### 4.2 Dynamic Object Layer Rendering

Dynamic objects, including the ego-vehicle and other agents like vehicles and pedestrians, are rendered on top of the static map layer. This process is handled by `navsim/common/bev_semantic_utils.py`, which programmatically draws agents onto the BEV grid based on their state vectors.

The core logic, primarily within the `generate_agent_mask` and `generate_simple_bev_semantic` functions, executes the following steps for each dynamic agent:

1. **State Retrieval**: The agent's position ($x, y$), heading ($\theta$), and physical dimensions (length, width) are retrieved from the dataset annotations.
2. **Coordinate Conversion**: The agent's center point, which is in the ego-vehicle's coordinate frame, is transformed into pixel coordinates ($row, col$) on the BEV grid. Agents outside the BEV's view are ignored.
3. **Rasterization**: A filled, rotated rectangle corresponding to the agent's size and orientation is drawn onto the map. This step uses the `draw_rotated_box` utility to accurately represent the agent's footprint.

Finally, the pixels covered by the rendered agents are assigned a specific integer class value based on the agent's type.

#### Dynamic Object Class Mapping

The system uses the following mapping to assign class values for dynamic objects rendered on the BEV map.

| Agent Type | BEV Class Value |
| :--- | :---: |
| Vehicle | 5 |
| Pedestrian | 6 |

### 4.3 Static Map Semantic Class Mapping

CARLA map elements are mapped to the semantic classes expected by DiffusionDrive's BEV loss function:

```python
# Bench2Drive to NavSim BEV class mapping for static elements
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

-----

## 5\. Usage

### 5.1 Setup Environment

Set the following environment variables to point to your project and data directories:

```bash
export NAVSIM_DEVKIT_ROOT=/workspace/DiffusionDrive
export NAVSIM_EXP_ROOT=/workspace/navsim_workspace/exp
```

### 5.2 Test Integration

Run the Pytest suite to verify that all components are working correctly. The tests will validate command mapping, scene loading, and feature/target building. Tests will skip gracefully if the `BENCH2DRIVE_ROOT` directory does not exist.

```bash
pytest tests/test_bench2drive_integration.py -v
```

### 5.3 Cache Dataset

**This is a mandatory prerequisite for training the model.** On-the-fly data processing during training is not supported.

**1. Generate BEV Map Cache (Optional but Recommended)**
This script pre-generates all BEV semantic maps from the HD map data and saves them to disk to speed up the main feature caching step.

```bash
python3 scripts/generate_bev_cache.py \
    --data-root /workspace/Bench2Drive-Base \
    --map-dir /workspace/Bench2Drive-Map \
    --output-dir data/bev_cache \
    --full-bev \
    --workers 4
```

**2. Cache Model Features**
This script processes the raw data and caches the resulting feature tensors required for training.

```bash
python3 navsim/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_agent_b2d \
    train_test_split=bench2drive \
    experiment_name=bench2drive_caching \
    split=train
```

### 5.4 Train Model

**Important:** Training on Bench2Drive **must** be done using a pre-computed cache. The on-the-fly data loading pipeline is no longer supported for training.

For detailed training and evaluation commands, see **[Bench2Drive Training & Evaluation Guide](train_eval.md)**.

Quick example:

```bash
# Simple training
python3 navsim/planning/script/run_bench2drive_training.py \
    agent=diffusiondrive_agent_b2d \
    experiment_name=my_b2d_experiment
```

Note: Since training relies entirely on the cache, the `BENCH2DRIVE_ROOT` environment variable is not used during the training process.

-----

## 6\. Limitations and Future Work

* **Caching Performance**: The mandatory caching process can be slow, especially for large datasets and BEV generation.
* **BEV Map Detail**:
  * **TODO**: Enhance the static BEV maps by generating more detailed objects from CARLA's map data (e.g., pedestrian crossings, specific lane markings).
  * **TODO**: Enhance the dynamic object layer by projecting 3D bounding box annotations of non-vehicle agents (e.g., pedestrians) onto the BEV map.
