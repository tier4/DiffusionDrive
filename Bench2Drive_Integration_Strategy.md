# Bench2Drive Integration Strategy

> **🚨 IMPLEMENTATION STATUS**: **Method 3 (CARLA-Native Pipeline) was chosen and implemented** as documented in `BENCH2DRIVE_INTEGRATION.md`. This document serves as a reference for the evaluation process and alternative approaches.

## Critical Context: Live CARLA Evaluation Requirement

**Important**: This project requires the trained model to control vehicles in real-time within the CARLA simulator. This requirement fundamentally influences the choice between the approaches below.

## Three Integration Methods Based on Your Goals

### Method Selection Guide

| Your Goal | Recommended Method |
| :--- | :--- |
| Train on Bench2Drive, evaluate on CARLA only | **Method 3** (CARLA-Native) ✅ **CHOSEN** |
| Train on Bench2Drive, compare with NavSim models | **Method 2** (Full Adaptation) |
| Train on mixed datasets (Bench2Drive + NavSim) | **Method 2** (Full Adaptation) |
| One-time conversion, no live CARLA needed | **Method 1** (Data Conversion) |

## Method 1: Data Conversion (CARLA → NavSim Format)

One-time conversion of Bench2Drive data to NavSim format. Requires separate pipeline for live CARLA evaluation.

### What You Need to Do

1. **Develop a Conversion Script:** Write a comprehensive script that iterates through every scenario in the Bench2Drive dataset.
2. **Downsample Data:** CARLA data is high-frequency (e.g., 10-20Hz). You must downsample the frames and annotations to match NAVSIM's **2Hz** rate.
3. **Perform Core Transformations:** For every data point in every sampled frame, you must:
    * **Convert Coordinates:** Transform 3D points from CARLA's coordinate system to NAVSIM's coordinate system (Note: exact transformation needs verification - conflicting information exists).
    * **Convert Rotations:** Transform yaw from **degrees (clockwise)** to **radians (counter-clockwise)**.
4. **Remap and Repackage:**
    * Map CARLA camera names to the expected NAVSIM camera slots.
    * Reorganize the data structure to match NavSim's expected format.

### Pros

* Clean separation of concerns

* No model modifications needed
* Can leverage existing NavSim training infrastructure

### Cons

* Massive upfront data processing task

* Requires live CARLA evaluation pipeline anyway
* Potential data loss during conversion
* Storage requirements doubled

## Method 2: Full Model Adaptation

Complete adaptation layer supporting both CARLA and NavSim coordinate systems with configurable transformations.

### Implementation Overview

Create a unified data pipeline that can handle both CARLA and NavSim formats with runtime coordinate transformation.

### Key Components

1. **Coordinate Transform Module**
2. **Data Loader Adaptation**
3. **Feature Builder Updates**
4. **Configuration System**

### Pros

* Supports both datasets

* Can compare with NavSim models
* Flexible for future datasets
* Runtime coordinate transformation

### Cons

* Complex implementation

* Potential for coordinate transformation bugs
* Higher computational overhead
* Requires extensive testing

## Method 3: CARLA-Native Pipeline ✅ **CHOSEN AND IMPLEMENTED**

Simplified adaptation keeping everything in CARLA coordinates, perfect for CARLA-specific projects.

### Implementation Strategy

Keep all data in CARLA coordinate system and adapt only the minimal necessary components:

1. **No Coordinate Transformation**: Keep CARLA's left-handed coordinate system
2. **Simple Data Mapping**: Map CARLA data structures to NavSim format without transformation
3. **Command Simplification**: Convert CARLA commands to discrete values
4. **Minimal Model Changes**: Only adapt data loading and feature extraction

### Pros

* Simplest implementation

* No coordinate transformation errors
* Direct CARLA integration
* Fastest development time

### Cons

* CARLA-specific solution

* Cannot compare with NavSim models
* Not suitable for mixed datasets

## Important Note on Coordinate Transformations

**WARNING**: The coordinate transformation logic between CARLA and NavSim shows conflicting information in different sources. Before implementing Method 2, you must verify the correct transformation by:

1. Testing with known reference points
2. Validating with sample data
3. Cross-referencing multiple sources
4. Implementing extensive unit tests

## Appendix: Example Implementation Details for Method 2 (Not Used)

> **Note**: The following code examples are preserved for reference but were NOT used in the final implementation.

### Coordinate Transformation Functions

```python
def carla_to_navsim_position(carla_pos, ego_transform):
    """
    Convert CARLA position to NavSim coordinate system
    
    CARLA: Left-handed, Z-up, Y-forward
    NavSim: Right-handed, Z-up, X-forward
    """
    # Transform from CARLA to NavSim coordinates
    navsim_x = carla_pos.y  # CARLA Y becomes NavSim X
    navsim_y = -carla_pos.x  # CARLA X becomes NavSim -Y
    navsim_z = carla_pos.z   # Z remains the same
    
    # Apply ego transform (convert to ego-centric coordinates)
    ego_x = ego_transform.location.y
    ego_y = -ego_transform.location.x
    ego_yaw = -math.radians(ego_transform.rotation.yaw)
    
    # Relative position
    rel_x = navsim_x - ego_x
    rel_y = navsim_y - ego_y
    
    # Rotate to ego frame
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)
    
    final_x = rel_x * cos_yaw - rel_y * sin_yaw
    final_y = rel_x * sin_yaw + rel_y * cos_yaw
    
    return (final_x, final_y, navsim_z)

def carla_to_navsim_rotation(carla_rotation):
    """
    Convert CARLA rotation to NavSim format
    
    CARLA: Degrees, clockwise
    NavSim: Radians, counter-clockwise
    """
    # Convert degrees to radians and flip direction
    navsim_yaw = -math.radians(carla_rotation.yaw)
    navsim_pitch = -math.radians(carla_rotation.pitch)
    navsim_roll = -math.radians(carla_rotation.roll)
    
    return (navsim_yaw, navsim_pitch, navsim_roll)

def transform_bounding_box(carla_bbox, ego_transform):
    """
    Transform CARLA bounding box to NavSim format
    """
    # Transform center position
    center_x, center_y, center_z = carla_to_navsim_position(
        carla_bbox.location, ego_transform
    )
    
    # Transform dimensions (no coordinate change needed)
    width = carla_bbox.extent.x * 2
    height = carla_bbox.extent.y * 2
    length = carla_bbox.extent.z * 2
    
    # Transform rotation
    yaw, pitch, roll = carla_to_navsim_rotation(carla_bbox.rotation)
    
    return {
        'center': [center_x, center_y, center_z],
        'dimensions': [width, height, length],
        'rotation': [yaw, pitch, roll]
    }
```

### Data Mapping Examples

```python
def map_carla_to_navsim_cameras(carla_cameras):
    """
    Map CARLA camera names to NavSim expected names
    """
    mapping = {
        'rgb_front': 'CAM_FRONT',
        'rgb_front_left': 'CAM_FRONT_LEFT',
        'rgb_front_right': 'CAM_FRONT_RIGHT',
        'rgb_back': 'CAM_BACK',
        'rgb_back_left': 'CAM_BACK_LEFT',
        'rgb_back_right': 'CAM_BACK_RIGHT'
    }
    
    navsim_cameras = {}
    for carla_name, navsim_name in mapping.items():
        if carla_name in carla_cameras:
            navsim_cameras[navsim_name] = carla_cameras[carla_name]
    
    return navsim_cameras

def convert_carla_command_to_navsim(carla_command):
    """
    Convert CARLA high-level command to NavSim discrete command
    """
    # Based on analysis from Bench2DriveZoo
    command_mapping = {
        1: 0,  # CARLA LEFT -> NavSim LEFT
        2: 2,  # CARLA RIGHT -> NavSim RIGHT  
        3: 1,  # CARLA STRAIGHT -> NavSim STRAIGHT
        4: 1,  # CARLA FOLLOW -> NavSim STRAIGHT
        5: 1,  # CARLA CHANGE_LEFT -> NavSim STRAIGHT
        6: 1,  # CARLA CHANGE_RIGHT -> NavSim STRAIGHT
    }
    
    return command_mapping.get(carla_command, 3)  # Default to UNKNOWN
```

## References

* Final implementation: `BENCH2DRIVE_INTEGRATION.md`
