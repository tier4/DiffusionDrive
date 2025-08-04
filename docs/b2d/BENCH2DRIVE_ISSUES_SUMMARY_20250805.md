# Bench2Drive Integration Issues Summary

## Overview

This document summarizes the key issues discovered during Bench2Drive dataset integration with DiffusionDrive.

## 1. BEV Map Vehicle Rendering Issue

### Problem

- No vehicles appear in BEV visualizations when using Base dataset
- Only static map elements (roads, lanes) are visible

### Root Cause

- BEV cache generation (`generate_bev_cache.py`) only creates static HD maps
- Vehicles are added dynamically during training by `get_bev_semantic_map()`
- Visualization was incorrectly showing static cache instead of complete BEV with vehicles

### Solution

- Modified `visualize_bench2drive_cached.py` to always use `targets["bev_semantic_map"]` which includes vehicles

## 2. Critical Coordinate System Issues

### Problems

1. **Ego Position Mismatch**:
   - Documentation says ego at bottom, but actually at center
   - This affects trajectory calculation, agent positioning, and BEV generation
   - Not just a visualization issue - impacts training data alignment

2. **Confirmed LiDAR vs BEV Coordinate Mismatch**:
   - LiDAR uses `np.add.at(hist_full, (y, x), 1)` - row-major indexing
   - BEV drawing uses OpenCV which expects (x, y) - column-major
   - Code has to swap coordinates: `[[center[1], center[0]]]` to fix this
   - This inconsistency affects alignment between LiDAR, BEV, trajectories, and agents

3. **Agent Bounding Box Misalignment**:
   - Agents appear outside roads in BEV map
   - Even ego vehicle is shown outside the road
   - Indicates coordinate transformation error between agent states and BEV

### Root Causes

1. **Documentation Bug**: Comments in `ego_to_bev_coordinates()` incorrectly state ego is at bottom
2. **Implementation Inconsistency**: Different components may use different coordinate conventions
3. **No Unified Coordinate System**: Each component (LiDAR, BEV, agents) handles coordinates differently

### Critical Requirements for Training Success

For training to work, ALL components must use consistent coordinate systems:

- **LiDAR BEV histogram** (currently row-major: y,x)
- **BEV semantic map** (mixed: internal row-major, OpenCV column-major)
- **Trajectory waypoints** (ego-centric x,y coordinates)
- **Agent states** (positions, headings must align with BEV)

### TODO

- [x] Verify LiDAR histogram indexing - CONFIRMED: uses (y,x) row-major
- [x] Check if BEV map uses same indexing - CONFIRMED: inconsistent, swaps for OpenCV
- [ ] Fix coordinate system to be consistent across all components
- [ ] Ensure trajectory points align with BEV representation
- [ ] Fix agent-to-BEV coordinate transformation
- [ ] Ensure ego vehicle aligns with road in BEV
- [ ] Update all documentation to reflect actual implementation

## 3. BEV Map Generation Architecture Issues

### Problems

1. **Static-Only Map Cache**:
   - Current design generates static maps during cache creation
   - Dynamic objects (vehicles, pedestrians) added later during training
   - This separation causes potential misalignment issues

2. **Missing Camera-Based Semantic**:
   - BEV semantic generation returns placeholder data
   - Not utilizing semantic segmentation from cameras
   - Limits model's understanding of scene

### Current Architecture Flaws

- Cache generation (`generate_bev_cache.py`) only creates static HD maps
- Dynamic object rendering happens separately in `get_bev_semantic_map()`
- Two-stage process increases chance of coordinate mismatches

### Proposed Solution

- Generate complete BEV maps (static + dynamic) during training, not caching
- Ensure all elements use consistent coordinate system
- Implement proper camera-based semantic segmentation

### TODO

- [ ] Redesign BEV generation to be single-stage (not split cache/runtime)
- [ ] Implement camera semantic segmentation from Bench2DriveZoo
- [ ] Ensure dynamic objects align with static map elements

## 4. Data Processing Pipeline

### Verified Working

- Agent detection and tracking ✓
- Trajectory generation ✓
- Command mapping (LEFT/RIGHT/STRAIGHT) ✓
- LiDAR BEV histogram generation ✓
- Camera stitching ✓

### Issues

- Coordinate system documentation doesn't match implementation
- BEV semantic placeholder limits model performance

## 5. Training Configuration

### Current Setup

- Using `extended_transfuser_config.py` to avoid modifying original code
- Successfully caching and training on Bench2Drive-Base dataset
- Multi-GPU training working with proper batch sizes

## Summary

Critical issues that affect training:

1. **Coordinate System Crisis**:
   - LiDAR may have X/Y axis flip vs BEV map
   - Agent bounding boxes don't align with roads
   - Ego vehicle appears outside road in BEV
   - These misalignments will prevent successful training

2. **BEV Generation Architecture**:
   - Split static/dynamic generation causes inconsistencies
   - Should generate complete maps during training
   - Missing camera-based semantic segmentation

3. **Documentation vs Reality**:
   - Code comments don't match implementation
   - Ego position, coordinate systems all incorrectly documented

**Current Status**: The integration has fundamental coordinate system issues that must be resolved before training can succeed. The visualization fixes revealed deeper problems in the data pipeline.

## Investigation Findings

### Coordinate System Analysis

1. **LiDAR BEV Generation** (`transfuser_features_b2d.py`):

   ```python
   np.add.at(hist_full, (pixel_coords_full[:, 1], pixel_coords_full[:, 0]), 1)
   ```

   - Uses (y, x) indexing (row-major)

2. **BEV Semantic Drawing** (`bev_semantic_utils.py`):

   ```python
   # In draw_rotated_box:
   abs_corners = rotated_corners + np.array([[center[1], center[0]]])
   ```

   - Has to swap coordinates for OpenCV which expects (x, y)

3. **Trajectory & Agent Alignment**:
   - Trajectories use ego-centric (x=forward, y=left) coordinates
   - Must be correctly mapped to BEV pixel space
   - Current implementation may have inconsistent mappings

This coordinate mismatch between components will cause:

- LiDAR features not aligning with BEV semantic maps
- Trajectories appearing in wrong positions
- Agents (including ego) appearing off-road
- Training failure due to misaligned multi-modal inputs

## Codes and files might be useful

- The script for training in Bench2Drive: `train_bench2drive_full.sh`
- The script for generating BEV cache: `scripts/generate_bev_cache.py`
- The script for generate training cache from BEV cache: `scripts/cache_bench2drive_dataset.py`
- The script for visualizing BEV cache: `visualize_bench2drive_cached.py`
- The current BEV cache location: `/workspace/navsim_workspace/cache/Bench2Drive-Base-full_bev_cache`
- The current training cache location: `/workspace/navsim_workspace/cache/Bench2Drive-Base-training_cache`
- The original Bench2Drive dataset: `/workspace/Bench2Drive-Base`
- The original Bench2Drive map: `/workspace/Bench2Drive-Map`
