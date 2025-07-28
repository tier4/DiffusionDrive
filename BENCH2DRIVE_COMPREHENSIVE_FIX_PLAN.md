# Bench2Drive NaN Investigation and TDD Fix Plan

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Investigation Findings](#investigation-findings)
3. [Root Cause Analysis](#root-cause-analysis)
4. [TDD Implementation Plan](#tdd-implementation-plan)
5. [Test Cases](#test-cases)
6. [Implementation Details](#implementation-details)
7. [Validation and Success Criteria](#validation-and-success-criteria)
8. [Timeline](#timeline)

## Executive Summary

Analysis of 10,000 cached samples from both NavSim and Bench2Drive datasets reveals significant differences that contribute to NaN losses during training. This document combines the investigation findings with a comprehensive Test-Driven Development (TDD) approach to fix the identified issues.

## Investigation Findings

### Cache Data Comparison Results

#### 1. Status Feature Differences

**NavSim**: min=-3.31, max=15.49, mean=0.696, std=1.94  
**Bench2Drive**: min=-26.97, max=14.32, mean=0.458, std=2.05

- Bench2Drive has an extreme minimum value of -26.97 (8x more negative than NavSim)

#### 2. Heading Distribution Differences  

**NavSim**: min=-1.65, max=1.44 radians, mean=0.050, std=0.233  
**Bench2Drive**: min=-0.110, max=0.110 radians, mean=-0.0006, std=0.017

**Root Cause**: Code incorrectly uses `anno['theta']` instead of `anno['bounding_boxes'][ego_index]['rotation'][2]`

#### 3. Trajectory Coordinate Differences

**NavSim**: min=-0.096, max=61.36 meters, mean=9.87  
**Bench2Drive**: min=-39.28, max=40.67 meters, mean=-0.082

- NavSim shows forward-biased distribution
- Bench2Drive is centered around zero

#### 4. Agent Information

- NavSim: Contains actual agent states
- Bench2Drive: All agent_states are zeros (100% failure rate)

#### 5. BEV Semantic Map

- NavSim: Uses classes 0-6
- Bench2Drive: Only classes 0-4 (missing vehicles and pedestrians)

### Root Cause Analysis

#### Critical Issues Identified

1. **Heading Bug** ✗ - Using wrong `anno['theta']` instead of ego vehicle yaw
2. **Trajectory Coordinates** ✗ - Not properly ego-relative (37.4% accuracy)
3. **Agent Detection** ✗ - All agent states are zeros
4. **BEV Generation** ✗ - No vehicles/pedestrians in semantic maps
5. **Driving Commands** ✓ - Correctly one-hot encoded
6. **Speed Values** ✓ - Correctly extracted

#### Key Discoveries

- **world2ego matrix available** in 100% of files
- **Object classes are string-based** ('vehicle', 'walker', etc.)
- **Pedestrians exist as 'walker'** class (0.30% of objects)
- **All vehicles grouped** under single 'vehicle' class

## TDD Implementation Plan

### Test Infrastructure

- Framework: pytest
- Test location: `tests/` directory
- Mock data: `/workspace/Bench2Drive-mini`

### Testing Philosophy

Following TDD principles:

1. **Red Phase**: Write tests first (they should fail)
2. **Green Phase**: Implement minimal code to pass tests
3. **Refactor Phase**: Improve code quality

## Test Cases

### Fix 0: Object Class Mapping Constants

**Test File**: `tests/test_bench2drive_class_mapping.py`

```python
import pytest
from navsim.common.bench2drive_constants import B2D_CLASS_TO_NAVSIM

class TestBench2DriveClassMapping:
    """Test Bench2Drive to NavSim class mappings."""
    
    def test_class_mapping_exists(self):
        """Test that class mapping constant exists."""
        assert hasattr('navsim.common.bench2drive_constants', 'B2D_CLASS_TO_NAVSIM')
        assert isinstance(B2D_CLASS_TO_NAVSIM, dict)
    
    def test_vehicle_mapping(self):
        """Test vehicle class mapping."""
        assert B2D_CLASS_TO_NAVSIM['vehicle'] == 5
    
    def test_walker_mapping(self):
        """Test walker/pedestrian class mapping."""
        assert B2D_CLASS_TO_NAVSIM['walker'] == 6
    
    def test_static_object_mappings(self):
        """Test static object mappings."""
        assert B2D_CLASS_TO_NAVSIM['traffic_light'] == 4
        assert B2D_CLASS_TO_NAVSIM['traffic_sign'] == 4
    
    def test_ego_vehicle_mapping(self):
        """Test ego vehicle is marked for skipping."""
        assert B2D_CLASS_TO_NAVSIM['ego_vehicle'] == -1
```

### Fix 1: Heading Correction

**Test File**: `tests/test_bench2drive_heading_fix.py`

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch
from navsim.common.bench2drive_scene import Bench2DriveScene

class TestBench2DriveHeadingFix:
    """Test heading extraction from ego vehicle bounding box."""
    
    def test_extract_ego_heading_from_bounding_box(self, mock_scene):
        """Test that ego heading is extracted from bounding box rotation."""
        test_anno = {
            'theta': 4.863,  # This should NOT be used
            'bounding_boxes': [{
                'class': 'ego_vehicle',
                'rotation': [0, 0, -171.349],  # Actual ego yaw
                'location': [100.0, 200.0, 0.5],
                'world2ego': np.eye(4).tolist()
            }]
        }
        
        with patch.object(mock_scene, '_load_annotation', return_value=test_anno):
            ego_status = mock_scene._extract_ego_status(test_anno)
            
            # Expected heading: -171.349 degrees = -2.990 radians
            expected_heading = np.radians(-171.349)
            assert np.isclose(ego_status.ego_pose[2], expected_heading, atol=0.001)
```

### Fix 2: Trajectory Transformation

**Test File**: `tests/test_bench2drive_trajectory_transform.py`

```python
class TestBench2DriveTrajectoryTransform:
    """Test trajectory transformation using world2ego matrix."""
    
    def test_trajectory_uses_world2ego_matrix(self, mock_scene):
        """Test that trajectory calculation uses world2ego transformation."""
        # Test that trajectories are properly transformed to ego-relative coordinates
        # using the world2ego matrix from bounding boxes
```

### Fix 3: Agent States

**Test File**: `tests/test_bench2drive_agent_states.py`

```python
class TestBench2DriveAgentStates:
    """Test agent state extraction with proper class mapping."""
    
    def test_agent_detection_with_world2ego(self, mock_scene):
        """Test agent detection using world2ego transformation."""
        # Test detection of vehicles and walkers
        # Test proper coordinate transformation
        # Test dimension capping for pedestrians
```

### Fix 4: BEV Integration

**Test File**: `tests/test_bench2drive_bev_integration.py`

```python
class TestBench2DriveBEVIntegration:
    """Test BEV semantic map generation with agent integration."""
    
    def test_bev_includes_vehicles_and_pedestrians(self, mock_scene):
        """Test that BEV map includes rendered vehicles and pedestrians."""
        # Test that vehicles (class 5) and pedestrians (class 6) are rendered
        # Test proper positioning in BEV coordinates
```

## Implementation Details

### Fix 0: Add Object Class Mapping

**File**: `navsim/common/bench2drive_constants.py`

```python
# Bench2Drive string class names to NavSim semantic classes
B2D_CLASS_TO_NAVSIM = {
    'vehicle': 5,           # All vehicles → NavSim class 5
    'walker': 6,            # Pedestrians → NavSim class 6
    'traffic_light': 4,     # Static objects → NavSim class 4
    'traffic_sign': 4,      # Static objects → NavSim class 4
    'ego_vehicle': -1,      # Skip ego vehicle
}
```

### Fix 1: Heading Correction

**File**: `navsim/common/bench2drive_scene.py` - `_extract_ego_status()`

Key changes:

- Find ego vehicle in bounding boxes
- Extract heading from `rotation[2]` (yaw component)
- Extract position from bounding box location
- Fallback to `anno['theta']` only if ego not found

### Fix 2: Trajectory Coordinate Transformation

**File**: `navsim/common/bench2drive_scene.py` - `get_future_trajectory()`

Key changes:

- Use `world2ego` matrix from ego bounding box
- Transform future positions using matrix multiplication
- Calculate relative heading from rotation differences

### Fix 3: Agent States Implementation

**File**: `navsim/common/bench2drive_scene.py` - `get_agents()`

Key changes:

- Use `world2ego` matrix for coordinate transformation
- Apply class mapping for vehicles and walkers
- Filter by lidar range (32m)
- Cap pedestrian dimensions
- Return agent types for BEV rendering

### Fix 4: BEV Semantic Map Integration

**File**: `navsim/common/bench2drive_scene.py` - `get_bev_semantic_map()`

Key changes:

- Get agents with types from Fix 3
- Render agents into BEV based on type
- Preserve all 7 NavSim semantic classes

## Validation and Success Criteria

### Validation Script

**File**: `scripts/validate_bench2drive_fixes.py`

```python
def validate_heading_distribution(loader, num_samples=1000):
    """Validate heading distribution after fix."""
    # Should have std > 0.1 rad (not all near zero)

def validate_trajectory_ego_relative(loader, num_samples=1000):
    """Validate trajectories are ego-relative."""
    # >95% should start within 1m of ego

def validate_agent_detection(loader, num_samples=1000):
    """Validate agent detection after fix."""
    # >50% of samples should have detected agents

def validate_bev_content(loader, num_samples=500):
    """Validate BEV maps contain vehicles/pedestrians."""
    # >50% should contain vehicles (class 5)
    # ~0.3% should contain pedestrians (class 6)
```

### Success Metrics

1. **All unit tests pass** - 100% test coverage for fixed methods
2. **Heading std > 0.1 rad** - Similar to NavSim distribution
3. **>95% trajectories ego-relative** - Start within 1m of ego
4. **>50% agent detection rate** - Agents detected in majority of samples
5. **>50% BEV vehicle presence** - Vehicles rendered in BEV maps
6. **No NaN losses** - Successful training without numerical issues

## Timeline

### Phase 1: TDD Implementation (Days 1-5)

#### Day 1: Setup and Class Mapping

- [ ] Write test for class mapping constants
- [ ] Implement B2D_CLASS_TO_NAVSIM
- [ ] Verify test passes

#### Day 2: Heading Fix

- [ ] Write comprehensive heading tests
- [ ] Implement heading fix in `_extract_ego_status()`
- [ ] Run validation for heading distribution

#### Day 3: Trajectory Transformation

- [ ] Write trajectory transformation tests
- [ ] Implement world2ego matrix usage
- [ ] Validate trajectory ego-relative accuracy

#### Day 4: Agent States

- [ ] Write agent detection tests
- [ ] Implement agent state extraction with class mapping
- [ ] Validate agent detection rates

#### Day 5: BEV Integration

- [ ] Write BEV integration tests
- [ ] Implement agent rendering in BEV maps
- [ ] Run full validation suite

### Phase 2: Integration and Validation (Days 6-7)

#### Day 6: Integration Testing

- [ ] Run all tests together
- [ ] Re-cache subset of data
- [ ] Verify cached data quality
- [ ] Performance optimization

#### Day 7: Full Dataset Validation

- [ ] Clear old cache
- [ ] Re-cache full dataset with fixes
- [ ] Run training test to verify no NaN losses

## Risk Mitigation

1. **Backward Compatibility**: Keep fallback code paths
2. **Performance**: Profile code to ensure no slowdown
3. **Memory Usage**: Monitor during caching
4. **Data Validation**: Add assertions for invalid data
5. **Incremental Testing**: Test each fix independently

## Continuous Integration

Add to `.github/workflows/test.yml`:

```yaml
- name: Run Bench2Drive Tests
  run: |
    pytest tests/test_bench2drive_*.py -v
    python scripts/validate_bench2drive_fixes.py
```

## Appendix: Normalization Parameters

Current settings that need adjustment after fixes:

```python
"bench2drive": {
    "x": {"offset": 35.0, "scale": 70.0},
    "y": {"offset": 35.0, "scale": 70.0},
    "heading": {"offset": 0.05, "scale": 0.5}  # Needs update after heading fix
}
```

---
*This document combines investigation findings from 2025-07-28/29 with a comprehensive TDD approach for implementation.*
