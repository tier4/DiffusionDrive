# Bench2Drive TDD Fix Plan

## Overview

This document outlines a Test-Driven Development (TDD) approach to fix the critical issues identified in the Bench2Drive dataset integration. Each fix will be implemented with tests written first to ensure correctness and prevent regressions.

## Current Test Infrastructure

### Existing Test Files
- `tests/test_bench2drive_caching.py` - Tests basic caching functionality
- `tests/test_bench2drive_minimal.py` - Tests basic integration and data loading
- `tests/test_bev_generation_dataset.py` - Tests BEV generation
- `tests/test_trajectory_normalization.py` - Tests trajectory normalization

### Test Framework
- Using pytest as the test framework
- Tests organized under `tests/` directory
- Mock data available in mini dataset at `/workspace/Bench2Drive-mini`

## Phase 2: Core Fixes Implementation with TDD

### Fix 0: Add Object Class Mapping Constants

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
    
    def test_all_expected_classes_present(self):
        """Test all expected classes are in mapping."""
        expected_classes = ['vehicle', 'walker', 'traffic_light', 'traffic_sign', 'ego_vehicle']
        for cls in expected_classes:
            assert cls in B2D_CLASS_TO_NAVSIM
```

**Implementation**: Add to `navsim/common/bench2drive_constants.py`

### Fix 1: Heading Correction

**Test File**: `tests/test_bench2drive_heading_fix.py`

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch
from navsim.common.bench2drive_scene import Bench2DriveScene
from navsim.common.bench2drive_dataloader import Bench2DriveConfig

class TestBench2DriveHeadingFix:
    """Test heading extraction from ego vehicle bounding box."""
    
    @pytest.fixture
    def mock_scene(self):
        """Create mock scene with test data."""
        config = Mock(spec=Bench2DriveConfig)
        scene_info = {
            'frames': ['frame_0000.json'],
            'base_path': '/test/path',
            'scenario': 'test_scenario',
            'start_idx': 0
        }
        scene = Bench2DriveScene(scene_info, config)
        return scene
    
    def test_extract_ego_heading_from_bounding_box(self, mock_scene):
        """Test that ego heading is extracted from bounding box rotation."""
        # Mock annotation with ego vehicle in bounding boxes
        test_anno = {
            'theta': 4.863,  # This should NOT be used
            'x': 100.0,
            'y': 200.0,
            'speed': 10.0,
            'acceleration': [0.5, 0.0, 0.0],
            'command_near': 3,
            'bounding_boxes': [
                {
                    'class': 'vehicle',
                    'rotation': [0, 0, 45.0]  # roll, pitch, yaw in degrees
                },
                {
                    'class': 'ego_vehicle',
                    'rotation': [0, 0, -171.349],  # Actual ego yaw
                    'location': [100.0, 200.0, 0.5],
                    'world2ego': np.eye(4).tolist()
                }
            ]
        }
        
        with patch.object(mock_scene, '_load_annotation', return_value=test_anno):
            ego_status = mock_scene._extract_ego_status(test_anno)
            
            # Expected heading: -171.349 degrees = -2.990 radians
            expected_heading = np.radians(-171.349)
            
            # Check ego heading (should be from bounding box, not theta)
            assert np.isclose(ego_status.ego_pose[2], expected_heading, atol=0.001)
            
            # Ensure theta was not used
            theta_heading = -np.radians(4.863)  # What would be used with old code
            assert not np.isclose(ego_status.ego_pose[2], theta_heading, atol=0.001)
    
    def test_ego_heading_fallback_when_no_bbox(self, mock_scene):
        """Test fallback to theta when ego not found in bounding boxes."""
        test_anno = {
            'theta': 45.0,
            'x': 100.0,
            'y': 200.0,
            'speed': 10.0,
            'acceleration': [0.5, 0.0, 0.0],
            'command_near': 3,
            'bounding_boxes': []  # No ego vehicle
        }
        
        ego_status = mock_scene._extract_ego_status(test_anno)
        
        # Should fallback to using theta
        expected_heading = -np.radians(45.0)
        assert np.isclose(ego_status.ego_pose[2], expected_heading, atol=0.001)
    
    def test_ego_position_from_bounding_box(self, mock_scene):
        """Test that ego position is also extracted from bounding box."""
        test_anno = {
            'x': 50.0,  # Old position (should not be used)
            'y': 60.0,
            'theta': 0.0,
            'speed': 5.0,
            'acceleration': [0, 0, 0],
            'command_near': 3,
            'bounding_boxes': [
                {
                    'class': 'ego_vehicle',
                    'location': [100.0, 200.0, 0.5],  # Actual position
                    'rotation': [0, 0, 90.0],
                    'world2ego': np.eye(4).tolist()
                }
            ]
        }
        
        ego_status = mock_scene._extract_ego_status(test_anno)
        
        # Check position from bounding box
        assert ego_status.ego_pose[0] == 100.0
        assert ego_status.ego_pose[1] == 200.0
```

### Fix 2: Trajectory Coordinate Transformation

**Test File**: `tests/test_bench2drive_trajectory_transform.py`

```python
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from navsim.common.bench2drive_scene import Bench2DriveScene

class TestBench2DriveTrajectoryTransform:
    """Test trajectory transformation using world2ego matrix."""
    
    @pytest.fixture
    def mock_scene(self):
        """Create mock scene for testing."""
        config = Mock()
        scene_info = {
            'frames': ['frame_0000.json', 'frame_0001.json', 'frame_0002.json'],
            'base_path': '/test/path',
            'scenario': 'test',
            'start_idx': 0
        }
        scene = Bench2DriveScene(scene_info, config)
        return scene
    
    def test_trajectory_uses_world2ego_matrix(self, mock_scene):
        """Test that trajectory calculation uses world2ego transformation."""
        # Current frame annotation
        current_anno = {
            'bounding_boxes': [{
                'class': 'ego_vehicle',
                'location': [100.0, 200.0, 0.0],
                'rotation': [0, 0, 45.0],  # 45 degrees yaw
                'world2ego': [
                    [0.707, 0.707, 0, -212.132],   # cos(45), sin(45)
                    [-0.707, 0.707, 0, 70.711],    # -sin(45), cos(45)
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ]
            }]
        }
        
        # Future frame annotation (ego moved forward and turned)
        future_anno = {
            'bounding_boxes': [{
                'class': 'ego_vehicle', 
                'location': [110.0, 210.0, 0.0],  # Moved 10m forward at 45 degrees
                'rotation': [0, 0, 60.0],  # Turned to 60 degrees
            }]
        }
        
        # Mock the annotation loading
        def mock_load_annotation(idx):
            if idx == 0:
                return current_anno
            else:
                return future_anno
        
        with patch.object(mock_scene, '_load_annotation', side_effect=mock_load_annotation):
            # Set history frames to 0 for this test
            mock_scene.history_frames = 0
            trajectory = mock_scene.get_future_trajectory(0)
        
        # Expected ego-relative position after transformation
        # World delta: [10, 10] at 45 degrees
        # In ego frame: should be approximately [14.14, 0] (forward only)
        expected_x = 14.142  # sqrt(10^2 + 10^2)
        expected_y = 0.0
        expected_heading = np.radians(15.0)  # 60 - 45 degrees
        
        assert isinstance(trajectory, torch.Tensor)
        assert trajectory.shape == (8, 3)  # NUM_FUTURE_WAYPOINTS x 3
        
        # Check first waypoint
        assert np.isclose(trajectory[0, 0].item(), expected_x, atol=0.5)
        assert np.isclose(trajectory[0, 1].item(), expected_y, atol=0.5)
        assert np.isclose(trajectory[0, 2].item(), expected_heading, atol=0.1)
    
    def test_trajectory_extrapolation(self, mock_scene):
        """Test trajectory extrapolation when running out of frames."""
        current_anno = {
            'bounding_boxes': [{
                'class': 'ego_vehicle',
                'location': [0, 0, 0],
                'rotation': [0, 0, 0],
                'world2ego': np.eye(4).tolist()
            }]
        }
        
        future_anno = {
            'bounding_boxes': [{
                'class': 'ego_vehicle',
                'location': [10, 0, 0],
                'rotation': [0, 0, 0]
            }]
        }
        
        # Only return 2 frames worth of data
        def mock_load_annotation(idx):
            if idx == 0:
                return current_anno
            elif idx == 1:
                return future_anno
            else:
                return None
        
        with patch.object(mock_scene, '_load_annotation', side_effect=mock_load_annotation):
            mock_scene.history_frames = 0
            trajectory = mock_scene.get_future_trajectory(0)
        
        # First waypoint should be calculated
        assert trajectory[0, 0].item() == 10.0
        
        # Rest should be extrapolated (same as last valid)
        for i in range(2, 8):
            assert torch.allclose(trajectory[i], trajectory[1])
```

### Fix 3: Agent States Implementation

**Test File**: `tests/test_bench2drive_agent_states.py`

```python
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from navsim.common.bench2drive_scene import Bench2DriveScene

class TestBench2DriveAgentStates:
    """Test agent state extraction with proper class mapping."""
    
    @pytest.fixture
    def mock_scene(self):
        """Create mock scene."""
        config = Mock()
        scene_info = {
            'frames': ['frame_0000.json'],
            'base_path': '/test/path'
        }
        scene = Bench2DriveScene(scene_info, config)
        return scene
    
    def test_agent_detection_with_world2ego(self, mock_scene):
        """Test agent detection using world2ego transformation."""
        test_anno = {
            'bounding_boxes': [
                {
                    'class': 'ego_vehicle',
                    'location': [100.0, 200.0, 0.0],
                    'rotation': [0, 0, 0],
                    'world2ego': np.eye(4).tolist()
                },
                {
                    'class': 'vehicle',
                    'location': [110.0, 200.0, 0.0],  # 10m ahead
                    'rotation': [0, 0, 90.0],  # Facing perpendicular
                    'extent': [2.0, 1.0, 1.0],  # Half-sizes
                    'speed': 5.0
                },
                {
                    'class': 'walker',  # Pedestrian
                    'location': [105.0, 205.0, 0.0],  # 5m ahead, 5m left
                    'rotation': [0, 0, 180.0],
                    'extent': [0.3, 0.3, 0.9],  # Small pedestrian
                    'speed': 1.0
                }
            ]
        }
        
        with patch.object(mock_scene, '_load_annotation', return_value=test_anno):
            agents, labels = mock_scene.get_agents(0)
        
        # Should detect 2 agents (vehicle and walker)
        assert torch.sum(labels).item() == 2
        
        # First agent (vehicle) - 10m ahead
        assert np.isclose(agents[0, 0].item(), 10.0, atol=0.1)  # x
        assert np.isclose(agents[0, 1].item(), 0.0, atol=0.1)   # y
        assert np.isclose(agents[0, 2].item(), np.pi/2, atol=0.1)  # heading
        assert np.isclose(agents[0, 3].item(), 4.0, atol=0.1)   # length (2*extent)
        assert np.isclose(agents[0, 4].item(), 2.0, atol=0.1)   # width
        
        # Second agent (walker) - 5m ahead, 5m left
        assert np.isclose(agents[1, 0].item(), 5.0, atol=0.1)   # x
        assert np.isclose(agents[1, 1].item(), 5.0, atol=0.1)   # y
        # Check pedestrian dimensions are capped
        assert agents[1, 3].item() <= 1.0  # Max pedestrian length
        assert agents[1, 4].item() <= 0.6  # Max pedestrian width
    
    def test_agent_filtering_by_range(self, mock_scene):
        """Test that agents outside lidar range are filtered."""
        test_anno = {
            'bounding_boxes': [
                {
                    'class': 'ego_vehicle',
                    'location': [0, 0, 0],
                    'rotation': [0, 0, 0],
                    'world2ego': np.eye(4).tolist()
                },
                {
                    'class': 'vehicle',
                    'location': [20.0, 0, 0],  # Within 32m range
                    'extent': [2.0, 1.0, 1.0]
                },
                {
                    'class': 'vehicle', 
                    'location': [50.0, 0, 0],  # Outside 32m range
                    'extent': [2.0, 1.0, 1.0]
                }
            ]
        }
        
        with patch.object(mock_scene, '_load_annotation', return_value=test_anno):
            agents, labels = mock_scene.get_agents(0)
        
        # Should only detect 1 agent (within range)
        assert torch.sum(labels).item() == 1
        assert agents[0, 0].item() == 20.0
    
    def test_agent_class_filtering(self, mock_scene):
        """Test that only vehicles and pedestrians are included."""
        test_anno = {
            'bounding_boxes': [
                {
                    'class': 'ego_vehicle',
                    'location': [0, 0, 0],
                    'rotation': [0, 0, 0],
                    'world2ego': np.eye(4).tolist()
                },
                {'class': 'vehicle', 'location': [10, 0, 0], 'extent': [2, 1, 1]},
                {'class': 'walker', 'location': [5, 5, 0], 'extent': [0.3, 0.3, 0.9]},
                {'class': 'traffic_light', 'location': [15, 0, 0]},  # Should be filtered
                {'class': 'traffic_sign', 'location': [20, 0, 0]}    # Should be filtered
            ]
        }
        
        with patch.object(mock_scene, '_load_annotation', return_value=test_anno):
            agents, labels = mock_scene.get_agents(0)
        
        # Should only detect vehicle and walker
        assert torch.sum(labels).item() == 2
```

### Fix 4: BEV Semantic Map Integration

**Test File**: `tests/test_bench2drive_bev_integration.py`

```python
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from navsim.common.bench2drive_scene import Bench2DriveScene

class TestBench2DriveBEVIntegration:
    """Test BEV semantic map generation with agent integration."""
    
    @pytest.fixture
    def mock_scene(self):
        """Create mock scene."""
        config = Mock()
        config.bev_cache_dir = None
        scene_info = {
            'frames': ['frame_0000.json'],
            'base_path': '/test/path'
        }
        scene = Bench2DriveScene(scene_info, config)
        return scene
    
    def test_bev_includes_vehicles_and_pedestrians(self, mock_scene):
        """Test that BEV map includes rendered vehicles and pedestrians."""
        # Mock agents to be rendered
        mock_agents = torch.tensor([
            [10.0, 0.0, 0.0, 4.0, 2.0],    # Vehicle ahead
            [5.0, 5.0, 0.0, 0.6, 0.4],      # Pedestrian
            [0.0, 0.0, 0.0, 0.0, 0.0],      # Invalid (padding)
        ])
        mock_labels = torch.tensor([True, True, False])
        mock_types = [5, 6, 0]  # Vehicle, pedestrian, invalid
        
        # Mock base BEV (just road)
        base_bev = torch.ones((128, 256), dtype=torch.int64) * 1  # All road
        
        with patch.object(mock_scene, 'get_agents', return_value=(mock_agents, mock_labels, mock_types)):
            with patch('navsim.common.bev_semantic_utils.generate_simple_bev_semantic', 
                      return_value=base_bev.numpy()):
                bev_map = mock_scene.get_bev_semantic_map(0)
        
        # Check that BEV contains vehicle pixels (class 5)
        vehicle_pixels = (bev_map == 5).sum().item()
        assert vehicle_pixels > 0, "BEV should contain vehicle pixels"
        
        # Check that BEV contains pedestrian pixels (class 6)
        pedestrian_pixels = (bev_map == 6).sum().item()
        assert pedestrian_pixels > 0, "BEV should contain pedestrian pixels"
        
        # Check dimensions
        assert bev_map.shape == (128, 256)
    
    def test_bev_agent_positioning(self, mock_scene):
        """Test correct positioning of agents in BEV."""
        # Single vehicle directly ahead at 10m
        mock_agents = torch.tensor([[10.0, 0.0, 0.0, 4.0, 2.0]])
        mock_labels = torch.tensor([True])
        mock_types = [5]
        
        base_bev = torch.zeros((128, 256), dtype=torch.int64)
        
        with patch.object(mock_scene, 'get_agents', return_value=(mock_agents, mock_labels, mock_types)):
            with patch('navsim.common.bev_semantic_utils.generate_simple_bev_semantic',
                      return_value=base_bev.numpy()):
                bev_map = mock_scene.get_bev_semantic_map(0)
        
        # BEV coordinate system: origin at center bottom (x=128, y=128)
        # 10m ahead at 0.25m/pixel = 40 pixels
        expected_x = 128  # Center
        expected_y = 128 - 40  # 40 pixels up from bottom
        
        # Check area around expected position
        region = bev_map[expected_y-8:expected_y+8, expected_x-4:expected_x+4]
        vehicle_in_region = (region == 5).sum().item()
        assert vehicle_in_region > 0, f"Vehicle should be around ({expected_x}, {expected_y})"
    
    def test_all_semantic_classes_preserved(self, mock_scene):
        """Test that all 7 NavSim semantic classes can be represented."""
        # Mock base BEV with various classes
        base_bev = np.zeros((128, 256), dtype=np.int64)
        base_bev[0:20, :] = 0    # Background
        base_bev[20:40, :] = 1   # Road
        base_bev[40:60, :] = 2   # Lane line
        base_bev[60:80, :] = 3   # Walkway
        base_bev[80:100, :] = 4  # Stop line
        
        # No agents for this test
        mock_agents = torch.zeros((30, 5))
        mock_labels = torch.zeros(30, dtype=torch.bool)
        mock_types = [0] * 30
        
        with patch.object(mock_scene, 'get_agents', return_value=(mock_agents, mock_labels, mock_types)):
            with patch('navsim.common.bev_semantic_utils.generate_simple_bev_semantic',
                      return_value=base_bev):
                bev_map = mock_scene.get_bev_semantic_map(0)
        
        # Check all base classes are preserved
        unique_classes = torch.unique(bev_map).tolist()
        for cls in [0, 1, 2, 3, 4]:
            assert cls in unique_classes, f"Class {cls} should be in BEV"
```

### Test Execution Plan

1. **Write Tests First** (Red Phase)
   - Create test files as outlined above
   - Run tests to ensure they fail (no implementation yet)
   - This validates our test cases are checking the right things

2. **Implement Fixes** (Green Phase)
   - Add class mapping constants
   - Fix `_extract_ego_status()` to use bounding box data
   - Fix `get_future_trajectory()` to use world2ego matrix
   - Fix `get_agents()` to properly detect and transform agents
   - Fix `get_bev_semantic_map()` to render agents

3. **Refactor** (Refactor Phase)
   - Clean up code
   - Add proper error handling
   - Optimize performance where needed

### Validation Scripts

Create validation scripts to verify fixes on cached data:

**File**: `scripts/validate_bench2drive_fixes.py`

```python
import numpy as np
import torch
from pathlib import Path
from navsim.common.bench2drive_dataloader import Bench2DriveSceneLoader, Bench2DriveConfig

def validate_heading_distribution(loader, num_samples=1000):
    """Validate heading distribution after fix."""
    headings = []
    
    for i in range(min(num_samples, len(loader))):
        scene = loader.get_scene(loader.scene_tokens[i])
        ego_status = scene._extract_ego_status(scene._load_annotation(0))
        headings.append(ego_status.ego_pose[2])
    
    headings = np.array(headings)
    
    print("Heading Distribution After Fix:")
    print(f"  Min: {np.min(headings):.3f} rad")
    print(f"  Max: {np.max(headings):.3f} rad")
    print(f"  Mean: {np.mean(headings):.3f} rad")
    print(f"  Std: {np.std(headings):.3f} rad")
    
    # Should have reasonable spread (not all near zero)
    assert np.std(headings) > 0.1, "Heading std should be > 0.1 rad"
    return True

def validate_trajectory_ego_relative(loader, num_samples=1000):
    """Validate trajectories are ego-relative."""
    start_distances = []
    
    for i in range(min(num_samples, len(loader))):
        scene = loader.get_scene(loader.scene_tokens[i])
        trajectory = scene.get_future_trajectory()
        
        # First waypoint distance from ego
        dist = np.sqrt(trajectory[0, 0]**2 + trajectory[0, 1]**2)
        start_distances.append(dist)
    
    start_distances = np.array(start_distances)
    within_threshold = np.sum(start_distances < 1.0) / len(start_distances)
    
    print(f"Trajectory Validation:")
    print(f"  Mean first waypoint distance: {np.mean(start_distances):.3f}m")
    print(f"  Trajectories starting < 1m from ego: {within_threshold*100:.1f}%")
    
    assert within_threshold > 0.95, "95% of trajectories should start near ego"
    return True

def validate_agent_detection(loader, num_samples=1000):
    """Validate agent detection after fix."""
    samples_with_agents = 0
    
    for i in range(min(num_samples, len(loader))):
        scene = loader.get_scene(loader.scene_tokens[i])
        agents, labels = scene.get_agents()
        
        if torch.sum(labels).item() > 0:
            samples_with_agents += 1
    
    detection_rate = samples_with_agents / num_samples
    
    print(f"Agent Detection:")
    print(f"  Samples with detected agents: {detection_rate*100:.1f}%")
    
    assert detection_rate > 0.5, "Should detect agents in >50% of samples"
    return True

def validate_bev_content(loader, num_samples=500):
    """Validate BEV maps contain vehicles/pedestrians."""
    samples_with_vehicles = 0
    samples_with_pedestrians = 0
    
    for i in range(min(num_samples, len(loader))):
        scene = loader.get_scene(loader.scene_tokens[i])
        bev = scene.get_bev_semantic_map()
        
        if (bev == 5).any():  # Vehicle class
            samples_with_vehicles += 1
        if (bev == 6).any():  # Pedestrian class
            samples_with_pedestrians += 1
    
    vehicle_rate = samples_with_vehicles / num_samples
    pedestrian_rate = samples_with_pedestrians / num_samples
    
    print(f"BEV Content:")
    print(f"  Samples with vehicles: {vehicle_rate*100:.1f}%")
    print(f"  Samples with pedestrians: {pedestrian_rate*100:.1f}%")
    
    assert vehicle_rate > 0.5, "Should have vehicles in >50% of BEV maps"
    return True

if __name__ == "__main__":
    # Load dataset
    config = Bench2DriveConfig(
        data_root=Path("/workspace/Bench2Drive-mini"),
        scenarios=["ConstructionObstacle_Town05_Route68_Weather8"],
    )
    loader = Bench2DriveSceneLoader(config)
    
    # Run all validations
    print("Running Bench2Drive Fix Validation...")
    print("=" * 50)
    
    validate_heading_distribution(loader)
    print()
    
    validate_trajectory_ego_relative(loader)
    print()
    
    validate_agent_detection(loader)
    print()
    
    validate_bev_content(loader)
    
    print("=" * 50)
    print("All validations passed!")
```

## Implementation Timeline

### Day 1: Setup and Class Mapping
- [ ] Write test for class mapping constants
- [ ] Implement B2D_CLASS_TO_NAVSIM in constants file
- [ ] Verify test passes

### Day 2: Heading Fix
- [ ] Write comprehensive heading tests
- [ ] Implement heading fix in `_extract_ego_status()`
- [ ] Run validation script for heading distribution

### Day 3: Trajectory Transformation
- [ ] Write trajectory transformation tests
- [ ] Implement world2ego matrix usage in `get_future_trajectory()`
- [ ] Validate trajectory ego-relative accuracy

### Day 4: Agent States
- [ ] Write agent detection tests
- [ ] Implement proper agent state extraction with class mapping
- [ ] Validate agent detection rates

### Day 5: BEV Integration
- [ ] Write BEV integration tests
- [ ] Implement agent rendering in BEV maps
- [ ] Full validation of all fixes

### Day 6: Integration Testing
- [ ] Run all tests together
- [ ] Re-cache subset of data
- [ ] Verify cached data quality
- [ ] Performance optimization if needed

### Day 7: Full Dataset Validation
- [ ] Clear old cache
- [ ] Re-cache full dataset with fixes
- [ ] Run training test to verify no NaN losses

## Success Metrics

1. **All unit tests pass** - 100% test coverage for fixed methods
2. **Heading std > 0.1 rad** - Similar to NavSim distribution
3. **>95% trajectories ego-relative** - Start within 1m of ego
4. **>50% agent detection rate** - Agents detected in majority of samples
5. **>50% BEV vehicle presence** - Vehicles rendered in BEV maps
6. **No NaN losses** - Successful training without numerical issues

## Risk Mitigation

1. **Backward Compatibility**: Keep fallback code paths for missing data
2. **Performance**: Profile code to ensure no significant slowdown
3. **Memory Usage**: Monitor memory usage during caching
4. **Data Validation**: Add assertions to catch invalid data early
5. **Incremental Testing**: Test each fix independently before integration

## Continuous Integration

Add to `.github/workflows/test.yml`:

```yaml
- name: Run Bench2Drive Tests
  run: |
    pytest tests/test_bench2drive_*.py -v
    python scripts/validate_bench2drive_fixes.py
```

This ensures fixes remain stable as codebase evolves.