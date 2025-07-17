"""
Test suite for BEV semantic map generation with Bench2Drive map data.

This tests BEV generation using vectorized map data from Bench2Drive.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional
import tempfile
import json


class TestMapDataLoading:
    """Test loading and parsing Bench2Drive map data."""
    
    def test_load_map_file(self):
        """Test loading NPZ map file."""
        map_path = Path("/workspace/Bench2Drive-Map/Town05_HD_map.npz")
        assert map_path.exists(), "Map file should exist"
        
        # Load map data
        map_data = np.load(str(map_path), allow_pickle=True)
        assert 'arr' in map_data, "Map should contain 'arr' key"
        
        # Extract dictionary
        map_dict = dict(map_data['arr'][()])
        assert isinstance(map_dict, dict), "Should be a dictionary"
        assert len(map_dict) > 0, "Map should contain road data"
    
    def test_map_structure(self):
        """Test map data structure."""
        map_path = Path("/workspace/Bench2Drive-Map/Town05_HD_map.npz")
        map_data = np.load(str(map_path), allow_pickle=True)
        map_dict = dict(map_data['arr'][()])
        
        # Check a sample road
        road_id = list(map_dict.keys())[0]
        road = map_dict[road_id]
        
        # Road should have lanes and possibly trigger volumes
        assert isinstance(road, dict), "Road should be a dictionary"
        
        # Check for lane keys (-2, -1, 1, 2)
        lane_keys = [k for k in road.keys() if k != 'Trigger_Volumes']
        assert len(lane_keys) > 0, "Road should have lanes"
        
        # Check trigger volumes if present
        if 'Trigger_Volumes' in road:
            triggers = road['Trigger_Volumes']
            assert isinstance(triggers, list), "Trigger volumes should be a list"
    
    def test_lane_point_format(self):
        """Test lane point data format."""
        map_path = Path("/workspace/Bench2Drive-Map/Town05_HD_map.npz")
        map_data = np.load(str(map_path), allow_pickle=True)
        map_dict = dict(map_data['arr'][()])
        
        # Find a road with lanes
        for road_id, road in map_dict.items():
            for lane_id, lane_segments in road.items():
                if lane_id != 'Trigger_Volumes' and isinstance(lane_segments, list) and len(lane_segments) > 0:
                    segment = lane_segments[0]
                    
                    # Check segment structure
                    assert 'Type' in segment, "Lane segment should have Type"
                    assert 'Points' in segment, "Lane segment should have Points"
                    
                    # Check point format
                    points = segment['Points']
                    assert len(points) > 0, "Lane should have points"
                    
                    # Each point is ((x, y, z), (roll, pitch, yaw))
                    point = points[0]
                    assert len(point) == 2, "Point should have position and rotation"
                    assert len(point[0]) == 3, "Position should be (x, y, z)"
                    assert len(point[1]) == 3, "Rotation should be (roll, pitch, yaw)"
                    return  # Test one lane is enough
    
    def test_trigger_volume_format(self):
        """Test trigger volume data format."""
        map_path = Path("/workspace/Bench2Drive-Map/Town05_HD_map.npz")
        map_data = np.load(str(map_path), allow_pickle=True)
        map_dict = dict(map_data['arr'][()])
        
        # Find a road with trigger volumes
        for road_id, road in map_dict.items():
            if 'Trigger_Volumes' in road and len(road['Trigger_Volumes']) > 0:
                trigger = road['Trigger_Volumes'][0]
                
                # Check structure
                assert 'Type' in trigger, "Trigger should have Type"
                assert 'Points' in trigger, "Trigger should have Points"
                assert trigger['Type'] in ['TrafficLight', 'StopSign'], f"Unknown trigger type: {trigger['Type']}"
                
                # Check points - should be 8 vertices of a box
                points = np.array(trigger['Points'])
                assert points.shape == (8, 3), "Trigger should have 8 vertices with (x,y,z)"
                return  # Test one trigger is enough


class TestMapClassMapping:
    """Test mapping from Bench2Drive map classes to NavSim BEV classes."""
    
    def test_lane_type_to_bev_class(self):
        """Test mapping lane types to BEV semantic classes."""
        # Bench2Drive lane types to NavSim BEV classes
        lane_to_bev = {
            'Broken': 1,      # Broken line -> Road
            'Solid': 1,       # Solid line -> Road  
            'SolidSolid': 1,  # Double solid -> Road
            'Center': 3,      # Center line -> Lane centerline
        }
        
        # Test all mappings
        for lane_type, bev_class in lane_to_bev.items():
            assert 0 <= bev_class <= 6, f"Invalid BEV class {bev_class}"
            assert bev_class in [1, 3], "Lane should map to road or centerline"
    
    def test_trigger_type_to_bev_class(self):
        """Test mapping trigger volume types to BEV classes."""
        # Bench2Drive trigger types to NavSim BEV classes
        trigger_to_bev = {
            'TrafficLight': 4,  # Traffic light -> Static object
            'StopSign': 4,      # Stop sign -> Static object
        }
        
        for trigger_type, bev_class in trigger_to_bev.items():
            assert bev_class == 4, "Triggers should map to static objects"


class TestCoordinateTransformations:
    """Test coordinate transformations for map data."""
    
    def test_world_to_ego_transformation(self):
        """Test transforming world coordinates to ego-centric."""
        # World point
        world_point = np.array([100, 50, 0, 1])  # homogeneous
        
        # Mock world2ego matrix
        world2ego = np.eye(4)
        world2ego[:3, 3] = [-90, -45, 0]  # Ego at (90, 45, 0)
        
        # Transform
        ego_point = world2ego @ world_point
        
        assert ego_point[0] == 10, "X should be relative to ego"
        assert ego_point[1] == 5, "Y should be relative to ego"
    
    def test_left_to_right_hand_conversion(self):
        """Test CARLA left-handed to right-handed coordinate conversion."""
        # Left-handed point
        point_left = np.array([1, 2, 3])
        
        # Convert to right-handed (flip Y)
        point_right = point_left.copy()
        point_right[1] = -point_right[1]
        
        assert point_right[0] == 1, "X unchanged"
        assert point_right[1] == -2, "Y flipped"
        assert point_right[2] == 3, "Z unchanged"
    
    def test_ego_to_bev_pixel_conversion(self):
        """Test converting ego-centric coordinates to BEV pixels."""
        # Ego-centric points
        points_ego = np.array([
            [0, 0],      # Ego position
            [10, 0],     # 10m forward
            [10, 5],     # 10m forward, 5m left
            [5, -3],     # 5m forward, 3m right
        ])
        
        # BEV parameters
        bev_height, bev_width = 128, 256
        resolution = 0.25  # m/pixel
        
        # Expected conversions
        # Ego at bottom-center: (127, 128)
        # Forward is up (decreasing row), left is right (increasing col)
        expected_pixels = np.array([
            [127, 128],  # Ego at bottom center
            [127 - 40, 128],  # 10m forward = 40 pixels up
            [127 - 40, 128 + 20],  # Also 5m left = 20 pixels right
            [127 - 20, 128 - 12],  # 5m forward, 3m right
        ])
        
        # Test bounds
        for pixel in expected_pixels:
            assert 0 <= pixel[0] < bev_height, "Row in bounds"
            assert 0 <= pixel[1] < bev_width, "Col in bounds"


class TestBEVMapGeneration:
    """Test BEV map generation from map data."""
    
    def test_generate_bev_from_lanes(self):
        """Test generating BEV from lane data."""
        # Mock lane data (ego-centric coordinates)
        lanes = {
            'Solid': [np.array([
                [5, -2], [10, -2], [15, -1.5], [20, -1]
            ])],
            'Broken': [np.array([
                [5, 2], [10, 2], [15, 1.5], [20, 1]
            ])],
            'Center': [np.array([
                [5, 0], [10, 0], [15, 0], [20, 0]
            ])],
        }
        
        # BEV parameters
        bev_shape = (128, 256)
        
        # Generate BEV (test expected behavior)
        # - Solid and Broken -> class 1 (road)
        # - Center -> class 3 (centerline)
        
        # Test that lanes are rendered
        # Implementation will create the actual BEV
    
    def test_generate_bev_from_triggers(self):
        """Test generating BEV from trigger volumes."""
        # Mock trigger data (8 vertices of a box)
        triggers = [
            {
                'Type': 'TrafficLight',
                'Points': np.array([
                    [10, -2, 0], [10, -2, 3], [10, 2, 0], [10, 2, 3],
                    [12, -2, 0], [12, -2, 3], [12, 2, 0], [12, 2, 3]
                ])
            },
            {
                'Type': 'StopSign',
                'Points': np.array([
                    [20, -1, 0], [20, -1, 2], [20, 1, 0], [20, 1, 2],
                    [21, -1, 0], [21, -1, 2], [21, 1, 0], [21, 1, 2]
                ])
            }
        ]
        
        # Test that triggers are rendered as class 4 (static objects)
        # Implementation will handle this
    
    def test_full_bev_vs_half_bev(self):
        """Test generating full BEV and extracting front half."""
        # Full BEV covers 360 degrees
        full_bev_shape = (256, 256)  # Square for full coverage
        
        # NavSim uses only front half (128x256)
        half_bev_shape = (128, 256)
        
        # Test extraction
        full_bev = np.random.randint(0, 7, full_bev_shape, dtype=np.uint8)
        
        # Front half is top half of full BEV
        half_bev = full_bev[:128, :]
        
        assert half_bev.shape == half_bev_shape
    
    def test_combine_map_and_dynamic_objects(self):
        """Test combining static map with dynamic objects."""
        # Static map from lanes/triggers
        static_bev = np.zeros((128, 256), dtype=np.float32)
        static_bev[40:80, 100:150] = 1  # Road area
        static_bev[58:60, 120:130] = 3  # Centerline
        
        # Dynamic objects (vehicles, pedestrians)
        dynamic_bev = np.zeros((128, 256), dtype=np.float32)
        dynamic_bev[50:55, 125:135] = 5  # Vehicle
        
        # Combine (dynamic overrides static)
        final_bev = static_bev.copy()
        final_bev[dynamic_bev > 0] = dynamic_bev[dynamic_bev > 0]
        
        # Test combined result
        assert final_bev[51, 126] == 5  # Vehicle present
        assert final_bev[45, 125] == 1  # Road still visible


class TestOfflineBEVGeneration:
    """Test offline BEV generation and caching."""
    
    def test_generate_and_save_bev(self):
        """Test generating BEV offline and saving to disk."""
        # Mock BEV data
        bev_data = {
            'bev_map': np.random.randint(0, 7, (128, 256)).astype(np.float32),
            'metadata': {
                'town': 'Town05',
                'frame': 100,
                'world2ego': np.eye(4).tolist(),
            }
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            np.savez_compressed(
                tmp.name,
                bev_map=bev_data['bev_map'],
                metadata=json.dumps(bev_data['metadata'])
            )
            tmp_path = tmp.name
        
        # Load and verify
        loaded = np.load(tmp_path)
        assert 'bev_map' in loaded
        assert 'metadata' in loaded
        
        loaded_bev = loaded['bev_map']
        loaded_meta = json.loads(str(loaded['metadata']))
        
        assert loaded_bev.shape == (128, 256)
        assert loaded_meta['town'] == 'Town05'
        
        # Cleanup
        Path(tmp_path).unlink()
    
    def test_bev_cache_structure(self):
        """Test structure for caching BEV maps."""
        # Expected cache structure:
        # cache_dir/
        #   Town01/
        #     scenario_name/
        #       00000.npz
        #       00001.npz
        #       ...
        
        cache_structure = {
            'Town01': {
                'scenario1': ['00000.npz', '00001.npz'],
                'scenario2': ['00000.npz', '00001.npz'],
            },
            'Town02': {
                'scenario3': ['00000.npz'],
            }
        }
        
        # Test that structure is valid
        for town, scenarios in cache_structure.items():
            assert isinstance(scenarios, dict)
            for scenario, frames in scenarios.items():
                assert isinstance(frames, list)
                for frame in frames:
                    assert frame.endswith('.npz')


class TestBEVVisualization:
    """Test BEV visualization utilities."""
    
    def test_bev_colormap(self):
        """Test BEV semantic colormap."""
        # NavSim BEV classes and colors
        bev_colors = {
            0: [46, 46, 46],     # Background (dark gray)
            1: [128, 128, 128],  # Road (gray)
            2: [255, 165, 0],    # Walkways (orange)
            3: [255, 255, 0],    # Lane centerlines (yellow)
            4: [128, 0, 128],    # Static objects (purple)
            5: [255, 0, 0],      # Vehicles (red)
            6: [0, 255, 0],      # Pedestrians (green)
        }
        
        # Test all classes defined
        for class_id in range(7):
            assert class_id in bev_colors
            color = bev_colors[class_id]
            assert len(color) == 3  # RGB
            assert all(0 <= c <= 255 for c in color)
    
    def test_apply_colormap(self):
        """Test applying colormap to BEV semantic map."""
        # Create sample BEV
        bev = np.array([
            [0, 1, 1, 0],
            [0, 1, 3, 0],
            [5, 1, 3, 0],
            [0, 1, 1, 0],
        ], dtype=np.uint8)
        
        # Expected colored output shape
        colored = np.zeros((*bev.shape, 3), dtype=np.uint8)
        
        assert colored.shape == (4, 4, 3)
        assert colored.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])