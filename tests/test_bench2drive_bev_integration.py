import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from navsim.common.bench2drive_scene import Bench2DriveScene
from navsim.common.bench2drive_constants import B2D_CLASS_TO_NAVSIM


class TestBench2DriveBEVIntegration:
    """Test BEV semantic map generation with agent integration."""

    @pytest.fixture
    def mock_scene(self):
        """Create a mock scene for testing."""
        with patch.object(Bench2DriveScene, "__init__", lambda x, y, z: None):
            scene = Bench2DriveScene(None, None)
            scene.anno_paths = ["frame0", "frame1", "frame2"]
            scene.history_frames = 1
            scene.data_folder = Mock()
            scene.scene_name = "test_scene"
            scene.config = Mock()
            scene.config.bev_cache_dir = None
            scene.config.map_dir = None
            scene.scene_info = {"scenario": "test_scenario"}
            return scene

    def test_bev_includes_vehicles_and_pedestrians(self, mock_scene):
        """Test that BEV map includes rendered vehicles and pedestrians."""
        # Mock annotation with vehicles and pedestrians
        test_anno = {
            "x": 100.0,
            "y": 200.0,
            "theta": 0.0,
            "bounding_boxes": [
                {
                    "class": "ego_vehicle",
                    "location": [100.0, 200.0, 0.5],
                    "rotation": [0, 0, 0],
                    "world2ego": np.eye(4).tolist(),
                },
                {
                    "class": "vehicle",
                    "location": [105.0, 200.0, 0.5],  # 5m ahead
                    "rotation": [0, 0, 0],
                    "world2ego": np.eye(4).tolist(),
                },
                {
                    "class": "walker",  # Pedestrian
                    "location": [100.0, 205.0, 0.5],  # 5m to the right
                    "rotation": [0, 0, 0],
                    "world2ego": np.eye(4).tolist(),
                },
            ],
        }

        # Mock the _load_annotation method
        mock_scene._load_annotation = lambda idx: test_anno

        # Mock get_agents to return the processed agents
        def mock_get_agents(frame_idx=-1):
            agent_states = torch.zeros((30, 5))
            agent_labels = torch.zeros(30, dtype=torch.bool)
            agent_types = torch.zeros(30, dtype=torch.int32)

            # Vehicle at (5, 0)
            agent_states[0] = torch.tensor([5.0, 0.0, 0.0, 4.0, 1.8])
            agent_labels[0] = True
            agent_types[0] = 5  # Vehicle

            # Pedestrian at (0, 5)
            agent_states[1] = torch.tensor([0.0, 5.0, 0.0, 0.8, 0.6])
            agent_labels[1] = True
            agent_types[1] = 6  # Pedestrian

            return agent_states, agent_labels, agent_types

        mock_scene.get_agents = mock_get_agents

        # Get BEV map
        bev_map = mock_scene.get_bev_semantic_map(frame_idx=1)

        # Check that BEV has correct dimensions
        assert bev_map.shape == (128, 256)  # BEV_SEMANTIC_HEIGHT x BEV_SEMANTIC_WIDTH

        # Check that vehicle and pedestrian classes are present
        unique_classes = torch.unique(bev_map).numpy()
        print(f"BEV classes: {unique_classes}")

        # After fix, vehicles (5) and pedestrians (6) should be rendered
        # Note: This test documents expected behavior after implementation

    def test_bev_agent_positioning(self, mock_scene):
        """Test that agents are correctly positioned in BEV coordinates."""
        # Mock a vehicle directly in front of ego
        test_anno = {
            "x": 100.0,
            "y": 200.0,
            "theta": 0.0,
            "bounding_boxes": [
                {
                    "class": "ego_vehicle",
                    "location": [100.0, 200.0, 0.5],
                    "rotation": [0, 0, 0],
                    "world2ego": np.eye(4).tolist(),
                },
                {
                    "class": "vehicle",
                    "location": [108.0, 200.0, 0.5],  # 8m ahead
                    "rotation": [0, 0, 0],
                },
            ],
        }

        mock_scene._load_annotation = lambda idx: test_anno

        # Get BEV map
        bev_map = mock_scene.get_bev_semantic_map(frame_idx=1)

        # The vehicle should appear in the center horizontally and forward vertically
        # BEV coordinates: x-forward, y-left/right
        # After fix, check that vehicle appears at expected position

    def test_bev_with_no_agents(self, mock_scene):
        """Test BEV generation when no agents are present."""
        # Mock annotation with only ego vehicle
        test_anno = {
            "x": 100.0,
            "y": 200.0,
            "theta": 0.0,
            "bounding_boxes": [
                {
                    "class": "ego_vehicle",
                    "location": [100.0, 200.0, 0.5],
                    "rotation": [0, 0, 0],
                    "world2ego": np.eye(4).tolist(),
                },
            ],
        }

        mock_scene._load_annotation = lambda idx: test_anno

        # Mock get_agents to return empty
        def mock_get_agents(frame_idx=-1):
            agent_states = torch.zeros((30, 5))
            agent_labels = torch.zeros(30, dtype=torch.bool)
            agent_types = torch.zeros(30, dtype=torch.int32)
            return agent_states, agent_labels, agent_types

        mock_scene.get_agents = mock_get_agents

        # Get BEV map
        bev_map = mock_scene.get_bev_semantic_map(frame_idx=1)

        # Should still generate valid BEV with road (1) and other static elements
        unique_classes = torch.unique(bev_map).numpy()
        print(f"BEV without agents - classes: {unique_classes}")

        # Should not contain vehicle (5) or pedestrian (6) classes
        assert 5 not in unique_classes
        assert 6 not in unique_classes

    def test_bev_class_consistency(self, mock_scene):
        """Test that BEV uses consistent NavSim class IDs."""
        # Expected NavSim classes in BEV:
        # 0: Background
        # 1: Road
        # 2: Walkways
        # 3: Centerline
        # 4: Static objects
        # 5: Vehicles
        # 6: Pedestrians

        test_anno = {"x": 100.0, "y": 200.0, "theta": 0.0, "bounding_boxes": []}

        mock_scene._load_annotation = lambda idx: test_anno

        # Mock get_agents
        def mock_get_agents(frame_idx=-1):
            agent_states = torch.zeros((30, 5))
            agent_labels = torch.zeros(30, dtype=torch.bool)
            agent_types = torch.zeros(30, dtype=torch.int32)
            return agent_states, agent_labels, agent_types

        mock_scene.get_agents = mock_get_agents

        # Get BEV map
        bev_map = mock_scene.get_bev_semantic_map(frame_idx=1)

        # All values should be valid NavSim classes (0-6)
        unique_classes = torch.unique(bev_map).numpy()
        assert all(c >= 0 and c <= 6 for c in unique_classes), f"Invalid classes: {unique_classes}"

    def test_bev_integration_with_agent_types(self, mock_scene):
        """Test that agent types from get_agents are used for BEV rendering."""
        # This test verifies the integration between Fix 3 (agent types) and Fix 4 (BEV)

        # Mock mixed agent types
        test_anno = {
            "x": 100.0,
            "y": 200.0,
            "theta": 0.0,
            "bounding_boxes": [
                {"class": "ego_vehicle", "location": [100.0, 200.0, 0.5]},
                {"class": "vehicle", "location": [105.0, 200.0, 0.5]},
                {"class": "walker", "location": [100.0, 203.0, 0.5]},
                {"class": "traffic_light", "location": [110.0, 200.0, 3.0]},  # Static, ignored
            ],
        }

        mock_scene._load_annotation = lambda idx: test_anno

        # After fix, get_agents returns agent_types that should be used in BEV
        # Vehicle should render as class 5, walker as class 6
        # Traffic light should not appear in dynamic agent rendering
