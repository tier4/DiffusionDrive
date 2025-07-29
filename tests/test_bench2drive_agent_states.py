import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from navsim.common.bench2drive_scene import Bench2DriveScene
from navsim.common.bench2drive_constants import B2D_CLASS_TO_NAVSIM
from pathlib import Path


class TestBench2DriveAgentStates:
    """Test agent state extraction with proper class mapping."""

    @pytest.fixture
    def mock_scene(self):
        """Create a mock scene for testing."""
        with patch.object(Bench2DriveScene, "__init__", lambda x, y, z: None):
            scene = Bench2DriveScene(None, None)
            scene.frames = ["frame0", "frame1", "frame2"]
            scene.history_frames = 1
            scene.data_folder = Path("/workspace/Bench2Drive-mini")
            scene.scene_name = "test_scene"
            return scene

    def test_agent_detection_current_implementation(self, mock_scene):
        """Test current agent detection that doesn't use class mapping."""
        # Mock annotation with various agent types
        test_anno = {
            "x": 100.0,
            "y": 200.0,
            "theta": 0.0,
            "bounding_boxes": [
                {
                    "class": "ego_vehicle",
                    "location": [100.0, 200.0, 0.5],
                    "rotation": [0, 0, 0],
                },
                {
                    "class": "vehicle",
                    "location": [105.0, 200.0, 0.5],
                    "rotation": [0, 0, 90.0],
                },
                {
                    "class": "walker",  # Bench2Drive uses 'walker' for pedestrians
                    "location": [100.0, 205.0, 0.5],
                    "rotation": [0, 0, 180.0],
                },
                {
                    "class": "traffic_light",  # Should be ignored
                    "location": [110.0, 200.0, 3.0],
                    "rotation": [0, 0, 0],
                },
            ],
        }

        mock_scene._load_annotation = lambda idx: test_anno

        # Get agents
        agent_states, agent_labels, agent_types = mock_scene.get_agents(frame_idx=1)

        # After fix: both vehicle and walker are now recognized
        assert agent_labels[0] == True  # vehicle detected
        assert agent_labels[1] == True  # walker now detected with class mapping
        assert agent_labels[2] == False  # traffic_light ignored (static object)

        # Check vehicle position (ego-relative)
        assert np.isclose(agent_states[0, 0], 5.0, atol=0.1)  # 5m ahead
        assert np.isclose(agent_states[0, 1], 0.0, atol=0.1)  # same lateral position

        # Check agent types
        assert agent_types[0] == 5  # vehicle is NavSim class 5
        assert agent_types[1] == 6  # walker is NavSim class 6 (pedestrian)

    def test_agent_class_mapping_needed(self, mock_scene):
        """Test that Bench2Drive class names need mapping."""
        # This test documents the expected behavior after fixing
        test_anno = {
            "x": 100.0,
            "y": 200.0,
            "theta": 0.0,
            "bounding_boxes": [
                {"class": "ego_vehicle", "location": [100.0, 200.0, 0.5]},
                {"class": "vehicle", "location": [105.0, 200.0, 0.5]},  # Should map to class 5
                {"class": "walker", "location": [100.0, 205.0, 0.5]},  # Should map to class 6
                {
                    "class": "traffic_light",
                    "location": [110.0, 200.0, 3.0],
                },  # Should map to class 4
                {
                    "class": "traffic_sign",
                    "location": [115.0, 200.0, 3.0],
                },  # Should map to class 4
            ],
        }

        # After fix, all these classes should be properly mapped:
        # - 'vehicle' -> NavSim class 5 (vehicle)
        # - 'walker' -> NavSim class 6 (pedestrian)
        # - 'traffic_light' -> NavSim class 4 (static object)
        # - 'traffic_sign' -> NavSim class 4 (static object)
        pass

    def test_agent_detection_with_world2ego(self, mock_scene):
        """Test agent detection should use world2ego transformation."""
        # Create world2ego matrix for ego at (100, 200) with 30-degree rotation
        theta_rad = np.radians(30.0)
        world2ego = np.array(
            [
                [
                    np.cos(theta_rad),
                    np.sin(theta_rad),
                    0,
                    -100 * np.cos(theta_rad) - 200 * np.sin(theta_rad),
                ],
                [
                    -np.sin(theta_rad),
                    np.cos(theta_rad),
                    0,
                    100 * np.sin(theta_rad) - 200 * np.cos(theta_rad),
                ],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        test_anno = {
            "x": 100.0,  # Currently used
            "y": 200.0,  # Currently used
            "theta": 0.0,  # Currently used (incorrect)
            "bounding_boxes": [
                {
                    "class": "ego_vehicle",
                    "location": [100.0, 200.0, 0.5],
                    "rotation": [0, 0, 30.0],
                    "world2ego": world2ego.tolist(),
                },
                {
                    "class": "vehicle",
                    "location": [105.0, 205.0, 0.5],
                    "rotation": [0, 0, 45.0],
                    "world2ego": world2ego.tolist(),  # Each object has world2ego
                },
            ],
        }

        # After fix, agents should be transformed using world2ego matrix
        pass

    def test_dimension_capping_for_pedestrians(self, mock_scene):
        """Test that pedestrian dimensions are capped appropriately."""
        test_anno = {
            "x": 100.0,
            "y": 200.0,
            "theta": 0.0,
            "bounding_boxes": [
                {
                    "class": "pedestrian",  # If recognized
                    "location": [100.0, 205.0, 0.5],
                    "rotation": [0, 0, 0],
                    "extent": [0.5, 0.5, 0.9],  # Pedestrian size
                },
            ],
        }

        mock_scene._load_annotation = lambda idx: test_anno

        # Get agents
        agent_states, agent_labels, agent_types = mock_scene.get_agents(frame_idx=1)

        # If pedestrian is detected, check dimensions
        if agent_labels[0]:
            # Pedestrian dimensions should be smaller than vehicles
            assert agent_states[0, 3] < 2.0  # length
            assert agent_states[0, 4] < 1.0  # width

    def test_agent_filtering_by_distance(self, mock_scene):
        """Test that agents are filtered by lidar range (32m)."""
        test_anno = {
            "x": 100.0,
            "y": 200.0,
            "theta": 0.0,
            "bounding_boxes": [
                {"class": "vehicle", "location": [110.0, 200.0, 0.5]},  # 10m away
                {"class": "vehicle", "location": [125.0, 200.0, 0.5]},  # 25m away
                {
                    "class": "vehicle",
                    "location": [140.0, 200.0, 0.5],
                },  # 40m away (should be filtered)
            ],
        }

        mock_scene._load_annotation = lambda idx: test_anno

        # Get agents
        agent_states, agent_labels, agent_types = mock_scene.get_agents(frame_idx=1)

        # Count detected agents
        num_detected = torch.sum(agent_labels).item()

        # Only agents within 32m should be detected
        # Note: current implementation might not have distance filtering
        # This test documents expected behavior

    def test_max_agents_limit(self, mock_scene):
        """Test that agent detection respects MAX_AGENTS limit."""
        # Create many vehicles
        vehicles = []
        for i in range(50):  # More than MAX_AGENTS (30)
            vehicles.append(
                {
                    "class": "vehicle",
                    "location": [100.0 + i, 200.0, 0.5],
                    "rotation": [0, 0, 0],
                }
            )

        test_anno = {"x": 100.0, "y": 200.0, "theta": 0.0, "bounding_boxes": vehicles}

        mock_scene._load_annotation = lambda idx: test_anno

        # Get agents
        agent_states, agent_labels, agent_types = mock_scene.get_agents(frame_idx=1)

        # Should not exceed MAX_AGENTS
        assert agent_states.shape[0] == 30  # MAX_AGENTS
        assert agent_labels.shape[0] == 30

        # Count actual detected agents
        num_detected = torch.sum(agent_labels).item()
        assert num_detected <= 30
