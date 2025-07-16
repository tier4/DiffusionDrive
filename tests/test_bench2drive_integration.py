"""
Test script for Bench2Drive integration with DiffusionDrive.
Verifies data loading, feature building, and basic functionality.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder,
)
from navsim.common.bench2drive_dataloader import (
    Bench2DriveConfig,
    Bench2DriveSceneLoader,
    map_carla_command_to_discrete,
)


class TestBench2DriveIntegration:
    """Test suite for Bench2Drive integration."""

    def test_command_mapping(self):
        """Test CARLA command to discrete mapping."""
        # Test cases based on CARLA RoadOption enum
        test_cases = [
            (-1, 3, "VOID → UNKNOWN"),
            (1, 0, "LEFT → LEFT"),
            (2, 2, "RIGHT → RIGHT"),
            (3, 1, "STRAIGHT → STRAIGHT"),
            (4, 1, "LANEFOLLOW → STRAIGHT"),
            (5, 0, "CHANGELANELEFT → LEFT"),
            (6, 2, "CHANGELANERIGHT → RIGHT"),
        ]

        for carla_cmd, expected, description in test_cases:
            result = map_carla_command_to_discrete(carla_cmd)
            assert (
                result == expected
            ), f"{description}: {carla_cmd} → {result} (expected {expected})"

    @pytest.fixture
    def scene_loader_config(self):
        """Create test configuration for scene loader."""
        return Bench2DriveConfig(
            data_root=Path("/workspace/Bench2Drive-Base"),
            scenarios=["ConstructionObstacle"],  # Just one scenario for testing
            sampling_rate=5,
            num_frames=30,
            num_history_frames=4,
            num_future_frames=26,
            extract_tar=False,  # Assume already extracted
        )

    def test_scene_loader_creation(self, scene_loader_config):
        """Test Bench2Drive scene loader creation."""
        try:
            loader = Bench2DriveSceneLoader(scene_loader_config)
            assert loader is not None
            print(f"Created scene loader with {len(loader)} scenes")

            # Skip further tests if no data available
            if len(loader) == 0:
                pytest.skip("No scenes found. Check if data is extracted.")

            return loader
        except Exception as e:
            pytest.fail(f"Error creating scene loader: {e}")

    def test_scene_loading(self, scene_loader_config):
        """Test loading a scene from the dataset."""
        loader = Bench2DriveSceneLoader(scene_loader_config)

        if len(loader) == 0:
            pytest.skip("No scenes found. Check if data is extracted.")

        # Get first scene
        token = loader.scene_tokens[0]
        scene = loader.get_scene(token)

        assert scene is not None
        assert len(scene) > 0
        assert hasattr(scene, "get_agent_input")
        assert hasattr(scene, "get_future_trajectory")
        assert hasattr(scene, "get_agents")

        return scene

    def test_feature_builder(self, scene_loader_config):
        """Test feature builder with a scene."""
        loader = Bench2DriveSceneLoader(scene_loader_config)

        if len(loader) == 0:
            pytest.skip("No scenes found. Check if data is extracted.")

        scene = loader.get_scene(loader.scene_tokens[0])

        # Create config and builders
        config = TransfuserConfig()
        feature_builder = Bench2DriveFeatureBuilder(config)
        target_builder = Bench2DriveTargetBuilder(config)

        # Get agent input
        agent_input = scene.get_agent_input()
        assert agent_input is not None

        # Build features
        features = feature_builder.compute_features(agent_input)
        assert "camera_feature" in features
        assert "lidar_feature" in features
        assert "status_feature" in features

        # Check feature shapes
        assert features["camera_feature"].shape == (3, 256, 1024)
        assert features["lidar_feature"].shape == (1, 256, 256)
        assert features["status_feature"].shape == (8,)

        # Build targets
        targets = target_builder.compute_targets(scene)
        assert "trajectory" in targets
        assert "agent_states" in targets
        assert "agent_labels" in targets
        assert "bev_semantic_map" in targets

        # Check target shapes
        assert targets["trajectory"].shape == (8, 3)
        assert targets["agent_states"].shape == (30, 5)
        assert targets["agent_labels"].shape == (30,)
        assert targets["bev_semantic_map"].shape == (128, 256)

        return features, targets

    @pytest.mark.skipif(
        not Path("/workspace/Bench2Drive-Base").exists(),
        reason="Bench2Drive dataset not available",
    )
    def test_end_to_end_integration(self, scene_loader_config, tmp_path):
        """Test end-to-end integration with visualization."""
        loader = Bench2DriveSceneLoader(scene_loader_config)

        if len(loader) == 0:
            pytest.skip("No scenes found.")

        scene = loader.get_scene(loader.scene_tokens[0])

        # Create builders
        config = TransfuserConfig()
        feature_builder = Bench2DriveFeatureBuilder(config)

        # Build features
        agent_input = scene.get_agent_input()
        features = feature_builder.compute_features(agent_input)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Camera feature
        cam = features["camera_feature"].numpy().transpose(1, 2, 0)
        axes[0].imshow(cam)
        axes[0].set_title("Camera Feature (Stitched)")
        axes[0].axis("off")

        # LiDAR feature
        lidar = features["lidar_feature"].numpy().squeeze()
        axes[1].imshow(lidar, cmap="viridis")
        axes[1].set_title("LiDAR BEV")
        axes[1].axis("off")

        # Status feature
        status = features["status_feature"].numpy()
        labels = ["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN", "Vx", "Vy", "Ax", "Ay"]
        axes[2].bar(range(len(status)), status)
        axes[2].set_xticks(range(len(status)))
        axes[2].set_xticklabels(labels, rotation=45)
        axes[2].set_title("Status Features")

        plt.tight_layout()

        # Save to temp directory
        output_path = tmp_path / "bench2drive_features_test.png"
        plt.savefig(output_path)
        plt.close()

        assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
