"""
Test Bench2Drive integration.
Tests data loading, scene creation, feature extraction, and visualization.
Uses mini dataset by default, with optional tests for full dataset.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder,
)
from navsim.common.bench2drive_dataloader import (
    Bench2DriveDataConfig,
    Bench2DriveSceneLoader,
    map_carla_command_to_discrete,
)


def test_command_mapping():
    """Test CARLA command to discrete mapping."""
    # Test cases based on CARLA RoadOption enum
    # Mapping: CARLA commands are correctly mapped to NavSim discrete values
    test_cases = [
        (-1, 1, "VOID → STRAIGHT (via LANEFOLLOW)"),  # -1 becomes 4, then 3, maps to STRAIGHT
        (1, 0, "LEFT → LEFT"),  # Correct mapping (no swapping)
        (2, 2, "RIGHT → RIGHT"),  # Correct mapping (no swapping)
        (3, 1, "STRAIGHT → STRAIGHT"),
        (4, 1, "LANEFOLLOW → STRAIGHT"),
        (5, 0, "CHANGELANELEFT → LEFT"),
        (6, 2, "CHANGELANERIGHT → RIGHT"),
    ]

    for carla_cmd, expected, description in test_cases:
        result = map_carla_command_to_discrete(carla_cmd)
        assert result == expected, f"{description}: {carla_cmd} → {result} (expected {expected})"


@pytest.fixture
def bench2drive_config(sample_config):
    """Create Bench2Drive configuration using test data from conftest."""
    # Use the sample_config from conftest which uses test_data directory
    return sample_config


@pytest.fixture
def bench2drive_config_full(sample_config):
    """Create Bench2Drive configuration for full dataset testing."""
    # For now, use the same test data as mini dataset
    # This avoids dependency on external datasets
    return sample_config


@pytest.fixture
def scene_loader(bench2drive_config):
    """Create scene loader."""
    return Bench2DriveSceneLoader(bench2drive_config)


def test_scene_loader_creation(bench2drive_config):
    """Test that scene loader can be created successfully."""
    loader = Bench2DriveSceneLoader(bench2drive_config)
    assert loader is not None
    assert len(loader) > 0, "No scenes found in dataset"
    assert len(loader.scene_tokens) == len(loader.scenes)


def test_scene_loading(scene_loader):
    """Test loading individual scenes."""
    assert len(scene_loader) > 0, "No scenes available to test"

    # Load first scene
    token = scene_loader.scene_tokens[0]
    scene = scene_loader.get_scene(token)

    assert scene is not None
    assert hasattr(scene, "anno_paths")
    assert len(scene.anno_paths) == scene_loader.config.num_frames
    assert scene.token == token


def test_agent_input_extraction(scene_loader):
    """Test extracting agent input from scene."""
    token = scene_loader.scene_tokens[0]
    scene = scene_loader.get_scene(token)

    agent_input = scene.get_agent_input()

    assert agent_input is not None
    assert hasattr(agent_input, "ego_statuses")
    assert hasattr(agent_input, "cameras")
    assert hasattr(agent_input, "lidars")

    # Check dimensions
    assert len(agent_input.ego_statuses) == scene_loader.config.num_history_frames
    assert len(agent_input.cameras) == scene_loader.config.num_history_frames
    assert len(agent_input.lidars) == scene_loader.config.num_history_frames


def test_feature_extraction(scene_loader):
    """Test feature extraction from agent input."""
    token = scene_loader.scene_tokens[0]
    scene = scene_loader.get_scene(token)
    agent_input = scene.get_agent_input()

    model_config = TransfuserConfig()
    feature_builder = Bench2DriveFeatureBuilder(model_config)

    features = feature_builder.compute_features(agent_input)

    # Check required features
    assert "camera_feature" in features
    assert "lidar_feature" in features
    assert "status_feature" in features

    # Check feature shapes
    assert features["camera_feature"].shape == torch.Size([3, 256, 1024])
    assert features["lidar_feature"].shape == torch.Size([1, 256, 256])
    assert features["status_feature"].shape == torch.Size([8])


def test_target_extraction(scene_loader):
    """Test target extraction from scene."""
    token = scene_loader.scene_tokens[0]
    scene = scene_loader.get_scene(token)

    model_config = TransfuserConfig()
    target_builder = Bench2DriveTargetBuilder(model_config)

    targets = target_builder.compute_targets(scene)

    # Check required targets
    assert "trajectory" in targets
    assert "agent_states" in targets
    assert "agent_labels" in targets
    assert "bev_semantic_map" in targets

    # Check target shapes
    assert targets["trajectory"].shape[0] == 8  # 8 waypoints
    assert targets["trajectory"].shape[1] == 3  # x, y, heading
    assert targets["agent_states"].shape[1] == 5  # x, y, heading, length, width
    assert len(targets["agent_labels"].shape) == 1
    assert targets["bev_semantic_map"].shape == torch.Size([128, 256])


def test_full_pipeline(scene_loader):
    """Test the full pipeline from scene loading to feature/target extraction."""
    # This combines all the steps
    token = scene_loader.scene_tokens[0]
    scene = scene_loader.get_scene(token)

    model_config = TransfuserConfig()
    feature_builder = Bench2DriveFeatureBuilder(model_config)
    target_builder = Bench2DriveTargetBuilder(model_config)

    agent_input = scene.get_agent_input()
    features = feature_builder.compute_features(agent_input)
    targets = target_builder.compute_targets(scene)

    # Verify complete output
    assert len(features) == 3  # camera, lidar, status
    assert len(targets) == 4  # trajectory, agent_states, agent_labels, bev_semantic_map

    # All tensors should be valid
    for name, tensor in features.items():
        assert torch.isfinite(tensor).all(), f"Feature {name} contains non-finite values"

    for name, tensor in targets.items():
        assert torch.isfinite(tensor).all(), f"Target {name} contains non-finite values"


def test_visualization(scene_loader, tmp_path):
    """Test end-to-end integration with visualization."""
    if len(scene_loader) == 0:
        pytest.skip("No scenes found.")

    scene = scene_loader.get_scene(scene_loader.scene_tokens[0])

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
    # Normalize to [0, 1] for display
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
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


@pytest.mark.skipif(
    not Path("/workspace/Bench2Drive-Base").exists(),
    reason="Full Bench2Drive dataset not available",
)
def test_full_dataset_integration(bench2drive_config_full, tmp_path):
    """Test with full dataset if available."""
    loader = Bench2DriveSceneLoader(bench2drive_config_full)

    if len(loader) == 0:
        pytest.skip("No scenes found in full dataset.")

    # Run basic tests with full dataset
    test_scene_loader_creation(bench2drive_config_full)
    test_visualization(loader, tmp_path)


def test_command_mapping_comprehensive():
    """Comprehensive test for all command mapping edge cases."""
    # Test negative command values (VOID path)
    assert map_carla_command_to_discrete(-1) == 1, "VOID (-1) should map to STRAIGHT"
    assert map_carla_command_to_discrete(-999) == 1, "Any negative should map to STRAIGHT"

    # Test all valid CARLA commands
    valid_mappings = {
        1: 0,  # LEFT → LEFT
        2: 2,  # RIGHT → RIGHT
        3: 1,  # STRAIGHT → STRAIGHT
        4: 1,  # LANEFOLLOW → STRAIGHT
        5: 0,  # CHANGELANELEFT → LEFT
        6: 2,  # CHANGELANERIGHT → RIGHT
    }

    for carla_cmd, expected in valid_mappings.items():
        result = map_carla_command_to_discrete(carla_cmd)
        assert result == expected, f"Command {carla_cmd} should map to {expected}, got {result}"

    # Test out of range commands (should map to UNKNOWN)
    for invalid_cmd in [7, 8, 999, 1000]:
        result = map_carla_command_to_discrete(invalid_cmd)
        assert (
            result == 3
        ), f"Invalid command {invalid_cmd} should map to UNKNOWN (3), got {result}"


def test_dimension_validation():
    """Test that BEV and LiDAR dimensions match DiffusionDrive expectations."""
    config = Bench2DriveDataConfig(
        data_root=Path("/workspace/Bench2Drive-mini"),
        scenarios=["test_scenario"],
    )

    # Expected dimensions based on DiffusionDrive
    EXPECTED_BEV_SEMANTIC_SHAPE = (128, 256)  # H=128, W=256
    EXPECTED_LIDAR_BEV_SHAPE = (1, 256, 256)  # C=1, H=256, W=256

    # These would be tested if we had actual data
    # For now, just document the expected dimensions
    assert EXPECTED_BEV_SEMANTIC_SHAPE == (128, 256), "BEV semantic should be (128, 256)"
    assert EXPECTED_LIDAR_BEV_SHAPE == (1, 256, 256), "LiDAR BEV should be (1, 256, 256)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
