"""
Test Bench2Drive integration with minimal dataset.
Tests data loading, scene creation, and feature extraction.
"""

import os
from pathlib import Path

import pytest
import torch

# Set environment variables before imports
os.environ["BENCH2DRIVE_ROOT"] = "/workspace/Bench2Drive-mini"
os.environ["NAVSIM_EXP_ROOT"] = "/workspace/navsim_workspace/exp"

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder,
)
from navsim.common.bench2drive_dataloader import Bench2DriveConfig, Bench2DriveSceneLoader


@pytest.fixture
def bench2drive_config():
    """Create Bench2Drive configuration for testing."""
    return Bench2DriveConfig(
        data_root=Path("/workspace/Bench2Drive-mini"),
        scenarios=["ConstructionObstacle_Town05_Route68_Weather8"],
        sampling_rate=5,
        num_frames=30,
        num_history_frames=4,
        num_future_frames=26,
        extract_tar=False,
    )


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
    assert hasattr(scene, "frames")
    assert len(scene.frames) == scene_loader.config.num_frames
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
