"""
Test Bench2Drive dataset caching functionality.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder,
)
from navsim.common.bench2drive_dataloader import Bench2DriveDataConfig, Bench2DriveSceneLoader


class TestBench2DriveCaching:
    """Test caching functionality for Bench2Drive dataset."""

    @pytest.fixture
    def scene_loader(self, sample_config):
        """Create scene loader."""
        return Bench2DriveSceneLoader(sample_config)

    @pytest.fixture
    def model_config(self):
        """Create model configuration."""
        return TransfuserConfig()

    def test_scene_loader_creation(self, scene_loader):
        """Test scene loader is created with correct number of scenes."""
        assert len(scene_loader) > 0
        assert len(scene_loader.get_scene_tokens()) == len(scene_loader)

    def test_feature_extraction(self, scene_loader, model_config):
        """Test feature extraction for a scene."""
        # Get first scene
        token = scene_loader.get_scene_tokens()[0]
        scene = scene_loader.get_scene(token)

        # Get agent input from current frame (NavSim convention)
        agent_input = scene.get_agent_input(4)  # num_history_frames - 1

        # Create feature builder
        feature_builder = Bench2DriveFeatureBuilder(model_config)

        # Extract features
        features = feature_builder.compute_features(agent_input)

        # Verify features
        assert "camera_feature" in features
        assert "lidar_feature" in features
        assert "status_feature" in features

        # Check shapes
        assert features["camera_feature"].shape == (3, 256, 1024)
        assert features["lidar_feature"].shape == (1, 256, 256)
        assert features["status_feature"].shape == (8,)

    def test_target_extraction(self, scene_loader, model_config):
        """Test target extraction for a scene."""
        # Get first scene
        token = scene_loader.get_scene_tokens()[0]
        scene = scene_loader.get_scene(token)

        # Create target builder
        target_builder = Bench2DriveTargetBuilder(model_config)

        # Extract targets
        targets = target_builder.compute_targets(scene)

        # Verify targets
        assert "trajectory" in targets
        assert "agent_states" in targets
        assert "agent_labels" in targets
        assert "bev_semantic_map" in targets

        # Check shapes
        assert targets["trajectory"].shape == (8, 3)
        assert targets["agent_states"].shape == (30, 5)
        assert targets["agent_labels"].shape == (30,)
        assert targets["bev_semantic_map"].shape == (128, 256)

    def test_cache_save_load(self, scene_loader, model_config):
        """Test saving and loading cached data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir)

            # Get first scene
            token = scene_loader.get_scene_tokens()[0]
            scene = scene_loader.get_scene(token)

            # Get agent input from current frame (NavSim convention)
            agent_input = scene.get_agent_input(4)  # num_history_frames - 1

            # Create builders
            feature_builder = Bench2DriveFeatureBuilder(model_config)
            target_builder = Bench2DriveTargetBuilder(model_config)

            # Extract features and targets
            features = feature_builder.compute_features(agent_input)
            targets = target_builder.compute_targets(scene)

            # Create cache entry
            cache_entry = {
                "token": token,
                "features": features,
                "targets": targets,
            }

            # Save cache
            cache_file = cache_path / f"{token}.pt"
            torch.save(cache_entry, cache_file)

            # Load cache
            loaded_entry = torch.load(cache_file)

            # Verify loaded data
            assert loaded_entry["token"] == token
            assert torch.allclose(
                loaded_entry["features"]["camera_feature"], features["camera_feature"]
            )
            assert torch.allclose(loaded_entry["targets"]["trajectory"], targets["trajectory"])

    def test_bev_cache_loading(self, sample_config_with_bev_cache, model_config):
        """Test that BEV cache is properly loaded."""
        # Create scene loader with BEV cache config
        scene_loader = Bench2DriveSceneLoader(sample_config_with_bev_cache)
        
        # Get first scene
        token = scene_loader.get_scene_tokens()[0]
        scene = scene_loader.get_scene(token)

        # Get BEV map for frame 4 (frame 00020) - we have cache for this
        bev_map = scene.get_bev_semantic_map(4)

        # Verify BEV map
        assert bev_map is not None
        assert bev_map.shape == (128, 256)
        assert bev_map.dtype == torch.float32

        # Check that it has meaningful content (not all zeros)
        assert bev_map.sum() > 0
        # Note: The cached BEV might be simpler, so we don't check for multiple classes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
