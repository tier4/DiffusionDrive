"""
Tests for TaCarla dataset pipeline.
Covers: Parquet reading, BEV class mapping, ego status, trajectory,
cache round-trip, and B2D regression.

These tests require Docker (for pyarrow) or run inside the Docker container.
Mark with @pytest.mark.docker for tests needing pyarrow.
"""

import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock

# ============================================================
# 1. Constants and BEV class mapping (no Docker needed)
# ============================================================


class TestTaCarlaConstants:
    """Test TaCarla constants are correctly defined."""

    def test_bev_rgb_to_class_mapping(self):
        from navsim.common.tacarla_constants import TACARLA_BEV_RGB_TO_CLASS

        assert len(TACARLA_BEV_RGB_TO_CLASS) == 6
        assert TACARLA_BEV_RGB_TO_CLASS[(0, 0, 0)] == 0  # background
        assert TACARLA_BEV_RGB_TO_CLASS[(0, 0, 255)] == 1  # ego
        assert TACARLA_BEV_RGB_TO_CLASS[(0, 255, 0)] == 2  # lane boundary
        assert TACARLA_BEV_RGB_TO_CLASS[(0, 255, 255)] == 3  # road
        assert TACARLA_BEV_RGB_TO_CLASS[(255, 0, 0)] == 4  # objects
        assert TACARLA_BEV_RGB_TO_CLASS[(255, 255, 255)] == 5  # lane markings

    def test_num_bev_classes(self):
        from navsim.common.tacarla_constants import TACARLA_NUM_BEV_CLASSES
        assert TACARLA_NUM_BEV_CLASSES == 6

    def test_class_to_navsim_mapping(self):
        from navsim.common.tacarla_constants import TACARLA_CLASS_TO_NAVSIM

        assert TACARLA_CLASS_TO_NAVSIM["Car"] == 5
        assert TACARLA_CLASS_TO_NAVSIM["Pedestrian"] == 6
        assert TACARLA_CLASS_TO_NAVSIM["Truck"] == 5

    def test_bev_dimensions(self):
        from navsim.common.tacarla_constants import (
            BEV_SEMANTIC_HEIGHT, BEV_SEMANTIC_WIDTH, BEV_SEMANTIC_RESOLUTION,
        )
        assert BEV_SEMANTIC_HEIGHT == 128
        assert BEV_SEMANTIC_WIDTH == 256
        assert BEV_SEMANTIC_RESOLUTION == 0.25


# ============================================================
# 2. Command mapping (no Docker needed)
# ============================================================


class TestCommandMapping:
    """Test CARLA command to NAVSIM discrete mapping."""

    def test_carla_raw_commands(self):
        from navsim.common.abstract_carla_dataloader import map_carla_command_to_discrete

        assert map_carla_command_to_discrete(1) == 0  # LEFT
        assert map_carla_command_to_discrete(2) == 2  # RIGHT
        assert map_carla_command_to_discrete(3) == 1  # STRAIGHT
        assert map_carla_command_to_discrete(4) == 1  # LANEFOLLOW -> STRAIGHT
        assert map_carla_command_to_discrete(5) == 0  # CHANGELANELEFT -> LEFT
        assert map_carla_command_to_discrete(6) == 2  # CHANGELANERIGHT -> RIGHT
        assert map_carla_command_to_discrete(-1) == 3  # VOID -> UNKNOWN
        assert map_carla_command_to_discrete(99) == 3  # unknown -> UNKNOWN


# ============================================================
# 3. BEV RGB to class conversion (no Docker needed)
# ============================================================


class TestBevConversion:
    """Test BEV label image RGB to class label conversion."""

    def test_rgb_to_class_mapping(self):
        from navsim.common.tacarla_constants import TACARLA_BEV_RGB_TO_CLASS

        # Create a small test BEV image (6x1 pixels, one per class)
        test_img = np.array([
            [[0, 0, 0]],       # background
            [[0, 0, 255]],     # ego
            [[0, 255, 0]],     # lane boundary
            [[0, 255, 255]],   # road
            [[255, 0, 0]],     # objects
            [[255, 255, 255]], # lane markings
        ], dtype=np.uint8)

        # Apply mapping
        class_map = np.zeros((6, 1), dtype=np.uint8)
        for rgb, cls_id in TACARLA_BEV_RGB_TO_CLASS.items():
            mask = np.all(test_img == np.array(rgb, dtype=np.uint8), axis=-1)
            class_map[mask] = cls_id

        expected = np.array([[0], [1], [2], [3], [4], [5]], dtype=np.uint8)
        np.testing.assert_array_equal(class_map, expected)


# ============================================================
# 4. Abstract base class contracts (no Docker needed)
# ============================================================


class TestAbstractBaseClasses:
    """Test that abstract base classes define the correct interface."""

    def test_abstract_scene_loader_interface(self):
        from navsim.common.abstract_carla_dataloader import AbstractCarlaSceneLoader

        # Verify abstract methods exist
        assert hasattr(AbstractCarlaSceneLoader, '_build_scene_index')
        assert hasattr(AbstractCarlaSceneLoader, 'get_scene')
        assert hasattr(AbstractCarlaSceneLoader, 'get_scene_tokens')

    def test_abstract_scene_interface(self):
        from navsim.common.abstract_carla_scene import AbstractCarlaScene

        assert hasattr(AbstractCarlaScene, 'get_agent_input')
        assert hasattr(AbstractCarlaScene, 'get_future_trajectory')
        assert hasattr(AbstractCarlaScene, 'get_agents')
        assert hasattr(AbstractCarlaScene, 'get_bev_semantic_map')


# ============================================================
# 5. Config inheritance (no Docker needed)
# ============================================================


class TestTaCarlaConfig:
    """Test TaCarla config correctly extends B2D config."""

    def test_config_creation(self):
        from navsim.agents.diffusiondrive.tacarla_config import TaCarlaConfig

        config = TaCarlaConfig()
        assert config.dataset_type == "tacarla"
        assert config.num_bev_classes == 6

    def test_normalization_params(self):
        from navsim.agents.diffusiondrive.tacarla_config import TaCarlaConfig

        config = TaCarlaConfig()
        params = config.get_normalization_params()
        assert "x_offset" in params
        assert "x_scale" in params
        assert "y_offset" in params
        assert "y_scale" in params
        assert "heading_offset" in params
        assert "heading_scale" in params

    def test_invalid_dataset_type(self):
        from navsim.agents.diffusiondrive.tacarla_config import TaCarlaConfig

        with pytest.raises(ValueError):
            config = TaCarlaConfig(dataset_type="invalid")
            config.__post_init__()


# ============================================================
# 6. Cache validation (requires cached data on NAS)
# ============================================================


CACHE_TEST_PATH = "/mnt/nas/private_workspace/chenglin/dataset/TaCarla_cache_test"


@pytest.mark.skipif(
    not os.path.exists(CACHE_TEST_PATH),
    reason="TaCarla test cache not available"
)
class TestCacheRoundTrip:
    """Validate cached features/targets have correct shapes."""

    def _load_first_sample(self):
        """Load the first cached sample."""
        import gzip
        import pickle

        cache_path = Path(CACHE_TEST_PATH)
        # Find first token directory
        for log_dir in sorted(cache_path.iterdir()):
            if not log_dir.is_dir():
                continue
            for token_dir in sorted(log_dir.iterdir()):
                feat_file = token_dir / "transfuser_feature.gz"
                tgt_file = token_dir / "transfuser_target.gz"
                if feat_file.exists() and tgt_file.exists():
                    with gzip.open(feat_file, "rb") as f:
                        features = pickle.load(f)
                    with gzip.open(tgt_file, "rb") as f:
                        targets = pickle.load(f)
                    return features, targets
        pytest.skip("No cached samples found")

    def test_feature_shapes(self):
        features, _ = self._load_first_sample()
        assert features["camera_feature"].shape == (3, 256, 1024)
        assert features["lidar_feature"].shape == (1, 256, 256)
        assert features["status_feature"].shape == (8,)

    def test_target_shapes(self):
        _, targets = self._load_first_sample()
        assert targets["trajectory"].shape == (8, 3)
        assert targets["agent_states"].shape == (30, 5)
        assert targets["agent_labels"].shape == (30,)
        assert targets["bev_semantic_map"].ndim == 2  # [H, W]

    def test_no_nan_values(self):
        features, targets = self._load_first_sample()
        for name, tensor in features.items():
            assert not torch.isnan(tensor).any(), f"NaN in feature {name}"
            assert not torch.isinf(tensor).any(), f"Inf in feature {name}"
        for name, tensor in targets.items():
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                assert not torch.isnan(tensor).any(), f"NaN in target {name}"

    def test_trajectory_first_waypoint_near_origin(self):
        _, targets = self._load_first_sample()
        traj = targets["trajectory"]
        # First waypoint should be close to ego (within ~5m)
        assert abs(traj[0, 0].item()) < 10.0, f"First waypoint X={traj[0, 0]:.2f} too far"
        assert abs(traj[0, 1].item()) < 10.0, f"First waypoint Y={traj[0, 1]:.2f} too far"

    def test_bev_class_values(self):
        _, targets = self._load_first_sample()
        bev = targets["bev_semantic_map"]
        unique_vals = torch.unique(bev)
        # All values should be valid TaCarla classes (0-5)
        assert unique_vals.min() >= 0
        assert unique_vals.max() <= 5

    def test_agent_labels_have_valid_agents(self):
        _, targets = self._load_first_sample()
        # At least one valid agent in most scenes
        # (sample was not filtered, so it should have agents)
        assert targets["agent_labels"].any(), "No valid agents in cached sample"


# ============================================================
# 7. B2D regression (no Docker needed for syntax, structure check)
# ============================================================


class TestB2DRegression:
    """Verify B2D still works after abstract base refactoring."""

    def test_b2d_dataloader_inherits_abstract(self):
        from navsim.common.abstract_carla_dataloader import AbstractCarlaSceneLoader
        from navsim.common.bench2drive_dataloader import Bench2DriveSceneLoader

        assert issubclass(Bench2DriveSceneLoader, AbstractCarlaSceneLoader)

    def test_b2d_config_still_works(self):
        from navsim.common.bench2drive_dataloader import Bench2DriveDataConfig
        from navsim.common.abstract_carla_dataloader import AbstractCarlaDataConfig

        config = Bench2DriveDataConfig(data_root=Path("/tmp"))
        assert isinstance(config, AbstractCarlaDataConfig)
        assert config.sampling_rate == 5

    def test_b2d_command_mapping_unchanged(self):
        """Verify B2D's command mapping still works (takes raw CARLA commands)."""
        from navsim.common.bench2drive_dataloader import map_carla_command_to_discrete as b2d_map

        # B2D function internally applies Bench2DriveZoo shift then maps
        # Input: raw CARLA command (1=LEFT, 2=RIGHT, 3=STRAIGHT, 4=LANEFOLLOW)
        assert b2d_map(1) == 0  # LEFT -> LEFT
        assert b2d_map(2) == 2  # RIGHT -> RIGHT
        assert b2d_map(3) == 1  # STRAIGHT -> STRAIGHT
        assert b2d_map(4) == 1  # LANEFOLLOW -> STRAIGHT
