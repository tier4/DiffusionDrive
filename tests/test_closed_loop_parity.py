"""Offline parity: adapter-built features must equal scene-loader-built features.

Proves that dataset frames replayed through the closed-loop input adapter
(`build_agent_input`) produce feature tensors identical to the training pipeline
(`Bench2DriveScene.get_agent_input` -> `Bench2DriveFeatureBuilder`).

The bundled sample scenario (`tests/test_data/bench2drive_sample`) ships with raw
annotations only -- no camera JPEGs or LiDAR LAZ files -- so it cannot exercise the
camera/lidar feature path. The compass-formula check runs hermetically on it. The
full feature-parity check runs against real Bench2Drive-Base data when mounted
(the Docker env mounts /mnt/nvme1); it skips gracefully otherwise, and the CLI
harness `run_parity_check.py` covers the same ground at scale.
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import Bench2DriveFeatureBuilder
from navsim.common.bench2drive_dataloader import (
    Bench2DriveDataConfig,
    Bench2DriveSceneLoader,
)
from navsim.evaluation.carla_closed_loop.input_adapter import (
    build_agent_input,
    world_yaw_from_compass,
)

from scripts.evaluation.closed_loop.run_parity_check import reconstruct_carla_inputs

REAL_DATA_ROOT = Path("/mnt/nvme1/dataset/Bench2Drive-Base")
REAL_SCENARIO = "Accident_Town03_Route101_Weather23"


class TestFeatureParity:
    @pytest.fixture
    def sample_scene(self, sample_config):
        """Scene from the bundled (annotation-only) sample scenario."""
        loader = Bench2DriveSceneLoader(sample_config)
        tokens = loader.get_scene_tokens()
        assert tokens, "sample scenario has no tokens"
        return loader.get_scene(tokens[0])

    @pytest.fixture
    def real_scene(self):
        """Scene from a real Bench2Drive-Base scenario (needs sensor data)."""
        if not (REAL_DATA_ROOT / REAL_SCENARIO / "camera").is_dir():
            pytest.skip(
                f"Real Bench2Drive data not available at {REAL_DATA_ROOT}; "
                "feature parity is covered by the run_parity_check.py CLI."
            )
        config = Bench2DriveDataConfig(
            data_root=REAL_DATA_ROOT,
            scenarios=[REAL_SCENARIO],
            sampling_rate=5,
            num_frames=6,
            num_history_frames=2,
            num_future_frames=4,
            extract_tar=False,
            map_dir=None,
            bev_cache_dir=None,
        )
        loader = Bench2DriveSceneLoader(config)
        tokens = loader.get_scene_tokens()
        assert tokens, "real scenario has no tokens"
        return loader.get_scene(tokens[0])

    def test_features_match_training_pipeline(self, real_scene):
        config = Bench2DriveConfig()
        builder = Bench2DriveFeatureBuilder(config)

        gt_input = real_scene.get_agent_input()
        gt_features = builder.compute_features(gt_input)

        kwargs = reconstruct_carla_inputs(real_scene)
        cl_input = build_agent_input(**kwargs)
        cl_features = builder.compute_features(cl_input)

        for name in ("camera_feature", "lidar_feature", "status_feature"):
            max_diff = (gt_features[name] - cl_features[name]).abs().max().item()
            assert torch.allclose(
                gt_features[name], cl_features[name], atol=1e-5
            ), f"{name} diverges: max abs diff {max_diff}"

    def test_compass_formula_matches_recorded_yaw(self, sample_scene):
        """The raw anno stores the CARLA IMU compass in `theta`; the ego-box
        rotation yaw is the world yaw. Validate the adapter's compass->yaw
        formula against this recorded pair (hermetic, annotation-only)."""
        frame_idx = sample_scene.history_frames
        anno = sample_scene.get_raw_annotation(frame_idx)
        assert "theta" in anno, "sample anno unexpectedly lacks compass field 'theta'"

        recorded_compass = float(anno["theta"])
        ego_box = next(
            b for b in anno["bounding_boxes"] if b["class"] == "ego_vehicle"
        )
        world_yaw = math.radians(ego_box["rotation"][2])
        world_yaw = math.atan2(math.sin(world_yaw), math.cos(world_yaw))

        assert world_yaw_from_compass(recorded_compass) == pytest.approx(
            world_yaw, abs=1e-3
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
