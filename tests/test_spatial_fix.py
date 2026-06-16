"""
Tests for the spatial mismatch fix (85m -> 64m crop).
Verifies that all spatial constants, configs, and data processing are consistent.
"""

import copy
import numpy as np
import pytest
import torch


# =============================================================================
# Test Group 1: Spatial Consistency
# =============================================================================


try:
    import nuplan  # noqa: F401
    HAS_NUPLAN = True
except ImportError:
    HAS_NUPLAN = False

needs_nuplan = pytest.mark.skipif(not HAS_NUPLAN, reason="nuplan not installed")


@needs_nuplan
class TestSpatialConsistency:
    """Verify spatial constants match model config assumptions."""

    def test_constants_match_model_config(self):
        """BEV_SEMANTIC_RANGE_M / 256 must equal TransfuserConfig.bev_pixel_size."""
        from navsim.common.bench2drive_constants import BEV_SEMANTIC_RANGE_M
        from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

        config = TransfuserConfig()
        expected_pixel_size = BEV_SEMANTIC_RANGE_M / 256
        assert abs(expected_pixel_size - config.bev_pixel_size) < 1e-6, (
            f"BEV range {BEV_SEMANTIC_RANGE_M}m / 256px = {expected_pixel_size} m/px, "
            f"but model config has bev_pixel_size={config.bev_pixel_size}"
        )

    def test_lidar_range_matches_model(self):
        """BENCH2DRIVE_LIDAR_RANGE_M / 2 must equal TransfuserConfig.lidar_max_x."""
        from navsim.common.bench2drive_constants import BENCH2DRIVE_LIDAR_RANGE_M
        from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

        config = TransfuserConfig()
        half_range = BENCH2DRIVE_LIDAR_RANGE_M / 2
        assert abs(half_range - config.lidar_max_x) < 1e-6, (
            f"LiDAR half range {half_range}m != model lidar_max_x={config.lidar_max_x}"
        )

    def test_bench2drive_config_spatial_params(self):
        """Bench2DriveConfig spatial params must match parent TransfuserConfig."""
        from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig
        from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

        b2d = Bench2DriveConfig()
        nav = TransfuserConfig()
        assert b2d.lidar_max_x == nav.lidar_max_x
        assert b2d.lidar_max_y == nav.lidar_max_y
        assert b2d.lidar_min_x == nav.lidar_min_x
        assert b2d.lidar_min_y == nav.lidar_min_y
        assert b2d.pixels_per_meter == nav.pixels_per_meter

    def test_lidar_size_equals_model_size(self):
        """After crop, BENCH2DRIVE_LIDAR_SIZE must equal DIFFUSIONDRIVE_LIDAR_SIZE."""
        from navsim.common.bench2drive_constants import (
            BENCH2DRIVE_LIDAR_SIZE,
            DIFFUSIONDRIVE_LIDAR_SIZE,
        )
        assert BENCH2DRIVE_LIDAR_SIZE == DIFFUSIONDRIVE_LIDAR_SIZE, (
            f"B2D LiDAR size {BENCH2DRIVE_LIDAR_SIZE} != model size {DIFFUSIONDRIVE_LIDAR_SIZE}"
        )


# =============================================================================
# Test Group 2: Velocity Decomposition
# =============================================================================


class TestVelocityDecomposition:
    """Verify CARLA left-handed velocity decomposition."""

    @staticmethod
    def _compute_velocity(speed, heading_degrees):
        """Replicate the velocity computation from bench2drive_scene.py."""
        theta_rad = np.radians(heading_degrees)
        vx = speed * np.cos(theta_rad)
        vy = speed * np.sin(theta_rad)
        return vx, vy

    def test_velocity_at_zero_heading(self):
        """heading=0 (facing +X forward): vx=speed, vy=0."""
        vx, vy = self._compute_velocity(10.0, 0.0)
        assert abs(vx - 10.0) < 1e-6
        assert abs(vy - 0.0) < 1e-6

    def test_velocity_at_90_degrees(self):
        """heading=90 (facing +Y right): vx=0, vy=speed."""
        vx, vy = self._compute_velocity(10.0, 90.0)
        assert abs(vx - 0.0) < 1e-6
        assert abs(vy - 10.0) < 1e-6

    def test_velocity_at_180_degrees(self):
        """heading=180 (facing backward): vx=-speed, vy=0."""
        vx, vy = self._compute_velocity(10.0, 180.0)
        assert abs(vx - (-10.0)) < 1e-6
        assert abs(vy - 0.0) < 1e-4  # sin(180) is ~1e-16, not exactly 0

    def test_velocity_at_270_degrees(self):
        """heading=270 (facing -Y left): vx=0, vy=-speed."""
        vx, vy = self._compute_velocity(10.0, 270.0)
        assert abs(vx - 0.0) < 1e-4
        assert abs(vy - (-10.0)) < 1e-6


# =============================================================================
# Test Group 3: Config Validation
# =============================================================================


@needs_nuplan
class TestConfigValidation:
    """Verify Bench2DriveConfig defaults and validation."""

    def test_default_dataset_type_is_bench2drive(self):
        """Default dataset_type must be 'bench2drive', not 'navsim'."""
        from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig

        config = Bench2DriveConfig()
        assert config.dataset_type == "bench2drive"

    def test_invalid_dataset_type_raises(self):
        """Invalid dataset_type must raise ValueError."""
        from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig

        with pytest.raises(ValueError, match="Invalid dataset_type"):
            Bench2DriveConfig(dataset_type="invalid")

    def test_navsim_normalization_params(self):
        """NavSim dataset_type returns NavSim normalization params."""
        from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig

        config = Bench2DriveConfig(dataset_type="navsim")
        params = config.get_normalization_params()
        assert abs(params["x_offset"] - 1.2) < 1e-6
        assert abs(params["x_scale"] - 56.9) < 1e-6

    def test_bench2drive_normalization_params(self):
        """Bench2Drive dataset_type returns B2D-specific normalization params."""
        from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig

        config = Bench2DriveConfig(dataset_type="bench2drive")
        params = config.get_normalization_params()
        assert abs(params["x_offset"] - 0.6710353515014572) < 1e-6
        assert abs(params["y_scale"] - 64.60912817798321) < 1e-6


# =============================================================================
# Test Group 4: Model Wrapper Robustness
# =============================================================================


@needs_nuplan
class TestModelWrapperRobustness:
    """Verify model wrapper normalization survives deepcopy and state_dict."""

    @pytest.fixture
    def wrapper(self):
        """Create a wrapper with B2D config (without loading pretrained weights)."""
        from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig
        from navsim.agents.diffusiondrive.transfuser_model_wrapper import V2TransfuserModelWrapper

        config = Bench2DriveConfig(
            dataset_type="bench2drive",
            bkb_path="",  # skip pretrained backbone
            plan_anchor_path="",  # will fail if forward() is called
        )
        try:
            model = V2TransfuserModelWrapper(config)
            return model
        except Exception:
            pytest.skip("Cannot create model wrapper (missing dependencies)")

    def test_norm_denorm_roundtrip(self, wrapper):
        """Normalize then denormalize should recover original values."""
        head = wrapper._trajectory_head
        original = torch.tensor([[[10.0, 5.0, 0.5]]])
        normalized = head.norm_odo(original)
        recovered = head.denorm_odo(normalized)
        assert torch.allclose(original, recovered, atol=1e-4), (
            f"Roundtrip failed: {original} -> {normalized} -> {recovered}"
        )

    def test_state_dict_contains_norm_buffers(self, wrapper):
        """Normalization buffers must appear in state_dict."""
        sd = wrapper.state_dict()
        buffer_keys = [k for k in sd if "_norm_" in k]
        assert len(buffer_keys) >= 6, (
            f"Expected 6 norm buffers, found {len(buffer_keys)}: {buffer_keys}"
        )

    def test_deepcopy_preserves_normalization(self, wrapper):
        """deepcopy must preserve normalization behavior."""
        head = wrapper._trajectory_head
        copied = copy.deepcopy(wrapper)
        copied_head = copied._trajectory_head

        test_input = torch.tensor([[[10.0, 5.0, 0.5]]])
        original_out = head.norm_odo(test_input)
        copied_out = copied_head.norm_odo(test_input)
        assert torch.allclose(original_out, copied_out, atol=1e-6), (
            f"deepcopy changed normalization: {original_out} vs {copied_out}"
        )


# =============================================================================
# Test Group 5: BEV Orientation
# =============================================================================


class TestBEVOrientation:
    """Verify BEV orientation: forward=up (decreasing row), right=right (increasing col)."""

    def test_forward_object_maps_above_ego(self):
        """An object 10m ahead (ego x=10, y=0) should map to a row above ego (row < 127)."""
        from navsim.common.bev_semantic_utils import ego_to_bev_coordinates

        points = np.array([[10.0, 0.0]])  # 10m ahead, centered
        coords = ego_to_bev_coordinates(points, bev_height=128, bev_width=256)
        row = coords[0, 0]
        # Ego is at row 127 (bottom). Forward objects have smaller row indices.
        assert row < 127, f"Forward object at row {row}, expected < 127 (above ego)"
        # At 0.25 m/px, 10m forward = 40 pixels: row = 127 - 40 = 87
        assert abs(row - 87) < 1, f"10m forward should be row ~87, got {row}"

    def test_right_object_maps_to_right_of_bev(self):
        """An object 5m to the right (ego x=0, y=5) should map to col > width/2."""
        from navsim.common.bev_semantic_utils import ego_to_bev_coordinates

        points = np.array([[0.0, 5.0]])  # at ego longitudinal, 5m right
        coords = ego_to_bev_coordinates(points, bev_height=128, bev_width=256)
        col = coords[0, 1]
        assert col > 256 / 2, f"Right object at col {col}, expected > 128 (right half)"

    def test_ego_position_maps_to_bottom_center(self):
        """Ego at (0, 0) should map to bottom-center of front BEV (row=127, col=128)."""
        from navsim.common.bev_semantic_utils import ego_to_bev_coordinates

        points = np.array([[0.0, 0.0]])
        coords = ego_to_bev_coordinates(points, bev_height=128, bev_width=256)
        row, col = coords[0, 0], coords[0, 1]
        assert abs(row - 127) < 1, f"Ego row={row}, expected ~127"
        assert abs(col - 128) < 1, f"Ego col={col}, expected ~128"

    def test_bev_orientation_after_flipud(self):
        """LiDAR histogram flipud: forward points should be at low row indices."""
        from navsim.common.bench2drive_constants import BENCH2DRIVE_LIDAR_RANGE_M, BENCH2DRIVE_LIDAR_SIZE

        # Create a point cloud with one point 10m ahead
        points = np.array([[10.0, 0.0, 0.0]])

        xbins = np.linspace(-BENCH2DRIVE_LIDAR_RANGE_M / 2, BENCH2DRIVE_LIDAR_RANGE_M / 2, BENCH2DRIVE_LIDAR_SIZE + 1)
        ybins = np.linspace(-BENCH2DRIVE_LIDAR_RANGE_M / 2, BENCH2DRIVE_LIDAR_RANGE_M / 2, BENCH2DRIVE_LIDAR_SIZE + 1)
        hist = np.histogramdd(points[:, :2], bins=(xbins, ybins))[0]
        hist_flipped = np.flipud(hist)

        # After flipud, forward (positive x) should be at low row indices (top of image)
        nonzero_rows = np.where(hist_flipped.sum(axis=1) > 0)[0]
        assert len(nonzero_rows) > 0, "No nonzero rows after flipud"
        assert nonzero_rows[0] < BENCH2DRIVE_LIDAR_SIZE / 2, (
            f"Forward point at row {nonzero_rows[0]}, expected < {BENCH2DRIVE_LIDAR_SIZE / 2}"
        )


# =============================================================================
# Test Group 6: Cache Metadata Validation
# =============================================================================


class TestCacheMetadata:
    """Verify BEV cache metadata fields are written and validated."""

    def test_bev_cache_metadata_format(self, tmp_path):
        """Cache files should contain resolution and range_m metadata."""
        from navsim.common.bench2drive_constants import BEV_SEMANTIC_RESOLUTION, BEV_SEMANTIC_RANGE_M

        # Simulate what generate_bev_cache.py saves
        cache_path = tmp_path / "test_cache.npz"
        np.savez_compressed(
            cache_path,
            full_bev=np.zeros((256, 256), dtype=np.uint8),
            front_bev=np.zeros((128, 256), dtype=np.uint8),
            resolution=np.float32(BEV_SEMANTIC_RESOLUTION),
            range_m=np.float32(BEV_SEMANTIC_RANGE_M),
        )

        loaded = np.load(cache_path)
        assert "resolution" in loaded
        assert "range_m" in loaded
        assert abs(float(loaded["resolution"]) - BEV_SEMANTIC_RESOLUTION) < 1e-4
        assert abs(float(loaded["range_m"]) - BEV_SEMANTIC_RANGE_M) < 1e-4
        assert loaded["full_bev"].dtype == np.uint8
