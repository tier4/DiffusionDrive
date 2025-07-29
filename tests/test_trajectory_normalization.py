"""
Comprehensive tests for trajectory normalization system.
Tests the TrajectoryNormalizer class with various scenarios and edge cases.
"""

import pytest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent.parent))

from navsim.agents.diffusiondrive.trajectory_normalizer import TrajectoryNormalizer


@pytest.fixture
def navsim_normalizer():
    """Create a NavSim normalizer."""
    return TrajectoryNormalizer(dataset_type="navsim")


@pytest.fixture
def b2d_normalizer():
    """Create a Bench2Drive normalizer."""
    return TrajectoryNormalizer(dataset_type="bench2drive")


@pytest.fixture
def test_trajectories_navsim():
    """Create test trajectories typical for NavSim."""
    return np.array(
        [
            [[0, 0, 0], [5, 0, 0], [10, 0, 0], [15, 0, 0]],  # Straight forward
            [[0, 0, 0], [5, 5, 0.5], [10, 10, 1.0], [15, 15, 1.5]],  # Diagonal
            [[0, 0, 0], [5, -5, -0.5], [10, -10, -1.0], [15, -15, -1.5]],  # Diagonal other way
        ]
    )


@pytest.fixture
def test_trajectories_b2d():
    """Create test trajectories typical for Bench2Drive."""
    return np.array(
        [
            [[-10, -10, -0.05], [-5, -5, -0.02], [0, 0, 0], [5, 5, 0.02]],  # Centered
            [[-15, 0, 0], [-10, 0, 0], [-5, 0, 0], [0, 0, 0]],  # Backward
            [[0, -15, 0], [0, -10, 0], [0, -5, 0], [0, 0, 0]],  # Upward
        ]
    )


class TestTrajectoryNormalizer:
    """Test cases for TrajectoryNormalizer class."""

    def test_normalization_initialization(self, navsim_normalizer, b2d_normalizer):
        """Test proper initialization of normalizers."""
        # Check NavSim parameters
        assert navsim_normalizer.dataset_type == "navsim"
        assert navsim_normalizer.normalization_params is not None
        assert navsim_normalizer.normalization_params["x"]["offset"] == pytest.approx(1.2)
        assert navsim_normalizer.normalization_params["x"]["scale"] == pytest.approx(56.9)

        # Check Bench2Drive parameters
        assert b2d_normalizer.dataset_type == "bench2drive"
        assert b2d_normalizer.normalization_params["x"]["offset"] == pytest.approx(35.0)
        assert b2d_normalizer.normalization_params["x"]["scale"] == pytest.approx(70.0)

    def test_dataset_detection(self):
        """Test automatic dataset type detection from paths."""
        normalizer = TrajectoryNormalizer(dataset_type="auto")

        # Test NavSim detection
        navsim_paths = [
            "/workspace/navsim_workspace/exp/training_cache",
            "/data/navtrain/cache",
            "/experiments/navsim_test",
        ]
        for path in navsim_paths:
            detected = normalizer.detect_dataset_from_path(path)
            assert detected == "navsim", f"Failed to detect NavSim from {path}"

        # Test Bench2Drive detection
        b2d_paths = [
            "/workspace/bench2drive_cache",
            "/data/b2d/training",
            "/experiments/carla_bench2drive",
        ]
        for path in b2d_paths:
            detected = normalizer.detect_dataset_from_path(path)
            assert detected == "bench2drive", f"Failed to detect Bench2Drive from {path}"

        # Test unknown detection
        unknown_path = "/data/custom_dataset/cache"
        detected = normalizer.detect_dataset_from_path(unknown_path)
        assert detected == "custom"

    def test_normalize_denormalize_numpy(
        self, navsim_normalizer, b2d_normalizer, test_trajectories_navsim, test_trajectories_b2d
    ):
        """Test normalization and denormalization with numpy arrays."""
        # Test NavSim normalization
        normalized = navsim_normalizer.normalize(test_trajectories_navsim)

        # Check shape preservation
        assert normalized.shape == test_trajectories_navsim.shape

        # Check range [-1, 1]
        assert np.all(normalized >= -1)
        assert np.all(normalized <= 1)

        # Test denormalization
        denormalized = navsim_normalizer.denormalize(normalized)
        np.testing.assert_allclose(denormalized, test_trajectories_navsim, rtol=1e-5, atol=1e-10)

        # Test Bench2Drive normalization
        normalized_b2d = b2d_normalizer.normalize(test_trajectories_b2d)
        assert np.all(normalized_b2d >= -1)
        assert np.all(normalized_b2d <= 1)

        denormalized_b2d = b2d_normalizer.denormalize(normalized_b2d)
        np.testing.assert_allclose(denormalized_b2d, test_trajectories_b2d, rtol=1e-5, atol=1e-10)

    def test_normalize_denormalize_torch(self, navsim_normalizer, test_trajectories_navsim):
        """Test normalization with PyTorch tensors."""
        # Convert to torch tensors
        torch_trajectories = torch.from_numpy(test_trajectories_navsim).float()

        # Test on CPU
        normalized = navsim_normalizer.normalize(torch_trajectories)
        assert isinstance(normalized, torch.Tensor)
        assert normalized.device == torch_trajectories.device

        # Check range
        assert torch.all(normalized >= -1)
        assert torch.all(normalized <= 1)

        # Test denormalization
        denormalized = navsim_normalizer.denormalize(normalized)
        torch.testing.assert_close(denormalized, torch_trajectories, rtol=1e-5, atol=1e-6)

        # Test on GPU if available
        if torch.cuda.is_available():
            torch_trajectories_gpu = torch_trajectories.cuda()
            normalized_gpu = navsim_normalizer.normalize(torch_trajectories_gpu)
            assert normalized_gpu.device == torch_trajectories_gpu.device
            assert normalized_gpu.is_cuda

    def test_fit_from_data(self):
        """Test computing normalization parameters from data."""
        # Generate synthetic data
        np.random.seed(42)
        synthetic_trajectories = np.random.randn(1000, 8, 3) * 10 + 5

        # Create custom normalizer
        custom_normalizer = TrajectoryNormalizer(dataset_type="custom")
        params = custom_normalizer.fit_from_data(synthetic_trajectories, percentile=95)

        # Check that parameters were computed
        assert "x" in params
        assert "y" in params
        assert "heading" in params

        # Check that parameters make sense
        for dim in ["x", "y", "heading"]:
            assert params[dim]["scale"] > 0
            assert isinstance(params[dim]["offset"], float)
            assert isinstance(params[dim]["scale"], float)

        # Test normalization with fitted parameters
        custom_normalizer.normalization_params = params
        normalized = custom_normalizer.normalize(synthetic_trajectories)

        # Check that most values are in [-1, 1]
        in_range_ratio = float(np.sum(np.abs(normalized) <= 1)) / float(normalized.size)
        assert in_range_ratio > 0.90  # At least 90% should be in range

    def test_edge_cases(self, navsim_normalizer):
        """Test edge cases and error conditions."""
        # Test with extreme values
        extreme_trajectories = np.array(
            [[[1000, 1000, 10], [-1000, -1000, -10]], [[0, 0, 0], [0, 0, 0]]]  # No movement
        )

        # Should still work with clamping
        normalized = navsim_normalizer.normalize(extreme_trajectories, clamp=True)
        assert np.all(normalized >= -1)
        assert np.all(normalized <= 1)

        # Test with single trajectory
        single_traj = np.array([[0, 0, 0], [10, 10, 1]])
        normalized_single = navsim_normalizer.normalize(single_traj)
        assert normalized_single.shape == single_traj.shape

        # Test with custom normalizer (has default params)
        custom_normalizer = TrajectoryNormalizer(dataset_type="custom")
        # Should work with default identity normalization
        custom_normalized = custom_normalizer.normalize(extreme_trajectories)
        # With identity normalization and clamping, extreme values become ±1
        assert np.all(np.abs(custom_normalized) <= 1)

    def test_validation_metrics(self, navsim_normalizer):
        """Test validation metrics computation."""
        # Create trajectories that will have some out-of-range values
        test_trajectories = np.array(
            [
                [[0, 0, 0], [50, 20, 1.5]],  # Within NavSim range
                [[100, 50, 3], [200, 100, 6]],  # Outside NavSim range
            ]
        )

        metrics = navsim_normalizer.validate_normalization(test_trajectories, tolerance=0.01)

        # Check that metrics are computed
        assert "x_min" in metrics
        assert "x_max" in metrics
        assert "overall_out_of_range" in metrics
        assert "has_nan" in metrics
        assert "has_inf" in metrics

        # Check that out-of-range is detected
        assert metrics["overall_out_of_range"] > 0
        assert not metrics["has_nan"]
        assert not metrics["has_inf"]

    def test_config_save_load(self, navsim_normalizer):
        """Test saving and loading normalization configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_norm_config.json"

            # Save config
            navsim_normalizer.save_config(str(config_path))
            assert config_path.exists()

            # Load config manually since config_path loading is not implemented
            with open(config_path, "r") as f:
                loaded_config = json.load(f)

            new_normalizer = TrajectoryNormalizer(dataset_type="custom")
            new_normalizer.normalization_params = loaded_config

            # Check that parameters match
            for dim in ["x", "y", "heading"]:
                assert new_normalizer.normalization_params[dim]["offset"] == pytest.approx(
                    navsim_normalizer.normalization_params[dim]["offset"]
                )
                assert new_normalizer.normalization_params[dim]["scale"] == pytest.approx(
                    navsim_normalizer.normalization_params[dim]["scale"]
                )

    def test_get_config_dict(self, navsim_normalizer):
        """Test getting configuration as dictionary for model config."""
        config_dict = navsim_normalizer.get_config_dict()

        # Check expected keys
        expected_keys = [
            "traj_norm_x_offset",
            "traj_norm_x_scale",
            "traj_norm_y_offset",
            "traj_norm_y_scale",
            "traj_norm_heading_offset",
            "traj_norm_heading_scale",
        ]

        for key in expected_keys:
            assert key in config_dict
            assert isinstance(config_dict[key], float)

    @pytest.mark.parametrize(
        "shape",
        [
            (3, 3),  # 2D array (single trajectory)
            (5, 3, 8, 3),  # 4D array (batch of trajectory sets)
            (3,),  # 1D array (single point)
        ],
    )
    def test_different_shapes(self, navsim_normalizer, shape):
        """Test with different input shapes."""
        if len(shape) == 2:
            traj = np.array([[0, 0, 0], [10, 10, 1], [20, 20, 2]])
        elif len(shape) == 4:
            traj = np.random.randn(*shape)
        else:
            traj = np.array([5, 5, 0.5])

        normalized = navsim_normalizer.normalize(traj)
        assert normalized.shape == traj.shape

    def test_nan_inf_handling(self, navsim_normalizer):
        """Test handling of NaN and Inf values."""
        # Create trajectories with NaN and Inf
        bad_trajectories = np.array(
            [[[0, 0, 0], [np.nan, 10, 0], [20, 20, 0]], [[0, 0, 0], [10, np.inf, 0], [20, 20, 0]]]
        )

        # Should handle gracefully with clamping
        normalized = navsim_normalizer.normalize(bad_trajectories, clamp=True)

        # Check that NaN is propagated but Inf is clamped to 1
        assert np.isnan(normalized[0, 1, 0])
        # When clamping is enabled, Inf becomes 1.0
        assert normalized[1, 1, 1] == 1.0

        # Validation should detect them
        metrics = navsim_normalizer.validate_normalization(bad_trajectories)
        assert metrics["has_nan"]
        assert metrics["has_inf"]


class TestNormalizationIntegration:
    """Integration tests for normalization system."""

    def test_navsim_bench2drive_difference(self):
        """Test that NavSim and Bench2Drive normalizations produce different results."""
        # Create trajectory that's typical for Bench2Drive (centered)
        trajectory = np.array([[[-10, -10, -0.05], [0, 0, 0], [10, 10, 0.05]]])

        navsim_norm = TrajectoryNormalizer(dataset_type="navsim")
        b2d_norm = TrajectoryNormalizer(dataset_type="bench2drive")

        navsim_result = navsim_norm.normalize(trajectory)
        b2d_result = b2d_norm.normalize(trajectory)

        # Results should be different
        assert not np.allclose(navsim_result, b2d_result)

        # Results should be different due to different normalization parameters
        # Can't always guarantee B2D will be closer to 0 for all trajectories
        # Just check they're different
        assert not np.array_equal(navsim_result, b2d_result)

    def test_percentile_robustness(self):
        """Test that percentile-based fitting is robust to outliers."""
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.randn(900, 8, 3) * 10 + 5
        outliers = np.random.randn(100, 8, 3) * 100 + 500
        data_with_outliers = np.vstack([normal_data, outliers])

        normalizer = TrajectoryNormalizer(dataset_type="custom")

        # Fit with different percentiles
        params_95 = normalizer.fit_from_data(data_with_outliers, percentile=95)
        params_99 = normalizer.fit_from_data(data_with_outliers, percentile=99)

        # 95th percentile should be more robust (smaller scale)
        assert params_95["x"]["scale"] < params_99["x"]["scale"]

        # Most normal data should still be in range with 95th percentile
        normalizer.normalization_params = params_95
        normalized_normal = normalizer.normalize(normal_data)
        in_range_ratio = float(np.sum(np.abs(normalized_normal) <= 1)) / float(
            normalized_normal.size
        )
        assert in_range_ratio > 0.90
