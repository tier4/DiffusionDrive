"""
Test Bench2Drive BEV cache error handling.
Ensures that missing BEV cache files cause proper errors when cache directory is specified.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from navsim.common.bench2drive_dataloader import Bench2DriveDataConfig, Bench2DriveSceneLoader

# Test data directory - contains real sample data from DynamicObjectCrossing_Town02_Route13_Weather6
TEST_DATA_DIR = Path(__file__).parent / "test_data" / "bench2drive_sample"


class TestBench2DriveBEVCacheError:
    """Test BEV cache error handling."""

    def test_cache_script_fails_with_nonexistent_bev_dir(self):
        """Test that cache script fails immediately with non-existent BEV cache dir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    "python3",
                    "scripts/cache_bench2drive_dataset.py",
                    "--data-root",
                    str(TEST_DATA_DIR),
                    "--cache-path",
                    str(Path(temp_dir) / "cache"),
                    "--bev-cache-dir",
                    "/nonexistent/bev/cache/dir",
                    "--map-dir",
                    str(TEST_DATA_DIR / "maps"),
                    "--scenarios",
                    "DynamicObjectCrossing_Town02_Route13_Weather6",
                    "--num-workers",
                    "1",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode != 0
            assert "BEV cache directory not found" in result.stderr
            assert "generate_bev_cache.py" in result.stderr

    def test_scene_fails_with_missing_bev_files(self):
        """Test that scene loading fails when BEV cache dir exists but files missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty BEV cache directory
            bev_cache_dir = Path(temp_dir) / "empty_bev_cache"
            bev_cache_dir.mkdir(parents=True)

            config = Bench2DriveDataConfig(
                data_root=TEST_DATA_DIR,
                scenarios=["DynamicObjectCrossing_Town02_Route13_Weather6"],
                bev_cache_dir=bev_cache_dir,
                map_dir=TEST_DATA_DIR / "maps",
                sampling_rate=5,  # Standard sampling rate
                num_frames=6,  # We have 30 frames, which gives us 6 scenes
                num_history_frames=2,
                num_future_frames=4,
                extract_tar=False,
            )

            loader = Bench2DriveSceneLoader(config)
            assert len(loader) > 0, "No scenes found in test data"

            scene = loader.get_scene(loader.get_scene_tokens()[0])

            # Attempting to get BEV semantic map should raise FileNotFoundError
            with pytest.raises(FileNotFoundError) as exc:
                scene.get_bev_semantic_map()

            error_msg = str(exc.value)
            assert "BEV cache file not found" in error_msg
            assert "generate_bev_cache.py" in error_msg
            assert "DynamicObjectCrossing_Town02_Route13_Weather6" in error_msg

    def test_cache_script_warns_without_bev_dir(self):
        """Test that cache script warns but continues without BEV cache dir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run with timeout to prevent long processing
            try:
                result = subprocess.run(
                    [
                        "python3",
                        "scripts/cache_bench2drive_dataset.py",
                        "--data-root",
                        str(TEST_DATA_DIR),
                        "--cache-path",
                        str(Path(temp_dir) / "cache"),
                        "--map-dir",
                        str(TEST_DATA_DIR / "maps"),
                        "--scenarios",
                        "DynamicObjectCrossing_Town02_Route13_Weather6",
                        "--num-workers",
                        "1",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,  # Kill after 10 seconds - we just want to see the warning
                )
                # If it completed quickly, check for warning
                assert "No BEV cache directory provided" in result.stderr
            except subprocess.TimeoutExpired as e:
                # Timeout is expected since processing takes time
                # Check if warning was printed before timeout
                stderr = e.stderr
                assert "No BEV cache directory provided" in stderr
                assert "will be generated on-the-fly" in stderr

    def test_scene_succeeds_with_valid_bev_cache(self):
        """Test that scene loads successfully with valid BEV cache."""
        config = Bench2DriveDataConfig(
            data_root=TEST_DATA_DIR,
            scenarios=["DynamicObjectCrossing_Town02_Route13_Weather6"],
            bev_cache_dir=TEST_DATA_DIR / "sample_bev_cache",
            map_dir=TEST_DATA_DIR / "maps",
            sampling_rate=5,
            num_frames=6,  # Just enough frames to include frame 00020
            num_history_frames=2,
            num_future_frames=4,
            extract_tar=False,
        )

        loader = Bench2DriveSceneLoader(config)
        assert len(loader) > 0, "No scenes found in test data"

        scene = loader.get_scene(loader.get_scene_tokens()[0])

        # Should not raise error since we have 00020.npz in sample_bev_cache
        bev_map = scene.get_bev_semantic_map(frame_idx=4)  # Frame 00020 is at index 4
        assert bev_map is not None
        assert bev_map.shape == (128, 256)  # BEV semantic map shape

    def test_no_bev_cache_generates_simple_bev(self):
        """Test that when no BEV cache dir is provided, simple BEV is generated."""
        config = Bench2DriveDataConfig(
            data_root=TEST_DATA_DIR,
            scenarios=["DynamicObjectCrossing_Town02_Route13_Weather6"],
            # No bev_cache_dir specified
            map_dir=None,  # Also no map to force simple BEV generation
            sampling_rate=5,
            num_frames=6,
            num_history_frames=2,
            num_future_frames=4,
            extract_tar=False,
        )

        loader = Bench2DriveSceneLoader(config)
        scene = loader.get_scene(loader.get_scene_tokens()[0])

        # Should generate simple BEV without error
        bev_map = scene.get_bev_semantic_map()
        assert bev_map is not None
        assert bev_map.shape == (128, 256)  # BEV semantic map shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])