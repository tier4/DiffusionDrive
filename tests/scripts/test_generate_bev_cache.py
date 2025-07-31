"""
Tests for BEV cache generation script.
"""

import pytest
import json
import gzip
import numpy as np
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import argparse

# Add the scripts directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


# Mock ray.remote decorator to return the original function
def mock_ray_remote(func):
    """Mock ray.remote decorator that returns the original function."""
    func.remote = func  # Make func.remote() just call func()
    return func


# Patch ray.remote before importing
with patch("ray.remote", mock_ray_remote):
    from generate_bev_cache import (
        setup_logging,
        load_annotation,
        worker_process_frame_ray,
        main,
    )


class TestSetupLogging:
    """Test logging setup."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        logger = setup_logging(verbose=False)
        assert logger.name == "generate_bev_cache"

    def test_setup_logging_verbose(self):
        """Test verbose logging setup."""
        logger = setup_logging(verbose=True)
        assert logger.name == "generate_bev_cache"


class TestLoadAnnotation:
    """Test annotation loading."""

    def test_load_annotation(self, tmp_path):
        """Test loading gzipped JSON annotation."""
        # Create test annotation
        test_anno = {
            "bounding_boxes": [
                {
                    "world2ego": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                }
            ]
        }

        # Save as gzipped JSON
        anno_path = tmp_path / "test.json.gz"
        with gzip.open(anno_path, "wt", encoding="utf-8") as f:
            json.dump(test_anno, f)

        # Load and verify
        loaded = load_annotation(anno_path)
        assert "bounding_boxes" in loaded
        assert len(loaded["bounding_boxes"]) == 1


class TestWorkerProcessFrame:
    """Test Ray worker function."""

    def create_test_annotation(self, tmp_path, scenario_name, frame_num):
        """Create a test annotation file."""
        scenario_dir = tmp_path / scenario_name / "anno"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        anno_data = {
            "bounding_boxes": [
                {
                    "world2ego": [
                        [1.0, 0.0, 0.0, 10.0],
                        [0.0, 1.0, 0.0, 20.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                }
            ]
        }

        anno_path = scenario_dir / f"{frame_num:06d}.json.gz"
        with gzip.open(anno_path, "wt", encoding="utf-8") as f:
            json.dump(anno_data, f)

        return anno_path

    @patch("generate_bev_cache.ray")
    def test_worker_process_frame_success(self, mock_ray, tmp_path):
        """Test successful frame processing."""
        # Create test frame
        frame_path = self.create_test_annotation(tmp_path, "Town01_scenario1", 0)
        output_dir = tmp_path / "output"

        # Create mock MapProcessor
        mock_processor = MagicMock()
        mock_processor.generate_bev.return_value = np.ones((128, 256), dtype=np.float32)

        # Mock ray.get to return the processor
        mock_ray.get.return_value = mock_processor

        # Create map processor refs
        map_processor_refs = {"Town01": "mock_ref"}

        # Process frame
        result = worker_process_frame_ray(
            frame_path, output_dir, map_processor_refs, generate_full=False, overwrite=True
        )

        assert result == True
        assert (output_dir / "Town01_scenario1" / "000000.npz").exists()

        # Verify saved data
        saved_data = np.load(output_dir / "Town01_scenario1" / "000000.npz")
        assert "front_bev" in saved_data
        assert "world2ego" in saved_data
        assert "frame_idx" in saved_data

    @patch("generate_bev_cache.ray")
    def test_worker_process_frame_full_bev(self, mock_ray, tmp_path):
        """Test frame processing with full BEV generation."""
        # Create test frame
        frame_path = self.create_test_annotation(tmp_path, "Town01_scenario1", 0)
        output_dir = tmp_path / "output"

        # Create mock MapProcessor
        mock_processor = MagicMock()

        # Mock generate_full_bev_from_map
        with patch("generate_bev_cache.generate_full_bev_from_map") as mock_full_bev:
            mock_full_bev.return_value = np.ones((256, 256), dtype=np.float32)
            with patch("generate_bev_cache.extract_front_half_bev") as mock_extract:
                mock_extract.return_value = np.ones((128, 256), dtype=np.float32)

                # Mock ray.get to return the processor
                mock_ray.get.return_value = mock_processor

                # Create map processor refs
                map_processor_refs = {"Town01": "mock_ref"}

                # Process frame
                result = worker_process_frame_ray(
                    frame_path, output_dir, map_processor_refs, generate_full=True, overwrite=True
                )

                assert result == True

                # Verify full BEV was saved
                saved_data = np.load(output_dir / "Town01_scenario1" / "000000.npz")
                assert "full_bev" in saved_data
                assert "front_bev" in saved_data

    @patch("generate_bev_cache.ray")
    def test_worker_skip_existing(self, mock_ray, tmp_path):
        """Test skipping existing files when overwrite=False."""
        # Create test frame
        frame_path = self.create_test_annotation(tmp_path, "Town01_scenario1", 0)
        output_dir = tmp_path / "output"

        # Create existing output file
        output_file = output_dir / "Town01_scenario1" / "000000.npz"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_file, dummy=True)

        # Process frame with overwrite=False
        result = worker_process_frame_ray(
            frame_path, output_dir, {}, generate_full=False, overwrite=False
        )

        assert result == False  # Should skip

    @patch("generate_bev_cache.ray")
    def test_worker_no_matching_town(self, mock_ray, tmp_path):
        """Test handling of frame with no matching map processor."""
        # Create test frame with unmatched town
        frame_path = self.create_test_annotation(tmp_path, "UnknownTown_scenario1", 0)
        output_dir = tmp_path / "output"

        # Process frame with no matching town
        result = worker_process_frame_ray(
            frame_path, output_dir, {"Town01": "mock_ref"}, generate_full=False, overwrite=True
        )

        assert result == False


class TestMainFunction:
    """Test main function."""

    def create_test_dataset(self, tmp_path):
        """Create a minimal test dataset structure."""
        data_root = tmp_path / "dataset"
        map_dir = tmp_path / "maps"

        # Create scenario directories
        scenarios = ["Town01_scenario1", "Town01_scenario2", "Town02_scenario1"]
        for scenario in scenarios:
            scenario_dir = data_root / scenario / "anno"
            scenario_dir.mkdir(parents=True, exist_ok=True)

            # Create a few frame files
            for i in range(3):
                anno_data = {
                    "bounding_boxes": [
                        {
                            "world2ego": [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        }
                    ]
                }
                anno_path = scenario_dir / f"{i:06d}.json.gz"
                with gzip.open(anno_path, "wt", encoding="utf-8") as f:
                    json.dump(anno_data, f)

        # Create map files
        map_dir.mkdir(parents=True, exist_ok=True)
        for town in ["Town01", "Town02"]:
            map_data = {
                "road1": {
                    "lane1": [
                        {
                            "Points": [
                                ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                                ((10.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                            ],
                            "Type": "Solid",
                        }
                    ],
                    "Trigger_Volumes": [],
                }
            }
            np_data = np.array(map_data, dtype=object)
            map_path = map_dir / f"{town}_HD_map.npz"
            np.savez(map_path, arr=np_data)

        return data_root, map_dir

    @patch("generate_bev_cache.ray")
    @patch("sys.argv")
    def test_main_basic(self, mock_argv, mock_ray, tmp_path):
        """Test basic main function execution."""
        data_root, map_dir = self.create_test_dataset(tmp_path)
        output_dir = tmp_path / "output"

        # Mock command line arguments
        mock_argv.__getitem__.side_effect = lambda x: [
            "generate_bev_cache.py",
            "--data-root",
            str(data_root),
            "--map-dir",
            str(map_dir),
            "--output-dir",
            str(output_dir),
            "--workers",
            "2",
        ][x]

        # Mock Ray initialization and worker execution
        mock_ray.init.return_value = None
        mock_ray.available_resources.return_value = {"CPU": 2}
        mock_ray.put.return_value = "mock_ref"
        # Fix: ray.get should return a list of results, not a single boolean
        mock_ray.get.return_value = [True, True, True, True, True, True, True, True, True]  # 9 frames
        mock_ray.wait.return_value = (["done_ref"], [])
        mock_ray.shutdown.return_value = None

        # Mock the remote decorator - already done at module level
        # Just ensure worker function has .remote attribute
        worker_process_frame_ray.remote = MagicMock(return_value="task_ref")

        # Run main
        result = main()

        assert result == 0
        assert mock_ray.init.called
        assert mock_ray.shutdown.called

        # Check metadata was saved
        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()

    @patch("generate_bev_cache.ray")
    @patch("sys.argv")
    def test_main_specific_scenarios(self, mock_argv, mock_ray, tmp_path):
        """Test main with specific scenarios."""
        data_root, map_dir = self.create_test_dataset(tmp_path)
        output_dir = tmp_path / "output"

        # Mock command line arguments
        mock_argv.__getitem__.side_effect = lambda x: [
            "generate_bev_cache.py",
            "--data-root",
            str(data_root),
            "--map-dir",
            str(map_dir),
            "--output-dir",
            str(output_dir),
            "--scenarios",
            "Town01_scenario1",
            "--max-frames",
            "2",
            "--full-bev",
            "--verbose",
        ][x]

        # Mock Ray
        mock_ray.init.return_value = None
        mock_ray.available_resources.return_value = {"CPU": 4}
        mock_ray.put.return_value = "mock_ref"
        # Fix: ray.get should return a list of results
        mock_ray.get.return_value = [True, True]  # 2 frames max
        mock_ray.wait.return_value = (["done_ref"], [])
        mock_ray.shutdown.return_value = None

        # Mock the remote decorator
        worker_process_frame_ray.remote = MagicMock(return_value="task_ref")

        # Run main
        result = main()

        assert result == 0

    @patch("sys.argv")
    def test_main_missing_directories(self, mock_argv, tmp_path):
        """Test main with missing directories."""
        # Mock command line arguments with non-existent paths
        mock_argv.__getitem__.side_effect = lambda x: [
            "generate_bev_cache.py",
            "--data-root",
            "/nonexistent/data",
            "--map-dir",
            "/nonexistent/maps",
            "--output-dir",
            str(tmp_path / "output"),
        ][x]

        # Run main - should return error code
        with patch("generate_bev_cache.ray") as mock_ray:
            mock_ray.init.return_value = None
            mock_ray.shutdown.return_value = None
            result = main()

        assert result == 1

    @patch("generate_bev_cache.ray")
    @patch("sys.argv")
    def test_main_no_frames(self, mock_argv, mock_ray, tmp_path):
        """Test main with no frames to process."""
        data_root = tmp_path / "empty_dataset"
        map_dir = tmp_path / "maps"
        output_dir = tmp_path / "output"

        # Create empty directories
        data_root.mkdir(parents=True, exist_ok=True)
        map_dir.mkdir(parents=True, exist_ok=True)

        # Mock command line arguments
        mock_argv.__getitem__.side_effect = lambda x: [
            "generate_bev_cache.py",
            "--data-root",
            str(data_root),
            "--map-dir",
            str(map_dir),
            "--output-dir",
            str(output_dir),
        ][x]

        # Mock Ray
        mock_ray.init.return_value = None
        mock_ray.available_resources.return_value = {"CPU": 4}
        mock_ray.shutdown.return_value = None

        # Run main
        result = main()

        assert result == 0  # Should succeed but process nothing


class TestIntegration:
    """Integration tests."""

    @pytest.mark.skipif(
        not shutil.which("ray"), reason="Ray not installed or not in PATH"
    )
    def test_end_to_end_small(self, tmp_path):
        """Test end-to-end with a small dataset (requires Ray)."""
        # This test would actually run Ray if available
        # Skipped by default to avoid dependencies in CI
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])