"""
Tests for BEV map generation utilities.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from navsim.common.bev_map_utils import (
    load_map_data,
    extract_lane_points,
    extract_trigger_points,
    transform_points_to_ego,
    ego_to_bev_pixels,
    filter_points_in_range,
    draw_lane_on_bev,
    draw_trigger_on_bev,
    generate_bev_from_map,
    generate_bev_from_map_legacy,
    generate_full_bev_from_map,
    extract_front_half_bev,
    combine_map_and_dynamic,
    MapProcessor,
    LANE_TYPE_TO_BEV_CLASS,
    TRIGGER_TYPE_TO_BEV_CLASS,
)


class TestMapDataLoading:
    """Test map data loading functionality."""

    def test_load_map_data_success(self, tmp_path):
        """Test successful loading of map data."""
        # Create mock NPZ file
        map_data = {"road1": {"lane1": [], "Trigger_Volumes": []}}
        np_data = np.array(map_data, dtype=object)
        map_path = tmp_path / "test_map.npz"
        np.savez(map_path, arr=np_data)

        # Load and verify
        loaded_data = load_map_data(map_path)
        assert isinstance(loaded_data, dict)
        assert "road1" in loaded_data

    def test_load_map_data_file_not_found(self):
        """Test error handling for missing map file."""
        with pytest.raises(FileNotFoundError):
            load_map_data("/nonexistent/path/map.npz")


class TestPointExtraction:
    """Test point extraction functions."""

    def test_extract_lane_points(self):
        """Test lane point extraction."""
        lane_segment = {
            "Points": [
                ((1.0, 2.0, 0.0), (0.0, 0.0, 0.0)),
                ((3.0, 4.0, 0.0), (0.0, 0.0, 0.0)),
            ],
            "Type": "Solid",
        }

        points, lane_type = extract_lane_points(lane_segment)
        assert points.shape == (2, 3)
        assert lane_type == "Solid"
        assert np.allclose(points[0], [1.0, 2.0, 0.0])
        assert np.allclose(points[1], [3.0, 4.0, 0.0])

    def test_extract_trigger_points(self):
        """Test trigger point extraction."""
        trigger = {
            "Points": [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0]],
            "Type": "TrafficLight",
        }

        points, trigger_type = extract_trigger_points(trigger)
        assert points.shape == (3, 3)
        assert trigger_type == "TrafficLight"
        assert np.allclose(points[0], [1.0, 2.0, 0.0])


class TestCoordinateTransformations:
    """Test coordinate transformation functions."""

    def test_transform_points_to_ego(self):
        """Test world to ego coordinate transformation."""
        points = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])

        # Identity transformation
        world2ego = np.eye(4)
        points_ego = transform_points_to_ego(points, world2ego)
        # Y-axis should be flipped due to left-to-right conversion
        assert np.allclose(points_ego[0], [10.0, 0.0, 0.0])
        assert np.allclose(points_ego[1], [0.0, -10.0, 0.0])

        # Test with left_to_right=False
        points_ego_no_flip = transform_points_to_ego(points, world2ego, left_to_right=False)
        assert np.allclose(points_ego_no_flip[1], [0.0, 10.0, 0.0])

    def test_ego_to_bev_pixels(self):
        """Test ego to BEV pixel conversion."""
        points_ego = np.array(
            [
                [0.0, 0.0, 0.0],  # Ego position
                [10.0, 0.0, 0.0],  # 10m forward
                [0.0, 5.0, 0.0],  # 5m right
            ]
        )

        bev_height, bev_width = 128, 256
        resolution = 0.5  # 0.5m per pixel

        pixels = ego_to_bev_pixels(points_ego, bev_height, bev_width, resolution)

        # Ego should be at bottom center
        assert pixels[0, 0] == bev_height - 1  # Bottom row
        assert pixels[0, 1] == bev_width // 2  # Center column

        # Forward point
        assert pixels[1, 0] == bev_height - 1 - 20  # 10m / 0.5m/px = 20px up

        # Right point
        assert pixels[2, 1] == bev_width // 2 + 10  # 5m / 0.5m/px = 10px right


class TestBEVGeneration:
    """Test BEV map generation functions."""

    def create_mock_map_data(self):
        """Create mock map data for testing."""
        return {
            "road1": {
                "lane1": [
                    {
                        "Points": [
                            ((0.0, -5.0, 0.0), (0.0, 0.0, 0.0)),
                            ((10.0, -5.0, 0.0), (0.0, 0.0, 0.0)),
                        ],
                        "Type": "Solid",
                    }
                ],
                "Trigger_Volumes": [
                    {
                        "Points": [
                            [5.0, 0.0, 0.0],
                            [7.0, 0.0, 0.0],
                            [7.0, 2.0, 0.0],
                            [5.0, 2.0, 0.0],
                            [5.0, 0.0, 1.0],
                            [7.0, 0.0, 1.0],
                            [7.0, 2.0, 1.0],
                            [5.0, 2.0, 1.0],
                        ],
                        "Type": "StopSign",
                    }
                ],
            }
        }

    def test_draw_lane_on_bev(self):
        """Test lane drawing on BEV map."""
        bev_map = np.zeros((128, 256), dtype=np.float32)
        lane_points = np.array([[64, 128], [80, 128]])  # Vertical line

        bev_map = draw_lane_on_bev(bev_map, lane_points, "Solid")

        # Check that some pixels were drawn
        assert np.any(bev_map > 0)
        assert np.any(bev_map == LANE_TYPE_TO_BEV_CLASS["Solid"])

    def test_draw_trigger_on_bev(self):
        """Test trigger drawing on BEV map."""
        bev_map = np.zeros((128, 256), dtype=np.float32)
        trigger_points = np.array(
            [[60, 120], [68, 120], [68, 136], [60, 136], [60, 120], [68, 120], [68, 136], [60, 136]]
        )

        bev_map = draw_trigger_on_bev(bev_map, trigger_points, "TrafficLight")

        # Check that polygon was filled
        assert np.any(bev_map > 0)
        assert np.any(bev_map == TRIGGER_TYPE_TO_BEV_CLASS["TrafficLight"])

    def test_generate_bev_from_map_legacy(self):
        """Test legacy BEV generation."""
        map_data = self.create_mock_map_data()
        world2ego = np.eye(4)

        bev_map = generate_bev_from_map_legacy(map_data, world2ego)

        assert bev_map.shape == (128, 256)
        assert bev_map.dtype == np.float32
        # Should have drawn something
        assert np.any(bev_map > 0)

    def test_generate_bev_from_map_with_processor(self):
        """Test BEV generation with MapProcessor."""
        map_data = self.create_mock_map_data()
        world2ego = np.eye(4)

        # Test with dictionary (should create MapProcessor internally)
        bev_map = generate_bev_from_map(map_data, world2ego)
        assert bev_map.shape == (128, 256)
        assert np.any(bev_map > 0)

        # Test with pre-created MapProcessor
        processor = MapProcessor(map_data)
        bev_map2 = generate_bev_from_map(processor, world2ego)
        assert bev_map2.shape == (128, 256)

    def test_generate_bev_legacy_mode(self):
        """Test that legacy mode can be enabled."""
        map_data = self.create_mock_map_data()
        world2ego = np.eye(4)

        bev_map = generate_bev_from_map(map_data, world2ego, use_legacy=True)
        assert bev_map.shape == (128, 256)

    def test_generate_full_bev(self):
        """Test full 360-degree BEV generation."""
        map_data = self.create_mock_map_data()
        world2ego = np.eye(4)

        full_bev = generate_full_bev_from_map(map_data, world2ego)
        assert full_bev.shape == (256, 256)

    def test_extract_front_half_bev(self):
        """Test front half extraction."""
        full_bev = np.ones((256, 256))
        full_bev[:128, :] = 2.0  # Different value for front half

        front_bev = extract_front_half_bev(full_bev)
        assert front_bev.shape == (128, 256)
        assert np.all(front_bev == 2.0)

    def test_combine_map_and_dynamic(self):
        """Test combining static and dynamic BEV maps."""
        static_bev = np.ones((128, 256)) * 1.0
        dynamic_bev = np.zeros((128, 256))
        dynamic_bev[50:60, 100:110] = 5.0  # Add a vehicle

        combined = combine_map_and_dynamic(static_bev, dynamic_bev)

        # Static areas should remain
        assert combined[0, 0] == 1.0
        # Dynamic objects should override
        assert combined[55, 105] == 5.0


class TestMapProcessor:
    """Test MapProcessor class."""

    def test_map_processor_initialization(self):
        """Test MapProcessor initialization."""
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

        processor = MapProcessor(map_data)
        assert processor.lane_kdtree is not None
        assert len(processor._flat_lanes) == 1

    def test_map_processor_generate_bev(self):
        """Test MapProcessor BEV generation."""
        map_data = {
            "road1": {
                "lane1": [
                    {
                        "Points": [
                            ((0.0, -5.0, 0.0), (0.0, 0.0, 0.0)),
                            ((20.0, -5.0, 0.0), (0.0, 0.0, 0.0)),
                        ],
                        "Type": "Solid",
                    }
                ],
                "Trigger_Volumes": [],
            }
        }

        processor = MapProcessor(map_data)
        world2ego = np.eye(4)

        bev_map = processor.generate_bev(world2ego)
        assert bev_map.shape == (128, 256)
        assert np.any(bev_map > 0)

    def test_map_processor_empty_map(self):
        """Test MapProcessor with empty map data."""
        processor = MapProcessor({})
        assert processor.lane_kdtree is None
        assert processor.trigger_kdtree is None

        bev_map = processor.generate_bev(np.eye(4))
        assert np.all(bev_map == 0)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_filter_points_in_range(self):
        """Test point filtering by range."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # In range
                [100.0, 0.0, 0.0],  # Out of range (forward)
                [0.0, 200.0, 0.0],  # Out of range (lateral)
            ]
        )

        filtered, mask = filter_points_in_range(points, 128, 256, 0.5)

        assert len(filtered) == 1
        assert mask[0] == True
        assert mask[1] == False
        assert mask[2] == False

    def test_empty_lane_segments(self):
        """Test handling of empty lane segments."""
        map_data = {"road1": {"lane1": [{"Points": [], "Type": "Solid"}]}}

        bev_map = generate_bev_from_map(map_data, np.eye(4))
        assert np.all(bev_map == 0)

    def test_invalid_lane_type(self):
        """Test handling of unknown lane types."""
        bev_map = np.zeros((128, 256), dtype=np.float32)
        lane_points = np.array([[64, 128], [80, 128]])

        bev_map = draw_lane_on_bev(bev_map, lane_points, "UnknownType")
        # Should default to road class (1)
        assert np.any(bev_map == 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])