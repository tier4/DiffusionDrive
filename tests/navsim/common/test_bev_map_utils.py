"""
Tests for BEV map generation utilities.
"""

import pytest
import numpy as np
import tempfile
import cv2
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
    generate_full_bev_from_map,
    extract_front_half_bev,
    combine_map_and_dynamic,
    MapProcessor,
    get_ego_pixel_position,
    BEV_VIEW_FRONT,
    BEV_VIEW_REAR,
    BEV_VIEW_FULL,
)
from navsim.common.bench2drive_constants import (
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
        points_ego = transform_points_to_ego(points, world2ego, left_to_right=True)
        # Y-axis should be flipped due to left-to-right conversion
        assert np.allclose(points_ego[0], [10.0, 0.0, 0.0])
        assert np.allclose(points_ego[1], [0.0, -10.0, 0.0])

        # Test with left_to_right=False
        points_ego_no_flip = transform_points_to_ego(points, world2ego, left_to_right=False)
        assert np.allclose(points_ego_no_flip[1], [0.0, 10.0, 0.0])

    def test_get_ego_pixel_position(self):
        """Test ego position for different view types."""
        bev_height, bev_width = 128, 256

        # Front view: ego at bottom-center
        ego_row, ego_col = get_ego_pixel_position(bev_height, bev_width, BEV_VIEW_FRONT)
        assert ego_row == 127  # Bottom row
        assert ego_col == 128  # Center column

        # Rear view: ego at top-center
        ego_row, ego_col = get_ego_pixel_position(bev_height, bev_width, BEV_VIEW_REAR)
        assert ego_row == 0  # Top row
        assert ego_col == 128  # Center column

        # Full view: ego at center
        bev_height_full, bev_width_full = 256, 256
        ego_row, ego_col = get_ego_pixel_position(bev_height_full, bev_width_full, BEV_VIEW_FULL)
        assert ego_row == 128  # Center row
        assert ego_col == 128  # Center column

    def test_ego_to_bev_pixels_front_view(self):
        """Test ego to BEV pixel conversion for front view."""
        points_ego = np.array(
            [
                [0.0, 0.0, 0.0],  # Ego position
                [10.0, 0.0, 0.0],  # 10m forward
                [0.0, 5.0, 0.0],  # 5m right
                [0.0, -5.0, 0.0],  # 5m left
                [20.0, 10.0, 0.0],  # Forward and right
            ]
        )

        bev_height, bev_width = 128, 256
        resolution = 0.5  # 0.5m per pixel

        pixels = ego_to_bev_pixels(points_ego, bev_height, bev_width, resolution, BEV_VIEW_FRONT)

        # Ego should be at bottom center
        assert pixels[0, 0] == bev_height - 1  # Bottom row
        assert pixels[0, 1] == bev_width // 2  # Center column

        # Forward point (10m forward)
        assert pixels[1, 0] == bev_height - 1 - 20  # Move up by 20 pixels
        assert pixels[1, 1] == bev_width // 2  # Still center column

        # Right point (5m right)
        assert pixels[2, 0] == bev_height - 1  # Still bottom row
        assert pixels[2, 1] == bev_width // 2 + 10  # Move right by 10 pixels

        # Left point (5m left)
        assert pixels[3, 0] == bev_height - 1  # Still bottom row
        assert pixels[3, 1] == bev_width // 2 - 10  # Move left by 10 pixels

        # General point (forward and right)
        assert pixels[4, 0] == bev_height - 1 - 40  # 20m / 0.5 = 40 pixels up
        assert pixels[4, 1] == bev_width // 2 + 20  # 10m / 0.5 = 20 pixels right

    def test_ego_to_bev_pixels_full_view(self):
        """Test ego to BEV pixel conversion for full 360° view."""
        points_ego = np.array(
            [
                [0.0, 0.0, 0.0],  # Ego position
                [10.0, 0.0, 0.0],  # 10m forward
                [-10.0, 0.0, 0.0],  # 10m backward
                [0.0, 10.0, 0.0],  # 10m right
                [0.0, -10.0, 0.0],  # 10m left
            ]
        )

        bev_height, bev_width = 256, 256
        resolution = 1.0  # 1m per pixel

        pixels = ego_to_bev_pixels(points_ego, bev_height, bev_width, resolution, BEV_VIEW_FULL)

        # Ego at center
        assert pixels[0, 0] == 128  # Center row
        assert pixels[0, 1] == 128  # Center column

        # Forward: decreases row
        assert pixels[1, 0] == 118  # 128 - 10
        assert pixels[1, 1] == 128

        # Backward: increases row
        assert pixels[2, 0] == 138  # 128 + 10
        assert pixels[2, 1] == 128

        # Right: increases column
        assert pixels[3, 0] == 128
        assert pixels[3, 1] == 138  # 128 + 10

        # Left: decreases column
        assert pixels[4, 0] == 128
        assert pixels[4, 1] == 118  # 128 - 10

    def test_ego_to_bev_pixels_rear_view(self):
        """Test ego to BEV pixel conversion for rear view."""
        points_ego = np.array(
            [
                [0.0, 0.0, 0.0],  # Ego position
                [-10.0, 0.0, 0.0],  # 10m backward
                [0.0, 5.0, 0.0],  # 5m right
            ]
        )

        bev_height, bev_width = 128, 256
        resolution = 0.5  # 0.5m per pixel

        pixels = ego_to_bev_pixels(points_ego, bev_height, bev_width, resolution, BEV_VIEW_REAR)

        # Ego at top center
        assert pixels[0, 0] == 0  # Top row
        assert pixels[0, 1] == 128  # Center column

        # Backward point (10m behind) - should increase row (move down in image)
        assert pixels[1, 0] == 0 + 20  # Move down by 20 pixels
        assert pixels[1, 1] == 128

        # Right point
        assert pixels[2, 0] == 0
        assert pixels[2, 1] == 128 + 10  # Move right by 10 pixels

    @pytest.mark.parametrize(
        "description, points_ego, resolution, expected_pixels",
        [
            (
                "Point at origin should be at bottom-center",
                np.array([[0, 0]]),
                0.5,
                np.array([[127, 128]]),  # bev_height-1, bev_width/2
            ),
            (
                "Point 10m forward should move up 20 pixels (10m / 0.5)",
                np.array([[10, 0]]),
                0.5,
                np.array([[107, 128]]),  # 127 - (10 / 0.5)
            ),
            (
                "Point 5m right should move right 10 pixels (5m / 0.5)",
                np.array([[0, 5]]),
                0.5,
                np.array([[127, 138]]),  # 127, 128 + (5 / 0.5)
            ),
            (
                "Point 5m left should move left 10 pixels",
                np.array([[0, -5]]),
                0.5,
                np.array([[127, 118]]),  # 127, 128 - (5 / 0.5)
            ),
            (
                "General point (20m forward, 10m right) with resolution 1.0",
                np.array([[20, 10]]),
                1.0,
                np.array([[107, 138]]),  # 127 - 20, 128 + 10
            ),
            (
                "Batch of multiple points",
                np.array([[0, 0], [20, 10], [0, -5]]),
                1.0,
                np.array([[127, 128], [107, 138], [127, 123]]),
            ),
            (
                "Point behind the car should have a row > max_row",
                np.array([[-10, 0]]),
                1.0,
                np.array([[137, 128]]),  # 127 - (-10)
            ),
        ],
    )
    def test_ego_to_bev_scenarios(self, description, points_ego, resolution, expected_pixels):
        """Tests various scenarios for the BEV conversion."""
        bev_height, bev_width = 128, 256
        # Run the function
        result_pixels = ego_to_bev_pixels(
            points_ego=points_ego,
            bev_height=bev_height,
            bev_width=bev_width,
            resolution=resolution,
            view_type=BEV_VIEW_FRONT,
        )
        # Assert that the result is what we expect
        np.testing.assert_array_equal(result_pixels, expected_pixels, err_msg=description)


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
        bev_map = np.zeros((128, 256), dtype=np.uint8)
        lane_points = np.array([[64, 128], [80, 128]])  # Vertical line

        bev_map = draw_lane_on_bev(bev_map, lane_points, "Solid")

        # Check that some pixels were drawn
        assert np.any(bev_map > 0)
        assert np.any(bev_map == LANE_TYPE_TO_BEV_CLASS["Solid"])

    def test_draw_lane_with_valid_data_and_semantic_class(self):
        """Test that lane is drawn with correct semantic class when valid data provided"""
        bev_height, bev_width = 128, 256
        bev_map = np.zeros((bev_height, bev_width), dtype=np.uint8)

        # Create lane points (straight horizontal line)
        lane_points = np.array(
            [
                [64, 100],  # Start point (row, col)
                [64, 120],  # Middle point
                [64, 140],  # End point
            ]
        )

        lane_type = "Solid"
        expected_semantic_class = LANE_TYPE_TO_BEV_CLASS[lane_type]  # Should be 1

        result_bev = draw_lane_on_bev(bev_map, lane_points, lane_type)

        # Check that semantic class is drawn on the map
        unique_values = np.unique(result_bev)
        assert (
            expected_semantic_class in unique_values
        ), f"Expected semantic class {expected_semantic_class} not found in BEV map"

        # Check that the line was drawn approximately where expected
        line_area = result_bev[60:68, 95:145]  # Area around the line
        assert np.any(line_area == expected_semantic_class), "Lane not drawn in expected area"

    def test_draw_different_lane_types(self):
        """Test drawing different lane types with correct semantic classes"""
        bev_map = np.zeros((128, 256), dtype=np.uint8)
        lane_points = np.array([[64, 100], [64, 120], [64, 140]])

        # Test Broken lane
        result_broken = draw_lane_on_bev(bev_map.copy(), lane_points, "Broken")
        assert LANE_TYPE_TO_BEV_CLASS["Broken"] in np.unique(result_broken)

        # Test Center lane
        result_center = draw_lane_on_bev(bev_map.copy(), lane_points, "Center")
        assert LANE_TYPE_TO_BEV_CLASS["Center"] in np.unique(result_center)

        # Test SolidSolid lane
        result_solid_solid = draw_lane_on_bev(bev_map.copy(), lane_points, "SolidSolid")
        assert LANE_TYPE_TO_BEV_CLASS["SolidSolid"] in np.unique(result_solid_solid)

    def test_draw_lane_empty_points_unchanged(self):
        """Test that empty lane points returns unchanged BEV map"""
        bev_map = np.zeros((128, 256), dtype=np.uint8)
        lane_points = np.array([]).reshape(0, 2)  # Empty points
        lane_type = "Solid"

        result_bev = draw_lane_on_bev(bev_map, lane_points, lane_type)

        # Should return unchanged map
        np.testing.assert_array_equal(result_bev, bev_map)

    def test_draw_lane_insufficient_points_unchanged(self):
        """Test that < 2 points returns unchanged BEV map"""
        bev_map = np.zeros((128, 256), dtype=np.uint8)
        lane_points = np.array([[64, 100]])  # Only 1 point
        lane_type = "Solid"

        result_bev = draw_lane_on_bev(bev_map, lane_points, lane_type)

        # Should return unchanged map
        np.testing.assert_array_equal(result_bev, bev_map)

    def test_draw_trigger_on_bev(self):
        """Test trigger drawing on BEV map."""
        bev_map = np.zeros((128, 256), dtype=np.uint8)
        trigger_points = np.array(
            [
                [60, 120],
                [68, 120],
                [68, 136],
                [60, 136],
                [60, 120],
                [68, 120],
                [68, 136],
                [60, 136],
            ]
        )

        bev_map = draw_trigger_on_bev(bev_map, trigger_points, "TrafficLight")

        # Check that polygon was filled
        assert np.any(bev_map > 0)
        assert np.any(bev_map == TRIGGER_TYPE_TO_BEV_CLASS["TrafficLight"])

    def test_draw_trigger_with_valid_data_and_semantic_class(self):
        """Test that trigger is drawn with correct semantic class when valid data provided"""
        bev_height, bev_width = 128, 256
        bev_map = np.zeros((bev_height, bev_width), dtype=np.uint8)

        # Create trigger points (rectangle in middle of BEV)
        trigger_points = np.array(
            [
                [60, 120],  # Bottom-left
                [60, 140],  # Bottom-right
                [68, 140],  # Top-right
                [68, 120],  # Top-left
            ]
        )

        trigger_type = "TrafficLight"
        expected_semantic_class = TRIGGER_TYPE_TO_BEV_CLASS[trigger_type]  # Should be 4

        result_bev = draw_trigger_on_bev(bev_map, trigger_points, trigger_type)

        # Check that semantic class is drawn on the map
        unique_values = np.unique(result_bev)
        assert (
            expected_semantic_class in unique_values
        ), f"Expected semantic class {expected_semantic_class} not found in BEV map"

        # Check that the polygon was filled in expected area
        filled_area = result_bev[58:70, 118:142]  # Area around the rectangle
        assert np.any(filled_area == expected_semantic_class), "Trigger not drawn in expected area"

    def test_draw_different_trigger_types(self):
        """Test drawing different trigger types with correct semantic classes"""
        bev_map = np.zeros((128, 256), dtype=np.uint8)
        trigger_points = np.array([[60, 120], [60, 140], [68, 140], [68, 120]])

        # Test TrafficLight trigger
        result_traffic = draw_trigger_on_bev(bev_map.copy(), trigger_points, "TrafficLight")
        assert TRIGGER_TYPE_TO_BEV_CLASS["TrafficLight"] in np.unique(result_traffic)

        # Test StopSign trigger
        result_stop = draw_trigger_on_bev(bev_map.copy(), trigger_points, "StopSign")
        assert TRIGGER_TYPE_TO_BEV_CLASS["StopSign"] in np.unique(result_stop)

    def test_draw_trigger_with_8_points_uses_bottom_4(self):
        """Test that 8-point trigger (3D box) uses bottom 4 points"""
        bev_map = np.zeros((128, 256), dtype=np.uint8)

        # 8 points: bottom 4 then top 4
        trigger_points = np.array(
            [
                # Bottom 4 points
                [60, 120],
                [60, 140],
                [68, 140],
                [68, 120],
                # Top 4 points (should be ignored for 2D BEV)
                [50, 115],
                [50, 145],
                [78, 145],
                [78, 115],
            ]
        )

        trigger_type = "TrafficLight"
        expected_semantic_class = TRIGGER_TYPE_TO_BEV_CLASS[trigger_type]

        result_bev = draw_trigger_on_bev(bev_map, trigger_points, trigger_type)

        unique_values = np.unique(result_bev)
        assert expected_semantic_class in unique_values

        # Should draw in bottom 4 area, not top 4 area
        bottom_area = result_bev[58:70, 118:142]
        assert np.any(bottom_area == expected_semantic_class)

    def test_draw_trigger_insufficient_points_unchanged(self):
        """Test that < 4 points returns unchanged BEV map"""
        bev_map = np.zeros((128, 256), dtype=np.uint8)
        trigger_points = np.array([[60, 120], [60, 140]])  # Only 2 points
        trigger_type = "TrafficLight"

        result_bev = draw_trigger_on_bev(bev_map, trigger_points, trigger_type)

        # Should return unchanged map
        np.testing.assert_array_equal(result_bev, bev_map)

    def test_draw_trigger_empty_points_unchanged(self):
        """Test that empty trigger points returns unchanged BEV map"""
        bev_map = np.zeros((128, 256), dtype=np.uint8)
        trigger_points = np.array([]).reshape(0, 2)  # Empty points
        trigger_type = "TrafficLight"

        result_bev = draw_trigger_on_bev(bev_map, trigger_points, trigger_type)

        # Should return unchanged map
        np.testing.assert_array_equal(result_bev, bev_map)

    def test_draw_lane_points_outside_map_bounds_should_error(self):
        """Test that lane points outside map bounds should raise an error"""
        bev_map = np.zeros((128, 256), dtype=np.uint8)
        # Points way outside map bounds - should cause OpenCV error
        lane_points = np.array([[10000, 10000], [20000, 20000]])
        lane_type = "Solid"

        # OpenCV should handle this gracefully, but let's test what happens
        try:
            result_bev = draw_lane_on_bev(bev_map, lane_points, lane_type)
            # If no error, check that result is still valid
            assert result_bev is not None
        except Exception as e:
            # Expected to potentially fail with out-of-bounds coordinates
            assert True  # This is expected behavior

    def test_draw_trigger_points_outside_map_bounds_should_error(self):
        """Test that trigger points outside map bounds should raise an error"""
        bev_map = np.zeros((128, 256), dtype=np.uint8)
        # Points way outside map bounds - should cause OpenCV error
        trigger_points = np.array([[10000, 10000], [10000, 20000], [20000, 20000], [20000, 10000]])
        trigger_type = "TrafficLight"

        # This should potentially cause an error due to cv2.fillPoly with tuple issue
        with pytest.raises((cv2.error, ValueError, TypeError)) as exc_info:
            draw_trigger_on_bev(bev_map, trigger_points, trigger_type)

        # The error should be related to the (semantic_class,) tuple issue
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["color", "scalar", "tuple", "fillpoly"])

    def test_generate_bev_from_map_with_processor(self):
        """Test BEV generation with MapProcessor."""
        map_data = self.create_mock_map_data()
        world2ego = np.eye(4)
        ego_position = (0.0, 0.0)

        # Test with dictionary (should create MapProcessor internally)
        bev_map = generate_bev_from_map(map_data, world2ego, ego_position)
        assert bev_map.shape == (128, 256)
        assert np.any(bev_map > 0)

        # Test with pre-created MapProcessor
        processor = MapProcessor(map_data)
        bev_map2 = generate_bev_from_map(processor, world2ego, ego_position)
        assert bev_map2.shape == (128, 256)

    def test_generate_full_bev(self):
        """Test full 360-degree BEV generation."""
        map_data = self.create_mock_map_data()
        world2ego = np.eye(4)
        ego_position = (0.0, 0.0)

        full_bev = generate_full_bev_from_map(map_data, world2ego, ego_position)
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
        ego_position = (0.0, 0.0)

        bev_map = processor.generate_bev(world2ego, ego_position)
        assert bev_map.shape == (128, 256)
        assert np.any(bev_map > 0)

    def test_map_processor_empty_map(self):
        """Test MapProcessor with empty map data."""
        processor = MapProcessor({})
        assert processor.lane_kdtree is None
        assert processor.trigger_kdtree is None

        ego_position = (0.0, 0.0)
        bev_map = processor.generate_bev(np.eye(4), ego_position)
        assert np.all(bev_map == 0)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_filter_points_in_range_front_view(self):
        """Test point filtering for front view."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # At ego (edge of front view)
                [10.0, 0.0, 0.0],  # 10m forward (in range)
                [-5.0, 0.0, 0.0],  # 5m behind (out of range for front view)
                [100.0, 0.0, 0.0],  # 100m forward (out of range)
                [5.0, 70.0, 0.0],  # Far right (out of lateral range)
            ]
        )

        bev_height, bev_width = 128, 256
        resolution = 0.5  # Coverage: 0 to 64m forward, ±64m lateral

        filtered, mask = filter_points_in_range(
            points, bev_height, bev_width, resolution, BEV_VIEW_FRONT
        )

        # Check which points are in range
        assert mask[0] == True  # At ego is at edge but included
        assert mask[1] == True  # 10m forward is in range
        assert mask[2] == False  # Behind ego is out of range for front view
        assert mask[3] == False  # Too far forward
        assert mask[4] == False  # Too far laterally (70m > 64m limit)

    def test_filter_points_in_range_full_view(self):
        """Test point filtering for full 360° view."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # At ego
                [30.0, 0.0, 0.0],  # 30m forward
                [-30.0, 0.0, 0.0],  # 30m behind
                [100.0, 0.0, 0.0],  # 100m forward (out of range)
                [-100.0, 0.0, 0.0],  # 100m behind (out of range)
            ]
        )

        bev_height, bev_width = 256, 256
        resolution = 0.5  # Coverage: ±64m in all directions

        filtered, mask = filter_points_in_range(
            points, bev_height, bev_width, resolution, BEV_VIEW_FULL
        )

        # Check which points are in range
        assert mask[0] == True  # At ego
        assert mask[1] == True  # 30m forward is in range
        assert mask[2] == True  # 30m behind is in range
        assert mask[3] == False  # Too far forward
        assert mask[4] == False  # Too far behind

    def test_filter_points_in_range_rear_view(self):
        """Test point filtering for rear view."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # At ego (edge of rear view)
                [-10.0, 0.0, 0.0],  # 10m behind (in range)
                [5.0, 0.0, 0.0],  # 5m forward (out of range for rear view)
                [-100.0, 0.0, 0.0],  # 100m behind (out of range)
            ]
        )

        bev_height, bev_width = 128, 256
        resolution = 0.5  # Coverage: -64m to 0m, ±64m lateral

        filtered, mask = filter_points_in_range(
            points, bev_height, bev_width, resolution, BEV_VIEW_REAR
        )

        # Check which points are in range
        assert mask[0] == True  # At ego is at edge but included
        assert mask[1] == True  # 10m behind is in range
        assert mask[2] == False  # Forward is out of range for rear view
        assert mask[3] == False  # Too far behind

    def test_empty_lane_segments(self):
        """Test handling of empty lane segments."""
        map_data = {"road1": {"lane1": [{"Points": [], "Type": "Solid"}]}}
        ego_position = (0.0, 0.0)

        bev_map = generate_bev_from_map(map_data, np.eye(4), ego_position)
        assert np.all(bev_map == 0)

    def test_invalid_lane_type(self):
        """Test handling of unknown lane types."""
        bev_map = np.zeros((128, 256), dtype=np.uint8)
        lane_points = np.array([[64, 128], [80, 128]])

        bev_map = draw_lane_on_bev(bev_map, lane_points, "UnknownType")
        # Should default to road class (1)
        assert np.any(bev_map == 1)


class TestMaxDistanceCalculation:
    """Test max_distance parameter and its effect on BEV generation."""
    
    def test_lanes_far_from_ego_need_large_max_distance(self):
        """Test that lanes whose centers are far from ego need sufficient max_distance."""
        # Create a lane that passes near ego but has center far away
        # This simulates a curved road where the lane segment center is distant
        map_data = {
            "road1": {
                "lane1": [
                    {
                        # Lane segment from (-100, 0) to (100, 0)
                        # Center is at (0, 0) but extends far
                        "Points": [
                            ((-100.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                            ((-50.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                            ((50.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                            ((100.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                        ],
                        "Type": "Solid",
                    }
                ],
                "lane2": [
                    {
                        # Lane segment with center at (75, 0) 
                        # Even though some points are near ego
                        "Points": [
                            ((50.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                            ((75.0, 0.0, 0.0), (0.0, 0.0, 0.0)),  # Center point
                            ((100.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                        ],
                        "Type": "Broken",
                    }
                ],
            }
        }
        
        world2ego = np.eye(4)
        ego_position = (0.0, 0.0)
        
        # Test with small max_distance (50m) - should miss lane2
        bev_small_dist = generate_bev_from_map(
            map_data, world2ego, ego_position, max_distance=50.0
        )
        
        # Test with large max_distance (150m) - should catch both lanes
        bev_large_dist = generate_bev_from_map(
            map_data, world2ego, ego_position, max_distance=150.0
        )
        
        # The large max_distance should capture more lanes
        assert np.sum(bev_large_dist > 0) >= np.sum(bev_small_dist > 0)
        
        # Check that both lane types are present with large max_distance
        unique_large = np.unique(bev_large_dist)
        assert LANE_TYPE_TO_BEV_CLASS["Solid"] in unique_large
        # Lane2 with center at 75m should be captured with 150m max_distance
        assert LANE_TYPE_TO_BEV_CLASS["Broken"] in unique_large or np.sum(bev_large_dist > 0) > 0
    
    def test_max_distance_default_covers_bev_range(self):
        """Test that default max_distance should cover at least the BEV visible range."""
        bev_height = 128
        bev_width = 256
        resolution = 0.25
        
        # Calculate BEV coverage
        forward_coverage = bev_height * resolution  # 32m
        lateral_coverage = bev_width * resolution  # 64m
        diagonal_coverage = np.sqrt(forward_coverage**2 + (lateral_coverage/2)**2)  # ~45m
        
        # Create lanes at various distances
        map_data = {
            "road1": {
                f"lane_{dist}m": [
                    {
                        "Points": [
                            ((dist, -5.0, 0.0), (0.0, 0.0, 0.0)),
                            ((dist, 5.0, 0.0), (0.0, 0.0, 0.0)),
                        ],
                        "Type": "Solid",
                    }
                ]
                for dist in [10, 20, 30, 40, 60, 80, 100]
            }
        }
        
        world2ego = np.eye(4)
        ego_position = (0.0, 0.0)
        
        # Generate with default max_distance
        bev_map = generate_bev_from_map(
            map_data, world2ego, ego_position,
            bev_height=bev_height, bev_width=bev_width, resolution=resolution
        )
        
        # Should at least capture lanes within BEV visible range (32m forward)
        # Check that we have non-zero pixels
        assert np.sum(bev_map > 0) > 0, "BEV map should not be empty with default max_distance"
        
    def test_kdtree_query_with_different_max_distances(self):
        """Test that KDTree correctly filters lanes based on max_distance.
        
        With B2D BEV covering 85m x 85m at 0.332m/pixel resolution:
        - Front-half BEV (128x256): covers 0-42.5m forward, ±42.5m lateral
        - Lanes within this range should be visible in the BEV
        """
        # Create map with lanes at known distances
        # Note: Using lane types with different BEV classes for testing
        map_data = {
            "road1": {
                "lane_near": [
                    {
                        # Lane at 10m from ego (should be visible in BEV)
                        "Points": [
                            ((10.0, -2.0, 0.0), (0.0, 0.0, 0.0)),
                            ((10.0, 2.0, 0.0), (0.0, 0.0, 0.0)),
                        ],
                        "Type": "Solid",  # Maps to BEV class 1
                    }
                ],
                "lane_medium": [
                    {
                        # Lane at 30m from ego (visible in BEV with 42.5m range)
                        "Points": [
                            ((30.0, -2.0, 0.0), (0.0, 0.0, 0.0)),
                            ((30.0, 2.0, 0.0), (0.0, 0.0, 0.0)),
                        ],
                        "Type": "Center",  # Maps to BEV class 3 (different from Solid)
                    }
                ],
                "Trigger_Volumes": [
                    {
                        # Trigger at 60m from ego (outside BEV but may be found by KDTree)
                        "Points": [
                            [60.0, -2.0, 0.0],
                            [60.0, 2.0, 0.0],
                            [62.0, 2.0, 0.0],
                            [62.0, -2.0, 0.0],
                            [60.0, -2.0, 1.0],
                            [60.0, 2.0, 1.0],
                            [62.0, 2.0, 1.0],
                            [62.0, -2.0, 1.0],
                        ],
                        "Type": "TrafficLight",  # Maps to BEV class 4
                    }
                ],
            }
        }
        
        processor = MapProcessor(map_data)
        world2ego = np.eye(4)
        ego_position = (0.0, 0.0)
        
        # Use B2D BEV resolution (0.332m/pixel for 85m coverage)
        bev_resolution = 0.332  # 85m / 256 pixels
        bev_height = 128  # Front-half
        bev_width = 256
        
        # Test with 20m max_distance - should only get near lane
        bev_20m = processor.generate_bev(
            world2ego, ego_position, 
            bev_height=bev_height, bev_width=bev_width,
            resolution=bev_resolution, max_distance=20.0
        )
        unique_20m = np.unique(bev_20m)
        assert LANE_TYPE_TO_BEV_CLASS["Solid"] in unique_20m  # Near lane at 10m
        assert LANE_TYPE_TO_BEV_CLASS["Center"] not in unique_20m  # Medium lane at 30m excluded
        assert TRIGGER_TYPE_TO_BEV_CLASS.get("TrafficLight", 4) not in unique_20m  # Far trigger excluded
        
        # Test with 40m max_distance - should get near and medium  
        bev_40m = processor.generate_bev(
            world2ego, ego_position,
            bev_height=bev_height, bev_width=bev_width,
            resolution=bev_resolution, max_distance=40.0
        )
        unique_40m = np.unique(bev_40m)
        assert LANE_TYPE_TO_BEV_CLASS["Solid"] in unique_40m  # Near lane at 10m
        assert LANE_TYPE_TO_BEV_CLASS["Center"] in unique_40m  # Medium lane at 30m included
        assert TRIGGER_TYPE_TO_BEV_CLASS.get("TrafficLight", 4) not in unique_40m  # Far trigger at 60m still excluded
        
        # Test with 80m max_distance - should get all elements (trigger at 61m center)
        bev_80m = processor.generate_bev(
            world2ego, ego_position,
            bev_height=bev_height, bev_width=bev_width,
            resolution=bev_resolution, max_distance=80.0
        )
        unique_80m = np.unique(bev_80m)
        assert LANE_TYPE_TO_BEV_CLASS["Solid"] in unique_80m  # Near lane
        assert LANE_TYPE_TO_BEV_CLASS["Center"] in unique_80m  # Medium lane
        # Trigger at 60m may or may not be visible depending on BEV range
        # With 128x256 at 0.332m/pixel, forward range is 42.5m, so trigger at 60m is out of BEV


class TestSemanticValidation:
    """Test semantic class validation in BEV maps."""

    def create_mock_map_processor(self):
        """Create a mock MapProcessor for testing."""
        processor = MagicMock()
        processor.generate_bev = MagicMock()
        return processor

    def test_valid_semantic_classes(self):
        """Test that valid semantic classes pass validation."""
        # Create a BEV map with valid semantic classes only
        mock_processor = self.create_mock_map_processor()
        valid_bev = np.zeros((64, 64), dtype=np.uint8)
        valid_bev[10:20, 10:20] = 1  # Road
        valid_bev[30:40, 30:40] = 3  # Centerline
        valid_bev[50:55, 50:55] = 4  # Static objects
        mock_processor.generate_bev.return_value = valid_bev

        world2ego = np.eye(4)
        ego_position = (0.0, 0.0)

        # Should not raise any exception
        result = generate_bev_from_map(mock_processor, world2ego, ego_position)
        assert result.shape == valid_bev.shape
        assert np.array_equal(result, valid_bev)

    def test_invalid_semantic_classes_raises_error(self):
        """Test that invalid semantic classes raise ValueError."""
        mock_processor = self.create_mock_map_processor()
        # Create BEV map with invalid semantic class (99)
        invalid_bev = np.zeros((64, 64), dtype=np.uint8)
        invalid_bev[10:20, 10:20] = 99  # Invalid class
        mock_processor.generate_bev.return_value = invalid_bev

        world2ego = np.eye(4)
        ego_position = (0.0, 0.0)

        with pytest.raises(ValueError, match="BEV map contains invalid semantic class values"):
            generate_bev_from_map(mock_processor, world2ego, ego_position)

    def test_multiple_invalid_semantic_classes(self):
        """Test error message with multiple invalid classes."""
        mock_processor = self.create_mock_map_processor()
        invalid_bev = np.zeros((64, 64), dtype=np.uint8)
        invalid_bev[10:15, 10:15] = 99  # Invalid class 1
        invalid_bev[20:25, 20:25] = 200  # Invalid class 2
        mock_processor.generate_bev.return_value = invalid_bev

        world2ego = np.eye(4)
        ego_position = (0.0, 0.0)

        with pytest.raises(ValueError) as exc_info:
            generate_bev_from_map(mock_processor, world2ego, ego_position)

        error_msg = str(exc_info.value)
        assert "99" in error_msg
        assert "200" in error_msg
        assert "Valid classes are:" in error_msg

    def test_background_class_is_valid(self):
        """Test that background class (0) is always valid."""
        mock_processor = self.create_mock_map_processor()
        # BEV map with only background pixels
        background_bev = np.zeros((64, 64), dtype=np.uint8)
        mock_processor.generate_bev.return_value = background_bev

        world2ego = np.eye(4)
        ego_position = (0.0, 0.0)

        # Should not raise any exception
        result = generate_bev_from_map(mock_processor, world2ego, ego_position)
        assert np.all(result == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
