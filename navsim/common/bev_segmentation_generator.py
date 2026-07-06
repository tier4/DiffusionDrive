"""
BEV semantic map generator using segmentation method - SIMPLIFIED VERSION.
Fills road surface area properly by drawing all lanes with sufficient width.
"""

import logging
from typing import Dict, Optional, List, Tuple
import numpy as np
import cv2

from navsim.common.bev_generator_base import BEVGenerator
from navsim.common.bev_map_utils import (
    BEV_VIEW_FRONT,
    BEV_VIEW_FULL,
    MapProcessor,
    extract_lane_points,
    extract_trigger_points,
    transform_points_to_ego,
    ego_to_bev_pixels,
    extract_front_half_bev,
)
from navsim.common.bench2drive_constants import (
    LANE_TYPE_TO_BEV_CLASS,
    TRIGGER_TYPE_TO_BEV_CLASS,
    BEV_SEMANTIC_RESOLUTION,
)

logger = logging.getLogger(__name__)


class SegmentationBEVGenerator(BEVGenerator):
    """
    Simplified BEV semantic map generator that actually fills roads properly.
    """

    def __init__(
        self,
        bev_height: int,
        bev_width: int,
        resolution: float = BEV_SEMANTIC_RESOLUTION,
        view_type: str = BEV_VIEW_FRONT,
        lane_thickness: float = 0.3,
        road_width: float = 5.0,  # Width for road filling
    ):
        """Initialize the segmentation BEV generator."""
        super().__init__(bev_height, bev_width, resolution, view_type)
        self.lane_thickness = lane_thickness
        self.road_width = road_width  # Width to fill for each lane

        # For front view (128x256), generate full (256x256) first, then extract
        self._extract_front = (view_type == BEV_VIEW_FRONT and bev_height == 128)

        if self._extract_front:
            self._internal_height = 256
            self._internal_width = 256
            self._internal_view = BEV_VIEW_FULL
        else:
            self._internal_height = bev_height
            self._internal_width = bev_width
            self._internal_view = view_type

    def generate_from_map(
        self,
        map_data: Dict,
        ego_points: np.ndarray,
        ego_heading_rad: float,
        **kwargs
    ) -> np.ndarray:
        """Generate segmentation-based BEV map from HD map data."""
        if map_data is None:
            raise ValueError("map_data cannot be None")

        if ego_points.shape != (3,):
            raise ValueError(f"ego_points must be shape (3,), got {ego_points.shape}")

        # Initialize BEV map with internal dimensions
        bev_map = np.zeros((self._internal_height, self._internal_width), dtype=np.uint8)

        # Process based on data type
        if isinstance(map_data, MapProcessor):
            bev_map = self._generate_from_processor(
                map_data, ego_points, ego_heading_rad, bev_map
            )
        else:
            bev_map = self._generate_from_dict(
                map_data, ego_points, ego_heading_rad, bev_map
            )

        # Extract front half if needed (like vector generator does)
        if self._extract_front:
            bev_map = extract_front_half_bev(bev_map)

        # Validate output
        self.validate_output(bev_map)

        return bev_map

    def _generate_from_dict(
        self,
        map_dict: Dict,
        ego_points: np.ndarray,
        ego_heading_rad: float,
        bev_map: np.ndarray
    ) -> np.ndarray:
        """Generate BEV from raw map dictionary."""

        # Collect all elements
        all_lanes = []  # (pixels, marking_type)
        centerlines = []
        triggers = []

        # Process all roads
        for road_id, road_data in map_dict.items():
            for lane_id, lane_segments in road_data.items():

                # Handle triggers
                if lane_id == "Trigger_Volumes":
                    if isinstance(lane_segments, list):
                        for trigger in lane_segments:
                            points, t_type = extract_trigger_points(trigger)
                            if points.shape[0] > 0:
                                points_ego = transform_points_to_ego(
                                    points, ego_points, ego_heading_rad
                                )
                                pixels = ego_to_bev_pixels(
                                    points_ego,
                                    self._internal_height,
                                    self._internal_width,
                                    self.resolution,
                                    self._internal_view
                                )
                                triggers.append((pixels, t_type))
                    continue

                # Handle lanes
                if not isinstance(lane_segments, list):
                    continue

                for segment in lane_segments:
                    points, _ = extract_lane_points(segment)
                    if points.shape[0] < 2:
                        continue

                    marking_type = segment.get('Type', 'Unknown')

                    # Transform to ego
                    points_ego = transform_points_to_ego(
                        points, ego_points, ego_heading_rad
                    )

                    # Convert to pixels
                    pixels = ego_to_bev_pixels(
                        points_ego,
                        self._internal_height,
                        self._internal_width,
                        self.resolution,
                        self._internal_view
                    )

                    # Filter valid pixels
                    mask = (
                        (pixels[:, 0] >= 0) & (pixels[:, 0] < self._internal_height) &
                        (pixels[:, 1] >= 0) & (pixels[:, 1] < self._internal_width)
                    )

                    if np.sum(mask) < 2:
                        continue

                    valid_pixels = pixels[mask]

                    if marking_type == "Center":
                        centerlines.append((valid_pixels, marking_type))
                    else:
                        all_lanes.append((valid_pixels, marking_type))

        # Step 1: Fill road surface - draw ALL lanes with thick lines
        road_width_pixels = max(5, int(self.road_width / self.resolution))

        for pixels, _ in all_lanes:
            cv2.polylines(
                bev_map,
                [pixels[:, [1, 0]].astype(np.int32)],
                isClosed=False,
                color=1,  # Road surface
                thickness=road_width_pixels
            )

        # Also fill from centerlines with smaller width
        center_width_pixels = max(3, int(3.0 / self.resolution))
        for pixels, _ in centerlines:
            cv2.polylines(
                bev_map,
                [pixels[:, [1, 0]].astype(np.int32)],
                isClosed=False,
                color=1,  # Road surface first
                thickness=center_width_pixels
            )

        # Step 2: Apply morphological closing to connect gaps
        road_mask = (bev_map == 1).astype(np.uint8)
        kernel_size = max(5, int(3.0 / self.resolution))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        bev_map[closed_mask == 1] = 1

        # Step 3: Draw centerlines on top (class 3)
        for pixels, _ in centerlines:
            thickness_pixels = max(1, int(self.lane_thickness / self.resolution))
            cv2.polylines(
                bev_map,
                [pixels[:, [1, 0]].astype(np.int32)],
                isClosed=False,
                color=3,  # Centerline class
                thickness=thickness_pixels
            )

        # Step 4: Draw triggers (class 4)
        for pixels, t_type in triggers:
            semantic_class = TRIGGER_TYPE_TO_BEV_CLASS.get(t_type, 4)
            if len(pixels) >= 4:
                bottom_pixels = pixels[:4] if len(pixels) >= 8 else pixels
                cv2.fillPoly(
                    bev_map,
                    [bottom_pixels[:, [1, 0]].astype(np.int32)],
                    semantic_class
                )

        return bev_map

    def _generate_from_processor(
        self,
        processor: MapProcessor,
        ego_points: np.ndarray,
        ego_heading_rad: float,
        bev_map: np.ndarray
    ) -> np.ndarray:
        """Generate BEV using preprocessed MapProcessor."""
        from navsim.common.bev_map_utils import generate_bev_from_map

        # Generate vector map
        vector_bev = generate_bev_from_map(
            map_data=processor,
            ego_points=ego_points,
            ego_heading_rad=ego_heading_rad,
            bev_height=self._internal_height,
            bev_width=self._internal_width,
            resolution=self.resolution,
            view_type=self._internal_view,
            lane_thickness=self.lane_thickness,
        )

        # Expand road areas
        road_mask = (vector_bev == 1).astype(np.uint8)
        centerlines = (vector_bev == 3).astype(np.uint8)
        static_objects = (vector_bev == 4).astype(np.uint8)

        # Dilate road areas significantly
        kernel_size = max(7, int(self.road_width / self.resolution))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        filled_road = cv2.dilate(road_mask, kernel, iterations=1)

        # Combine
        bev_map[filled_road == 1] = 1
        bev_map[centerlines == 1] = 3
        bev_map[static_objects == 1] = 4

        return bev_map

    def get_generation_type(self) -> str:
        """Get the type of BEV generation."""
        return "segmentation_v2_simple"

    def __repr__(self) -> str:
        """String representation of the generator."""
        return (
            f"SegmentationBEVGeneratorV2Simple("
            f"size={self.bev_height}x{self.bev_width}, "
            f"resolution={self.resolution}m/px, "
            f"view={self.view_type}, "
            f"road_width={self.road_width}m)"
        )