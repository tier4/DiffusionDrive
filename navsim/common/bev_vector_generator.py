"""
Vector-based BEV generator that wraps the existing MapProcessor.
Maintains backward compatibility with the current vector map approach.
"""

from typing import Dict, Optional
import numpy as np

from navsim.common.bev_generator_base import BEVGenerator
from navsim.common.bev_map_utils import (
    MapProcessor,
    generate_bev_from_map,
    extract_front_half_bev,
    BEV_VIEW_FRONT,
    BEV_VIEW_FULL,
)


class VectorBEVGenerator(BEVGenerator):
    """
    Vector-based BEV generator using sparse lane line representation.

    This generator creates BEV maps by drawing lane boundaries and
    discrete elements, resulting in a sparse representation.
    """

    def __init__(
        self,
        bev_height: int = 128,
        bev_width: int = 256,
        resolution: float = 0.332,
        view_type: str = "front",
        lane_thickness: float = 0.4,
    ):
        """
        Initialize the vector BEV generator.

        Args:
            bev_height: Height of BEV map in pixels
            bev_width: Width of BEV map in pixels
            resolution: Meters per pixel (0.332 for 85m coverage)
            view_type: Type of BEV view ("front", "rear", or "full")
            lane_thickness: Thickness of lane lines in meters
        """
        super().__init__(bev_height, bev_width, resolution, view_type)
        self.lane_thickness = lane_thickness

        # Determine internal generation parameters based on requested output
        if view_type == "front" and bev_height == 128:
            # Generate full 256x256 then extract front half
            self._internal_height = 256
            self._internal_width = 256
            self._internal_view = BEV_VIEW_FULL
            self._extract_front = True
        else:
            # Generate directly at requested size
            self._internal_height = bev_height
            self._internal_width = bev_width
            self._internal_view = BEV_VIEW_FRONT if view_type == "front" else view_type
            self._extract_front = False

    def generate_from_map(
        self,
        map_data: Dict,
        ego_points: np.ndarray,
        ego_heading_rad: float,
        **kwargs
    ) -> np.ndarray:
        """
        Generate vector-based BEV map from HD map data.

        This method uses the existing MapProcessor to draw sparse lane lines
        and traffic elements.

        Args:
            map_data: Dictionary containing road/lane data or MapProcessor instance
            ego_points: (3,) array of ego position in world coordinates [x, y, z]
            ego_heading_rad: Current ego heading in radians
            **kwargs: Additional parameters (e.g., max_distance)

        Returns:
            BEV semantic map with shape (H, W) and values 0-6

        Raises:
            ValueError: If map_data is None or invalid
        """
        if map_data is None:
            raise ValueError("map_data cannot be None")

        if ego_points.shape != (3,):
            raise ValueError(f"ego_points must be shape (3,), got {ego_points.shape}")

        # Use existing generate_bev_from_map function
        # This handles both Dict and MapProcessor inputs
        bev_map = generate_bev_from_map(
            map_data=map_data,
            ego_points=ego_points,
            ego_heading_rad=ego_heading_rad,
            bev_height=self._internal_height,
            bev_width=self._internal_width,
            resolution=self.resolution,
            view_type=self._internal_view,
            lane_thickness=self.lane_thickness,
            max_distance=kwargs.get("max_distance", None),
        )

        # Extract front half if needed (for 128x256 format)
        if self._extract_front:
            bev_map = extract_front_half_bev(bev_map)

        # Validate output
        self.validate_output(bev_map)

        return bev_map

    def get_generation_type(self) -> str:
        """
        Get the type of BEV generation.

        Returns:
            "vector" for vector-based generation
        """
        return "vector"

    def __repr__(self) -> str:
        """String representation of the generator."""
        return (
            f"VectorBEVGenerator("
            f"size={self.bev_height}x{self.bev_width}, "
            f"resolution={self.resolution}m/px, "
            f"view={self.view_type}, "
            f"lane_thickness={self.lane_thickness}m)"
        )