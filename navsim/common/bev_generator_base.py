"""
Base abstract class for BEV map generators.
Provides a common interface for both vector and segmentation-based BEV generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class BEVGenerator(ABC):
    """
    Abstract base class for BEV map generators.

    This class defines the interface that all BEV generators must implement,
    allowing for different generation strategies (vector, segmentation, etc.)
    while maintaining a consistent API.
    """

    def __init__(
        self,
        bev_height: int = 128,
        bev_width: int = 256,
        resolution: float = 0.332,  # meters per pixel for 85m coverage
        view_type: str = "front",
    ):
        """
        Initialize the BEV generator.

        Args:
            bev_height: Height of BEV map in pixels
            bev_width: Width of BEV map in pixels
            resolution: Meters per pixel
            view_type: Type of BEV view ("front", "rear", or "full")
        """
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.resolution = resolution
        self.view_type = view_type

        # Validate parameters
        if bev_height <= 0 or bev_width <= 0:
            raise ValueError(f"Invalid BEV dimensions: {bev_height}x{bev_width}")
        if resolution <= 0:
            raise ValueError(f"Invalid resolution: {resolution}")
        if view_type not in ["front", "rear", "full"]:
            raise ValueError(f"Invalid view type: {view_type}. Must be 'front', 'rear', or 'full'")

    @abstractmethod
    def generate_from_map(
        self,
        map_data: Dict,
        ego_points: np.ndarray,
        ego_heading_rad: float,
        **kwargs
    ) -> np.ndarray:
        """
        Generate BEV map from HD map data.

        Args:
            map_data: Dictionary containing road/lane data or MapProcessor instance
            ego_points: (3,) array of ego position in world coordinates [x, y, z]
            ego_heading_rad: Current ego heading in radians
            **kwargs: Additional generator-specific parameters

        Returns:
            BEV semantic map with shape (H, W) and values 0-6
        """
        pass

    @abstractmethod
    def get_generation_type(self) -> str:
        """
        Get the type of BEV generation.

        Returns:
            String identifier for the generation type (e.g., "vector", "segmentation")
        """
        pass

    def validate_output(self, bev_map: np.ndarray) -> None:
        """
        Validate the generated BEV map.

        Args:
            bev_map: Generated BEV map to validate

        Raises:
            ValueError: If the BEV map is invalid
        """
        # Check shape
        expected_shape = (self.bev_height, self.bev_width)
        if bev_map.shape != expected_shape:
            raise ValueError(
                f"Invalid BEV shape: {bev_map.shape}. Expected {expected_shape}"
            )

        # Check data type
        if bev_map.dtype != np.uint8:
            raise ValueError(
                f"Invalid BEV dtype: {bev_map.dtype}. Expected np.uint8"
            )

        # Check semantic class values (0-6 for 7 classes)
        unique_values = np.unique(bev_map)
        invalid_values = unique_values[unique_values > 6]
        if len(invalid_values) > 0:
            raise ValueError(
                f"Invalid semantic class values in BEV: {invalid_values}. "
                f"Valid range is 0-6"
            )

    def get_ego_pixel_position(self) -> tuple:
        """
        Get ego vehicle pixel position based on view type.

        Returns:
            Tuple of (row, col) for ego position in pixels
        """
        if self.view_type == "front":
            # Front-only view: ego at bottom-center
            return (self.bev_height - 1, self.bev_width // 2)
        elif self.view_type == "rear":
            # Rear-only view: ego at top-center
            return (0, self.bev_width // 2)
        elif self.view_type == "full":
            # Full 360° view: ego at center
            return (self.bev_height // 2, self.bev_width // 2)
        else:
            # Should never reach here due to validation in __init__
            raise ValueError(f"Invalid view type: {self.view_type}")