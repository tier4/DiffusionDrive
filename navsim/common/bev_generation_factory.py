"""
Factory pattern for creating BEV generators.
Allows switching between vector and segmentation approaches through configuration.
"""

from typing import Optional, Dict, Any
import logging

from navsim.common.bev_generator_base import BEVGenerator
from navsim.common.bev_vector_generator import VectorBEVGenerator
from navsim.common.bev_segmentation_generator import SegmentationBEVGenerator

logger = logging.getLogger(__name__)


class BEVGeneratorFactory:
    """
    Factory class for creating BEV generators based on configuration.

    This factory allows the system to switch between different BEV
    generation strategies (vector vs segmentation) without changing
    the calling code.
    """

    # Registry of available generators
    _generators = {
        "vector": VectorBEVGenerator,
        "segmentation": SegmentationBEVGenerator,
    }

    @classmethod
    def create_generator(
        cls,
        generation_type: str = "vector",
        bev_height: int = 128,
        bev_width: int = 256,
        resolution: float = 0.332,
        view_type: str = "front",
        **kwargs
    ) -> BEVGenerator:
        """
        Create a BEV generator based on the specified type.

        Args:
            generation_type: Type of generator ("vector" or "segmentation")
            bev_height: Height of BEV map in pixels
            bev_width: Width of BEV map in pixels
            resolution: Meters per pixel
            view_type: Type of BEV view ("front", "rear", or "full")
            **kwargs: Additional generator-specific parameters

        Returns:
            BEVGenerator instance of the specified type

        Raises:
            ValueError: If generation_type is not recognized
        """
        if generation_type not in cls._generators:
            raise ValueError(
                f"Unknown generation type: {generation_type}. "
                f"Available types: {list(cls._generators.keys())}"
            )

        generator_class = cls._generators[generation_type]

        # Extract generator-specific parameters
        if generation_type == "vector":
            specific_params = cls._extract_vector_params(kwargs)
        elif generation_type == "segmentation":
            specific_params = cls._extract_segmentation_params(kwargs)
        else:
            specific_params = {}

        # Create and return generator instance
        generator = generator_class(
            bev_height=bev_height,
            bev_width=bev_width,
            resolution=resolution,
            view_type=view_type,
            **specific_params
        )

        logger.info(f"Created {generation_type} BEV generator: {generator}")

        return generator

    @staticmethod
    def _extract_vector_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract parameters specific to VectorBEVGenerator.

        Args:
            kwargs: All keyword arguments

        Returns:
            Dictionary of vector-specific parameters
        """
        vector_params = {}

        if "lane_thickness" in kwargs:
            vector_params["lane_thickness"] = kwargs["lane_thickness"]

        return vector_params

    @staticmethod
    def _extract_segmentation_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract parameters specific to SegmentationBEVGenerator.

        Args:
            kwargs: All keyword arguments

        Returns:
            Dictionary of segmentation-specific parameters
        """
        segmentation_params = {}

        if "lane_width" in kwargs:
            segmentation_params["road_width"] = kwargs["lane_width"]  # Map to road_width parameter

        # Note: fill_drivable_area and use_lane_connectivity are ignored
        # as the simplified generator always fills roads

        return segmentation_params

    @classmethod
    def register_generator(cls, name: str, generator_class: type) -> None:
        """
        Register a new generator type.

        This allows extending the factory with custom generators.

        Args:
            name: Name identifier for the generator
            generator_class: Generator class (must inherit from BEVGenerator)

        Raises:
            TypeError: If generator_class doesn't inherit from BEVGenerator
        """
        if not issubclass(generator_class, BEVGenerator):
            raise TypeError(
                f"{generator_class} must inherit from BEVGenerator"
            )

        cls._generators[name] = generator_class
        logger.info(f"Registered new BEV generator type: {name}")

    @classmethod
    def get_available_types(cls) -> list:
        """
        Get list of available generator types.

        Returns:
            List of registered generator type names
        """
        return list(cls._generators.keys())


def create_generator_from_config(config: Any) -> BEVGenerator:
    """
    Convenience function to create a generator from a configuration object.

    Args:
        config: Configuration object with BEV generation parameters

    Returns:
        BEVGenerator instance

    Raises:
        AttributeError: If config is missing required attributes
    """
    # Extract generation type with fallback to vector
    generation_type = getattr(config, "bev_generation_type", "vector")

    # Extract common parameters
    params = {
        "generation_type": generation_type,
        "bev_height": getattr(config, "bev_pixel_height", 128),
        "bev_width": getattr(config, "bev_pixel_width", 256),
        "resolution": getattr(config, "bev_pixel_size", 0.332),
    }

    # Add type-specific parameters
    if generation_type == "segmentation":
        if hasattr(config, "lane_width_meters"):
            params["lane_width"] = config.lane_width_meters
        if hasattr(config, "fill_drivable_area"):
            params["fill_drivable_area"] = config.fill_drivable_area
    elif generation_type == "vector":
        if hasattr(config, "lane_thickness_meters"):
            params["lane_thickness"] = config.lane_thickness_meters

    return BEVGeneratorFactory.create_generator(**params)