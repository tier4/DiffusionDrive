"""
Utilities for generating BEV semantic maps from Bench2Drive vectorized map data.

This module handles the generation of BEV semantic maps using lane lines and
trigger volumes from Bench2Drive HD maps.

The module provides two implementations:
1. MapProcessor (default): High-performance implementation using KDTree for spatial queries
2. Legacy functions: Original implementation for compatibility and testing
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


# Bench2Drive lane types to NavSim BEV semantic classes
LANE_TYPE_TO_BEV_CLASS = {
    "Broken": 1,  # Broken line -> Road
    "Solid": 1,  # Solid line -> Road
    "SolidSolid": 1,  # Double solid -> Road
    "Center": 3,  # Center line -> Lane centerline
}

# Bench2Drive trigger types to NavSim BEV semantic classes
TRIGGER_TYPE_TO_BEV_CLASS = {
    "TrafficLight": 4,  # Traffic light -> Static object
    "StopSign": 4,  # Stop sign -> Static object
}


def load_map_data(map_path: Union[str, Path]) -> Dict:
    """
    Load Bench2Drive map data from NPZ file.

    Args:
        map_path: Path to the NPZ map file

    Returns:
        Dictionary containing road data with lanes and trigger volumes
    """
    map_path = Path(map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    # Load NPZ file
    map_data = np.load(str(map_path), allow_pickle=True)

    # Extract dictionary from 0-d object array
    map_dict = dict(map_data["arr"][()])

    return map_dict


def extract_lane_points(lane_segment: Dict) -> Tuple[np.ndarray, str]:
    """
    Extract points and type from a lane segment.

    Args:
        lane_segment: Lane segment dictionary with 'Points' and 'Type'

    Returns:
        Tuple of (points array, lane type)
    """
    points_raw = lane_segment["Points"]
    lane_type = lane_segment.get("Type", "Unknown")

    # Extract positions from ((x, y, z), (roll, pitch, yaw)) format
    points = []
    for point_data in points_raw:
        if isinstance(point_data, tuple) and len(point_data) >= 1:
            pos = point_data[0]  # (x, y, z)
            points.append([pos[0], pos[1], pos[2]])

    return np.array(points), lane_type


def extract_trigger_points(trigger: Dict) -> Tuple[np.ndarray, str]:
    """
    Extract points and type from a trigger volume.

    Args:
        trigger: Trigger volume dictionary with 'Points' and 'Type'

    Returns:
        Tuple of (points array, trigger type)
    """
    points = np.array(trigger["Points"])  # Already in simple format
    trigger_type = trigger.get("Type", "Unknown")

    return points, trigger_type


def transform_points_to_ego(
    points: np.ndarray, world2ego: np.ndarray, left_to_right: bool = True
) -> np.ndarray:
    """
    Transform points from world coordinates to ego-centric coordinates.

    Args:
        points: Nx3 array of points in world coordinates
        world2ego: 4x4 transformation matrix from world to ego
        left_to_right: Whether to convert from left-handed to right-handed

    Returns:
        Nx3 array of points in ego-centric coordinates
    """
    # Convert to homogeneous coordinates
    points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)

    # Apply transformation
    points_ego_h = points_h @ world2ego.T
    points_ego = points_ego_h[:, :3]

    # Convert from left-handed to right-handed if needed
    if left_to_right:
        points_ego[:, 1] = -points_ego[:, 1]  # Flip Y axis

    return points_ego


def ego_to_bev_pixels(
    points_ego: np.ndarray,
    bev_height: int,
    bev_width: int,
    resolution: float,
    coverage_behind: float = 0.0,
) -> np.ndarray:
    """
    Convert ego-centric coordinates to BEV pixel coordinates.

    Args:
        points_ego: Nx2 or Nx3 array of points in ego coordinates (meters)
        bev_height: Height of BEV map in pixels
        bev_width: Width of BEV map in pixels
        resolution: Meters per pixel
        coverage_behind: Meters of coverage behind ego (default 0 for front-only)

    Returns:
        Nx2 array of (row, col) pixel coordinates
    """
    # Extract x, y coordinates
    x = points_ego[:, 0]
    y = points_ego[:, 1]

    # BEV coverage
    # Forward coverage: bev_height * resolution - coverage_behind
    # Lateral coverage: bev_width * resolution (centered)

    # Convert to pixels
    # Row: ego at bottom, forward is up
    rows = (bev_height - 1) - (x - coverage_behind) / resolution

    # Col: ego at center, left is positive in ego coords, right in image
    cols = bev_width / 2 + y / resolution

    pixels = np.stack([rows, cols], axis=1).astype(np.int32)

    return pixels


def filter_points_in_range(
    points_ego: np.ndarray,
    bev_height: int,
    bev_width: int,
    resolution: float,
    coverage_behind: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter points to those within BEV range.

    Args:
        points_ego: Nx2 or Nx3 array of points in ego coordinates
        bev_height: Height of BEV map
        bev_width: Width of BEV map
        resolution: Meters per pixel
        coverage_behind: Coverage behind ego in meters

    Returns:
        Tuple of (filtered points, valid mask)
    """
    x = points_ego[:, 0]
    y = points_ego[:, 1]

    # Calculate bounds in meters
    x_min = coverage_behind
    x_max = coverage_behind + bev_height * resolution
    y_min = -bev_width * resolution / 2
    y_max = bev_width * resolution / 2

    # Create mask
    valid_mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)

    filtered_points = points_ego[valid_mask]

    return filtered_points, valid_mask


def draw_lane_on_bev(
    bev_map: np.ndarray,
    lane_points: np.ndarray,
    lane_type: str,
    resolution: float = 0.25,
    thickness: float = 0.3,
) -> np.ndarray:
    """
    Draw a lane on the BEV map.

    Args:
        bev_map: BEV semantic map to draw on
        lane_points: Nx2 array of pixel coordinates
        lane_type: Type of lane (Broken, Solid, etc.)
        resolution: Meters per pixel
        thickness: Lane thickness in meters

    Returns:
        Updated BEV map
    """
    if len(lane_points) < 2:
        return bev_map

    # Get semantic class
    semantic_class = LANE_TYPE_TO_BEV_CLASS.get(lane_type, 1)

    # Convert thickness to pixels
    thickness_pixels = max(1, int(thickness / resolution))

    # Draw polyline
    # OpenCV expects (x,y) = (col,row) format
    points = lane_points[:, [1, 0]].astype(np.int32)

    cv2.polylines(
        bev_map, [points], isClosed=False, color=semantic_class, thickness=thickness_pixels
    )

    return bev_map


def draw_trigger_on_bev(
    bev_map: np.ndarray, trigger_points: np.ndarray, trigger_type: str
) -> np.ndarray:
    """
    Draw a trigger volume on the BEV map.

    Args:
        bev_map: BEV semantic map to draw on
        trigger_points: 8x2 array of pixel coordinates (box vertices)
        trigger_type: Type of trigger (TrafficLight, StopSign)

    Returns:
        Updated BEV map
    """
    if len(trigger_points) < 4:
        return bev_map

    # Get semantic class
    semantic_class = TRIGGER_TYPE_TO_BEV_CLASS.get(trigger_type, 4)

    # Project 3D box to 2D (use bottom 4 vertices)
    # Assuming points are ordered: bottom 4, then top 4
    bottom_points = trigger_points[:4] if len(trigger_points) >= 8 else trigger_points

    # Convert to integer pixel coordinates
    points = bottom_points[:, [1, 0]].astype(np.int32)

    # Draw filled polygon
    cv2.fillPoly(bev_map, [points], semantic_class)

    return bev_map


def generate_bev_from_map(
    map_data: Dict,
    world2ego: np.ndarray,
    bev_height: int = 128,
    bev_width: int = 256,
    resolution: float = 0.25,
    coverage_behind: float = 0.0,
    lane_thickness: float = 0.3,
    max_distance: float = 100.0,
) -> np.ndarray:
    """
    Generate BEV semantic map from vectorized map data.

    Args:
        map_data: Dictionary containing road data
        world2ego: 4x4 transformation matrix from world to ego
        bev_height: Height of BEV map in pixels
        bev_width: Width of BEV map in pixels
        resolution: Meters per pixel
        coverage_behind: Coverage behind ego in meters
        lane_thickness: Lane line thickness in meters
        max_distance: Maximum distance to consider map elements

    Returns:
        BEV semantic map with shape (H, W) and values 0-6
    """
    # Initialize BEV map
    bev_map = np.zeros((bev_height, bev_width), dtype=np.float32)

    # Process each road
    for road_id, road_data in map_data.items():
        # Process lanes
        for lane_id, lane_segments in road_data.items():
            if lane_id == "Trigger_Volumes":
                continue

            if not isinstance(lane_segments, list):
                continue

            # Process each segment
            for segment in lane_segments:
                try:
                    # Extract points and type
                    points_world, lane_type = extract_lane_points(segment)

                    if len(points_world) == 0:
                        continue

                    # Transform to ego coordinates
                    points_ego = transform_points_to_ego(points_world, world2ego)

                    # Filter by distance
                    distances = np.linalg.norm(points_ego[:, :2], axis=1)
                    valid_dist = distances < max_distance
                    points_ego = points_ego[valid_dist]

                    if len(points_ego) == 0:
                        continue

                    # Filter points in BEV range
                    points_filtered, _ = filter_points_in_range(
                        points_ego, bev_height, bev_width, resolution, coverage_behind
                    )

                    if len(points_filtered) < 2:
                        continue

                    # Convert to pixel coordinates
                    pixels = ego_to_bev_pixels(
                        points_filtered, bev_height, bev_width, resolution, coverage_behind
                    )

                    # Draw lane
                    bev_map = draw_lane_on_bev(
                        bev_map, pixels, lane_type, resolution, lane_thickness
                    )

                except Exception as e:
                    logger.debug(f"Error processing lane {lane_id} in road {road_id}: {e}")
                    continue

        # Process trigger volumes
        if "Trigger_Volumes" in road_data:
            for trigger in road_data["Trigger_Volumes"]:
                try:
                    # Extract points and type
                    points_world, trigger_type = extract_trigger_points(trigger)

                    # Transform to ego coordinates
                    points_ego = transform_points_to_ego(points_world, world2ego)

                    # Check distance (use centroid)
                    centroid = points_ego.mean(axis=0)
                    if np.linalg.norm(centroid[:2]) > max_distance:
                        continue

                    # Convert to pixel coordinates
                    pixels = ego_to_bev_pixels(
                        points_ego, bev_height, bev_width, resolution, coverage_behind
                    )

                    # Check if any points are in view
                    in_bounds = (
                        (pixels[:, 0] >= 0)
                        & (pixels[:, 0] < bev_height)
                        & (pixels[:, 1] >= 0)
                        & (pixels[:, 1] < bev_width)
                    )

                    if not np.any(in_bounds):
                        continue

                    # Draw trigger
                    bev_map = draw_trigger_on_bev(bev_map, pixels, trigger_type)

                except Exception as e:
                    logger.debug(f"Error processing trigger in road {road_id}: {e}")
                    continue

    return bev_map


def generate_full_bev_from_map(
    map_data: Union[Dict, MapProcessor],
    world2ego: np.ndarray,
    full_height: int = 256,
    full_width: int = 256,
    resolution: float = 0.25,
    **kwargs,
) -> np.ndarray:
    """
    Generate full 360-degree BEV map from vectorized map data.

    Args:
        map_data: Dictionary containing road data or MapProcessor instance
        world2ego: 4x4 transformation matrix
        full_height: Height of full BEV map
        full_width: Width of full BEV map
        resolution: Meters per pixel
        **kwargs: Additional arguments for generate_bev_from_map

    Returns:
        Full BEV semantic map with shape (full_height, full_width)
    """
    # For full BEV, coverage is centered around ego
    coverage_behind = full_height * resolution / 2

    return generate_bev_from_map(
        map_data=map_data,
        world2ego=world2ego,
        bev_height=full_height,
        bev_width=full_width,
        resolution=resolution,
        coverage_behind=coverage_behind,
        **kwargs,
    )


def extract_front_half_bev(full_bev: np.ndarray, front_ratio: float = 0.5) -> np.ndarray:
    """
    Extract front half of BEV map for NavSim format.

    Args:
        full_bev: Full BEV map (H, W)
        front_ratio: Ratio of front coverage (default 0.5)

    Returns:
        Front half BEV with shape (H//2, W)
    """
    height = full_bev.shape[0]
    front_height = int(height * front_ratio)

    # Front half is top portion of full BEV
    return full_bev[:front_height, :].copy()


def combine_map_and_dynamic(static_bev: np.ndarray, dynamic_bev: np.ndarray) -> np.ndarray:
    """
    Combine static map BEV with dynamic objects.

    Args:
        static_bev: BEV from map data (lanes, triggers)
        dynamic_bev: BEV from dynamic objects (vehicles, pedestrians)

    Returns:
        Combined BEV map
    """
    # Dynamic objects override static map
    combined = static_bev.copy()
    dynamic_mask = dynamic_bev > 0
    combined[dynamic_mask] = dynamic_bev[dynamic_mask]

    return combined


# --- HIGH-PERFORMANCE IMPLEMENTATION WITH KDTREE ---
class MapProcessor:
    """
    Pre-processes map data and provides a fast method to generate BEV maps.
    Uses KDTree for efficient spatial queries.
    """

    def __init__(self, map_data: Dict):
        logger.info("Initializing MapProcessor: Pre-processing map data...")
        self._flat_lanes = []
        self._flat_triggers = []
        lane_centers = []
        trigger_centers = []

        # Flatten the map data structure for efficient access
        for road_data in map_data.values():
            for lane_id, lane_segments in road_data.items():
                if lane_id == "Trigger_Volumes":
                    for trigger in lane_segments:
                        points, t_type = extract_trigger_points(trigger)
                        if points.shape[0] > 0:
                            self._flat_triggers.append({"points": points, "type": t_type})
                            trigger_centers.append(np.mean(points[:, :2], axis=0))
                elif isinstance(lane_segments, list):
                    for segment in lane_segments:
                        points, l_type = extract_lane_points(segment)
                        if points.shape[0] > 0:
                            self._flat_lanes.append({"points": points, "type": l_type})
                            lane_centers.append(np.mean(points[:, :2], axis=0))

        # Build KDTrees for fast spatial queries
        self.lane_kdtree = KDTree(lane_centers) if lane_centers else None
        self.trigger_kdtree = KDTree(trigger_centers) if trigger_centers else None
        logger.info(
            f"MapProcessor initialized with {len(self._flat_lanes)} lanes and {len(self._flat_triggers)} triggers."
        )

    def generate_bev(
        self,
        world2ego: np.ndarray,
        bev_height: int = 128,
        bev_width: int = 256,
        resolution: float = 0.25,
        coverage_behind: float = 0.0,
        lane_thickness: float = 0.3,
        max_distance: float = 100.0,
    ) -> np.ndarray:
        """
        Efficiently generates a BEV map by querying nearby map elements.
        """
        bev_map = np.zeros((bev_height, bev_width), dtype=np.float32)

        # Get ego position in world coordinates to query the KDTree
        ego2world = np.linalg.inv(world2ego)
        ego_pos_world = ego2world[:2, 3]

        # --- Process Lanes ---
        if self.lane_kdtree:
            # Query KDTree to find indices of lanes within max_distance
            nearby_indices = self.lane_kdtree.query_ball_point(ego_pos_world, r=max_distance)

            for idx in nearby_indices:
                segment = self._flat_lanes[idx]
                points_ego = transform_points_to_ego(segment["points"], world2ego)
                pixels = ego_to_bev_pixels(
                    points_ego, bev_height, bev_width, resolution, coverage_behind
                )
                draw_lane_on_bev(bev_map, pixels, segment["type"], resolution, lane_thickness)

        # --- Process Triggers ---
        if self.trigger_kdtree:
            nearby_indices = self.trigger_kdtree.query_ball_point(ego_pos_world, r=max_distance)

            for idx in nearby_indices:
                trigger = self._flat_triggers[idx]
                points_ego = transform_points_to_ego(trigger["points"], world2ego)
                pixels = ego_to_bev_pixels(
                    points_ego, bev_height, bev_width, resolution, coverage_behind
                )
                draw_trigger_on_bev(bev_map, pixels, trigger["type"])

        return bev_map


# --- LEGACY IMPLEMENTATION (ORIGINAL) ---
def generate_bev_from_map_legacy(
    map_data: Dict,
    world2ego: np.ndarray,
    bev_height: int = 128,
    bev_width: int = 256,
    resolution: float = 0.25,
    coverage_behind: float = 0.0,
    lane_thickness: float = 0.3,
    max_distance: float = 100.0,
) -> np.ndarray:
    """
    Generate BEV semantic map from vectorized map data (legacy implementation).
    This is the original implementation that processes the entire map for each frame.
    Use MapProcessor for better performance.

    Args:
        map_data: Dictionary containing road data
        world2ego: 4x4 transformation matrix from world to ego
        bev_height: Height of BEV map in pixels
        bev_width: Width of BEV map in pixels
        resolution: Meters per pixel
        coverage_behind: Coverage behind ego in meters
        lane_thickness: Lane line thickness in meters
        max_distance: Maximum distance to consider map elements

    Returns:
        BEV semantic map with shape (H, W) and values 0-6
    """
    # Initialize BEV map
    bev_map = np.zeros((bev_height, bev_width), dtype=np.float32)

    # Process each road
    for road_id, road_data in map_data.items():
        # Process lanes
        for lane_id, lane_segments in road_data.items():
            if lane_id == "Trigger_Volumes":
                continue

            if not isinstance(lane_segments, list):
                continue

            # Process each segment
            for segment in lane_segments:
                try:
                    # Extract points and type
                    points_world, lane_type = extract_lane_points(segment)

                    if len(points_world) == 0:
                        continue

                    # Transform to ego coordinates
                    points_ego = transform_points_to_ego(points_world, world2ego)

                    # Filter by distance
                    distances = np.linalg.norm(points_ego[:, :2], axis=1)
                    valid_dist = distances < max_distance
                    points_ego = points_ego[valid_dist]

                    if len(points_ego) == 0:
                        continue

                    # Filter points in BEV range
                    points_filtered, _ = filter_points_in_range(
                        points_ego, bev_height, bev_width, resolution, coverage_behind
                    )

                    if len(points_filtered) < 2:
                        continue

                    # Convert to pixel coordinates
                    pixels = ego_to_bev_pixels(
                        points_filtered, bev_height, bev_width, resolution, coverage_behind
                    )

                    # Draw lane
                    draw_lane_on_bev(
                        bev_map, pixels, lane_type, resolution, lane_thickness
                    )

                except Exception as e:
                    logger.debug(f"Error processing lane {lane_id} in road {road_id}: {e}")
                    continue

        # Process trigger volumes
        if "Trigger_Volumes" in road_data:
            for trigger in road_data["Trigger_Volumes"]:
                try:
                    # Extract points and type
                    points_world, trigger_type = extract_trigger_points(trigger)

                    # Transform to ego coordinates
                    points_ego = transform_points_to_ego(points_world, world2ego)

                    # Check distance (use centroid)
                    centroid = points_ego.mean(axis=0)
                    if np.linalg.norm(centroid[:2]) > max_distance:
                        continue

                    # Convert to pixel coordinates
                    pixels = ego_to_bev_pixels(
                        points_ego, bev_height, bev_width, resolution, coverage_behind
                    )

                    # Check if any points are in view
                    in_bounds = (
                        (pixels[:, 0] >= 0)
                        & (pixels[:, 0] < bev_height)
                        & (pixels[:, 1] >= 0)
                        & (pixels[:, 1] < bev_width)
                    )

                    if not np.any(in_bounds):
                        continue

                    # Draw trigger
                    draw_trigger_on_bev(bev_map, pixels, trigger_type)

                except Exception as e:
                    logger.debug(f"Error processing trigger in road {road_id}: {e}")
                    continue

    return bev_map


# --- DEFAULT IMPLEMENTATION (USES MAPPROCESSOR) ---
def generate_bev_from_map(
    map_data: Union[Dict, MapProcessor],
    world2ego: np.ndarray,
    bev_height: int = 128,
    bev_width: int = 256,
    resolution: float = 0.25,
    coverage_behind: float = 0.0,
    lane_thickness: float = 0.3,
    max_distance: float = 100.0,
    use_legacy: bool = False,
) -> np.ndarray:
    """
    Generate BEV semantic map from vectorized map data.
    By default uses high-performance MapProcessor implementation.

    Args:
        map_data: Dictionary containing road data or MapProcessor instance
        world2ego: 4x4 transformation matrix from world to ego
        bev_height: Height of BEV map in pixels
        bev_width: Width of BEV map in pixels
        resolution: Meters per pixel
        coverage_behind: Coverage behind ego in meters
        lane_thickness: Lane line thickness in meters
        max_distance: Maximum distance to consider map elements
        use_legacy: If True, uses the original implementation (for testing)

    Returns:
        BEV semantic map with shape (H, W) and values 0-6
    """
    if use_legacy and isinstance(map_data, dict):
        return generate_bev_from_map_legacy(
            map_data, world2ego, bev_height, bev_width, resolution,
            coverage_behind, lane_thickness, max_distance
        )
    
    # Use MapProcessor for performance
    if isinstance(map_data, dict):
        processor = MapProcessor(map_data)
    else:
        processor = map_data
    
    return processor.generate_bev(
        world2ego, bev_height, bev_width, resolution,
        coverage_behind, lane_thickness, max_distance
    )
