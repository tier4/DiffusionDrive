from navsim.common.bench2drive_constants import (
    LANE_TYPE_TO_BEV_CLASS,
    TRIGGER_TYPE_TO_BEV_CLASS,
)

import numpy as np
import cv2
from pathlib import Path
import logging
from scipy.spatial import KDTree

from typing import Dict, Tuple, Optional, Union
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from navsim.common.bench2drive_constants import BEV_SEMANTIC_RESOLUTION

logger = logging.getLogger(__name__)

# BEV view type constants
BEV_VIEW_FRONT = "front"  # Front-only view (ego at bottom-center)
BEV_VIEW_REAR = "rear"  # Rear-only view (ego at top-center)
BEV_VIEW_FULL = "full"  # Full 360° view (ego at center)


def get_ego_pixel_position(
    bev_height: int, bev_width: int, view_type: str = BEV_VIEW_FRONT
) -> Tuple[int, int]:
    """
    Get ego vehicle pixel position based on view type.

    Args:
        bev_height: Height of BEV map in pixels
        bev_width: Width of BEV map in pixels
        view_type: Type of BEV view (front, rear, or full)

    Returns:
        Tuple of (row, col) for ego position in pixels
    """
    if view_type == BEV_VIEW_FRONT:
        # Front-only view: ego at bottom-center
        return (bev_height - 1, bev_width // 2)
    elif view_type == BEV_VIEW_REAR:
        # Rear-only view: ego at top-center
        return (0, bev_width // 2)
    elif view_type == BEV_VIEW_FULL:
        # Full 360° view: ego at center
        return (bev_height // 2, bev_width // 2)
    else:
        raise ValueError(
            f"Invalid view_type: {view_type}. Must be one of {BEV_VIEW_FRONT}, {BEV_VIEW_REAR}, {BEV_VIEW_FULL}"
        )


def calculate_recommended_max_distance(
    bev_height: int,
    bev_width: int,
    resolution: float,
    view_type: str = BEV_VIEW_FRONT,
    safety_factor: float = 8,
) -> float:
    """
    Calculate recommended max_distance for KDTree queries based on BEV coverage.

    The max_distance needs to be larger than the actual BEV visible range because:
    1. Lane segments are indexed by their center points in the KDTree
    2. A lane segment with center far from ego might still pass through the BEV area
    3. Curved roads can have segment centers significantly offset from the actual lane path

    Args:
        bev_height: Height of BEV map in pixels
        bev_width: Width of BEV map in pixels
        resolution: Meters per pixel
        view_type: Type of BEV view (front, rear, or full)
        safety_factor: Multiplier to ensure we capture all relevant lanes (default 8)
                      Higher values capture more distant lane segments but may reduce performance

    Returns:
        Recommended max_distance in meters
    """
    # Calculate the actual BEV coverage in meters
    if view_type in [BEV_VIEW_FRONT, BEV_VIEW_REAR]:
        # For front/rear views, coverage is one-directional
        longitudinal_range = bev_height * resolution
        lateral_range = bev_width * resolution
    else:  # BEV_VIEW_FULL
        # For full view, coverage is bidirectional
        longitudinal_range = bev_height * resolution
        lateral_range = bev_width * resolution

    # Calculate diagonal distance from ego to farthest visible corner
    # This is the minimum distance needed to see all BEV area
    diagonal_distance = np.sqrt(longitudinal_range**2 + (lateral_range / 2) ** 2)

    # Apply safety factor to account for:
    # - Lane segments with centers outside BEV but passing through it
    # - Curved roads where segment centers are offset
    # - Discrete sampling of continuous lanes
    recommended_distance = diagonal_distance * safety_factor

    # Ensure minimum distance for urban scenarios
    # (intersections, roundabouts often need wider search)
    min_distance = 100.0

    return max(recommended_distance, min_distance)


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
    map_dict = dict(map_data["arr"])

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
    points: np.ndarray, ego_points: np.ndarray, ego_heading_rad: float, left_to_right: bool = False
) -> np.ndarray:
    """
    Transform points from world coordinates to ego-centric coordinates.
    Default is left-handed coordinate system (e.g. Carla coordinate),
    can convert to right-handed (e.g. NAVSIM coordinate).

    Args:
        points: Nx3 array of points in world coordinates
        ego_points: (3,) array of ego position in world coordinates [x, y, z]
        ego_heading_rad: Current ego heading in radians
        left_to_right: Whether to convert from left-handed to right-handed

    Returns:
        Nx3 array of points in ego-centric coordinates
    """
    if points.shape[-1] != 3 or not (points.ndim == 2 or points.ndim == 1):
        raise ValueError(
            f"Points must be Nx3 array with (x, y, z) coordinates. Instead got {points.shape}"
        )
    if ego_points.shape[0] != 3 or ego_points.ndim != 1:
        raise ValueError(f"Ego points must be a 3-element array. Instead got {ego_points.shape}")

    if points.ndim == 1:
        points = points.reshape(1, -1)  # Ensure points is Nx3

    ego_x_world, ego_y_world, ego_z_world = ego_points

    # Calculate relative world position
    rel_world_pos = [
        points[:, 0] - ego_x_world,
        points[:, 1] - ego_y_world,
        points[:, 2],
    ]

    # Pre-calculate sine and cosine for efficiency
    ego_heading_rad = normalize_angle(ego_heading_rad)  # Normalize angle to [-π, π]
    cos_heading = np.cos(ego_heading_rad)
    sin_heading = np.sin(ego_heading_rad)

    # Convert from left-handed to right-handed if needed
    if left_to_right:
        # need flip y axis and change the rotation matrix
        raise NotImplementedError("The right-handed conversion is not implemented yet.")
    else:
        # Transform to ego coordinates using rotation matrix
        ego_x = rel_world_pos[0] * cos_heading + rel_world_pos[1] * sin_heading
        ego_y = -rel_world_pos[0] * sin_heading + rel_world_pos[1] * cos_heading
    ego_z = rel_world_pos[2]

    return np.column_stack([ego_x, ego_y, ego_z])  # pyright: ignore[reportCallIssue]


def transform_heading_to_ego(
    object_heading_rad: float, ego_heading_rad: float, normalize: bool = True
) -> float:
    """
    Transform object heading from world to ego-relative coordinates.

    Args:
        object_heading_rad: Object heading in world coordinates (radians)
        ego_heading_rad: Ego vehicle heading in world coordinates (radians)
        normalize: Whether to normalize result to [-π, π] range (default: True)

    Returns:
        Relative heading in ego coordinates (radians)
    """
    relative_heading = object_heading_rad - ego_heading_rad

    if normalize:
        relative_heading = normalize_angle(relative_heading)

    return relative_heading


def ego_to_bev_pixels(
    points_ego: np.ndarray,
    bev_height: int,
    bev_width: int,
    resolution: float,
    view_type: str = BEV_VIEW_FRONT,
) -> np.ndarray:
    """
    Convert ego-centric coordinates to BEV pixel coordinates.

    Args:
        points_ego: Nx2 or Nx3 array of points in ego coordinates (meters)
                   In ego coordinates: X=forward, Y=right (CARLA coordinates)
        bev_height: Height of BEV map in pixels
        bev_width: Width of BEV map in pixels
        resolution: Meters per pixel
        view_type: Type of BEV view (front, rear, or full)

    Returns:
        Nx2 array of (row, col) pixel coordinates
    """
    # Get ego position in pixels based on view type
    ego_row, ego_col = get_ego_pixel_position(bev_height, bev_width, view_type)

    # Extract x, y coordinates
    x_ego = points_ego[:, 0]  # Forward distance in meters
    y_ego = points_ego[:, 1]  # Left distance in meters (CARLA coordinates)

    # Convert to pixels
    # X (forward in ego) decreases row (moves up in image)
    # Y (right in ego) increases column (moves right in image) - CARLA coordinates
    rows = ego_row - x_ego / resolution
    cols = ego_col + y_ego / resolution

    pixels = np.stack([rows, cols], axis=1).astype(np.int32)

    return pixels


def filter_points_in_range(
    points_ego: np.ndarray,
    bev_height: int,
    bev_width: int,
    resolution: float,
    view_type: str = BEV_VIEW_FRONT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter points to those within BEV range based on view type.

    Args:
        points_ego: Nx2 or Nx3 array of points in ego coordinates
        bev_height: Height of BEV map
        bev_width: Width of BEV map
        resolution: Meters per pixel
        view_type: Type of BEV view (front, rear, or full)

    Returns:
        Tuple of (filtered points, valid mask)
    """
    x = points_ego[:, 0]  # Forward distance
    y = points_ego[:, 1]  # Right distance

    # Calculate bounds in meters based on view type
    if view_type == BEV_VIEW_FRONT:
        # Front-only view: covers area ahead of ego
        x_min = 0
        x_max = bev_height * resolution
    elif view_type == BEV_VIEW_REAR:
        # Rear-only view: covers area behind ego
        x_min = -bev_height * resolution
        x_max = 0
    elif view_type == BEV_VIEW_FULL:
        # Full 360° view: ego at center
        half_range = bev_height * resolution / 2
        x_min = -half_range
        x_max = half_range
    else:
        raise ValueError(f"Invalid view_type: {view_type}")

    # Lateral bounds (same for all view types)
    y_min = -bev_width * resolution / 2
    y_max = bev_width * resolution / 2

    # Create mask
    valid_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    filtered_points = points_ego[valid_mask]

    return filtered_points, valid_mask


def draw_lane_on_bev(
    bev_map: np.ndarray,
    lane_points: np.ndarray,
    lane_type: str,
    resolution: float = BEV_SEMANTIC_RESOLUTION,
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
    # But lane_points is in [row, col] format, so we need to swap
    cv2_points = lane_points[:, [1, 0]].astype(np.int32)

    cv2.polylines(
        bev_map,
        [cv2_points],
        isClosed=False,
        color=semantic_class,  # int can be asign to color # type: ignore
        thickness=thickness_pixels,
    )  # type: ignore

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
    # OpenCV expects (x,y) = (col,row) format
    # But bottom_points is in [row, col] format, so we need to swap
    cv2_points = bottom_points[:, [1, 0]].astype(np.int32)

    # Draw filled polygon
    cv2.fillPoly(bev_map, [cv2_points], color=semantic_class)  # type: ignore

    return bev_map


def generate_full_bev_from_map(
    map_data: "MapProcessor",
    ego_points: np.ndarray,
    ego_heading_rad: float,
    full_height: int = 256,
    full_width: int = 256,
    resolution: float = BEV_SEMANTIC_RESOLUTION,
    **kwargs,
) -> np.ndarray:
    """
    Generate full 360-degree BEV map from vectorized map data.

    Args:
        map_data: Dictionary containing road data or MapProcessor instance
        ego_points: (3,) array of ego position in world coordinates [x, y, z]
        ego_heading_rad: Current ego heading in radians
        full_height: Height of full BEV map
        full_width: Width of full BEV map
        resolution: Meters per pixel
        **kwargs: Additional arguments for generate_bev_from_map

    Returns:
        Full BEV semantic map with shape (full_height, full_width)
    """
    # For full BEV, use full view type
    return generate_bev_from_map(
        map_data=map_data,
        ego_points=ego_points,
        ego_heading_rad=ego_heading_rad,
        bev_height=full_height,
        bev_width=full_width,
        resolution=resolution,
        view_type=BEV_VIEW_FULL,
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
            f"MapProcessor initialized with {len(self._flat_lanes)} lanes "
            f"and {len(self._flat_triggers)} triggers."
        )

    def generate_bev(
        self,
        ego_points: np.ndarray,
        ego_heading_rad: float,
        bev_height: int = 128,
        bev_width: int = 256,
        resolution: float = BEV_SEMANTIC_RESOLUTION,
        view_type: str = BEV_VIEW_FRONT,
        lane_thickness: float = 0.3,
        max_distance: Optional[float] = None,
    ) -> np.ndarray:
        """
        Efficiently generates a BEV map by querying nearby map elements.

        Args:
            ego_points: (3,) array of ego position in world coordinates [x, y, z]
            ego_heading_rad: Current ego heading in radians
            bev_height: Height of BEV map in pixels
            bev_width: Width of BEV map in pixels
            resolution: Meters per pixel
            view_type: Type of BEV view (front, rear, or full)
            lane_thickness: Lane line thickness in meters
            max_distance: Maximum distance to consider map elements. If None, automatically
                        calculated based on BEV coverage with safety factor.

        Returns:
            BEV semantic map
        """
        # Calculate recommended max_distance if not provided
        if max_distance is None:
            max_distance = calculate_recommended_max_distance(
                bev_height, bev_width, resolution, view_type
            )
        bev_map = np.zeros((bev_height, bev_width), dtype=np.uint8)

        # Use the provided ego position in world coordinates to query the KDTree
        ego_pos_world = ego_points[:2]  # Use x, y coordinates for KDTree query

        # --- Process Lanes ---
        if self.lane_kdtree:
            # Query KDTree to find indices of lanes within max_distance
            nearby_indices = self.lane_kdtree.query_ball_point(ego_pos_world, r=max_distance, p=2)

            for idx in nearby_indices:
                segment = self._flat_lanes[idx]
                points_ego = transform_points_to_ego(
                    segment["points"], ego_points, ego_heading_rad
                )
                pixels = ego_to_bev_pixels(
                    points_ego, bev_height, bev_width, resolution, view_type
                )
                bev_map = draw_lane_on_bev(
                    bev_map, pixels, segment["type"], resolution, lane_thickness
                )

        # --- Process Triggers ---
        if self.trigger_kdtree:
            nearby_indices = self.trigger_kdtree.query_ball_point(
                ego_pos_world, r=max_distance, p=2
            )

            for idx in nearby_indices:
                trigger = self._flat_triggers[idx]
                points_ego = transform_points_to_ego(
                    trigger["points"], ego_points, ego_heading_rad
                )
                pixels = ego_to_bev_pixels(
                    points_ego, bev_height, bev_width, resolution, view_type
                )
                bev_map = draw_trigger_on_bev(bev_map, pixels, trigger["type"])

        return bev_map


# --- DEFAULT IMPLEMENTATION (USES MAPPROCESSOR) ---
def generate_bev_from_map(
    map_data: Union[Dict, MapProcessor],
    ego_points: np.ndarray,
    ego_heading_rad: float,
    bev_height: int = 128,
    bev_width: int = 256,
    resolution: float = BEV_SEMANTIC_RESOLUTION,
    view_type: str = BEV_VIEW_FRONT,
    lane_thickness: float = 0.3,
    max_distance: Optional[float] = None,
) -> np.ndarray:
    """
    Generate BEV semantic map from vectorized map data.
    By default uses high-performance MapProcessor implementation.

    Args:
        map_data: Dictionary containing road data or MapProcessor instance
        ego_points: (3,) array of ego position in world coordinates [x, y, z]
        ego_heading_rad: Current ego heading in radians
        bev_height: Height of BEV map in pixels
        bev_width: Width of BEV map in pixels
        resolution: Meters per pixel
        view_type: Type of BEV view (front, rear, or full)
        lane_thickness: Lane line thickness in meters
        max_distance: Maximum distance to consider map elements. If None, automatically
                     calculated based on BEV coverage with safety factor.

    Returns:
        BEV semantic map with shape (H, W) and values 0-6
    """

    # Calculate recommended max_distance if not provided
    if max_distance is None:
        max_distance = calculate_recommended_max_distance(
            bev_height, bev_width, resolution, view_type
        )
        logger.debug(f"Using calculated max_distance: {max_distance:.1f}m")

    # Use MapProcessor for performance
    if isinstance(map_data, dict):
        processor = MapProcessor(map_data)
    else:
        processor = map_data

    bev_map = processor.generate_bev(
        ego_points,
        ego_heading_rad,
        bev_height,
        bev_width,
        resolution,
        view_type,
        lane_thickness,
        max_distance,
    )

    # Validate semantic map values are valid semantic classes
    valid_classes = (
        set(LANE_TYPE_TO_BEV_CLASS.values()) | set(TRIGGER_TYPE_TO_BEV_CLASS.values()) | {0}
    )  # Include background class
    unique_values = np.unique(bev_map)
    invalid_values = [val for val in unique_values if val not in valid_classes]
    if invalid_values:
        raise ValueError(
            f"BEV map contains invalid semantic class values: {invalid_values}. "
            "Valid classes are: {sorted(valid_classes)}"
        )

    return bev_map
