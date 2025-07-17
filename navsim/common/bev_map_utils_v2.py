"""
Utilities for generating BEV semantic maps from Bench2Drive vectorized map data.

This module handles the generation of BEV semantic maps using lane lines and
trigger volumes from Bench2Drive HD maps.

MODIFIED: Includes a MapProcessor class that uses a KDTree for high-performance
spatial queries, avoiding full map scans per frame.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from scipy.spatial import KDTree  # Import KDTree

logger = logging.getLogger(__name__)

# Constants remain the same
LANE_TYPE_TO_BEV_CLASS = {"Broken": 1, "Solid": 1, "SolidSolid": 1, "Center": 3}
TRIGGER_TYPE_TO_BEV_CLASS = {"TrafficLight": 4, "StopSign": 4}


# Original utility functions remain mostly the same
def load_map_data(map_path: Union[str, Path]) -> Dict:
    map_path = Path(map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")
    map_data = np.load(str(map_path), allow_pickle=True)
    return dict(map_data["arr"][()])


def extract_lane_points(lane_segment: Dict) -> Tuple[np.ndarray, str]:
    points_raw = lane_segment["Points"]
    lane_type = lane_segment.get("Type", "Unknown")
    points = np.array([p[0] for p in points_raw if isinstance(p, tuple) and len(p) >= 1])
    return points, lane_type


def extract_trigger_points(trigger: Dict) -> Tuple[np.ndarray, str]:
    return np.array(trigger["Points"]), trigger.get("Type", "Unknown")


def transform_points_to_ego(
    points: np.ndarray, world2ego: np.ndarray, left_to_right: bool = True
) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    points_ego_h = points_h @ world2ego.T
    points_ego = points_ego_h[:, :3]
    if left_to_right:
        points_ego[:, 1] = -points_ego[:, 1]
    return points_ego


def ego_to_bev_pixels(
    points_ego: np.ndarray,
    bev_height: int,
    bev_width: int,
    resolution: float,
    coverage_behind: float = 0.0,
) -> np.ndarray:
    x, y = points_ego[:, 0], points_ego[:, 1]
    rows = (bev_height - 1) - (x + coverage_behind) / resolution
    cols = bev_width / 2 - y / resolution  # Flipped y-axis for pixel coords
    return np.stack([rows, cols], axis=1).astype(np.int32)


def draw_lane_on_bev(
    bev_map: np.ndarray,
    lane_points: np.ndarray,
    lane_type: str,
    resolution: float,
    thickness: float,
) -> None:
    if len(lane_points) < 2:
        return
    semantic_class = LANE_TYPE_TO_BEV_CLASS.get(lane_type, 1)
    thickness_pixels = max(1, int(thickness / resolution))
    points = lane_points[:, [1, 0]].astype(np.int32)  # OpenCV expects (col, row)
    cv2.polylines(
        bev_map, [points], isClosed=False, color=semantic_class, thickness=thickness_pixels
    )


def draw_trigger_on_bev(
    bev_map: np.ndarray, trigger_points: np.ndarray, trigger_type: str
) -> None:
    if len(trigger_points) < 4:
        return
    semantic_class = TRIGGER_TYPE_TO_BEV_CLASS.get(trigger_type, 4)
    bottom_points = trigger_points[:4] if len(trigger_points) >= 8 else trigger_points
    points = bottom_points[:, [1, 0]].astype(np.int32)
    cv2.fillPoly(bev_map, [points], color=semantic_class)


# --- NEW HIGH-PERFORMANCE IMPLEMENTATION ---
class MapProcessor:
    """
    Pre-processes map data and provides a fast method to generate BEV maps.
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
                    points_ego, bev_height, bev_width, resolution, -coverage_behind
                )
                draw_lane_on_bev(bev_map, pixels, segment["type"], resolution, lane_thickness)

        # --- Process Triggers ---
        if self.trigger_kdtree:
            nearby_indices = self.trigger_kdtree.query_ball_point(ego_pos_world, r=max_distance)

            for idx in nearby_indices:
                trigger = self._flat_triggers[idx]
                points_ego = transform_points_to_ego(trigger["points"], world2ego)
                pixels = ego_to_bev_pixels(
                    points_ego, bev_height, bev_width, resolution, -coverage_behind
                )
                draw_trigger_on_bev(bev_map, pixels, trigger["type"])

        return bev_map


# --- LEGACY FUNCTIONS (kept for API compatibility but are slow) ---
def generate_bev_from_map(*args, **kwargs):
    # This function is now just a placeholder for the old, slow method.
    # It's recommended to use the MapProcessor class instead.
    raise NotImplementedError(
        "Use the MapProcessor class for efficient BEV generation. See documentation."
    )


def generate_full_bev_from_map(
    map_processor: MapProcessor,  # Changed to accept the processor
    world2ego: np.ndarray,
    full_height: int = 256,
    full_width: int = 256,
    resolution: float = 0.25,
    **kwargs,
) -> np.ndarray:
    coverage_behind = full_height * resolution / 2
    # Note: ego_to_bev_pixels now expects negative coverage_behind
    # to correctly handle the coordinate system shift.
    return map_processor.generate_bev(
        world2ego=world2ego,
        bev_height=full_height,
        bev_width=full_width,
        resolution=resolution,
        coverage_behind=-coverage_behind,  # Centered coverage
        **kwargs,
    )


def extract_front_half_bev(full_bev: np.ndarray, front_ratio: float = 0.5) -> np.ndarray:
    """
    Extract front half of BEV map for NavSim format.

    Args:
        full_bev: Full BEV map (H, W)
        front_ratio: Ratio of front coverage (default 0.5)

    Returns:
        Front half BEV with shape (H*ratio, W)
    """
    height = full_bev.shape[0]
    front_height = int(height * front_ratio)

    # Front half is the top portion of the full BEV array
    return full_bev[:front_height, :].copy()
