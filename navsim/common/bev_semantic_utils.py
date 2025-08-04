"""
Utilities for generating BEV semantic maps at the dataset level.

This is the CORRECT approach - BEV generation happens during data loading,
not through model modifications.
"""

# TODO: this script should be merge with bev_map_utils.py, or other refacotring method

import numpy as np
import cv2
from typing import Tuple, Optional


# TODO: to b2d constant
# CARLA to BEV semantic class mapping
CARLA_TO_BEV_MAPPING = {
    # Background (0)
    0: 0,  # Unlabeled
    3: 0,  # Building
    9: 0,  # Vegetation
    10: 0,  # Terrain
    11: 0,  # Sky
    22: 0,  # Other
    23: 0,  # Water
    26: 0,  # Bridge
    27: 0,  # RailTrack
    # Road (1)
    1: 1,  # Roads
    24: 1,  # RoadLine
    25: 1,  # Ground
    # Walkways (2)
    2: 2,  # SideWalks
    # Lane centerlines (3) - special handling needed
    # Will be generated from trajectory or left as road
    # Static objects (4)
    4: 4,  # Wall
    5: 4,  # Fence
    6: 4,  # Pole
    7: 4,  # TrafficLight
    8: 4,  # TrafficSign
    20: 4,  # Static
    21: 4,  # Dynamic (when static)
    28: 4,  # GuardRail
    # Vehicles (5)
    14: 5,  # Car
    15: 5,  # Truck
    16: 5,  # Bus (if present)
    17: 5,  # Train (if present)
    18: 5,  # Motorcycle
    19: 5,  # Bicycle
    # Pedestrians (6)
    12: 6,  # Pedestrian
    13: 6,  # Rider
}


def ego_to_bev_coordinates(
    points_ego: np.ndarray, bev_height: int = 128, bev_width: int = 256, resolution: float = 0.25
) -> np.ndarray:
    """
    Convert ego-centric coordinates to BEV pixel coordinates.

    Args:
        points_ego: Nx2 array of (x, y) points in ego coordinates (meters)
        bev_height: Height of BEV map in pixels
        bev_width: Width of BEV map in pixels
        resolution: Meters per pixel

    Returns:
        Nx2 array of (row, col) pixel coordinates
    """
    # In ego coordinates:
    # x: forward (positive forward)
    # y: left (positive left)

    # In BEV image:
    # row: 0 at top (far ahead), increases downward (toward ego)
    # col: 0 at left, increases rightward

    # The BEV covers:
    # - Forward/Backward: ±32m (256 pixels * 0.25 m/pixel for full view)
    # - Lateral: ±32m (256 pixels * 0.25 m/pixel)
    # - For 128x256 BEV: Forward: +32m, Backward: 0m (front-only view)
    #
    # Ego vehicle is at CENTER of the full 360° view:
    # - For 256x256: row = 128, col = 128 (center)
    # - For 128x256: row = 128 (bottom edge), col = 128 (center)

    # Convert coordinates based on BEV dimensions
    if bev_height == 256 and bev_width == 256:
        # Full 360° BEV: ego at center
        rows = bev_height / 2 - points_ego[:, 0] / resolution
        cols = bev_width / 2 + points_ego[:, 1] / resolution
    else:
        # Front-only BEV (128x256): ego at bottom center
        rows = (bev_height - 1) - points_ego[:, 0] / resolution
        cols = bev_width / 2 + points_ego[:, 1] / resolution

    pixels = np.stack([rows, cols], axis=1).astype(np.int32)

    return pixels


def draw_rotated_box(
    image: np.ndarray, center: Tuple[int, int], size: Tuple[int, int], angle: float, value: int
) -> np.ndarray:
    """
    Draw a filled rotated rectangle on the image.

    Args:
        image: Image to draw on
        center: (row, col) center of box in image coordinates
        size: (height, width) of box in pixels
        angle: Rotation angle in radians
        value: Fill value

    Returns:
        Image with box drawn
    """
    # Create rotation matrix
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Box corners (relative to center)
    # Using row-major convention: height corresponds to Y, width to X
    h, w = size
    corners = np.array([[-h / 2, -w / 2], [h / 2, -w / 2], [h / 2, w / 2], [-h / 2, w / 2]])

    # Rotate corners
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = corners @ rot_matrix.T

    # Translate to absolute position
    # Input center is (row, col), but we need to be consistent
    # corners are in (y, x) format matching row-major indexing
    abs_corners = rotated_corners + np.array([[center[0], center[1]]])

    # OpenCV expects points in (x, y) format, so swap columns
    cv2_corners = abs_corners[:, [1, 0]].astype(np.int32)

    # Draw filled polygon
    cv2.fillPoly(image, [cv2_corners], value)  # type: ignore

    return image


def generate_road_mask_from_trajectory(
    trajectory: np.ndarray,
    bev_height: int = 128,
    bev_width: int = 256,
    resolution: float = 0.25,
    road_width: float = 4.0,
) -> np.ndarray:
    """
    Generate road mask from ego trajectory.

    Args:
        trajectory: Nx3 array of (x, y, heading) waypoints
        bev_height: Height of BEV map
        bev_width: Width of BEV map
        resolution: Meters per pixel
        road_width: Width of road in meters

    Returns:
        Binary mask with road area
    """
    road_mask = np.zeros((bev_height, bev_width), dtype=np.uint8)

    # Convert trajectory to pixel coordinates
    traj_pixels = ego_to_bev_coordinates(trajectory[:, :2], bev_height, bev_width, resolution)

    # Filter out points outside image
    valid_mask = (
        (traj_pixels[:, 0] >= 0)
        & (traj_pixels[:, 0] < bev_height)
        & (traj_pixels[:, 1] >= 0)
        & (traj_pixels[:, 1] < bev_width)
    )
    traj_pixels = traj_pixels[valid_mask]

    if len(traj_pixels) < 2:
        return road_mask

    # Draw thick polyline for road
    # OpenCV expects (x,y) = (col,row) format
    points = traj_pixels[:, [1, 0]].astype(np.int32)
    thickness = int(road_width / resolution)
    cv2.polylines(
        road_mask, [points], isClosed=False, color=1, thickness=thickness  # type: ignore
    )  # type: ignore

    return road_mask


def generate_agent_mask(
    agents: np.ndarray,
    agent_labels: np.ndarray,
    bev_height: int = 128,
    bev_width: int = 256,
    resolution: float = 0.25,
) -> np.ndarray:
    """
    Generate vehicle mask from detected agents.

    Args:
        agents: Nx5 array of (x, y, heading, length, width) for each agent
        agent_labels: N boolean array indicating valid agents
        bev_height: Height of BEV map
        bev_width: Width of BEV map
        resolution: Meters per pixel

    Returns:
        Binary mask with vehicle areas
    """
    vehicle_mask = np.zeros((bev_height, bev_width), dtype=np.uint8)

    # Process each valid agent
    for agent, valid in zip(agents, agent_labels):
        if not valid:
            continue

        x, y, heading, length, width = agent

        # Convert center to pixel coordinates
        center_pixel = ego_to_bev_coordinates(
            np.array([[x, y]]), bev_height, bev_width, resolution
        )[0]

        # Skip if outside image
        if (
            center_pixel[0] < 0
            or center_pixel[0] >= bev_height
            or center_pixel[1] < 0
            or center_pixel[1] >= bev_width
        ):
            continue

        # Convert size to pixels
        length_pixels = int(length / resolution)
        width_pixels = int(width / resolution)

        # Draw rotated box
        draw_rotated_box(
            vehicle_mask,
            center=(center_pixel[0], center_pixel[1]),
            size=(length_pixels, width_pixels),
            angle=-heading,  # Negative because of coordinate system
            value=1,
        )

    return vehicle_mask


# TODO: Shoud not be so many None, can be more strict
def generate_simple_bev_semantic(
    agents: np.ndarray,
    agent_labels: np.ndarray,
    agent_types: np.ndarray,
    bev_height: int,
    bev_width: int,
    resolution: float,
    map_bev: np.ndarray,
) -> np.ndarray:
    """
    Generate a simple BEV semantic map from trajectory and agents.

    This is a fallback method when we don't have access to semantic cameras
    or vectorized map data.

    Args:
        trajectory: Nx3 array of (x, y, heading) future waypoints
        agents: Mx5 array of (x, y, heading, length, width) for other agents
        agent_labels: M boolean array indicating valid agents
        agent_types: M array of NavSim class IDs (5=vehicle, 6=pedestrian)
        bev_height: Height of BEV map
        bev_width: Width of BEV map
        resolution: Meters per pixel
        map_bev: Optional pre-generated BEV from map data

    Returns:
        BEV semantic map with shape (H, W) and values 0-6
    """
    # Initialize with background or map data
    if map_bev is not None:
        bev_map = map_bev.copy()
    else:
        raise ValueError("map_bev must be provided for simple BEV generation")

    # Add agents with proper class types (vehicles and pedestrians)
    if agents is not None and agent_labels is not None:
        if agent_types is not None:
            # Use agent types to render different classes
            for i in range(len(agent_labels)):
                if agent_labels[i]:
                    agent_class = agent_types[i] if i < len(agent_types) else 5
                    if agent_class in [5, 6]:  # Only render vehicles and pedestrians
                        agent_mask = generate_agent_mask(
                            agents[i : i + 1], np.array([True]), bev_height, bev_width, resolution
                        )
                        bev_map[agent_mask > 0] = agent_class
        else:
            # Fallback: treat all agents as vehicles
            vehicle_mask = generate_agent_mask(
                agents, agent_labels, bev_height, bev_width, resolution
            )
            bev_map[vehicle_mask > 0] = 5  # Vehicle class
    else:
        raise ValueError("Agents and agent labels must be provided for BEV generation")

    # Ensure integer type for class labels
    return bev_map.astype(np.uint8)
