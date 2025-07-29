"""
Utilities for generating BEV semantic maps at the dataset level.

This is the CORRECT approach - BEV generation happens during data loading,
not through model modifications.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional


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


def map_carla_to_bev_semantic(carla_semantic: np.ndarray) -> np.ndarray:
    """
    Map CARLA semantic segmentation values to BEV semantic classes.

    Args:
        carla_semantic: Array with CARLA semantic values (0-28)

    Returns:
        Array with BEV semantic values (0-6)
    """
    bev_semantic = np.zeros_like(carla_semantic, dtype=np.uint8)

    for carla_val, bev_val in CARLA_TO_BEV_MAPPING.items():
        mask = carla_semantic == carla_val
        bev_semantic[mask] = bev_val

    return bev_semantic


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
    # - Forward: 0 to +32m (128 pixels * 0.25 m/pixel)
    # - Lateral: -32m to +32m (256 pixels * 0.25 m/pixel)
    #
    # Ego vehicle is at:
    # - row = bev_height - 1 (bottom of image)
    # - col = bev_width / 2 (center of image)

    # Convert coordinates
    # For row: ego is at bottom, forward is up
    rows = (bev_height - 1) - points_ego[:, 0] / resolution

    # For col: ego is at center, left is positive in ego coords but right in image
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
        center: (row, col) center of box
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
    h, w = size
    corners = np.array([[-h / 2, -w / 2], [h / 2, -w / 2], [h / 2, w / 2], [-h / 2, w / 2]])

    # Rotate corners
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = corners @ rot_matrix.T

    # Translate to absolute position
    # Note: OpenCV uses (x,y) = (col,row) convention
    abs_corners = rotated_corners + np.array([[center[1], center[0]]])
    abs_corners = abs_corners.astype(np.int32)

    # Draw filled polygon
    cv2.fillPoly(image, [abs_corners], value)

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
    cv2.polylines(road_mask, [points], isClosed=False, color=1, thickness=thickness)

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


def generate_simple_bev_semantic(
    trajectory: Optional[np.ndarray] = None,
    agents: Optional[np.ndarray] = None,
    agent_labels: Optional[np.ndarray] = None,
    agent_types: Optional[np.ndarray] = None,
    bev_height: int = 128,
    bev_width: int = 256,
    resolution: float = 0.25,
    map_bev: Optional[np.ndarray] = None,
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
        bev_height: Height of BEV map (default: 128)
        bev_width: Width of BEV map (default: 256)
        resolution: Meters per pixel (default: 0.25)
        map_bev: Optional pre-generated BEV from map data

    Returns:
        BEV semantic map with shape (H, W) and values 0-6
    """
    # Initialize with background or map data
    if map_bev is not None:
        bev_map = map_bev.copy()
    else:
        bev_map = np.zeros((bev_height, bev_width), dtype=np.uint8)

        # Add road from trajectory only if no map data
        if trajectory is not None and len(trajectory) > 0:
            road_mask = generate_road_mask_from_trajectory(
                trajectory, bev_height, bev_width, resolution
            )
            bev_map[road_mask > 0] = 1  # Road class

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
            vehicle_mask = generate_vehicle_mask_from_agents(
                agents, agent_labels, bev_height, bev_width, resolution
            )
            bev_map[vehicle_mask > 0] = 5  # Vehicle class

    # Ensure integer type for class labels
    return bev_map.astype(np.uint8)


# Backward compatibility alias
generate_vehicle_mask_from_agents = generate_agent_mask


def project_semantic_to_bev(
    semantic_images: Dict[str, np.ndarray],
    camera_intrinsics: Dict[str, np.ndarray],
    camera_extrinsics: Dict[str, np.ndarray],
    bev_height: int = 128,
    bev_width: int = 256,
    resolution: float = 0.25,
    camera_height: float = 1.5,
) -> np.ndarray:
    """
    Project perspective semantic segmentation images to BEV.

    This is a more advanced method that uses camera geometry to project
    semantic segmentation from multiple cameras to BEV space.

    Args:
        semantic_images: Dict of camera_name -> semantic image
        camera_intrinsics: Dict of camera_name -> 3x3 intrinsic matrix
        camera_extrinsics: Dict of camera_name -> 4x4 extrinsic matrix
        bev_height: Height of BEV map
        bev_width: Width of BEV map
        resolution: Meters per pixel
        camera_height: Assumed height of camera above ground

    Returns:
        BEV semantic map
    """
    # This is a placeholder for the more complex projection method
    # For now, return simple BEV
    return generate_simple_bev_semantic(
        bev_height=bev_height, bev_width=bev_width, resolution=resolution
    )


# Coordinate transformation utilities
def transform_carla_to_navsim(points_carla: np.ndarray) -> np.ndarray:
    """
    Transform points from CARLA (left-handed) to NavSim (right-handed) coordinates.

    Args:
        points_carla: Nx3 or Nx4 array of points

    Returns:
        Transformed points
    """
    points_navsim = points_carla.copy()
    points_navsim[:, 1] = -points_navsim[:, 1]  # Flip Y axis
    return points_navsim
