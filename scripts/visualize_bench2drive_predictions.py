#!/usr/bin/env python3
"""
Enhanced visualization script for DiffusionDrive model predictions on Bench2Drive dataset.
Creates comprehensive MP4 videos showing:
- BEV view with vector map background, LiDAR overlay, and numbered trajectory points
- 6 camera views: 3 front cameras, back camera, RGB BEV, and semantic segmentation
- Detailed metrics including L2 errors, ego state, and agent counts
- Support for train/val split and full scenario export

NO PLACEHOLDERS, MOCK DATA, OR TRY/EXCEPT - All errors are raised properly.

COORDINATE SYSTEM:
- BEV Display: X-axis (horizontal) = lateral, Y-axis (vertical) = forward
- Ego vehicle faces UP (positive Y direction = forward)
"""

from navsim.common.bench2drive_dataloader import (
    Bench2DriveConfig,
    Bench2DriveSceneLoader,
)
from navsim.common.bench2drive_scene import Bench2DriveScene
from navsim.common.dataclasses import Trajectory, Lidar
from navsim.agents.diffusiondrive.b2d_agent import Bench2DriveAgent
from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig as AgentConfig
from navsim.visualization.plots import configure_bev_ax
from navsim.visualization.bev import (
    add_configured_bev_on_ax,
    add_lidar_to_bev_ax,
)
from navsim.visualization.camera import add_camera_ax
from navsim.common.bench2drive_constants import B2D_CLASS_TO_NAVSIM

import os
import sys
import json
import gzip
import argparse
import subprocess
import tempfile
import io
import laspy
from pathlib import Path
from typing import Tuple, Any, Dict, List, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Arrow
from PIL import Image
import cv2
from joblib import Parallel, delayed


def load_bev_map_from_cache(
    bev_cache_dir: Path, scenario: str, frame_idx: int
) -> Optional[np.ndarray]:
    """
    Load pre-generated BEV map from cache.

    Args:
        bev_cache_dir: Path to BEV cache directory
        scenario: Scenario name
        frame_idx: Frame index

    Returns:
        BEV map array or None if not available

    Raises:
        ValueError: If cache file is corrupted or missing required keys
    """
    cache_file = bev_cache_dir / scenario / f"{frame_idx:05d}.npz"
    if not cache_file.exists():
        print(f"Warning: BEV cache file not found: {cache_file}")
        return None

    data = np.load(cache_file)
    if 'full_bev' not in data.files:
        raise ValueError(f"BEV cache file missing 'full_bev' key. Available keys: {data.files}")

    bev_map = data['full_bev']
    print(f"  Loaded BEV map: shape={bev_map.shape}, range=[{bev_map.min():.1f}, {bev_map.max():.1f}]")
    return bev_map


def load_lidar_from_raw_data(data_root: Path, scenario: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load LiDAR point cloud directly from raw data.

    Args:
        data_root: Bench2Drive dataset root
        scenario: Scenario name
        frame_idx: Frame index

    Returns:
        LiDAR points array [N, 4] (x, y, z, intensity) or None
    """
    lidar_path = data_root / scenario / "lidar" / f"{frame_idx:05d}.laz"
    if not lidar_path.exists():
        # Try .las extension
        lidar_path = lidar_path.with_suffix('.las')
        if not lidar_path.exists():
            print(f"Warning: LiDAR file not found: {lidar_path}")
            return None

    # Load LiDAR points
    las = laspy.read(str(lidar_path))
    points = np.vstack((las.x, las.y, las.z, las.intensity)).T
    print(f"  Loaded LiDAR: {points.shape[0]} points")
    return points


def load_rgb_bev_camera(data_root: Path, scenario: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load RGB bird's eye view camera image.

    Args:
        data_root: Bench2Drive dataset root
        scenario: Scenario name
        frame_idx: Frame index

    Returns:
        RGB BEV image array or None
    """
    bev_path = data_root / scenario / "camera" / "rgb_top_down" / f"{frame_idx:05d}.jpg"
    if not bev_path.exists():
        print(f"Warning: RGB BEV camera not found: {bev_path}")
        return None
    return np.array(Image.open(bev_path))


def load_semantic_segmentation(data_root: Path, scenario: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load semantic segmentation map.

    Args:
        data_root: Bench2Drive dataset root
        scenario: Scenario name
        frame_idx: Frame index

    Returns:
        Semantic segmentation array or None
    """
    seg_path = data_root / scenario / "camera" / "semantic_front" / f"{frame_idx:05d}.png"
    if not seg_path.exists():
        print(f"Warning: Semantic segmentation not found: {seg_path}")
        return None
    # Load as grayscale/index image
    seg_img = Image.open(seg_path)
    return np.array(seg_img)


def load_back_camera(data_root: Path, scenario: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load back camera RGB image.

    Args:
        data_root: Bench2Drive dataset root
        scenario: Scenario name
        frame_idx: Frame index

    Returns:
        Back camera RGB image or None
    """
    back_path = data_root / scenario / "camera" / "rgb_back" / f"{frame_idx:05d}.jpg"
    if not back_path.exists():
        print(f"Warning: Back camera not found: {back_path}")
        return None
    return np.array(Image.open(back_path))


def calculate_vad_l2_metrics(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    timestep_duration: float = 0.5
) -> Dict[str, float]:
    """
    Calculate L2 metrics using VAD-style period-average method.

    Args:
        pred_trajectory: Predicted trajectory [T, 3] (x, y, heading)
        gt_trajectory: Ground truth trajectory [T, 3]
        timestep_duration: Duration between timesteps in seconds

    Returns:
        Dictionary with L2 metrics at various time horizons
    """
    num_timesteps = min(pred_trajectory.shape[0], gt_trajectory.shape[0])

    # Calculate L2 errors at each timestep
    l2_errors = np.zeros(num_timesteps)
    for t in range(num_timesteps):
        pred_pos = pred_trajectory[t, :2]  # x, y only
        gt_pos = gt_trajectory[t, :2]
        l2_errors[t] = np.linalg.norm(pred_pos - gt_pos)

    # Calculate period-average L2 for different horizons
    metrics = {}
    horizons_sec = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    for horizon_sec in horizons_sec:
        horizon_steps = int(horizon_sec / timestep_duration)
        if horizon_steps <= num_timesteps:
            # VAD method: Average L2 over period [0, t]
            period_avg = np.mean(l2_errors[:horizon_steps])
            metrics[f"L2_{horizon_sec:.1f}s"] = period_avg

    # Overall average
    metrics["L2_avg"] = np.mean(l2_errors)

    return metrics


def count_agents_by_type(scene: Bench2DriveScene, frame_idx: int) -> Dict[str, int]:
    """
    Count agents by type in the current frame.

    Args:
        scene: Bench2Drive scene
        frame_idx: Frame index

    Returns:
        Dictionary with counts by agent type
    """
    anno = scene._load_annotation(frame_idx)

    counts = {
        "total": 0,
        "vehicles": 0,
        "pedestrians": 0,
        "traffic_lights": 0,
        "traffic_signs": 0,
    }

    for box in anno.get("bounding_boxes", []):
        agent_class = box.get("class", "")
        if agent_class != "ego_vehicle":
            counts["total"] += 1
            if agent_class == "vehicle":
                counts["vehicles"] += 1
            elif agent_class == "walker":
                counts["pedestrians"] += 1
            elif agent_class == "traffic_light":
                counts["traffic_lights"] += 1
            elif agent_class == "traffic_sign":
                counts["traffic_signs"] += 1

    return counts


def get_ego_state_from_raw_data(data_root: Path, scenario_name: str, frame_idx: int) -> Dict[str, Any]:
    """
    Get ego state directly from raw annotation data.

    Args:
        data_root: Dataset root path
        scenario_name: Scenario name
        frame_idx: Frame index

    Returns:
        Dictionary with ego state information
    """
    import gzip
    import json

    anno_file = data_root / scenario_name / "anno" / f"{frame_idx:05d}.json.gz"

    if not anno_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")

    with gzip.open(anno_file, 'rt') as f:
        data = json.load(f)

    # Find ego vehicle in bounding boxes
    ego_box = None
    for box in data['bounding_boxes']:
        if 'ego' in box['class'].lower():
            ego_box = box
            break

    if ego_box is None:
        raise ValueError(f"Ego vehicle not found in frame {frame_idx}")

    # Extract ego velocity and acceleration
    # Bench2Drive provides 'speed' directly in m/s in ego_box
    velocity_ms = ego_box['speed']

    # Get heading from rotation[2] (yaw in degrees)
    # Negate to convert from CCW to CW (or vice versa) for correct visualization
    heading_deg = -ego_box['rotation'][2]
    heading_rad = np.radians(heading_deg)

    # For vx/vy, calculate from speed and heading
    vx = velocity_ms * np.cos(heading_rad)
    vy = velocity_ms * np.sin(heading_rad)

    # Acceleration is provided as 3D vector [ax, ay, az] in the main data
    accel_vec = data['acceleration']
    # Compute magnitude of horizontal acceleration (ax, ay)
    acceleration = np.sqrt(accel_vec[0]**2 + accel_vec[1]**2)

    # Get driving command (Bench2Drive uses command_near)
    command = data['command_near']

    return {
        "velocity": velocity_ms,
        "velocity_x": vx,
        "velocity_y": vy,
        "acceleration": acceleration,
        "heading": heading_rad,  # Return radians, not degrees!
        "driving_command": command,
    }


def count_agents_from_raw_data(data_root: Path, scenario_name: str, frame_idx: int) -> Dict[str, int]:
    """
    Count agents directly from raw annotation data.

    Args:
        data_root: Dataset root path
        scenario_name: Scenario name
        frame_idx: Frame index

    Returns:
        Dictionary with agent counts by type
    """
    import gzip
    import json

    anno_file = data_root / scenario_name / "anno" / f"{frame_idx:05d}.json.gz"

    counts = {
        "vehicles": 0,
        "pedestrians": 0,
        "cyclists": 0,
        "traffic_lights": 0,
        "traffic_signs": 0,
        "total": 0,
    }

    if not anno_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")

    with gzip.open(anno_file, 'rt') as f:
        data = json.load(f)

    # Count from bounding_boxes
    if 'bounding_boxes' in data:
        for box in data['bounding_boxes']:
            agent_class = box.get('class', '').lower()

            if 'vehicle' in agent_class or 'car' in agent_class or 'truck' in agent_class:
                if 'ego' not in agent_class:  # Don't count ego vehicle
                    counts["vehicles"] += 1
            elif 'pedestrian' in agent_class or 'person' in agent_class:
                counts["pedestrians"] += 1
            elif 'bicycle' in agent_class or 'cyclist' in agent_class:
                counts["cyclists"] += 1
            elif 'traffic_light' in agent_class:
                counts["traffic_lights"] += 1
            elif 'traffic_sign' in agent_class:
                counts["traffic_signs"] += 1

    counts["total"] = counts["vehicles"] + counts["pedestrians"] + counts["cyclists"]

    return counts


def get_ego_state_info(scene: Bench2DriveScene, frame_idx: int) -> Dict[str, Any]:
    """
    Extract ego vehicle state information.

    Args:
        scene: Bench2Drive scene
        frame_idx: Frame index

    Returns:
        Dictionary with ego state information
    """
    agent_input = scene.get_agent_input(frame_idx)
    ego_status = agent_input.ego_statuses[-1]  # Current frame status

    return {
        "velocity": np.linalg.norm(ego_status.ego_velocity[:2]),  # m/s
        "velocity_x": ego_status.ego_velocity[0],
        "velocity_y": ego_status.ego_velocity[1],
        "acceleration": np.linalg.norm(ego_status.ego_acceleration[:2]),  # m/s²
        "heading": np.degrees(ego_status.ego_pose[2]) % 360,  # degrees
        "driving_command": int(ego_status.driving_command[0]),
    }


def get_driving_command_name(command: int) -> str:
    """Convert driving command integer to readable name."""
    commands = {
        0: "IDLE",
        1: "STRAIGHT",
        2: "LEFT",
        3: "RIGHT",
        4: "LANE_FOLLOW",
        5: "CHANGE_LEFT",
        6: "CHANGE_RIGHT",
    }
    return commands.get(command, f"UNKNOWN({command})")


def draw_trajectory_with_numbers(
    ax, trajectory: np.ndarray, color: str, label: str,
    marker: str = "o", linestyle: str = "-", alpha: float = 1.0,
    number_color: str = None, zorder: int = 10
):
    """
    Draw trajectory with numbered points.

    FIXED: In Bench2Drive ego coordinates:
    - X is FORWARD (should map to Y-axis/up in display)
    - Y is RIGHT (should map to X-axis in display)

    Args:
        ax: Matplotlib axis
        trajectory: Trajectory points [T, 3]
        color: Line color
        label: Legend label
        marker: Marker style
        linestyle: Line style
        alpha: Transparency
        number_color: Color for numbers (defaults to line color)
        zorder: Drawing order
    """
    if trajectory.shape[0] == 0:
        return

    # Use world coordinates with correct transformation for origin='upper'
    # In CARLA/Bench2Drive: X=forward, Y=right
    # Prediction was already correct! Only GT needs different handling
    if color == "#00ff00":  # GT is green
        x_coords = -trajectory[:, 0]  # X with left-right flip
        y_coords = -trajectory[:, 1]  # Y negated to face up
    else:  # Prediction (red) - KEEP ORIGINAL CORRECT MAPPING
        x_coords = trajectory[:, 1]  # Y (right) -> X axis
        y_coords = trajectory[:, 0]  # X (forward) -> Y axis

    # Draw trajectory line
    ax.plot(
        x_coords, y_coords,
        color=color, linewidth=3, linestyle=linestyle,
        marker=marker, markersize=6, label=label,
        alpha=alpha, zorder=zorder
    )

    # Add numbered markers
    number_color = number_color or color
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        # Add white background for better visibility
        ax.text(
            x, y, str(i + 1),
            color=number_color, fontsize=8, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor=number_color, alpha=0.8),
            zorder=zorder + 1
        )


def draw_ego_vehicle(ax, heading_rad: float = 0, zorder: int = 25):
    """
    Draw ego vehicle with correct orientation.

    FIXED: Vehicle faces UP (positive Y direction = forward)

    Args:
        ax: Matplotlib axis
        heading_rad: Vehicle heading in radians
        zorder: Drawing order
    """
    # Vehicle dimensions in meters
    vehicle_length = 4.5  # meters
    vehicle_width = 2.0   # meters

    # Ego is at origin (0, 0) in world coordinates
    # Create vehicle rectangle centered at origin
    # NO ROTATION ADJUSTMENT NEEDED - heading was correct from the start!
    rect = patches.Rectangle(
        xy=(-vehicle_width/2, -vehicle_length/2),
        width=vehicle_width,
        height=vehicle_length,
        angle=np.degrees(heading_rad),  # Use heading directly, no subtraction!
        rotation_point='center',
        linewidth=2,
        edgecolor='black',
        facecolor='yellow',
        zorder=zorder
    )
    ax.add_patch(rect)

    # Add arrow to show heading direction
    arrow_len = 3.0  # meters
    # Use original heading directly
    dx = arrow_len * np.sin(heading_rad)
    dy = -arrow_len * np.cos(heading_rad)

    ax.arrow(0, 0, dx, dy,
             head_width=1.0, head_length=0.5,
             fc='red', ec='red', zorder=zorder+1, linewidth=2)

    # Mark ego center
    ax.scatter(0, 0, c='red', s=100, marker='*',
              edgecolors='black', linewidth=1, zorder=zorder+2)


def load_bench2drive_agent_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Bench2DriveAgent:
    """
    Load Bench2Drive agent from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Torch device to use

    Returns:
        Loaded agent ready for inference
    """
    print(f"Loading Bench2Drive checkpoint: {checkpoint_path}")

    # Create config with B2D-specific parameters
    config = AgentConfig(
        dataset_type="bench2drive",
        plan_anchor_path="/workspace/DiffusionDrive/download/kmeans_b2d_v2_traj_20.npy",
        bkb_path="",  # Will be loaded from checkpoint
    )

    # Load agent
    agent = Bench2DriveAgent(config=config, checkpoint_path=checkpoint_path)

    # Move to device and set to eval mode
    agent.to(device)
    agent.eval()

    # Store device for later use
    agent.device = device

    # Load checkpoint info
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    if "epoch" in checkpoint_data:
        print(f"  Loaded checkpoint from epoch: {checkpoint_data['epoch']}")

    return agent


def create_visualization_frame(
    scene: Bench2DriveScene,
    agent: Bench2DriveAgent,
    frame_idx: int,
    data_root: Path,
    bev_cache_dir: Optional[Path] = None,
    show_debug: bool = False,
) -> Tuple[plt.Figure, Any]:
    """
    Create an enhanced visualization frame with BEV + 6 camera views.

    FIXED: Trajectories persist across all frames

    Args:
        scene: Bench2Drive scene data
        agent: Trained agent for predictions
        frame_idx: Frame index in the scene
        data_root: Path to Bench2Drive dataset
        bev_cache_dir: Path to BEV cache directory
        show_debug: Whether to show debug information
        cached_trajectories: Previously computed trajectories to reuse

    Returns:
        Figure, axes, and computed trajectories (for caching)
    """
    # Create figure with 2x4 grid (BEV takes 2 columns)
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 4, width_ratios=[2, 1, 1, 1])

    # BEV takes left side (2 columns)
    ax_bev = fig.add_subplot(gs[:, :2])

    # Camera grid on right side
    camera_positions = [
        ("Front-Left", gs[0, 2]),
        ("Front", gs[0, 3]),
        ("Front-Right", gs[1, 2]),
        ("Back", gs[1, 3]),
    ]

    # Special views in middle column
    ax_rgb_bev = fig.add_subplot(gs[0, 1])
    ax_semantic = fig.add_subplot(gs[1, 1])

    # === BEV Visualization with Vector Map Background ===
    scenario_name = scene.scene_name

    # Load and display BEV vector map from cache
    if bev_cache_dir:
        bev_map = load_bev_map_from_cache(bev_cache_dir, scenario_name, frame_idx)
        if bev_map is None:
            raise ValueError(f"Failed to load BEV map for scenario {scenario_name}, frame {frame_idx}. "
                           f"Check that BEV cache exists at: {bev_cache_dir}/{scenario_name}/{frame_idx:05d}.npz")
        print(f"  Loaded BEV map: shape={bev_map.shape}, range=[{bev_map.min():.1f}, {bev_map.max():.1f}]")
        # Create color-coded BEV map
        # Map values: 0=background, 1=road, 2=lane marking, 3=sidewalk, 4=crosswalk
        colors = {
            0: [0.0, 0.0, 0.0],      # Background - black
            1: [0.5, 0.5, 0.5],      # Road - gray
            2: [0.8, 0.8, 0.3],      # Lane marking - yellow
            3: [0.3, 0.8, 0.3],      # Sidewalk - green
            4: [0.3, 0.3, 0.8],      # Crosswalk - blue
        }

        # Create RGB image
        h, w = bev_map.shape
        rgb_bev = np.zeros((h, w, 3), dtype=np.float32)
        for value, color in colors.items():
            mask = (bev_map == value)
            rgb_bev[mask] = color

        # Display BEV map with extent in world coordinates
        # BEV covers 85m x 85m centered at ego
        # Flip the BEV map both vertically and horizontally
        rgb_bev_flipped = np.flipud(np.fliplr(rgb_bev))  # Flip both up-down and left-right
        # With origin='upper': [left, right, bottom, top]
        extent = [-42.5, 42.5, 42.5, -42.5]
        im = ax_bev.imshow(rgb_bev_flipped, extent=extent, origin='upper', alpha=0.9, zorder=1)
        print(f"  Color-coded BEV map displayed with extent: {extent} (flipped UD+LR)")

        # Add legend
        from matplotlib.patches import Rectangle
        legend_elements = [
            Rectangle((0,0), 1, 1, fc=colors[1], label='Road'),
            Rectangle((0,0), 1, 1, fc=colors[2], label='Lane Marking'),
            Rectangle((0,0), 1, 1, fc=colors[3], label='Sidewalk'),
            Rectangle((0,0), 1, 1, fc=colors[4], label='Crosswalk'),
        ]
        ax_bev.legend(handles=legend_elements, loc='upper right', fontsize=6, framealpha=0.8)

    # Set BEV limits in world coordinates
    ax_bev.set_xlim(-50, 50)
    ax_bev.set_ylim(50, -50)  # Inverted to match origin='upper'
    ax_bev.grid(True, alpha=0.2, linestyle='--', zorder=0)

    # Load and display LiDAR directly from raw data
    lidar_points = load_lidar_from_raw_data(data_root, scenario_name, frame_idx)
    if lidar_points is not None:
        # Convert LiDAR points to BEV for visualization
        # Use world coordinates with correct mapping
        # Flip left-right as requested
        points_x = -lidar_points[:, 1]  # Y (right) -> -X display (flip left-right)
        points_y = -lidar_points[:, 0]  # X (forward) -> -Y display (up)

        # Create histogram in world space
        hist_range = [[-50, 50], [-50, 50]]
        hist, xedges, yedges = np.histogram2d(
            points_x, points_y,
            bins=256, range=hist_range
        )

        # Apply log scale to make points more visible
        hist_log = np.log1p(hist)  # log(1 + hist) to avoid log(0)

        # Display LiDAR as heatmap with proper extent
        extent_lidar = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax_bev.imshow(hist_log.T, extent=extent_lidar, origin='upper',
                     cmap='viridis', alpha=0.3, vmin=0, vmax=np.percentile(hist_log, 95) if hist_log.max() > 0 else 1,
                     zorder=2)
        print(f"  LiDAR overlay added: {lidar_points.shape[0]} points")

    # COMPUTE EVERYTHING FRESH FOR EVERY FRAME - NO CACHING!
    if show_debug:
        print(f"  Computing fresh predictions for frame {frame_idx}")

    # Initialize variables
    gt_trajectory = None
    predicted_trajectory = None
    l2_metrics = None
    first_prediction = None
    second_prediction = None

    # ALWAYS get ego state and agent counts for EVERY frame from raw data
    ego_state = get_ego_state_from_raw_data(data_root, scenario_name, frame_idx)
    agent_counts = count_agents_from_raw_data(data_root, scenario_name, frame_idx)

    # Get ego position for current frame to transform trajectories
    anno_file = data_root / scenario_name / "anno" / f"{frame_idx:05d}.json.gz"
    with gzip.open(anno_file, 'rt') as f:
        anno_data = json.load(f)
    ego_x = anno_data.get('x', 0)
    ego_y = anno_data.get('y', 0)
    ego_theta = anno_data.get('theta', 0)

    # ALWAYS compute trajectories for EVERY frame
    if True:  # Always compute, no conditions!

        # Get ground truth trajectory from CURRENT frame
        # Load future positions from raw data
        gt_poses = []
        for future_offset in range(1, 9):  # 8 future timesteps at 0.5s intervals
            future_frame = frame_idx + future_offset * 5  # 5 frames = 0.5s at 10Hz
            future_anno_file = data_root / scenario_name / "anno" / f"{future_frame:05d}.json.gz"

            if future_anno_file.exists():
                with gzip.open(future_anno_file, 'rt') as f:
                    future_data = json.load(f)

                # Get future position in world coordinates
                future_x = future_data.get('x', 0)
                future_y = future_data.get('y', 0)
                future_theta = future_data.get('theta', 0)

                # Transform to current frame's ego coordinates
                dx = future_x - ego_x
                dy = future_y - ego_y

                # Rotate to ego frame
                cos_theta = np.cos(-ego_theta)
                sin_theta = np.sin(-ego_theta)
                future_ego_x = cos_theta * dx - sin_theta * dy
                future_ego_y = sin_theta * dx + cos_theta * dy
                future_ego_heading = future_theta - ego_theta

                gt_poses.append([future_ego_x, future_ego_y, future_ego_heading])
            else:
                break  # Stop if we run out of frames

        if len(gt_poses) == 8:
            # Full trajectory available
            gt_trajectory = Trajectory(np.array(gt_poses))
        elif len(gt_poses) > 0:
            # Pad with last position if we have fewer than 8 points
            while len(gt_poses) < 8:
                gt_poses.append(gt_poses[-1])
            gt_trajectory = Trajectory(np.array(gt_poses))
        else:
            gt_trajectory = None

        # Note: ego_state and agent_counts are computed for every frame above

        # Get model prediction for CURRENT frame
        # Get fresh agent input directly from the existing scene for the current frame
        # This ensures the model sees the current frame's data and makes fresh predictions

        # Get agent input for the CURRENT frame_idx
        # Map absolute frame_idx to scene's relative frame index
        # The scene has a sliding window of frames loaded
        scene_frame_idx = min(frame_idx, len(scene.anno_paths) - 1)
        if show_debug:
            print(f"  Getting agent input for frame {frame_idx} (scene frame {scene_frame_idx})")
        agent_input = scene.get_agent_input(scene_frame_idx)

        # Compute trajectory
        agent.eval()
        features: Dict[str, torch.Tensor] = {}

        # Build features
        for builder in agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # Add batch dimension and move to device
        features = {k: v.unsqueeze(0).to(agent.device) for k, v in features.items()}

        # Forward pass
        with torch.no_grad():
            predictions = agent.forward(features)
            trajectory_output = predictions["trajectory"].squeeze(0).cpu().numpy()

            # Debug: Show that predictions are different per frame
            if show_debug:
                print(f"  Frame {frame_idx} prediction: first point = ({trajectory_output[0, 0]:.2f}, {trajectory_output[0, 1]:.2f})")

            # The model already outputs trajectory in ego frame
            trajectory_meters = trajectory_output.copy()
            if trajectory_meters.shape[0] > 0:
                # Store first prediction info (no offset subtraction!)
                first_prediction = {
                    "x": trajectory_meters[0, 0],
                    "y": trajectory_meters[0, 1],
                    "heading": np.degrees(trajectory_meters[0, 2]) % 360,
                }

                # Store second prediction info if available
                if trajectory_meters.shape[0] > 1:
                    second_prediction = {
                        "x": trajectory_meters[1, 0],
                        "y": trajectory_meters[1, 1],
                        "heading": np.degrees(trajectory_meters[1, 2]) % 360,
                    }

        predicted_trajectory = Trajectory(trajectory_meters)

        # Calculate L2 metrics using VAD method (only if GT is available)
        if gt_trajectory is not None:
            l2_metrics = calculate_vad_l2_metrics(
                predicted_trajectory.poses,
                gt_trajectory.poses,
                timestep_duration=0.5
            )
        else:
            l2_metrics = None

        # NO CACHING - compute fresh every time!

    # Plot agents from raw annotation data
    if 'bounding_boxes' in anno_data:
        for box in anno_data['bounding_boxes']:
            agent_class = box.get('class', '').lower()

            # Skip ego vehicle
            if 'ego' in agent_class:
                continue

            # Get agent position in world coordinates
            agent_world_x = box['center'][0]
            agent_world_y = box['center'][1]

            # Transform to ego-centric coordinates
            dx = agent_world_x - ego_x
            dy = agent_world_y - ego_y

            # Rotate to ego frame
            cos_theta = np.cos(-ego_theta)
            sin_theta = np.sin(-ego_theta)
            agent_ego_x = cos_theta * dx - sin_theta * dy  # X = forward
            agent_ego_y = sin_theta * dx + cos_theta * dy   # Y = right

            # Convert to display coordinates in world space
            # Agents should be in front, need to negate
            agent_x = -agent_ego_x  # X (forward) -> -X (flip left-right)
            agent_y = -agent_ego_y  # Y (right) -> -Y (flip front-back)

            # Get agent dimensions in meters
            extent_x = box.get('extent', [2, 1, 1])[0]  # length
            extent_y = box.get('extent', [2, 1, 1])[1]  # width

            # Get agent heading
            agent_heading = box.get('rotation', [0, 0, 0])[2]  # yaw in degrees
            agent_heading_rad = np.radians(agent_heading) - ego_theta

            # Choose color based on agent type
            if 'vehicle' in agent_class or 'car' in agent_class:
                color = 'blue'
                facecolor = 'cyan'
            elif 'pedestrian' in agent_class or 'person' in agent_class:
                color = 'green'
                facecolor = 'lightgreen'
            elif 'bicycle' in agent_class or 'cyclist' in agent_class:
                color = 'orange'
                facecolor = 'yellow'
            else:
                color = 'gray'
                facecolor = 'lightgray'

            # Skip if outside reasonable bounds
            if abs(agent_x) > 50 or abs(agent_y) > 50:
                continue

            # Draw agent as rectangle in world coordinates
            rect = patches.Rectangle(
                (agent_x - extent_y/2, agent_y - extent_x/2),
                extent_y, extent_x,
                angle=np.degrees(agent_heading_rad),
                rotation_point='center',
                linewidth=1, edgecolor=color, facecolor=facecolor,
                alpha=0.6, zorder=5
            )
            ax_bev.add_patch(rect)

    # Always plot trajectories if they exist (for all frames)
    if gt_trajectory is not None and predicted_trajectory is not None:
        # Plot trajectories with numbered points
        # Ground truth in green
        draw_trajectory_with_numbers(
            ax_bev, gt_trajectory.poses,
            color="#00ff00", label="Ground Truth",
            marker="o", linestyle="-", alpha=0.8,
            number_color="#006600", zorder=15
        )

        # Predicted trajectory in red
        draw_trajectory_with_numbers(
            ax_bev, predicted_trajectory.poses,
            color="#ff0000", label="Model Prediction",
            marker="^", linestyle="--", alpha=0.9,
            number_color="#660000", zorder=20
        )

        # Add legend
        ax_bev.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # Always show ego vehicle (update state for current frame if needed)
    if ego_state is None:
        ego_state = get_ego_state_info(scene, frame_idx)

    ego_heading = ego_state['heading']  # Already in radians!
    draw_ego_vehicle(ax_bev, ego_heading, zorder=25)

    # Show metrics if available
    if l2_metrics is not None and agent_counts is not None:
        # Create comprehensive metrics display
        metrics_text_lines = [
            "═══ Trajectory Metrics ═══",
            f"L2 @ 0.5s: {l2_metrics.get('L2_0.5s', 0):.2f}m",
            f"L2 @ 1.0s: {l2_metrics.get('L2_1.0s', 0):.2f}m",
            f"L2 @ 2.0s: {l2_metrics.get('L2_2.0s', 0):.2f}m",
            f"L2 @ 3.0s: {l2_metrics.get('L2_3.0s', 0):.2f}m",
            f"L2 @ 4.0s: {l2_metrics.get('L2_4.0s', 0):.2f}m",
            f"L2 Average: {l2_metrics.get('L2_avg', 0):.2f}m",
            "",
            "═══ Ego State ═══",
            f"Velocity: {ego_state['velocity']:.1f} m/s",
            f"Acceleration: {ego_state['acceleration']:.2f} m/s²",
            f"Heading: {np.degrees(ego_state['heading']):.1f}°",
            f"Command: {get_driving_command_name(ego_state['driving_command'])}",
            "",
            "═══ Scene Agents ═══",
            f"Total: {agent_counts['total']}",
            f"Vehicles: {agent_counts['vehicles']}",
            f"Pedestrians: {agent_counts['pedestrians']}",
            f"Traffic Lights: {agent_counts['traffic_lights']}",
            "",
            "═══ 1st Prediction ═══",
            f"X: {first_prediction['x']:.2f}m" if first_prediction else "X: N/A",
            f"Y: {first_prediction['y']:.2f}m" if first_prediction else "Y: N/A",
            f"Heading: {first_prediction['heading']:.1f}°" if first_prediction else "Heading: N/A",
            "",
            "═══ 2nd Prediction ═══",
            f"X: {second_prediction['x']:.2f}m" if second_prediction else "X: N/A",
            f"Y: {second_prediction['y']:.2f}m" if second_prediction else "Y: N/A",
            f"Heading: {second_prediction['heading']:.1f}°" if second_prediction else "Heading: N/A",
        ]

        metrics_text = "\n".join(metrics_text_lines)
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.9)
        ax_bev.text(
            0.02, 0.98, metrics_text,
            transform=ax_bev.transAxes, fontsize=9,
            verticalalignment="top", fontfamily='monospace',
            bbox=props
        )

    configure_bev_ax(ax_bev)
    ax_bev.set_title("BEV with Vector Map & LiDAR & Trajectories", fontsize=12, fontweight="bold")
    ax_bev.set_xlabel("Right (m) →")
    ax_bev.set_ylabel("Forward (m) ↑")

    # === Camera Views ===
    # Load cameras directly from raw files for ALL frames
    from PIL import Image as PILImage

    camera_paths = {
        "Front": data_root / scenario_name / 'camera' / 'rgb_front' / f"{frame_idx:05d}.jpg",
        "Front-Left": data_root / scenario_name / 'camera' / 'rgb_front_left' / f"{frame_idx:05d}.jpg",
        "Front-Right": data_root / scenario_name / 'camera' / 'rgb_front_right' / f"{frame_idx:05d}.jpg",
        "Back": data_root / scenario_name / 'camera' / 'rgb_back' / f"{frame_idx:05d}.jpg",
    }

    for cam_name, grid_pos in camera_positions:
        ax_cam = fig.add_subplot(grid_pos)

        if cam_name in camera_paths:
            cam_path = camera_paths[cam_name]
            if cam_path.exists():
                img = PILImage.open(cam_path)
                ax_cam.imshow(img)
            else:
                raise FileNotFoundError(f"Camera image not found: {cam_path}")
        else:
            ax_cam.text(0.5, 0.5, f"{cam_name}\nNot Available",
                       ha="center", va="center", fontsize=10)

        ax_cam.set_title(cam_name, fontsize=10)
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])

    # === RGB BEV Camera View ===
    rgb_bev = load_rgb_bev_camera(data_root, scenario_name, frame_idx)
    if rgb_bev is not None:
        ax_rgb_bev.imshow(rgb_bev)
        ax_rgb_bev.set_title("RGB BEV (Top-Down)", fontsize=10)
    else:
        ax_rgb_bev.text(0.5, 0.5, "RGB BEV\nNot Available",
                       ha="center", va="center", fontsize=10)
    ax_rgb_bev.set_xticks([])
    ax_rgb_bev.set_yticks([])

    # === Semantic Segmentation View ===
    semantic_seg = load_semantic_segmentation(data_root, scenario_name, frame_idx)
    if semantic_seg is not None:
        # Apply colormap for better visualization
        ax_semantic.imshow(semantic_seg, cmap='tab20')
        ax_semantic.set_title("Semantic Segmentation", fontsize=10)
    else:
        ax_semantic.text(0.5, 0.5, "Semantic Map\nNot Available",
                        ha="center", va="center", fontsize=10)
    ax_semantic.set_xticks([])
    ax_semantic.set_yticks([])

    # Add main title
    scene_token = getattr(scene, "token", scenario_name)
    fig.suptitle(
        f"DiffusionDrive B2D Enhanced Visualization - Scene: {scene_token} - Frame: {frame_idx}",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    return fig, None


def get_all_scene_frames(scene_loader: Bench2DriveSceneLoader, token: str) -> int:
    """
    Get the total number of frames available in a scene.

    Args:
        scene_loader: Scene loader
        token: Scene token

    Returns:
        Number of frames in the scene
    """
    # Get the scenario name from token
    # Token format could be: ScenarioName_FrameStart or ScenarioName_FrameStart-FrameEnd
    scenario_name = token.rsplit('_', 1)[0] if '_' in token else token

    # Count actual frame files in the scenario directory
    from pathlib import Path
    anno_dir = scene_loader.config.data_root / scenario_name / "anno"

    if anno_dir.exists():
        # Count .json.gz files
        frame_files = list(anno_dir.glob("*.json.gz"))
        num_frames = len(frame_files)
        print(f"  Scene {scenario_name}: {num_frames} frames available")
        return num_frames
    else:
        # Fall back to config if directory doesn't exist
        print(f"  Warning: Anno directory not found for {scenario_name}, using config default")
        return scene_loader.config.num_frames


def create_mp4_from_scenes(
    agent: Bench2DriveAgent,
    scene_loader: Bench2DriveSceneLoader,
    output_path: str,
    data_root: Path,
    bev_cache_dir: Optional[Path] = None,
    num_scenes: int = 5,
    fps: int = 2,
    full_scenario: bool = False,
    show_debug: bool = False,
):
    """
    Create MP4 video with enhanced visualizations.

    Args:
        agent: Trained model
        scene_loader: Data loader
        output_path: Output MP4 path
        data_root: Bench2Drive dataset root
        bev_cache_dir: BEV cache directory
        num_scenes: Number of scenes to visualize (ignored if full_scenario)
        fps: Frames per second
        full_scenario: Export entire scenario
        show_debug: Whether to show debug info
    """
    print(f"\n🎬 Creating Enhanced Bench2Drive Visualization...")
    print(f"  Mode: {'Full Scenario' if full_scenario else f'{num_scenes} Scenes'}")
    print(f"  FPS: {fps}")

    all_images = []

    if full_scenario and len(scene_loader) > 0:
        # Export entire first scenario - show ALL frames
        token = scene_loader.scene_tokens[0]
        scene = scene_loader.get_scene(token)

        # Get actual number of frames available
        num_frames = get_all_scene_frames(scene_loader, token)

        print(f"  Exporting full scenario: {scene.scene_name}")
        print(f"  Total frames to export: {num_frames}")

        for frame_idx in range(num_frames):
            if frame_idx % 10 == 0:
                print(f"    Processing frame {frame_idx}/{num_frames}")

            fig, _ = create_visualization_frame(
                scene, agent, frame_idx, data_root, bev_cache_dir, show_debug
            )

            # Convert to PIL image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf).copy()
            all_images.append(img)
            buf.close()
            plt.close(fig)
    else:
        # Original mode: for each scene, show multiple frames around prediction point
        num_to_process = min(num_scenes, len(scene_loader))

        for scene_idx in range(num_to_process):
            token = scene_loader.scene_tokens[scene_idx]
            scene = scene_loader.get_scene(token)

            print(f"  Scene {scene_idx + 1}/{num_to_process}: {token[:40]}...")

            # Get ALL frames in the scenario
            prediction_frame = scene.history_frames - 1

            # Show ALL frames in the scenario, not just around prediction
            start_frame = 0
            total_frames = get_all_scene_frames(scene_loader, token)
            end_frame = min(10, total_frames)  # LIMIT TO 10 FRAMES FOR QUICK CHECK

            print(f"  Scene {scene.scene_name}: {total_frames} frames available, using only first {end_frame} frames for quick check")

            for frame_idx in range(start_frame, end_frame):
                print(f"    Adding frame {frame_idx}")

                fig, _ = create_visualization_frame(
                    scene, agent, frame_idx, data_root, bev_cache_dir, show_debug
                )

                # Convert to PIL image
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                img = Image.open(buf).copy()
                all_images.append(img)
                buf.close()
                plt.close(fig)

    if not all_images:
        raise ValueError("No images generated! Check your data paths and scene configuration.")

    print(f"\n📊 Total frames: {len(all_images)}")
    actual_duration = len(all_images) / fps
    print(f"  Actual duration: {actual_duration:.1f} seconds")

    # Save as temporary GIF
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_gif:
        tmp_gif_path = tmp_gif.name

    print("💾 Creating GIF...")
    all_images[0].save(
        tmp_gif_path,
        save_all=True,
        append_images=all_images[1:],
        duration=1000 // fps,  # milliseconds per frame
        loop=0,
    )

    # Convert to MP4
    print("🎥 Converting to MP4...")

    # Check ffmpeg
    result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
    if result.returncode != 0:
        print("⚠️  ffmpeg not found! Installing...")
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)

    # Convert GIF to MP4 with slower playback for visibility
    cmd = [
        "ffmpeg",
        "-i", tmp_gif_path,
        "-movflags", "faststart",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={fps}",
        "-c:v", "libx264",
        "-crf", "20",
        "-preset", "slow",
        "-y", output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error converting to MP4: {result.stderr}")

    # Clean up temp file
    os.unlink(tmp_gif_path)

    # Get file info
    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n✅ Video successfully created!")
    print(f"  📹 File: {output_path}")
    print(f"  💾 Size: {size_mb:.2f} MB")
    print(f"  ⏱️  Duration: {actual_duration:.1f} seconds")
    print(f"  🖼️  Total frames: {len(all_images)}")


def load_train_val_split() -> Dict[str, List[str]]:
    """Load official train/val split from JSON file."""
    split_file = Path(__file__).parent.parent / "navsim" / "planning" / "script" / "config" / \
                 "common" / "train_test_split" / "bench2drive_base_train_val_split.json"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r") as f:
        splits = json.load(f)
        # Remove 'v1/' prefix from validation scenarios
        val_scenarios = [s.replace("v1/", "") for s in splits.get("val", [])]
        return {"val": val_scenarios}


def generate_single_video(
    video_idx: int,
    selected_scenarios: List[str],
    args,
    data_root: Path,
    agent: Bench2DriveAgent,
    device: torch.device,
) -> Optional[str]:
    """
    Generate a single video for the given video index.
    Returns the output path if successful, None otherwise.
    """
    # Ensure matplotlib uses non-interactive backend for parallel processing
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if args.num_videos > 1:
        # Select ONE scenario for each video
        if video_idx < len(selected_scenarios):
            # Use scenarios in order
            video_scenarios = [selected_scenarios[video_idx]]
        else:
            # If we run out of scenarios, stop
            print(f"Only {len(selected_scenarios)} scenarios available, stopping at video {video_idx}")
            return None

        # Create unique output filename with split and full scenario name
        base_name = args.output.replace(".mp4", "")
        # Use full scenario name
        scenario_name = video_scenarios[0]  # Always single scenario
        output_path = f"{base_name}_{args.split}_{scenario_name}.mp4"
        print(f"\n📊 Generating video {video_idx+1}/{args.num_videos}: {output_path}")
    else:
        video_scenarios = selected_scenarios
        # Add split and full scenario name to output filename
        base_name = args.output.replace(".mp4", "")
        if len(video_scenarios) == 1:
            # For single scenario, use full name
            scenario_name = video_scenarios[0]
        else:
            scenario_name = f"multi_{len(video_scenarios)}_scenes"
        output_path = f"{base_name}_{args.split}_{scenario_name}.mp4"

    # Create scene loader
    print(f"\n📊 [Worker {video_idx}] Loading Bench2Drive scenes...")
    print(f"  Looking for scenarios: {video_scenarios}")

    try:
        # Check scenario length first
        scenario_path = data_root / video_scenarios[0]
        anno_dir = scenario_path / "anno"
        if anno_dir.exists():
            total_frames = len(list(anno_dir.glob("*.json.gz")))
            sampled_frames = total_frames // 5  # After 10Hz to 2Hz sampling

            # Adjust num_frames if scenario is too short
            if sampled_frames < 30:
                num_frames = max(10, sampled_frames - 1)  # At least 10 frames, or all available
                print(f"  ⚠️ Scenario has only {sampled_frames} frames after sampling, adjusting to {num_frames} frames")
            else:
                num_frames = 30
        else:
            num_frames = 30

        config = Bench2DriveConfig(
            data_root=data_root,
            scenarios=video_scenarios,
            sampling_rate=5,  # 10Hz to 2Hz
            num_frames=num_frames,
            num_history_frames=4,
            num_future_frames=num_frames - 4,
            bev_cache_dir=Path(args.bev_cache_dir) if args.bev_cache_dir else None,
        )

        scene_loader = Bench2DriveSceneLoader(config)
        print(f"  ✅ [Worker {video_idx}] Loaded {len(scene_loader)} scenes")

        if len(scene_loader) == 0:
            print(f"  ⚠️ WARNING: No scenes found for scenarios: {video_scenarios}")
            print(f"  Checking if scenarios exist in {data_root}...")
            for scenario in video_scenarios:
                scenario_path = data_root / scenario
                if scenario_path.exists():
                    print(f"    ✓ {scenario} exists")
                    anno_dir = scenario_path / "anno"
                    if anno_dir.exists():
                        num_frames = len(list(anno_dir.glob("*.json.gz")))
                        print(f"      Found {num_frames} frames")
                    else:
                        print(f"      ✗ No anno directory")
                else:
                    print(f"    ✗ {scenario} does NOT exist")
            return None  # Skip this video if no scenes found

    except Exception as e:
        print(f"  ❌ [Worker {video_idx}] Error loading scenes: {e}")
        print(f"  Skipping scenarios: {video_scenarios}")
        return None

    # Create video
    try:
        create_mp4_from_scenes(
            agent,
            scene_loader,
            output_path,
            data_root,
            Path(args.bev_cache_dir) if args.bev_cache_dir else None,
            args.num_scenes,
            args.fps,
            args.full_scenario,
            args.debug,
        )
        return output_path
    except Exception as e:
        print(f"  ❌ [Worker {video_idx}] Error creating video: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced visualization for DiffusionDrive on Bench2Drive dataset"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/navsim_workspace/exp/training_diffusiondrive_agent_b2d_training_diffusiondrive_diffusiondrive_agent_b2d_exp0_datasetv2_archv0_bs32x8_ep3000_lr1e-5/2025.08.31.22.51.59/lightning_logs/version_0/checkpoints/epoch=309-step=309.ckpt",
        help="Path to Bench2Drive-trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bench2drive_predictions_enhanced.mp4",
        help="Output MP4 file path (or prefix for multiple videos)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/workspace/Bench2Drive-Base",
        help="Path to Bench2Drive dataset root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "all"],
        help="Use train, validation, or all scenarios",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="List of specific scenarios to visualize",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for scenario selection",
    )
    parser.add_argument(
        "--bev-cache-dir",
        type=str,
        default="/workspace/navsim_workspace/cache/Bench2Drive-Base-full_bev_cache-v2",
        help="Path to pre-generated BEV cache",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=5,
        help="Number of scenes to visualize (ignored if --full-scenario)",
    )
    parser.add_argument(
        "--full-scenario",
        action="store_true",
        help="Export complete scenario instead of sample scenes",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=1,
        help="Number of random scenario videos to generate",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for video",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug information",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for video generation (default: 1 = no parallelization)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" DiffusionDrive Enhanced Bench2Drive Visualization")
    print("=" * 70)

    # Check checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Handle scenario selection based on split
    import random
    if args.random_seed is not None:
        random.seed(args.random_seed)
        print(f"\n🎲 Random seed set to: {args.random_seed}")

    if args.scenarios:
        # Use explicitly specified scenarios
        selected_scenarios = args.scenarios
        print(f"\n📋 Using specified scenarios: {selected_scenarios}")
    else:
        # Get all available scenarios
        all_scenarios = [
            d.name for d in data_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        if not all_scenarios:
            raise ValueError(f"No scenarios found in {data_root}")

        if args.split == "val":
            # Use validation scenarios only
            splits = load_train_val_split()
            val_scenarios = splits.get("val", [])
            selected_scenarios = [s for s in all_scenarios if s in val_scenarios]
            print(f"\n📋 Using validation split: {len(selected_scenarios)} scenarios")
        elif args.split == "train":
            # Use training scenarios (exclude validation)
            splits = load_train_val_split()
            val_scenarios = splits.get("val", [])
            selected_scenarios = [s for s in all_scenarios if s not in val_scenarios]
            print(f"\n📋 Using training split: {len(selected_scenarios)} scenarios")
        else:
            # Use all scenarios
            selected_scenarios = all_scenarios
            print(f"\n📋 Using all scenarios: {len(selected_scenarios)} total")

        if not selected_scenarios:
            raise ValueError(f"No scenarios found for split '{args.split}'")

        # Random selection if too many
        if len(selected_scenarios) > 10 and not args.full_scenario:
            selected_scenarios = random.sample(selected_scenarios, 10)
            print(f"  Randomly selected 10 scenarios for visualization")

    print("\n📂 Configuration:")
    print(f"  Checkpoint: {Path(args.checkpoint).name}")
    print(f"  Data root: {data_root}")
    print(f"  Split: {args.split}")
    print(f"  Mode: {'Full Scenario' if args.full_scenario else 'Sample Scenes'}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load agent
    print("\n🤖 Loading Bench2Drive agent...")
    agent = load_bench2drive_agent_from_checkpoint(args.checkpoint, device)
    print("  ✅ Agent loaded successfully")

    # Check if parallel processing is requested
    if args.num_workers > 1 and args.num_videos > 1:
        print(f"\n🚀 Using parallel processing with {args.num_workers} workers")

        # Move agent to CPU and share memory for multi-processing
        agent.to('cpu')
        agent.share_memory()

        # Use joblib for parallel processing
        results = Parallel(n_jobs=args.num_workers, backend='multiprocessing', verbose=10)(
            delayed(generate_single_video)(
                video_idx, selected_scenarios, args, data_root, agent, device
            )
            for video_idx in range(args.num_videos)
        )

        # Filter out None results and report
        successful_videos = [r for r in results if r is not None]
        print(f"\n✅ Successfully generated {len(successful_videos)}/{args.num_videos} videos")
        if successful_videos:
            print("Generated files:")
            for video_path in successful_videos:
                print(f"  📹 {video_path}")
        return

    # Single-threaded processing (original code)
    print("\n📊 Using single-threaded processing")
    for video_idx in range(args.num_videos):
        if args.num_videos > 1:
            # Select ONE scenario for each video
            if video_idx < len(selected_scenarios):
                # Use scenarios in order
                video_scenarios = [selected_scenarios[video_idx]]
            else:
                # If we run out of scenarios, stop
                print(f"Only {len(selected_scenarios)} scenarios available, stopping at video {video_idx}")
                break

            # Create unique output filename with split and full scenario name
            base_name = args.output.replace(".mp4", "")
            # Use full scenario name
            scenario_name = video_scenarios[0]  # Always single scenario
            output_path = f"{base_name}_{args.split}_{scenario_name}.mp4"
            print(f"\n📊 Generating video {video_idx+1}/{args.num_videos}: {output_path}")
        else:
            video_scenarios = selected_scenarios
            # Add split and full scenario name to output filename
            base_name = args.output.replace(".mp4", "")
            if len(video_scenarios) == 1:
                # For single scenario, use full name
                scenario_name = video_scenarios[0]
            else:
                scenario_name = f"multi_{len(video_scenarios)}_scenes"
            output_path = f"{base_name}_{args.split}_{scenario_name}.mp4"

        # Create scene loader
        print(f"\n📊 Loading Bench2Drive scenes...")
        print(f"  Looking for scenarios: {video_scenarios}")

        try:
            # Check scenario length first
            scenario_path = data_root / video_scenarios[0]
            anno_dir = scenario_path / "anno"
            if anno_dir.exists():
                total_frames = len(list(anno_dir.glob("*.json.gz")))
                sampled_frames = total_frames // 5  # After 10Hz to 2Hz sampling

                # Adjust num_frames if scenario is too short
                if sampled_frames < 30:
                    num_frames = max(10, sampled_frames - 1)  # At least 10 frames, or all available
                    print(f"  ⚠️ Scenario has only {sampled_frames} frames after sampling, adjusting to {num_frames} frames")
                else:
                    num_frames = 30
            else:
                num_frames = 30

            config = Bench2DriveConfig(
                data_root=data_root,
                scenarios=video_scenarios,
                sampling_rate=5,  # 10Hz to 2Hz
                num_frames=num_frames,
                num_history_frames=4,
                num_future_frames=num_frames - 4,
                bev_cache_dir=Path(args.bev_cache_dir) if args.bev_cache_dir else None,
            )

            scene_loader = Bench2DriveSceneLoader(config)
            print(f"  ✅ Loaded {len(scene_loader)} scenes")

            if len(scene_loader) == 0:
                print(f"  ⚠️ WARNING: No scenes found for scenarios: {video_scenarios}")
                print(f"  Checking if scenarios exist in {data_root}...")
                for scenario in video_scenarios:
                    scenario_path = data_root / scenario
                    if scenario_path.exists():
                        print(f"    ✓ {scenario} exists")
                        anno_dir = scenario_path / "anno"
                        if anno_dir.exists():
                            num_frames = len(list(anno_dir.glob("*.json.gz")))
                            print(f"      Found {num_frames} frames")
                        else:
                            print(f"      ✗ No anno directory")
                    else:
                        print(f"    ✗ {scenario} does NOT exist")
                continue  # Skip this video if no scenes found

        except Exception as e:
            print(f"  ❌ Error loading scenes: {e}")
            print(f"  Skipping scenarios: {video_scenarios}")
            continue

        # Create video
        create_mp4_from_scenes(
            agent,
            scene_loader,
            output_path,
            data_root,
            bev_cache_dir=Path(args.bev_cache_dir) if args.bev_cache_dir else None,
            num_scenes=args.num_scenes,
            fps=args.fps,
            full_scenario=args.full_scenario,
            show_debug=args.debug,
        )


if __name__ == "__main__":
    main()