#!/usr/bin/env python3
"""
Visualization script for DiffusionDrive model predictions.
Creates MP4 videos showing:
- BEV view with predicted trajectory vs ground truth
- All 8 camera views
- Multiple continuous scenes for longer videos
"""

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Scene, Trajectory
from navsim.agents.abstract_agent import AbstractAgent
from navsim.visualization.plots import configure_bev_ax
from navsim.visualization.bev import (
    add_configured_bev_on_ax,
    add_lidar_to_bev_ax,
)
from navsim.visualization.camera import add_camera_ax

import os
import sys
import argparse
import subprocess
import tempfile
import io
import pickle
from pathlib import Path
from typing import List, Tuple, Any, Dict
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import cv2
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


def draw_trajectory_on_camera(image, camera, trajectory, color=(255, 0, 0), thickness=3):
    """
    Project and draw a trajectory onto a camera image.

    Args:
        image: Input image to draw on (will be modified)
        camera: Camera object with intrinsics and transforms
        trajectory: Trajectory object with poses (x, y, heading)
        color: RGB color tuple for trajectory
        thickness: Line thickness

    Returns:
        Image with trajectory overlay
    """
    image = image.copy()

    if trajectory is None or trajectory.poses.shape[0] == 0:
        return image

    # Convert trajectory points to 3D (add z=0 for ground plane)
    traj_points_3d = np.zeros((trajectory.poses.shape[0], 3))
    traj_points_3d[:, 0] = trajectory.poses[:, 0]  # X (forward)
    traj_points_3d[:, 1] = trajectory.poses[:, 1]  # Y (left/right)
    traj_points_3d[:, 2] = 0.0  # Z (height) - assume ground plane

    # Transform trajectory points to camera coordinates
    lidar2cam_r = np.linalg.inv(camera.sensor2lidar_rotation)
    lidar2cam_t = camera.sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    # Add homogeneous coordinate
    traj_points_homo = np.concatenate(
        [traj_points_3d, np.ones((traj_points_3d.shape[0], 1))], axis=1
    )

    # Transform to camera frame
    traj_cam = lidar2cam_rt.T @ traj_points_homo.T
    traj_cam = traj_cam.T[:, :3]

    # Project to image plane
    viewpad = np.eye(4)
    viewpad[: camera.intrinsics.shape[0], : camera.intrinsics.shape[1]] = camera.intrinsics

    traj_img_homo = np.concatenate([traj_cam, np.ones((traj_cam.shape[0], 1))], axis=1)
    traj_img = viewpad @ traj_img_homo.T
    traj_img = traj_img.T

    # Normalize by depth
    eps = 1e-3
    valid_mask = traj_img[:, 2] > eps
    traj_img[:, :2] = traj_img[:, :2] / np.maximum(traj_img[:, 2:3], eps)

    # Filter points within image bounds
    img_h, img_w = image.shape[:2]
    valid_mask = valid_mask & (traj_img[:, 0] >= 0) & (traj_img[:, 0] < img_w)
    valid_mask = valid_mask & (traj_img[:, 1] >= 0) & (traj_img[:, 1] < img_h)

    # Draw trajectory as connected line
    valid_points = traj_img[valid_mask, :2].astype(np.int32)

    if len(valid_points) > 1:
        # Draw lines connecting consecutive points
        for i in range(len(valid_points) - 1):
            cv2.line(
                image,
                tuple(valid_points[i]),
                tuple(valid_points[i + 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )

        # Draw circles at each point for visibility
        for point in valid_points:
            cv2.circle(image, tuple(point), thickness + 1, color, -1)

    return image


# Note: The model already outputs trajectories in absolute coordinates (meters)
# No denormalization is needed as the model only uses normalization internally
# for the diffusion process, but outputs unnormalized trajectories


def load_agent_from_checkpoint(checkpoint_path: str, device: torch.device) -> AbstractAgent:
    """
    Load DiffusionDrive agent from checkpoint using proper Hydra config.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Torch device to use

    Returns:
        Loaded agent ready for inference
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Create config using Hydra structure
    config_dict = {
        "_target_": "navsim.agents.diffusiondrive.transfuser_agent.TransfuserAgent",
        "_convert_": "all",
        "config": {
            "_target_": "navsim.agents.diffusiondrive.transfuser_config.TransfuserConfig",
            "_convert_": "all",
            "trajectory_sampling": {
                "_target_": "nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling",  # noqa: E501
                "_convert_": "all",
                "time_horizon": 4,
                "interval_length": 0.5,
            },
            "latent": False,
            "plan_anchor_path": "/workspace/DiffusionDrive/download/kmeans_navsim_traj_20.npy",
            "bkb_path": "",  # Will be loaded from checkpoint
        },
        "checkpoint_path": checkpoint_path,
        "lr": 6e-4,  # Not used for inference
    }

    # Convert to DictConfig
    cfg = OmegaConf.create(config_dict)

    # Instantiate agent using Hydra
    agent: AbstractAgent = instantiate(cfg)

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
    scene: Scene,
    agent: AbstractAgent,
    frame_idx: int,
    show_all_cameras: bool = True,
    show_debug: bool = False,
) -> Tuple[plt.Figure, Any]:
    """
    Create a visualization frame with BEV + cameras showing predicted and GT trajectories.

    Args:
        scene: NAVSIM scene data
        agent: Trained agent for predictions
        frame_idx: Current frame index
        show_all_cameras: Whether to show all 8 cameras
        show_debug: Whether to show debug information

    Returns:
        Figure and axes
    """
    # Determine layout based on camera option
    if show_all_cameras:
        # Layout: Large BEV on left, 8 cameras in grid on right
        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(3, 4, width_ratios=[2, 1, 1, 1])

        # BEV takes left column spanning all rows
        ax_bev = fig.add_subplot(gs[:, 0])

        # 8 cameras in 3x3 grid (with BEV spot empty)
        camera_positions = [
            ("CAM_FRONT_LEFT", gs[0, 1]),
            ("CAM_FRONT", gs[0, 2]),
            ("CAM_FRONT_RIGHT", gs[0, 3]),
            ("CAM_SIDE_LEFT", gs[1, 1]),
            ("CAM_SIDE_RIGHT", gs[1, 3]),
            ("CAM_REAR_LEFT", gs[2, 1]),
            ("CAM_REAR", gs[2, 2]),
            ("CAM_REAR_RIGHT", gs[2, 3]),
        ]
    else:
        # Simpler layout with just front cameras
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])

        ax_bev = fig.add_subplot(gs[:, 0])

        camera_positions = [
            ("CAM_FRONT", gs[0, 1:]),
            ("CAM_FRONT_LEFT", gs[1, 1]),
            ("CAM_FRONT_RIGHT", gs[1, 2]),
        ]

    # === BEV Visualization ===
    # Add map and scene elements
    add_configured_bev_on_ax(ax_bev, scene.map_api, scene.frames[frame_idx])

    # Add LiDAR if available
    if scene.frames[frame_idx].lidar is not None:
        add_lidar_to_bev_ax(ax_bev, scene.frames[frame_idx].lidar)

    # Compute and plot trajectories only at the current frame (frame 3)
    current_frame_idx = scene.scene_metadata.num_history_frames - 1

    if frame_idx == current_frame_idx:
        # Get ground truth trajectory
        gt_trajectory = scene.get_future_trajectory()

        # Get model prediction
        agent_input = scene.get_agent_input()

        # Custom compute_trajectory that handles device placement
        agent.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension and move to device
        features = {k: v.unsqueeze(0).to(agent.device) for k, v in features.items()}

        # forward pass
        with torch.no_grad():
            predictions = agent.forward(features)
            # Get normalized trajectory from model
            # The model outputs [batch_size, num_poses, 3] where 3 = (x, y, heading)
            trajectory_output = predictions["trajectory"].squeeze(0).cpu().numpy()

            if show_debug:
                print(f"      Raw trajectory output shape: {trajectory_output.shape}")
                print(
                    f"      Raw trajectory range: X:[{trajectory_output[:,0].min():.3f}, {trajectory_output[:,0].max():.3f}], Y:[{trajectory_output[:,1].min():.3f}, {trajectory_output[:,1].max():.3f}]"
                )

            # The model outputs absolute positions in meters
            # We need to shift them to be relative to the current ego position
            trajectory_meters = trajectory_output.copy()

            # Shift trajectory to start from origin (ego position)
            # This ensures the trajectory is relative to the current ego position
            if trajectory_meters.shape[0] > 0:
                offset = trajectory_meters[0, :2].copy()
                trajectory_meters[:, :2] = trajectory_meters[:, :2] - offset

        predicted_trajectory = Trajectory(trajectory_meters)

        if show_debug:
            print(f"    Frame {frame_idx}: Plotting trajectories")
            print(
                f"      GT shape: {gt_trajectory.poses.shape}, range X:[{gt_trajectory.poses[:,0].min():.1f}, {gt_trajectory.poses[:,0].max():.1f}], Y:[{gt_trajectory.poses[:,1].min():.1f}, {gt_trajectory.poses[:,1].max():.1f}]"
            )
            print(
                f"      Pred shape: {predicted_trajectory.poses.shape}, range X:[{predicted_trajectory.poses[:,0].min():.1f}, {predicted_trajectory.poses[:,0].max():.1f}], Y:[{predicted_trajectory.poses[:,1].min():.1f}, {predicted_trajectory.poses[:,1].max():.1f}]"
            )
            # Check if predictions look reasonable
            if predicted_trajectory.poses.shape[0] > 0:
                first_point = predicted_trajectory.poses[0]
                last_point = predicted_trajectory.poses[-1]
                print(f"      Pred first point: ({first_point[0]:.2f}, {first_point[1]:.2f})")
                print(f"      Pred last point: ({last_point[0]:.2f}, {last_point[1]:.2f})")

        # Plot trajectories directly with matplotlib for better visibility
        # Both trajectories should start from ego position (0, 0)

        # Ground truth in green (semi-transparent solid line)
        # In NAVSIM BEV: X is left/right, Y is forward/back
        # Don't negate - use coordinates as-is (positive Y = left, negative Y = right)
        gt_x = gt_trajectory.poses[:, 1]  # Y is left/right in BEV
        gt_y = gt_trajectory.poses[:, 0]  # X is forward/back
        ax_bev.plot(
            gt_x,
            gt_y,
            color="#00ff00",
            linewidth=3,
            linestyle="-",
            marker="o",
            markersize=5,
            label="Ground Truth",
            alpha=0.4,  # Semi-transparent - more visible but still distinguishable
            zorder=15,  # Higher zorder to ensure visibility
        )

        # Predicted trajectory in red (thick dashed line)
        # Apply same coordinate system
        pred_x = predicted_trajectory.poses[:, 1]  # Y is left/right
        pred_y = predicted_trajectory.poses[:, 0]  # X is forward/back

        # Check if trajectory starts near origin, if not, there might be an issue
        start_dist = np.sqrt(pred_x[0] ** 2 + pred_y[0] ** 2)
        if start_dist > 1.0:  # More than 1 meter from origin is suspicious
            print(f"      WARNING: Predicted trajectory starts {start_dist:.2f}m from origin!")
            print(f"      Start point: ({pred_x[0]:.2f}, {pred_y[0]:.2f})")

        ax_bev.plot(
            pred_x,
            pred_y,
            color="#ff0000",
            linewidth=3,
            linestyle="--",
            marker="^",
            markersize=6,
            label="Model Prediction",
            zorder=20,  # Even higher zorder to ensure it's on top
        )

        # Add ego vehicle marker at origin
        ax_bev.plot(0, 0, "ko", markersize=8, label="Ego Vehicle", zorder=25)

        # Add legend
        ax_bev.legend(loc="upper right", fontsize=10, framealpha=0.9)

        # Add metrics if available
        if gt_trajectory.poses.shape[0] > 0 and predicted_trajectory.poses.shape[0] > 0:
            # Calculate L2 error at common points
            min_len = min(gt_trajectory.poses.shape[0], predicted_trajectory.poses.shape[0])
            errors = np.linalg.norm(
                gt_trajectory.poses[:min_len, :2] - predicted_trajectory.poses[:min_len, :2],
                axis=1,
            )
            final_idx = min_len - 1
            final_error = errors[final_idx]
            avg_error = errors.mean()

            # Add text box with metrics
            textstr = f"Avg L2: {avg_error:.2f}m\nFinal L2: {final_error:.2f}m"
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax_bev.text(
                0.02,
                0.98,
                textstr,
                transform=ax_bev.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
            )

    configure_bev_ax(ax_bev)
    ax_bev.set_title("Birds Eye View - Trajectory Prediction", fontsize=12, fontweight="bold")
    ax_bev.set_xlabel("Lateral (m)")
    ax_bev.set_ylabel("Longitudinal (m)")

    # === Camera Views ===
    # Map camera names to actual camera attributes
    cameras = scene.frames[frame_idx].cameras
    camera_mapping = {
        "CAM_FRONT": cameras.cam_f0,
        "CAM_FRONT_LEFT": cameras.cam_l0,
        "CAM_FRONT_RIGHT": cameras.cam_r0,
        "CAM_SIDE_LEFT": cameras.cam_l1,
        "CAM_SIDE_RIGHT": cameras.cam_r1,
        "CAM_REAR_LEFT": cameras.cam_l2,
        "CAM_REAR": cameras.cam_b0,
        "CAM_REAR_RIGHT": cameras.cam_r2,
    }

    for cam_name, grid_pos in camera_positions:
        ax_cam = fig.add_subplot(grid_pos)

        if cam_name in camera_mapping and camera_mapping[cam_name] is not None:
            camera = camera_mapping[cam_name]

            # Draw trajectories on front camera only (like in the paper)
            if cam_name == "CAM_FRONT" and frame_idx == current_frame_idx:
                # Get base image (already in RGB)
                base_img = camera.image.copy()

                # Draw ground truth trajectory in green
                img_with_gt = draw_trajectory_on_camera(
                    base_img,
                    camera,
                    gt_trajectory,
                    color=(0, 255, 0),  # Green for ground truth (RGB format)
                    thickness=3,
                    label="Ground Truth",
                )

                # Draw predicted trajectory in red on top (matching BEV)
                img_with_both = draw_trajectory_on_camera(
                    img_with_gt,
                    camera,
                    predicted_trajectory,
                    color=(255, 0, 0),  # Red for prediction (RGB format) - matches BEV
                    thickness=4,
                    label="Prediction",
                )

                # Show the combined image
                ax_cam.imshow(img_with_both)
            else:
                # For other cameras or non-prediction frames, show normal image
                add_camera_ax(ax_cam, camera)

            ax_cam.set_title(cam_name.replace("CAM_", "").replace("_", " "), fontsize=10)
        else:
            # Handle missing camera
            ax_cam.text(
                0.5, 0.5, f"{cam_name}\nNot Available", ha="center", va="center", fontsize=10
            )
            ax_cam.set_xlim(0, 1)
            ax_cam.set_ylim(0, 1)

        ax_cam.set_xticks([])
        ax_cam.set_yticks([])

    # Add main title with scene info
    fig.suptitle(
        f"DiffusionDrive Prediction - Scene: {scene.scene_metadata.log_name}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig, None


def get_overlapping_scenes(
    scene_loader: SceneLoader,
    target_frames: int = 120,
    cache_dir: str = "/tmp",
    force_rebuild: bool = False,
    test_mode: bool = False,
    custom_cache_name: str = None,
) -> List[Tuple[str, int]]:
    """
    Get overlapping scenes for continuous prediction video.
    Uses cached scene metadata if available to avoid re-analyzing.

    Args:
        scene_loader: Scene loader with all tokens
        target_frames: Target number of frames for video
        cache_dir: Directory to store cache files
        force_rebuild: Force rebuild cache even if it exists

    Returns:
        List of (token, start_frame_idx) tuples for continuous video
    """
    print("  Finding overlapping scenes for continuous predictions...")

    # Create cache filename - use custom name if provided
    if custom_cache_name:
        # Use custom cache name
        cache_file = Path(cache_dir) / f"{custom_cache_name}.pkl"
    elif test_mode:
        # Use test mode cache
        cache_file = Path(cache_dir) / "navsim_scene_metadata_cache_test.pkl"
    else:
        # Use default cache
        cache_file = Path(cache_dir) / "navsim_scene_metadata_cache.pkl"

    # Try to load cached scene metadata (unless force rebuild)
    scenes_by_log = None
    if cache_file.exists() and not force_rebuild:
        try:
            print(f"  Loading cached scene metadata from {cache_file}")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                # Verify it has enough data
                if len(cached_data) > 0:
                    scenes_by_log = cached_data
                    print(
                        f"  Loaded {sum(len(v) for v in scenes_by_log.values())} cached scenes from {len(scenes_by_log)} logs"
                    )
        except Exception as e:
            print(f"  Warning: Could not load cache: {e}")
            scenes_by_log = None

    # If no cache, analyze ALL scenes once
    if scenes_by_log is None:
        print("  Analyzing ALL scenes (this only happens once, will be cached)...")
        scenes_by_log = defaultdict(list)

        # Analyze ALL scenes - do it once, do it right!
        total_scenes = len(scene_loader.tokens)
        print(f"  Processing {total_scenes} scenes...")

        for token in tqdm(scene_loader.tokens, desc="Analyzing scenes"):
            scene = scene_loader.get_scene_from_token(token)
            log_name = scene.scene_metadata.log_name

            # Store scene info: token, first timestamp, last timestamp
            if len(scene.frames) >= 4:  # Need at least 4 frames for prediction
                first_timestamp = scene.frames[0].timestamp
                last_timestamp = scene.frames[-1].timestamp
                scenes_by_log[log_name].append(
                    {
                        "token": token,
                        "first_ts": first_timestamp,
                        "last_ts": last_timestamp,
                        "num_frames": len(scene.frames),
                    }
                )

        # Save cache for next time
        try:
            print(f"  Saving scene metadata cache to {cache_file}")
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(dict(scenes_by_log), f)
            print(
                f"  Cached {sum(len(v) for v in scenes_by_log.values())} scenes from {len(scenes_by_log)} logs"
            )
        except Exception as e:
            print(f"  Warning: Could not save cache: {e}")

    # Find log with most scenes
    best_log = max(scenes_by_log.items(), key=lambda x: len(x[1]))
    log_name, log_scenes = best_log
    print(f"  Selected log '{log_name}' with {len(log_scenes)} scenes")

    # Sort scenes by timestamp
    log_scenes.sort(key=lambda x: x["first_ts"])

    # Build chain of overlapping scenes
    overlapping_chain = []
    frames_collected = 0

    for i, scene_info in enumerate(log_scenes):
        if frames_collected >= target_frames:
            break

        if i == 0:
            # First scene: use frames 0-3 (show progression to prediction)
            overlapping_chain.append((scene_info["token"], 0))
            frames_collected += 4
        else:
            # Check if this scene overlaps with previous
            # We want each new scene to start 1 frame after the previous
            prev_scene = log_scenes[i - 1]

            # Estimate if scenes are consecutive (allowing some gap)
            time_gap = scene_info["first_ts"] - prev_scene["last_ts"]

            # If gap is reasonable (less than 1 second), use this scene
            if time_gap < 1000000:  # 1 second in microseconds
                # Only use frame 3 (prediction frame) for subsequent scenes
                overlapping_chain.append((scene_info["token"], 3))
                frames_collected += 1

    print(f"  Found {len(overlapping_chain)} scenes for {frames_collected} total frames")
    return overlapping_chain


def create_mp4_from_scenes(
    agent: AbstractAgent,
    scene_loader: SceneLoader,
    output_path: str,
    duration_seconds: int = 60,
    fps: int = 2,
    show_all_cameras: bool = True,
    show_debug: bool = False,
    rebuild_cache: bool = False,
    test_mode: bool = False,
    custom_cache_name: str = None,
):
    """
    Create MP4 video with continuous predictions at every frame.
    Uses overlapping scenes to ensure predictions are available at each timestamp.

    Args:
        agent: Trained model
        scene_loader: Data loader
        output_path: Output MP4 path
        duration_seconds: Target video duration in seconds
        fps: Frames per second
        show_all_cameras: Whether to show all 8 cameras
        show_debug: Whether to show debug info
    """
    print(f"\n🎬 Creating continuous prediction video...")
    print(f"  Target duration: {duration_seconds} seconds")
    print(f"  FPS: {fps}")
    print(f"  Cameras: {'All 8' if show_all_cameras else 'Front 3'}")

    # Get overlapping scenes for continuous predictions
    target_frames = duration_seconds * fps
    cache_dir = "/workspace/navsim_workspace/cache"  # Better cache location
    overlapping_scenes = get_overlapping_scenes(
        scene_loader,
        target_frames,
        cache_dir,
        force_rebuild=rebuild_cache,
        test_mode=test_mode,
        custom_cache_name=custom_cache_name,
    )

    all_images = []

    for scene_idx, (token, start_frame) in enumerate(overlapping_scenes):
        scene = scene_loader.get_scene_from_token(token)

        # Always use frame 3 (prediction frame) for all scenes
        # This ensures every frame has predictions
        print(f"  Scene {scene_idx + 1}/{len(overlapping_scenes)}: {token[:20]} (frame 3)")
        frame_indices = [3]

        def create_frame(s, idx):
            return create_visualization_frame(s, agent, idx, show_all_cameras, show_debug)

        # Generate images for selected frames
        for frame_idx in frame_indices:
            fig, _ = create_frame(scene, frame_idx)

            # Convert to PIL image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf).copy()
            all_images.append(img)
            buf.close()
            plt.close(fig)

            if show_debug:
                print(f"    Frame {frame_idx}: Added to video")

    if not all_images:
        print("❌ No images generated!")
        return

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
        duration=1000 // fps,  # Convert FPS to duration in ms
        loop=0,
    )

    # Convert to MP4
    print("🎥 Converting to MP4...")

    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  ffmpeg not found! Installing...")
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)

    # Convert GIF to MP4
    cmd = [
        "ffmpeg",
        "-i",
        tmp_gif_path,
        "-movflags",
        "faststart",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={fps}",
        "-c:v",
        "libx264",
        "-crf",
        "20",  # Higher quality
        "-preset",
        "slow",  # Better compression
        "-y",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Clean up temp file
        os.unlink(tmp_gif_path)

        # Get file info
        size_mb = os.path.getsize(output_path) / (1024 * 1024)

        print("\n✅ Video successfully created!")
        print(f"  📹 File: {output_path}")
        print(f"  💾 Size: {size_mb:.2f} MB")
        print(f"  ⏱️  Duration: {actual_duration:.1f} seconds")
        print(f"  🖼️  Total frames: {len(all_images)}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting to MP4: {e}")
        print(f"  GIF saved at: {tmp_gif_path}")
        print(f"  stderr: {e.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DiffusionDrive model predictions with trajectories"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diffusiondrive_predictions.mp4",
        help="Output MP4 file path",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/workspace/navsim_workspace/dataset",
        help="Path to NAVSIM dataset root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "trainval"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Target video duration in seconds",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for video",
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Show all 8 cameras (default: front 3 only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug information",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild scene metadata cache",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: use only first 50 scenes for quick testing",
    )
    parser.add_argument(
        "--cache-name",
        type=str,
        default="navsim_scene_metadata_cache_test",
        help="Custom cache file name (e.g., 'my_cache'). If not specified, uses default naming.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" DiffusionDrive Model Prediction Visualization")
    print("=" * 70)

    # Check checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    data_root = Path(args.data_root)

    print("\n📂 Configuration:")
    print(f"  Checkpoint: {Path(args.checkpoint).name}")
    print(f"  Data root: {data_root}")
    print(f"  Split: {args.split}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load agent
    print("\n🤖 Loading DiffusionDrive agent...")
    agent = load_agent_from_checkpoint(args.checkpoint, device)
    print("  ✅ Agent loaded successfully")

    # Initialize Hydra for scene filter
    config_path = (
        Path(__file__).parent.parent
        / "navsim/planning/script/config/common/train_test_split/scene_filter"
    )
    hydra.initialize_config_dir(config_dir=str(config_path), version_base=None)
    cfg = hydra.compose(config_name="all_scenes")
    scene_filter: SceneFilter = instantiate(cfg)

    # Load scenes
    print(f"\n📊 Loading {args.split} split...")
    scene_loader = SceneLoader(
        data_root / f"navsim_logs/{args.split}",
        data_root / f"sensor_blobs/{args.split}",
        scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )
    # In test mode, limit scenes
    if args.test_mode:
        orig_count = len(scene_loader.tokens)
        # Modify the internal dictionary to only keep first 50 scenes
        tokens_list = list(scene_loader.scene_frames_dicts.keys())[:50]
        scene_loader.scene_frames_dicts = {
            k: scene_loader.scene_frames_dicts[k] for k in tokens_list
        }
        print(f"  ⚠️  TEST MODE: Using only first 50 scenes (out of {orig_count})")

    print(f"  ✅ Working with {len(scene_loader.tokens)} scenes")

    # Create video
    create_mp4_from_scenes(
        agent,
        scene_loader,
        args.output,
        duration_seconds=args.duration,
        fps=args.fps,
        show_all_cameras=args.all_cameras,
        show_debug=args.debug,
        rebuild_cache=args.rebuild_cache,
        test_mode=args.test_mode,
        custom_cache_name=args.cache_name,
    )


if __name__ == "__main__":
    main()
