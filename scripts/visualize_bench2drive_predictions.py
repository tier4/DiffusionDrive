#!/usr/bin/env python3
"""
Visualization script for DiffusionDrive model predictions on Bench2Drive dataset.
Creates MP4 videos showing:
- BEV view with predicted trajectory vs ground truth
- 3 front camera views used by the model
- Properly denormalized trajectories using B2D-specific parameters
- Multiple continuous scenes for longer videos
"""

from navsim.common.bench2drive_dataloader import (
    Bench2DriveConfig,
    Bench2DriveSceneLoader,
)
from navsim.common.bench2drive_scene import Bench2DriveScene
from navsim.common.dataclasses import Trajectory
from navsim.agents.diffusiondrive.b2d_agent import Bench2DriveAgent
from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig as AgentConfig
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
from pathlib import Path
from typing import Tuple, Any, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


def draw_trajectory_on_camera(image, camera, trajectory, color=(255, 0, 0), thickness=3):
    """
    Project and draw a trajectory onto a camera image.
    NOTE: For Bench2Drive, cameras don't have transformation matrices,
    so we'll skip camera projection for now.

    Args:
        image: Input image to draw on (will be modified)
        camera: Camera object
        trajectory: Trajectory object with poses (x, y, heading)
        color: RGB color tuple for trajectory
        thickness: Line thickness

    Returns:
        Image with trajectory overlay (or original if projection not available)
    """
    # For Bench2Drive, we don't have camera intrinsics/extrinsics readily available
    # Return original image for now
    # TODO: Implement proper camera projection when transformation matrices are available
    return image.copy()


def denormalize_bench2drive_trajectory(normalized_traj: np.ndarray) -> np.ndarray:
    """
    Denormalize trajectory from model output to world coordinates using B2D parameters.
    Based on Bench2Drive normalization parameters.

    Args:
        normalized_traj: Normalized trajectory from model [N, 3] or [N, 2]

    Returns:
        Denormalized trajectory in world coordinates
    """
    # Ensure we have the right shape
    if len(normalized_traj.shape) == 1:
        normalized_traj = normalized_traj.reshape(-1, 2)

    denorm_traj = normalized_traj.copy()

    # Bench2Drive denormalization parameters
    # Denormalize X: x = (x_norm + 1) * 59.609 / 2 - 0.671
    denorm_traj[..., 0] = (normalized_traj[..., 0] + 1) * 59.609 / 2 - 0.671

    # Denormalize Y: y = (y_norm + 1) * 64.609 / 2 - 32.956
    denorm_traj[..., 1] = (normalized_traj[..., 1] + 1) * 64.609 / 2 - 32.956

    return denorm_traj


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
    show_debug: bool = False,
) -> Tuple[plt.Figure, Any]:
    """
    Create a visualization frame with BEV + 3 front cameras showing predicted and GT trajectories.

    Args:
        scene: Bench2Drive scene data
        agent: Trained agent for predictions
        frame_idx: Current frame index
        show_debug: Whether to show debug information

    Returns:
        Figure and axes
    """
    # Layout: BEV on left, 3 front cameras on right
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])

    # BEV takes left column spanning all rows
    ax_bev = fig.add_subplot(gs[:, 0])

    # 3 front cameras on the right
    camera_positions = [
        ("Front", gs[0, 1:]),  # Front camera spans top-right
        ("Front-Left", gs[1, 1]),  # Bottom-left camera
        ("Front-Right", gs[1, 2]),  # Bottom-right camera
    ]

    # === BEV Visualization ===
    # Try to add BEV if available
    try:
        # For Bench2Drive, we might not have map_api, so handle gracefully
        if hasattr(scene, "map_api") and scene.map_api is not None:
            add_configured_bev_on_ax(ax_bev, scene.map_api, scene.frames[frame_idx])
        else:
            # Just create a basic BEV plot
            ax_bev.set_xlim(-50, 50)
            ax_bev.set_ylim(-50, 50)
            ax_bev.grid(True, alpha=0.3)
    except Exception as e:
        print(f"Warning: Could not add BEV map: {e}")
        ax_bev.set_xlim(-50, 50)
        ax_bev.set_ylim(-50, 50)
        ax_bev.grid(True, alpha=0.3)

    # Add LiDAR if available
    try:
        if hasattr(scene, "frames") and scene.frames[frame_idx].lidar is not None:
            add_lidar_to_bev_ax(ax_bev, scene.frames[frame_idx].lidar)
    except:
        pass  # LiDAR not available for Bench2Drive scenes

    # Compute and plot trajectories at the current frame
    current_frame_idx = scene.history_frames - 1

    if frame_idx == current_frame_idx:
        # Get ground truth trajectory (returns torch.Tensor)
        gt_trajectory_tensor = scene.get_future_trajectory()
        # Convert to numpy and create Trajectory object
        gt_poses = (
            gt_trajectory_tensor.numpy()
            if isinstance(gt_trajectory_tensor, torch.Tensor)
            else gt_trajectory_tensor
        )
        gt_trajectory = Trajectory(gt_poses)

        # Get model prediction
        agent_input = scene.get_agent_input()

        # Custom compute_trajectory that handles device placement
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
            # Get normalized trajectory from model
            trajectory_output = predictions["trajectory"].squeeze(0).cpu().numpy()

            if show_debug:
                print(f"      Raw trajectory output shape: {trajectory_output.shape}")
                print(
                    f"      Raw trajectory range: X:[{trajectory_output[:,0].min():.3f}, "
                    f"{trajectory_output[:,0].max():.3f}], Y:[{trajectory_output[:,1].min():.3f}, "
                    f"{trajectory_output[:,1].max():.3f}]"
                )

            # The model outputs absolute positions, but we need relative positions from ego
            denorm_poses = trajectory_output.copy()

            # Shift trajectory to start from origin (ego position)
            if denorm_poses.shape[0] > 0:
                offset = denorm_poses[0, :2].copy()
                denorm_poses[:, :2] = denorm_poses[:, :2] - offset

        predicted_trajectory = Trajectory(denorm_poses)

        if show_debug:
            print(f"    Frame {frame_idx}: Plotting trajectories")
            print(
                f"      GT shape: {gt_trajectory.poses.shape}, "
                f"range X:[{gt_trajectory.poses[:,0].min():.1f}, {gt_trajectory.poses[:,0].max():.1f}], "
                f"Y:[{gt_trajectory.poses[:,1].min():.1f}, {gt_trajectory.poses[:,1].max():.1f}]"
            )
            print(
                f"      Pred shape: {predicted_trajectory.poses.shape}, "
                f"range X:[{predicted_trajectory.poses[:,0].min():.1f}, {predicted_trajectory.poses[:,0].max():.1f}], "
                f"Y:[{predicted_trajectory.poses[:,1].min():.1f}, {predicted_trajectory.poses[:,1].max():.1f}]"
            )

        # Plot trajectories on BEV
        # For Bench2Drive/CARLA: X is forward, Y is right
        # In BEV visualization: typically X is right, Y is forward
        gt_x = gt_trajectory.poses[:, 1]  # Y (right) becomes X in BEV
        gt_y = gt_trajectory.poses[:, 0]  # X (forward) becomes Y in BEV
        ax_bev.plot(
            gt_x,
            gt_y,
            color="#00ff00",
            linewidth=5,
            linestyle="-",
            marker="o",
            markersize=12,
            label="Ground Truth",
            alpha=0.8,
            zorder=15,
        )

        # Predicted trajectory in red
        pred_x = predicted_trajectory.poses[:, 1]
        pred_y = predicted_trajectory.poses[:, 0]
        ax_bev.plot(
            pred_x,
            pred_y,
            color="#ff0000",
            linewidth=5,
            linestyle="--",
            marker="^",
            markersize=14,
            label="Model Prediction",
            zorder=20,
        )

        # Add ego vehicle marker at origin
        ax_bev.plot(0, 0, "ko", markersize=15, label="Ego Vehicle", zorder=25)

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
    # Get cameras from the current frame
    current_cameras = scene.get_agent_input().cameras[-1]  # Get last (current) frame cameras

    # Bench2Drive uses these 3 front cameras
    camera_mapping = {
        "Front": current_cameras.cam_f0,
        "Front-Left": current_cameras.cam_l0,
        "Front-Right": current_cameras.cam_r0,
    }

    # Store trajectories for camera overlay (only if we computed them)
    cam_gt_trajectory = gt_trajectory if frame_idx == current_frame_idx else None
    cam_pred_trajectory = predicted_trajectory if frame_idx == current_frame_idx else None

    for cam_name, grid_pos in camera_positions:
        ax_cam = fig.add_subplot(grid_pos)

        if cam_name in camera_mapping and camera_mapping[cam_name] is not None:
            camera = camera_mapping[cam_name]

            # Draw trajectories on front camera only
            if (
                cam_name == "Front"
                and frame_idx == current_frame_idx
                and cam_gt_trajectory is not None
            ):
                # Get base image (already in RGB)
                base_img = camera.image.copy()

                # Draw ground truth trajectory in green
                img_with_gt = draw_trajectory_on_camera(
                    base_img,
                    camera,
                    cam_gt_trajectory,
                    color=(0, 255, 0),  # Green for ground truth
                    thickness=3,
                )

                # Draw predicted trajectory in red on top
                img_with_both = draw_trajectory_on_camera(
                    img_with_gt,
                    camera,
                    cam_pred_trajectory,
                    color=(255, 0, 0),  # Red for prediction
                    thickness=4,
                )

                # Show the combined image
                ax_cam.imshow(img_with_both)
            else:
                # For other cameras, show normal image
                add_camera_ax(ax_cam, camera)

            ax_cam.set_title(cam_name, fontsize=10)
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
    scene_name = getattr(scene, "token", "Unknown Scene")
    fig.suptitle(
        f"DiffusionDrive B2D Prediction - Scene: {scene_name}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig, None


def create_mp4_from_scenes(
    agent: Bench2DriveAgent,
    scene_loader: Bench2DriveSceneLoader,
    output_path: str,
    num_scenes: int = 5,
    fps: int = 2,
    show_debug: bool = False,
):
    """
    Create MP4 video with predictions from multiple Bench2Drive scenes.

    Args:
        agent: Trained model
        scene_loader: Data loader
        output_path: Output MP4 path
        num_scenes: Number of scenes to visualize
        fps: Frames per second
        show_debug: Whether to show debug info
    """
    print(f"\n🎬 Creating Bench2Drive prediction video...")
    print(f"  Number of scenes: {num_scenes}")
    print(f"  FPS: {fps}")

    all_images = []

    # Process the requested number of scenes
    num_to_process = min(num_scenes, len(scene_loader))

    for scene_idx in range(num_to_process):
        token = scene_loader.scene_tokens[scene_idx]
        scene = scene_loader.get_scene(token)

        print(f"  Scene {scene_idx + 1}/{num_to_process}: {token[:40]}...")

        # Use the history frame (where prediction happens)
        frame_idx = scene.history_frames - 1

        # Create visualization frame
        fig, _ = create_visualization_frame(scene, agent, frame_idx, show_debug)

        # Convert to PIL image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf).copy()
        all_images.append(img)
        buf.close()
        plt.close(fig)

        if show_debug:
            print(f"    Frame added to video")

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
        duration=1000 // fps,
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
        "20",
        "-preset",
        "slow",
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
        description="Visualize DiffusionDrive model predictions on Bench2Drive dataset"
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
        default="bench2drive_predictions.mp4",
        help="Output MP4 file path",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/workspace/Bench2Drive-Base",
        help="Path to Bench2Drive dataset root",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="List of scenarios to visualize (full scenario names). If not specified, randomly chooses from available scenarios",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly select scenarios from the dataset",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for scenario selection (for reproducibility)",
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
        help="Number of scenes to visualize",
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
    args = parser.parse_args()

    print("=" * 70)
    print(" DiffusionDrive Bench2Drive Prediction Visualization")
    print("=" * 70)

    # Check checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    data_root = Path(args.data_root)

    # Handle scenario selection
    import random

    if args.random_seed is not None:
        random.seed(args.random_seed)
        print(f"\n🎲 Random seed set to: {args.random_seed}")

    if args.random or args.scenarios is None:
        # Get all available scenarios from the dataset
        all_scenarios = [
            d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

        if not all_scenarios:
            print(f"❌ No scenarios found in {data_root}")
            sys.exit(1)

        # Randomly select scenarios
        num_to_select = min(10, len(all_scenarios))  # Select up to 10 random scenarios
        selected_scenarios = random.sample(all_scenarios, num_to_select)
        print(
            f"\n🎲 Randomly selected {num_to_select} scenarios from {len(all_scenarios)} available"
        )
        print(
            f"  Selected: {', '.join(s[:30] + '...' if len(s) > 30 else s for s in selected_scenarios[:3])}"
        )
        if len(selected_scenarios) > 3:
            print(f"  ... and {len(selected_scenarios) - 3} more")
    else:
        selected_scenarios = args.scenarios
        print(f"\n📋 Using specified scenarios: {selected_scenarios}")

    print("\n📂 Configuration:")
    print(f"  Checkpoint: {Path(args.checkpoint).name}")
    print(f"  Data root: {data_root}")
    print(f"  Number of scenarios: {len(selected_scenarios)}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load agent
    print("\n🤖 Loading Bench2Drive agent...")
    agent = load_bench2drive_agent_from_checkpoint(args.checkpoint, device)
    print("  ✅ Agent loaded successfully")

    # Create scene loader
    print(f"\n📊 Loading Bench2Drive scenes...")

    # Create Bench2Drive configuration
    config = Bench2DriveConfig(
        data_root=data_root,
        scenarios=selected_scenarios,
        sampling_rate=5,  # 10Hz to 2Hz
        num_frames=30,
        num_history_frames=4,
        num_future_frames=26,
        bev_cache_dir=Path(args.bev_cache_dir) if args.bev_cache_dir else None,
    )

    scene_loader = Bench2DriveSceneLoader(config)
    print(f"  ✅ Loaded {len(scene_loader)} scenes from {len(selected_scenarios)} scenarios")

    # Create video
    create_mp4_from_scenes(
        agent,
        scene_loader,
        args.output,
        num_scenes=args.num_scenes,
        fps=args.fps,
        show_debug=args.debug,
    )


if __name__ == "__main__":
    main()
