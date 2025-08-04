#!/usr/bin/env python3
"""
Visualize Bench2Drive cached features and targets.

This script loads cached feature/target pairs and creates comprehensive visualizations.
"""

import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
import random
from navsim.common.bev_map_utils import get_ego_pixel_position, BEV_VIEW_FRONT, BEV_VIEW_FULL


def create_bev_semantic_colormap():
    """Create colormap for BEV semantic segmentation."""
    # NavSim BEV semantic classes
    colors = {
        0: [0, 0, 0],  # Background (black)
        1: [128, 128, 128],  # Road (gray)
        2: [255, 178, 102],  # Walkways (orange)
        3: [255, 255, 0],  # Lane centerlines (yellow)
        4: [0, 0, 255],  # Static objects (blue)
        5: [255, 0, 0],  # Vehicles (red)
        6: [0, 255, 0],  # Pedestrians (green)
    }

    # Create colormap array
    colormap = np.zeros((256, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        colormap[class_id] = color

    return colormap


def load_cached_sample(cache_dir: Path, scenario_idx: int = 0, sample_idx: int = 0):
    """Load a cached feature/target pair."""
    # List all scenario directories
    scenario_dirs = sorted([d for d in cache_dir.iterdir() if d.is_dir()])
    if not scenario_dirs:
        raise ValueError(f"No scenario directories found in {cache_dir}")

    print(f"Found {len(scenario_dirs)} scenarios")

    # Select scenario
    if scenario_idx >= len(scenario_dirs):
        print(f"Scenario index {scenario_idx} out of range, using 0")
        scenario_idx = 0

    scenario_dir = scenario_dirs[scenario_idx]
    scenario_name = scenario_dir.name
    print(f"Selected scenario: {scenario_name}")

    # List sample directories in scenario
    sample_dirs = sorted([d for d in scenario_dir.iterdir() if d.is_dir()])
    if not sample_dirs:
        raise ValueError(f"No sample directories found in {scenario_dir}")

    print(f"Found {len(sample_dirs)} samples in scenario")

    # Select sample
    if sample_idx >= len(sample_dirs):
        print(f"Sample index {sample_idx} out of range (max: {len(sample_dirs)-1}), using 0")
        sample_idx = 0

    sample_dir = sample_dirs[sample_idx]
    sample_name = sample_dir.name
    print(f"Selected sample: {sample_name}")

    # Load feature and target files
    feature_file = sample_dir / "transfuser_feature.gz"
    target_file = sample_dir / "transfuser_target.gz"

    if not feature_file.exists() or not target_file.exists():
        raise ValueError(f"Feature or target file not found in {sample_dir}")

    # Load features
    with gzip.open(feature_file, "rb") as f:
        features = pickle.load(f)

    # Load targets
    with gzip.open(target_file, "rb") as f:
        targets = pickle.load(f)

    # Extract frame index from sample name (last number)
    # NOTE: This represents the sliding window start index, NOT the actual frame used
    # The cached data uses frame at index 4 (middle frame) within this window
    try:
        frame_idx = int(sample_name.split("_")[-1])
    except:
        frame_idx = 0

    return features, targets, scenario_name, frame_idx


def load_bev_cache(bev_cache_dir: Path, scenario_name: str, frame_idx: int):
    """Load BEV map from cache."""
    # Try different file formats
    bev_file_npz = bev_cache_dir / scenario_name / f"{frame_idx:05d}.npz"
    bev_file_npy = bev_cache_dir / scenario_name / f"frame_{frame_idx:05d}.npy"

    if bev_file_npz.exists():
        # Load NPZ file
        data = np.load(bev_file_npz)
        # Extract the BEV map (usually stored as the first array)
        if "bev_map" in data:
            return data["bev_map"]
        elif "arr_0" in data:
            return data["arr_0"]
        else:
            # Try to get the first array
            keys = list(data.keys())
            if keys:
                return data[keys[0]]
    elif bev_file_npy.exists():
        return np.load(bev_file_npy)
    else:
        print(f"BEV cache file not found: {bev_file_npz} or {bev_file_npy}")
        return None


def visualize_features(features, targets, bev_map=None, save_path=None, title_prefix=""):
    """Create comprehensive visualization of all features."""

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=16)

    # 1. RGB Camera (stitched view)
    ax1 = plt.subplot(3, 3, 1)
    camera_feature = features["camera_feature"]
    # Convert from CHW to HWC and denormalize
    camera_img = camera_feature.permute(1, 2, 0).numpy()
    camera_img = np.clip(camera_img, 0, 1)  # Already in [0,1] range
    ax1.imshow(camera_img)
    ax1.set_title("Stitched Front Cameras (Left-Front-Right)")
    ax1.axis("off")

    # 2. LiDAR BEV
    ax2 = plt.subplot(3, 3, 2)
    lidar_feature = features["lidar_feature"][0].numpy()  # Remove channel dimension
    im2 = ax2.imshow(lidar_feature, cmap="viridis", origin="upper")
    ax2.set_title("LiDAR BEV Histogram")
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    plt.colorbar(im2, ax=ax2, label="Normalized Count")

    # Add ego vehicle and front direction indicator
    # LiDAR has ego at CENTER of image
    lidar_h, lidar_w = lidar_feature.shape
    ego_x_lidar = lidar_w // 2
    ego_y_lidar = lidar_h // 2  # CENTER, not bottom!
    ax2.scatter(ego_x_lidar, ego_y_lidar, c="yellow", s=100, marker="*", label="Ego", zorder=5)
    # Arrow pointing forward (upward in image, decreasing row)
    ax2.arrow(
        ego_x_lidar,
        ego_y_lidar,
        0,
        -20,  # Negative to go up (decreasing row)
        head_width=5,
        head_length=3,
        fc="yellow",
        ec="yellow",
        linewidth=2,
        label="Front",
        zorder=5,
    )
    ax2.text(
        ego_x_lidar + 5,
        ego_y_lidar + 15,
        "FRONT",
        fontsize=8,
        color="yellow",
        fontweight="bold",
        va="center",
    )

    # 3. BEV Semantic Map (always from targets - includes vehicles)
    ax3 = plt.subplot(3, 3, 3)
    # Always use BEV from targets which includes dynamic objects (vehicles)
    bev_semantic = targets["bev_semantic_map"].numpy()

    # Note if static map was available for comparison
    if bev_map is not None:
        ax3.set_title("BEV Semantic Map (with vehicles)")
    else:
        ax3.set_title("BEV Semantic Map")

    # Apply colormap
    colormap = create_bev_semantic_colormap()
    bev_colored = colormap[bev_semantic.astype(np.uint8)]
    ax3.imshow(bev_colored, origin="upper")
    ax3.set_xlabel("X (pixels)")
    ax3.set_ylabel("Y (pixels)")

    # Add ego vehicle and front direction indicator
    # Use proper ego position calculation from BEV utilities
    bev_h, bev_w = bev_semantic.shape
    if bev_h == 256 and bev_w == 256:
        # Full 360° BEV: ego at center
        ego_y, ego_x = get_ego_pixel_position(bev_h, bev_w, BEV_VIEW_FULL)
    else:
        # Front-only BEV (128x256): ego at bottom center
        ego_y, ego_x = get_ego_pixel_position(bev_h, bev_w, BEV_VIEW_FRONT)

    ax3.scatter(
        ego_x, ego_y, c="yellow", s=100, marker="*", edgecolors="black", linewidth=1, zorder=5
    )
    # Arrow pointing forward (up in image)
    ax3.arrow(
        ego_x,
        ego_y,
        0,
        -15,
        head_width=4,
        head_length=2,
        fc="yellow",
        ec="black",
        linewidth=1,
        zorder=5,
    )
    ax3.text(
        ego_x + 5,
        ego_y - 10,
        "FRONT",
        fontsize=8,
        color="yellow",
        fontweight="bold",
        va="center",
        bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5),
    )

    # Add legend for BEV classes
    class_names = ["Background", "Road", "Walkway", "Lane", "Static", "Vehicle", "Pedestrian"]
    colors_rgb = [[c / 255 for c in colormap[i]] for i in range(7)]
    legend_patches = [patches.Patch(color=colors_rgb[i], label=class_names[i]) for i in range(7)]
    ax3.legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1, 0.5))

    # 4. Trajectory on BEV
    ax4 = plt.subplot(3, 3, 4)
    # Show BEV as background
    ax4.imshow(bev_colored, origin="upper", alpha=0.5)

    # Plot trajectory
    trajectory = targets["trajectory"].numpy()  # [8, 3] (x, y, heading)

    # Convert trajectory to BEV pixels
    bev_h, bev_w = bev_semantic.shape
    resolution = 0.332  # meters per pixel

    # Use proper ego position calculation from BEV utilities
    if bev_h == 256 and bev_w == 256:
        # Full 360° BEV: ego at center
        ego_y, ego_x = get_ego_pixel_position(bev_h, bev_w, BEV_VIEW_FULL)
    else:
        # Front-only BEV (128x256): ego at bottom center
        ego_y, ego_x = get_ego_pixel_position(bev_h, bev_w, BEV_VIEW_FRONT)

    # Convert trajectory points to pixels
    # Note: trajectory is in left-handed CARLA coordinates
    # X (forward) goes up in image, Y (left) goes right
    traj_x = ego_x + trajectory[:, 1] / resolution  # Y in ego -> X in image
    traj_y = ego_y - trajectory[:, 0] / resolution  # X in ego -> Y in image (forward is up)

    # Plot trajectory
    ax4.plot(traj_x, traj_y, "r-", linewidth=3, label="Future Trajectory")
    ax4.scatter(traj_x, traj_y, c="red", s=50, zorder=5)
    ax4.scatter(
        ego_x,
        ego_y,
        c="yellow",
        s=100,
        marker="*",
        label="Ego Vehicle",
        edgecolors="black",
        linewidth=1,
    )

    # Add front direction arrow
    ax4.arrow(
        ego_x,
        ego_y,
        0,
        -15,
        head_width=4,
        head_length=2,
        fc="yellow",
        ec="black",
        linewidth=1,
        zorder=5,
    )
    ax4.text(
        ego_x + 5,
        ego_y - 10,
        "FRONT",
        fontsize=8,
        color="yellow",
        fontweight="bold",
        va="center",
        bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5),
    )

    # Add heading arrows
    for i in range(len(trajectory)):
        if i % 2 == 0:  # Show every other arrow to avoid clutter
            heading = trajectory[i, 2]
            arrow_len = 5
            dx = arrow_len * np.sin(heading)
            dy = -arrow_len * np.cos(heading)
            ax4.arrow(
                traj_x[i],
                traj_y[i],
                dx,
                dy,
                head_width=2,
                head_length=1,
                fc="red",
                ec="red",
                alpha=0.7,
            )

    ax4.set_title("Trajectory on BEV (Left-handed)")
    ax4.set_xlabel("X (pixels)")
    ax4.set_ylabel("Y (pixels)")
    ax4.legend()
    ax4.set_xlim(0, bev_w)
    ax4.set_ylim(bev_h, 0)  # Inverted to match origin="upper"

    # 5. Agent States
    ax5 = plt.subplot(3, 3, 5)
    ax5.imshow(bev_colored, origin="upper", alpha=0.5)

    # Plot agents
    agent_states = targets["agent_states"].numpy()  # [30, 5] (x, y, heading, length, width)
    agent_labels = targets["agent_labels"].numpy()  # [30] boolean

    # Use proper ego position calculation from BEV utilities
    if bev_h == 256 and bev_w == 256:
        ego_y, ego_x = get_ego_pixel_position(bev_h, bev_w, BEV_VIEW_FULL)
    else:
        ego_y, ego_x = get_ego_pixel_position(bev_h, bev_w, BEV_VIEW_FRONT)

    num_valid_agents = 0
    for i, (state, valid) in enumerate(zip(agent_states, agent_labels)):
        if valid:
            num_valid_agents += 1
            x, y, heading, length, width = state

            # Convert to BEV pixels (left-handed coordinates)
            agent_x = ego_x + y / resolution
            agent_y = ego_y - x / resolution

            # Skip if outside bounds
            if agent_x < 0 or agent_x >= bev_w or agent_y < 0 or agent_y >= bev_h:
                continue

            # Create rectangle for agent
            rect = patches.Rectangle(
                (agent_x - width / (2 * resolution), agent_y - length / (2 * resolution)),
                width / resolution,
                length / resolution,
                angle=np.degrees(-heading),  # Convert to degrees
                rotation_point="center",
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
            )
            ax5.add_patch(rect)

            # Add arrow for heading
            arrow_len = length / (2 * resolution)
            arrow_x = agent_x + arrow_len * np.sin(heading)
            arrow_y = agent_y - arrow_len * np.cos(heading)
            ax5.arrow(
                agent_x,
                agent_y,
                arrow_x - agent_x,
                arrow_y - agent_y,
                head_width=2,
                head_length=2,
                fc="blue",
                ec="blue",
            )

    ax5.scatter(
        ego_x, ego_y, c="yellow", s=100, marker="*", label="Ego", edgecolors="black", linewidth=1
    )

    # Add front direction arrow
    ax5.arrow(
        ego_x,
        ego_y,
        0,
        -15,
        head_width=4,
        head_length=2,
        fc="yellow",
        ec="black",
        linewidth=1,
        zorder=5,
    )
    ax5.text(
        ego_x + 5,
        ego_y - 10,
        "FRONT",
        fontsize=8,
        color="yellow",
        fontweight="bold",
        va="center",
        bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5),
    )

    ax5.set_title(f"Agent States on BEV ({num_valid_agents} agents)")
    ax5.set_xlabel("X (pixels)")
    ax5.set_ylabel("Y (pixels)")
    ax5.set_xlim(0, bev_w)
    ax5.set_ylim(bev_h, 0)  # Inverted to match origin="upper"

    # 6. Status Information
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis("off")
    status_feature = features["status_feature"].numpy()

    # Decode status
    command_one_hot = status_feature[:4]
    velocity = status_feature[4:6]
    acceleration = status_feature[6:8]

    command_idx = np.argmax(command_one_hot)
    command_names = ["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]

    status_text = f"Driving Command: {command_names[command_idx]}\n"
    status_text += f"Velocity: ({velocity[0]:.2f}, {velocity[1]:.2f}) m/s\n"
    status_text += f"Speed: {np.linalg.norm(velocity):.2f} m/s\n"
    status_text += f"Acceleration: ({acceleration[0]:.2f}, {acceleration[1]:.2f}) m/s²\n"

    ax6.text(
        0.1,
        0.5,
        status_text,
        fontsize=14,
        verticalalignment="center",
        transform=ax6.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
    )
    ax6.set_title("Ego Status")

    # 7. Coordinate System Info
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis("off")

    coord_text = "Coordinate Systems:\n\n"
    coord_text += "BEV/LiDAR Orientation:\n"
    if bev_h == 256 and bev_w == 256:
        coord_text += "• Ego at CENTER (256x256)\n"
        coord_text += "• Coverage: ±32m all dirs\n"
        coord_text += "• 360° view around ego\n"
    else:
        coord_text += "• Ego at BOTTOM CENTER\n"
        coord_text += "• Coverage: +32m forward\n"
        coord_text += "• Front-only view\n"
    coord_text += "\n"
    coord_text += "• Front = Up (toward top)\n"
    coord_text += "• Back = Down (toward bottom)\n"
    coord_text += "• Left = Left side\n"
    coord_text += "• Right = Right side\n\n"

    coord_text += "Fixed Issues:\n"
    coord_text += "✓ Coordinate system unified\n"
    coord_text += "✓ Agent placement fixed\n"
    coord_text += "✓ Documentation updated"

    ax7.text(
        0.1,
        0.5,
        coord_text,
        fontsize=12,
        verticalalignment="center",
        transform=ax7.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
    )
    ax7.set_title("Important Notes")

    # 8. Trajectory Details
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis("off")

    traj_text = "Trajectory Waypoints:\n\n"
    for i, (x, y, h) in enumerate(trajectory[:4]):  # Show first 4
        traj_text += f"WP{i}: X={x:6.2f}m, Y={y:6.2f}m\n"
        traj_text += f"      θ={np.degrees(h):6.1f}°\n"
    traj_text += "...\n"

    # Calculate total trajectory length
    traj_points = trajectory[:, :2]
    traj_diffs = np.diff(traj_points, axis=0)
    traj_dists = np.linalg.norm(traj_diffs, axis=1)
    total_dist = np.sum(traj_dists)
    traj_text += f"\nTotal distance: {total_dist:.1f}m"

    ax8.text(
        0.1,
        0.5,
        traj_text,
        fontsize=11,
        verticalalignment="center",
        transform=ax8.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
        family="monospace",
    )
    ax8.set_title("Trajectory Details")

    # 9. Feature Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis("off")

    stats_text = "Feature Statistics:\n\n"
    stats_text += f"Camera: {list(camera_feature.shape)}\n"
    stats_text += f"  Range: [{camera_feature.min():.3f}, {camera_feature.max():.3f}]\n\n"
    stats_text += f"LiDAR: {list(lidar_feature.shape)}\n"
    stats_text += f"  Range: [{lidar_feature.min():.3f}, {lidar_feature.max():.3f}]\n"
    stats_text += f"  Non-zero: {(lidar_feature > 0).sum()} pixels\n\n"
    stats_text += f"BEV Semantic: {list(bev_semantic.shape)}\n"
    stats_text += f"  Classes: {np.unique(bev_semantic)}\n\n"
    stats_text += f"Trajectory: {trajectory.shape[0]} waypoints\n"
    stats_text += f"Valid Agents: {agent_labels.sum()}/{len(agent_labels)}\n"

    ax9.text(
        0.1,
        0.5,
        stats_text,
        fontsize=12,
        verticalalignment="center",
        transform=ax9.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
    )
    ax9.set_title("Data Statistics")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()


def load_target_scenes():
    """Load target scenes from debug/target_scenes.txt."""
    target_scenes_file = Path(__file__).parent / "target_scenes.txt"
    if not target_scenes_file.exists():
        raise FileNotFoundError(f"Target scenes file not found: {target_scenes_file}")
    
    target_scenes = []
    with open(target_scenes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                # Parse format: ScenarioName_FrameNumber
                if '_' in line:
                    parts = line.split('_')
                    frame_number = parts[-1]  # Last part should be frame number
                    scenario_name = '_'.join(parts[:-1])  # Everything except last part
                    target_scenes.append((scenario_name, frame_number))
                else:
                    raise ValueError(f"Invalid target scene format: {line}. Expected format: ScenarioName_FrameNumber")
    
    if not target_scenes:
        raise ValueError("No valid target scenes found in target_scenes.txt")
    
    return target_scenes


def load_cached_sample_direct(sample_dir: Path, scenario_name: str, frame_number: str):
    """Load cached sample directly from a specific sample directory."""
    # Find feature and target files
    feature_files = list(sample_dir.glob("*feature*.gz"))
    target_files = list(sample_dir.glob("*target*.gz"))
    
    if not feature_files:
        raise ValueError(f"No feature files found in {sample_dir}")
    if not target_files:
        raise ValueError(f"No target files found in {sample_dir}")
    
    # Load features
    with gzip.open(feature_files[0], "rb") as f:
        features = pickle.load(f)
    
    # Load targets  
    with gzip.open(target_files[0], "rb") as f:
        targets = pickle.load(f)
    
    # Return in same format as load_cached_sample
    return features, targets, scenario_name, int(frame_number)


def main():
    parser = argparse.ArgumentParser(description="Visualize Bench2Drive cached features")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/workspace/navsim_workspace/cache/Bench2Drive-Base-training_cache/",
        help="Path to training cache directory",
    )
    parser.add_argument(
        "--bev-cache-dir",
        type=str,
        default="/workspace/navsim_workspace/cache/Bench2Drive-Base-full_bev_cache/",
        help="Path to BEV cache directory",
    )
    parser.add_argument("--scenario-idx", type=int, default=0, help="Scenario index to visualize")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index within scenario")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save visualization")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly select both scenario and frame (sample) within that scenario",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=1,
        help="Number of visualizations to generate (useful with --random and --save-path)",
    )
    parser.add_argument(
        "--use-target-scenes",
        action="store_true",
        help="Use target scenes from debug/target_scenes.txt",
    )
    args = parser.parse_args()

    # Validate mutually exclusive arguments
    if args.use_target_scenes and args.random:
        raise ValueError("--use-target-scenes and --random are mutually exclusive")

    # Convert to Path objects
    cache_dir = Path(args.cache_dir)
    bev_cache_dir = Path(args.bev_cache_dir)

    if not cache_dir.exists():
        raise ValueError(f"Cache directory not found: {cache_dir}")

    # Pre-compute scenario directories for random selection if needed
    if args.random:
        scenario_dirs = sorted([d for d in cache_dir.iterdir() if d.is_dir()])
        if not scenario_dirs:
            raise ValueError(f"No scenario directories found in {cache_dir}")
    else:
        scenario_dirs = []

    # Load target scenes if specified
    target_scenes = None
    if args.use_target_scenes:
        target_scenes = load_target_scenes()
        print(f"Loaded {len(target_scenes)} target scenes from target_scenes.txt")
        # Override num_vis to match number of target scenes
        args.num_vis = len(target_scenes)

    # Generate multiple visualizations
    for i in range(args.num_vis):
        print(f"\n=== Generating visualization {i+1}/{args.num_vis} ===")
        
        # Set current indices (copy original values for each iteration)
        current_scenario_idx = args.scenario_idx
        current_sample_idx = args.sample_idx
        
        if args.use_target_scenes and target_scenes:
            # Use target scenes
            scenario_name, target_frame = target_scenes[i]
            print(f"Using target scene: {scenario_name}, frame: {target_frame}")
            
            # Find the scenario directory
            scenario_dir = cache_dir / scenario_name
            if not scenario_dir.exists():
                raise ValueError(f"Target scenario '{scenario_name}' not found in cache directory: {cache_dir}")
            
            # Find the specific token/sample
            target_token = f"{scenario_name}_{target_frame.zfill(5)}"
            sample_dir = scenario_dir / target_token
            if not sample_dir.exists():
                # List available samples for debugging
                available_samples = [d.name for d in scenario_dir.iterdir() if d.is_dir()]
                raise ValueError(f"Target token '{target_token}' not found in scenario '{scenario_name}'. Available samples: {available_samples[:5]}...")
            
            # Load the specific sample directly
            features, targets, scenario_name, frame_idx = load_cached_sample_direct(sample_dir, scenario_name, target_frame)
            
        elif args.random:
            # Randomly select scenario
            current_scenario_idx = random.randint(0, len(scenario_dirs) - 1)
            selected_scenario_dir = scenario_dirs[current_scenario_idx]
            
            # Get all sample directories in the selected scenario
            sample_dirs = sorted([d for d in selected_scenario_dir.iterdir() if d.is_dir()])
            if not sample_dirs:
                raise ValueError(f"No sample directories found in {selected_scenario_dir}")
            
            # Randomly select sample within scenario
            current_sample_idx = random.randint(0, len(sample_dirs) - 1)
            
            print(f"Randomly selected scenario index: {current_scenario_idx} ({selected_scenario_dir.name})")
            print(f"Randomly selected sample index: {current_sample_idx} (out of {len(sample_dirs)} samples)")
            
            # Load cached sample for random selection
            print(f"Loading from cache directory: {cache_dir}")
            features, targets, scenario_name, frame_idx = load_cached_sample(
                cache_dir, current_scenario_idx, current_sample_idx
            )
        else:
            # Load cached sample for normal selection (non-random, non-target-scenes)
            print(f"Loading from cache directory: {cache_dir}")
            features, targets, scenario_name, frame_idx = load_cached_sample(
                cache_dir, current_scenario_idx, current_sample_idx
            )

        # Try to load BEV cache
        bev_map = None
        if bev_cache_dir.exists():
            print(f"Loading BEV cache for scenario: {scenario_name}, frame: {frame_idx}")
            bev_map = load_bev_cache(bev_cache_dir, scenario_name, frame_idx)
            if bev_map is not None:
                print(f"Loaded BEV map with shape: {bev_map.shape}")

        # Create title
        title = f"Scenario: {scenario_name}, Frame: {frame_idx}"

        # Determine save path for this visualization
        current_save_path = args.save_path
        if args.save_path and args.num_vis > 1:
            # Add index to save path for multiple visualizations
            if args.save_path.endswith('.png') or args.save_path.endswith('.jpg'):
                # Remove extension, add index, add extension back
                base_path = args.save_path.rsplit('.', 1)[0]
                extension = args.save_path.rsplit('.', 1)[1]
                current_save_path = f"{base_path}_{i}.{extension}"
            else:
                # No extension, just add index
                current_save_path = f"{args.save_path}_{i}.png"

        # Create visualization
        visualize_features(features, targets, bev_map, current_save_path, title)


if __name__ == "__main__":
    main()
