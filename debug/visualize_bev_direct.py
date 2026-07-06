#!/usr/bin/env python3
"""
Direct BEV visualization from test data without using cache.
This script loads raw Bench2Drive data and processes it through the feature/target builders
to visualize the BEV generation logic.
"""

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder,
)
from navsim.common.bench2drive_dataloader import Bench2DriveDataConfig, Bench2DriveSceneLoader

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))


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


def visualize_direct_bev(features, targets, title="Direct BEV Visualization", save_path=None):
    """Create comprehensive visualization of features and BEV."""

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(title, fontsize=16)

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
    # Use origin="upper" for consistency with BEV semantic map
    im2 = ax2.imshow(lidar_feature, cmap="viridis", origin="upper")
    ax2.set_title("LiDAR BEV Histogram")
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    plt.colorbar(im2, ax=ax2, label="Normalized Count")

    # Add ego vehicle
    lidar_h, lidar_w = lidar_feature.shape
    ego_col_lidar = lidar_w // 2
    ego_row_lidar = lidar_h // 2  # LiDAR is typically full view (ego at center)
    ax2.scatter(ego_col_lidar, ego_row_lidar, c="yellow", s=100, marker="*", label="Ego", zorder=5)
    # Arrow pointing forward (upward in image, decreasing row)
    ax2.arrow(
        ego_col_lidar,
        ego_row_lidar,
        0,
        -20,  # Negative to go up (decreasing row)
        head_width=5,
        head_length=3,
        fc="yellow",
        ec="yellow",
        linewidth=2,
        zorder=5,
    )

    # 3. BEV Semantic Map (with agents)
    ax3 = plt.subplot(3, 3, 3)
    bev_semantic = targets["bev_semantic_map"].numpy()

    # Apply colormap
    colormap = create_bev_semantic_colormap()
    bev_colored = colormap[bev_semantic.astype(np.uint8)]
    # Note: origin="upper" because row 0 is at top in our BEV convention
    ax3.imshow(bev_colored, origin="upper")
    ax3.set_title("BEV Semantic Map (with agents)")
    ax3.set_xlabel("X (pixels)")
    ax3.set_ylabel("Y (pixels)")

    # Add ego vehicle
    # B2D BEV convention:
    # - Front-half BEV (128x256): ego at bottom-center (row=127, col=128)
    # - Full BEV (256x256): ego at center (row=128, col=128)
    from navsim.common.bev_map_utils import get_ego_pixel_position, BEV_VIEW_FRONT, BEV_VIEW_FULL

    bev_h, bev_w = bev_semantic.shape
    if bev_h == 128 and bev_w == 256:
        # Front-half view
        ego_row, ego_col = get_ego_pixel_position(bev_h, bev_w, BEV_VIEW_FRONT)
    elif bev_h == 256 and bev_w == 256:
        # Full view
        ego_row, ego_col = get_ego_pixel_position(bev_h, bev_w, BEV_VIEW_FULL)
    else:
        # Default to center
        ego_col = bev_w // 2
        ego_row = bev_h // 2

    # For matplotlib with origin="upper", we plot (col, row)
    ax3.scatter(
        ego_col, ego_row, c="yellow", s=100, marker="*", edgecolors="black", linewidth=1, zorder=5
    )
    # Arrow pointing forward (upward in image, decreasing row)
    ax3.arrow(
        ego_col,
        ego_row,
        0,
        -15,  # Negative to go up (decreasing row)
        head_width=4,
        head_length=2,
        fc="yellow",
        ec="black",
        linewidth=1,
        zorder=5,
    )

    # Add legend
    class_names = ["Background", "Road", "Walkway", "Lane", "Static", "Vehicle", "Pedestrian"]
    colors_rgb = [[c / 255 for c in colormap[i]] for i in range(7)]
    legend_patches = [patches.Patch(color=colors_rgb[i], label=class_names[i]) for i in range(7)]
    ax3.legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1, 0.5))

    # 4. Trajectory on BEV
    ax4 = plt.subplot(3, 3, 4)
    ax4.imshow(bev_colored, origin="upper", alpha=0.5)

    # Plot trajectory
    trajectory = targets["trajectory"].numpy()
    # Use B2D BEV resolution: 85m / 256 pixels = 0.332m/pixel
    from navsim.common.bench2drive_constants import BEV_SEMANTIC_RESOLUTION

    resolution = BEV_SEMANTIC_RESOLUTION  # 0.332 meters per pixel for B2D

    # Convert trajectory to pixels using CARLA coordinate system
    # trajectory[:, 0] is X (forward in ego) -> decreases row
    # trajectory[:, 1] is Y (right in ego) -> increases column
    traj_cols = ego_col + trajectory[:, 1] / resolution
    traj_rows = ego_row - trajectory[:, 0] / resolution

    # Debug trajectory coordinates
    print(f"  Trajectory pixel coordinates:")
    for i in range(min(4, len(trajectory))):
        print(
            f"    WP{i}: col={traj_cols[i]:.1f}, row={traj_rows[i]:.1f} (ego: col={ego_col}, row={ego_row})"
        )
    print(f"    BEV bounds: cols=[0,{bev_colored.shape[1]}], rows=[0,{bev_colored.shape[0]}]")

    ax4.plot(traj_cols, traj_rows, "r-", linewidth=3, label="Future Trajectory")
    ax4.scatter(traj_cols, traj_rows, c="red", s=50, zorder=5)
    ax4.scatter(
        ego_col,
        ego_row,
        c="yellow",
        s=100,
        marker="*",
        label="Ego Vehicle",
        edgecolors="black",
        linewidth=1,
    )

    # Add heading arrows
    for i in range(0, len(trajectory), 2):
        heading = trajectory[i, 2]
        arrow_len = 5
        # In image coordinates with origin="upper":
        # dx (column change) = sin(heading) for rightward
        # dy (row change) = -cos(heading) for upward (decreasing row)
        dx = arrow_len * np.sin(heading)
        dy = -arrow_len * np.cos(heading)
        ax4.arrow(
            traj_cols[i],
            traj_rows[i],
            dx,
            dy,
            head_width=2,
            head_length=1,
            fc="red",
            ec="red",
            alpha=0.7,
        )

    ax4.set_title("Trajectory on BEV")
    ax4.set_xlabel("X (pixels)")
    ax4.set_ylabel("Y (pixels)")
    ax4.legend()
    ax4.set_xlim(0, bev_w)
    ax4.set_ylim(bev_h, 0)  # Inverted to match origin="upper"

    # 5. Agent States on BEV
    ax5 = plt.subplot(3, 3, 5)
    ax5.imshow(bev_colored, origin="upper", alpha=0.5)

    # Plot agents
    agent_states = targets["agent_states"].numpy()
    agent_labels = targets["agent_labels"].numpy()

    num_valid_agents = 0
    for i, (state, valid) in enumerate(zip(agent_states, agent_labels)):
        if valid:
            num_valid_agents += 1
            x, y, heading, length, width = state

            # DEBUG: Print agent heading information
            print(f"Agent {i}: heading={heading:.3f} rad = {np.degrees(heading):.1f}°")

            # Convert to BEV pixels using CARLA coordinate system
            # x (forward) -> decreases row, y (right) -> increases column
            agent_col = ego_col + y / resolution
            agent_row = ego_row - x / resolution

            # Skip if outside bounds
            if agent_col < 0 or agent_col >= bev_w or agent_row < 0 or agent_row >= bev_h:
                continue

            # Create rectangle for agent
            # For matplotlib with origin="upper", Rectangle expects (x, y) = (col, row_from_top)
            rect = patches.Rectangle(
                (agent_col - width / (2 * resolution), agent_row - length / (2 * resolution)),
                width / resolution,
                length / resolution,
                angle=np.degrees(heading),
                rotation_point="center",
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
            )
            ax5.add_patch(rect)

            # Add arrow for heading (use same transformation as rectangle)
            arrow_len = length / (2 * resolution)
            arrow_dx = arrow_len * np.sin(heading)  # Match rectangle angle convention
            arrow_dy = -arrow_len * np.cos(
                heading
            )  # Negative for upward in matplotlib (origin="upper")
            ax5.arrow(
                agent_col,
                agent_row,
                arrow_dx,
                arrow_dy,
                head_width=2,
                head_length=2,
                fc="blue",
                ec="blue",
            )

    ax5.scatter(
        ego_col,
        ego_row,
        c="yellow",
        s=100,
        marker="*",
        label="Ego",
        edgecolors="black",
        linewidth=1,
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

    # DEBUG: Get ego heading from trajectory for comparison
    trajectory = targets["trajectory"].numpy()
    if len(trajectory) > 0:
        ego_heading_from_traj = trajectory[0, 2]  # First waypoint should be close to ego heading
        print(
            f"DEBUG: Ego heading from trajectory[0]: {ego_heading_from_traj:.3f} rad = {np.degrees(ego_heading_from_traj):.1f}°"
        )

        # Analyze coordinate system orientation
        print("DEBUG: Coordinate system analysis:")
        print(f"  Trajectory arrows use: dx=sin(heading), dy=-cos(heading)")
        print(f"  Agent arrows now use:  dx=sin(heading), dy=-cos(heading)")
        print(f"  Both should have consistent orientation now")

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

    # 7. BEV Generation Process
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis("off")

    process_text = "BEV Generation Process:\n\n"
    process_text += "1. Load HD map (static elements)\n"
    process_text += "2. Process agent states:\n"
    process_text += "   - Convert to BEV coordinates\n"
    process_text += "   - Render as semantic class 5\n"
    process_text += "3. Combine map + agents\n"
    process_text += "4. Output semantic map\n\n"
    process_text += f"BEV Shape: {bev_semantic.shape}\n"
    process_text += f"Unique classes: {np.unique(bev_semantic)}\n"

    ax7.text(
        0.1,
        0.5,
        process_text,
        fontsize=12,
        verticalalignment="center",
        transform=ax7.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
    )
    ax7.set_title("BEV Generation Info")

    # 8. Trajectory Details
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis("off")

    traj_text = "Trajectory Waypoints:\n\n"
    for i in range(min(4, len(trajectory))):
        x, y, h = trajectory[i]
        traj_text += f"WP{i}: X={x:6.2f}m, Y={y:6.2f}m\n"
        traj_text += f"      θ={np.degrees(h):6.1f}°\n"

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
        plt.close()  # Close figure to free memory when saving multiple files
    else:
        plt.savefig("direct_bev_visualization.png", dpi=150, bbox_inches="tight")
        print("Saved visualization to direct_bev_visualization.png")
        plt.show()  # Only show interactively when not saving to specific path

    # Generate coordinate assumption tests
    try:
        from debug.test_agent_coordinate_assumptions import (
            test_coordinate_assumptions,
            test_heading_assumptions,
        )
        from debug.test_rectangle_heading import test_rectangle_vs_arrow_heading

        # Extract valid agents for testing
        valid_agents = []
        for i, (state, valid) in enumerate(zip(agent_states, agent_labels)):
            if valid:
                valid_agents.append(state)

        # if valid_agents and save_path:
        #     # Create coordinate test file path
        #     coord_test_path = save_path.replace(".png", "_coordinate_tests.png")
        #     test_coordinate_assumptions(
        #         agents=valid_agents,
        #         bev_colored=bev_colored,
        #         ego_row=ego_row,
        #         ego_col=ego_col,
        #         resolution=resolution,
        #         save_path=coord_test_path,
        #     )

        #     # Create heading test file path
        #     heading_test_path = save_path.replace(".png", "_heading_tests.png")
        #     test_heading_assumptions(
        #         agents=valid_agents,
        #         bev_colored=bev_colored,
        #         ego_row=ego_row,
        #         ego_col=ego_col,
        #         resolution=resolution,
        #         save_path=heading_test_path,
        #     )

        #     # Create rectangle vs arrow test file path
        #     rect_test_path = save_path.replace(".png", "_rectangle_tests.png")
        #     test_rectangle_vs_arrow_heading(
        #         agents=valid_agents,
        #         bev_colored=bev_colored,
        #         ego_row=ego_row,
        #         ego_col=ego_col,
        #         resolution=resolution,
        #         save_path=rect_test_path,
        #     )
        # elif valid_agents:
        #     test_coordinate_assumptions(
        #         agents=valid_agents,
        #         bev_colored=bev_colored,
        #         ego_row=ego_row,
        #         ego_col=ego_col,
        #         resolution=resolution,
        #         save_path="coordinate_assumption_tests.png",
        #     )

        #     test_heading_assumptions(
        #         agents=valid_agents,
        #         bev_colored=bev_colored,
        #         ego_row=ego_row,
        #         ego_col=ego_col,
        #         resolution=resolution,
        #         save_path="heading_assumption_tests.png",
        #     )

    except ImportError as e:
        print(f"Could not import coordinate/heading test module: {e}")
    except Exception as e:
        print(f"Error generating coordinate tests: {e}")


def load_target_scenes():
    """Load target scenes from debug/target_scenes.txt."""
    target_scenes_file = Path(__file__).parent / "target_scenes.txt"
    if not target_scenes_file.exists():
        raise FileNotFoundError(f"Target scenes file not found: {target_scenes_file}")

    target_scenes = []
    with open(target_scenes_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # Skip empty lines and comments
                # Parse format: ScenarioName_FrameNumber
                if "_" in line:
                    parts = line.split("_")
                    frame_number = parts[-1]  # Last part should be frame number
                    scenario_name = "_".join(parts[:-1])  # Everything except last part
                    target_scenes.append((scenario_name, frame_number))
                else:
                    raise ValueError(
                        f"Invalid target scene format: {line}. Expected format: ScenarioName_FrameNumber"
                    )

    if not target_scenes:
        raise ValueError("No valid target scenes found in target_scenes.txt")

    return target_scenes


def create_gif_from_pngs(png_paths, gif_path, duration=1000, loop=0):
    """
    Create an animated GIF from a list of PNG files.

    Args:
        png_paths: List of paths to PNG files
        gif_path: Output path for the GIF file
        duration: Duration of each frame in milliseconds (default: 1000ms = 1s)
        loop: Number of loops (0 = infinite loop)
    """
    if not png_paths:
        print("No PNG files provided for GIF creation")
        return

    print(f"Creating GIF from {len(png_paths)} PNG files...")

    # Load all images
    images = []
    for png_path in sorted(
        png_paths, key=lambda x: int(Path(x).stem.split("_")[-1])
    ):  # Sort to ensure consistent order
        if Path(png_path).exists():
            img = Image.open(png_path)
            # Convert to RGB to ensure compatibility
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        else:
            print(f"Warning: PNG file not found: {png_path}")

    if not images:
        print("No valid PNG files found for GIF creation")
        return

    # Save as animated GIF
    if images:
        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=duration, loop=loop
        )
        print(f"GIF saved to: {gif_path}")


def main():
    """Main function to load test data and visualize BEV generation."""

    import argparse
    import random

    parser = argparse.ArgumentParser(description="Visualize Bench2Drive direct BEV generation")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/workspace/Bench2Drive-Base",
        help="Path to Bench2Drive dataset root directory",
    )
    parser.add_argument(
        "--bev-cache-dir",
        type=str,
        default=None,
        help="Path to BEV cache directory (optional, generates on the fly if not specified)",
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        default="/workspace/Bench2Drive-Map",
        help="Path to HD map directory",
    )
    parser.add_argument("--scenario-idx", type=int, default=0, help="Scenario index to visualize")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save visualization")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly select both scenario and frame",
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
    parser.add_argument(
        "--generation-type",
        type=str,
        default="vector",
        choices=["vector", "segmentation"],
        help="BEV generation type for debug mode (default: vector)",
    )
    args = parser.parse_args()

    # Validate mutually exclusive arguments
    if args.use_target_scenes and args.random:
        raise ValueError("--use-target-scenes and --random are mutually exclusive")

    # Configuration - use workspace data structure
    data_root = Path(args.data_root)

    if not data_root.exists():
        raise ValueError(f"Data directory not found: {data_root}")

    # Find available scenarios (subdirectories with anno folders)
    scenarios = []
    for d in data_root.iterdir():
        if d.is_dir() and not d.name.startswith(".") and (d / "anno").exists():
            scenarios.append(d.name)

    if not scenarios:
        print(f"No scenarios found in {data_root}")
        raise ValueError("No scenarios found in test data")

    print(f"Found scenarios: {scenarios}")

    # Map directory - REQUIRED
    map_dir = Path(args.map_dir)
    if not map_dir.exists():
        raise FileNotFoundError(f"HD map directory not found: {map_dir}")

    # BEV cache directory (optional)
    if args.bev_cache_dir:
        bev_cache_dir = Path(args.bev_cache_dir)
        if not bev_cache_dir.exists():
            raise FileNotFoundError(f"BEV cache directory not found: {bev_cache_dir}")
        print(f"Using BEV cache directory: {bev_cache_dir}")
    else:
        bev_cache_dir = None
        print("No BEV cache specified, will generate on the fly")

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

        # Select scenario
        target_frame = None  # Initialize target_frame
        if args.use_target_scenes and target_scenes:
            scenario, target_frame = target_scenes[i]
            print(f"Using target scene: {scenario}, frame: {target_frame}")
            # Verify scenario exists in available scenarios
            if scenario not in scenarios:
                raise ValueError(
                    f"Target scenario '{scenario}' not found in available scenarios: {scenarios}"
                )
        elif args.random:
            scenario = random.choice(scenarios)
            print(f"Randomly selected scenario: {scenario}")
        else:
            if args.scenario_idx >= len(scenarios):
                print(f"Scenario index {args.scenario_idx} out of range, using 0")
                scenario_idx = 0
            else:
                scenario_idx = args.scenario_idx
            scenario = scenarios[scenario_idx]
            print(f"Using scenario: {scenario}")

        # Validate HD map file for the scenario
        scenario_parts = scenario.split("_")
        town_name = None
        for part in scenario_parts:
            if part.startswith("Town"):
                town_name = part
                break

        if not town_name:
            raise ValueError(f"Cannot extract town name from scenario: {scenario}")

        map_file = map_dir / f"{town_name}_HD_map.npz"
        if not map_file.exists():
            raise FileNotFoundError(f"HD map file not found: {map_file}")

        print(f"Using HD map: {map_file}")

        # Create Bench2Drive configuration
        config = Bench2DriveDataConfig(
            data_root=data_root,
            scenarios=[scenario],
            sampling_rate=5,  # No downsampling for test data
            num_frames=13,  # Need at least 4 history + 1 current + 8 future = 13 frames
            num_history_frames=4,
            num_future_frames=8,  # Need 8 future frames for 8 waypoints
            extract_tar=False,
            map_dir=map_dir,
            bev_cache_dir=bev_cache_dir,
            debug_mode=True,  # Enable debug mode to allow on-the-fly BEV generation
        )

        # Store generation type in config for debug mode
        config.debug_generation_type = args.generation_type

        # Create scene loader
        scene_loader = Bench2DriveSceneLoader(config)

        # Get scene tokens
        tokens = scene_loader.get_scene_tokens()
        if not tokens:
            print(f"No scene tokens found for scenario {scenario}")
            raise ValueError("No scene tokens found")

        print(f"Found {len(tokens)} scene tokens")

        # Select token/frame
        if args.use_target_scenes and target_frame:
            # Use specific target frame
            target_token = f"{scenario}_{target_frame:05}"
            if target_token in tokens:
                token = target_token
                print(f"Found target token: {token}")
            else:
                raise ValueError(
                    f"Target token '{target_token}' not found in available tokens: {tokens[:5]}..."
                )
        elif args.random:
            # Randomly select a token
            token = random.choice(tokens)
            print(f"Randomly selected token: {token}")
        else:
            # Use specific frame logic (frame 20) or fallback
            token = None
            for t in tokens:
                # Extract frame index from token
                frame_idx = int(t.split("_")[-1])
                # Check if this scene window would include frame 20
                # (frame_idx is the starting frame of the window)
                if frame_idx <= 20 and frame_idx + 13 > 20:  # 13 frames in window
                    token = t
                    print(f"Using token {token} which includes frame 20")
                    break

            if token is None:
                print("No suitable token found that includes frame 20")
                # Fall back to first token
                token = tokens[0]
                print(f"Using first token: {token}")

        # Load scene
        scene = scene_loader.get_scene(token)

        # Create model config
        model_config = TransfuserConfig()

        # Create feature and target builders
        feature_builder = Bench2DriveFeatureBuilder(model_config)
        target_builder = Bench2DriveTargetBuilder(model_config)

        # Debug: Check how many frames are in the scene
        print(f"\nScene info:")
        print(f"  Total frames in scene: {len(scene.anno_paths)}")
        print(f"  History frames: {scene.history_frames}")
        print(f"  Future frames: {scene.future_frames}")

        # Get agent input for the current frame (NavSim convention: middle frame)
        agent_input = scene.get_agent_input(4)  # num_history_frames - 1

        # Compute features
        print("Computing features...")
        features = feature_builder.compute_features(agent_input)

        # Compute targets (including BEV with agents)
        print("Computing targets (including BEV generation)...")
        targets = target_builder.compute_targets(scene)

        if targets is None:
            # FAIL FAST AND LOUD - don't silently return
            raise ValueError(
                f"Cannot generate BEV: compute_targets returned None. "
                f"This typically happens when there aren't enough future frames for trajectory. "
                f"Scene has {len(scene.anno_paths)} total frames. "
                f"Try using an earlier frame in the sequence."
            )

        # DEBUG: Add detailed vehicle detection debugging
        print("\n=== DETAILED VEHICLE DETECTION DEBUG ===")

        # Get the current frame annotation data for debugging
        current_frame_idx = 4  # History frames index for current frame
        current_anno_path = scene.anno_paths[current_frame_idx]
        print(f"Current frame annotation: {current_anno_path}")

        import json
        import gzip

        # Handle gzipped JSON files
        if str(current_anno_path).endswith(".gz"):
            with gzip.open(current_anno_path, "rt") as f:
                anno_data = json.load(f)
        else:
            with open(current_anno_path, "r") as f:
                anno_data = json.load(f)

        # Continue with normal visualization flow

        # Print info about BEV generation
        bev_map = targets["bev_semantic_map"].numpy()
        print(f"\nBEV Semantic Map Info:")
        print(f"  Shape: {bev_map.shape}")
        print(f"  Unique classes: {np.unique(bev_map)}")
        print(f"  Class counts:")
        for class_id in np.unique(bev_map):
            count = (bev_map == class_id).sum()
            class_names = [
                "Background",
                "Road",
                "Walkway",
                "Lane",
                "Static",
                "Vehicle",
                "Pedestrian",
            ]
            class_id_int = int(class_id)  # Convert to int for indexing
            if class_id_int < len(class_names):
                print(f"    {class_names[class_id_int]}: {count} pixels")
            else:
                print(f"    Class {class_id_int}: {count} pixels")

        # Check agent data
        agent_states = targets["agent_states"].numpy()
        agent_labels = targets["agent_labels"].numpy()
        num_agents = agent_labels.sum()
        print(f"\nAgent Info:")
        print(f"  Total slots: {len(agent_labels)}")
        print(f"  Valid agents: {num_agents}")

        # Debug trajectory data
        trajectory = targets["trajectory"].numpy()
        print(f"\nTrajectory Info:")
        print(f"  Shape: {trajectory.shape}")
        print(f"  First 4 waypoints:")
        for wp_i in range(min(4, len(trajectory))):
            print(
                f"    WP{wp_i}: X={trajectory[wp_i, 0]:.3f}m, Y={trajectory[wp_i, 1]:.3f}m, θ={np.degrees(trajectory[wp_i, 2]):.1f}°"
            )

        # Determine save path for this visualization
        current_save_path = args.save_path
        if args.save_path and args.num_vis > 1:
            # Add index to save path for multiple visualizations
            if args.save_path.endswith(".png") or args.save_path.endswith(".jpg"):
                # Remove extension, add index, add extension back
                base_path = args.save_path.rsplit(".", 1)[0]
                extension = args.save_path.rsplit(".", 1)[1]
                current_save_path = f"{base_path}_{i}.{extension}"
            else:
                # No extension, just add index
                current_save_path = f"{args.save_path}_{i}.png"

        # Visualize
        print("\nGenerating visualization...")
        title = f"Direct BEV - Scenario: {scenario}, Token: {token}"

        # Visualize with save path
        visualize_direct_bev(features, targets, title, current_save_path)

        print(f"Visualization {i+1}/{args.num_vis} complete!")

    # Create GIF if multiple visualizations were generated and save_path was specified
    if args.num_vis > 1 and args.save_path:
        # Collect all generated PNG files
        png_paths = []
        for i in range(args.num_vis):
            if args.save_path.endswith((".png", ".jpg")):
                base_path = args.save_path.rsplit(".", 1)[0]
                extension = args.save_path.rsplit(".", 1)[1]
                png_path = f"{base_path}_{i}.{extension}"
            else:
                png_path = f"{args.save_path}_{i}.png"
            png_paths.append(png_path)

        # Create GIF path
        if args.save_path.endswith((".png", ".jpg")):
            gif_path = args.save_path.rsplit(".", 1)[0] + ".gif"
        else:
            gif_path = args.save_path + ".gif"

        # Generate GIF
        print(f"\nCreating animated GIF from {len(png_paths)} visualizations...")
        create_gif_from_pngs(png_paths, gif_path, duration=1000, loop=0)

    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()
