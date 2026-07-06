#!/usr/bin/env python3
"""
Comprehensive visualization tool for trajectory anchors.
Creates multiple plots to analyze anchor quality and distribution.
"""

import os
import sys
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def load_anchors_and_metadata(anchor_path: str):
    """Load anchors and associated metadata."""
    anchors = np.load(anchor_path)

    metadata_path = anchor_path.replace(".npy", "_metadata.json")
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return anchors, metadata


def plot_anchor_overview(anchors: np.ndarray, title_suffix: str = ""):
    """Create overview plot of all anchors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: All anchors overlaid
    colors = plt.cm.rainbow(np.linspace(0, 1, len(anchors)))
    for i, (anchor, color) in enumerate(zip(anchors, colors)):
        ax1.plot(
            anchor[:, 0], anchor[:, 1], "o-", color=color, alpha=0.6, markersize=4, linewidth=2
        )
        ax1.plot(anchor[0, 0], anchor[0, 1], "o", color=color, markersize=8)
        ax1.plot(anchor[-1, 0], anchor[-1, 1], "s", color=color, markersize=8)

    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.set_title(f"All Trajectory Anchors{title_suffix}")
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # Add legend for start/end markers
    start_patch = mpatches.Patch(color="gray", label="○ Start")
    end_patch = mpatches.Patch(color="gray", label="□ End")
    ax1.legend(handles=[start_patch, end_patch], loc="upper right")

    # Right plot: Individual anchors in grid
    n_anchors = len(anchors)
    grid_size = int(np.ceil(np.sqrt(n_anchors)))

    for i, anchor in enumerate(anchors):
        ax2.plot(
            anchor[:, 0] + (i % grid_size) * 100,
            anchor[:, 1] + (i // grid_size) * 100,
            "o-",
            alpha=0.8,
            markersize=3,
        )

    ax2.set_xlabel("X (meters, offset by grid)")
    ax2.set_ylabel("Y (meters, offset by grid)")
    ax2.set_title("Anchors in Grid Layout")
    ax2.grid(True, alpha=0.3)
    ax2.axis("equal")

    plt.tight_layout()
    return fig


def plot_anchor_statistics(anchors: np.ndarray):
    """Plot statistical analysis of anchors."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    # 1. Trajectory length distribution
    lengths = []
    for anchor in anchors:
        length = np.sum(np.linalg.norm(np.diff(anchor, axis=0), axis=1))
        lengths.append(length)

    axes[0].hist(lengths, bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Trajectory Length (meters)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Trajectory Lengths")
    axes[0].axvline(
        np.mean(lengths), color="red", linestyle="--", label=f"Mean: {np.mean(lengths):.1f}m"
    )
    axes[0].legend()

    # 2. Starting position heatmap
    start_positions = anchors[:, 0, :]
    axes[1].hexbin(start_positions[:, 0], start_positions[:, 1], gridsize=20, cmap="YlOrRd")
    axes[1].set_xlabel("X (meters)")
    axes[1].set_ylabel("Y (meters)")
    axes[1].set_title("Starting Position Heatmap")
    axes[1].axis("equal")

    # 3. Ending position heatmap
    end_positions = anchors[:, -1, :]
    axes[2].hexbin(end_positions[:, 0], end_positions[:, 1], gridsize=20, cmap="YlOrRd")
    axes[2].set_xlabel("X (meters)")
    axes[2].set_ylabel("Y (meters)")
    axes[2].set_title("Ending Position Heatmap")
    axes[2].axis("equal")

    # 4. Velocity distribution
    velocities = []
    dt = 0.5  # Assumed time step
    for anchor in anchors:
        vel = np.linalg.norm(np.diff(anchor, axis=0), axis=1) / dt
        velocities.extend(vel)

    axes[3].hist(velocities, bins=50, edgecolor="black", alpha=0.7)
    axes[3].set_xlabel("Velocity (m/s)")
    axes[3].set_ylabel("Count")
    axes[3].set_title("Velocity Distribution")
    axes[3].axvline(
        np.mean(velocities),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(velocities):.1f} m/s",
    )
    axes[3].legend()

    # 5. Curvature analysis
    curvatures = []
    for anchor in anchors:
        if len(anchor) >= 3:
            for i in range(1, len(anchor) - 1):
                v1 = anchor[i] - anchor[i - 1]
                v2 = anchor[i + 1] - anchor[i]
                angle = np.arccos(
                    np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1
                    )
                )
                curvatures.append(angle)

    axes[4].hist(np.degrees(curvatures), bins=50, edgecolor="black", alpha=0.7)
    axes[4].set_xlabel("Turn Angle (degrees)")
    axes[4].set_ylabel("Count")
    axes[4].set_title("Turn Angle Distribution")

    # 6. Anchor similarity matrix
    n_anchors = len(anchors)
    similarity_matrix = np.zeros((n_anchors, n_anchors))
    anchors_flat = anchors.reshape(n_anchors, -1)

    for i in range(n_anchors):
        for j in range(n_anchors):
            similarity_matrix[i, j] = np.linalg.norm(anchors_flat[i] - anchors_flat[j])

    im = axes[5].imshow(similarity_matrix, cmap="viridis")
    axes[5].set_xlabel("Anchor Index")
    axes[5].set_ylabel("Anchor Index")
    axes[5].set_title("Anchor Distance Matrix")
    plt.colorbar(im, ax=axes[5], label="Distance")

    plt.tight_layout()
    return fig


def plot_anchor_clusters(anchors: np.ndarray, metadata: dict = None):
    """Visualize anchors grouped by characteristics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    # Group by direction (forward, left turn, right turn, backward)
    directions = {"forward": [], "left": [], "right": [], "backward": []}

    for i, anchor in enumerate(anchors):
        start_to_end = anchor[-1] - anchor[0]
        angle = np.arctan2(start_to_end[1], start_to_end[0])

        # Determine overall turn direction
        total_turn = 0
        for j in range(1, len(anchor)):
            v1 = anchor[j] - anchor[j - 1]
            if j < len(anchor) - 1:
                v2 = anchor[j + 1] - anchor[j]
                cross = np.cross(v1, v2)
                total_turn += np.sign(cross)

        if abs(angle) < np.pi / 4:  # Mainly forward
            directions["forward"].append(i)
        elif total_turn > 2:  # Left turn
            directions["left"].append(i)
        elif total_turn < -2:  # Right turn
            directions["right"].append(i)
        else:  # Other/backward
            directions["backward"].append(i)

    # Plot each group
    titles = ["Forward Motion", "Left Turns", "Right Turns", "Complex/Other"]
    groups = ["forward", "left", "right", "backward"]

    for ax, title, group in zip(axes, titles, groups):
        indices = directions[group]
        if indices:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(indices)))
            for idx, color in zip(indices, colors):
                anchor = anchors[idx]
                ax.plot(anchor[:, 0], anchor[:, 1], "o-", color=color, alpha=0.6, markersize=4)
                ax.plot(anchor[0, 0], anchor[0, 1], "go", markersize=6)
                ax.plot(anchor[-1, 0], anchor[-1, 1], "ro", markersize=6)

        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title(f"{title} ({len(indices)} anchors)")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    plt.tight_layout()
    return fig


def plot_quality_metrics(metadata: dict):
    """Plot quality metrics if available."""
    if not metadata or "cluster_metrics" not in metadata:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Cluster metrics
    cluster_metrics = metadata["cluster_metrics"]
    metrics_names = ["Silhouette\nScore", "Coverage", "Mean\nDistance"]
    metrics_values = [
        cluster_metrics.get("silhouette_score", 0),
        cluster_metrics.get("coverage", 0),
        cluster_metrics.get("mean_distance", 0) / 10,  # Normalize for display
    ]

    bars1 = ax1.bar(metrics_names, metrics_values, color=["green", "blue", "orange"])
    ax1.set_ylabel("Score")
    ax1.set_title("Clustering Quality Metrics")
    ax1.set_ylim(0, 1)

    # Add value labels on bars
    for bar, value in zip(bars1, metrics_values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{value:.3f}", ha="center", va="bottom"
        )

    # Feasibility checks
    if "feasibility_metrics" in metadata:
        feasibility = metadata["feasibility_metrics"]
        checks = list(feasibility.keys())
        values = [1 if feasibility[k] else 0 for k in checks]

        bars2 = ax2.bar(
            range(len(checks)), values, color=["green" if v else "red" for v in values]
        )
        ax2.set_xticks(range(len(checks)))
        ax2.set_xticklabels([c.replace("_", "\n") for c in checks], rotation=0)
        ax2.set_ylabel("Pass/Fail")
        ax2.set_title("Kinematic Feasibility Checks")
        ax2.set_ylim(0, 1.2)

        # Add pass/fail labels
        for i, (bar, value) in enumerate(zip(bars2, values)):
            label = "PASS" if value else "FAIL"
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                0.5,
                label,
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if value else "black",
            )

    plt.tight_layout()
    return fig


def create_comparison_plot(navsim_path: str, bench2drive_path: str):
    """Create side-by-side comparison of NavSim and Bench2Drive anchors."""
    navsim_anchors, _ = load_anchors_and_metadata(navsim_path)
    b2d_anchors, _ = load_anchors_and_metadata(bench2drive_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # NavSim anchors
    for anchor in navsim_anchors:
        ax1.plot(anchor[:, 0], anchor[:, 1], "o-", alpha=0.5, markersize=3)
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.set_title("NavSim Trajectory Anchors")
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # Add statistics
    x_range = (navsim_anchors[:, :, 0].min(), navsim_anchors[:, :, 0].max())
    y_range = (navsim_anchors[:, :, 1].min(), navsim_anchors[:, :, 1].max())
    ax1.text(
        0.02,
        0.98,
        f"X range: [{x_range[0]:.1f}, {x_range[1]:.1f}]\n"
        f"Y range: [{y_range[0]:.1f}, {y_range[1]:.1f}]",
        transform=ax1.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    # Bench2Drive anchors
    for anchor in b2d_anchors:
        ax2.plot(anchor[:, 0], anchor[:, 1], "o-", alpha=0.5, markersize=3)
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters)")
    ax2.set_title("Bench2Drive Trajectory Anchors")
    ax2.grid(True, alpha=0.3)
    ax2.axis("equal")

    # Add statistics
    x_range = (b2d_anchors[:, :, 0].min(), b2d_anchors[:, :, 0].max())
    y_range = (b2d_anchors[:, :, 1].min(), b2d_anchors[:, :, 1].max())
    ax2.text(
        0.02,
        0.98,
        f"X range: [{x_range[0]:.1f}, {x_range[1]:.1f}]\n"
        f"Y range: [{y_range[0]:.1f}, {y_range[1]:.1f}]",
        transform=ax2.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightblue"),
    )

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize trajectory anchors")
    parser.add_argument(
        "--anchor-path", type=str, required=True, help="Path to anchor file (.npy)"
    )
    parser.add_argument(
        "--compare-with", type=str, help="Path to another anchor file for comparison"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Directory to save plots (default: same as anchor file)"
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")

    args = parser.parse_args()

    # Load anchors
    anchors, metadata = load_anchors_and_metadata(args.anchor_path)
    print(f"Loaded {len(anchors)} anchors from {args.anchor_path}")

    if metadata:
        print(f"Dataset type: {metadata.get('dataset_type', 'unknown')}")
        if "cluster_metrics" in metadata:
            print(f"Silhouette score: {metadata['cluster_metrics']['silhouette_score']:.3f}")
            print(f"Coverage: {metadata['cluster_metrics']['coverage']:.1%}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.anchor_path).parent
    output_dir.mkdir(exist_ok=True)

    # Create base name for outputs
    base_name = Path(args.anchor_path).stem

    # Generate plots
    plots = []

    # 1. Overview plot
    fig = plot_anchor_overview(anchors, f" ({base_name})")
    plots.append((fig, f"{base_name}_overview.png"))

    # 2. Statistics plot
    fig = plot_anchor_statistics(anchors)
    plots.append((fig, f"{base_name}_statistics.png"))

    # 3. Cluster plot
    fig = plot_anchor_clusters(anchors, metadata)
    plots.append((fig, f"{base_name}_clusters.png"))

    # 4. Quality metrics plot
    if metadata:
        fig = plot_quality_metrics(metadata)
        if fig:
            plots.append((fig, f"{base_name}_quality.png"))

    # 5. Comparison plot if requested
    if args.compare_with:
        fig = create_comparison_plot(args.anchor_path, args.compare_with)
        plots.append((fig, "anchor_comparison.png"))

    # Save or show plots
    for fig, filename in plots:
        if args.show:
            plt.show()
        else:
            save_path = output_dir / filename
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
            plt.close(fig)


if __name__ == "__main__":
    main()
