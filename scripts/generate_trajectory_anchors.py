#!/usr/bin/env python3
"""
Enhanced trajectory anchor generation with quality metrics.
Supports both NavSim and Bench2Drive datasets with automatic optimization.
"""


from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features import (
    TransfuserFeatureBuilder,
    TransfuserTargetBuilder,
)
from navsim.agents.diffusiondrive.trajectory_normalizer import TrajectoryNormalizer

import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, Tuple, Optional


class AnchorQualityAnalyzer:
    """Analyzes quality of trajectory anchors."""

    def __init__(self):
        self.metrics = {}

    def compute_cluster_metrics(
        self, trajectories: np.ndarray, kmeans: KMeans
    ) -> Dict[str, float]:
        """
        Compute clustering quality metrics.

        Args:
            trajectories: Shape (N, features) flattened trajectories
            kmeans: Fitted KMeans model

        Returns:
            Dictionary of quality metrics
        """
        labels = kmeans.labels_

        # Silhouette score: measures how similar an object is to its own cluster
        # compared to other clusters. Range: [-1, 1], higher is better
        silhouette = silhouette_score(trajectories, labels)

        # Davies-Bouldin score: ratio of within-cluster to between-cluster distances
        # Lower is better
        davies_bouldin = davies_bouldin_score(trajectories, labels)

        # Inertia: sum of squared distances to nearest cluster center
        inertia = kmeans.inertia_

        # Coverage: percentage of trajectories well-represented by their cluster
        distances = kmeans.transform(trajectories).min(axis=1)
        coverage_threshold = np.percentile(distances, 95)
        coverage = np.sum(distances < coverage_threshold) / len(distances)

        return {
            "silhouette_score": float(silhouette),
            "davies_bouldin_score": float(davies_bouldin),
            "inertia": float(inertia),
            "coverage": float(coverage),
            "mean_distance": float(distances.mean()),
            "std_distance": float(distances.std()),
        }

    def analyze_anchor_diversity(self, anchors: np.ndarray) -> Dict[str, float]:
        """
        Analyze diversity of anchor trajectories.

        Args:
            anchors: Shape (K, T, 2) anchor trajectories

        Returns:
            Dictionary of diversity metrics
        """
        K, T, _ = anchors.shape

        # Pairwise distances between anchors
        anchor_flat = anchors.reshape(K, -1)
        pairwise_dists = np.zeros((K, K))
        for i in range(K):
            for j in range(i + 1, K):
                dist = np.linalg.norm(anchor_flat[i] - anchor_flat[j])
                pairwise_dists[i, j] = dist
                pairwise_dists[j, i] = dist

        # Minimum distance between any two anchors
        min_dist = pairwise_dists[pairwise_dists > 0].min()

        # Average distance between anchors
        avg_dist = pairwise_dists[pairwise_dists > 0].mean()

        # Trajectory length statistics
        lengths = np.zeros(K)
        for i in range(K):
            traj = anchors[i]
            length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
            lengths[i] = length

        return {
            "min_anchor_distance": float(min_dist),
            "avg_anchor_distance": float(avg_dist),
            "min_trajectory_length": float(lengths.min()),
            "max_trajectory_length": float(lengths.max()),
            "avg_trajectory_length": float(lengths.mean()),
            "trajectory_length_std": float(lengths.std()),
        }

    def check_kinematic_feasibility(self, anchors: np.ndarray, dt: float = 0.5) -> Dict[str, bool]:
        """
        Check if anchors follow kinematic constraints.

        Args:
            anchors: Shape (K, T, 2) anchor trajectories
            dt: Time step between waypoints

        Returns:
            Dictionary of feasibility checks
        """
        K, T, _ = anchors.shape

        # Maximum reasonable velocity and acceleration for vehicles
        max_velocity = 30.0  # m/s (108 km/h)
        max_acceleration = 10.0  # m/s^2

        feasible_velocity = True
        feasible_acceleration = True
        smooth_trajectories = True

        for i in range(K):
            traj = anchors[i]

            # Compute velocities
            velocities = np.diff(traj, axis=0) / dt
            speeds = np.linalg.norm(velocities, axis=1)

            # Check velocity constraints
            if np.any(speeds > max_velocity):
                feasible_velocity = False

            # Compute accelerations
            if len(velocities) > 1:
                accelerations = np.diff(velocities, axis=0) / dt
                accel_magnitudes = np.linalg.norm(accelerations, axis=1)

                # Check acceleration constraints
                if np.any(accel_magnitudes > max_acceleration):
                    feasible_acceleration = False

                # Check smoothness (sudden direction changes)
                for j in range(len(velocities) - 1):
                    angle_change = np.arccos(
                        np.clip(
                            np.dot(velocities[j], velocities[j + 1])
                            / (
                                np.linalg.norm(velocities[j]) * np.linalg.norm(velocities[j + 1])
                                + 1e-6
                            ),
                            -1,
                            1,
                        )
                    )
                    if angle_change > np.pi / 2:  # 90 degree turn
                        smooth_trajectories = False

        return {
            "feasible_velocity": feasible_velocity,
            "feasible_acceleration": feasible_acceleration,
            "smooth_trajectories": smooth_trajectories,
        }


def find_optimal_clusters(
    trajectories: np.ndarray, min_k: int = 10, max_k: int = 30, sample_size: int = 5000
) -> Tuple[int, Dict[int, Dict]]:
    """
    Find optimal number of clusters using elbow method and quality metrics.

    Args:
        trajectories: Flattened trajectories
        min_k: Minimum number of clusters
        max_k: Maximum number of clusters
        sample_size: Number of samples to use for evaluation

    Returns:
        Optimal k and metrics for each k
    """
    print(f"\nFinding optimal number of clusters (k={min_k} to {max_k})...")

    # Sample trajectories if too many
    if len(trajectories) > sample_size:
        idx = np.random.choice(len(trajectories), sample_size, replace=False)
        sample_traj = trajectories[idx]
    else:
        sample_traj = trajectories

    analyzer = AnchorQualityAnalyzer()
    all_metrics = {}

    for k in tqdm(range(min_k, max_k + 1)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(sample_traj)

        metrics = analyzer.compute_cluster_metrics(sample_traj, kmeans)
        metrics["k"] = k
        all_metrics[k] = metrics

    # Find elbow point using second derivative of inertia
    inertias = [all_metrics[k]["inertia"] for k in range(min_k, max_k + 1)]

    # Compute second derivatives
    first_deriv = np.diff(inertias)
    second_deriv = np.diff(first_deriv)

    # Find elbow (maximum second derivative)
    elbow_idx = np.argmax(second_deriv) + 1  # +1 because of diff
    optimal_k = min_k + elbow_idx

    # Also consider silhouette score
    silhouettes = [all_metrics[k]["silhouette_score"] for k in range(min_k, max_k + 1)]
    best_silhouette_k = min_k + np.argmax(silhouettes)

    # Use weighted decision
    if abs(optimal_k - best_silhouette_k) <= 2:
        optimal_k = best_silhouette_k

    print(f"\nOptimal number of clusters: {optimal_k}")
    print(f"  Elbow method: {min_k + elbow_idx}")
    print(f"  Best silhouette: {best_silhouette_k}")

    return optimal_k, all_metrics


def generate_quality_anchors(
    cache_path: str,
    num_samples: int = -1,
    k: Optional[int] = None,
    auto_optimize: bool = False,
    dataset_type: str = "auto",
) -> Tuple[np.ndarray, Dict]:
    """
    Generate high-quality trajectory anchors with metrics.

    Args:
        cache_path: Path to cached dataset
        num_samples: Number of samples to use
        k: Number of clusters (None for auto)
        auto_optimize: Whether to automatically find optimal k
        dataset_type: Dataset type for normalization

    Returns:
        Anchors and quality metrics
    """
    print(f"Generating trajectory anchors from: {cache_path}")

    # Create feature/target builders
    config = TransfuserConfig()
    feature_builder = TransfuserFeatureBuilder(config)
    target_builder = TransfuserTargetBuilder(config)

    # Create dataset
    dataset = CacheOnlyDataset(
        cache_path=cache_path,
        feature_builders=[feature_builder],
        target_builders=[target_builder],
    )

    print(f"Dataset size: {len(dataset)}")
    if num_samples == -1:
        num_samples = len(dataset)
        print("Using all available samples")
    else:
        num_samples = min(num_samples, len(dataset))

    # Initialize normalizer
    normalizer = TrajectoryNormalizer(dataset_type=dataset_type)
    if dataset_type == "auto":
        detected_type = normalizer.detect_dataset_from_path(cache_path)
        normalizer = TrajectoryNormalizer(dataset_type=detected_type)
        print(f"Detected dataset type: {detected_type}")

    # Collect trajectories
    trajectories = []
    print(f"\nCollecting {num_samples} trajectories...")

    for i in tqdm(range(num_samples)):
        try:
            features, targets = dataset[i]
            trajectory = targets["trajectory"]  # Shape: [8, 3]

            # Only use x, y coordinates
            xy_trajectory = trajectory[:, :2].numpy()
            trajectories.append(xy_trajectory)

        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue

    trajectories = np.array(trajectories)  # Shape: (N, 8, 2)
    print(f"Collected {len(trajectories)} valid trajectories")

    # Filter outliers using percentiles
    print("\nFiltering outliers...")
    traj_flat = trajectories.reshape(-1, 2)
    x_low, x_high = np.percentile(traj_flat[:, 0], [1, 99])
    y_low, y_high = np.percentile(traj_flat[:, 1], [1, 99])

    valid_mask = np.all(
        [
            trajectories[:, :, 0].min(axis=1) >= x_low,
            trajectories[:, :, 0].max(axis=1) <= x_high,
            trajectories[:, :, 1].min(axis=1) >= y_low,
            trajectories[:, :, 1].max(axis=1) <= y_high,
        ],
        axis=0,
    )

    trajectories_filtered = trajectories[valid_mask]
    print(f"Kept {len(trajectories_filtered)} trajectories after filtering")

    # Flatten for clustering
    trajectories_flat = trajectories_filtered.reshape(len(trajectories_filtered), -1)

    # Find optimal k if auto_optimize is requested
    if auto_optimize:
        optimal_k, k_metrics = find_optimal_clusters(trajectories_flat)
        k = optimal_k
    elif k is None:
        k = 20  # Default when neither specified nor auto-optimized

    # Generate final anchors
    print(f"\nGenerating {k} anchor trajectories...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(trajectories_flat)

    # Get anchors
    anchors = kmeans.cluster_centers_.reshape(k, 8, 2)

    # Compute quality metrics
    analyzer = AnchorQualityAnalyzer()

    print("\nComputing quality metrics...")
    cluster_metrics = analyzer.compute_cluster_metrics(trajectories_flat, kmeans)
    diversity_metrics = analyzer.analyze_anchor_diversity(anchors)
    feasibility_metrics = analyzer.check_kinematic_feasibility(anchors)

    # Combine all metrics
    quality_metrics = {
        "num_anchors": k,
        "num_trajectories": len(trajectories_filtered),
        "cluster_metrics": cluster_metrics,
        "diversity_metrics": diversity_metrics,
        "feasibility_metrics": feasibility_metrics,
        "dataset_type": normalizer.dataset_type,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("ANCHOR GENERATION SUMMARY")
    print("=" * 60)
    print(f"Number of anchors: {k}")
    print(f"Silhouette score: {cluster_metrics['silhouette_score']:.3f}")
    print(f"Coverage: {cluster_metrics['coverage']:.1%}")
    print(f"Average anchor distance: {diversity_metrics['avg_anchor_distance']:.2f}")
    print(f"Kinematic feasibility: {all(feasibility_metrics.values())}")

    return anchors, quality_metrics


def visualize_anchors(anchors: np.ndarray, save_path: Optional[str] = None):
    """Visualize anchor trajectories."""
    plt.figure(figsize=(12, 8))

    # Plot all anchors
    for i, anchor in enumerate(anchors):
        plt.plot(anchor[:, 0], anchor[:, 1], "o-", alpha=0.6, label=f"Anchor {i}" if i < 5 else "")
        plt.plot(anchor[0, 0], anchor[0, 1], "go", markersize=8)  # Start
        plt.plot(anchor[-1, 0], anchor[-1, 1], "ro", markersize=8)  # End

    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title(f"Trajectory Anchors ({len(anchors)} clusters)")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    if len(anchors) <= 5:
        plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate trajectory anchors with quality metrics"
    )
    parser.add_argument("--cache-path", type=str, required=True, help="Path to the cached dataset")
    parser.add_argument(
        "--output-path", type=str, help="Output path for anchors (default: auto-generated)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to use (default: -1, use all samples)",
    )
    parser.add_argument(
        "--num-clusters", type=int, help="Number of clusters (default: auto-detect)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="auto",
        choices=["navsim", "bench2drive", "auto"],
        help="Dataset type (default: auto)",
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize the generated anchors")
    parser.add_argument(
        "--auto-optimize",
        action="store_true",
        help="Enable automatic k optimization (will override --num-clusters if specified)",
    )

    args = parser.parse_args()

    # Handle conflicting arguments
    if args.auto_optimize and args.num_clusters:
        print("\nWarning: --num-clusters will be ignored when --auto-optimize is enabled")

    # Generate anchors
    anchors, metrics = generate_quality_anchors(
        cache_path=args.cache_path,
        num_samples=args.num_samples,
        k=args.num_clusters,
        auto_optimize=args.auto_optimize,
        dataset_type=args.dataset_type,
    )

    # Determine output path
    if args.output_path:
        output_path = args.output_path
    else:
        cache_name = Path(args.cache_path).name
        dataset_type = metrics["dataset_type"]
        k = metrics["num_anchors"]
        output_path = f"./download/kmeans_{dataset_type}_traj_{k}.npy"

    # Save anchors
    np.save(output_path, anchors)
    print(f"\nSaved anchors to: {output_path}")

    # Save metadata
    metadata_path = output_path.replace(".npy", "_metadata.json")
    with open(metadata_path, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        json.dump(convert_types(metrics), f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    # Visualize if requested
    if args.visualize:
        viz_path = output_path.replace(".npy", "_visualization.png")
        visualize_anchors(anchors, viz_path)


if __name__ == "__main__":
    main()
