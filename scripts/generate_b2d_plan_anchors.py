#!/usr/bin/env python3
"""
Generate k-means plan anchors from Bench2Drive dataset.
This creates trajectory anchors specific to Bench2Drive data distribution.
"""

# Fix OpenBLAS threading issue on high-core machines
import os
os.environ['OPENBLAS_NUM_THREADS'] = '64'
os.environ['GOTO_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64'
os.environ['MKL_NUM_THREADS'] = '64'

from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder,
)
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import json


def collect_trajectories(cache_path: str, num_samples: int = 5000):
    """
    Collect trajectory data from cached Bench2Drive dataset.

    Args:
        cache_path: Path to the cached dataset
        num_samples: Number of samples to collect

    Returns:
        numpy array of trajectories with shape (N, 8, 2) - only x,y coordinates
    """
    print(f"Collecting trajectories from: {cache_path}")

    # Create feature/target builders
    config = TransfuserConfig()
    feature_builder = Bench2DriveFeatureBuilder(config)
    target_builder = Bench2DriveTargetBuilder(config)

    # Create dataset
    dataset = CacheOnlyDataset(
        cache_path=cache_path,
        feature_builders=[feature_builder],
        target_builders=[target_builder],
    )

    print(f"Dataset size: {len(dataset)}")
    if num_samples == -1:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    # Collect trajectories
    trajectories = []

    print(f"\nCollecting {num_samples} trajectories...")
    for i in tqdm(range(num_samples)):
        try:
            features, targets = dataset[i]
            trajectory = targets["trajectory"]  # Shape: [8, 3] - (x, y, heading)

            # Extract only x, y coordinates
            xy_trajectory = trajectory[:, :2].numpy()  # Shape: [8, 2]
            trajectories.append(xy_trajectory)
            
            # Also collect heading for normalization statistics
            if i == 0:
                all_headings = [trajectory[:, 2].numpy()]  # Shape: [8]
            else:
                all_headings.append(trajectory[:, 2].numpy())

        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue

    trajectories = np.array(trajectories)  # Shape: (N, 8, 2)
    all_headings = np.array(all_headings)  # Shape: (N, 8)
    print(f"\nCollected {len(trajectories)} valid trajectories")

    # Compute normalization statistics
    x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
    y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max() 
    h_min, h_max = all_headings.min(), all_headings.max()
    
    normalization_params = {
        "x_offset": float(-x_min),
        "x_scale": float(x_max - x_min),
        "y_offset": float(-y_min), 
        "y_scale": float(y_max - y_min),
        "heading_offset": float(-h_min),
        "heading_scale": float(h_max - h_min),
        "raw_ranges": {
            "x_range": [float(x_min), float(x_max)],
            "y_range": [float(y_min), float(y_max)],
            "heading_range": [float(h_min), float(h_max)]
        }
    }
    
    print(f"\nNormalization parameters computed:")
    print(f"  X: range=[{x_min:.3f}, {x_max:.3f}] -> offset={normalization_params['x_offset']:.3f}, scale={normalization_params['x_scale']:.3f}")
    print(f"  Y: range=[{y_min:.3f}, {y_max:.3f}] -> offset={normalization_params['y_offset']:.3f}, scale={normalization_params['y_scale']:.3f}")
    print(f"  Heading: range=[{h_min:.3f}, {h_max:.3f}] -> offset={normalization_params['heading_offset']:.3f}, scale={normalization_params['heading_scale']:.3f}")

    return trajectories, normalization_params


def generate_kmeans_anchors(trajectories: np.ndarray, num_clusters: int = 20):
    """
    Generate k-means anchors from collected trajectories.

    Args:
        trajectories: numpy array of shape (N, 8, 2)
        num_clusters: number of k-means clusters

    Returns:
        k-means cluster centers of shape (num_clusters, 8, 2)
    """
    print(f"\nGenerating {num_clusters} k-means clusters...")

    # Reshape trajectories for k-means: (N, 8*2) = (N, 16)
    N, timesteps, coords = trajectories.shape
    trajectories_flat = trajectories.reshape(N, -1)

    # Print statistics
    print("Trajectory statistics before clustering:")
    print(f"  Shape: {trajectories.shape}")
    print(f"  X range: [{trajectories[:, :, 0].min():.3f}, {trajectories[:, :, 0].max():.3f}]")
    print(f"  Y range: [{trajectories[:, :, 1].min():.3f}, {trajectories[:, :, 1].max():.3f}]")

    # Run k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, verbose=1)
    kmeans.fit(trajectories_flat)

    # Get cluster centers and reshape back
    anchors = kmeans.cluster_centers_.reshape(num_clusters, timesteps, coords)

    # Print anchor statistics
    print("\nGenerated anchor statistics:")
    print(f"  Shape: {anchors.shape}")
    print(f"  X range: [{anchors[:, :, 0].min():.3f}, {anchors[:, :, 0].max():.3f}]")
    print(f"  Y range: [{anchors[:, :, 1].min():.3f}, {anchors[:, :, 1].max():.3f}]")

    # Calculate average distance between consecutive waypoints
    distances = []
    for i in range(anchors.shape[0]):
        for j in range(anchors.shape[1] - 1):
            dist = np.linalg.norm(anchors[i, j + 1] - anchors[i, j])
            distances.append(dist)

    print(f"  Average waypoint distance: {np.mean(distances):.3f}")
    print(f"  Distance range: [{np.min(distances):.3f}, {np.max(distances):.3f}]")

    # Verify anchors cover the data distribution
    assigned_clusters = kmeans.labels_
    for i in range(num_clusters):
        cluster_count = np.sum(assigned_clusters == i)
        print(f"  Cluster {i}: {cluster_count} trajectories ({100*cluster_count/N:.1f}%)")

    return anchors


def compare_with_navsim_anchors(b2d_anchors: np.ndarray, navsim_anchor_path: str):
    """
    Compare Bench2Drive anchors with NavSim anchors.

    Args:
        b2d_anchors: Bench2Drive anchors
        navsim_anchor_path: Path to NavSim anchors
    """
    if os.path.exists(navsim_anchor_path):
        navsim_anchors = np.load(navsim_anchor_path)

        print("\n" + "=" * 60)
        print("COMPARISON WITH NAVSIM ANCHORS")
        print("=" * 60)

        print("\nNavSim anchors:")
        print(f"  Shape: {navsim_anchors.shape}")
        print(
            f"  X range: [{navsim_anchors[:, :, 0].min():.3f}, {navsim_anchors[:, :, 0].max():.3f}]"
        )
        print(
            f"  Y range: [{navsim_anchors[:, :, 1].min():.3f}, {navsim_anchors[:, :, 1].max():.3f}]"
        )

        print("\nBench2Drive anchors:")
        print(f"  Shape: {b2d_anchors.shape}")
        print(f"  X range: [{b2d_anchors[:, :, 0].min():.3f}, {b2d_anchors[:, :, 0].max():.3f}]")
        print(f"  Y range: [{b2d_anchors[:, :, 1].min():.3f}, {b2d_anchors[:, :, 1].max():.3f}]")

        # Calculate scale differences
        navsim_x_range = navsim_anchors[:, :, 0].max() - navsim_anchors[:, :, 0].min()
        navsim_y_range = navsim_anchors[:, :, 1].max() - navsim_anchors[:, :, 1].min()
        b2d_x_range = b2d_anchors[:, :, 0].max() - b2d_anchors[:, :, 0].min()
        b2d_y_range = b2d_anchors[:, :, 1].max() - b2d_anchors[:, :, 1].min()

        print("\nScale comparison:")
        print(f"  X scale ratio (B2D/NavSim): {b2d_x_range/navsim_x_range:.2f}")
        print(f"  Y scale ratio (B2D/NavSim): {b2d_y_range/navsim_y_range:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Generate k-means plan anchors for Bench2Drive")
    parser.add_argument("--cache-path", type=str, required=True, help="Path to the cached dataset")
    parser.add_argument(
        "--output-path",
        type=str,
        default="./download/kmeans_bench2drive_traj_20.npy",
        help="Path to save the generated anchors",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to use for k-means. If -1, use all samples (default: -1)",
    )
    parser.add_argument(
        "--num-clusters", type=int, default=20, help="Number of k-means clusters (default: 20)"
    )
    parser.add_argument(
        "--navsim-anchor-path",
        type=str,
        default="./download/kmeans_navsim_traj_20.npy",
        help="Path to NavSim anchors for comparison",
    )

    args = parser.parse_args()

    if not os.path.exists(args.cache_path):
        print(f"Error: Cache path does not exist: {args.cache_path}")
        sys.exit(1)

    # Collect trajectories
    trajectories, normalization_params = collect_trajectories(args.cache_path, args.num_samples)

    # Generate k-means anchors
    anchors = generate_kmeans_anchors(trajectories, args.num_clusters)

    # Save anchors
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, anchors)
    print(f"\nAnchors saved to: {args.output_path}")

    # Compare with NavSim anchors
    compare_with_navsim_anchors(anchors, args.navsim_anchor_path)

    # Save metadata
    metadata = {
        "num_samples_used": len(trajectories),
        "num_clusters": args.num_clusters,
        "cache_path": args.cache_path,
        "anchor_shape": list(anchors.shape),
        "anchor_ranges": {
            "x_range": [float(anchors[:, :, 0].min()), float(anchors[:, :, 0].max())],
            "y_range": [float(anchors[:, :, 1].min()), float(anchors[:, :, 1].max())],
        },
        "normalization_params": normalization_params
    }

    metadata_path = args.output_path.replace(".npy", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
