#!/usr/bin/env python3
"""
Generate k-means plan anchors and trajectory normalization stats from TaCarla cache.
Adapted from generate_b2d_plan_anchors.py.

Usage (inside Docker):
    python3 scripts/generate_tacarla_plan_anchors.py \
        --cache-path /mnt/nas/.../TaCarla_cache_test \
        --output-path ./download/kmeans_tacarla_traj_20.npy \
        --num-samples -1
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '64'
os.environ['GOTO_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64'
os.environ['MKL_NUM_THREADS'] = '64'

import argparse
import json
import sys

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from navsim.agents.diffusiondrive.tacarla_config import TaCarlaConfig
from navsim.agents.diffusiondrive.transfuser_features_tacarla import (
    TaCarlaFeatureBuilder,
    TaCarlaTargetBuilder,
)
from navsim.planning.training.dataset import CacheOnlyDataset


def collect_trajectories(cache_path: str, num_samples: int = -1):
    """Collect trajectory data from cached TaCarla dataset."""
    print(f"Collecting trajectories from: {cache_path}")

    config = TaCarlaConfig()
    feature_builder = TaCarlaFeatureBuilder(config)
    target_builder = TaCarlaTargetBuilder(config)

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

    trajectories = []
    all_headings = []

    print(f"\nCollecting {num_samples} trajectories...")
    for i in tqdm(range(num_samples)):
        try:
            features, targets = dataset[i]
            trajectory = targets["trajectory"]  # [8, 3]
            trajectories.append(trajectory[:, :2].numpy())
            all_headings.append(trajectory[:, 2].numpy())
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue

    trajectories = np.array(trajectories)
    all_headings = np.array(all_headings)
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
            "heading_range": [float(h_min), float(h_max)],
        },
    }

    print(f"\nNormalization parameters:")
    print(f"  X: [{x_min:.3f}, {x_max:.3f}] -> offset={-x_min:.3f}, scale={x_max-x_min:.3f}")
    print(f"  Y: [{y_min:.3f}, {y_max:.3f}] -> offset={-y_min:.3f}, scale={y_max-y_min:.3f}")
    print(f"  H: [{h_min:.3f}, {h_max:.3f}] -> offset={-h_min:.3f}, scale={h_max-h_min:.3f}")

    return trajectories, normalization_params


def generate_kmeans_anchors(trajectories: np.ndarray, num_clusters: int = 20):
    """Generate k-means anchors from trajectories."""
    print(f"\nGenerating {num_clusters} k-means clusters...")

    N, timesteps, coords = trajectories.shape
    trajectories_flat = trajectories.reshape(N, -1)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, verbose=1)
    kmeans.fit(trajectories_flat)

    anchors = kmeans.cluster_centers_.reshape(num_clusters, timesteps, coords)

    print(f"\nAnchor stats:")
    print(f"  Shape: {anchors.shape}")
    print(f"  X: [{anchors[:,:,0].min():.3f}, {anchors[:,:,0].max():.3f}]")
    print(f"  Y: [{anchors[:,:,1].min():.3f}, {anchors[:,:,1].max():.3f}]")

    # Cluster distribution
    for i in range(num_clusters):
        count = np.sum(kmeans.labels_ == i)
        print(f"  Cluster {i}: {count} ({100*count/N:.1f}%)")

    return anchors


def main():
    parser = argparse.ArgumentParser(description="Generate k-means plan anchors for TaCarla")
    parser.add_argument("--cache-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="./download/kmeans_tacarla_traj_20.npy")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--num-clusters", type=int, default=20)
    parser.add_argument("--navsim-anchor-path", type=str, default="./download/kmeans_navsim_traj_20.npy")
    args = parser.parse_args()

    if not os.path.exists(args.cache_path):
        print(f"Error: Cache path does not exist: {args.cache_path}")
        sys.exit(1)

    trajectories, normalization_params = collect_trajectories(args.cache_path, args.num_samples)
    anchors = generate_kmeans_anchors(trajectories, args.num_clusters)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    np.save(args.output_path, anchors)
    print(f"\nAnchors saved to: {args.output_path}")

    # Compare with NAVSIM anchors if available
    if os.path.exists(args.navsim_anchor_path):
        navsim = np.load(args.navsim_anchor_path)
        print(f"\nNAVSIM anchors: shape={navsim.shape}")
        print(f"  X: [{navsim[:,:,0].min():.3f}, {navsim[:,:,0].max():.3f}]")
        print(f"  Y: [{navsim[:,:,1].min():.3f}, {navsim[:,:,1].max():.3f}]")

    # Save metadata
    metadata = {
        "num_samples_used": len(trajectories),
        "num_clusters": args.num_clusters,
        "cache_path": args.cache_path,
        "anchor_shape": list(anchors.shape),
        "normalization_params": normalization_params,
    }
    metadata_path = args.output_path.replace(".npy", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    # Print config snippet for tacarla_config.py
    p = normalization_params
    print(f"\n--- Update tacarla_config.py with these values ---")
    print(f"tc_traj_norm_x_offset: float = {p['x_offset']}")
    print(f"tc_traj_norm_x_scale: float = {p['x_scale']}")
    print(f"tc_traj_norm_y_offset: float = {p['y_offset']}")
    print(f"tc_traj_norm_y_scale: float = {p['y_scale']}")
    print(f"tc_traj_norm_heading_offset: float = {p['heading_offset']}")
    print(f"tc_traj_norm_heading_scale: float = {p['heading_scale']}")


if __name__ == "__main__":
    main()
