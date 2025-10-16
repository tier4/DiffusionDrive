#!/usr/bin/env python3
"""
Generate 15 sample comparisons with RGB top-down views.
Uses the fixed segmentation V2 generator with correct ego heading.
"""

import sys
import os
sys.path.append('/workspace/DiffusionDrive')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gzip
import json
from PIL import Image
from datetime import datetime

from navsim.common.bev_generation_factory import BEVGeneratorFactory
from navsim.common.bev_segmentation_generator import SegmentationBEVGenerator
from navsim.common.bev_map_utils import MapProcessor, load_map_data, normalize_angle


def load_ego_data_correct(scenario_path, frame_number):
    """Load ego data CORRECTLY from bounding boxes, not theta."""
    anno_file = scenario_path / "anno" / f"{frame_number}.json.gz"

    if not anno_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")

    with gzip.open(anno_file, 'rt') as f:
        data = json.load(f)

    # Get position from x, y fields
    ego_x = data['x']
    ego_y = data['y']
    ego_points = np.array([ego_x, ego_y, 0.0])

    # Find ego vehicle in bounding boxes - CORRECT METHOD
    ego_box = None
    bounding_boxes = data['bounding_boxes']
    for box in bounding_boxes:
        if box['class'] == 'ego_vehicle':
            ego_box = box
            break

    if ego_box is None:
        raise ValueError(f"Ego vehicle not found in bounding boxes for frame {frame_number}")

    # Get heading from rotation[2] (z-axis rotation in degrees)
    ego_yaw_degrees = ego_box['rotation'][2]
    ego_heading_rad = normalize_angle(np.radians(ego_yaw_degrees))

    return ego_points, ego_heading_rad, ego_yaw_degrees


def load_rgb_topdown(scenario_path, frame_number):
    """Load RGB top-down image."""
    rgb_file = scenario_path / "camera" / "rgb_top_down" / f"{frame_number}.jpg"

    if not rgb_file.exists():
        return None

    img = Image.open(rgb_file)
    return np.array(img)


def generate_comparison(scenario_path, frame_number, map_processor):
    """Generate vector and segmentation BEV maps for comparison."""

    # Load ego data with CORRECT heading
    ego_points, ego_heading_rad, ego_yaw_degrees = load_ego_data_correct(scenario_path, frame_number)

    # Load RGB top-down view
    rgb_topdown = load_rgb_topdown(scenario_path, frame_number)

    # Create generators
    vector_gen = BEVGeneratorFactory.create_generator(
        generation_type="vector",
        bev_height=256,  # Use full view for better comparison
        bev_width=256,
        resolution=0.332,
        view_type="full"
    )

    seg_gen = SegmentationBEVGenerator(
        bev_height=256,  # Use full view
        bev_width=256,
        resolution=0.332,
        view_type="full"
    )

    # Generate BEV maps
    vector_bev = vector_gen.generate_from_map(
        map_data=map_processor,
        ego_points=ego_points,
        ego_heading_rad=ego_heading_rad
    )

    seg_bev = seg_gen.generate_from_map(
        map_data=map_processor,
        ego_points=ego_points,
        ego_heading_rad=ego_heading_rad
    )

    return {
        'vector_bev': vector_bev,
        'seg_bev': seg_bev,
        'rgb_topdown': rgb_topdown,
        'ego_points': ego_points,
        'ego_heading_rad': ego_heading_rad,
        'ego_yaw_degrees': ego_yaw_degrees
    }


def visualize_samples(results, output_path):
    """Create visualization grid with all samples."""
    n_samples = len(results)
    n_cols = 4  # RGB, Vector, Segmentation, Overlay
    n_rows = n_samples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Color map for BEV semantic classes
    colors = np.array([
        [0, 0, 0],      # 0: background (black)
        [128, 0, 128],  # 1: road (purple)
        [255, 255, 0],  # 2: walkway (yellow) - NOT GENERATED
        [0, 255, 0],    # 3: centerline (green)
        [255, 0, 0],    # 4: static (red)
        [0, 0, 255],    # 5: vehicle (blue)
        [255, 128, 0],  # 6: pedestrian (orange)
    ], dtype=np.uint8)

    for idx, (scenario_name, frame_number, result) in enumerate(results):
        # RGB top-down
        if result['rgb_topdown'] is not None:
            axes[idx, 0].imshow(result['rgb_topdown'])
            axes[idx, 0].set_title(f"RGB Top-down\n{scenario_name[:20]}/F{frame_number}")
        else:
            axes[idx, 0].text(0.5, 0.5, "No RGB", ha='center', va='center')
            axes[idx, 0].set_title("RGB Top-down (N/A)")
        axes[idx, 0].axis('off')

        # Vector BEV
        vector_rgb = colors[result['vector_bev'].astype(int)]
        axes[idx, 1].imshow(vector_rgb)
        non_zero = np.sum(result['vector_bev'] > 0)
        axes[idx, 1].set_title(f"Vector BEV\n{non_zero} pixels")
        axes[idx, 1].axis('off')

        # Segmentation BEV
        seg_rgb = colors[result['seg_bev'].astype(int)]
        axes[idx, 2].imshow(seg_rgb)
        non_zero = np.sum(result['seg_bev'] > 0)
        axes[idx, 2].set_title(f"Segmentation V2\n{non_zero} pixels")
        axes[idx, 2].axis('off')

        # Overlay comparison
        overlay = seg_rgb.copy()
        vector_mask = result['vector_bev'] > 0
        # Add white outline for vector
        for c in range(3):
            overlay[:, :, c] = np.where(vector_mask, 255, overlay[:, :, c])
        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title(f"Overlay\nEgo heading: {result['ego_yaw_degrees']:.1f}°")
        axes[idx, 3].axis('off')

        # Add text info on the left
        info_text = (f"Ego: ({result['ego_points'][0]:.1f}, {result['ego_points'][1]:.1f})\n"
                    f"Heading: {result['ego_yaw_degrees']:.1f}°")
        fig.text(0.01, 1 - (idx + 0.5) / n_rows, info_text,
                fontsize=8, ha='left', va='center')

    plt.suptitle(f"BEV Comparison: Vector vs Segmentation V2 (15 Samples)\n"
                f"Full 360° View (256×256) - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.05, 0.02, 1, 0.98])

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    plt.close()


def main():
    """Main function to generate 15 sample comparisons."""
    print("=" * 60)
    print("Generating 15 Sample BEV Comparisons")
    print("Vector vs Segmentation V2 with RGB Top-down Views")
    print("=" * 60)

    data_root = Path("/workspace/Bench2Drive-Base")
    map_dir = Path("/workspace/Bench2Drive-Map")

    # Define 15 test scenarios
    test_scenarios = [
        # Town03 scenarios
        ("Accident_Town03_Route156_Weather0", "00050", "Town03"),
        ("Accident_Town03_Route156_Weather0", "00100", "Town03"),
        ("Accident_Town03_Route156_Weather0", "00150", "Town03"),
        ("Accident_Town03_Route146_Weather8", "00050", "Town03"),
        ("Accident_Town03_Route146_Weather8", "00100", "Town03"),
        ("BlockedIntersection_Town03_Route134_Weather3", "00050", "Town03"),
        ("BlockedIntersection_Town03_Route134_Weather3", "00100", "Town03"),
        ("Accident_Town03_Route101_Weather23", "00050", "Town03"),
        ("Accident_Town03_Route101_Weather23", "00100", "Town03"),
        # Town10 scenarios (if available)
        ("BlockedIntersection_Town10_Route88_Weather6", "00050", "Town10"),
        ("BlockedIntersection_Town10_Route88_Weather6", "00100", "Town10"),
        # Town02 scenarios (if available)
        ("Accident_Town02_Route63_Weather4", "00050", "Town02"),
        ("Accident_Town02_Route63_Weather4", "00100", "Town02"),
        # Additional Town03 scenarios
        ("Accident_Town03_Route165_Weather0", "00050", "Town03"),
        ("Accident_Town03_Route165_Weather0", "00100", "Town03"),
    ]

    # Load maps
    print("\nLoading HD maps...")
    map_processors = {}
    for town in ["Town02", "Town03", "Town10"]:
        map_path = map_dir / f"{town}_HD_map.npz"
        if map_path.exists():
            print(f"  Loading {town} map...")
            map_data = load_map_data(map_path)
            map_processors[town] = MapProcessor(map_data)
            print(f"    ✓ {town} loaded")

    # Process scenarios
    results = []
    processed = 0
    target = 15

    print(f"\nProcessing scenarios (target: {target} samples)...")
    for scenario_name, frame_number, town in test_scenarios:
        if processed >= target:
            break

        scenario_path = data_root / scenario_name
        if not scenario_path.exists():
            print(f"  ⚠️ Skipping {scenario_name} (not found)")
            continue

        if town not in map_processors:
            print(f"  ⚠️ Skipping {scenario_name} (no map for {town})")
            continue

        try:
            print(f"  Processing {scenario_name}/{frame_number}...")
            result = generate_comparison(scenario_path, frame_number, map_processors[town])
            results.append((scenario_name, frame_number, result))
            processed += 1
            print(f"    ✓ {processed}/{target} completed")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue

    if results:
        print(f"\n✓ Successfully processed {len(results)} samples")

        # Create visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_path = f'/workspace/DiffusionDrive/segmentation_15samples_{timestamp}.png'
        visualize_samples(results, output_path)

        # Print summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)

        total_vector_pixels = 0
        total_seg_pixels = 0
        has_centerlines = 0
        has_static = 0

        for _, _, result in results:
            total_vector_pixels += np.sum(result['vector_bev'] > 0)
            total_seg_pixels += np.sum(result['seg_bev'] > 0)
            if 3 in result['seg_bev']:
                has_centerlines += 1
            if 4 in result['seg_bev']:
                has_static += 1

        print(f"Average vector pixels: {total_vector_pixels / len(results):.0f}")
        print(f"Average segmentation pixels: {total_seg_pixels / len(results):.0f}")
        print(f"Coverage increase: {total_seg_pixels / total_vector_pixels:.1f}x")
        print(f"Samples with centerlines: {has_centerlines}/{len(results)}")
        print(f"Samples with static objects: {has_static}/{len(results)}")

        print("\n✅ Visualization complete!")
        print(f"Check: {output_path}")
    else:
        print("\n❌ No samples could be processed")


if __name__ == "__main__":
    main()