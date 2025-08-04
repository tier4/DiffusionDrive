#!/usr/bin/env python3
"""
Simple script to convert BEV cache .npz files to PNG images.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def npz_to_png(npz_file, output_file=None):
    """Convert BEV cache .npz file directly to PNG."""

    # Load the .npz file
    data = np.load(npz_file)

    # Get the BEV map (try different possible keys)
    if "bev_map" in data:
        bev_map = data["bev_map"]
    elif "arr_0" in data:
        bev_map = data["arr_0"]
    else:
        # Use first available array
        keys = list(data.keys())
        if keys:
            bev_map = data[keys[0]]
        else:
            raise ValueError(f"No data found in {npz_file}")

    print(f"Loaded BEV map: {bev_map.shape}, dtype: {bev_map.dtype}")
    print(f"Value range: {bev_map.min()} - {bev_map.max()}")
    print(f"Unique values: {np.unique(bev_map)}")

    # Create colormap for semantic classes
    colors = {
        0: [0, 0, 0],  # Background (black)
        1: [128, 128, 128],  # Road (gray)
        2: [255, 178, 102],  # Walkways (orange)
        3: [255, 255, 0],  # Lane centerlines (yellow)
        4: [0, 0, 255],  # Static objects (blue)
        5: [255, 0, 0],  # Vehicles (red)
        6: [0, 255, 0],  # Pedestrians (green)
    }

    # Convert to RGB image
    rgb_image = np.zeros((*bev_map.shape, 3), dtype=np.uint8)

    for class_id, color in colors.items():
        mask = bev_map == class_id
        rgb_image[mask] = color

    # Determine output filename
    if output_file is None:
        output_file = npz_file.with_suffix(".png")

    # Save as PNG using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image, origin="lower")
    plt.title(f"BEV Cache: {npz_file.name}")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")

    # Add ego vehicle marker
    h, w = bev_map.shape
    if h == 256 and w == 256:
        ego_x, ego_y = w // 2, h // 2  # Center for 360° view
    else:
        ego_x, ego_y = w // 2, h - 1  # Bottom center for front view

    plt.scatter(
        ego_x,
        ego_y,
        c="yellow",
        s=200,
        marker="*",
        edgecolors="black",
        linewidth=2,
        label="Ego Vehicle",
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved PNG to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert BEV cache .npz to PNG")
    parser.add_argument("input", type=str, help="Input .npz file or directory")
    parser.add_argument("--output", "-o", type=str, help="Output PNG file (optional)")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file() and input_path.suffix == ".npz":
        # Single file
        output_path = Path(args.output) if args.output else None
        npz_to_png(input_path, output_path)

    elif input_path.is_dir():
        # Directory - process all .npz files
        npz_files = list(input_path.glob("**/*.npz"))
        if not npz_files:
            print(f"No .npz files found in {input_path}")
            return

        print(f"Found {len(npz_files)} .npz files")
        for npz_file in npz_files:
            try:
                npz_to_png(npz_file)
            except Exception as e:
                print(f"Error processing {npz_file}: {e}")
    else:
        print(f"Invalid input: {input_path}")
        print("Please provide a .npz file or directory containing .npz files")


if __name__ == "__main__":
    main()
