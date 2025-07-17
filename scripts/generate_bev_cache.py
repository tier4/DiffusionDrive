#!/usr/bin/env python3
"""
Generate BEV semantic maps offline for Bench2Drive dataset.

This script processes Bench2Drive scenarios and generates BEV semantic maps
using vectorized map data, saving them to disk for efficient loading during training.
"""
import os
import sys
import argparse
import json
import gzip
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from navsim.common.bev_map_utils import (
    load_map_data,
    generate_bev_from_map,
    generate_full_bev_from_map,
    extract_front_half_bev,
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def load_annotation(anno_path: Path) -> Dict:
    """Load Bench2Drive annotation file."""
    with gzip.open(anno_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def process_frame(
    frame_path: Path,
    map_data: Dict,
    output_dir: Path,
    generate_full: bool = True,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Process a single frame to generate BEV map.

    Args:
        frame_path: Path to annotation file
        map_data: Loaded map data for the town
        output_dir: Directory to save BEV maps
        generate_full: Whether to generate full 360° BEV
        overwrite: Whether to overwrite existing files

    Returns:
        Path to saved BEV file or None if skipped
    """
    # Output path
    frame_number = frame_path.stem.split(".")[0]  # Remove .json from stem
    output_path = output_dir / f"{frame_number}.npz"

    # Skip if exists and not overwriting
    if output_path.exists() and not overwrite:
        return None

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load annotation
        anno = load_annotation(frame_path)

        # Get world2ego transformation
        # Note: B2D uses left-handed coordinates
        world2ego = np.array(anno["bounding_boxes"][0]["world2ego"])

        # Generate BEV
        if generate_full:
            # Generate full 360° BEV (256x256)
            logging.debug(f"Generating full BEV for frame {frame_number}")
            full_bev = generate_full_bev_from_map(
                map_data=map_data,
                world2ego=world2ego,
                full_height=256,
                full_width=256,
                resolution=0.25,
                lane_thickness=0.4,
                max_distance=75.0,
            )

            # Extract front half for NavSim format (128x256)
            front_bev = extract_front_half_bev(full_bev)

            # Save both versions
            np.savez_compressed(
                output_path,
                full_bev=full_bev.astype(np.float32),
                front_bev=front_bev.astype(np.float32),
                world2ego=world2ego,
                frame_idx=int(frame_number),
            )

            # Log statistics
            full_unique = np.unique(full_bev)
            front_unique = np.unique(front_bev)
            logging.debug(
                f"Frame {frame_number}: Full BEV classes {full_unique}, Front BEV classes {front_unique}"
            )
        else:
            # Generate only front BEV (128x256)
            logging.debug(f"Generating front-only BEV for frame {frame_number}")
            front_bev = generate_bev_from_map(
                map_data=map_data,
                world2ego=world2ego,
                bev_height=128,
                bev_width=256,
                resolution=0.25,
                coverage_behind=0.0,
                lane_thickness=0.4,
                max_distance=50.0,
            )

            # Save
            np.savez_compressed(
                output_path,
                front_bev=front_bev.astype(np.float32),
                world2ego=world2ego,
                frame_idx=int(frame_number),
            )

            # Log statistics
            front_unique = np.unique(front_bev)
            logging.debug(f"Frame {frame_number}: Front BEV classes {front_unique}")

        return output_path

    except Exception as e:
        logging.error(f"Error processing {frame_path}: {e}")
        return None


def process_scenario(
    scenario_dir: Path,
    map_dir: Path,
    output_dir: Path,
    generate_full: bool = True,
    overwrite: bool = False,
    max_frames: Optional[int] = None,
) -> Tuple[str, int, int]:
    """
    Process all frames in a scenario.

    Args:
        scenario_dir: Path to scenario directory
        map_dir: Path to map directory
        output_dir: Output directory for BEV maps
        generate_full: Whether to generate full 360° BEV
        overwrite: Whether to overwrite existing files
        max_frames: Maximum number of frames to process

    Returns:
        Tuple of (scenario_name, processed_count, skipped_count)
    """
    scenario_name = scenario_dir.name

    # Extract town name from scenario
    # Format: ScenarioType_TownXX_RouteXX_WeatherXX
    parts = scenario_name.split("_")
    town_name = None
    for part in parts:
        if part.startswith("Town"):
            town_name = part
            break

    if not town_name:
        logging.warning(f"Could not extract town name from {scenario_name}")
        return scenario_name, 0, 0

    # Load map data
    map_path = map_dir / f"{town_name}_HD_map.npz"
    if not map_path.exists():
        logging.warning(f"Map not found: {map_path}")
        return scenario_name, 0, 0

    try:
        logging.info(f"Loading map data from {map_path}")
        map_data = load_map_data(map_path)
        logging.info(f"Map loaded successfully with {len(map_data)} roads")
    except Exception as e:
        logging.error(f"Error loading map {map_path}: {e}")
        return scenario_name, 0, 0

    # Get annotation files
    anno_dir = scenario_dir / "anno"
    if not anno_dir.exists():
        logging.warning(f"No annotation directory: {anno_dir}")
        return scenario_name, 0, 0

    anno_files = sorted(anno_dir.glob("*.json.gz"))
    if max_frames:
        anno_files = anno_files[:max_frames]

    logging.info(f"Processing {len(anno_files)} frames for scenario {scenario_name}")

    # Process frames
    processed = 0
    skipped = 0

    for i, anno_file in enumerate(anno_files):
        if i > 0 and i % 50 == 0:
            logging.info(f"  Progress: {i}/{len(anno_files)} frames processed")

        result = process_frame(
            anno_file, map_data, output_dir / scenario_name, generate_full, overwrite
        )

        if result:
            processed += 1
        else:
            skipped += 1

    logging.info(f"Scenario {scenario_name} completed: {processed} processed, {skipped} skipped")
    return scenario_name, processed, skipped


def worker_process_scenario(args):
    """Worker function for multiprocessing."""
    return process_scenario(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        default="/workspace/Bench2Drive-mini",
        help="Root directory of Bench2Drive dataset",
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        default="/workspace/Bench2Drive-Map",
        help="Directory containing map NPZ files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/DiffusionDrive/data/bev_cache",
        help="Output directory for BEV cache",
    )
    parser.add_argument(
        "--scenarios", type=str, nargs="+", help="Specific scenarios to process (default: all)"
    )
    parser.add_argument(
        "--full-bev", action="store_true", help="Generate full 360° BEV maps (default: front only)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing BEV files")
    parser.add_argument("--max-frames", type=int, help="Maximum frames per scenario to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)

    # Paths
    data_root = Path(args.data_root)
    map_dir = Path(args.map_dir)
    output_dir = Path(args.output_dir)

    # Validate paths
    if not data_root.exists():
        logger.error(f"Data root not found: {data_root}")
        return 1

    if not map_dir.exists():
        logger.error(f"Map directory not found: {map_dir}")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get scenarios to process
    if args.scenarios:
        scenario_dirs = [data_root / s for s in args.scenarios]
    else:
        # Get all scenario directories
        scenario_dirs = [
            d for d in data_root.iterdir() if d.is_dir() and "Town" in d.name and "Route" in d.name
        ]

    # Filter valid scenarios
    valid_scenarios = [s for s in scenario_dirs if s.exists()]

    logger.info(f"Found {len(valid_scenarios)} scenarios to process")

    # Prepare arguments for workers
    worker_args = [
        (scenario_dir, map_dir, output_dir, args.full_bev, args.overwrite, args.max_frames)
        for scenario_dir in valid_scenarios
    ]

    # Process scenarios
    total_processed = 0
    total_skipped = 0

    if args.workers > 1:
        # Parallel processing
        with mp.Pool(args.workers) as pool:
            results = list(
                tqdm(
                    pool.imap(worker_process_scenario, worker_args),
                    total=len(worker_args),
                    desc="Processing scenarios",
                )
            )
    else:
        # Sequential processing
        results = []
        for args_tuple in tqdm(worker_args, desc="Processing scenarios"):
            results.append(process_scenario(*args_tuple))

    # Summarize results
    for scenario_name, processed, skipped in results:
        total_processed += processed
        total_skipped += skipped
        if processed > 0:
            logger.info(f"{scenario_name}: {processed} processed, {skipped} skipped")

    logger.info(f"\nTotal: {total_processed} processed, {total_skipped} skipped")

    # Save metadata
    metadata = {
        "data_root": str(data_root),
        "map_dir": str(map_dir),
        "generate_full": args.full_bev,
        "scenarios": [s.name for s in valid_scenarios],
        "total_processed": total_processed,
        "total_skipped": total_skipped,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to {metadata_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
