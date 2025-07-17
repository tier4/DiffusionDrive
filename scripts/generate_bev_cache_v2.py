#!/usr/bin/env python3
"""
Generate BEV semantic maps offline for Bench2Drive dataset.

This script uses Ray and a KDTree-based MapProcessor for high-performance,
memory-efficient BEV map generation.
"""
import os
import sys
import argparse
import json
import gzip
import numpy as np
import logging
import ray
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

from navsim.common.bev_map_utils_v2 import (
    MapProcessor,
    load_map_data,
    extract_front_half_bev,
    generate_full_bev_from_map,  # Assuming this is the updated version
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Sets up logging for the main driver."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def load_annotation(anno_path: Path) -> Dict:
    """Loads a gzipped JSON annotation file."""
    with gzip.open(anno_path, "rt", encoding="utf-8") as f:
        return json.load(f)


@ray.remote
def worker_process_frame_ray(
    frame_path: Path,
    output_root_dir: Path,
    map_processor_refs: Dict[str, ray.ObjectRef],
    generate_full: bool,
    overwrite: bool,
) -> bool:
    """
    Ray worker task that uses the shared MapProcessor to generate a BEV map.
    """
    frame_number = frame_path.stem.split(".")[0]
    scenario_name = frame_path.parent.parent.name
    output_path = output_root_dir / scenario_name / f"{frame_number}.npz"

    if output_path.exists() and not overwrite:
        return False  # Skip if already exists and not overwriting

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        anno = load_annotation(frame_path)
        world2ego = np.array(anno["bounding_boxes"][0]["world2ego"])

        # Find the correct map processor for this frame's town
        town_name = next((p for p in scenario_name.split("_") if p.startswith("Town")), None)
        if not town_name or town_name not in map_processor_refs:
            logging.warning(f"No map processor found for town {town_name} in frame {frame_path}")
            return False

        # Get the MapProcessor object from Ray's object store
        map_processor = ray.get(map_processor_refs[town_name])

        if generate_full:
            # Call the optimized function that uses the processor
            full_bev = generate_full_bev_from_map(map_processor, world2ego)
            front_bev = extract_front_half_bev(full_bev)
            np.savez_compressed(
                output_path,
                full_bev=full_bev.astype(np.float32),
                front_bev=front_bev.astype(np.float32),
                world2ego=world2ego,
                frame_idx=int(frame_number),
            )
        else:
            # Call the processor's fast generation method directly
            front_bev = map_processor.generate_bev(world2ego)
            np.savez_compressed(
                output_path,
                front_bev=front_bev.astype(np.float32),
                world2ego=world2ego,
                frame_idx=int(frame_number),
            )
        return True
    except Exception as e:
        logging.error(f"Error processing {frame_path}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=str, required=True, help="Root directory of Bench2Drive dataset"
    )
    parser.add_argument(
        "--map-dir", type=str, required=True, help="Directory containing map NPZ files"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for BEV cache"
    )
    parser.add_argument(
        "--scenarios", type=str, nargs="+", help="Specific scenarios to process (default: all)"
    )
    parser.add_argument(
        "--full-bev", action="store_true", help="Generate full 360° BEV maps (default: front only)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing BEV files")
    parser.add_argument("--max-frames", type=int, help="Maximum frames per scenario to process")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (CPUs for Ray)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    ray.init(
        num_cpus=args.workers, logging_level=logging.INFO if not args.verbose else logging.DEBUG
    )
    logger.info(f"Ray initialized with {ray.available_resources().get('CPU', 0)} CPUs.")

    try:
        data_root, map_dir, output_dir = (
            Path(args.data_root),
            Path(args.map_dir),
            Path(args.output_dir),
        )
        if not data_root.is_dir() or not map_dir.is_dir():
            logger.error(f"Data root ({data_root}) or map directory ({map_dir}) not found.")
            return 1
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Collect all frames and unique towns ---
        scenario_dirs = (
            [data_root / s for s in args.scenarios]
            if args.scenarios
            else [d for d in data_root.iterdir() if d.is_dir() and "Town" in d.name]
        )

        all_frame_files, unique_towns = [], set()
        for scenario_dir in (s for s in scenario_dirs if s.is_dir()):
            town_name = next(
                (p for p in scenario_dir.name.split("_") if p.startswith("Town")), None
            )
            if town_name:
                unique_towns.add(town_name)

            anno_dir = scenario_dir / "anno"
            if anno_dir.is_dir():
                frames = sorted(anno_dir.glob("*.json.gz"))
                all_frame_files.extend(frames[: args.max_frames] if args.max_frames else frames)

        if not all_frame_files:
            logger.warning("No frames found to process.")
            return 0

        # --- 2. Pre-process maps ONCE and share with Ray ---
        logger.info(f"Found {len(unique_towns)} unique towns. Pre-processing maps with KDTree...")
        map_processor_refs = {}
        for town_name in tqdm(unique_towns, desc="Pre-processing maps"):
            map_path = map_dir / f"{town_name}_HD_map.npz"
            if map_path.exists():
                map_data = load_map_data(map_path)
                processor = MapProcessor(map_data)
                map_processor_refs[town_name] = ray.put(processor)
        logger.info("All maps pre-processed and shared in Ray's object store.")

        # --- 3. Launch Ray tasks ---
        logger.info(f"Dispatching {len(all_frame_files)} frames to Ray workers...")
        result_refs = [
            worker_process_frame_ray.remote(
                frame, output_dir, map_processor_refs, args.full_bev, args.overwrite
            )
            for frame in all_frame_files
        ]

        # --- 4. Collect results with a progress bar ---
        results = []
        with tqdm(total=len(result_refs), desc="Processing frames") as pbar:
            while result_refs:
                done_refs, result_refs = ray.wait(
                    result_refs, num_returns=min(len(result_refs), 100)
                )
                results.extend(ray.get(done_refs))
                pbar.update(len(done_refs))

        # --- 5. Summarize results ---
        total_processed = sum(1 for r in results if r)
        total_skipped = len(results) - total_processed
        logger.info(
            f"\n--- Processing Complete ---\nTotal Frames Processed: {total_processed}\nTotal Frames Skipped/Failed: {total_skipped}"
        )

        # Save metadata...
        metadata = {
            "data_root": str(data_root),
            "map_dir": str(map_dir),
            "generate_full": args.full_bev,
            "total_processed": total_processed,
            "total_skipped": total_skipped,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {output_dir / 'metadata.json'}")

        return 0
    finally:
        ray.shutdown()
        logger.info("Ray has been shut down.")


if __name__ == "__main__":
    sys.exit(main())
