"""
TaCarla dataset loader.
Reads Parquet labels and provides scene access for feature/target building.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import logging

from navsim.common.abstract_carla_dataloader import (
    AbstractCarlaDataConfig,
    AbstractCarlaSceneLoader,
    map_carla_command_to_discrete,
)

logger = logging.getLogger(__name__)


@dataclass
class TaCarlaDataConfig(AbstractCarlaDataConfig):
    """Configuration for TaCarla dataset loading."""

    labels_root: Path = None  # Root of TaCarla_labels/ (contains Town12_labels/, Town13_labels/)
    sensor_root: Path = None  # Root of extracted sensor data
    towns: List[str] = field(default_factory=lambda: ["Town12", "Town13"])
    sampling_rate: int = 5  # 10Hz -> 2Hz
    num_frames: int = 9  # 0 history + 1 current + 8 future
    num_history_frames: int = 0
    num_future_frames: int = 8
    max_routes: Optional[int] = None  # Limit routes for debugging (None = all)


class TaCarlaSceneLoader(AbstractCarlaSceneLoader):
    """
    Scene loader for TaCarla dataset.
    Reads Parquet labels, builds frame index with 2Hz downsampling.
    """

    def __init__(self, config: TaCarlaDataConfig):
        super().__init__(config)
        self._parquet_cache = {}  # Cache: route_name -> lazy-loaded DataFrame
        self._build_scene_index()

    def _build_scene_index(self) -> None:
        """Build index of available scenes from Parquet labels."""
        self.scenes = {}
        self.scene_tokens = []

        config = self.config
        labels_root = Path(config.labels_root)

        total_routes = 0
        for town in config.towns:
            town_dir = labels_root / f"{town}_labels"
            if not town_dir.exists():
                logger.warning(f"Town labels directory not found: {town_dir}")
                continue

            parquet_files = sorted(town_dir.glob("*.parquet"))
            logger.info(f"Found {len(parquet_files)} parquet files in {town_dir.name}")

            for pq_path in parquet_files:
                # Route name from filename (strip .parquet)
                route_name = pq_path.stem

                # Check that extracted sensor data exists for this route
                sensor_dir = Path(config.sensor_root) / route_name
                if not sensor_dir.exists():
                    continue

                # Get frame count without loading full data
                # Use pyarrow metadata for fast row count
                try:
                    import pyarrow.parquet as pq
                    pq_meta = pq.read_metadata(pq_path)
                    n_frames = pq_meta.num_rows
                except Exception as e:
                    logger.warning(f"Cannot read metadata from {pq_path}: {e}")
                    continue

                if n_frames < config.num_frames * config.sampling_rate:
                    logger.debug(f"Skipping {route_name}: only {n_frames} frames (need {config.num_frames * config.sampling_rate})")
                    continue

                # Downsample to 2Hz: take every sampling_rate-th frame
                sampled_indices = list(range(0, n_frames, config.sampling_rate))

                # Create sliding windows of size num_frames over 2Hz indices
                for window_idx in range(len(sampled_indices) - config.num_frames + 1):
                    window_frames = sampled_indices[window_idx : window_idx + config.num_frames]
                    token = f"{route_name}_{window_idx:05d}"

                    self.scenes[token] = {
                        "route_name": route_name,
                        "parquet_path": str(pq_path),
                        "frame_indices": window_frames,  # 10Hz indices into Parquet rows
                        "window_idx": window_idx,
                        "token": token,
                        "town": town,
                    }
                    self.scene_tokens.append(token)

                total_routes += 1
                if config.max_routes and total_routes >= config.max_routes:
                    break

            if config.max_routes and total_routes >= config.max_routes:
                break

        logger.info(f"Built TaCarla scene index: {total_routes} routes, {len(self.scenes)} scenes")

    def get_scene(self, token: str):
        """Load a scene by token."""
        if token not in self.scenes:
            raise ValueError(f"Scene token {token} not found")

        scene_info = self.scenes[token]

        from navsim.common.tacarla_scene import TaCarlaScene
        return TaCarlaScene(
            scene_info=scene_info,
            config=self.config,
        )

    def _load_parquet_lazy(self, parquet_path: str):
        """Load Parquet file lazily with caching."""
        if parquet_path not in self._parquet_cache:
            import pyarrow.parquet as pq
            # Only load needed columns
            columns = [
                "measurements", "new_boxes", "lanes",
                "front", "front_left", "front_right",
                "back", "back_left", "back_right",
                "lidar", "bev_label_image",
            ]
            table = pq.read_table(parquet_path, columns=columns)
            self._parquet_cache[parquet_path] = table.to_pandas()
        return self._parquet_cache[parquet_path]

    def clear_cache(self):
        """Clear Parquet cache to free memory."""
        self._parquet_cache.clear()
