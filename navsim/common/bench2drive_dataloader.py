"""
Bench2Drive dataset loader for CARLA-native training.
Implements Method 3 from bench2drive_integration_strategy.md
"""

import gzip
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from navsim.common.abstract_carla_dataloader import AbstractCarlaDataConfig, AbstractCarlaSceneLoader


@dataclass
class Bench2DriveDataConfig(AbstractCarlaDataConfig):
    """Configuration for Bench2Drive dataset loading."""

    scenarios: List[str] = None  # List of scenario names to load
    sampling_rate: int = 5  # Downsample from 10Hz to 2Hz (take every 5th frame)
    num_frames: int = 30  # Number of frames per scene (15 seconds at 2Hz)
    num_history_frames: int = 4  # Past frames
    num_future_frames: int = 26  # Future frames
    extract_tar: bool = True  # Whether to extract tar files
    map_dir: Optional[Path] = None  # Directory containing HD map NPZ files
    bev_cache_dir: Optional[Path] = None  # Directory containing pre-generated BEV maps
    debug_mode: bool = False  # Allow on-the-fly BEV generation for debugging (bypasses cache requirement)
    # DEPRECATED: 10Hz sliding window mode removed to simplify codebase
    # sliding_mode: bool = True  # If True, use true sliding window through all 10Hz frames
    sliding_mode: bool = False  # Force legacy mode only


class Bench2DriveSceneLoader(AbstractCarlaSceneLoader):
    """
    Scene loader for Bench2Drive dataset.
    Implements CARLA-native loading (Method 3) without coordinate transformations.
    """

    def __init__(
        self,
        config: Bench2DriveDataConfig,
        planner: Optional[any] = None,
        trajectory_sampling: Optional[any] = None,
    ):
        """
        Initialize Bench2Drive scene loader.

        Args:
            config: Dataset configuration
            planner: Optional planner for expert demonstrations
            trajectory_sampling: Trajectory sampling configuration
        """
        super().__init__(config)
        self.planner = planner
        self.trajectory_sampling = trajectory_sampling

        # Build scene index
        self._build_scene_index()

    def _build_scene_index(self) -> None:
        """Build index of available scenes from dataset."""
        self.scenes = {}
        self.scene_tokens = []

        # Check if data needs extraction
        if self.config.extract_tar:
            self._extract_tar_files()

        # Iterate through scenarios
        for scenario_name in self.config.scenarios:
            # Check if this is a full scenario path (e.g., "ConstructionObstacle_Town05_Route68_Weather8")
            # or just a scenario type (e.g., "ConstructionObstacle")
            if "_Town" in scenario_name:
                # Full scenario path - use directly
                scenario_dirs = [self.config.data_root / scenario_name]
            else:
                # Scenario type - find all matching directories
                scenario_dirs = list(self.config.data_root.glob(f"{scenario_name}_*"))

            for scenario_path in scenario_dirs:
                if not scenario_path.exists() or not scenario_path.is_dir():
                    raise FileNotFoundError(f"Scenario path not found: {scenario_path}")
                # For mini dataset, scenario_path is already the run directory
                run_dir = scenario_path

                # Check if annotation directory exists
                anno_dir = run_dir / "anno"
                if not anno_dir.exists():
                    raise FileNotFoundError(f"Annotation directory not found: {anno_dir}")

                # Get all frame annotations
                frames = sorted(anno_dir.glob("*.json.gz"))

                if len(frames) == 0:
                    raise ValueError(f"No frames found in scenario: {scenario_path}")

                # DEPRECATED: 10Hz sliding window mode removed
                # if self.config.sliding_mode:
                #     # TRUE SLIDING WINDOW: Slide through all 10Hz frames
                #     # Calculate frames needed for one sample
                #     # History span: 0 when num_history_frames=0 (start from frame 0)
                #     history_span = self.config.num_history_frames * self.config.sampling_rate
                #
                #     # Future span: Use config value (8 future frames * sampling_rate)
                #     # At 2Hz training: 8 * 5 = 40 raw frames needed
                #     future_span = self.config.num_future_frames * self.config.sampling_rate
                #
                #     # Iterate through every possible starting position
                #     # Start from history_span (0 if no history) to ensure we have enough past frames
                #     for start_idx in range(history_span, len(frames) - future_span):
                #         # Generate unique token
                #         token = f"{scenario_path.name}_{start_idx:05d}"
                #
                #         self.scenes[token] = {
                #             "scenario": scenario_path.name,
                #             "run": run_dir.name,
                #             "all_frames": frames,  # Store all frames for on-demand access
                #             "start_idx": start_idx,  # Current frame index in 10Hz
                #             "token": token,
                #         }
                #         self.scene_tokens.append(token)
                # else:

                # LEGACY MODE ONLY: Downsample first, then slide
                sampled_frames = frames[:: self.config.sampling_rate]

                # Create scenes with sliding window
                for i in range(len(sampled_frames) - self.config.num_frames + 1):
                    scene_frames = sampled_frames[i : i + self.config.num_frames]

                    # Generate unique token
                    token = f"{scenario_path.name}_{i:05d}"

                    self.scenes[token] = {
                        "scenario": scenario_path.name,
                        "run": run_dir.name,
                        "frames": scene_frames,
                        "base_path": run_dir,
                        "start_idx": i * self.config.sampling_rate,  # Original frame index
                        "token": token,  # Store token in scene info
                    }
                    self.scene_tokens.append(token)

        # mode_str = "true sliding window" if self.config.sliding_mode else "legacy downsampling"
        print(f"Built scene index with {len(self.scenes)} scenes using legacy downsampling mode")

    def _extract_tar_files(self) -> None:
        """Extract tar.gz files if needed."""
        # Look for tar.gz files in data root
        tar_files = list(self.config.data_root.glob("*.tar.gz"))

        for tar_file in tar_files:
            # Extract scenario name from tar filename
            scenario_name = tar_file.stem.replace(".tar", "")
            scenario_path = self.config.data_root / scenario_name

            # Skip if already extracted
            if scenario_path.exists():
                continue

            print(f"Extracting {tar_file.name}...")
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(self.config.data_root)

    def get_scene_tokens(self) -> List[str]:
        """Get list of all scene tokens."""
        return self.scene_tokens

    def get_scene(self, token: str) -> "Bench2DriveScene":
        """
        Load a scene by token.

        Args:
            token: Scene identifier

        Returns:
            Bench2DriveScene object
        """
        if token not in self.scenes:
            raise ValueError(f"Scene token {token} not found")

        scene_info = self.scenes[token]

        # Create and return scene object
        from navsim.common.bench2drive_scene import Bench2DriveScene

        return Bench2DriveScene(
            scene_info=scene_info,
            config=self.config,
            planner=self.planner,
            trajectory_sampling=self.trajectory_sampling,
        )

    def __len__(self) -> int:
        """Get number of scenes."""
        return len(self.scenes)

    def __iter__(self):
        """Iterate over scene tokens."""
        return iter(self.scene_tokens)


def load_bench2drive_annotation(anno_path: Path) -> Dict:
    """
    Load a single Bench2Drive annotation file.

    Args:
        anno_path: Path to .json.gz annotation file

    Returns:
        Dictionary containing annotation data
    """
    with gzip.open(anno_path, "rt") as f:
        return json.load(f)


def map_carla_command_to_discrete(command: int) -> int:
    """
    Map CARLA navigation command to discrete values.

    CARLA commands (from local_planner.py):
    - VOID = -1
    - LEFT = 1
    - RIGHT = 2
    - STRAIGHT = 3
    - LANEFOLLOW = 4
    - CHANGELANELEFT = 5
    - CHANGELANERIGHT = 6

    NavSim discrete commands:
    - 0 = LEFT (turns, lane changes, sharp curves)
    - 1 = STRAIGHT
    - 2 = RIGHT (turns, lane changes, sharp curves)
    - 3 = UNKNOWN

    Args:
        command: CARLA command value

    Returns:
        Discrete command (0-3)
    """
    # Bench2DriveZoo transformation (from vad_b2d_agent_visualize.py lines 390-392):
    if command < 0:
        command = 4
    command -= 1

    # After Bench2DriveZoo transformation:
    # VOID (-1) → 4 → 3
    # LEFT (1) → 0
    # RIGHT (2) → 1
    # STRAIGHT (3) → 2
    # LANEFOLLOW (4) → 3
    # CHANGELANELEFT (5) → 4
    # CHANGELANERIGHT (6) → 5

    # Map transformed values to NavSim format
    # After Bench2DriveZoo transformation, we correctly map:
    # - CARLA LEFT (originally 1, now 0) → NavSim LEFT (0)
    # - CARLA RIGHT (originally 2, now 1) → NavSim RIGHT (2)
    # - CARLA STRAIGHT (originally 3, now 2) → NavSim STRAIGHT (1)
    # This ensures proper command semantics are preserved
    if command == 0:  # Was LEFT in CARLA (1 → 0 after transformation)
        return 0  # LEFT in NavSim
    elif command == 1:  # Was RIGHT in CARLA (2 → 1 after transformation)
        return 2  # RIGHT in NavSim
    elif command == 2:  # Was STRAIGHT in CARLA (3 → 2 after transformation)
        return 1  # STRAIGHT in NavSim
    elif command == 3:  # Was LANEFOLLOW in CARLA (4 → 3 after transformation)
        return 1  # STRAIGHT in NavSim
    elif command == 4:  # Was CHANGELANELEFT in CARLA (5 → 4 after transformation)
        return 0  # LEFT in NavSim
    elif command == 5:  # Was CHANGELANERIGHT in CARLA (6 → 5 after transformation)
        return 2  # RIGHT in NavSim
    else:
        return 3  # UNKNOWN
