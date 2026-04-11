"""
Abstract base classes for CARLA-based dataset loaders.
Defines the interface that Bench2Drive and TaCarla loaders must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AbstractCarlaDataConfig:
    """Base configuration for CARLA-based dataset loading."""

    data_root: Path
    sampling_rate: int = 5  # Downsample from 10Hz to 2Hz
    num_frames: int = 9  # 0 history + 1 current + 8 future
    num_history_frames: int = 0
    num_future_frames: int = 8


class AbstractCarlaSceneLoader(ABC):
    """
    Abstract scene loader for CARLA-based datasets.
    Subclasses must implement scene index building and scene creation.
    """

    def __init__(self, config: AbstractCarlaDataConfig):
        self.config = config
        self.scenes: Dict = {}
        self.scene_tokens: List[str] = []

    @abstractmethod
    def _build_scene_index(self) -> None:
        """Build index of available scenes from dataset."""
        ...

    def get_scene_tokens(self) -> List[str]:
        """Get list of all scene tokens."""
        return self.scene_tokens

    @abstractmethod
    def get_scene(self, token: str):
        """Load a scene by token."""
        ...

    def __len__(self) -> int:
        return len(self.scenes)

    def __iter__(self):
        return iter(self.scene_tokens)


def map_carla_command_to_discrete(command: int) -> int:
    """
    Map CARLA navigation command to discrete values.

    CARLA commands:
    - VOID = -1, LEFT = 1, RIGHT = 2, STRAIGHT = 3
    - LANEFOLLOW = 4, CHANGELANELEFT = 5, CHANGELANERIGHT = 6

    NavSim discrete: 0=LEFT, 1=STRAIGHT, 2=RIGHT, 3=UNKNOWN
    """
    command_map = {
        1: 0,   # LEFT -> LEFT
        2: 2,   # RIGHT -> RIGHT
        3: 1,   # STRAIGHT -> STRAIGHT
        4: 1,   # LANEFOLLOW -> STRAIGHT
        5: 0,   # CHANGELANELEFT -> LEFT
        6: 2,   # CHANGELANERIGHT -> RIGHT
    }
    return command_map.get(command, 3)  # default UNKNOWN


def map_carla_command_b2d_format(command: int) -> int:
    """
    Map CARLA command that has been pre-transformed by Bench2DriveZoo format.
    B2D shifts: if command < 0: command = 4; command -= 1
    So CARLA LEFT(1) -> 0, RIGHT(2) -> 1, STRAIGHT(3) -> 2, etc.
    """
    b2d_map = {
        0: 0,   # Was LEFT
        1: 2,   # Was RIGHT
        2: 1,   # Was STRAIGHT
        3: 1,   # Was LANEFOLLOW
        4: 0,   # Was CHANGELANELEFT
        5: 2,   # Was CHANGELANERIGHT
    }
    return b2d_map.get(command, 3)
