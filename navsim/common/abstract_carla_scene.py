"""
Abstract base class for CARLA-based dataset scenes.
Defines the interface that Bench2Drive and TaCarla scenes must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from navsim.common.dataclasses import AgentInput


class AbstractCarlaScene(ABC):
    """
    Abstract scene for CARLA-based datasets.
    Provides data in formats compatible with NAVSIM's feature/target builders.
    """

    @abstractmethod
    def get_agent_input(self, frame_idx: int = -1) -> AgentInput:
        """
        Get agent input for a specific frame.

        Returns AgentInput with:
        - ego_statuses: List[EgoStatus] (ego pose, velocity, acceleration, command)
        - cameras: List[Cameras] (6+ camera images)
        - lidars: List[Lidar] (point cloud)
        """
        ...

    @abstractmethod
    def get_future_trajectory(self, frame_idx: int = -1) -> Optional[torch.Tensor]:
        """
        Get future ego trajectory.

        Returns:
            Tensor of shape [8, 3] (x, y, heading) in ego-local coordinates,
            or None if insufficient future frames.
        """
        ...

    @abstractmethod
    def get_agents(self, frame_idx: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get agent states for a specific frame.

        Returns:
            agent_states: [max_agents, 5] (x, y, heading, length, width) in ego frame
            agent_labels: [max_agents] boolean mask for valid agents
            agent_types: [max_agents] class IDs (5=vehicle, 6=pedestrian)
        """
        ...

    @abstractmethod
    def get_bev_semantic_map(self, frame_idx: int = -1) -> torch.Tensor:
        """
        Get BEV semantic map for a specific frame.

        Returns:
            Tensor with semantic class labels. Shape and class count
            are dataset-specific (NAVSIM: [128,256] 7 classes,
            TaCarla: [128,256] or [200,200] 6 classes).
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Get number of frames in this scene."""
        ...
