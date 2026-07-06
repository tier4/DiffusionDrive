"""
Bench2Drive agent that combines normalization with B2D-specific features.
"""

import logging
from typing import List
from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_model_wrapper import V2TransfuserModelWrapper
from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder,
)
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

logger = logging.getLogger(__name__)


class Bench2DriveAgent(TransfuserAgent):
    """Agent specifically for Bench2Drive with proper features and normalization."""

    def __init__(self, config: Bench2DriveConfig, lr: float = 6e-4, checkpoint_path: str = None):
        """Initialize with B2D config for normalization support."""
        self._b2d_config = config
        super().__init__(config, lr, checkpoint_path)
        # super().__init__ creates a plain V2TransfuserModel — replace it with
        # the wrapper that registers B2D-specific normalization buffers.
        logger.info(
            f"[Bench2DriveAgent] Replacing model with B2D wrapper: "
            f"dataset_type={self._b2d_config.dataset_type}"
        )
        self._transfuser_model = V2TransfuserModelWrapper(self._b2d_config)
        self.init_from_pretrained()

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Return Bench2Drive feature builders."""
        return [Bench2DriveFeatureBuilder(config=self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Return Bench2Drive target builders."""
        return [Bench2DriveTargetBuilder(config=self._config)]
