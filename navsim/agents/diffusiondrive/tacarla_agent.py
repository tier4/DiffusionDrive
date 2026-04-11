"""
TaCarla agent that combines normalization with TaCarla-specific features.
"""

import logging
from typing import List

from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_model_wrapper import V2TransfuserModelWrapper
from navsim.agents.diffusiondrive.tacarla_config import TaCarlaConfig
from navsim.agents.diffusiondrive.transfuser_features_tacarla import (
    TaCarlaFeatureBuilder,
    TaCarlaTargetBuilder,
)
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

logger = logging.getLogger(__name__)


class TaCarlaAgent(TransfuserAgent):
    """Agent specifically for TaCarla with proper features and normalization."""

    def __init__(self, config: TaCarlaConfig, lr: float = 5e-5, checkpoint_path: str = None):
        self._tc_config = config
        super().__init__(config, lr, checkpoint_path)

    def _build_model(self) -> V2TransfuserModelWrapper:
        logger.info(
            f"[TaCarlaAgent] Building model with config: dataset_type={self._tc_config.dataset_type}, "
            f"num_bev_classes={self._tc_config.num_bev_classes}"
        )
        return V2TransfuserModelWrapper(self._tc_config)

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [TaCarlaFeatureBuilder(config=self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TaCarlaTargetBuilder(config=self._config)]
