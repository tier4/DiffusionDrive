"""
Extended TransfuserAgent that uses the model wrapper for normalization support.
"""

from typing import Type
from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_model_wrapper import V2TransfuserModelWrapper
from navsim.agents.diffusiondrive.extended_transfuser_config import ExtendedTransfuserConfig


class ExtendedTransfuserAgent(TransfuserAgent):
    """Extended agent that supports dataset-specific normalization."""
    
    def __init__(self, config: ExtendedTransfuserConfig, lr: float = 6e-4, checkpoint_path: str = None):
        """Initialize with extended config."""
        # Store extended config before calling parent
        self._extended_config = config
        super().__init__(config, lr, checkpoint_path)
    
    def _build_model(self) -> V2TransfuserModelWrapper:
        """Build model using the wrapper."""
        return V2TransfuserModelWrapper(self._extended_config)