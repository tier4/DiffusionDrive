"""Model loading + single-frame inference for closed-loop CARLA evaluation."""
import os
from typing import Optional

import numpy as np
import torch

from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import Bench2DriveFeatureBuilder
from navsim.common.dataclasses import AgentInput
from navsim.evaluation.checkpoint_loading import load_b2d_model

DEFAULT_BKB_PATH = "/mnt/nvme1/diffusiondrive/weights/resnet34_a1_in1k.bin"


def make_inference_config(
    bkb_path: Optional[str] = None, anchor_path: Optional[str] = None
) -> Bench2DriveConfig:
    config = Bench2DriveConfig()
    # Bench2DriveConfig inherits a nonexistent bkb_path default; the backbone falls
    # back to it when timm cannot download resnet34 (offline container) -- set it
    # to a file that exists. See transfuser_backbone.py:19-28.
    config.bkb_path = bkb_path or os.environ.get("DD_BKB_PATH", DEFAULT_BKB_PATH)
    if anchor_path:
        config.plan_anchor_path = anchor_path
    return config


class ModelRunner:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        bkb_path: Optional[str] = None,
        anchor_path: Optional[str] = None,
    ):
        self.device = device
        self.config = make_inference_config(bkb_path, anchor_path)
        if not os.path.exists(self.config.plan_anchor_path):
            raise FileNotFoundError(f"Plan anchors not found: {self.config.plan_anchor_path}")
        self.feature_builder = Bench2DriveFeatureBuilder(self.config)
        self.model, self.load_info = load_b2d_model(checkpoint_path, self.config, device)

    @torch.no_grad()
    def infer(self, agent_input: AgentInput) -> np.ndarray:
        features = self.feature_builder.compute_features(agent_input)
        batched = {k: v.unsqueeze(0).to(self.device) for k, v in features.items()}
        trajectory = self.model(batched)["trajectory"]  # [1, 8, 3], best mode
        if trajectory.shape != (1, 8, 3):
            raise RuntimeError(f"Unexpected trajectory shape {tuple(trajectory.shape)}")
        return trajectory[0].float().cpu().numpy()
