"""Tests for closed-loop model loading and inference plumbing."""
import numpy as np
import pytest
import torch

from navsim.evaluation.checkpoint_loading import clean_lightning_state_dict
from navsim.evaluation.carla_closed_loop.model_runner import make_inference_config


class TestStateDictCleaning:
    def test_strips_agent_transfuser_prefix(self):
        raw = {"agent._transfuser_model.backbone.w": torch.zeros(1)}
        assert "backbone.w" in clean_lightning_state_dict(raw)

    def test_strips_agent_prefix(self):
        raw = {"agent.head.b": torch.zeros(1)}
        assert "head.b" in clean_lightning_state_dict(raw)

    def test_keeps_unprefixed_keys(self):
        raw = {"backbone.w": torch.zeros(1)}
        assert "backbone.w" in clean_lightning_state_dict(raw)


class TestInferenceConfig:
    def test_config_is_bench2drive(self):
        cfg = make_inference_config(bkb_path="/tmp/x.bin", anchor_path=None)
        assert cfg.dataset_type == "bench2drive"
        assert cfg.bkb_path == "/tmp/x.bin"
        assert cfg.trajectory_sampling.num_poses == 8

    def test_anchor_override(self):
        cfg = make_inference_config(bkb_path=None, anchor_path="/tmp/anchors.npy")
        assert cfg.plan_anchor_path == "/tmp/anchors.npy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
