"""Tests for closed-loop model loading and inference plumbing."""
import pytest
import torch

from navsim.evaluation.checkpoint_loading import clean_lightning_state_dict


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
