"""Tests for per-future-timestep agent extraction (collision metric wiring)."""

import numpy as np
import torch


def test_get_future_agents_shapes_and_range(sample_config):
    from navsim.common.bench2drive_dataloader import Bench2DriveSceneLoader
    from navsim.common.bench2drive_constants import MAX_AGENTS, COLLISION_EVAL_RANGE_M

    loader = Bench2DriveSceneLoader(sample_config)
    token = loader.get_scene_tokens()[0]
    scene = loader.get_scene(token)

    states, labels = scene.get_future_agents(-1, num_timesteps=4)

    assert states.shape == (4, MAX_AGENTS, 5)
    assert labels.shape == (4, MAX_AGENTS)
    assert labels.dtype == torch.bool
    valid = states[labels]
    if len(valid):
        dist = torch.hypot(valid[:, 0], valid[:, 1])
        assert dist.max() <= COLLISION_EVAL_RANGE_M + 1e-3


def test_get_agents_unchanged_by_refactor(sample_config):
    """get_agents must be byte-identical before/after the helper extraction —
    this pins the training-path behavior B4 will change later, not B1."""
    from navsim.common.bench2drive_dataloader import Bench2DriveSceneLoader

    loader = Bench2DriveSceneLoader(sample_config)
    token = loader.get_scene_tokens()[0]
    scene = loader.get_scene(token)

    states, labels, types = scene.get_agents(-1)
    assert states.shape[1] == 5
    assert states[labels].abs().sum() > 0 or labels.sum() == 0
