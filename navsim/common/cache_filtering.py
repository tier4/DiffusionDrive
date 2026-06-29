"""
Filtering functions for dataset caching.
Skips samples that would cause NaN during training.
"""

import torch


def has_valid_agents(targets):
    """Check if sample has at least one valid agent (prevents NaN in Hungarian matching)."""
    if "agent_labels" not in targets:
        return False
    labels = targets["agent_labels"]
    if isinstance(labels, torch.Tensor):
        return labels.any().item()
    return any(labels)


def has_nonzero_trajectory(targets):
    """Check if ego trajectory is non-zero (prevents NaN in FP16 normalization)."""
    if "trajectory" not in targets:
        return False
    traj = targets["trajectory"]
    if isinstance(traj, torch.Tensor):
        return traj.abs().sum().item() > 1e-6
    return True


def should_filter_sample(targets):
    """Return True if sample should be filtered (skipped) from cache."""
    return not has_valid_agents(targets) or not has_nonzero_trajectory(targets)
