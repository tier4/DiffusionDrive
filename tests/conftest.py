"""
Shared pytest fixtures for tests.
"""

import pytest
from pathlib import Path
from navsim.common.bench2drive_dataloader import Bench2DriveConfig


@pytest.fixture
def sample_data_dir():
    """Path to the sample test data directory."""
    return Path(__file__).parent / "test_data" / "bench2drive_sample"


@pytest.fixture
def sample_config(sample_data_dir):
    """Standard configuration using sample data."""
    return Bench2DriveConfig(
        data_root=sample_data_dir,
        scenarios=["DynamicObjectCrossing_Town02_Route13_Weather6"],
        sampling_rate=5,
        num_frames=6,
        num_history_frames=2,
        num_future_frames=4,
        extract_tar=False,
        map_dir=sample_data_dir / "maps",
        bev_cache_dir=None,  # No BEV cache by default
    )


@pytest.fixture
def sample_config_with_bev_cache(sample_data_dir):
    """Configuration with BEV cache for specific tests."""
    return Bench2DriveConfig(
        data_root=sample_data_dir,
        scenarios=["DynamicObjectCrossing_Town02_Route13_Weather6"],
        sampling_rate=5,
        num_frames=6,
        num_history_frames=2,
        num_future_frames=4,
        extract_tar=False,
        map_dir=sample_data_dir / "maps",
        bev_cache_dir=sample_data_dir / "sample_bev_cache",
    )


@pytest.fixture
def sample_scenario_name():
    """Name of the sample scenario."""
    return "DynamicObjectCrossing_Town02_Route13_Weather6"