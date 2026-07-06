import pytest
from navsim.common.bench2drive_constants import B2D_CLASS_TO_NAVSIM


class TestBench2DriveClassMapping:
    """Test Bench2Drive to NavSim class mappings."""

    def test_class_mapping_exists(self):
        """Test that class mapping constant exists."""
        assert B2D_CLASS_TO_NAVSIM is not None
        assert isinstance(B2D_CLASS_TO_NAVSIM, dict)

    def test_vehicle_mapping(self):
        """Test vehicle class mapping."""
        assert B2D_CLASS_TO_NAVSIM["vehicle"] == 5

    def test_walker_mapping(self):
        """Test walker/pedestrian class mapping."""
        assert B2D_CLASS_TO_NAVSIM["walker"] == 6

    def test_static_object_mappings(self):
        """Test static object mappings."""
        assert B2D_CLASS_TO_NAVSIM["traffic_light"] == 4
        assert B2D_CLASS_TO_NAVSIM["traffic_sign"] == 4

    def test_ego_vehicle_mapping(self):
        """Test ego vehicle is marked for skipping."""
        assert B2D_CLASS_TO_NAVSIM["ego_vehicle"] == -1
