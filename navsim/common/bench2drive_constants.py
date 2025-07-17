"""Constants for Bench2Drive integration with DiffusionDrive.

This file contains all constants used in the Bench2Drive integration,
documenting the differences between Bench2Drive's native data format
and DiffusionDrive's expected inputs.
"""

# LiDAR Coverage Parameters
# -------------------------
# Bench2Drive provides larger LiDAR coverage than DiffusionDrive expects.
# We process at full resolution then resize to preserve maximum information.

BENCH2DRIVE_LIDAR_RANGE_M = 85.0  # Bench2Drive native LiDAR range in meters
DIFFUSIONDRIVE_LIDAR_RANGE_M = 64.0  # DiffusionDrive expected range (-32 to +32)
LIDAR_PIXELS_PER_METER = 4.0  # Resolution for LiDAR BEV processing

# BEV Sizes
# ---------
# LiDAR BEV sizes at different stages of processing
BENCH2DRIVE_LIDAR_SIZE = int(BENCH2DRIVE_LIDAR_RANGE_M * LIDAR_PIXELS_PER_METER)  # 340 pixels
DIFFUSIONDRIVE_LIDAR_SIZE = 256  # DiffusionDrive expected LiDAR BEV size

# BEV Semantic Dimensions
# -----------------------
# Semantic BEV dimensions match DiffusionDrive's expected format
# Note: These are different from LiDAR BEV dimensions
BEV_SEMANTIC_HEIGHT = 128  # Height of semantic BEV map
BEV_SEMANTIC_WIDTH = 256   # Width of semantic BEV map

# Fixed Parameters
# ----------------
NUM_FUTURE_WAYPOINTS = 8  # Number of future trajectory points
MAX_AGENTS = 30  # Maximum number of agents to track
LIDAR_NORMALIZATION_FACTOR = 10.0  # Normalization factor for LiDAR histogram

# Command Mapping
# ---------------
# NavSim discrete command values
NAVSIM_CMD_LEFT = 0
NAVSIM_CMD_STRAIGHT = 1
NAVSIM_CMD_RIGHT = 2
NAVSIM_CMD_UNKNOWN = 3

# Temporal Parameters
# -------------------
BENCH2DRIVE_FREQUENCY_HZ = 10  # Bench2Drive recording frequency
DIFFUSIONDRIVE_FREQUENCY_HZ = 2  # DiffusionDrive training frequency
TEMPORAL_DOWNSAMPLE_RATE = 5  # Take every 5th frame (10Hz → 2Hz)