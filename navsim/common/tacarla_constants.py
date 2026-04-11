"""
TaCarla dataset constants.
"""

import numpy as np

# Frame rate
TACARLA_NATIVE_HZ = 10
TACARLA_TRAINING_HZ = 2
TACARLA_DOWNSAMPLE_STRIDE = 5  # 10Hz / 2Hz

# Camera resolution (NuScenes-style)
TACARLA_CAMERA_HEIGHT = 900
TACARLA_CAMERA_WIDTH = 1600

# Camera names as they appear in Parquet columns and tar.gz paths
TACARLA_RGB_CAMERAS = ["front", "front_left", "front_right", "back", "back_left", "back_right"]

# Map TaCarla camera names to tar.gz subdirectory names
TACARLA_CAMERA_TAR_DIRS = {
    "front": "detection/rgb_camera/front",
    "front_left": "detection/rgb_camera/front_left",
    "front_right": "detection/rgb_camera/front_right",
    "back": "detection/rgb_camera/back",
    "back_left": "detection/rgb_camera/back_left",
    "back_right": "detection/rgb_camera/back_right",
}

# File naming pattern in tar.gz: {camera_name}_{frame_idx}_.jpg
# e.g. front_0_.jpg, front_left_100_.jpg

# LiDAR
TACARLA_LIDAR_RANGE_M = 64.0
TACARLA_LIDAR_PIXELS_PER_METER = 4.0
TACARLA_LIDAR_SIZE = 256

# BEV semantic map from bev_label_image
# 200x200 RGB, ~0.5 m/pixel, ego-centered at (100, 101)
TACARLA_BEV_ORIGINAL_SIZE = 200
TACARLA_BEV_APPROX_RESOLUTION = 0.5  # m/pixel (approximate)
TACARLA_BEV_APPROX_RANGE = 100.0  # meters total coverage (approximate)

# BEV RGB to class mapping
TACARLA_BEV_RGB_TO_CLASS = {
    (0, 0, 0): 0,        # background
    (0, 0, 255): 1,      # ego vehicle
    (0, 255, 0): 2,      # lane boundaries
    (0, 255, 255): 3,    # road / drivable area
    (255, 0, 0): 4,      # objects
    (255, 255, 255): 5,  # lane markings
}
TACARLA_NUM_BEV_CLASSES = 6

# Model input BEV dimensions (front-half, matching NAVSIM convention)
BEV_SEMANTIC_HEIGHT = 128
BEV_SEMANTIC_WIDTH = 256
BEV_SEMANTIC_RESOLUTION = 0.25  # m/pixel

# Trajectory
NUM_FUTURE_WAYPOINTS = 8
FUTURE_TRAJECTORY_FRAME_STRIDE = 5  # 10Hz frames per 0.5s interval

# Agent detection
MAX_AGENTS = 30
AGENT_DISTANCE_THRESHOLD_M = 32.0

# TaCarla vehicle type_id classification
# new_boxes 'class' field values
TACARLA_VEHICLE_CLASSES = {"Car", "Truck", "Van", "Bus", "Motorcycle", "Bicycle"}
TACARLA_PEDESTRIAN_CLASSES = {"Pedestrian"}

# Map to NAVSIM agent type IDs
TACARLA_CLASS_TO_NAVSIM = {
    "Car": 5,
    "Truck": 5,
    "Van": 5,
    "Bus": 5,
    "Motorcycle": 5,
    "Bicycle": 5,
    "Pedestrian": 6,
}

# Ego vehicle offset in TaCarla new_boxes
# The ego vehicle appears in new_boxes with position [-1.3, 0.0, -2.5]
# This is the sensor-to-rear-axle offset
TACARLA_EGO_OFFSET = np.array([-1.3, 0.0, -2.5])
