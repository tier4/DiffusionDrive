# Bench2Drive string class names to NavSim semantic classes
B2D_CLASS_TO_NAVSIM = {
    "vehicle": 5,  # All vehicles → NavSim class 5
    "walker": 6,  # Pedestrians → NavSim class 6
    "traffic_light": 4,  # Static objects → NavSim class 4
    "traffic_sign": 4,  # Static objects → NavSim class 4
    "ego_vehicle": -1,  # Skip ego vehicle
}

# BEV semantic map dimensions (from TransfuserConfig)
BEV_SEMANTIC_HEIGHT = 128  # lidar_resolution_height // 2 = 256 // 2
BEV_SEMANTIC_WIDTH = 256  # lidar_resolution_width

# Agent tracking parameters
MAX_AGENTS = 30  # From num_bounding_boxes in TransfuserConfig

# Trajectory parameters
NUM_FUTURE_WAYPOINTS = 8  # From trajectory_sampling: 4s / 0.5s = 8 waypoints

# LiDAR parameters
BENCH2DRIVE_LIDAR_RANGE_M = 64.0  # 64 meters range for Bench2Drive LiDAR
LIDAR_PIXELS_PER_METER = 4.0  # 4 pixels per meter resolution
BENCH2DRIVE_LIDAR_SIZE = int(BENCH2DRIVE_LIDAR_RANGE_M * LIDAR_PIXELS_PER_METER)  # 256x256
DIFFUSIONDRIVE_LIDAR_SIZE = 256  # Target size for DiffusionDrive model
