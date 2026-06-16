# Bench2Drive string class names to NavSim semantic classes
B2D_CLASS_TO_NAVSIM = {
    "vehicle": 5,  # All vehicles → NavSim class 5
    "walker": 6,  # Pedestrians → NavSim class 6
    "traffic_light": 4,  # Static objects → NavSim class 4
    "traffic_sign": 4,  # Static objects → NavSim class 4
    "ego_vehicle": -1,  # Skip ego vehicle
}

# BEV semantic map dimensions (from TransfuserConfig)
# B2D data is cropped to 64m to match DiffusionDrive model's spatial assumptions
# (model uses lidar_max_x/y=32, bev_pixel_size=0.25, agent head tanh*32)
# Front-half BEV is 128x256 pixels (crop rear half to match front camera coverage)
BEV_SEMANTIC_HEIGHT = 128  # Front-half only (full would be 256)
BEV_SEMANTIC_WIDTH = 256  # Full width
BEV_SEMANTIC_RANGE_M = 64.0  # Cropped to match DiffusionDrive model (256px * 0.25 m/px)
BEV_SEMANTIC_RESOLUTION = BEV_SEMANTIC_RANGE_M / BEV_SEMANTIC_WIDTH  # 64m / 256 pixels = 0.25m/pixel

# Agent tracking parameters
MAX_AGENTS = 30  # From num_bounding_boxes in TransfuserConfig

# Trajectory parameters
NUM_FUTURE_WAYPOINTS = 8  # From trajectory_sampling: 4s / 0.5s = 8 waypoints
FUTURE_TRAJECTORY_FRAME_STRIDE = 5  # Sample future trajectory every 5 frames (0.5s at 10Hz)
# This ensures ground truth trajectories are always at 2Hz regardless of input sampling rate
# Critical for evaluation: even when evaluating at 10Hz (sampling_rate=1), GT must be 2Hz

# LiDAR parameters
# B2D LiDAR is cropped to 64m to match model spatial assumptions (lidar_max_x/y=32)
BENCH2DRIVE_LIDAR_RANGE_M = 64.0  # Cropped to match DiffusionDrive model expectations
LIDAR_PIXELS_PER_METER = 4.0  # 4 pixels per meter resolution
BENCH2DRIVE_LIDAR_SIZE = int(BENCH2DRIVE_LIDAR_RANGE_M * LIDAR_PIXELS_PER_METER)  # 256x256
NAVSIM_LIDAR_RANGE_M = 64.0  # NavSim expects 64m range
DIFFUSIONDRIVE_LIDAR_SIZE = 256  # Target size for DiffusionDrive model (NavSim format)

# Bench2Drive lane types to NavSim BEV semantic classes
LANE_TYPE_TO_BEV_CLASS = {
    "Broken": 1,  # Broken line -> Road
    "Solid": 1,  # Solid line -> Road
    "SolidSolid": 1,  # Double solid -> Road
    "Center": 3,  # Center line -> Lane centerline
}

# Bench2Drive trigger types to NavSim BEV semantic classes
TRIGGER_TYPE_TO_BEV_CLASS = {
    "TrafficLight": 4,  # Traffic light -> Static object
    "StopSign": 4,  # Stop sign -> Static object
}

