# Bench2Drive string class names to NavSim semantic classes
B2D_CLASS_TO_NAVSIM = {
    "vehicle": 5,  # All vehicles → NavSim class 5
    "walker": 6,  # Pedestrians → NavSim class 6
    "traffic_light": 4,  # Static objects → NavSim class 4
    "traffic_sign": 4,  # Static objects → NavSim class 4
    "ego_vehicle": -1,  # Skip ego vehicle
}

# BEV semantic map dimensions (from TransfuserConfig)
# For B2D: Full BEV covers 85m x 85m with 256x256 pixels
# Front-half BEV is 128x256 pixels (crop rear half to match front camera coverage)
BEV_SEMANTIC_HEIGHT = 128  # Front-half only (full would be 256)
BEV_SEMANTIC_WIDTH = 256  # Full width
BEV_SEMANTIC_RANGE_M = 85.0  # BEV covers same range as B2D LiDAR
BEV_SEMANTIC_RESOLUTION = BEV_SEMANTIC_RANGE_M / 256  # 85m / 256 pixels = 0.332m/pixel

# Agent tracking parameters
MAX_AGENTS = 30  # From num_bounding_boxes in TransfuserConfig

# Trajectory parameters
NUM_FUTURE_WAYPOINTS = 8  # From trajectory_sampling: 4s / 0.5s = 8 waypoints

# LiDAR parameters
BENCH2DRIVE_LIDAR_RANGE_M = 85.0  # 85 meters range for Bench2Drive LiDAR (actual B2D data)
LIDAR_PIXELS_PER_METER = 4.0  # 4 pixels per meter resolution
BENCH2DRIVE_LIDAR_SIZE = int(BENCH2DRIVE_LIDAR_RANGE_M * LIDAR_PIXELS_PER_METER)  # 340x340
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

