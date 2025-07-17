# Data Mapping and Model Input Guide

This comprehensive guide explains how raw data is mapped and formatted for the DiffusionDrive model, covering driving commands, semantic category mapping, and model input requirements.

## Driving Commands

### Command Format and Data Flow

**Key Point**: Commands are stored as integers in the data structure but **the model expects one-hot encoded vectors**.

#### 1. Storage Format

- **Type**: Single integer value (0, 1, 2, or 3)
- **Location in code**: `navsim/common/dataclasses.py`, line 138
- **Access path**: `agent_input.ego_statuses[-1].driving_command`

#### 2. Model Input Format

The DiffusionDrive model's status encoding layer (`transfuser_model_v2.py`) uses:

```python
self._status_encoding = nn.Linear(config.num_driving_commands + 2 + 2, config.tf_d_model)
```

This requires **one-hot encoded input** where:

- `config.num_driving_commands`: Number of one-hot command dimensions (default: 4)
- `+ 2`: Velocity components (vx, vy)
- `+ 2`: Acceleration components (ax, ay)

#### 3. Command Values and Meanings

According to official NAVSIM documentation:

- **`0` = LEFT** (turns, lane changes left, sharp left curves)
- **`1` = STRAIGHT** (continue straight)
- **`2` = RIGHT** (turns, lane changes right, sharp right curves)
- **`3` = UNKNOWN** (can be filtered during training)

**Important Notes**:

- Commands are based solely on the desired route
- They do NOT include information about obstacles or traffic signs (unlike nuScenes)
- Left and right commands cover all maneuvers including turns, lane changes, and sharp curves
- The 'unknown' command can be used to filter out ambiguous data during training

#### 4. Data Conversion Process

```python
# Integer to one-hot conversion (required for model)
driving_cmd = agent_input.ego_statuses[-1].driving_command  # Integer 0-3
driving_cmd_one_hot = F.one_hot(torch.tensor(driving_cmd), num_classes=4).float()

# Complete status feature construction
features["status_feature"] = torch.concatenate([
    driving_cmd_one_hot,  # 4D one-hot encoded
    torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),     # 2D
    torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32), # 2D
])
```

#### 5. Bench2Drive Command Mapping

When converting from CARLA/Bench2Drive commands to NAVSIM format:

```python
def carla_to_navsim_command(carla_command: str) -> int:
    """Convert CARLA command string to NAVSIM integer command"""
    
    command_map = {
        # Left commands (turns, lane changes, sharp curves)
        'CHANGELANELEFT': 0,
        'TURNLEFT': 0,
        'LEFT': 0,
        
        # Straight commands
        'STRAIGHT': 1,
        'LANEFOLLOW': 1,
        
        # Right commands (turns, lane changes, sharp curves)
        'CHANGELANERIGHT': 2,
        'TURNRIGHT': 2,
        'RIGHT': 2,
        
        # Unknown (can be filtered during training)
        'STOP': 3,
        'UNKNOWN': 3
    }
    
    return command_map.get(carla_command.upper(), 1)  # Default to straight
```

## Semantic Category Mapping

### Overview

This section covers the mapping strategy between CARLA's 23 semantic classes and DiffusionDrive's 7 BEV semantic classes.

### Analysis Results (Bench2Drive Base Dataset)

**Analysis of 13,770 semantic images from Bench2Drive base dataset:**

- **Total unique pixel values found**: 27 values
- **Actual values present**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
- **Missing values**: [16, 17] (Bus=16, Train=17)

### Mapping Validation Status

✅ **Present in dataset**: Unlabeled(0), Roads(1), SideWalks(2), Building(3), Wall(4), Fence(5), Pole(6), TrafficLight(7), TrafficSign(8), Vegetation(9), Terrain(10), Sky(11), Pedestrian(12), Rider(13), Car(14), Truck(15), Motorcycle(18), Bicycle(19), Static(20), Dynamic(21), Other(22), Water(23), RoadLine(24), Ground(25), Bridge(26), RailTrack(27), GuardRail(28)

❌ **Missing in dataset**: Bus(16), Train(17)

### BEV Semantic Map Classes (7 Classes)

#### 0: Background

Elements that are generally not interactive or are part of the distant environment.

- **Semantic Tags**: `Unlabeled` (0), `Building` (3), `Vegetation` (9), `Terrain` (10), `Sky` (11), `Other` (22), `Water` (23), `Bridge` (26), `RailTrack` (27)

#### 1: Road (including lanes and intersections)

All drivable surfaces and markings, excluding specific centerlines.

- **BEV/HD-Map Categories**: All Lane Marking Types (Broken, Solid, SolidSolid, Other, NONE) and Colors (White, Yellow, Blue). Also includes crosswalks.
- **Semantic Tags**: `Roads` (1), `RoadLine` (24), `Ground` (25)

#### 2: Walkways

Areas designated for pedestrians.

- **Semantic Tags**: `SideWalks` (2)

#### 3: Lane centerlines

Annotated center of driving lanes.

- **BEV/HD-Map Categories**: Lane markings where Type == 'Center'. Various Topology statuses (e.g., Junction, Normal) are attributes of these centerlines.

#### 4: Static objects (like barriers and signs)

Non-actor obstacles, traffic control infrastructure, and roadside furniture.

- **BEV/HD-Map Categories**: Trigger Volumes for StopSign and TrafficLight
- **Anno Structure**: traffic_light, traffic_sign
- **Semantic Tags**: `Wall` (4), `Fence` (5), `Pole` (6), `TrafficLight` (7), `TrafficSign` (8), `Static` (20), `Dynamic` (21), `GuardRail` (28)

#### 5: Vehicles

All types of vehicles, including the ego vehicle.

- **Anno Structure**: ego_vehicle, vehicle
- **Semantic Tags**: `Car` (14), `Truck` (15), `Bus` (16), `Train` (17), `Motorcycle` (18), `Bicycle` (19)
- **Note**: Bus(16) and Train(17) are missing from the dataset but can be handled by grouping with other vehicles

#### 6: Pedestrians

All human actors.

- **Anno Structure**: pedestrian (class 'walker')
- **Semantic Tags**: `Pedestrian` (12), `Rider` (13)

### Implementation Code

```python
# CARLA to BEV mapping (based on semantic analysis)
CARLA_TO_BEV_MAPPING = {
    # Background (0)
    0: 0,   # Unlabeled
    3: 0,   # Building
    9: 0,   # Vegetation
    10: 0,  # Terrain
    11: 0,  # Sky
    22: 0,  # Other
    23: 0,  # Water
    26: 0,  # Bridge
    27: 0,  # RailTrack
    
    # Road (1)
    1: 1,   # Roads
    24: 1,  # RoadLine
    25: 1,  # Ground
    
    # Walkways (2)
    2: 2,   # SideWalks
    
    # Lane centerlines (3) - requires custom detection
    24: 3,  # RoadLine (when centerline)
    
    # Static objects (4)
    4: 4,   # Wall
    5: 4,   # Fence
    6: 4,   # Pole
    7: 4,   # TrafficLight
    8: 4,   # TrafficSign
    20: 4,  # Static
    21: 4,  # Dynamic
    28: 4,  # GuardRail
    
    # Vehicles (5)
    14: 5,  # Car
    15: 5,  # Truck
    # 16: 5,  # Bus (missing in dataset)
    # 17: 5,  # Train (missing in dataset)
    18: 5,  # Motorcycle
    19: 5,  # Bicycle
    
    # Pedestrians (6)
    12: 6,  # Pedestrian
    13: 6,  # Rider
}
```

### Important Findings

#### BEV Semantic Map Limitations

- **Bench2Drive has NO `semantic_top_down` views** - only `rgb_top_down`
- Current implementation uses **placeholder BEV semantic maps**
- This limits the effectiveness of DiffusionDrive's BEV semantic auxiliary task (loss weight: 14.0)
- Need to generate BEV semantic maps from available data (perspective semantic views, LiDAR, annotations)

#### Implications for BEV Semantic Map Generation

- All major categories (Road, Walkways, Vehicles, Pedestrians, Static objects) are present
- Missing Bus/Train categories can be handled by grouping with other vehicles
- Current 7-class BEV mapping should work with available semantic data

## Extending Model Capabilities

### Supporting More Driving Commands

To extend DiffusionDrive to support more than 4 driving commands:

#### 1. Configuration Updates

```python
# In navsim/agents/diffusiondrive/transfuser_config.py
@dataclass
class TransfuserConfig:
    num_driving_commands: int = 8  # Change from 4 to desired number
```

#### 2. Model Architecture Updates

```python
# In navsim/agents/diffusiondrive/transfuser_model_v2.py
# Change from:
self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

# To:
self._status_encoding = nn.Linear(config.num_driving_commands + 2 + 2, config.tf_d_model)
```

#### 3. Data Preprocessing Updates

```python
def convert_command_to_onehot(command_id: int, num_commands: int) -> np.ndarray:
    """Convert integer command to one-hot encoding."""
    onehot = np.zeros(num_commands, dtype=np.float32)
    if 0 <= command_id < num_commands:
        onehot[command_id] = 1.0
    return onehot
```

#### 4. Example: Carla Command System

Carla command system

### Alternative: Continuous Commands

Instead of discrete one-hot encoding, you can use continuous values:

```python
# Use 2D continuous command (angle, magnitude)
driving_command = np.array([angle, magnitude])  # 2D instead of one-hot

# Update model accordingly
self._status_encoding = nn.Linear(2 + 2 + 2, config.tf_d_model)  # 2 for command + 2 vel + 2 acc
```

## Validation and Testing

### Command Validation

```python
# Verify one-hot encoding
def validate_command_encoding(status_feature, num_commands):
    driving_cmd = status_feature[:, :num_commands]
    assert torch.allclose(driving_cmd.sum(dim=1), torch.ones(batch_size))
    print("✅ Command encoding validation passed")

# Check data shape consistency
def validate_data_shapes(features, num_commands):
    expected_shape = (num_commands + 4,)  # commands + velocity + acceleration
    actual_shape = features['status_feature'].shape
    assert actual_shape == expected_shape, f"Expected {expected_shape}, got {actual_shape}"
    print("✅ Data shape validation passed")
```

### Semantic Mapping Validation

```python
def validate_semantic_mapping(semantic_image, mapping_dict):
    """Validate that all semantic values can be mapped to BEV classes"""
    unique_values = np.unique(semantic_image)
    unmapped_values = []
    
    for value in unique_values:
        if value not in mapping_dict:
            unmapped_values.append(value)
    
    if unmapped_values:
        print(f"⚠️ Warning: Unmapped semantic values found: {unmapped_values}")
    else:
        print("✅ All semantic values have valid mappings")
```

## Usage Examples

### Accessing Driving Commands

```python
# Access driving command from agent input
driving_cmd = agent_input.ego_statuses[-1].driving_command
print(f"Current command: {driving_cmd}")

# Interpret the command
command_names = ["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]
if 0 <= driving_cmd < len(command_names):
    action = command_names[driving_cmd]
else:
    action = "Invalid command"
print(f"Action: {action}")

# Convert to one-hot for model input
driving_cmd_one_hot = F.one_hot(torch.tensor(driving_cmd), num_classes=4)
```

### Processing Semantic Data

```python
# Process semantic image to BEV format
def process_semantic_to_bev(semantic_image, mapping_dict):
    """Convert semantic image to BEV format using mapping dictionary"""
    bev_image = np.zeros_like(semantic_image)
    
    for carla_class, bev_class in mapping_dict.items():
        bev_image[semantic_image == carla_class] = bev_class
    
    return bev_image

# Apply mapping
bev_semantic = process_semantic_to_bev(semantic_image, CARLA_TO_BEV_MAPPING)
```

## Important Notes

1. **Data Consistency**: Ensure all data uses the same number of commands consistently across training and evaluation
2. **Backward Compatibility**: Pre-trained models with 4 commands won't work with modified architecture
3. **BEV Semantic Maps**: Current implementation uses placeholder BEV maps - real BEV generation is needed for optimal performance
4. **Command Interpretation**: Commands are route-based, not reactive (unlike nuScenes)
5. **Missing Categories**: Bus and Train categories are missing from Bench2Drive but can be grouped with other vehicles
6. **LiDAR Coverage**: Bench2Drive provides 85m x 85m coverage, which is processed at full resolution then resized to DiffusionDrive's expected 64m x 64m
7. **Dimension Standards**:
   - LiDAR BEV: 256 x 256 (single channel, square coverage)
   - Semantic BEV: 128 x 256 (H x W, **native NavSim format matching frontal RGB camera coverage**)
   - Camera: 1024 x 256 (W x H, 3 stitched frontal cameras)
   - **Key Insight**: The 128×256 BEV semantic format is the original NavSim design, not a DiffusionDrive modification. This rectangular shape optimizes for forward-driving scenarios where lateral awareness is more critical than forward/backward range.

## References

- Original driving command analysis: `driving_command_analysis.md`
- Command extension guide: `adaptation_guide_more_commands.md`
- Category mapping strategy: `becn2drive_category_mapping_strategy.md`
- BEV implementation plan: `BEV_SEGMENTATION_GENERATION_PLAN.md`
- NAVSIM Documentation: <https://navsim.readthedocs.io/>
- CARLA Documentation: <https://carla.readthedocs.io/>
