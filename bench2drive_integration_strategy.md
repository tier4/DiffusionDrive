# Bench2Drive Integration Strategy

## Overview

This document outlines implementation strategies for integrating Bench2Drive with DiffusionDrive. We present three methods based on different use cases and requirements.

## Method Selection Guide

| Your Goal | Recommended Method |
| :--- | :--- |
| Train on Bench2Drive, evaluate on CARLA only | **Method 3** (CARLA-Native) |
| Train on Bench2Drive, compare with NavSim models | **Method 2** (Full Adaptation) |
| Train on mixed datasets (Bench2Drive + NavSim) | **Method 2** (Full Adaptation) |
| One-time conversion, no live CARLA needed | **Method 1** (Data Conversion) |

## Method 1: Data Conversion (Not Recommended)

One-time conversion of Bench2Drive data to NavSim format. Requires separate pipeline for live CARLA evaluation.

## Method 2: Full Model Adaptation

Complete adaptation layer supporting both CARLA and NavSim coordinate systems with configurable transformations.

## Method 3: CARLA-Native Pipeline (Recommended for CARLA-only)

Simplified adaptation keeping everything in CARLA coordinates, perfect for CARLA-specific projects.

## Architecture Design

### 1. Core Components

```
navsim/
├── common/
│   ├── bench2drive_dataloader.py      # Bench2Drive-specific data loader
│   └── coordinate_transforms.py       # Coordinate transformation utilities
├── agents/diffusiondrive/
│   └── transfuser_features_b2d.py     # Bench2Drive feature builder
└── planning/
    ├── script/config/common/
    │   ├── train_test_split/
    │   │   └── bench2drive.yaml       # Dataset configuration
    │   └── agent/
    │       └── diffusiondrive_agent_b2d.yaml
    └── training/
        └── dataset_factory.py         # Factory for dataset selection
```

### 2. Class Hierarchy

```python
# Base classes from NavSim
AbstractSceneLoader
    └── SceneLoader (NavSim implementation)
    └── Bench2DriveSceneLoader (New implementation)

AbstractScene
    └── Scene (NavSim implementation)
    └── Bench2DriveScene (New implementation)

AbstractFeatureBuilder
    └── TransfuserFeatureBuilder (NavSim)
    └── Bench2DriveFeatureBuilder (New)
```

## Method 2: Full Implementation Details

### 1. Coordinate System Transformation

#### A. Position Transformation

```python
class CoordinateTransformer:
    """Handles coordinate transformations between CARLA and NavSim"""
    
    @staticmethod
    def carla_to_navsim_position(carla_pos: Dict[str, float]) -> np.ndarray:
        """
        Transform CARLA position to NavSim coordinate system
        
        CARLA (left-handed):  X-forward, Y-right, Z-up
        NavSim (right-handed): X-right, Y-forward, Z-up
        """
        return np.array([
            -carla_pos['y'],  # CARLA Y → NavSim -X
            carla_pos['x'],   # CARLA X → NavSim Y
            carla_pos['z']    # Z remains the same
        ])
    
    @staticmethod
    def carla_to_navsim_rotation(carla_rot: Dict[str, float]) -> float:
        """
        Transform CARLA rotation to NavSim
        
        CARLA: degrees, clockwise positive
        NavSim: radians, counter-clockwise positive
        """
        # Convert yaw from degrees to radians and flip direction
        return -np.radians(carla_rot['yaw'])
    
    @staticmethod
    def transform_velocity(velocity: Dict[str, float], yaw: float) -> np.ndarray:
        """Transform velocity from world frame to ego frame"""
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        vx_world = velocity['x']
        vy_world = velocity['y']
        
        # Rotate to ego frame
        vx_ego = vx_world * cos_yaw + vy_world * sin_yaw
        vy_ego = -vx_world * sin_yaw + vy_world * cos_yaw
        
        return np.array([vx_ego, vy_ego, 0.0])
```

#### B. Bounding Box Transformation

```python
def transform_bounding_box(carla_bbox: Dict) -> np.ndarray:
    """
    Transform CARLA bounding box to NavSim format
    
    Returns: [x, y, z, length, width, height, yaw]
    """
    location = carla_bbox['location']
    extent = carla_bbox['extent']
    rotation = carla_bbox['rotation']
    
    # Transform position
    pos = CoordinateTransformer.carla_to_navsim_position(location)
    
    # CARLA extent is half-size, NavSim expects full size
    size = [
        2 * extent['x'],  # length
        2 * extent['y'],  # width
        2 * extent['z']   # height
    ]
    
    # Transform rotation
    yaw = CoordinateTransformer.carla_to_navsim_rotation(rotation)
    
    return np.array([*pos, *size, yaw])
```

### 2. Sensor Data Mapping

#### A. Camera Configuration

```python
class CameraMapper:
    """Maps Bench2Drive cameras to NavSim configuration"""
    
    # NavSim expects 8 cameras, Bench2Drive has 6
    CAMERA_MAPPING = {
        'cam_f0': 'rgb_front',
        'cam_l0': 'rgb_front_left',
        'cam_l1': 'rgb_front_left',    # Duplicate (no side camera)
        'cam_l2': 'rgb_back_left',
        'cam_r0': 'rgb_front_right',
        'cam_r1': 'rgb_front_right',   # Duplicate (no side camera)
        'cam_r2': 'rgb_back_right',
        'cam_b0': 'rgb_back'
    }
    
    @classmethod
    def load_camera_data(cls, b2d_path: Path, frame_idx: str) -> Dict:
        """Load and map camera data"""
        cameras = {}
        
        for navsim_name, b2d_name in cls.CAMERA_MAPPING.items():
            img_path = b2d_path / 'camera' / b2d_name / f"{frame_idx}.jpg"
            
            if img_path.exists():
                # Load image (handle JPEG compression matching)
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Apply same JPEG compression as training to avoid train-val gap
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
                _, buffer = cv2.imencode('.jpg', img, encode_param)
                img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # Create placeholder
                img = np.zeros((900, 1600, 3), dtype=np.uint8)
            
            cameras[navsim_name] = img
            
        return cameras
```

#### B. LiDAR Data Processing

```python
class LidarProcessor:
    """Process Bench2Drive LiDAR data"""
    
    @staticmethod
    def load_lidar_data(lidar_path: Path) -> np.ndarray:
        """
        Load LAZ file and convert to NavSim format
        
        Returns: [N, 6] array (x, y, z, intensity, ring_index, lidar_id)
        """
        import laspy
        
        las = laspy.read(str(lidar_path))
        
        # Transform coordinates
        points_carla = np.vstack([las.x, las.y, las.z]).T
        points_navsim = np.apply_along_axis(
            CoordinateTransformer.carla_to_navsim_position,
            1,
            points_carla
        )
        
        # Build full point cloud array
        num_points = len(las.x)
        lidar_data = np.zeros((num_points, 6), dtype=np.float32)
        lidar_data[:, :3] = points_navsim
        lidar_data[:, 3] = las.intensity if hasattr(las, 'intensity') else 0
        lidar_data[:, 4] = 0  # ring_index (not in Bench2Drive)
        lidar_data[:, 5] = 0  # lidar_id (single LiDAR)
        
        return lidar_data
```

### 3. Annotation Processing

#### A. Command Mapping

```python
class CommandMapper:
    """Map Bench2Drive commands to NavSim format"""
    
    # NavSim uses simple commands: left=0, straight=1, right=2
    COMMAND_MAP = {
        # Bench2Drive command → NavSim command
        'CHANGELANELEFT': 0,
        'TURNLEFT': 0,
        'LEFT': 0,
        'STRAIGHT': 1,
        'LANEFOLLOW': 1,
        'CHANGELANERIGHT': 2,
        'TURNRIGHT': 2,
        'RIGHT': 2
    }
    
    @classmethod
    def map_command(cls, b2d_command: str) -> int:
        """Convert Bench2Drive command to NavSim format"""
        return cls.COMMAND_MAP.get(b2d_command.upper(), 1)
```

#### B. Actor Type Mapping

```python
class ActorTypeMapper:
    """Map CARLA actor types to NavSim categories"""
    
    TYPE_MAP = {
        # CARLA type → NavSim type
        'vehicle.car': 'vehicle',
        'vehicle.truck': 'vehicle',
        'vehicle.bus': 'vehicle',
        'vehicle.motorcycle': 'vehicle',
        'vehicle.bicycle': 'bicycle',
        'walker.pedestrian': 'pedestrian',
        'traffic.traffic_light': 'traffic_light',
        'traffic.stop_sign': 'traffic_cone',  # NavSim doesn't have stop signs
        'traffic.speed_limit': 'traffic_cone'
    }
    
    @classmethod
    def map_actor_type(cls, carla_type: str) -> str:
        """Map CARLA actor type to NavSim category"""
        for pattern, navsim_type in cls.TYPE_MAP.items():
            if carla_type.startswith(pattern):
                return navsim_type
        return 'generic_object'
```

### 4. Data Loader Implementation

```python
class Bench2DriveSceneLoader:
    """Scene loader for Bench2Drive dataset and live CARLA data"""
    
    def __init__(
        self,
        data_root: Path = None,
        carla_client = None,
        scene_filter: SceneFilter = None,
        sensor_config: SensorConfig = None,
        enable_caching: bool = True
    ):
        self.data_root = Path(data_root) if data_root else None
        self.carla_client = carla_client
        self.scene_filter = scene_filter
        self.sensor_config = sensor_config
        self.enable_caching = enable_caching
        self.cache = {}
        
        if self.data_root:
            self._load_scene_index()
        elif self.carla_client:
            self._setup_carla_connection()
    
    def _load_scene_index(self):
        """Build index of available scenes"""
        self.scenes = {}
        
        # Iterate through scenario directories
        for scenario_dir in self.data_root.iterdir():
            if not scenario_dir.is_dir():
                continue
                
            # Each scenario has multiple runs (Town_Weather_Route)
            for run_dir in scenario_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                # Check if annotation directory exists
                anno_dir = run_dir / 'anno'
                if not anno_dir.exists():
                    continue
                
                # Get all frame annotations
                frames = sorted(anno_dir.glob('*.json.gz'))
                
                # Apply temporal downsampling (10Hz → 2Hz)
                sampling_rate = 5  # Take every 5th frame
                sampled_frames = frames[::sampling_rate]
                
                # Create scenes with sliding window
                for i in range(len(sampled_frames) - self.scene_filter.num_frames + 1):
                    scene_frames = sampled_frames[i:i + self.scene_filter.num_frames]
                    
                    # Generate unique token
                    token = f"{scenario_dir.name}_{run_dir.name}_{i:05d}"
                    
                    self.scenes[token] = {
                        'scenario': scenario_dir.name,
                        'run': run_dir.name,
                        'frames': scene_frames,
                        'base_path': run_dir
                    }
    
    def get_scene_from_token(self, token: str) -> 'Bench2DriveScene':
        """Load scene by token"""
        # Check cache first
        if self.enable_caching and token in self.cache:
            return self.cache[token]
        
        scene_info = self.scenes[token]
        scene = Bench2DriveScene(
            scene_info=scene_info,
            scene_filter=self.scene_filter,
            sensor_config=self.sensor_config
        )
        
        if self.enable_caching:
            self.cache[token] = scene
            
        return scene
    
    def get_live_scene(self) -> 'Bench2DriveScene':
        """Get current scene from live CARLA simulation"""
        # Collect sensor data from CARLA
        sensor_data = self._collect_carla_sensors()
        
        # Transform to same format as offline data
        scene = Bench2DriveLiveScene(
            sensor_data=sensor_data,
            transformer=self.coordinate_transformer
        )
        
        return scene
```

### 5. Feature Builder Adaptation

```python
class Bench2DriveFeatureBuilder(AbstractFeatureBuilder):
    """Feature builder for Bench2Drive data"""
    
    def __init__(self, config: TransfuserConfig):
        super().__init__()
        self.config = config
        
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Build features from Bench2Drive agent input"""
        features = {}
        
        # Camera features (stitch 3 front cameras)
        front_cameras = [
            agent_input.cameras[-1].cam_l0,  # left
            agent_input.cameras[-1].cam_f0,  # front
            agent_input.cameras[-1].cam_r0   # right
        ]
        
        # Stitch cameras horizontally
        stitched_image = np.concatenate([cam.image for cam in front_cameras], axis=1)
        
        # Resize to expected dimensions (1024x256)
        stitched_image = cv2.resize(stitched_image, (1024, 256))
        
        # Normalize and convert to tensor
        features['camera'] = torch.from_numpy(stitched_image).float() / 255.0
        features['camera'] = features['camera'].permute(2, 0, 1)  # HWC → CHW
        
        # LiDAR features (create BEV histogram)
        lidar_bev = self._create_lidar_bev(agent_input.lidars[-1].lidar_pc)
        features['lidar'] = torch.from_numpy(lidar_bev).float()
        
        # Status features
        ego_status = agent_input.ego_statuses[-1]
        features['velocity'] = torch.tensor(ego_status.ego_velocity[:2]).float()
        # Convert integer command to one-hot if needed by model
driving_cmd = ego_status.driving_command  # Integer 0-3
features['command'] = torch.tensor(driving_cmd).long()
# Or if model expects one-hot:
# features['command_one_hot'] = F.one_hot(torch.tensor(driving_cmd), num_classes=4).float()
        
        return features
```

### 6. Configuration System

#### A. Dataset Configuration

```yaml
# config/common/train_test_split/bench2drive.yaml
_target_: navsim.common.dataclasses.SceneFilter
_convert_: all

log_names: null  # Use all available scenarios
tokens: null
num_frames: 30    # 15 seconds at 2Hz
num_history_frames: 4
num_future_frames: 26
frame_interval: 1  # Already downsampled in loader
has_route: true
max_scenes: null

# Bench2Drive specific settings
bench2drive:
  data_root: ${oc.env:BENCH2DRIVE_ROOT}
  scenarios:
    train: [
      "ConstructionCrash",
      "CutIn", 
      "FollowLeadingVehicle",
      # ... more scenarios
    ]
    val: [
      "NoSignalJunction",
      "ParkingExit",
      # ... validation scenarios
    ]
  sampling_rate: 5  # Downsample from 10Hz to 2Hz
```

#### B. Agent Configuration

```yaml
# config/common/agent/diffusiondrive_agent_b2d.yaml
_target_: navsim.agents.diffusiondrive.transfuser_agent.TransfuserAgent

checkpoint_path: null
config:
  _target_: navsim.agents.diffusiondrive.transfuser_config.TransfuserConfig
  _convert_: all
  
  # Use Bench2Drive feature builder
  feature_builder:
    _target_: navsim.agents.diffusiondrive.transfuser_features_b2d.Bench2DriveFeatureBuilder
    
  # Rest of configuration remains the same
  backbone: "resnet34"
  n_views: 3
  # ...
```

### 7. Dataset Factory Pattern

```python
class DatasetFactory:
    """Factory for creating appropriate dataset based on configuration"""
    
    @staticmethod
    def create_dataset(
        cfg: DictConfig,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder]
    ) -> Dataset:
        """Create dataset based on configuration"""
        
        dataset_type = cfg.get('dataset_type', 'navsim')
        
        if dataset_type == 'bench2drive':
            # Create Bench2Drive scene loader
            scene_loader = Bench2DriveSceneLoader(
                data_root=Path(cfg.bench2drive.data_root),
                scene_filter=instantiate(cfg.scene_filter),
                sensor_config=instantiate(cfg.sensor_config)
            )
        else:
            # Default NavSim scene loader
            scene_loader = SceneLoader(
                data_path=Path(cfg.data_path),
                sensor_blobs_path=Path(cfg.sensor_blobs_path),
                scene_filter=instantiate(cfg.scene_filter),
                sensor_config=instantiate(cfg.sensor_config)
            )
        
        # Create dataset with appropriate loader
        return Dataset(
            scene_loader=scene_loader,
            feature_builders=feature_builders,
            target_builders=target_builders,
            cache_path=cfg.get('cache_path'),
            force_cache_computation=cfg.get('force_cache_computation', False)
        )
```

### 8. Usage Example

```bash
# Training with Bench2Drive dataset
python navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent_b2d \
    train_test_split=bench2drive \
    dataset_type=bench2drive \
    experiment_name=diffusiondrive_bench2drive \
    trainer.params.max_epochs=100

# Training with original NavSim dataset (default)
python navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    train_test_split=navtrain \
    experiment_name=diffusiondrive_navsim \
    trainer.params.max_epochs=100
```

## Method 3: CARLA-Native Implementation (Simplified)

### Overview

Method 3 is a simplified version of Method 2, designed for when you're training and evaluating exclusively within the CARLA ecosystem. No coordinate transformations are needed.

### Key Simplifications

```python
class CARLANativeLoader(Bench2DriveSceneLoader):
    """Simplified loader for CARLA-only training and evaluation"""
    
    def __init__(self, data_root: Path = None, carla_client = None):
        super().__init__(data_root, carla_client)
        # No coordinate transformer needed
        self.transform_coordinates = False
    
    def _process_frame(self, b2d_anno: Dict) -> Dict:
        """Process Bench2Drive frame without coordinate transformation"""
        return {
            'token': self._generate_token(),
            # Keep CARLA coordinates as-is
            'ego2global_translation': [b2d_anno['x'], b2d_anno['y'], b2d_anno.get('z', 0)],
            'ego2global_rotation': self._euler_to_quaternion(b2d_anno['theta']),  # Just format conversion
            'ego_dynamic_state': [
                b2d_anno['speed'] * np.cos(np.radians(b2d_anno['theta'])),
                b2d_anno['speed'] * np.sin(np.radians(b2d_anno['theta'])),
                b2d_anno['acceleration']['x'],
                b2d_anno['acceleration']['y']
            ],
            'driving_command': self._simplify_command(b2d_anno['command_near']),
        }
    
    def _process_actors(self, actors: List[Dict]) -> List[Dict]:
        """Process actor annotations without coordinate transformation"""
        processed_actors = []
        
        for actor in actors:
            # Keep CARLA coordinates and units
            processed_actor = {
                'translation': [actor['location']['x'], actor['location']['y'], actor['location']['z']],
                'size': [2 * actor['extent']['x'], 2 * actor['extent']['y'], 2 * actor['extent']['z']],
                'rotation': np.radians(actor['rotation']['yaw']),  # Only convert degrees to radians
                'velocity': [actor['velocity']['x'], actor['velocity']['y'], 0],
                'type': self._map_actor_type(actor['type_id'])
            }
            processed_actors.append(processed_actor)
            
        return processed_actors
```

### What Still Needs Adaptation

Even without coordinate transformation, you still need to:

1. **Data Structure Mapping**
   - Convert Bench2Drive's file structure to DiffusionDrive's expected format
   - Create scene tokens and frame groupings
   - Handle pickle serialization

2. **Sensor Mapping**
   - Map 6 CARLA cameras to 8 expected cameras (with duplicates)
   - Convert LAZ LiDAR files to numpy arrays
   - Match image dimensions and formats

3. **Feature Extraction**
   - Stitch front cameras for wide-view input
   - Create BEV representations from point clouds
   - Format velocity and command inputs

4. **Command Simplification**

   ```python
   def _simplify_command(self, carla_command: str) -> int:
       """Convert complex CARLA commands to NavSim discrete commands
       
       NavSim commands represent desired route only:
       0 = left (turns, lane changes, sharp curves)
       1 = straight
       2 = right (turns, lane changes, sharp curves)
       3 = unknown (can be filtered during training)
       """
       command_map = {
           'CHANGELANELEFT': 0, 'TURNLEFT': 0, 'LEFT': 0,
           'STRAIGHT': 1, 'LANEFOLLOW': 1,
           'CHANGELANERIGHT': 2, 'TURNRIGHT': 2, 'RIGHT': 2,
           'STOP': 3, 'UNKNOWN': 3
       }
       return command_map.get(carla_command.upper(), 1)
   ```

### Configuration for Method 3

```yaml
# config/common/train_test_split/bench2drive_carla_native.yaml
dataset_type: bench2drive_carla_native
transform_coordinates: false  # Key difference
coordinate_system: carla      # Stay in CARLA coordinates

# Everything else remains similar
bench2drive:
  data_root: ${oc.env:BENCH2DRIVE_ROOT}
  sampling_rate: 5  # 10Hz → 2Hz
```

### Advantages of Method 3

1. **Simplicity**: No complex coordinate transformations
2. **Performance**: Faster data loading without transformations
3. **Debugging**: Easier to verify data correctness
4. **Live CARLA**: Perfect alignment between training and deployment

### Limitations

1. **No NavSim Compatibility**: Cannot compare with NavSim-trained models
2. **No Mixed Training**: Cannot combine with NavSim datasets
3. **Metric Adaptation**: Some evaluation metrics may expect NavSim coordinates

## Testing and Validation

### 1. Coordinate Transformation Tests

```python
def test_coordinate_transformation():
    """Verify coordinate transformations are correct"""
    
    # Test position transformation
    carla_pos = {'x': 10.0, 'y': 5.0, 'z': 2.0}
    navsim_pos = CoordinateTransformer.carla_to_navsim_position(carla_pos)
    assert np.allclose(navsim_pos, [-5.0, 10.0, 2.0])
    
    # Test rotation transformation
    carla_rot = {'yaw': 90.0}  # 90 degrees clockwise
    navsim_yaw = CoordinateTransformer.carla_to_navsim_rotation(carla_rot)
    assert np.isclose(navsim_yaw, -np.pi/2)  # -90 degrees in radians
```

### 2. Visualization Tools

```python
def visualize_transformation(b2d_scene, navsim_scene):
    """Visualize coordinate transformations for validation"""
    
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot Bench2Drive coordinates
    ax1.set_title("Bench2Drive (CARLA)")
    ax1.set_xlabel("X (Forward)")
    ax1.set_ylabel("Y (Right)")
    
    # Plot NavSim coordinates
    ax2.set_title("NavSim")
    ax2.set_xlabel("X (Right)")
    ax2.set_ylabel("Y (Forward)")
    
    # Plot ego vehicle and other actors
    # ... plotting code ...
```

## Performance Considerations

### 1. Caching Strategy

- Cache transformed annotations in memory
- Lazy load sensor data only when needed
- Pre-compute frequently used transformations

### 2. Parallel Processing

- Use multiprocessing for loading multiple scenes
- Batch coordinate transformations using numpy
- Implement async sensor data loading

### 3. Memory Management

- Stream large sensor files (LiDAR, images)
- Use memory-mapped files for frequently accessed data
- Implement LRU cache for transformed data

## Migration Path

### Phase 1: Core Infrastructure (1-2 weeks)

1. Implement coordinate transformation utilities
2. Create Bench2DriveSceneLoader with caching support
3. Adapt feature builders for Bench2Drive format
4. Add configuration files for dataset switching

### Phase 2: Training Integration (1 week)

1. Integrate with existing training pipeline
2. Implement efficient caching for transformed data
3. Validate data loading and transformations
4. Performance optimization

### Phase 3: Live CARLA Support (2-3 weeks)

1. Extend loader for CARLA client connection
2. Implement real-time sensor data collection
3. Create evaluation harness for closed-loop control
4. Test live inference and vehicle control

### Phase 4: Validation & Testing (1 week)

1. Unit tests for all transformations
2. End-to-end training validation
3. Live CARLA evaluation testing
4. Documentation and examples

## Advantages Over Method 1

1. **Live CARLA Compatible**: Essential for real-time vehicle control in simulator
2. **Unified Pipeline**: Same code for training and live evaluation
3. **No Data Duplication**: Original data remains untouched
4. **Flexible Caching**: Can cache transformed data when needed for performance
5. **Maintainability**: Transformations in code are easier to debug and update
6. **Version Control**: Code changes are tracked, data remains stable
7. **Faster Iteration**: No need to re-convert entire dataset for fixes

## Key Implementation Considerations

### Caching Strategy

- Cache transformed data during first epoch for training efficiency
- Use memory-mapped files for large sensor data
- Implement cache warming before training starts

### Live CARLA Integration

```python
# Example usage for live evaluation
carla_client = carla.Client('localhost', 2000)
loader = Bench2DriveSceneLoader(carla_client=carla_client)

# In evaluation loop
while True:
    scene = loader.get_live_scene()
    features = feature_builder.compute_features(scene.get_agent_input())
    action = model.predict(features)
    carla_client.apply_control(action)
```

## Conclusion

We've presented three implementation strategies for integrating Bench2Drive with DiffusionDrive:

### Method 1: Data Conversion

- **Use case**: One-time experiments without live evaluation
- **Pros**: Works with existing tools
- **Cons**: No live CARLA support, data duplication

### Method 2: Full Model Adaptation

- **Use case**: Research requiring NavSim compatibility or mixed datasets
- **Pros**: Most flexible, supports all scenarios
- **Cons**: Most complex implementation

### Method 3: CARLA-Native Pipeline (Recommended for CARLA-only)

- **Use case**: Training and evaluating exclusively on CARLA data
- **Pros**: Simplest implementation, best performance, perfect train-eval alignment
- **Cons**: Limited to CARLA ecosystem

For most CARLA-focused projects, **Method 3 provides the best balance** of simplicity and functionality while maintaining perfect alignment between training and live evaluation.
