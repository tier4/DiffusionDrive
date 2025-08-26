# DiffusionDrive CARLA Integration Plan for Bench2Drive

## Overview

This document outlines the implementation plan for integrating DiffusionDrive model trained on Bench2Drive dataset into CARLA for closed-loop evaluation and video generation. The focus is on creating a custom agent that can run in CARLA simulator without requiring open-loop PDM evaluation.

## Objectives

1. Create a CARLA-compatible agent for DiffusionDrive model inference
2. Enable closed-loop evaluation in CARLA environment
3. Support automatic video generation during evaluation
4. Minimize changes to existing codebase

## Implementation Architecture

### Phase 1: CARLA Agent Development

#### 1.1 Create DiffusionDrive B2D Agent

**File:** `Bench2DriveZoo/team_code/diffusiondrive_b2d_agent.py`

**Key Components:**

```python
class DiffusionDriveAgent(autonomous_agent.AutonomousAgent):
    """
    CARLA agent for DiffusionDrive model trained on Bench2Drive.
    Processes sensor data and outputs control commands.
    """
    
    def setup(self, path_to_conf_file):
        # Parse config: "config_path+checkpoint_path+save_name"
        # Load DiffusionDrive model from checkpoint
        # Initialize PID controller
        # Set up camera transformation matrices
        
    def sensors(self):
        # Define 6 cameras + GPS + IMU + speedometer
        # Match Bench2Drive sensor configuration
        
    def tick(self, input_data):
        # Collect sensor data
        # Process GPS/IMU for ego status
        # Store for inference
        
    def run_step(self, input_data, timestamp):
        # Process camera images (stitch 3 front cameras)
        # Convert to DiffusionDrive input format
        # Run model inference
        # Convert trajectory to waypoints
        # PID controller to control commands
        # Save data if SAVE_PATH is set
```

**Sensor Processing:**

- **Cameras:** 6 RGB cameras (front, front_left, front_right, back, back_left, back_right)
  - Resolution: 1600×900 per camera
  - Stitch 3 front cameras to 1024×256 for model input
- **GPS/IMU:** Convert to ego position and orientation
- **Speedometer:** Extract velocity information

**Model Inference Pipeline:**

1. Load trained DiffusionDrive checkpoint (without MMCV dependencies)
2. Preprocess sensor data to match training format
3. Build features using `Bench2DriveFeatureBuilder`
4. Run diffusion model to generate trajectory
5. Post-process trajectory to waypoints

#### 1.2 Adapt Model Loading

**Challenge:** Remove MMCV dependencies while preserving model functionality

**Solution:**

```python
# Direct PyTorch model loading
import torch
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

class ModelWrapper:
    def __init__(self, checkpoint_path, config_path):
        # Load config
        self.config = TransfuserConfig()
        
        # Initialize model
        self.model = V2TransfuserModel(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        # Remove 'agent.' prefix if present
        state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.cuda()
        self.model.eval()
    
    def inference(self, features):
        with torch.no_grad():
            trajectory = self.model.compute_trajectory(features)
        return trajectory
```

### Phase 2: Feature Processing

#### 2.1 Sensor Data Conversion

**Adapt Bench2Drive features for CARLA input:**

```python
def process_carla_sensors(self, sensor_data):
    """Convert CARLA sensor data to DiffusionDrive format."""
    
    # Camera processing
    camera_images = self.stitch_front_cameras(
        sensor_data['rgb_front'],
        sensor_data['rgb_front_left'], 
        sensor_data['rgb_front_right']
    )
    camera_feature = self.normalize_images(camera_images)
    
    # LiDAR processing (if available, otherwise use placeholder)
    lidar_feature = self.process_lidar(sensor_data.get('lidar'))
    
    # Status processing
    status_feature = self.process_ego_status(
        sensor_data['gps'],
        sensor_data['imu'],
        sensor_data['speedometer']
    )
    
    return {
        'camera_feature': camera_feature,
        'lidar_feature': lidar_feature,
        'status_feature': status_feature
    }
```

#### 2.2 Coordinate System Alignment

**Ensure consistency between CARLA and training coordinate systems:**

- CARLA uses left-handed coordinate system (X-forward, Y-right, Z-up)
- Bench2Drive training uses the same system (native CARLA data)
- No coordinate transformation needed

### Phase 3: Trajectory Planning and Control

#### 3.1 Trajectory Post-processing

```python
def trajectory_to_waypoints(self, trajectory, ego_transform):
    """Convert model trajectory output to CARLA waypoints."""
    
    # Trajectory shape: (20 modes, 8 timesteps, 2)
    # Select best mode (e.g., mode with highest confidence)
    best_trajectory = trajectory[0]  # or use scoring mechanism
    
    # Convert to world coordinates
    waypoints = []
    for t in range(len(best_trajectory)):
        # Model outputs relative positions at 0.5s intervals
        rel_x, rel_y = best_trajectory[t]
        
        # Transform to world coordinates
        world_point = ego_transform.transform(
            carla.Location(x=rel_x, y=rel_y, z=0)
        )
        waypoints.append(world_point)
    
    return waypoints
```

#### 3.2 PID Control

```python
def waypoints_to_control(self, waypoints, current_speed):
    """Convert waypoints to vehicle control commands."""
    
    # Use existing PID controller from team_code
    control = self.pid_controller.run_step(
        waypoints=waypoints,
        current_speed=current_speed
    )
    
    return control
```

### Phase 4: Data Saving and Video Generation

#### 4.1 Automatic Data Collection

```python
def save_evaluation_data(self, sensor_data, control, step):
    """Save data for video generation when SAVE_PATH is set."""
    
    if self.save_path is None:
        return
    
    # Save every 10 steps (matching VAD agent)
    if step % 10 == 0:
        # Save RGB images
        for camera_name, image in sensor_data.items():
            if 'rgb' in camera_name:
                save_dir = self.save_path / camera_name
                save_dir.mkdir(exist_ok=True)
                image.save(save_dir / f'{step:05d}.png')
        
        # Save BEV visualization (if generated)
        if hasattr(self, 'bev_map'):
            bev_dir = self.save_path / 'bev'
            bev_dir.mkdir(exist_ok=True)
            self.save_bev(bev_dir / f'{step:05d}.png')
        
        # Save control metadata
        meta = {
            'step': step,
            'control': {
                'steer': control.steer,
                'throttle': control.throttle,
                'brake': control.brake
            },
            'timestamp': sensor_data['timestamp']
        }
        meta_dir = self.save_path / 'meta'
        meta_dir.mkdir(exist_ok=True)
        with open(meta_dir / f'{step:05d}.json', 'w') as f:
            json.dump(meta, f)
```

#### 4.2 Video Generation Workflow

```bash
# After evaluation completes
python Bench2Drive/tools/generate_video.py -f ${SAVE_PATH}/route_folder/
```

### Phase 5: Configuration and Execution

#### 5.1 Agent Configuration

**File:** `Bench2DriveZoo/configs/diffusiondrive_b2d_agent.yaml`

```yaml
agent:
  name: diffusiondrive_b2d
  checkpoint: /path/to/diffusiondrive_b2d.pth
  config:
    model:
      trajectory_sampling:
        time_horizon: 4.0
        interval_length: 0.5
      num_modes: 20
      num_timesteps: 8
    sensors:
      camera:
        width: 1600
        height: 900
        fov: 100
      use_lidar: false  # Optional for DiffusionDrive
    control:
      max_steering: 0.7
      max_throttle: 0.75
      max_brake: 1.0
```

#### 5.2 Execution Commands

```bash
# Environment setup
export CARLA_ROOT=/path/to/carla
export SAVE_PATH=/path/to/save/evaluation_data
export IS_BENCH2DRIVE=1

# Start CARLA server (headless)
cd ${CARLA_ROOT}
./CarlaUE4.sh -RenderOffScreen -nosound -fps=10 -carla-rpc-port=2000 &

# Run evaluation
cd /workspace/Bench2DriveZoo/Bench2Drive
python leaderboard/leaderboard/leaderboard_evaluator.py \
  --routes=leaderboard/data/routes_devtest.xml \
  --repetitions=1 \
  --track=SENSORS \
  --agent=../team_code/diffusiondrive_b2d_agent.py \
  --agent-config=/path/to/config.yaml+/path/to/checkpoint.pth+experiment_name \
  --checkpoint=results/ \
  --port=2000 \
  --traffic-manager-port=8000

# Generate video
python tools/generate_video.py -f ${SAVE_PATH}/route_*/
```

## Implementation Timeline

### ✅ Completed (2025-01-22)

#### Core Agent Development
- ✅ Implemented basic agent structure (`diffusiondrive_b2d_agent.py`)
- ✅ Added sensor processing pipeline (`sensor_processor.py`)
- ✅ Integrated model loading without MMCV (`diffusion_model_wrapper.py`)

#### Model Integration
- ✅ Adapted feature builders for CARLA input (camera stitching, ego status)
- ✅ Implemented inference pipeline (direct PyTorch inference)
- ✅ Added trajectory post-processing (`trajectory_planner.py`)

#### Control Integration
- ✅ Integrated PID controller (reused from VAD agent)
- ✅ Added control smoothing and limits
- ✅ Implemented waypoint generation from trajectories

#### Data Saving and Configuration
- ✅ Implemented data saving for video generation
- ✅ Created comprehensive configuration YAML
- ✅ Added support for all 6 cameras + metadata saving

### 🔄 Pending Testing (Requires Environment Setup)

#### Testing Phase
- ⏳ Test with CARLA 0.9.15 installation
- ⏳ Validate with trained DiffusionDrive-B2D checkpoint
- ⏳ Run mini routes for initial validation
- ⏳ Debug and optimize performance

#### Full Evaluation
- ⏳ Run complete Bench2Drive evaluation suite
- ⏳ Generate evaluation videos
- ⏳ Collect performance metrics

## Key Differences from Original Plan

1. **No Open-Loop Evaluation:** Removed PDM score computation for Bench2Drive as it's not needed
2. **Focus on CARLA Integration:** All effort on closed-loop evaluation
3. **Direct Model Loading:** Bypass MMCV dependencies for simpler integration
4. **Reuse Existing Infrastructure:** Leverage VAD agent structure as reference

## Testing Strategy

### Unit Tests

1. Sensor processing correctness
2. Model inference with dummy inputs
3. Trajectory to waypoint conversion
4. Control command generation

### Integration Tests

1. Agent initialization with checkpoint
2. Full inference pipeline
3. Data saving functionality
4. Small route completion

### System Tests

1. Full Bench2Drive route evaluation
2. Multi-scenario testing
3. Video generation quality
4. Performance benchmarking

## Potential Challenges and Solutions

### Challenge 1: Model Loading Dependencies

**Issue:** DiffusionDrive uses MMCV for model building
**Solution:** Create standalone model loader using pure PyTorch

### Challenge 2: Sensor Data Format Mismatch

**Issue:** CARLA sensor format differs from training
**Solution:** Implement robust conversion pipeline with validation

### Challenge 3: Real-time Performance

**Issue:** Diffusion model may be slow for 10Hz control
**Solution:**

- Use truncated diffusion (50 steps)
- Implement caching for static features
- Consider model quantization if needed

### Challenge 4: Coordinate System Alignment

**Issue:** Potential mismatch between training and inference
**Solution:** Verify coordinate systems match (both use CARLA native)

## Success Metrics

1. **Functional:** Agent completes evaluation routes without crashes
2. **Performance:** Achieves >5Hz inference rate
3. **Quality:** Generated videos show reasonable driving behavior
4. **Compatibility:** Works with existing Bench2Drive evaluation pipeline

## Implementation Status

### ✅ Completed Files

1. **Main Agent** (`Bench2DriveZoo/team_code/diffusiondrive_b2d_agent.py`)
   - Full CARLA agent implementation
   - Sensor setup matching Bench2Drive configuration
   - Integration with PID controller
   - Data saving for video generation
   - Status: **COMPLETE**

2. **Model Wrapper** (`Bench2DriveZoo/team_code/diffusion_model_wrapper.py`)
   - Standalone PyTorch model loading (no MMCV dependencies)
   - Checkpoint loading with state dict handling
   - Inference pipeline for trajectory generation
   - Status: **COMPLETE**

3. **Sensor Processor** (`Bench2DriveZoo/team_code/sensor_processor.py`)
   - CARLA sensor data to DiffusionDrive format conversion
   - Camera stitching (3 front cameras → 1024×256)
   - Ego status processing from GPS/IMU
   - Placeholder LiDAR (model can work without it)
   - Status: **COMPLETE**

4. **Trajectory Planner** (`Bench2DriveZoo/team_code/trajectory_planner.py`)
   - Trajectory mode selection strategies
   - Coordinate transformation (ego → world)
   - Trajectory smoothing
   - Waypoint generation for PID controller
   - Status: **COMPLETE**

5. **Configuration** (`Bench2DriveZoo/configs/diffusiondrive_b2d_agent.yaml`)
   - Complete agent configuration
   - Model, sensor, and control parameters
   - Data saving settings
   - Status: **COMPLETE**

## Important Assumptions and Mock Implementations

### ⚠️ CARLA Environment

- **Assumption**: CARLA is not installed in the current environment
- **Implementation**: All CARLA imports are assumed to work at runtime
- **Testing**: Will require actual CARLA 0.9.15 installation for execution

### ⚠️ Model Checkpoint

- **Assumption**: Trained DiffusionDrive-B2D checkpoint exists
- **Path**: Expected at `/path/to/diffusiondrive_b2d_final.pth`
- **Fallback**: Code handles missing anchors by falling back to NavSim anchors

### ⚠️ Trajectory Anchors

- **Expected**: `kmeans_b2d_traj_20.npy` for Bench2Drive-specific anchors
- **Fallback**: Uses `kmeans_navsim_traj_20.npy` if B2D anchors not found
- **Location**: `/workspace/DiffusionDrive/download/`

### ⚠️ LiDAR Processing

- **Implementation**: Placeholder zero tensor (model can work without LiDAR)
- **Note**: Full LiDAR implementation would require point cloud processing

### ⚠️ GPS/IMU Conversion

- **Implementation**: Simple flat-earth approximation for coordinate conversion
- **Note**: Matches CARLA's internal coordinate system

## File Structure (As Implemented)

```
Bench2DriveZoo/
├── team_code/
│   ├── diffusiondrive_b2d_agent.py  # ✅ Main agent implementation
│   ├── diffusion_model_wrapper.py   # ✅ Model loading without MMCV
│   ├── sensor_processor.py          # ✅ Sensor data conversion
│   ├── trajectory_planner.py        # ✅ Trajectory post-processing
│   ├── pid_controller.py            # ✓ Existing (reused from VAD)
│   └── planner.py                   # ✓ Existing (reused from VAD)
├── configs/
│   └── diffusiondrive_b2d_agent.yaml # ✅ Agent configuration
└── docs/
    └── b2d/
        └── diffusiondrive_carla_integration.md # ✅ This document

DiffusionDrive/
├── navsim/agents/diffusiondrive/    # ✓ Existing model code (unchanged)
├── download/
│   ├── pytorch_model.bin            # ⚠️ Required: ResNet backbone
│   ├── kmeans_navsim_traj_20.npy   # ✓ Existing: NavSim anchors
│   └── kmeans_b2d_traj_20.npy      # ⚠️ Optional: B2D-specific anchors
└── checkpoints/
    └── diffusiondrive_b2d_final.pth # ⚠️ Required: Trained model
```

## Execution Commands (Updated)

```bash
# Environment setup
export CARLA_ROOT=/path/to/carla
export SAVE_PATH=/path/to/save/evaluation_data
export IS_BENCH2DRIVE=1
export ROUTES=/path/to/routes.xml

# Start CARLA server (headless)
cd ${CARLA_ROOT}
./CarlaUE4.sh -RenderOffScreen -nosound -fps=10 -carla-rpc-port=2000 &

# Run evaluation with the new agent
cd /workspace/Bench2DriveZoo/Bench2Drive
python leaderboard/leaderboard/leaderboard_evaluator.py \
  --routes=leaderboard/data/routes_devtest.xml \
  --repetitions=1 \
  --track=SENSORS \
  --agent=../team_code/diffusiondrive_b2d_agent.py \
  --agent-config=/workspace/DiffusionDrive/Bench2DriveZoo/configs/diffusiondrive_b2d_agent.yaml+/path/to/checkpoint.pth+experiment_name \
  --checkpoint=results/ \
  --port=2000 \
  --traffic-manager-port=8000

# Generate video from saved data
python tools/generate_video.py -f ${SAVE_PATH}/route_experiment_name/
```

## Next Steps for Testing

1. **Obtain trained checkpoint**: Need the actual DiffusionDrive model trained on Bench2Drive
2. **Install CARLA 0.9.15**: Required for running the evaluation
3. **Prepare test routes**: Use mini routes for initial testing
4. **Verify anchor files**: Ensure trajectory anchors are available
5. **Run small-scale test**: Start with single route evaluation
6. **Debug and optimize**: Based on test results

## References

1. Original DiffusionDrive paper and code
2. Bench2Drive dataset documentation
3. CARLA autonomous agent API
4. VAD agent implementation (reference)
