# Bench2Drive Training NaN Investigation

## Executive Summary

Analysis of 10,000 cached samples from both NavSim and Bench2Drive datasets reveals significant differences that may contribute to NaN losses during training. Further investigation has identified the root cause of the heading issue and verified speed data integrity.

## Cache Data Comparison Results

### 1. Status Feature Differences

**NavSim**: min=-3.31, max=15.49, mean=0.696, std=1.94  
**Bench2Drive**: min=-26.97, max=14.32, mean=0.458, std=2.05

- Bench2Drive has an extreme minimum value of -26.97
- This is approximately 8x more negative than NavSim's minimum

### 2. Heading Distribution Differences  

**NavSim**: min=-1.65, max=1.44 radians, mean=0.050, std=0.233  
**Bench2Drive**: min=-0.110, max=0.110 radians, mean=-0.0006, std=0.017

**Root Cause Identified**: The Bench2Drive caching code incorrectly uses `anno['theta']` for ego vehicle heading instead of `anno['bounding_boxes'][ego_index]['rotation'][2]` (yaw component). Analysis of raw data shows:

- `anno['theta']` does NOT represent ego vehicle heading/yaw
- The actual ego heading is in the bounding box rotation field: `anno['bounding_boxes'][ego_index]['rotation'][2]`
- 100% mismatch rate between these values (e.g., anno['theta']=4.863° vs ego yaw=-171.349°)
- This explains why cached trajectories have near-zero heading changes
- The incorrect heading affects trajectory normalization and relative trajectory calculations

### 3. Trajectory X Coordinate Differences

**NavSim**: min=-0.096, max=61.36 meters, mean=9.87  
**Bench2Drive**: min=-39.28, max=40.67 meters, mean=-0.082

- NavSim shows forward-biased distribution (positive mean)
- Bench2Drive is centered around zero

### 4. Trajectory Y Coordinate Differences

**NavSim**: min=-14.86, max=25.58 meters, mean=0.375, std=1.99  
**Bench2Drive**: min=-39.41, max=36.62 meters, mean=-0.055, std=6.80

- Bench2Drive has 3.4x larger standard deviation
- Range is approximately 2x larger

### 5. Agent Information

- NavSim: agent_states and agent_labels contain actual values
- Bench2Drive: All agent_states and agent_labels are exactly zero

### 6. BEV Semantic Map Classes

- NavSim: Uses classes 0-6
- Bench2Drive: Uses classes 0-4 only

## Normalization Impact Analysis

With current normalization parameters:

```python
"bench2drive": {
    "x": {"offset": 35.0, "scale": 70.0},
    "y": {"offset": 35.0, "scale": 70.0},
    "heading": {"offset": 0.05, "scale": 0.5}
}
```

The heading normalization produces:

- Input range: [-0.110, 0.110]
- Normalized range: [-0.32, -0.12]
- All values map to negative range

## Observations Requiring Investigation

1. **Status Feature**: Why does Bench2Drive have -26.97 as minimum value?
2. ~~**Heading**: Why is the range exactly ±0.110? Is this clipping or different units?~~ **SOLVED**: Code uses wrong `anno['theta']` instead of ego vehicle yaw from bounding boxes
3. **Coordinates**: Why are X values centered at zero in Bench2Drive vs forward-biased in NavSim?
4. **Agent States**: Why are all agent values zero in Bench2Drive?

## Additional Findings

### Speed Data Verification

- Checked 2000+ random samples across 43 different scenario types
- `anno['speed']` ALWAYS matches `anno['bounding_boxes'][ego_index]['speed']`
- 100% match rate with 0.0 difference
- Speed data is correctly extracted and cached

### Comprehensive Cache Format Verification (2025-07-28)

#### 1. Driving Command Encoding ✓

- Analyzed 2000 cached samples
- **100% valid one-hot encoding** (all commands properly encoded)
- Status feature shape correct: [8] = [4 command + 2 velocity + 2 acceleration]
- Command distribution: LEFT (5.1%), STRAIGHT (93.3%), RIGHT (1.6%), UNKNOWN (0%)

#### 2. Trajectory Coordinates ✗

- Analyzed 1000 cached samples
- **Only 37.4% start near ego position** (should be ~100% for ego-relative)
- First waypoint mean distance: 1.886m from ego (should be ~0.5m)
- **Heading variation extremely low**: std=0.020 rad (confirms theta bug)
- Many trajectories have near-zero heading changes

#### 3. Agent States ✗

- Analyzed 1000 cached samples  
- **100% of samples have all agent states as zeros**
- No valid agents detected in any sample
- Confirms the agent detection failure reported earlier

#### 4. BEV Semantic Map ✗

- Analyzed 500 cached samples
- **No vehicle (class 5) or pedestrian (class 6) pixels found**
- Only contains: background (94.88%), road (2.94%), centerlines (1.96%), static objects (0.22%)
- BEV generation using simplified method without proper agent/vehicle integration

## Summary of Critical Issues

1. **Heading Bug** ✗ - Using wrong `anno['theta']` instead of ego vehicle yaw
2. **Trajectory Coordinates** ✗ - Not properly ego-relative (37.4% accuracy)
3. **Agent Detection** ✗ - All agent states are zeros
4. **BEV Generation** ✗ - No vehicles/pedestrians in semantic maps
5. **Driving Commands** ✓ - Correctly one-hot encoded
6. **Speed Values** ✓ - Correctly extracted from ego vehicle

## Comprehensive Fix Plan for Bench2Drive Integration

### Phase 1: Investigation of Available Data ✓ COMPLETED

1. **world2ego Matrix Verification** ✓
   - **CONFIRMED**: `anno['bounding_boxes'][ego_index]['world2ego']` exists in 100% of files
   - It's a proper 4x4 transformation matrix (homogeneous coordinates)
   - Successfully transforms world coordinates to ego-relative coordinates
   - Matrix can be inverted to get ego position in world

2. **Bounding Box Structure Analysis** ✓
   - **Ego vehicle fields**: location, rotation, speed, world2ego, extent, etc.
   - **Other vehicle fields**: location, rotation, speed, world2vehicle, extent, class
   - **Object classes found**:
     - `ego_vehicle` - the ego car
     - `vehicle` - other cars (maps to NavSim class 5)
     - `traffic_light`, `traffic_sign` - infrastructure
     - Missing: `pedestrian` class (need to investigate further)

3. **Comprehensive Object Class Investigation** ✓ (2025-07-29)
   - **Analyzed FULL dataset: 247,656 annotation files, 5,472,774 objects**
   - **Class Format**: Bench2Drive uses **string labels only**, NOT numeric CARLA labels
   - **Object Classes Distribution**:
     - `vehicle`: 2,953,662 objects (53.97%) → NavSim class 5
     - `traffic_sign`: 1,311,386 objects (23.96%) → NavSim class 4
     - `traffic_light`: 943,768 objects (17.24%) → NavSim class 4
     - `ego_vehicle`: 247,656 objects (4.53%) → Skip (ego)
     - `walker`: 16,302 objects (0.30%) → NavSim class 6 (pedestrians!)
   - **Key Finding**: Pedestrians ARE present as `'walker'` class
   - All vehicles grouped under single `'vehicle'` class (no car/truck/bus distinction)
   - Walkers have special fields: gender, age, bone, world2ped matrix

### Phase 2: Core Fixes Implementation - UPDATED (2025-07-29)

#### Fix 0: Add Object Class Mapping Constants

**File**: `navsim/common/bench2drive_constants.py` - Add class mappings

```python
# Bench2Drive string class names to NavSim semantic classes
B2D_CLASS_TO_NAVSIM = {
    # Vehicles (NavSim class 5)
    'vehicle': 5,
    
    # Pedestrians (NavSim class 6) 
    'walker': 6,  # This is how pedestrians are labeled in B2D
    
    # Static objects (NavSim class 4)
    'traffic_light': 4,
    'traffic_sign': 4,
    
    # Skip
    'ego_vehicle': -1,
}
```

#### Fix 1: Heading Correction

**File**: `navsim/common/bench2drive_scene.py` - `_extract_ego_status()`

```python
# Find ego vehicle in bounding boxes
ego_index = None
for i, bbox in enumerate(anno.get('bounding_boxes', [])):
    if bbox.get('class') == 'ego_vehicle':
        ego_index = i
        break

if ego_index is not None:
    ego_bbox = anno['bounding_boxes'][ego_index]
    # Use actual ego yaw from rotation (already in degrees)
    ego_yaw_degrees = ego_bbox['rotation'][2]  # yaw component
    ego_heading = np.radians(ego_yaw_degrees)  # Convert to radians
    
    # Also extract position from bbox for consistency
    ego_location = ego_bbox['location']
    ego_x = ego_location[0]
    ego_y = ego_location[1]
else:
    # Fallback to original (incorrect) method
    theta = anno.get("theta", 0.0)
    ego_heading = -np.radians(theta)
```

#### Fix 2: Trajectory Coordinate Transformation  

**File**: `navsim/common/bench2drive_scene.py` - `get_future_trajectory()`

```python
# Get current frame ego world2ego matrix
current_anno = self._load_annotation(frame_idx)
ego_bbox = None
for bbox in current_anno.get('bounding_boxes', []):
    if bbox.get('class') == 'ego_vehicle':
        ego_bbox = bbox
        break

if ego_bbox and 'world2ego' in ego_bbox:
    # Use world2ego matrix for transformation
    W2E = np.array(ego_bbox['world2ego'])
    
    # For each future frame
    for i in range(1, NUM_FUTURE_WAYPOINTS + 1):
        future_anno = self._load_annotation(frame_idx + i)
        
        # Get future ego position in world
        future_ego = None
        for bbox in future_anno.get('bounding_boxes', []):
            if bbox.get('class') == 'ego_vehicle':
                future_ego = bbox
                break
        
        if future_ego:
            # Transform using matrix
            world_pos = np.array([
                future_ego['location'][0],
                future_ego['location'][1], 
                future_ego['location'][2],
                1.0  # homogeneous
            ])
            ego_pos = W2E @ world_pos
            
            # Extract ego-relative x, y
            ego_x = ego_pos[0]
            ego_y = ego_pos[1]
            
            # Calculate relative heading
            future_yaw = np.radians(future_ego['rotation'][2])
            current_yaw = np.radians(ego_bbox['rotation'][2])
            ego_heading = future_yaw - current_yaw
```

#### Fix 3: Agent States Implementation with Walker/Pedestrian Support

**File**: `navsim/common/bench2drive_scene.py` - `get_agents()`

```python
# Get ego world2ego matrix
ego_bbox = None
for bbox in anno.get('bounding_boxes', []):
    if bbox.get('class') == 'ego_vehicle':
        ego_bbox = bbox
        break

if not ego_bbox or 'world2ego' not in ego_bbox:
    return current_implementation  # fallback

W2E = np.array(ego_bbox['world2ego'])
agent_states = []
agent_types = []  # Track if vehicle (5) or pedestrian (6)

# Process all non-ego objects
for bbox in anno.get('bounding_boxes', []):
    obj_class = bbox.get('class')
    
    # Skip if not a relevant object
    navsim_class = B2D_CLASS_TO_NAVSIM.get(obj_class, -1)
    if navsim_class <= 0:  # Skip ego_vehicle (-1) and unknown objects
        continue
        
    # Only process vehicles and pedestrians
    if navsim_class not in [5, 6]:
        continue
        
    # Transform to ego coordinates
    world_pos = np.array([
        bbox['location'][0],
        bbox['location'][1],
        bbox['location'][2], 
        1.0
    ])
    ego_pos = W2E @ world_pos
    
    # Check if within lidar range
    if abs(ego_pos[0]) <= 32 and abs(ego_pos[1]) <= 32:
        # Extract agent info
        ego_x = ego_pos[0]
        ego_y = ego_pos[1]
        
        # Relative heading
        agent_yaw = np.radians(bbox['rotation'][2])
        ego_yaw = np.radians(ego_bbox['rotation'][2])
        ego_heading = agent_yaw - ego_yaw
        
        # Dimensions
        extent = bbox['extent']
        length = extent[0] * 2  # extent is half-size
        width = extent[1] * 2
        
        # For walkers/pedestrians, ensure reasonable dimensions
        if navsim_class == 6:  # Pedestrian
            length = min(length, 1.0)  # Max 1m length
            width = min(width, 0.6)   # Max 0.6m width
        
        agent_states.append([ego_x, ego_y, ego_heading, length, width])
        agent_types.append(navsim_class)

# Sort by distance and limit to MAX_AGENTS
combined = list(zip(agent_states, agent_types))
combined = sorted(combined, key=lambda x: x[0][0]**2 + x[0][1]**2)[:MAX_AGENTS]

# Separate back into states and types
agent_states = [x[0] for x in combined]
agent_types = [x[1] for x in combined]

# Pad to MAX_AGENTS
while len(agent_states) < MAX_AGENTS:
    agent_states.append([0, 0, 0, 0, 0])
    agent_types.append(0)

# Return with types for BEV rendering
return agent_states, agent_labels, agent_types
```

#### Fix 4: BEV Semantic Map Integration with Vehicles and Pedestrians

**File**: `navsim/common/bench2drive_scene.py` - `get_bev_semantic_map()`

```python
# After getting base BEV from simplified method
bev_map = generate_simple_bev_semantic(...)

# Get agents with types and render them into BEV
agent_states, agent_labels, agent_types = self.get_agents(frame_idx)

# Render agents based on their type
for i, (agent, valid, agent_type) in enumerate(zip(agent_states, agent_labels, agent_types)):
    if not valid:
        continue
        
    x, y, heading, length, width = agent
    
    # Convert to BEV pixel coordinates
    # BEV is 128x256, resolution 0.25m/pixel
    # Origin at center bottom (ego position)
    pixel_x = int((x / 0.25) + 128)  # 256/2 = 128
    pixel_y = int(128 - (y / 0.25))  # Flip Y, start from bottom
    
    # Create object rectangle
    half_length = int(length / 0.25 / 2)
    half_width = int(width / 0.25 / 2)
    
    # Simple axis-aligned rendering (can improve with rotation)
    x1 = max(0, pixel_x - half_width)
    x2 = min(256, pixel_x + half_width)
    y1 = max(0, pixel_y - half_length)
    y2 = min(128, pixel_y + half_length)
    
    # Set pixels based on agent type
    # agent_type is already the NavSim class (5 for vehicles, 6 for pedestrians)
    bev_map[y1:y2, x1:x2] = agent_type

return torch.tensor(bev_map, dtype=torch.int64)
```

### Phase 3: Validation and Testing

1. **Unit Tests**
   - Test each fix independently with sample annotations
   - Verify coordinate transformations
   - Check class mappings

2. **Integration Testing**  
   - Re-run verification scripts on fixed code
   - Ensure all metrics improve:
     - Trajectory ego-relative rate > 95%
     - Agent detection rate > 0%
     - BEV contains vehicles/pedestrians

3. **Re-cache Dataset**
   - Clear old cache
   - Run caching with fixed implementation
   - Verify cache quality

### Phase 4: Training Validation

1. **Small-scale Training**
   - Train on subset to check for NaN losses
   - Monitor loss curves and gradients

2. **Full Training**
   - Complete training run if small-scale succeeds
   - Compare metrics with NavSim baseline

## Implementation Priority

1. **Heading Fix** (Critical - affects all trajectories)
2. **Agent States** (Critical - currently all zeros)
3. **Trajectory Coordinates** (Important - affects trajectory quality)
4. **BEV Integration** (Important - affects perception features)

## Success Criteria

- Heading standard deviation > 0.1 rad (similar to NavSim)
- >95% of trajectories start within 1m of ego
- >50% of samples contain detected agents (vehicles and/or walkers)
- BEV maps show vehicles (class 5) in >50% of samples
- BEV maps show pedestrians (class 6) in samples with walkers (0.30% based on full dataset analysis)
- All 7 NavSim semantic classes properly represented in BEV
- No NaN losses during training

## Implementation Timeline

1. **Day 1**: Implement heading fix and trajectory transformation
2. **Day 2**: Implement agent states and BEV integration
3. **Day 3**: Testing and validation with verification scripts
4. **Day 4**: Re-cache dataset with fixes
5. **Day 5**: Training validation

## Key Insights from Investigation

- **world2ego matrix is available** - This simplifies coordinate transformations significantly
- **Rotation order**: [roll, pitch, yaw] in degrees in bounding boxes
- **Object classes are string-based**: NOT numeric CARLA labels as initially expected
- **Vehicle class**: All vehicles grouped under single 'vehicle' class → NavSim class 5
- **Pedestrians FOUND**: Labeled as 'walker' (0.30% of objects) → NavSim class 6
- **Static objects**: 'traffic_light' and 'traffic_sign' → NavSim class 4
- **Coordinate system**: Bench2Drive uses different origin but world2ego handles conversion
- **Class distribution**: Vehicles (53.97%), Traffic signs (23.96%), Traffic lights (17.24%), Walkers (0.30%)
- **Walker attributes**: Include special fields like gender, age, bone structure, and world2ped matrix
