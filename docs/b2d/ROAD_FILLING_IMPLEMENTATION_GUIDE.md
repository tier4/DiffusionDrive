# Road Filling Implementation Guide

## Overview
Transform BEV generation from drawing lane lines to filling road surfaces between lane boundaries.

## Current State vs Desired State

### Current Implementation
- **Lane lines are drawn as lines** on the BEV map
- Lane types (Solid, Broken, Center) map to BEV classes 1 and 3
- Results in sparse BEV with only thin lines representing roads

### Desired Implementation  
- **Fill the area between lane lines** to create solid road surfaces
- Lane lines themselves disappear (not drawn)
- Create new "driving lane" class that represents the filled road area

## Semantic Class Mapping Changes

### New Class Structure (3 classes only)
1. **Class 0**: Background (non-road areas)
2. **Class 1**: Road surface (filled area between lane boundaries) 
3. **Class 3**: Driving lane (driveable area, subset of road surface)

### Key Changes
- Lane line types (Solid, Broken, etc.) are NO LONGER drawn as lines
- Instead, they define the BOUNDARIES of road polygons to fill
- The lines themselves become invisible - only used for polygon generation

## Implementation Strategy

### 1. Add New Method (Non-Breaking)
Create a new method alongside existing implementation to avoid breaking current functionality:
- Add `generate_bev_with_road_filling()` method to `MapProcessor` class in `navsim/common/bev_map_utils.py`
- Add parameter `fill_roads=False` to maintain backward compatibility
- When `fill_roads=True`, use new road filling logic

### 2. Lane Boundary Detection Algorithm

#### Step 1: Group Lanes by Road Segment
- Parse map data to identify lanes belonging to same road
- Group parallel lanes that form road boundaries
- Identify lane pairs (left boundary, right boundary)

#### Step 2: Generate Road Polygons
- For each road segment:
  - Find leftmost and rightmost lane lines
  - Connect lane points to form closed polygon
  - Handle gaps and discontinuities

#### Step 3: Classify Polygon Types
- Determine if polygon represents:
  - General road surface (Class 1)
  - Driveable lane area (Class 3)
  - Based on lane types and positions

### 3. Polygon Filling Implementation

#### Key Functions to Add
1. **`identify_road_boundaries(lanes)`**
   - Input: List of lane segments
   - Output: Pairs of (left_boundary, right_boundary)
   - Logic: Use lateral position and lane type to identify boundaries

2. **`create_road_polygon(left_lane, right_lane)`**
   - Input: Two lane point arrays forming boundaries
   - Output: Closed polygon points
   - Logic: Connect start/end points of lanes to form polygon

3. **`fill_road_surface(bev_map, polygon, semantic_class)`**
   - Input: BEV map, polygon points, class to fill
   - Output: Updated BEV with filled polygon
   - Logic: Use cv2.fillPoly() to fill area

4. **`classify_road_area(lane_types, position)`**
   - Input: Lane boundary types and position
   - Output: Semantic class (1 for road, 3 for driving lane)
   - Logic: Determine based on lane configuration

### 4. Processing Pipeline

```python
def generate_bev_with_road_filling():
    # 1. Initialize empty BEV
    bev_map = np.zeros()
    
    # 2. Process lanes to find road boundaries
    for road_id, road_data in map_data:
        boundaries = identify_road_boundaries(road_data.lanes)
        
        # 3. Generate polygons from boundaries
        for left_bound, right_bound in boundaries:
            polygon = create_road_polygon(left_bound, right_bound)
            
            # 4. Classify and fill polygon
            semantic_class = classify_road_area(...)
            fill_road_surface(bev_map, polygon, semantic_class)
    
    # 5. Process triggers (unchanged)
    # 6. Return filled BEV
    return bev_map
```

### 5. Edge Cases to Handle

#### Curved Roads
- Interpolate between lane points for smooth curves
- Maintain consistent road width through curves
- Handle varying point densities

#### Intersections
- Detect lane convergence/divergence
- Fill intersection areas appropriately
- Handle multiple overlapping road segments

#### Missing Lane Boundaries
- Estimate road width from available lanes
- Use default width when only centerline available
- Handle single-lane roads

### 6. Testing Strategy

#### Unit Tests
- Test polygon generation from various lane configurations
- Test filling algorithms with different polygon shapes
- Verify semantic class assignment logic

#### Integration Tests
- Test with real Bench2Drive map data
- Verify road continuity in generated BEV
- Check handling of complex road geometries

#### Visual Validation
- Create visualization script to compare:
  - Original line-based BEV
  - New filled road BEV
- Overlay on camera images for validation

## Implementation Order

1. **Phase 1**: Basic polygon generation
   - Implement `create_road_polygon()` for straight roads
   - Test with simple parallel lanes

2. **Phase 2**: Polygon filling
   - Implement `fill_road_surface()` using OpenCV
   - Add semantic class assignment

3. **Phase 3**: Boundary detection
   - Implement `identify_road_boundaries()`
   - Handle various lane configurations

4. **Phase 4**: Integration
   - Add `generate_bev_with_road_filling()` to MapProcessor
   - Wire up with existing pipeline

5. **Phase 5**: Edge cases
   - Handle curves and intersections
   - Add robustness for missing data

## Files to Modify

1. **`navsim/common/bev_map_utils.py`**
   - Add new road filling functions
   - Extend MapProcessor class

2. **`navsim/common/bench2drive_constants.py`**
   - Update semantic class mappings if needed
   - Add road width constants

3. **`tests/navsim/common/test_bev_map_utils.py`**
   - Add comprehensive tests for new functionality

4. **`navsim/common/bench2drive_scene.py`**
   - Add option to use road filling in BEV generation

## Validation Criteria

### Correctness
- Road surfaces correctly fill area between lane boundaries
- No gaps or overlaps in road polygons
- Proper semantic class assignment

### Performance
- Processing time comparable to line drawing
- Memory usage within acceptable limits
- Scalable to large maps

### Compatibility
- Existing functionality unchanged when `fill_roads=False`
- Model can consume new BEV format
- Visualization tools work with filled roads

## Next Steps

1. Review and approve this implementation guide
2. Create feature branch for development
3. Implement Phase 1 (basic polygon generation)
4. Iterate through phases with testing
5. Integrate and validate with full pipeline