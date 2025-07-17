# Bench2drive mapping strategy

## ⚠️ IMPORTANT FINDINGS

### BEV Semantic Map Limitations

- **Bench2Drive has NO `semantic_top_down` views** - only `rgb_top_down`
- Current implementation uses **placeholder BEV semantic maps**
- This limits the effectiveness of DiffusionDrive's BEV semantic auxiliary task (loss weight: 14.0)
- Need to generate BEV semantic maps from available data (perspective semantic views, LiDAR, annotations)

### Category Mapping Validation Required

- Similar to command mapping issues, semantic category mapping needs validation
- Ensure no "swapping" issues between Bench2Drive semantic tags and DiffusionDrive's expected classes
- Test consistency between perspective semantic views and generated BEV semantic maps

### Bench2Drive Semantic Analysis Results

**Analysis of 13,770 semantic images from Bench2Drive base dataset:**

- **Total unique pixel values found**: 27 values
- **Actual values present**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
- **Missing values**: [16, 17] (Bus=16, Train=17)
- **Key finding**: Not all semantic categories defined in CARLA are present in the base dataset

### Mapping Validation Status

✅ **Present in dataset**: Unlabeled(0), Roads(1), SideWalks(2), Building(3), Wall(4), Fence(5), Pole(6), TrafficLight(7), TrafficSign(8), Vegetation(9), Terrain(10), Sky(11), Pedestrian(12), Rider(13), Car(14), Truck(15), Motorcycle(18), Bicycle(19), Static(20), Dynamic(21), Other(22), Water(23), RoadLine(24), Ground(25), Bridge(26), RailTrack(27), GuardRail(28)

❌ **Missing in dataset**: Bus(16), Train(17)

⚠️ **Implications for BEV semantic map generation**:

- All major categories (Road, Walkways, Vehicles, Pedestrians, Static objects) are present
- Missing Bus/Train categories can be handled by grouping with other vehicles
- Current 7-class BEV mapping should work with available semantic data

***

## 0: Background

This category includes elements that are generally not interactive or are part of the distant environment.

- **Semantic Tags**: `Unlabeled` (0), `Building` (3), `Vegetation` (9), `Terrain` (10), `Sky` (11), `Other` (22), `Water` (23), `Bridge` (26), `RailTrack` (27)

***

## 1: Road (including lanes and intersections)

This includes all drivable surfaces and the markings on them, excluding the specific centerline.

- **BEV/HD-Map Categories**: All `Lane Marking Types` (Broken, Solid, SolidSolid, Other, NONE) and `Colors` (White, Yellow, Blue). Also includes other markings like crosswalks.
- **Semantic Tags**: `Roads` (1), `RoadLine` (24), `Ground` (25)

***

## 2: Walkways

This category is for areas designated for pedestrians.

- **Semantic Tags**: `SideWalks` (2)

***

## 3: Lane centerlines

This specifically refers to the annotated center of a driving lane.

- **BEV/HD-Map Categories**: Lane markings where `Type == 'Center'`. The various `Topology` statuses (e.g., Junction, Normal) are attributes of these centerlines.

***

## 4: Static objects (like barriers and signs)

This includes all non-actor obstacles, traffic control infrastructure, and roadside furniture.

- **BEV/HD-Map Categories**: `Trigger Volumes` for `StopSign` and `TrafficLight`.
- **Anno Structure**: `traffic_light`, `traffic_sign`.
- **Semantic Tags**: `Wall` (4), `Fence` (5), `Pole` (6), `TrafficLight` (7), `TrafficSign` (8), `Static` (20), `Dynamic` (21), `GuardRail` (28).

***

## 5: Vehicles

This category encompasses all types of vehicles, including the ego vehicle.

- **Anno Structure**: `ego_vehicle`, `vehicle`.
- **Semantic Tags**: `Car` (14), `Truck` (15), `Bus` (16), `Train` (17), `Motorcycle` (18), `Bicycle` (19).

***

## 6: Pedestrians

This includes all human actors.

- **Anno Structure**: `pedestrian` (class 'walker').
- **Semantic Tags**: `Pedestrian` (12), `Rider` (13).
