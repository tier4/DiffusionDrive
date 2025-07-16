# Bench2drive mapping strategy

***

## 0: Background

This category includes elements that are generally not interactive or are part of the distant environment.

* **Semantic Tags**: `Unlabeled` (0), `Building` (3), `Vegetation` (9), `Terrain` (10), `Sky` (11), `Other` (22), `Water` (23), `Bridge` (26), `RailTrack` (27)

***

## 1: Road (including lanes and intersections)

This includes all drivable surfaces and the markings on them, excluding the specific centerline.

* **BEV/HD-Map Categories**: All `Lane Marking Types` (Broken, Solid, SolidSolid, Other, NONE) and `Colors` (White, Yellow, Blue). Also includes other markings like crosswalks.
* **Semantic Tags**: `Roads` (1), `RoadLine` (24), `Ground` (25)

***

## 2: Walkways

This category is for areas designated for pedestrians.

* **Semantic Tags**: `SideWalks` (2)

***

## 3: Lane centerlines

This specifically refers to the annotated center of a driving lane.

* **BEV/HD-Map Categories**: Lane markings where `Type == 'Center'`. The various `Topology` statuses (e.g., Junction, Normal) are attributes of these centerlines.

***

## 4: Static objects (like barriers and signs)

This includes all non-actor obstacles, traffic control infrastructure, and roadside furniture.

* **BEV/HD-Map Categories**: `Trigger Volumes` for `StopSign` and `TrafficLight`.
* **Anno Structure**: `traffic_light`, `traffic_sign`.
* **Semantic Tags**: `Wall` (4), `Fence` (5), `Pole` (6), `TrafficLight` (7), `TrafficSign` (8), `Static` (20), `Dynamic` (21), `GuardRail` (28).

***

## 5: Vehicles

This category encompasses all types of vehicles, including the ego vehicle.

* **Anno Structure**: `ego_vehicle`, `vehicle`.
* **Semantic Tags**: `Car` (14), `Truck` (15), `Bus` (16), `Train` (17), `Motorcycle` (18), `Bicycle` (19).

***

## 6: Pedestrians

This includes all human actors.

* **Anno Structure**: `pedestrian` (class 'walker').
* **Semantic Tags**: `Pedestrian` (12), `Rider` (13).
