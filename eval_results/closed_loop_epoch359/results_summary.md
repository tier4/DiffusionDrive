# DiffusionDrive Bench2Drive Closed-Loop Evaluation Results

**Checkpoint:** epoch=359, val_traj_loss=10.9327 (b2d_scratch_1000ep)
**Benchmark:** Bench2Drive 220-route full benchmark (CARLA 0.9.15)
**Date:** 2026-07-13 → 2026-07-17

## Comparison with Published Baselines

All numbers from the official Bench2DriveZoo evaluation (new version).
Note: VAD/UniAD baselines had 213-218 completed routes (some CARLA crashes);
DiffusionDrive completed all 220 (1 sim-crashed route = CARLA spawn timeout on Town13).

| Method | Backbone | DS ↑ | RC (%) ↑ | SR (%) ↑ | Completed | Blocked | Timed Out |
|--------|----------|------|----------|----------|-----------|---------|-----------|
| UniAD-Tiny | ResNet-50 | 40.73 | 63.5 | 13.2 | 93/215 | 44 | 78 |
| **DiffusionDrive** | **ResNet-34** | **42.17** | **73.9** | **14.5** | **113/220** | **36** | **70** |
| VAD | ResNet-50 | 42.35 | 65.5 | 15.0 | 99/213 | 49 | 65 |
| UniAD-Base | ResNet-101 | 45.81 | 68.2 | 16.4 | 110/218 | 37 | 71 |

### Multi-Ability Success Rates

| Ability | DiffusionDrive |
|---------|---------------|
| Give_Way | 40.0% |
| Traffic_Signs | 22.2% |
| Emergency_Brake | 20.0% |
| Merging | 12.5% |
| Overtaking | 8.9% |
| **Mean** | **20.7%** |

(Per-ability baselines not published in the Bench2DriveZoo README; numbers above are
from our run of the official ability_benchmark.py tool.)

## Key Observations

1. **DiffusionDrive (ResNet-34) achieves competitive DS (42.17) with a lighter backbone**
   than both VAD (ResNet-50, DS 42.35) and UniAD-Base (ResNet-101, DS 45.81).

2. **Highest Route Completion (73.9%)** among all methods — the model gets further along
   routes before failing, suggesting better lane-following behavior.

3. **Lowest block rate (16.4%)** vs VAD (23.0%) and UniAD-Tiny (20.5%) — fewer situations
   where the car stops and can't recover.

4. **Success Rate (14.5%)** is comparable to the baselines (13.2-16.4%), meaning the
   infraction rate when completing routes is similar.

5. **Give_Way is the strongest ability (40%)**, consistent with the model learning to
   yield to emergency vehicles and at invading turns from B2D's diverse training data.

6. **Overtaking is the weakest (8.9%)** — the model struggles to plan lane changes around
   parked/slow obstacles, likely due to the conservative from-scratch training.

## Run Configuration

- CARLA 0.9.15 with AdditionalMaps (Towns 11/12/13/15)
- Agent: DiffusionDrive closed-loop agent (this repo, navsim/evaluation/carla_closed_loop/)
- Controller: TCP/VAD PID (vendored, unmodified gains)
- Sensor rig: 3 front cameras (fov 70) + LiDAR (matching dataset collection)
- Features: Bench2DriveFeatureBuilder (same as training — parity verified offline)
- Resource caps: CARLA 16GB/8CPU, Agent 20GB/8CPU, with host watchdog
