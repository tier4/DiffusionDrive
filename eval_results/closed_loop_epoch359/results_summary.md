# DiffusionDrive Bench2Drive Closed-Loop Evaluation Results

**Checkpoint:** epoch=359, val_traj_loss=10.9327 (b2d_scratch_1000ep)
**Benchmark:** Bench2Drive 220-route full benchmark (CARLA 0.9.15)
**Date:** 2026-07-13 → 2026-07-17

## Comparison with Published Baselines

Baseline numbers from the official Bench2DriveZoo evaluation JSONs (new version).
Note: baselines had 213-218 completed routes (some CARLA crashes); DiffusionDrive
ran all 220 (1 sim-crashed = CARLA spawn timeout on Town13, recorded as DS 0).

| Method | Backbone | DS | RC (%) | SR (%) | Completed | Blocked | Timed Out |
|--------|----------|------|--------|--------|-----------|---------|-----------|
| UniAD-Tiny | ResNet-50 | 40.73 | 63.5 | 13.2 | 93/215 | 44 | 78 |
| DiffusionDrive | ResNet-34 | 42.17 | 73.9 | 14.5 | 113/220 | 36 | 70 |
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
| Mean | 20.7% |

Per-ability baselines not published in the Bench2DriveZoo README.

## Run Configuration

- CARLA 0.9.15 with AdditionalMaps (Towns 11/12/13/15)
- Agent: DiffusionDrive closed-loop agent (this repo, navsim/evaluation/carla_closed_loop/)
- Controller: TCP/VAD PID (vendored, unmodified gains)
- Sensor rig: 3 front cameras (fov 70) + LiDAR (matching dataset collection)
- Features: Bench2DriveFeatureBuilder (same as training — parity verified offline)
- Resource caps: CARLA 16GB/8CPU, Agent 20GB/8CPU, with host watchdog
