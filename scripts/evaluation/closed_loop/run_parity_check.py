"""Offline parity check: dataset frames replayed through the closed-loop adapter
must produce feature tensors identical to the training pipeline.

This is the linchpin proof for closed-loop CARLA evaluation: it demonstrates that
`build_agent_input` (the live-CARLA input adapter) fed with the inverse-reconstructed
raw sensor inputs of a dataset frame yields feature tensors bit-for-bit comparable
(atol 1e-5) to what `Bench2DriveScene.get_agent_input` -> `Bench2DriveFeatureBuilder`
produces during training. It also validates the compass->world-yaw formula offline
against the recorded IMU compass in the raw annotation, where available.

Usage (full check against real data, inside the Docker image, 20GB/8CPU caps):

  python3 scripts/evaluation/closed_loop/run_parity_check.py \
      --data-root /mnt/nvme1/dataset/Bench2Drive-Base --num-samples 50

  # optional end-to-end model parity (needs GPU + checkpoint):
  python3 scripts/evaluation/closed_loop/run_parity_check.py \
      --data-root /mnt/nvme1/dataset/Bench2Drive-Base --num-samples 50 \
      --with-model --checkpoint /path/to/ckpt.pth
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import Bench2DriveFeatureBuilder
from navsim.common.bench2drive_dataloader import (
    Bench2DriveDataConfig,
    Bench2DriveSceneLoader,
)
from navsim.evaluation.carla_closed_loop.input_adapter import (
    LIDAR_TO_EGO_TRANSLATION,
    build_agent_input,
    world_yaw_from_compass,
)

FEATURE_NAMES = ("camera_feature", "lidar_feature", "status_feature")


def reconstruct_carla_inputs(scene, frame_idx: int = -1) -> dict:
    """Rebuild the raw CARLA-sensor-style kwargs for ``build_agent_input`` from a
    loaded dataset scene frame -- the inverse of what ``data_collect.py`` recorded.

    The adapter round-trips exactly against the training loader:
      * RGB -> BGRA:      adapter's carla_bgra_to_rgb reverses this back to RGB.
      * ego-frame lidar -> sensor frame: adapter re-adds LIDAR_TO_EGO_TRANSLATION.
      * speed:            the SIGNED recorded scalar anno["speed"]. Training built
                          (vx, vy) = speed * (cos theta, -sin theta) from this signed
                          scalar, and the adapter applies the identical formula, so
                          feeding anno["speed"] is the exact inverse. hypot(vx, vy)
                          would drop the sign and flip velocity on negative-speed
                          frames (near-zero jitter / reverse) -- see report finding.
      * compass:          theta + pi/2, the inverse of world_yaw_from_compass.
      * accel / command:  read straight from the raw annotation (lossless).
    """
    if frame_idx == -1:
        resolved_idx = scene.history_frames
    else:
        resolved_idx = frame_idx

    agent_input = scene.get_agent_input(frame_idx)
    cams = agent_input.cameras[-1]
    ego = agent_input.ego_statuses[-1]
    lidar_ego = agent_input.lidars[-1].lidar_pc

    def rgb_to_bgra(rgb):
        bgra = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
        bgra[:, :, :3] = rgb[:, :, ::-1]  # RGB -> BGR
        bgra[:, :, 3] = 255
        return bgra

    theta = float(ego.ego_pose[2])
    compass = theta + math.pi / 2.0  # inverse of world_yaw_from_compass

    lidar_sensor = lidar_ego.copy()
    lidar_sensor[:, :3] = lidar_ego[:, :3] - LIDAR_TO_EGO_TRANSLATION

    anno = scene.get_raw_annotation(resolved_idx)
    # Signed scalar speed, exactly as the training loader consumed it. See docstring:
    # the CARLA speedometer sign must be preserved to reconstruct reverse/jitter frames.
    speed = float(anno["speed"])
    return dict(
        camera_images={
            "CAM_FRONT": rgb_to_bgra(cams.cam_f0.image),
            "CAM_FRONT_LEFT": rgb_to_bgra(cams.cam_l0.image),
            "CAM_FRONT_RIGHT": rgb_to_bgra(cams.cam_r0.image),
        },
        lidar_points=lidar_sensor,
        speed=speed,
        compass=compass,
        imu_accel_xy=np.asarray(anno["acceleration"][:2], dtype=np.float32),
        road_option_value=int(anno["command_near"]),
        map_xy=np.asarray(ego.ego_pose[:2], dtype=np.float64),
    )


def feature_max_diffs(gt_features: dict, cl_features: dict) -> dict:
    """Per-feature max absolute difference between two feature dicts."""
    return {
        name: (gt_features[name] - cl_features[name]).abs().max().item()
        for name in FEATURE_NAMES
    }


def compass_formula_error(scene, frame_idx: int):
    """Validate the compass->yaw formula against the recorded CARLA IMU compass.

    The raw annotation stores the IMU compass in ``theta`` and the world yaw in the
    ego bounding box rotation. Returns (abs_error_rad, recorded_compass, world_yaw)
    or None if the annotation has no compass-like field.
    """
    anno = scene.get_raw_annotation(frame_idx)
    if "theta" not in anno:
        return None
    recorded_compass = float(anno["theta"])
    ego_box = next(
        (b for b in anno["bounding_boxes"] if b["class"] == "ego_vehicle"), None
    )
    if ego_box is None:
        return None
    world_yaw = math.radians(ego_box["rotation"][2])
    world_yaw = math.atan2(math.sin(world_yaw), math.cos(world_yaw))
    predicted = world_yaw_from_compass(recorded_compass)
    err = abs(math.atan2(math.sin(predicted - world_yaw), math.cos(predicted - world_yaw)))
    return err, recorded_compass, world_yaw


def discover_scenarios(data_root):
    """Find scenario directories (those containing an ``anno`` subdir)."""
    root = Path(data_root)
    scenarios = sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "anno").is_dir()
    )
    if not scenarios:
        raise FileNotFoundError(
            f"No scenario directories (with an 'anno/' subdir) found under {root}"
        )
    return scenarios


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", required=True, help="Bench2Drive(-Base) dataset root"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario names to load (default: auto-discover under --data-root)",
    )
    parser.add_argument("--map-dir", default=None, help="Optional HD map directory")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--sampling-rate", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--num-history-frames", type=int, default=2)
    parser.add_argument("--num-future-frames", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Also compare model trajectories from gt vs reconstructed features",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for --with-model")
    parser.add_argument(
        "--device", default="cuda", help="Device for --with-model inference"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    scenarios = args.scenarios or discover_scenarios(args.data_root)

    data_config = Bench2DriveDataConfig(
        data_root=Path(args.data_root),
        scenarios=scenarios,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        extract_tar=False,
        map_dir=Path(args.map_dir) if args.map_dir else None,
        bev_cache_dir=None,
    )
    loader = Bench2DriveSceneLoader(data_config)
    tokens = loader.get_scene_tokens()[: args.num_samples]
    if not tokens:
        print("ERROR: no scenes/tokens found for the given config", file=sys.stderr)
        sys.exit(1)

    model_runner = None
    if args.with_model:
        if not args.checkpoint:
            print("ERROR: --with-model requires --checkpoint", file=sys.stderr)
            sys.exit(1)
        from navsim.evaluation.carla_closed_loop.model_runner import ModelRunner

        model_runner = ModelRunner(args.checkpoint, device=args.device)

    config = Bench2DriveConfig()
    builder = Bench2DriveFeatureBuilder(config)

    worst = {name: 0.0 for name in FEATURE_NAMES}
    worst_compass_err = 0.0
    compass_checked = 0
    worst_traj_diff = 0.0
    n_pass = 0
    n_fail = 0
    failures = []

    header = (
        f"{'token':<44} {'camera':>12} {'lidar':>12} {'status':>12} "
        f"{'compass_err':>12}"
    )
    if model_runner is not None:
        header += f" {'traj_diff':>12}"
    print(header)
    print("-" * len(header))

    for token in tokens:
        scene = loader.get_scene(token)
        frame_idx = scene.history_frames

        gt_input = scene.get_agent_input()
        gt_features = builder.compute_features(gt_input)

        kwargs = reconstruct_carla_inputs(scene)
        cl_input = build_agent_input(**kwargs)
        cl_features = builder.compute_features(cl_input)

        diffs = feature_max_diffs(gt_features, cl_features)
        for name in FEATURE_NAMES:
            worst[name] = max(worst[name], diffs[name])

        compass = compass_formula_error(scene, frame_idx)
        compass_str = "n/a"
        if compass is not None:
            compass_err = compass[0]
            worst_compass_err = max(worst_compass_err, compass_err)
            compass_checked += 1
            compass_str = f"{compass_err:.2e}"

        row_ok = all(diffs[name] <= args.atol for name in FEATURE_NAMES)

        traj_str = ""
        if model_runner is not None:
            gt_traj = model_runner.infer(gt_input)
            cl_traj = model_runner.infer(cl_input)
            traj_diff = float(np.abs(gt_traj - cl_traj).max())
            worst_traj_diff = max(worst_traj_diff, traj_diff)
            traj_str = f" {traj_diff:>12.2e}"
            if traj_diff > 1e-4:
                row_ok = False

        print(
            f"{token:<44} {diffs['camera_feature']:>12.2e} "
            f"{diffs['lidar_feature']:>12.2e} {diffs['status_feature']:>12.2e} "
            f"{compass_str:>12}{traj_str}"
        )

        if row_ok:
            n_pass += 1
        else:
            n_fail += 1
            failures.append((token, diffs))

    print("-" * len(header))
    print(f"samples checked : {len(tokens)}")
    print(f"passed          : {n_pass}")
    print(f"failed          : {n_fail}")
    print("worst max-abs-diff per feature (atol={:.0e}):".format(args.atol))
    for name in FEATURE_NAMES:
        print(f"  {name:<16}: {worst[name]:.3e}")
    if compass_checked:
        print(
            f"compass formula : checked {compass_checked} frames, "
            f"worst |world_yaw_from_compass(theta) - ego_yaw| = "
            f"{worst_compass_err:.3e} rad"
        )
    else:
        print(
            "compass formula : SKIPPED (no compass-like field in raw anno; "
            "covered by the CARLA smoke test)"
        )
    if model_runner is not None:
        print(f"model traj      : worst max-abs-diff = {worst_traj_diff:.3e} (atol 1e-4)")

    if n_fail:
        print("\nFAILURES:", file=sys.stderr)
        for token, diffs in failures:
            print(f"  {token}: {diffs}", file=sys.stderr)
        sys.exit(1)

    print("\nPARITY OK: reconstructed features match the training pipeline.")
    sys.exit(0)


if __name__ == "__main__":
    main()
