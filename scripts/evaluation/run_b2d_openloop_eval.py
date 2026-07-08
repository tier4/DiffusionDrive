#!/usr/bin/env python3
"""
Open-loop evaluation for DiffusionDrive on Bench2Drive validation set.

Computes:
- Absolute position L2 (VAD-compatible, primary)
- Offset L2 (secondary)
- Collision rates

Usage:
    python scripts/evaluation/run_b2d_openloop_eval.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --dev-mode  # for quick test with 2 scenarios

    python scripts/evaluation/run_b2d_openloop_eval.py \
        --use-gt --dev-mode  # verify pipeline with GT (should give 0.0 L2)
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from navsim.common.bench2drive_dataloader import Bench2DriveDataConfig, Bench2DriveSceneLoader
from navsim.agents.diffusiondrive.transfuser_model_wrapper import V2TransfuserModelWrapper
from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import Bench2DriveFeatureBuilder
from navsim.evaluate.b2d_metrics import B2DOpenLoopMetrics


class B2DEvalDataset(Dataset):
    """Multi-scenario evaluation dataset."""

    def __init__(
        self,
        scenario_paths: List[str],
        feature_builder: Bench2DriveFeatureBuilder,
        use_gt: bool = False,
        token_stride: int = 1,
    ):
        if token_stride < 1:
            raise ValueError(f"token_stride must be >= 1, got {token_stride}")

        self.feature_builder = feature_builder
        self.use_gt = use_gt
        self.samples = []

        print(f"Indexing {len(scenario_paths)} scenarios...")
        for scenario_path in tqdm(scenario_paths, desc="Indexing"):
            path = Path(scenario_path)
            config = Bench2DriveDataConfig(
                data_root=path.parent,
                scenarios=[path.name],
                sampling_rate=1,
                num_frames=50,
                num_history_frames=0,
            )
            loader = Bench2DriveSceneLoader(config)
            tokens = loader.get_scene_tokens()
            tokens = tokens[::token_stride]

            valid = 0
            for token in tokens:
                scene = loader.get_scene(token)
                gt = scene.get_future_trajectory(-1)
                if gt is not None:
                    self.samples.append({
                        "scenario_name": path.name,
                        "token": token,
                        "scene_loader": loader,
                    })
                    valid += 1
            print(f"  {path.name}: {valid}/{len(tokens)} valid")

        if not self.samples:
            raise ValueError("No valid samples found")
        print(f"Total: {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        scene = info["scene_loader"].get_scene(info["token"])
        gt = np.array(scene.get_future_trajectory(-1), dtype=np.float32)

        features = None
        if not self.use_gt:
            agent_input = scene.get_agent_input(-1)
            features = self.feature_builder.compute_features(agent_input)

        agent_states, agent_labels = scene.get_future_agents(-1, num_timesteps=8)

        return {
            "features": features,
            "gt_trajectory": gt,
            "scenario_name": info["scenario_name"],
            "agent_states": np.array(agent_states, dtype=np.float32),
            "agent_labels": np.array(agent_labels, dtype=np.float32),
        }


def eval_collate_fn(batch: List[Dict]) -> Dict:
    """Collate preserving scenario names."""
    has_features = batch[0]["features"] is not None

    batched_features = None
    if has_features:
        keys = batch[0]["features"].keys()
        batched_features = {
            k: torch.stack([item["features"][k] for item in batch]) for k in keys
        }

    gt_trajectories = torch.stack(
        [torch.from_numpy(item["gt_trajectory"]) for item in batch]
    )

    agent_states = torch.stack(
        [torch.from_numpy(item["agent_states"]) for item in batch]
    )
    agent_labels = torch.stack(
        [torch.from_numpy(item["agent_labels"]) for item in batch]
    )

    return {
        "features": batched_features,
        "gt_trajectories": gt_trajectories,
        "scenario_names": [item["scenario_name"] for item in batch],
        "agent_states": agent_states,
        "agent_labels": agent_labels,
    }


def load_model(checkpoint_path: str, config: Bench2DriveConfig, device: str = "cuda", allow_partial_load: bool = False):
    """
    Load DiffusionDrive model with B2D normalization from checkpoint.

    Args:
        checkpoint_path: Path to Lightning checkpoint.
        config: Bench2DriveConfig with correct normalization params.
        device: Target device.
        allow_partial_load: If True, continue even if checkpoint keys don't
            match exactly (missing/unexpected keys). Otherwise raises.

    Returns:
        (model, load_info) where model is in eval mode on the target device
        and load_info is a dict with strict/missing_keys/unexpected_keys.
    """
    model = V2TransfuserModelWrapper(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw_state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip Lightning prefix: agent._transfuser_model.xxx → xxx
    cleaned = {}
    prefix = "agent._transfuser_model."
    for k, v in raw_state_dict.items():
        if k.startswith(prefix):
            cleaned[k[len(prefix):]] = v
        elif k.startswith("agent."):
            cleaned[k[len("agent."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    load_info = {
        "strict": not (missing or unexpected),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
    if missing or unexpected:
        msg = (
            f"Checkpoint mismatch: {len(missing)} missing keys "
            f"(e.g. {missing[:3]}), {len(unexpected)} unexpected keys "
            f"(e.g. {unexpected[:3]}). A partially loaded model produces "
            "garbage metrics."
        )
        if not allow_partial_load:
            raise RuntimeError(msg + " Pass --allow-partial-load to override.")
        print(f"WARNING: {msg} Continuing due to --allow-partial-load.")
    else:
        print(f"Loaded checkpoint (all keys matched): {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model, load_info


def evaluate(
    model,
    dataloader: DataLoader,
    metrics_calc: B2DOpenLoopMetrics,
    device: str = "cuda",
    use_gt: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Run evaluation over all batches.

    Returns:
        (overall_metrics, per_scenario_results)
    """
    per_scenario = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            gt_trajs = batch["gt_trajectories"]  # [B, 8, 3]
            names = batch["scenario_names"]

            if use_gt:
                pred_trajs = gt_trajs.numpy()
            else:
                features = batch["features"]
                for k in features:
                    features[k] = features[k].to(device)
                outputs = model(features)
                pred_trajs = outputs["trajectory"].cpu().numpy()

            agent_states_batch = batch["agent_states"]
            agent_labels_batch = batch["agent_labels"]

            for i in range(len(names)):
                m = metrics_calc.compute_metrics(
                    pred_trajectory=pred_trajs[i],
                    gt_trajectory=gt_trajs[i].numpy(),
                    gt_agent_states=agent_states_batch[i].numpy(),
                    gt_agent_labels=agent_labels_batch[i].numpy().astype(bool),
                )
                per_scenario[names[i]].append(m)

    # Aggregate per scenario
    per_scenario_results = {}
    all_metrics = []
    for name, sample_list in per_scenario.items():
        per_scenario_results[name] = metrics_calc.aggregate_metrics(sample_list)
        per_scenario_results[name]["num_samples"] = len(sample_list)
        all_metrics.extend(sample_list)

    overall = metrics_calc.aggregate_metrics(all_metrics)
    overall["num_samples"] = len(all_metrics)

    return overall, per_scenario_results


def parse_args():
    parser = argparse.ArgumentParser(description="B2D open-loop evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="/mnt/nvme1/dataset/Bench2Drive-Base")
    parser.add_argument(
        "--split-file", type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "splits" / "bench2drive_base_train_val_split.json"),
    )
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dev-mode", action="store_true")
    parser.add_argument("--use-gt", action="store_true")
    parser.add_argument("--allow-partial-load", action="store_true",
                        help="Evaluate even if checkpoint keys are missing/unexpected (results are suspect)")
    parser.add_argument("--token-stride", type=int, default=1,
                        help="Evaluate every Nth 10Hz token (5 ≈ 2Hz VAD keyframe density)")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    if not args.use_gt and not args.checkpoint:
        raise ValueError("Must provide --checkpoint or --use-gt")

    # Load split
    with open(args.split_file) as f:
        split_data = json.load(f)
    scenarios = split_data["val"]

    if args.dev_mode:
        scenarios = scenarios[:2]
        print(f"Dev mode: using {len(scenarios)} scenarios")
    else:
        print(f"Full evaluation: {len(scenarios)} scenarios")

    # Build scenario paths (strip v1/ prefix)
    scenario_paths = []
    for s in scenarios:
        name = s[3:] if s.startswith("v1/") else s
        scenario_paths.append(str(Path(args.data_root) / name))

    # Config
    config = Bench2DriveConfig()

    # Feature builder
    feature_builder = Bench2DriveFeatureBuilder(config)

    # Model
    if not args.use_gt:
        model, load_info = load_model(args.checkpoint, config, args.device, args.allow_partial_load)
    else:
        model = None
        load_info = {"strict": True, "missing_keys": 0, "unexpected_keys": 0}
        print("Using ground truth as predictions (pipeline verification)")

    # Dataset + loader
    dataset = B2DEvalDataset(scenario_paths, feature_builder, use_gt=args.use_gt, token_stride=args.token_stride)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Metrics calculator
    metrics_calc = B2DOpenLoopMetrics(num_timesteps=8, timestep_sec=0.5)

    # Evaluate
    overall, per_scenario = evaluate(
        model, dataloader, metrics_calc, args.device, args.use_gt,
    )

    # Print results
    print(f"\n{'=' * 60}")
    print("Overall Results (Absolute L2 — VAD-compatible)")
    print(f"{'=' * 60}")
    for k, v in overall["absolute_l2"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nOffset L2:")
    for k, v in overall["offset_l2"].items():
        print(f"  {k}: {v:.4f}")
    if overall["collision"]:
        print(f"\nCollision:")
        for k, v in overall["collision"].items():
            print(f"  {k}: {v:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    ckpt_name = Path(args.checkpoint).stem if args.checkpoint else "gt_model"
    mode = "dev" if args.dev_mode else "full"
    result_dir = output_dir / f"{ckpt_name}_val_{mode}"
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "meta": {
            "checkpoint": args.checkpoint or "GT",
            "eval_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": overall["num_samples"],
            "num_scenarios": len(scenarios),
            "split": "val",
            "batch_size": args.batch_size,
            "dev_mode": args.dev_mode,
            "use_gt": args.use_gt,
            "checkpoint_load": load_info,
            "token_stride": args.token_stride,
            "metric_definitions": {
                "collision": "per-horizon mean of per-step flags over [0,t], GT-collision-masked (VAD/STP3); col_any_* = cumulative any()",
                "L2_avg_vad": "mean of period-average L2 at 1s/2s/3s",
                "agent_occupancy": "per-future-timestep agent states (get_future_agents)",
                "deviations_from_bench2drivezoo": (
                    "ego box centered on waypoint (no +0.5m forward offset), "
                    "ego box rotated by per-timestep heading (reference sweeps axis-aligned), "
                    "out-of-grid ego-box pixels dropped (reference clips onto border), "
                    "unified np.round quantization — physically corrected, not bit-identical "
                    "to Bench2DriveZoo-published numbers"
                ),
            },
        },
        "results": overall,
        "per_scenario_results": per_scenario,
    }

    output_path = result_dir / f"results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nEvaluation complete in {elapsed / 60:.1f}m")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
