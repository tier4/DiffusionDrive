# B2D Open-Loop Evaluation Results

## Results (epoch=199, corrected metrics, 50 scenarios, 2095 samples @ 2 Hz)

Produced 2026-07-04 with the fixed evaluation pipeline (commits `33450f8..f866f85`),
`--token-stride 5` (2 Hz keyframe density, VAD-comparable), strict checkpoint load
(0 missing / 0 unexpected keys).
Source: `eval_results/epoch=199-val_traj_loss=12.1346_val_full/results_20260703_185628.json`

| Metric | 1.0s | 2.0s | 3.0s | avg (1/2/3s) |
|---|---|---|---|---|
| L2 (m) | 0.441 | 0.977 | 1.664 | 1.027 |
| Collision, traj-point (%) | 0.00 | 0.10 | 0.34 | 0.15 |
| Collision, ego-box (%) | 0.45 | 0.93 | 1.81 | 1.07 |

Full-horizon (0.5–4.0 s) averages: L2 1.233 m, point-collision 0.70 %, box-collision 2.98 %.
Reference row for context (Bench2DriveZoo VAD open-loop): L2 0.451 / 0.907 / 1.470,
Col% 0.102 / 0.197 / 0.296 — but see the comparability caveat below.

Checkpoint: `b2d_normfix_full_200ep/epoch=199-val_traj_loss=12.1346.ckpt`
(trained 200 epochs vs VAD's 3000; note the LR-schedule wrap issue documented in the
fix-rollout spec means this checkpoint trained with an unintended cyclic LR after
epoch 100 — a retrain is planned in branch B4/OP of
`docs/superpowers/specs/2026-07-03-b2d-fix-rollout-design.md`).

### Metric definitions (corrected, VAD/STP3 semantics)

- **Occupancy**: agents rasterized at their position AT each future timestep
  (`get_future_agents`, nearest-first within 50 m), on the VAD 200×200 @ 0.5 m grid.
- **GT-collision masking**: timesteps where the GT trajectory itself box-collides are
  excluded from both flags (STP3 m1/m2). GT-as-prediction reports exactly 0.
- **Per-horizon collision** = MEAN of per-step flags over steps [0, t]
  (VAD: `plan_obj_col_{t}s = obj_coll.mean()`). Cumulative-any diagnostics are kept
  under `col_any_*` keys.
- **`L2_avg_vad` / `col_avg_vad`** = mean of the 1 s / 2 s / 3 s values only.

### Deviations from Bench2DriveZoo (documented, deliberate)

Our numbers are physically corrected but **not bit-comparable** to
Bench2DriveZoo-published VAD/UniAD tables: ego box centered on the waypoint (no
+0.5 m forward offset), ego box rotated by per-timestep heading (reference sweeps an
axis-aligned footprint), out-of-grid ego-box pixels dropped (reference clips them onto
the border), unified `np.round` quantization. These are also recorded in every results
JSON under `meta.metric_definitions.deviations_from_bench2drivezoo`.

## ⚠ Superseded results

All results produced before 2026-07-04 — including the previously published table
claiming L2 1s = 0.438 and collision rates (0.002 / 0.078 / 0.151) "roughly half of
VAD" — were computed with a broken collision metric and are **superseded**:

1. **Frozen agents**: occupancy used current-frame agent positions copied to every
   future timestep, so moving-actor collisions were missed and vacated positions
   counted as false positives.
2. **No GT-collision masking**: annotation noise and lead-vehicle geometry counted as
   prediction collisions.
3. **`any()` instead of `mean()`** per horizon, inflating rates relative to the VAD
   definition (while the frozen-agent bug simultaneously deflated them — the two
   errors partially cancelled, unpredictably per scene).
4. Dense 10 Hz sampling (10358 samples) overweighted long scenarios ~5× vs VAD's
   2 Hz keyframes.

The L2 numbers were computed correctly and changed only marginally (1s: 0.438 → 0.441,
sampling-density effect); `L2_avg_vad` changed definition (was a 6-horizon mean, now
the VAD 1/2/3 s mean: 0.879 → 1.027 — a definition change, not a regression).

## Historical: why the first evaluation attempt (pre-2026-06-29) was wrong

Three compounding bugs made early DiffusionDrive results on B2D look broken
(L2 of 3.685 at 1s):

1. **Wrong normalization at inference**: plain `TransfuserConfig()` applied NAVSIM
   denormalization (`x=(x+1)/2*56.9-1.2`) to a model trained with B2D stats
   (`x=(x+1)/2*59.6-0.67`), producing a ~13 m systematic offset.
   Fix: `V2TransfuserModelWrapper` with `Bench2DriveConfig`.
2. **Offset-based L2 instead of absolute**: `np.diff()` offsets hide cumulative drift.
   Fix: absolute ego-centric L2 as primary; offset L2 kept as secondary.
3. **Class name mismatch**: `Bench2DriveConfig` vs renamed `Bench2DriveDataConfig`.

## How to Run

Resource safety: run inside Docker with explicit caps and shared memory — the default
64 MB `/dev/shm` crashes multi-worker DataLoaders with bus errors. Check `free -g` and
`df -h` before launching; do not run alongside training/caching jobs.

```bash
# Verify pipeline (GT as prediction -> must give 0.0 L2 AND 0.0 collision)
python3 scripts/evaluation/run_b2d_openloop_eval.py --use-gt --dev-mode --num-workers 4

# Evaluate a checkpoint (VAD-comparable 2 Hz density)
docker run --rm --entrypoint "" --gpus all --cpus=8 --memory=20g --shm-size=4g \
    -v "$PWD":/workspace/DiffusionDrive -v /mnt/nvme1:/mnt/nvme1 \
    -w /workspace/DiffusionDrive -e PYTHONPATH=/workspace/DiffusionDrive \
    diffusiondrive:latest \
    python3 scripts/evaluation/run_b2d_openloop_eval.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --num-workers 8 --token-stride 5 --device cuda
```

Partial checkpoint loads now fail hard; pass `--allow-partial-load` only for debugging
(results are marked suspect in the JSON meta).
