# B2D Open-Loop Evaluation Results

## Results (epoch=199, 50 scenarios, 10358 samples)

|  | L2 1s | L2 2s | L2 3s | Col% 1s | Col% 2s | Col% 3s |
|---|---|---|---|---|---|---|
| VAD-b2d-benchmark | 0.451 | 0.907 | 1.470 | 0.102 | 0.197 | 0.296 |
| **DiffusionDrive (ours)** | **0.438** | 0.974 | 1.666 | **0.002** | **0.078** | **0.151** |

- L2 at 1s beats VAD; slightly behind at 2-3s (trained 200 epochs vs VAD's 3000)
- Collision rates roughly half of VAD at every horizon

Checkpoint: `b2d_normfix_full_200ep/epoch=199-val_traj_loss=12.1346.ckpt`

## Why Previous Evaluation Was Wrong

Three compounding bugs made previous DiffusionDrive results on B2D look broken (L2 of 3.685 at 1s vs VAD's 0.451):

### 1. Wrong normalization at inference (critical)

The previous evaluation created a plain `TransfuserConfig()` which uses hardcoded NAVSIM normalization in `denorm_odo`:

```
x = (x+1)/2 * 56.9 - 1.2    # NAVSIM
y = (y+1)/2 * 46.0 - 20.0    # NAVSIM
```

But the model was trained with B2D normalization (different data range):

```
x = (x+1)/2 * 59.6 - 0.67   # B2D
y = (y+1)/2 * 64.6 - 33.0   # B2D
```

The diffusion model produces values in [-1, 1] normalized space. Denormalizing with NAVSIM scale when the model learned B2D scale produces a ~13m systematic offset on Y alone, making every prediction garbage.

**Fix:** Use `V2TransfuserModelWrapper` with `Bench2DriveConfig`, which loads the correct normalization buffers from the checkpoint.

### 2. Offset-based L2 instead of absolute (incorrect metric)

The previous evaluation converted trajectories to frame-to-frame offsets via `np.diff()` and computed L2 on those. VAD evaluates L2 on absolute ego-centric positions (distance from ego at T=0). Offset L2 hides cumulative drift — if the prediction drifts 3m by T=3s, absolute L2 correctly reports 3m, but offset L2 only shows the small per-step error.

**Fix:** Compute L2 on absolute positions as the primary metric. Offset L2 kept as secondary for reference.

### 3. Class name mismatch

The reference code imported `Bench2DriveConfig` from `navsim.common.bench2drive_dataloader`, but this was renamed to `Bench2DriveDataConfig`.

**Fix:** Use the correct class name.

## How to Run

```bash
# Verify pipeline (GT as prediction -> should give 0.0 L2)
python3 scripts/evaluation/run_b2d_openloop_eval.py --use-gt --dev-mode

# Evaluate a checkpoint
python3 scripts/evaluation/run_b2d_openloop_eval.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --batch-size 4 --num-workers 0 --device cpu
```
