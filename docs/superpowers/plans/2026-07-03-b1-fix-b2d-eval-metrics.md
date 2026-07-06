# B1 — fix/b2d-eval-metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the B2D open-loop collision/L2 metrics actually match the VAD/STP3 definition (per-timestep agent occupancy, GT-collision masking, mean-based horizons) and make the eval script fail loudly on partial checkpoint loads.

**Architecture:** All metric math lives in `navsim/evaluate/b2d_planning_utils.py` (`PlanningMetric`) and `navsim/evaluate/b2d_metrics.py` (`B2DOpenLoopMetrics`). The scene (`navsim/common/bench2drive_scene.py`) gains a `get_future_agents()` accessor that returns agent states at each future waypoint time in the *current* ego frame; the eval script wires those through. The reference implementation being matched is `reference/Bench2DriveZoo/mmcv/models/dense_heads/planning_head_plugin/metric_stp3.py` (read it whenever semantics are unclear).

**Tech Stack:** Python 3.9/3.10, numpy, torch, cv2 (already a dependency), pytest.

## Global Constraints

- **Branch:** create `fix/b2d-eval-metrics` from the current `feature/b2d-openloop-eval` HEAD (PR #8 has not merged yet; the files only exist there). After PR #8 merges to `tier4-main`, rebase/retarget this branch onto `tier4-main` — the diff collapses to just these fixes.
- **NEVER `pip install` anything on the host.** The host env is delicate ($10M machine). If `pytest`/`torch`/`cv2` are missing on the host, run tests inside the project Docker container (`docker/run.sh` pattern), capped at **≤20 GB RAM / ≤8 CPUs**.
- **Resource safety (mandatory):** before any eval run, check `free -g` and `df -h /mnt/nvme1` (results JSON is small, but the model + dataloader workers are not). Use `--dev-mode` first, `--num-workers` ≤ 8, and never launch a full-split eval while training or caching is running.
- **Do not change training-path behavior.** `get_agents()` semantics (32 m circular filter etc.) stay untouched in this branch — B4 owns those changes. This branch only *adds* accessors and fixes eval-side code.
- **Coordinate conventions:** CARLA left-handed (+X fwd, +Y right, heading CW); the collision grid is the deliberate VAD convention — 200×200 @ 0.5 m over [−50, 50] m, nuScenes ego dims 1.85×4.084. Do NOT "fix" these to B2D values; they are kept for VAD comparability.
- Commit after every task with the exact messages given. No `git add -A` — stage only the named files.

## Grid math cheat-sheet (used by test assertions)

`bev_start_position = -49.75`, `bev_resolution = 0.5`, `pixel(v) = np.round((v + 49.75) / 0.5)`.
So: x=0 → row 100, x=5 → row 110, x=10 → row 120, y=0 → col 100. A 4.0×2.0 agent at (5, 0) spans rows ≈106–114, cols ≈98–102.

---

### Task 1: `cv2.fillPoly` rasterization helper in `PlanningMetric`

**Files:**
- Modify: `navsim/evaluate/b2d_planning_utils.py`
- Test: `tests/test_b2d_planning_utils.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `PlanningMetric._rasterize_polygon(pixel_corners: np.ndarray[(4,2) int (row,col)], height: int, width: int) -> Tuple[np.ndarray, np.ndarray]` returning `(rr, cc)` int32 index arrays of filled pixels, already clipped to the canvas. Tasks 2 and 3 call this.

- [ ] **Step 1: Write the failing test** — append to `tests/test_b2d_planning_utils.py`:

```python
def test_rasterize_polygon_fills_box():
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    pm = PlanningMetric()
    # axis-aligned box in pixel space: rows 10..20, cols 30..40 (corners as (row, col))
    corners = np.array([[10, 30], [10, 40], [20, 40], [20, 30]], dtype=np.int32)
    rr, cc = pm._rasterize_polygon(corners, 200, 200)

    filled = set(zip(rr.tolist(), cc.tolist()))
    assert (15, 35) in filled          # interior
    assert (10, 30) in filled          # corner (cv2 includes boundary)
    assert rr.min() >= 10 and rr.max() <= 20
    assert cc.min() >= 30 and cc.max() <= 40


def test_rasterize_polygon_clips_out_of_bounds():
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    pm = PlanningMetric()
    # box partially outside the canvas must be clipped, not wrapped or crashed
    corners = np.array([[-5, -5], [-5, 5], [5, 5], [5, -5]], dtype=np.int32)
    rr, cc = pm._rasterize_polygon(corners, 200, 200)
    assert len(rr) > 0
    assert rr.min() >= 0 and cc.min() >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_b2d_planning_utils.py::test_rasterize_polygon_fills_box -v`
Expected: FAIL with `AttributeError: 'PlanningMetric' object has no attribute '_rasterize_polygon'`

- [ ] **Step 3: Implement.** In `navsim/evaluate/b2d_planning_utils.py`, add `import cv2` at the top (after `import numpy as np`), and add this method to `PlanningMetric` (after `_world_to_pixel`):

```python
    def _rasterize_polygon(self, pixel_corners, height, width):
        """Fill a polygon on a (height, width) canvas.

        Args:
            pixel_corners: [4, 2] integer corners as (row, col).

        Returns:
            (rr, cc): int32 arrays of filled pixel indices, clipped to canvas.
        """
        canvas = np.zeros((height, width), dtype=np.uint8)
        # cv2 points are (x=col, y=row)
        cv2.fillPoly(canvas, [pixel_corners[:, [1, 0]].astype(np.int32)], 1)
        rr, cc = np.nonzero(canvas)
        return rr.astype(np.int32), cc.astype(np.int32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_b2d_planning_utils.py -v`
Expected: both new tests PASS; the three pre-existing tests still PASS (nothing else changed yet).

- [ ] **Step 5: Commit**

```bash
git add navsim/evaluate/b2d_planning_utils.py tests/test_b2d_planning_utils.py
git commit -m "feat: add cv2.fillPoly polygon rasterizer to PlanningMetric"
```

---

### Task 2: Per-timestep agent occupancy in `get_label`

The frozen-agent bug: `get_label` currently rasterizes each agent once and copies it to all T timesteps. The reference (`metric_stp3.py:118-133`) rasterizes agents at their *per-timestep* positions.

**Files:**
- Modify: `navsim/evaluate/b2d_planning_utils.py:128-174` (`get_label`)
- Test: `tests/test_b2d_planning_utils.py`

**Interfaces:**
- Consumes: `_rasterize_polygon` from Task 1.
- Produces: **new signature** `get_label(gt_agent_states, gt_agent_labels, num_timesteps=6)` where `gt_agent_states` is `[B, T, N, 5]` or `[T, N, 5]` (x, y, heading, length, width per timestep, current-ego frame) and `gt_agent_labels` is `[B, T, N]` or `[T, N]` bool. Returns `(segmentation, pedestrian)` each `[B, T, H, W]` torch tensors (pedestrian stays all-zeros — documented API-compat placeholder). Static `[N, 5]`/`[B, N, 5]` input now raises `ValueError`. Tasks 4 and 6 depend on this shape.

- [ ] **Step 1: Write the failing test** — append to `tests/test_b2d_planning_utils.py`:

```python
def test_moving_agent_occupancy_tracks_motion():
    """Occupancy at timestep t must reflect where agents are AT t, not at t=0."""
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    pm = PlanningMetric()
    T, N = 2, 1
    states = np.zeros((T, N, 5), dtype=np.float32)
    labels = np.ones((T, N), dtype=bool)
    states[0, 0] = [5.0, 0.0, 0.0, 4.0, 2.0]    # t=0: agent at x=5
    states[1, 0] = [10.0, 0.0, 0.0, 4.0, 2.0]   # t=1: agent moved to x=10

    seg, _ = pm.get_label(states, labels, num_timesteps=2)

    assert seg.shape == (1, 2, 200, 200)
    # pixel of (5,0) is (110,100); of (10,0) is (120,100) — see grid cheat-sheet
    assert seg[0, 0, 110, 100] == 1 and seg[0, 0, 120, 100] == 0
    assert seg[0, 1, 120, 100] == 1 and seg[0, 1, 110, 100] == 0


def test_get_label_rejects_static_agent_states():
    """The old frozen-at-t0 [N, 5] input shape must be an explicit error."""
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    pm = PlanningMetric()
    states = np.zeros((5, 5), dtype=np.float32)   # legacy [N, 5]
    labels = np.zeros(5, dtype=bool)
    with pytest.raises(ValueError, match="per-timestep"):
        pm.get_label(states, labels, num_timesteps=2)
```

Add `import pytest` at the top of the file if not already present.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_b2d_planning_utils.py::test_moving_agent_occupancy_tracks_motion -v`
Expected: FAIL (old code broadcasts `[T, N, 5]` as `[B=T, N=N, 5]` and produces the wrong shape `(2, 2, 200, 200)` or wrong content).

- [ ] **Step 3: Replace `get_label` entirely** with:

```python
    def get_label(
        self,
        gt_agent_states: np.ndarray,
        gt_agent_labels: np.ndarray,
        num_timesteps: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate per-timestep BEV occupancy grids from agent states.

        Matches STP3/VAD get_birds_eye_view_label: agents are rasterized at
        their position AT each timestep (metric_stp3.py:118-133), never
        frozen at t=0.

        Args:
            gt_agent_states: [B, T, N, 5] or [T, N, 5] of
                (x, y, heading, length, width) in the current ego frame.
            gt_agent_labels: [B, T, N] or [T, N] boolean validity mask.
            num_timesteps: Number of future timesteps T to rasterize.

        Returns:
            (segmentation, pedestrian) each [B, T, H, W]. The pedestrian
            tensor is always zeros (walkers are merged into segmentation
            upstream); it is kept for VAD API compatibility only.
        """
        gt_agent_states = np.asarray(gt_agent_states)
        gt_agent_labels = np.asarray(gt_agent_labels)
        if gt_agent_states.ndim == 3:
            gt_agent_states = gt_agent_states[np.newaxis, ...]
            gt_agent_labels = gt_agent_labels[np.newaxis, ...]
        if gt_agent_states.ndim != 4:
            raise ValueError(
                "gt_agent_states must be per-timestep [B, T, N, 5] or [T, N, 5]; "
                f"got shape {gt_agent_states.shape}. Static [N, 5] states would "
                "freeze agents at t=0 and invalidate the collision metric."
            )
        if gt_agent_states.shape[1] < num_timesteps:
            raise ValueError(
                f"Need agent states for {num_timesteps} timesteps, "
                f"got {gt_agent_states.shape[1]}"
            )

        B = gt_agent_states.shape[0]
        T = num_timesteps
        H = int(self.bev_dimension[0])
        W = int(self.bev_dimension[1])

        segmentation = torch.zeros((B, T, H, W))
        pedestrian = torch.zeros((B, T, H, W))

        for b in range(B):
            for t in range(T):
                for agent_idx in range(gt_agent_states.shape[2]):
                    if not gt_agent_labels[b, t, agent_idx]:
                        continue
                    x, y, heading, length, width = gt_agent_states[b, t, agent_idx]
                    corners = self._get_agent_corners(x, y, heading, length, width)
                    pixel_corners = self._world_to_pixel(corners)
                    rr, cc = self._rasterize_polygon(pixel_corners, H, W)
                    if len(rr) > 0:
                        segmentation[b, t, rr, cc] = 1

        return segmentation, pedestrian
```

Also delete the now-unused module-level `polygon_simple` usage inside `get_label` only — keep the `polygon_simple` function itself (still used by `evaluate_coll` until Task 3 and by `test_polygon_rasterization`).

- [ ] **Step 4: Update the two pre-existing tests in `tests/test_b2d_planning_utils.py`** to the new shapes:

In `test_planning_metric_no_agents` (line ~13), change:

```python
    agent_states = np.zeros((1, 4, 5, 5), dtype=np.float32)  # [B, T, N, 5]
    agent_labels = np.zeros((1, 4, 5), dtype=bool)
```

In `test_planning_metric_collision_free`, make the states/labels per-timestep the same way (add the T axis sized to the `num_timesteps` used there).

- [ ] **Step 5: Run the whole file**

Run: `pytest tests/test_b2d_planning_utils.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add navsim/evaluate/b2d_planning_utils.py tests/test_b2d_planning_utils.py
git commit -m "fix: rasterize agents per-timestep in collision occupancy (frozen-agent bug)"
```

---

### Task 3: GT-collision masking + unified quantization in `evaluate_coll`

The reference (`metric_stp3.py:267-287`) masks out timesteps where the GT trajectory itself box-collides (`m1 = logical_and(m1, logical_not(gt_box_coll))`, `m2 = logical_not(gt_box_coll)`). Our `evaluate_coll` ignores `gt_trajs` entirely, and quantizes points with `int()` truncation while boxes use `np.round`.

**Files:**
- Modify: `navsim/evaluate/b2d_planning_utils.py:176-230` (`evaluate_coll`)
- Test: `tests/test_b2d_planning_utils.py`

**Interfaces:**
- Consumes: `_rasterize_polygon` (Task 1), per-timestep occupancy (Task 2).
- Produces: same signature `evaluate_coll(trajs [B,T,3], gt_trajs [B,T,3], occupancy [B,T,H,W]) -> (obj_coll, obj_box_coll)` each `[B, T]` — but `gt_trajs` is now USED for masking. New helpers: `_traj_point_to_pixel(x, y) -> (px, py)` and `_box_collision_flags(traj [T,>=2], occupancy [T,H,W]) -> np.ndarray[T] bool`.

- [ ] **Step 1: Write the failing tests** — append:

```python
def _single_parked_agent_occupancy(pm, T, x=5.0, y=0.0):
    states = np.zeros((T, 1, 5), dtype=np.float32)
    labels = np.ones((T, 1), dtype=bool)
    states[:, 0] = [x, y, 0.0, 4.0, 2.0]
    seg, _ = pm.get_label(states, labels, num_timesteps=T)
    return seg


def test_gt_collision_timesteps_are_masked():
    """STP3 semantics: if the GT trajectory itself collides at t, that t is
    excluded — GT-as-prediction must report zero collisions."""
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    pm = PlanningMetric()
    seg = _single_parked_agent_occupancy(pm, T=2)
    # pred == gt, both drive straight through the parked agent at t=0
    traj = np.array([[[5.0, 0.0, 0.0], [12.0, 0.0, 0.0]]], dtype=np.float32)

    obj_coll, obj_box_coll = pm.evaluate_coll(traj, traj.copy(), seg)
    assert obj_coll.sum() == 0
    assert obj_box_coll.sum() == 0


def test_pred_collision_with_offpath_agent_detected():
    """A predicted trajectory hitting an agent the GT avoids IS a collision."""
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    pm = PlanningMetric()
    seg = _single_parked_agent_occupancy(pm, T=1)
    pred = np.array([[[5.0, 0.0, 0.0]]], dtype=np.float32)   # drives into agent
    gt = np.array([[[5.0, 10.0, 0.0]]], dtype=np.float32)    # GT swerves clear

    obj_coll, obj_box_coll = pm.evaluate_coll(pred, gt, seg)
    assert obj_coll[0, 0] == 1
    assert obj_box_coll[0, 0] == 1


def test_point_quantization_rejects_out_of_grid():
    """int() truncation used to map x=-50.4 into cell 0; np.round must not."""
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    pm = PlanningMetric()
    px, _ = pm._traj_point_to_pixel(-50.4, 0.0)
    assert px < 0  # out of grid → rejected by the bounds check
    px, _ = pm._traj_point_to_pixel(-49.9, 0.0)
    assert px == 0  # genuinely inside cell 0 ([-50, -49.5))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_b2d_planning_utils.py::test_gt_collision_timesteps_are_masked tests/test_b2d_planning_utils.py::test_point_quantization_rejects_out_of_grid -v`
Expected: FAIL (`_traj_point_to_pixel` missing; masking test reports collisions).

- [ ] **Step 3: Replace `evaluate_coll` and add helpers:**

```python
    def _traj_point_to_pixel(self, x, y):
        """Quantize a trajectory point with the SAME rounding as _world_to_pixel."""
        px = int(np.round((x - self.bev_start_position[0]) / self.bev_resolution[0]))
        py = int(np.round((y - self.bev_start_position[1]) / self.bev_resolution[1]))
        return px, py

    def _box_collision_flags(self, traj, occupancy):
        """Per-timestep ego-box collision flags for one trajectory.

        Args:
            traj: [T, >=2] (x, y[, heading]) trajectory.
            occupancy: [T, H, W] occupancy grid (tensor or ndarray).

        Returns:
            np.ndarray [T] bool.
        """
        H = int(self.bev_dimension[0])
        W = int(self.bev_dimension[1])
        T = traj.shape[0]
        flags = np.zeros(T, dtype=bool)
        for t in range(T):
            x, y = traj[t, 0], traj[t, 1]
            heading = traj[t, 2] if traj.shape[1] > 2 else 0.0
            corners = self._get_agent_corners(x, y, heading, self.H, self.W)
            pixel_corners = self._world_to_pixel(corners)
            rr, cc = self._rasterize_polygon(pixel_corners, H, W)
            if len(rr) > 0:
                occ = occupancy[t]
                if torch.is_tensor(occ):
                    occ = occ.numpy()
                flags[t] = bool(np.any(occ[rr, cc] > 0))
        return flags

    def evaluate_coll(
        self,
        trajs: np.ndarray,
        gt_trajs: np.ndarray,
        occupancy: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check trajectory collisions against per-timestep occupancy grids.

        STP3/VAD semantics (metric_stp3.py:267-287): timesteps where the GT
        trajectory itself box-collides are masked out of BOTH the point and
        box collision flags (annotation noise / unavoidable-contact filter).

        Args:
            trajs: [B, T, 3] predicted trajectories (x, y, heading).
            gt_trajs: [B, T, 3] ground truth — used for collision masking.
            occupancy: [B, T, H, W] occupancy grids.

        Returns:
            (obj_coll, obj_box_coll) each [B, T] {0,1} tensors.
        """
        if torch.is_tensor(trajs):
            trajs = trajs.cpu().numpy()
        if torch.is_tensor(gt_trajs):
            gt_trajs = gt_trajs.cpu().numpy()

        B, T, _ = trajs.shape
        H = int(self.bev_dimension[0])
        W = int(self.bev_dimension[1])

        obj_coll = np.zeros((B, T))
        obj_box_coll = np.zeros((B, T))

        for b in range(B):
            occ_b = occupancy[b]
            gt_box_coll = self._box_collision_flags(gt_trajs[b], occ_b)

            for t in range(T):
                if gt_box_coll[t]:
                    continue  # STP3 m1 mask
                x, y = trajs[b, t, 0], trajs[b, t, 1]
                px, py = self._traj_point_to_pixel(x, y)
                if 0 <= px < H and 0 <= py < W:
                    occ = occ_b[t]
                    if torch.is_tensor(occ):
                        occ = occ.numpy()
                    if occ[px, py] > 0:
                        obj_coll[b, t] = 1

            pred_box_coll = self._box_collision_flags(trajs[b], occ_b)
            # STP3 m2 mask
            obj_box_coll[b] = np.logical_and(
                pred_box_coll, np.logical_not(gt_box_coll)
            ).astype(float)

        return torch.tensor(obj_coll), torch.tensor(obj_box_coll)
```

Then delete `polygon_simple` **only if** `test_polygon_rasterization` is updated in the same step to test `_rasterize_polygon` instead; otherwise keep it. Preferred: update the test and delete the dead ray-caster (also delete `_gen_dx_bx` and the unused `self.dx`/`self.bx` attributes — pure dead code; keep `_calc_bev_params`).

- [ ] **Step 4: Run the whole file**

Run: `pytest tests/test_b2d_planning_utils.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add navsim/evaluate/b2d_planning_utils.py tests/test_b2d_planning_utils.py
git commit -m "fix: add STP3 GT-collision masking and unify pixel quantization in evaluate_coll"
```

---

### Task 4: VAD metric definitions in `B2DOpenLoopMetrics`

Two definition bugs: per-horizon collision uses cumulative `any()` where VAD uses `mean()` over steps [0, t] (`VAD.py: plan_obj_col_{t}s = obj_coll.mean()`), and `L2_avg_vad` averages six horizons where VAD averages only 1s/2s/3s.

**Files:**
- Modify: `navsim/evaluate/b2d_metrics.py`
- Test: `tests/test_b2d_metrics.py`, `tests/test_eval_collision_wiring.py`

**Interfaces:**
- Consumes: Task 2/3 `PlanningMetric` (per-timestep `get_label`, masked `evaluate_coll`).
- Produces: `compute_metrics(pred_trajectory [T,3], gt_trajectory [T,3], gt_agent_states [T,N,5], gt_agent_labels [T,N])` — **agent args are now per-timestep** (3-D). Collision dict keys: `col_{h}s`, `box_col_{h}s` (mean-based), `col_any_{h}s`, `box_col_any_{h}s` (diagnostic), `col_avg_vad`, `box_col_avg_vad`, `col_avg_full`, `box_col_avg_full`. The old `col_avg`/`box_col_avg` keys are REMOVED. L2 dicts unchanged except `L2_avg_vad` now averages the 1.0s/2.0s/3.0s values only.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_b2d_metrics.py`:

```python
def test_l2_avg_vad_uses_integer_horizons_only():
    """VAD's headline average is mean(L2@1s, L2@2s, L2@3s), not all six horizons."""
    from navsim.evaluate.b2d_metrics import B2DOpenLoopMetrics

    metrics = B2DOpenLoopMetrics(num_timesteps=8, timestep_sec=0.5)
    gt = np.zeros((8, 3), dtype=np.float32)
    pred = gt.copy()
    # per-step L2 error = 1, 2, ..., 8  → period_avg(t) = (t+2)/2
    pred[:, 1] = np.arange(1, 9)

    result = metrics.compute_metrics(pred, gt)
    l2 = result["absolute_l2"]
    # period averages: 1s→1.5, 2s→2.5, 3s→3.5
    assert abs(l2["L2_1.0s"] - 1.5) < 1e-6
    assert abs(l2["L2_2.0s"] - 2.5) < 1e-6
    assert abs(l2["L2_3.0s"] - 3.5) < 1e-6
    assert abs(l2["L2_avg_vad"] - 2.5) < 1e-6   # (1.5+2.5+3.5)/3, NOT 2.25


def test_collision_horizons_are_mean_not_any():
    """VAD reports mean per-step collision fraction up to each horizon."""
    from navsim.evaluate.b2d_metrics import B2DOpenLoopMetrics

    metrics = B2DOpenLoopMetrics(num_timesteps=8, timestep_sec=0.5)
    T = 8
    pred = np.zeros((T, 3), dtype=np.float32)
    pred[:, 0] = np.arange(1, 9)          # drives forward through x=1..8
    gt = pred.copy()
    gt[:, 1] = 10.0                        # GT is far off to the side (no masking)

    # agent sits on the pred waypoint x=1 ONLY at t=0, then disappears
    states = np.zeros((T, 1, 5), dtype=np.float32)
    labels = np.zeros((T, 1), dtype=bool)
    states[0, 0] = [1.0, 0.0, 0.0, 4.0, 2.0]
    labels[0, 0] = True

    col = metrics.compute_metrics(pred, gt, states, labels)["collision"]
    # per-step flags = [1, 0, 0, 0, 0, 0, 0, 0]
    assert abs(col["col_0.5s"] - 1.0) < 1e-6
    assert abs(col["col_1.0s"] - 0.5) < 1e-6
    assert abs(col["col_2.0s"] - 0.25) < 1e-6
    assert abs(col["col_3.0s"] - 1.0 / 6.0) < 1e-6
    assert abs(col["col_any_1.0s"] - 1.0) < 1e-6      # diagnostic keeps any()
    assert abs(col["col_avg_full"] - 1.0 / 8.0) < 1e-6
    expected_vad = (0.5 + 0.25 + 1.0 / 6.0) / 3.0
    assert abs(col["col_avg_vad"] - expected_vad) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_b2d_metrics.py::test_l2_avg_vad_uses_integer_horizons_only tests/test_b2d_metrics.py::test_collision_horizons_are_mean_not_any -v`
Expected: FAIL (`L2_avg_vad` == 2.25; collision keys/values wrong).

- [ ] **Step 3: Implement in `navsim/evaluate/b2d_metrics.py`:**

At module top, replace the `_VAD_NUM_TIMESTEPS = 6` constant block with:

```python
# VAD reports 1s/2s/3s horizons; at 0.5s per step these are indices 1, 3, 5.
_VAD_NUM_TIMESTEPS = 6
_VAD_HORIZON_INDICES = (1, 3, 5)
```

In `__init__`, add validation and a shared `PlanningMetric` instance (moves the import to module level: `from navsim.evaluate.b2d_planning_utils import PlanningMetric`):

```python
        if abs(timestep_sec - 0.5) > 1e-9:
            raise ValueError(
                f"VAD horizon indices assume timestep_sec=0.5, got {timestep_sec}"
            )
        self._planning_metric = PlanningMetric()
```

In `_compute_horizon_metrics`, replace the `L2_avg_vad` line:

```python
        vad_values = [all_horizon_values[i] for i in _VAD_HORIZON_INDICES if i < T]
        result["L2_avg_vad"] = float(np.mean(vad_values)) if vad_values else 0.0
```

Replace `_compute_collision` entirely:

```python
    def _compute_collision(
        self,
        pred_traj: np.ndarray,
        gt_traj: np.ndarray,
        gt_agent_states: np.ndarray,
        gt_agent_labels: np.ndarray,
        T: int,
    ) -> Dict[str, float]:
        """Collision rates with VAD/STP3 semantics.

        Per-horizon values are the MEAN of per-step collision flags over
        steps [0, t] (VAD: plan_obj_col_{t}s = obj_coll.mean()). Timesteps
        where the GT trajectory itself collides are masked inside
        evaluate_coll. col_any_* keys keep the cumulative-any diagnostic.
        """
        import torch

        gt_agent_states = np.asarray(gt_agent_states)
        gt_agent_labels = np.asarray(gt_agent_labels).astype(bool)
        if gt_agent_states.ndim == 3:  # [T, N, 5] → [1, T, N, 5]
            gt_agent_states = gt_agent_states[np.newaxis, ...]
            gt_agent_labels = gt_agent_labels[np.newaxis, ...]

        seg, ped = self._planning_metric.get_label(
            gt_agent_states, gt_agent_labels, num_timesteps=T
        )
        occupancy = torch.logical_or(seg, ped)

        obj_coll, obj_box_coll = self._planning_metric.evaluate_coll(
            pred_traj[np.newaxis, ...], gt_traj[np.newaxis, ...], occupancy
        )

        result = {}
        for t in range(T):
            label = f"{(t + 1) * self.timestep_sec:.1f}s"
            result[f"col_{label}"] = float(obj_coll[0, : t + 1].float().mean())
            result[f"box_col_{label}"] = float(obj_box_coll[0, : t + 1].float().mean())
            result[f"col_any_{label}"] = float(obj_coll[0, : t + 1].any().float())
            result[f"box_col_any_{label}"] = float(obj_box_coll[0, : t + 1].any().float())

        for prefix, flags in (("col", obj_coll), ("box_col", obj_box_coll)):
            vad_values = [
                result[f"{prefix}_{(i + 1) * self.timestep_sec:.1f}s"]
                for i in _VAD_HORIZON_INDICES
                if i < T
            ]
            result[f"{prefix}_avg_vad"] = float(np.mean(vad_values)) if vad_values else 0.0
            result[f"{prefix}_avg_full"] = float(flags.float().mean())

        return result
```

Update the `compute_metrics` docstring: `gt_agent_states: Optional [T, N, 5]`, `gt_agent_labels: Optional [T, N]`.

- [ ] **Step 4: Update stale assertions.** In `tests/test_eval_collision_wiring.py`, change agent fixtures to per-timestep shapes and the avg keys:

```python
    agent_states = np.zeros((8, 5, 5), dtype=np.float32)   # [T, N, 5]
    agent_labels = np.zeros((8, 5), dtype=bool)             # [T, N]
```

and replace `assert "col_avg" in col` / `assert "box_col_avg" in col` with `col_avg_vad`/`col_avg_full`/`box_col_avg_vad`/`box_col_avg_full`; same in `test_aggregate_preserves_collision_metrics` (both fixture shapes and the aggregated keys).

- [ ] **Step 5: Run all metric tests**

Run: `pytest tests/test_b2d_metrics.py tests/test_eval_collision_wiring.py tests/test_b2d_planning_utils.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add navsim/evaluate/b2d_metrics.py tests/test_b2d_metrics.py tests/test_eval_collision_wiring.py
git commit -m "fix: use VAD mean-based collision horizons and 1s/2s/3s L2 average"
```

---

### Task 5: `Bench2DriveScene.get_future_agents()`

The eval needs agent states at each future waypoint time, transformed into the *current* ego frame — mirroring how `get_future_trajectory` walks future annotation frames.

**Files:**
- Modify: `navsim/common/bench2drive_scene.py:717-876` (factor helper out of `get_agents`), `navsim/common/bench2drive_constants.py`
- Test: `tests/test_bench2drive_future_agents.py` (new)

**Interfaces:**
- Consumes: existing `_load_annotation`, `transform_points_to_ego`, `transform_heading_to_ego`, `normalize_angle`, `FUTURE_TRAJECTORY_FRAME_STRIDE`, `MAX_AGENTS`, `B2D_CLASS_TO_NAVSIM` (all already imported in the file — verify the import list at the top and extend it if a name is missing).
- Produces:
  - `bench2drive_constants.COLLISION_EVAL_RANGE_M = 50.0` (collision grid half-extent; matches `PlanningMetric.X_BOUND`).
  - `Bench2DriveScene._extract_agents_from_anno(anno, ego_points, ego_heading_rad, max_distance, use_anno_distance) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(agent_states [MAX_AGENTS,5] float32, agent_labels [MAX_AGENTS] bool, agent_types [MAX_AGENTS] int32)`.
  - `Bench2DriveScene.get_future_agents(frame_idx=-1, num_timesteps=8) -> Tuple[torch.Tensor, torch.Tensor]` with shapes `[T, MAX_AGENTS, 5]` float32 and `[T, MAX_AGENTS]` bool. Task 6 consumes this.
- **Behavioral guarantee:** `get_agents()` output is bit-identical to before (it delegates to the helper with `use_anno_distance=True, max_distance=BENCH2DRIVE_LIDAR_RANGE_M / 2`).

- [ ] **Step 1: Add the constant.** In `navsim/common/bench2drive_constants.py`, next to the other BEV constants:

```python
# Half-extent of the VAD-convention collision-evaluation grid ([-50, 50] m).
# Used by get_future_agents so occupancy covers everywhere the ego can reach
# within the 4 s horizon; must match PlanningMetric.X_BOUND/Y_BOUND.
COLLISION_EVAL_RANGE_M = 50.0
```

- [ ] **Step 2: Write the failing test** — create `tests/test_bench2drive_future_agents.py`:

```python
"""Tests for per-future-timestep agent extraction (collision metric wiring)."""

import numpy as np
import torch


def test_get_future_agents_shapes_and_range(sample_config):
    from navsim.common.bench2drive_dataloader import Bench2DriveSceneLoader
    from navsim.common.bench2drive_constants import MAX_AGENTS, COLLISION_EVAL_RANGE_M

    loader = Bench2DriveSceneLoader(sample_config)
    token = loader.get_scene_tokens()[0]
    scene = loader.get_scene(token)

    states, labels = scene.get_future_agents(-1, num_timesteps=4)

    assert states.shape == (4, MAX_AGENTS, 5)
    assert labels.shape == (4, MAX_AGENTS)
    assert labels.dtype == torch.bool
    valid = states[labels]
    if len(valid):
        dist = torch.hypot(valid[:, 0], valid[:, 1])
        assert dist.max() <= COLLISION_EVAL_RANGE_M + 1e-3


def test_get_agents_unchanged_by_refactor(sample_config):
    """get_agents must be byte-identical before/after the helper extraction —
    this pins the training-path behavior B4 will change later, not B1."""
    from navsim.common.bench2drive_dataloader import Bench2DriveSceneLoader

    loader = Bench2DriveSceneLoader(sample_config)
    token = loader.get_scene_tokens()[0]
    scene = loader.get_scene(token)

    states, labels, types = scene.get_agents(-1)
    assert states.shape[1] == 5
    assert states[labels].abs().sum() > 0 or labels.sum() == 0
```

(`sample_config` comes from `tests/conftest.py`; sampling_rate=5, num_frames=6, history=2 → current frame 1, four future frames.)

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_bench2drive_future_agents.py -v`
Expected: first test FAILS with `AttributeError: ... has no attribute 'get_future_agents'`; second PASSES (it pins existing behavior).

- [ ] **Step 4: Refactor + implement.** In `bench2drive_scene.py`:

(a) Move the body of `get_agents` from the `max_agents = MAX_AGENTS` line (~772) through the processing loop end (~870) into a new method placed just above `get_agents`:

```python
    def _extract_agents_from_anno(
        self,
        anno: dict,
        ego_points: np.ndarray,
        ego_heading_rad: float,
        max_distance: float,
        use_anno_distance: bool,
    ):
        """Extract agent states from one annotation, in the frame of the
        given ego pose (which need not be the annotation's own frame).

        Args:
            max_distance: keep agents within this distance of the ego pose.
            use_anno_distance: if True, filter on the annotation's precomputed
                obj["distance"] (distance to the annotation frame's OWN ego) —
                the historical get_agents behavior. If False, filter on the
                ego-centric planar distance to the GIVEN ego pose (correct for
                future-frame agents relative to the current ego).
        """
        agent_states = np.zeros((MAX_AGENTS, 5), dtype=np.float32)
        agent_labels = np.zeros(MAX_AGENTS, dtype=bool)
        agent_types = np.zeros(MAX_AGENTS, dtype=np.int32)
        ...  # (verbatim body moved from get_agents, with ONE change below)
```

The one change inside the moved loop — replace the distance filter (old lines 826-830):

```python
            if use_anno_distance:
                distance = obj["distance"]  # to the annotation frame's own ego
            else:
                distance = float(np.hypot(ego_centric_x, ego_centric_y))
            if distance > max_distance:
                continue
```

and end with `return agent_states, agent_labels, agent_types`.

(b) `get_agents` becomes (keeping its docstring and the ego lookup at 737-768):

```python
        agent_states, agent_labels, agent_types = self._extract_agents_from_anno(
            anno,
            ego_points,
            ego_heading_rad,
            max_distance=BENCH2DRIVE_LIDAR_RANGE_M / 2,
            use_anno_distance=True,
        )
        return (
            torch.from_numpy(agent_states),
            torch.from_numpy(agent_labels),
            torch.from_numpy(agent_types),
        )
```

(c) Add `get_future_agents` after `get_agents` (import `COLLISION_EVAL_RANGE_M` at the top of the file with the other constants):

```python
    def get_future_agents(
        self, frame_idx: int = -1, num_timesteps: int = NUM_FUTURE_WAYPOINTS
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Agent states at each future waypoint time, in the CURRENT ego frame.

        Timestep t corresponds to the same future annotation frame as
        get_future_trajectory waypoint t (same stride logic), so occupancy
        grids align 1:1 with trajectory waypoints.

        Returns:
            agent_states: [T, MAX_AGENTS, 5] float32 (x, y, heading, length, width)
            agent_labels: [T, MAX_AGENTS] bool
        """
        if frame_idx == -1:
            frame_idx = max(0, self.history_frames - 1)

        current_anno = self._load_annotation(frame_idx)
        ego_box = None
        for box in current_anno["bounding_boxes"]:
            if box["class"] == "ego_vehicle":
                ego_box = box
                break
        if ego_box is None:
            raise ValueError(
                f"Ego vehicle not found in bounding boxes for frame {frame_idx}"
            )
        ego_points = np.array(ego_box["center"])
        ego_heading_rad = normalize_angle(np.radians(ego_box["rotation"][2]))

        sampling_rate = getattr(self.config, "sampling_rate", 5)
        if sampling_rate == 1:
            frame_stride = FUTURE_TRAJECTORY_FRAME_STRIDE
        elif sampling_rate == 5:
            frame_stride = 1
        else:
            raise ValueError(f"Unsupported sampling_rate={sampling_rate}")

        states = np.zeros((num_timesteps, MAX_AGENTS, 5), dtype=np.float32)
        labels = np.zeros((num_timesteps, MAX_AGENTS), dtype=bool)
        for t in range(num_timesteps):
            future_idx = frame_idx + (t + 1) * frame_stride
            if future_idx >= len(self.anno_paths):
                # Callers must pair this with a valid get_future_trajectory
                # check; remaining timesteps stay empty rather than fabricated.
                break
            anno = self._load_annotation(future_idx)
            s, l, _ = self._extract_agents_from_anno(
                anno,
                ego_points,
                ego_heading_rad,
                max_distance=COLLISION_EVAL_RANGE_M,
                use_anno_distance=False,
            )
            states[t], labels[t] = s, l

        return torch.from_numpy(states), torch.from_numpy(labels)
```

- [ ] **Step 5: Run new + regression tests**

Run: `pytest tests/test_bench2drive_future_agents.py tests/test_bench2drive_minimal.py tests/test_bench2drive_heading_fix.py -v`
Expected: all PASS (regression files prove `get_agents` behavior is unchanged).

- [ ] **Step 6: Commit**

```bash
git add navsim/common/bench2drive_scene.py navsim/common/bench2drive_constants.py tests/test_bench2drive_future_agents.py
git commit -m "feat: add get_future_agents for per-timestep collision occupancy"
```

---

### Task 6: Eval-script wiring + strict checkpoint loading

**Files:**
- Modify: `scripts/evaluation/run_b2d_openloop_eval.py`
- Test: manual command checks (script-level; unit coverage lives in Tasks 1–5)

**Interfaces:**
- Consumes: `scene.get_future_agents(-1, num_timesteps=8)` (Task 5); `B2DOpenLoopMetrics` per-timestep agent args (Task 4).
- Produces: results JSON `meta` gains `checkpoint_load` (`{"strict": bool, "missing_keys": int, "unexpected_keys": int}`), `token_stride`, and `metric_definitions`. New CLI flags `--allow-partial-load` and `--token-stride` (default 1).

- [ ] **Step 1: Dataset returns per-timestep agents.** In `B2DEvalDataset.__getitem__` (line 95), replace:

```python
        agent_states, agent_labels, _ = scene.get_agents(-1)
```

with:

```python
        agent_states, agent_labels = scene.get_future_agents(-1, num_timesteps=8)
```

(the returned arrays are now `[8, N, 5]` / `[8, N]`; the existing `np.array(..., dtype=np.float32)` conversions and `eval_collate_fn` stacking work unchanged). Add token subsampling to `__init__`: give the constructor a `token_stride: int = 1` parameter and apply `tokens = tokens[::token_stride]` right after `tokens = loader.get_scene_tokens()`.

- [ ] **Step 2: Strict checkpoint loading.** Replace the `try/except` block in `load_model` (lines 165-176) with:

```python
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
```

Change the signature to `def load_model(checkpoint_path, config, device="cuda", allow_partial_load=False):` and the return to `return model, load_info`. Update the call site in `main()`:

```python
        model, load_info = load_model(args.checkpoint, config, args.device, args.allow_partial_load)
```

(and `load_info = {"strict": True, "missing_keys": 0, "unexpected_keys": 0}` in the `--use-gt` branch).

- [ ] **Step 3: CLI + meta.** In `parse_args()` add:

```python
    parser.add_argument("--allow-partial-load", action="store_true",
                        help="Evaluate even if checkpoint keys are missing/unexpected (results are suspect)")
    parser.add_argument("--token-stride", type=int, default=1,
                        help="Evaluate every Nth 10Hz token (5 ≈ 2Hz VAD keyframe density)")
```

Pass `token_stride=args.token_stride` to `B2DEvalDataset(...)`. In `output_data["meta"]` add:

```python
            "checkpoint_load": load_info,
            "token_stride": args.token_stride,
            "metric_definitions": {
                "collision": "per-horizon mean of per-step flags over [0,t], GT-collision-masked (VAD/STP3); col_any_* = cumulative any()",
                "L2_avg_vad": "mean of period-average L2 at 1s/2s/3s",
                "agent_occupancy": "per-future-timestep agent states (get_future_agents)",
            },
```

- [ ] **Step 4: Syntax + help check** (no GPU needed):

Run: `python -c "import ast; ast.parse(open('scripts/evaluation/run_b2d_openloop_eval.py').read())" && python scripts/evaluation/run_b2d_openloop_eval.py --help`
Expected: help text shows `--allow-partial-load` and `--token-stride`. (If host python lacks torch, run the `--help` check inside the Docker container.)

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluation/run_b2d_openloop_eval.py
git commit -m "feat: wire per-timestep agents through eval; fail hard on partial checkpoint load"
```

---

### Task 7: Verification gate — GT-as-prediction run + honest baseline

**Files:**
- Create: `docs/b2d/openloop_results.md`
- No source changes.

**Interfaces:**
- Consumes: everything above.
- Produces: the B1 merge-gate evidence.

- [ ] **Step 1: Resource check (mandatory before any run):**

```bash
free -g && df -h /mnt/nvme1 && nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Expected: ≥ 8 GB free RAM headroom, ≥ 5 GB disk free, GPU not saturated by another job. **If training/caching is running, STOP and wait — do not compete for RAM.**

- [ ] **Step 2: Full unit-test sweep:**

Run: `pytest tests/test_b2d_planning_utils.py tests/test_b2d_metrics.py tests/test_eval_collision_wiring.py tests/test_bench2drive_future_agents.py -v`
Expected: all PASS.

- [ ] **Step 3: GT-as-prediction gate (dev mode, in Docker, ≤20 GB/≤8 CPU caps):**

```bash
python scripts/evaluation/run_b2d_openloop_eval.py --use-gt --dev-mode --num-workers 4
```

Expected: all `L2_*` ≈ 0.0000 AND all `col_*` / `box_col_*` ≈ 0.0000 (GT masking guarantees GT-as-prediction reports no collisions). If any collision value is nonzero, the masking or occupancy wiring is wrong — stop and debug before proceeding.

- [ ] **Step 4: Honest epoch-199 baseline (needs user go-ahead — GPU time):**
Ask the user before running the full-split eval. When approved:

```bash
python scripts/evaluation/run_b2d_openloop_eval.py \
    --checkpoint <epoch=199 checkpoint path> --num-workers 8 --token-stride 5
```

(`--token-stride 5` for VAD-comparable 2 Hz density; also run once with stride 1 if a dense variant is wanted.) Then write `docs/b2d/openloop_results.md`: a table of the new L2/collision numbers, the metric definitions, and one paragraph stating that all previously committed results (in `eval_results/` and prior docs) were produced with the frozen-agent/any()-based metric and are superseded.

- [ ] **Step 5: Commit**

```bash
git add docs/b2d/openloop_results.md
git commit -m "docs: honest open-loop baseline with corrected VAD-compatible metrics"
```

---

## Self-review notes (done at authoring time)

- **Spec coverage:** frozen agents (T2+T5+T6), GT masking (T3), any→mean (T4), L2_avg_vad (T4), strict load (T6), quantization (T3), positive-path tests (T2/T3/T4 tests), reference parity (semantics mirrored from `metric_stp3.py` with file:line cites; runtime import of the reference's mmcv stack is deliberately NOT attempted — it doesn't import in this env), perf items (T1 fillPoly, T4 hoisted PlanningMetric), 10 Hz density (T6 `--token-stride` + meta). Ego dims / grid convention: explicitly kept (Global Constraints).
- **Known trap:** `np.round` uses banker's rounding; the test pixel values (110, 120, 100) in the cheat-sheet were computed with that in mind — don't "simplify" to `int(x + 0.5)`.
- **Out of scope for B1:** `get_agents` filter changes (B4), pedestrian-tensor API collapse, caching paths, anything touching training.
