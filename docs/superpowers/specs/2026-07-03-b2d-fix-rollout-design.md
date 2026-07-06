# B2D Fix Rollout — Branch Sequencing Design

**Date:** 2026-07-03
**Status:** Approved
**Source:** Multi-pass code review of `feature/b2d-openloop-eval` vs `tier4-main`
(6 Claude specialist passes + Codex cross-model adversarial pass, ~45 merged findings).

## Goal

Land the review findings as a sequence of small, ordered branches against `tier4-main`
such that:

1. Eval numbers become trustworthy as fast as possible (they are currently invalid).
2. The expensive operations — cache regeneration and retraining — happen **exactly once**,
   after every change that would invalidate them has merged.
3. Each branch is independently reviewable, testable, and revertable.

## Decisions (confirmed with user)

- **PR #8 merges as-is.** All fix branches are cut from `tier4-main` after the merge.
- **Single-retrain constraint.** All cache/checkpoint-invalidating fixes batch before one
  v7 cache regeneration and one retrain.
- **Medium refactor appetite.** Config/normalization single-sourcing is in scope (it is a
  correctness prerequisite for the retrain). The large dataset-adapter refactor is deferred
  to a final branch gated on TaCarla timing.

## Branch sequence

Dependency graph: `B1 ∥ B6` (independent, anytime); `B2 → B3 → B4 → B5 → OP`; `B7` after OP.

### B1 · `fix/b2d-eval-metrics` — first; independent of all others

The open-loop metrics are invalid; fixing them requires no retraining.

| Change | Where |
|---|---|
| Rasterize agents per future timestep instead of copying t=0 occupancy to all horizons; wire per-timestep agent states from the scene through the eval script | `navsim/evaluate/b2d_planning_utils.py:157-176`, `scripts/evaluation/run_b2d_openloop_eval.py:95,216`, `navsim/common/bench2drive_scene.py` (`get_agents`) |
| Add VAD/STP3 GT-collision masking (exclude timesteps where GT itself collides) | `b2d_planning_utils.py:187` (`evaluate_coll`, currently ignores `gt_trajs`) |
| Per-horizon collision: cumulative `any()` → `mean()` over steps [0, t] (VAD definition); keep `any` variant under `col_any_{t}s` if wanted | `navsim/evaluate/b2d_metrics.py:141-151` |
| `L2_avg_vad` = mean of 1s/2s/3s values only; keep 6-horizon mean under a distinct key | `b2d_metrics.py:103` |
| Checkpoint load: fail hard on missing keys unless `--allow-partial-load`; record missing/unexpected counts in results JSON | `run_b2d_openloop_eval.py:165-172` |
| Unify pixel quantization (`int()` truncation vs `np.round`) in point vs box checks | `b2d_planning_utils.py:121,208` |
| Positive-path tests: agent straddling a known waypoint (assert collision at exact timesteps), rotated-agent case, lateral box-only case; synthetic parity test against `reference/Bench2DriveZoo` `metric_stp3.py` | `tests/test_b2d_metrics.py`, `tests/test_eval_collision_wiring.py` |
| Optional (same branch, cheap): hoist `PlanningMetric` instance; `polygon_simple` → `cv2.fillPoly`; document/subsample the 10 Hz token density (`--eval-hz` flag or results-meta field) | `b2d_metrics.py:124`, `b2d_planning_utils.py:41-57`, `run_b2d_openloop_eval.py:58` |

**Non-goals:** the nuScenes ego dims (1.85×4.084) and the 200×200@0.5 m grid stay — they
are deliberate VAD parity; add a comment noting the real MKZ extents.

**Merge gate:** GT-as-prediction eval reports ≈0 collision on `val_dev`; parity test passes;
re-run epoch-199 eval and record the honest baseline in `docs/b2d/` (a results summary doc,
since B6 gitignores raw `eval_results/`), superseding the previously committed numbers.

### B2 · `fix/shared-code-reverts` — restores NAVSIM reproducibility

| Change | Where |
|---|---|
| Remove unconditional `clip_grad_norm_(max_norm=1.0)` hook; set `trainer.params.gradient_clip_val: 1.0` in the B2D training config only | `navsim/planning/training/agent_lightning_module.py:46-63`, `config/training/default_training_w_callbacks.yaml:49` |
| Replace per-parameter `.item()` grad-norm loop with the value `clip_grad_norm_` returns (single kernel) | `agent_lightning_module.py:53-61` |
| Hungarian cost `nan_to_num` guard behind a config flag; `print` → `logging` with occurrence counts | `navsim/agents/diffusiondrive/transfuser_loss.py:93-103` |
| Fix ABC drift: `AbstractAgent.forward(features, targets=None)` | `navsim/agents/abstract_agent.py:42` |

**Merge gate:** NAVSIM unit tests pass; a short NAVSIM smoke training step behaves as upstream.

### B3 · `refactor/normalization-single-source` — stabilize model code before retrain

| Change | Where |
|---|---|
| Move `norm_odo`/`denorm_odo` into `TrajectoryHead`, parameterized by config (NAVSIM constants `1.2/56.9/2/3.9` become defaults) | `transfuser_model_v2.py:433-449` |
| Delete `V2TransfuserModelWrapper` (subclass + `types.MethodType` binding); `Bench2DriveAgent` passes config only | `navsim/agents/diffusiondrive/transfuser_model_wrapper.py`, `b2d_agent.py` |
| Norm buffers `persistent=False` (state_dict schema identical to upstream) | wrapper → `TrajectoryHead` |
| `init_from_pretrained`: strip `plan_anchor` and `_norm_*` keys from incoming state dicts; hard-warn when checkpoint anchors differ from config anchors | `transfuser_agent.py:70`, `b2d_agent.py:35` (also fix the double model build/checkpoint load) |
| Delete dead, contradictory `normalization_profiles.yaml` | `navsim/planning/script/config/normalization_profiles.yaml` |
| Machine paths → `${oc.env:...}` interpolation; dataclass path defaults → `None` + fail-fast in `initialize()` | `diffusiondrive_agent_b2d.yaml:22-23`, `bench2drive_config.py:26`, `transfuser_config.py:18-19` |
| LR scheduler: plumb `max_epochs` from config and clamp cosine progress at 1.0 | `transfuser_agent.py:174`, `modules/scheduler.py:48` |
| Single canonical split JSON (`config/common/train_test_split/`); delete `data/splits/` copy; point all consumers at one path; warn when val scenarios are missing from cache | `run_bench2drive_training.py:99-106`, `run_b2d_openloop_eval.py:244`, `scripts/visualize_bench2drive_predictions.py` |

**Merge gate:** epoch-199 checkpoint still loads and evaluates identically (B1 metrics);
NAVSIM checkpoint loads unchanged; unit test for norm round-trip per profile.

### B4 · `fix/b2d-data-correctness` — the cache-invalidating batch

Everything here changes cached features/targets. Nothing regenerates until B5 merges.

| Change | Where |
|---|---|
| Ego-frame dynamics: velocity `[speed, 0]` (or world vector rotated by −yaw, left-handed), acceleration rotated into ego frame; remove the `-sin` flip | `bench2drive_scene.py:545-564` (`_extract_ego_status`) |
| Fix 1-px map/agent misalignment: single ego-row convention through both paths + regression test (x=0 maps to same row via map and agent paths) | `bev_semantic_utils.py:99` vs `bev_map_utils.py:273` (crop at `bench2drive_scene.py` full→front) |
| Agent filter: 32 m circle → BEV square (−32≤x,y≤32) via a dedicated `AGENT_FILTER` constant; fix stale "42.5m/85m" comments (3 sites) | `bench2drive_scene.py:827-832`, `transfuser_features_b2d.py:333` |
| `generate_agent_mask`: skip only when the agent's bounding circle cannot intersect the raster (not when center is off-raster) | `bev_semantic_utils.py:196` |
| Lane polyline: remove per-vertex out-of-bounds mask (cv2 clips correctly) | `bev_segmentation_generator.py:161` |
| Unknown lane types → background 0 explicitly + log-once (not silent Road) | `bev_map_utils.py:357`, `bench2drive_constants.py` (`LANE_TYPE_TO_BEV_CLASS`) |
| BEV cache load: require + validate `resolution`/`range_m`/shape; raise on mismatch or missing metadata | `bench2drive_scene.py:914-937` |
| Delete every `0.332` default (base class, factory, getattr fallback) → `BEV_SEMANTIC_RESOLUTION`; fix stale comments | `bev_generator_base.py:24`, `bev_generation_factory.py:37,180`, `bench2drive_scene.py:732,1030` |
| Camera stitch: crop margins giving exact 4.0 aspect before 1024×256 resize; drop the double JPEG q20 re-encode | `transfuser_features_b2d.py:184`, `bench2drive_scene.py:315-319` |
| Missing f-prefix in invalid-class error | `bev_map_utils.py:663` |

**Merge gate:** unit tests (alignment regression, filter bounds, lane rasterization);
A/B render of a sample frame showing agents aligned with map; no cache regen yet.

### B5 · `fix/b2d-caching-robustness` — last code branch before regen (may stack on B4)

| Change | Where |
|---|---|
| One caching path: keep the Ray `scripts/cache_bench2drive_dataset.py` pipeline (has None-target filtering); delete `Bench2DriveDataset`'s private 3-level cache format; make `run_bench2drive_caching.py` a thin wrapper (or delete it) | `bench2drive_dataset.py:104-143`, `run_bench2drive_caching.py:41-62` |
| `compute_targets` returning `None` → typed `SkipSample` handling documented in the builder contract | `transfuser_features_b2d.py:329-331` |
| Atomic cache writes: tmp + `os.replace`; treat zero-byte files as invalid | `navsim/planning/training/dataset.py:27-28` |
| `cache_metadata.json` at cache root (schema version, git SHA, config hash, coordinate convention, resolution/range); loader refuses mismatches | `scripts/cache_bench2drive_dataset.py`, `scripts/generate_bev_cache.py:288`, `dataset.py` loader |
| Tar handling: `extractall(filter='data')`, temp-dir + atomic rename, `extract_tar=False` default | `bench2drive_dataloader.py:160-165`, `:22-26` |
| Scenario glob → directories only, distinct error messages | `bench2drive_dataloader.py:80-84` |
| `get_bev_semantic_map(-1)` → shared `_resolve_frame_idx()` used by all four accessors | `bench2drive_scene.py:878-914` |
| Ray BEV cache: batch frames per town (deserialize MapProcessor once per batch); conservative default `--workers` (respect the 20 GB / 8 CPU budget) | `scripts/generate_bev_cache.py:103,168` |
| Align `Bench2DriveDataConfig` defaults with `bench2drive.yaml` (9/0/8, `extract_tar=False`) | `bench2drive_dataloader.py:22-26` |

**Merge gate:** caching smoke test on the committed sample data produces a valid,
metadata-stamped cache; kill-and-resume test leaves no truncated files.

### OP · Regenerate + retrain (operational; not a branch)

Preconditions: B2–B5 merged. Steps, in Docker per user workflow, within resource caps:

1. Regenerate BEV cache (v7) → 2. regenerate dataset cache (v7) → 3. retrain
   (correct LR schedule, gradient clipping via trainer config) → 4. evaluate with B1
   metrics → publish the new baseline. Verify `num_history_frames` resolves to 0
   everywhere (the documented off-by-one stays dormant).

### B6 · `chore/repo-hygiene` — independent; parallel with anything

Untrack-or-unignore `debug/` (resolve tracked-but-ignored, incl. `CLAUDE.md` in
`.gitignore`); remove `debug/reference_b2d/` third-party copies (link upstream + commit
hash in docs); strip notebook outputs, delete `notebooks/tmp.json`; gitignore
`eval_results/`; trim `tests/test_data` to frames tests consume; pytest
`testpaths`/markers in `pyproject.toml`; move `tests/scripts/test_generate_bev_cache.py`
into `tests/`, rename `scripts/test_segmentation_15samples.py`; delete the dead shell-test
framework (target `scripts/evaluation/eval.sh` never existed) or commit the script;
fork-correct `setup.py` metadata; docker `carla_net` join made conditional; restore
`docker/build.sh` exec bit; `common.sh` `LOG_FILE` default; move NaN-hunt scripts out of
`navsim/planning/utils/{data_validation,debugging}/`.

### B7 · `refactor/dataset-adapter` — deferred; gate before TaCarla

`AbstractScene`/`AbstractSceneLoader` protocol (what builders actually consume);
`CoordinateFrame` object applied at exactly one boundary (scene construction);
collapse `run_bench2drive_training.py` into `run_training.py` via a Hydra `dataset=`
group; one naming convention (`bench2drive_` subpackage) in a mechanical rename commit.
Do not start until OP completes and TaCarla work is scheduled.

## Addendum (2026-07-07) — pre-merge review findings

Source: 6-pass pre-merge review of `feature/bench2drive-dataset-support` vs `tier4-main`
(4 Claude specialists + Claude adversarial + Codex cross-model), run before the merge to
`tier4-main`. ~23 findings not covered by the branch tables above. Routing:

### B1.5 · `fix/b2d-eval-metrics-followup` — cut anytime; REQUIRED before publishing collision numbers

| Change | Where |
|---|---|
| The B1 merge gate was vacuous: with STP3 masking, GT-as-prediction yields collision ≡ 0 by construction (`pred_box ∧ ¬gt_box ≡ 0`), so it cannot detect broken occupancy wiring. Add an injected-known-collision integration test (stub model + 1-sample dataset, assert `col_0.5s > 0`) and/or a masking-disabled GT run | `b2d_planning_utils.py:230-246`, `run_b2d_openloop_eval.py` |
| Agent filter is a 50 m **circle** but the metric grid is a ±50 m **square** (corners at 70.7 m): in-raster agents beyond 50 m radial are silently dropped from occupancy (false-negative collisions; undocumented parity deviation) | `bench2drive_scene.py:829`, `bench2drive_constants.py:25` |
| Nearest-first overflow truncation (>MAX_AGENTS) sorts by distance to the **current** pose, so the agent at a far future waypoint can be the one dropped | `bench2drive_scene.py:782-806` |
| `_compute_collision` pre-expands any ndim-3 array to 4-D, bypassing `get_label`'s ambiguous-shape guard (b4de8d1) on the primary path | `b2d_metrics.py:135-137` |
| Deliver the B1-spec'd but never-written tests: rotated-agent occupancy (no non-zero heading exists in the metric test suite — a transposed rotation matrix would pass) and lateral box-only collision (`box_col=1, col=0`) | `tests/test_b2d_planning_utils.py`, `tests/test_b2d_metrics.py` |
| `col_avg_full` (mean of flags ≡ `col_4.0s`) is a different statistic than `L2_avg_full` (mean of cumulative horizons) under the same suffix — rename or redefine | `b2d_metrics.py:163` |
| Fail fast instead of silent degradation: `min(T)` horizon truncation, NaN/Inf trajectories (also `json.dump(..., allow_nan=False)`), `get_future_agents` zero-fill past scene end, `_box_collision_flags` heading=0 fallback for [T,2] input | `b2d_metrics.py:69`, `run_b2d_openloop_eval.py:383`, `bench2drive_scene.py:984-987`, `b2d_planning_utils.py:182` |
| Zero-assertion test files pass unconditionally; wiring test asserts key presence only | `tests/test_heading_fix_simple.py`, `tests/test_heading_fix_integration.py`, `tests/test_eval_collision_wiring.py` |

For the record: Codex's "wrong handedness mirrors collision boxes" claim was verified a
**false positive** — trajectory and agent paths share one self-consistent left-handed
convention through the same corner/rasterization code.

### OP preconditions (add to the regen/retrain checklist)

- **MUST FIX before regen:** `ray.init(num_cpus=num_workers)` runs while `num_workers` is
  still `None` (cap applied only later) → Ray claims ALL host CPUs + ~30% RAM object store.
  Resolve the capped default before `ray.init` and pass bounded `object_store_memory`
  (`scripts/cache_bench2drive_dataset.py:153` vs `:217`).
- Measure full-split eval peak RAM vs the 20 GB budget (defaults `num_workers=8`,
  `batch_size=32`, `persistent_workers=True`); eval indexing re-decodes each anno ~18×
  serially; per-scenario configs inherit `extract_tar=True`, so "read-only" eval can
  extract stray tars as a write side effect.

### B6 additions (hygiene/security)

Remove `${HOME}/.ssh` mount from `docker/docker-compose.yml:41` (container deserializes
third-party artifacts); remove silent `apt-get install` fallbacks from both visualization
scripts (fail fast instead; `visualize_model_predictions.py:684`,
`visualize_bench2drive_predictions.py:1162`); `torch.load(..., weights_only=True)` at the
3 checkpoint-load sites; document the `allow_pickle=True` trust assumption for downloaded
HD-map npz; make `pytest tests/` green on a fresh clone (stale `KeyError: 'center'` mock
schemas, `tests/test_data` fixture lacks camera images its consumers require,
dataset-dependent tests need skip markers); fix the CHANGELOG claim that left→right-handed
conversion was implemented (it raises `NotImplementedError`); machine-path default in
`run_b2d_openloop_eval.py:255` (add to the B3 path list); delete dead `sliding_mode` flag
and the nine commented-out sliding-window fossil blocks; per-frame `print()` → logging in
`get_bev_semantic_map`; unbounded frame accumulation in both video-rendering scripts.

## Resource safety — MANDATORY for every implementing agent

Any agent executing a branch of this plan MUST treat RAM and disk as scarce and
protected resources. This machine hosts other critical workloads; an OOM has
previously destroyed a very expensive training run.

- **RAM:** stay within the standing budget (≤20 GB / ≤8 CPUs for Docker jobs). Before
  launching anything memory-heavy (Ray caching, training, full-split eval), check free
  memory (`free -g`) and cap worker counts explicitly — never default to
  `os.cpu_count()`. Prefer streaming/batched processing over loading whole towns or
  splits into memory. If a step's peak memory is unknown, measure it on a small sample
  first.
- **Storage:** before writing any cache, checkpoint, or eval output, check free space
  (`df -h` on the target volume) and estimate the write size. The v7 cache regeneration
  writes hundreds of GB — confirm the target volume (`/mnt/nvme1/...`) has headroom and
  never delete an old cache version until the new one is validated. Clean up temp files
  (`.tmp` cache writes, extracted tars) on failure paths.
- **Docker only** for heavy jobs, with explicit `--memory`/`--cpus` limits per the user's
  workflow. No pip installs into the host environment, ever.
- Every future design/spec/plan document in this repo MUST carry this section (or an
  equivalent) so downstream agents inherit the constraint.

## Error handling & testing philosophy

- Every silent fallback touched by these branches becomes fail-fast (cache metadata,
  checkpoint loads, unknown lane types) — the review's dominant bug class was silent
  wrong-data paths.
- Each branch carries its own regression tests; B1 additionally carries the
  reference-parity test so metric definitions can never silently drift again.

## Out of scope

Known documented history-frame off-by-one (TODOS.md) beyond the OP verification step;
closed-loop CARLA evaluation; TaCarla integration itself.
