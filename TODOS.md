# TODOs

Each item below lands as its own branch + PR. Full `file:line` detail for the first
three lives in `docs/superpowers/specs/2026-07-03-b2d-fix-rollout-design.md`
(§ Addendum 2026-07-07).

## Re-validate Collision Metrics — GT-as-Prediction Gate Is Vacuous

**What:** The B1 merge gate ("GT-as-prediction eval reports ≈0 collision") cannot fail:
with STP3 masking in `evaluate_coll`, pred=GT gives point collision ≡ 0 and box collision
≡ `gt ∧ ¬gt` ≡ 0 regardless of occupancy wiring, frame alignment, or transforms. The
published collision baselines in `docs/b2d/openloop_eval_results.md` were never validated
by a test that can fail. Batch with the other metric-validity fixes: circle-vs-square
agent filter gap, nearest-first truncation referent, `_compute_collision` guard bypass,
missing rotated-agent/lateral-box tests, `col_avg_full` naming, fail-fast items.

**Why:** Do NOT cite the current collision numbers externally until an
injected-known-collision test passes.

**Where:** `navsim/evaluate/b2d_planning_utils.py:230-246`, `navsim/evaluate/b2d_metrics.py`,
`navsim/common/bench2drive_scene.py:782-832`, `tests/test_b2d_*.py`

**Fix in:** branch `fix/b2d-eval-metrics-followup` (B1.5), separate PR.

**Found by:** 2026-07-07 pre-merge review (PR #9), Claude adversarial pass.

**Priority:** High (blocks publishing any collision numbers).

## Ray Resource-Cap Ordering Bug in Dataset Caching

**What:** `ray.init(num_cpus=num_workers)` runs while `num_workers` is still `None`; the
`min(os.cpu_count(), 8)` cap is applied only afterwards. Ray therefore autodetects ALL
host CPUs and reserves its default object store (~30% of system RAM).

**Why:** This machine has a hard 20 GB / 8 CPU budget and an OOM here previously destroyed
an expensive training run. MUST be fixed before the v7 cache regeneration (OP step).

**Where:** `scripts/cache_bench2drive_dataset.py:153` (init) vs `:217` (late cap). Fix:
resolve the capped default before `ray.init` and pass a bounded `object_store_memory`.

**Fix in:** branch `fix/cache-ray-resource-caps`, separate PR.

**Found by:** 2026-07-07 pre-merge review (PR #9), performance specialist.

**Priority:** High (hard precondition for the v7 cache regen).

## Security Hygiene: .ssh Mount, apt-get Auto-Install, Unsafe torch.load

**What:** (1) `docker/docker-compose.yml:41` mounts `${HOME}/.ssh` into a container that
deserializes third-party artifacts (checkpoints via `torch.load`, HD-map npz via
`allow_pickle=True`, tar extraction) — key exfiltration risk if any artifact is malicious.
(2) Both visualization scripts silently run `apt-get update && apt-get install -y ffmpeg`
when ffmpeg is missing (`scripts/visualize_model_predictions.py:684`,
`scripts/visualize_bench2drive_predictions.py:1162`) — mutates the host if run outside
Docker. (3) Three checkpoint-load sites call `torch.load` without `weights_only=True`.

**Why:** The host is a protected machine; silent package installs and credential exposure
inside artifact-deserializing containers are unacceptable side effects.

**Where:** see above; full list in the spec addendum (B6 additions).

**Fix in:** branch `chore/b2d-security-hygiene`, separate PR.

**Found by:** 2026-07-07 pre-merge review (PR #9), security specialist.

**Priority:** Medium (fix before sharing the compose file or running viz outside Docker).

## History Frame Off-by-One Bug

**What:** `num_history_frames == 0` assumption masks an off-by-one between feature caching and target building. With nonzero history frames, features are cached from `num_history_frames - 1` but targets are built from `scene.history_frames`, which disagree.

**Why:** Any future attempt to use history frames (e.g., temporal modeling) will produce misaligned features and targets, causing subtle training degradation.

**Where:**
- `scripts/cache_bench2drive_dataset.py:69`
- `navsim/agents/diffusiondrive/transfuser_features_b2d.py:347`
- `navsim/planning/script/config/common/train_test_split/bench2drive.yaml:17`

**Found by:** Codex outside voice during eng review (2026-04-03).

**Priority:** Low (only manifests with nonzero history config, which is not currently used).
