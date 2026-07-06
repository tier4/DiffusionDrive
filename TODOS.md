# TODOs

## History Frame Off-by-One Bug

**What:** `num_history_frames == 0` assumption masks an off-by-one between feature caching and target building. With nonzero history frames, features are cached from `num_history_frames - 1` but targets are built from `scene.history_frames`, which disagree.

**Why:** Any future attempt to use history frames (e.g., temporal modeling) will produce misaligned features and targets, causing subtle training degradation.

**Where:**
- `scripts/cache_bench2drive_dataset.py:69`
- `navsim/agents/diffusiondrive/transfuser_features_b2d.py:347`
- `navsim/planning/script/config/common/train_test_split/bench2drive.yaml:17`

**Found by:** Codex outside voice during eng review (2026-04-03).

**Priority:** Low (only manifests with nonzero history config, which is not currently used).
