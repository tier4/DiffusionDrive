#!/usr/bin/env bash
# Render MP4s from a closed-loop run directory using Bench2Drive's video tools.
# Usage: make_videos.sh <run_dir (contains frames/<route>_*/rgb_front + meta)>
#
# The Bench2Drive video scripts call cv2.destroyAllWindows() at the end, which
# raises on headless OpenCV builds (no GTK/Qt) *after* the MP4 is already
# written -- crashing the run and skipping the BEV video. We invoke them through
# a tiny shim that no-ops the GUI calls (keeping Bench2Drive files untouched).
set -euo pipefail
RUN_DIR=${1:?run dir required}
B2D_REPO=${B2D_REPO:-$HOME/workspace/Bench2Drive}

# Headless shim: monkeypatch cv2 GUI fns to no-ops, then run the target script.
SHIM='
import sys, runpy, cv2
for _fn in ("destroyAllWindows", "destroyWindow", "imshow", "waitKey", "namedWindow"):
    if hasattr(cv2, _fn):
        setattr(cv2, _fn, lambda *a, **k: None)
_script, _folder, _out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [_script, "-f", _folder, "-o", _out]
runpy.run_path(_script, run_name="__main__")
'

for route_dir in "${RUN_DIR}"/frames/*/; do
  name=$(basename "${route_dir}")
  python3 -c "${SHIM}" "${B2D_REPO}/generate_video_ddrive.py" \
    "${route_dir}" "${RUN_DIR}/${name}.mp4"
  python3 -c "${SHIM}" "${B2D_REPO}/generate_bev_video_ddrive.py" \
    "${route_dir}" "${RUN_DIR}/${name}_bev.mp4"
done
echo "Videos in ${RUN_DIR}/"
