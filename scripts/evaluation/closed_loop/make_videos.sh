#!/usr/bin/env bash
# Render MP4s from a closed-loop run directory using Bench2Drive's video tools.
# Usage: make_videos.sh <run_dir (contains frames/<route>_*/rgb_front + meta)>
set -euo pipefail
RUN_DIR=${1:?run dir required}
B2D_REPO=${B2D_REPO:-$HOME/workspace/Bench2Drive}
for route_dir in "${RUN_DIR}"/frames/*/; do
  name=$(basename "${route_dir}")
  python3 "${B2D_REPO}/generate_video_ddrive.py" -f "${route_dir}" -o "${RUN_DIR}/${name}.mp4"
  python3 "${B2D_REPO}/generate_bev_video_ddrive.py" -f "${route_dir}" -o "${RUN_DIR}/${name}_bev.mp4"
done
echo "Videos in ${RUN_DIR}/"
