#!/usr/bin/env bash
# Closed-loop Bench2Drive evaluation orchestrator.
# Usage: run_closed_loop_eval.sh <routes.xml (IN-CONTAINER path, e.g. /workspace/Bench2Drive/...)> <checkpoint.ckpt> [results_dir]
#   NOTE: <routes.xml> must be the path as seen INSIDE the container (i.e. under
#   /workspace/Bench2Drive/...), because both the evaluator and the agent read it
#   from inside the container, not from the host filesystem.
# Env: DD_REPO (default ~/workspace/DiffusionDrive), B2D_REPO (default ~/workspace/Bench2Drive)
#      RUN_DIR (default <results_dir>/run_<timestamp>) — override to resume/reuse a specific run dir
set -euo pipefail
trap 'docker rm -f ddrive-carla 2>/dev/null || true' EXIT

ROUTES_XML=${1:?routes xml required}
CKPT=${2:?checkpoint path required}
RESULTS_ROOT=${3:-/mnt/nvme1/diffusiondrive/closed_loop_results}
DD_REPO=${DD_REPO:-$HOME/workspace/DiffusionDrive}
B2D_REPO=${B2D_REPO:-$HOME/workspace/Bench2Drive}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR=${RUN_DIR:-${RESULTS_ROOT}/run_${STAMP}}
NET=ddrive-eval
MIN_FREE_GB=100

# --- results must live under /mnt/nvme1 (the only data mount) ---
# /mnt/nvme1 is the sole host path mounted into the container, and the storage
# guard below only checks that filesystem. A RUN_DIR anywhere else would silently
# break persistence (results never land in the container) and check the wrong disk.
if [[ "${RUN_DIR}" != /mnt/nvme1/* ]]; then
  echo "FATAL: RUN_DIR must live under /mnt/nvme1 (the only data mount into the container)."
  echo "       Got: ${RUN_DIR}"
  exit 1
fi

# --- storage guard (never start a 16h run onto a filling disk) ---
free_gb=$(df --output=avail -BG /mnt/nvme1 | tail -1 | tr -dc '0-9')
if [ -z "${free_gb}" ]; then
  echo "FATAL: cannot determine free space on /mnt/nvme1"; exit 1
fi
if [ "${free_gb}" -lt "${MIN_FREE_GB}" ]; then
  echo "FATAL: only ${free_gb}G free on /mnt/nvme1 (need ${MIN_FREE_GB}G)"; exit 1
fi
mkdir -p "${RUN_DIR}"

docker network create ${NET} 2>/dev/null || true

start_carla() {
  docker rm -f ddrive-carla 2>/dev/null || true
  docker run -d --name ddrive-carla --network ${NET} \
    --memory=16g --cpus=8 --gpus device=0 \
    carlasim/carla:0.9.15 \
    /bin/bash ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000
  sleep 30
}

run_leaderboard() {
  docker rm -f ddrive-agent 2>/dev/null || true
  # RESUME=True is safe on a fresh checkpoint file (leaderboard treats it as new).
  docker run --rm --name ddrive-agent --network ${NET} \
    --memory=20g --cpus=8 --shm-size=4g --gpus device=0 \
    -v "${DD_REPO}":/workspace/DiffusionDrive \
    -v "${B2D_REPO}":/workspace/Bench2Drive \
    -v /mnt/nvme1:/mnt/nvme1 \
    -e PYTHONPATH=/workspace/DiffusionDrive:/workspace/Bench2Drive/leaderboard:/workspace/Bench2Drive/scenario_runner:/workspace/Bench2Drive/carla/PythonAPI:/workspace/Bench2Drive/carla/PythonAPI/carla \
    -e SCENARIO_RUNNER_ROOT=/workspace/Bench2Drive/scenario_runner \
    -e LEADERBOARD_ROOT=/workspace/Bench2Drive/leaderboard \
    -e IS_BENCH2DRIVE=True \
    -e SAVE_PATH="${RUN_DIR}/frames" \
    -e ROUTES="${ROUTES_XML}" \
    -w /workspace/Bench2Drive \
    diffusiondrive:blackwell-carla \
    python3 leaderboard/leaderboard/leaderboard_evaluator.py \
      --routes="${ROUTES_XML}" --repetitions=1 --track=SENSORS \
      --checkpoint="${RUN_DIR}/results.json" \
      --agent=/workspace/DiffusionDrive/navsim/evaluation/carla_closed_loop/diffusiondrive_agent.py \
      --agent-config="${CKPT}" \
      --resume=True --host=ddrive-carla --port=2000 --traffic-manager-port=8000 \
      2>&1 | tee -a "${RUN_DIR}/evaluation.log"
}

# --- crash-resume loop: restart CARLA + evaluator until routes complete ---
attempt=0
max_attempts=40
while [ ${attempt} -lt ${max_attempts} ]; do
  attempt=$((attempt + 1))
  echo "=== attempt ${attempt} $(date) ===" | tee -a "${RUN_DIR}/evaluation.log"
  start_carla
  if run_leaderboard; then
    echo "Evaluator exited cleanly." | tee -a "${RUN_DIR}/evaluation.log"
    break
  fi
  echo "Evaluator crashed; resuming from checkpoint." | tee -a "${RUN_DIR}/evaluation.log"
  sleep 10
done
docker rm -f ddrive-carla 2>/dev/null || true
echo "Results: ${RUN_DIR}/results.json"
