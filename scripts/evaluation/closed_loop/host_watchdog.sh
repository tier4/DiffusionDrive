#!/usr/bin/env bash
# Host safety watchdog for long closed-loop benchmark runs.
# Runs OUTSIDE the containers and independent of any orchestrating session.
# If host memory, swap, or disk cross the thresholds below, it creates the
# STOP sentinel in RUN_DIR (the orchestrator halts cleanly at the next attempt
# boundary) and force-removes the eval containers immediately.
#
# Usage: nohup host_watchdog.sh <RUN_DIR> >> <RUN_DIR>/watchdog.log 2>&1 &
set -u

RUN_DIR=${1:?RUN_DIR required}
MIN_AVAIL_MEM_GB=4      # host MemAvailable floor
MAX_SWAP_USED_GB=48     # swap-thrash ceiling
MIN_NVME1_FREE_GB=40    # results disk floor
MIN_NVME2_FREE_GB=15    # docker-root disk floor
INTERVAL_S=120

log() { echo "[watchdog $(date '+%F %T')] $*"; }

emergency() {
  log "EMERGENCY: $1 — creating STOP sentinel and removing eval containers"
  touch "${RUN_DIR}/STOP"
  docker rm -f ddrive-agent ddrive-carla 2>/dev/null || true
  log "containers removed; watchdog exiting (run halted; resume with same RUN_DIR after investigation)"
  exit 0
}

log "started: RUN_DIR=${RUN_DIR} thresholds: mem>=${MIN_AVAIL_MEM_GB}G swap<=${MAX_SWAP_USED_GB}G nvme1>=${MIN_NVME1_FREE_GB}G nvme2>=${MIN_NVME2_FREE_GB}G"
tick=0
while true; do
  avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  swap_used_gb=$(free -g | awk '/Swap:/ {print $3}')
  nvme1_gb=$(df --output=avail -BG /mnt/nvme1 2>/dev/null | tail -1 | tr -dc '0-9')
  nvme2_gb=$(df --output=avail -BG /mnt/nvme2 2>/dev/null | tail -1 | tr -dc '0-9')

  [ -n "${avail_gb}" ] && [ "${avail_gb}" -lt "${MIN_AVAIL_MEM_GB}" ] && emergency "MemAvailable ${avail_gb}G < ${MIN_AVAIL_MEM_GB}G"
  [ -n "${swap_used_gb}" ] && [ "${swap_used_gb}" -gt "${MAX_SWAP_USED_GB}" ] && emergency "swap used ${swap_used_gb}G > ${MAX_SWAP_USED_GB}G"
  [ -n "${nvme1_gb}" ] && [ "${nvme1_gb}" -lt "${MIN_NVME1_FREE_GB}" ] && emergency "/mnt/nvme1 free ${nvme1_gb}G < ${MIN_NVME1_FREE_GB}G"
  [ -n "${nvme2_gb}" ] && [ "${nvme2_gb}" -lt "${MIN_NVME2_FREE_GB}" ] && emergency "/mnt/nvme2 free ${nvme2_gb}G < ${MIN_NVME2_FREE_GB}G"

  # Stop yourself when the run is over (no containers and evaluator done marker).
  if [ -f "${RUN_DIR}/DONE" ]; then
    log "DONE marker found — watchdog exiting"
    exit 0
  fi

  tick=$((tick + 1))
  if [ $((tick % 15)) -eq 0 ]; then
    log "heartbeat: mem_avail=${avail_gb}G swap_used=${swap_used_gb}G nvme1=${nvme1_gb}G nvme2=${nvme2_gb}G"
  fi
  sleep "${INTERVAL_S}"
done
