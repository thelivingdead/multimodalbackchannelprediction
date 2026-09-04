#!/usr/bin/env bash
# Otter job: two-branch Pose CNN (temporal + scalar MLP).
# Same size as the locked 1D CNN plus a 2→8→4 head. CPU is enough.
# DEV only. Does not overwrite the locked original CNN. TEST is not loaded.
#
#   ssh otterdiv
#   cd ~/multimodalbackchannelprediction
#   git pull
#   cd dissertation-behaviour-recognition
#   bash scripts/run_pose_cnn_scalar_branch_otter.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/scratch/db01550/venv/bin/python}"
LOGDIR="${LOGDIR:-$ROOT/logs}"
mkdir -p "$LOGDIR"

if [[ ! -x "$PY" ]]; then
  echo "STOP: python not found at $PY"
  exit 1
fi

if [[ -f results/windowed_nod/pose_cnn_loco_dev_scalar_branch/metrics_dev.json ]]; then
  echo "already written; skipping train."
else
  echo "TEST will not be loaded."
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "$PY" \
    scripts/crossval_windowed_pose_cnn_scalar_branch_dev.py \
    2>&1 | tee "$LOGDIR/pose_cnn_scalar_branch_dev.log"
fi

echo "done. Commit results/windowed_nod/pose_cnn_loco_dev_scalar_branch/ and push."
