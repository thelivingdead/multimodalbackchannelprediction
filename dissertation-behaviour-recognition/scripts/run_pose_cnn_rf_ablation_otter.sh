#!/usr/bin/env bash
# Otter job: DEV Pose CNN receptive-field ablation.
# Kernels 5,5,3 / 11,9,7 / 21,15,13 on the locked 75-frame windowed input.
# CPU is enough. Does not overwrite the locked original CNN. TEST is not loaded.
# Fusion search and the TEST return-ratio rule are not run.
#
#   ssh otterdiv
#   cd ~/multimodalbackchannelprediction
#   git pull
#   cd dissertation-behaviour-recognition
#   bash scripts/run_pose_cnn_rf_ablation_otter.sh

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

LOCKED="results/windowed_nod/pose_cnn_loco_dev/metrics_dev.json"
OUT="results/windowed_nod/pose_cnn_loco_dev_rf_ablation/comparison.json"

if [[ ! -f "$LOCKED" ]]; then
  echo "STOP: missing locked original $LOCKED"
  exit 1
fi

if [[ -f "$OUT" ]]; then
  echo "already written; skipping train."
  exit 0
fi

echo "TEST will not be loaded."
echo "Input is 75 frames at 25 fps, not 128 resampled steps."
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "$PY" \
  scripts/crossval_windowed_pose_cnn_rf_ablation_dev.py \
  2>&1 | tee "$LOGDIR/pose_cnn_rf_ablation_dev.log"

echo "done. Commit results/windowed_nod/pose_cnn_loco_dev_rf_ablation/ and push."
echo "Do not score TEST."
