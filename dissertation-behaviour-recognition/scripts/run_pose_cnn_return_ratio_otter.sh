#!/usr/bin/env bash
# DEV leave-one-clip-out pose CNN + return-ratio channel.
# Does not rerun the locked original CNN. Does not load TEST.
#
#   ssh otterdiv
#   cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
#   bash scripts/run_pose_cnn_return_ratio_otter.sh
#
# This is a small 1D CNN (CPU is enough). GPU is unused.

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

ORIGINAL="results/windowed_nod/pose_cnn_loco_dev/metrics_dev.json"
NEW="results/windowed_nod/pose_cnn_loco_dev_return_ratio/metrics_dev.json"

if [[ ! -f "$ORIGINAL" ]]; then
  echo "STOP: missing locked original $ORIGINAL"
  exit 1
fi

echo "python: $PY"
echo "repo:   $ROOT"
echo "Original pose CNN is locked. This run only adds return ratio."
echo "TEST will not be loaded."

if [[ -f "$NEW" ]]; then
  echo "already written: $NEW"
else
  echo "==== pose CNN + return ratio, DEV LOCO ===="
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "$PY" \
    scripts/crossval_windowed_pose_cnn_dev.py --task nod --return-ratio \
    2>&1 | tee "$LOGDIR/pose_cnn_return_ratio_loco_dev.log"
fi

echo "==== comparison figures ===="
PYTHONUNBUFFERED=1 "$PY" scripts/plot_pose_cnn_return_ratio_ablation.py \
  2>&1 | tee "$LOGDIR/pose_cnn_return_ratio_ablation.log"

echo "done. TEST was not scored."
