#!/usr/bin/env bash
# Otter: extract 3 s DEV audio, train MFCC LR, write DEV figures.
# TEST is not loaded. Does not overwrite locked visual dirs.
#
#   ssh otterdiv
#   cd ~/multimodalbackchannelprediction
#   git pull
#   cd dissertation-behaviour-recognition
#   bash scripts/run_windowed_audio_3s_otter.sh

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

echo "TEST will not be loaded."
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "$PY" \
  scripts/run_windowed_audio_3s_dev.py --extract --train --figures \
  2>&1 | tee "$LOGDIR/windowed_audio_3s_dev.log"

echo "done. Commit results/windowed_dev/audio_3s/ and push. Do not score TEST."
