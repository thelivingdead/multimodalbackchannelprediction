#!/usr/bin/env bash
# Otter: frozen HuBERT-base on the same 3 s DEV windows as MFCC.
# Reuses wavs in AUDIO_WINDOWED_CACHE. Does not fine-tune. TEST is not loaded.
#
#   ssh otterdiv
#   cd ~/multimodalbackchannelprediction
#   git pull
#   cd dissertation-behaviour-recognition
#   bash scripts/run_windowed_hubert_3s_otter.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/scratch/db01550/venv/bin/python}"
LOGDIR="${LOGDIR:-$ROOT/logs}"
export AUDIO_WINDOWED_CACHE="${AUDIO_WINDOWED_CACHE:-/scratch/db01550/audio_windowed_dev}"
export HUBERT_WINDOWED_CACHE="${HUBERT_WINDOWED_CACHE:-/scratch/db01550/audio_windowed_hubert}"
export HF_HOME="${HF_HOME:-/scratch/db01550/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$LOGDIR" "$AUDIO_WINDOWED_CACHE" "$HUBERT_WINDOWED_CACHE" "$HF_HOME"
df -h "$HUBERT_WINDOWED_CACHE" "$AUDIO_WINDOWED_CACHE" ~ || true

if [[ ! -x "$PY" ]]; then
  echo "STOP: python not found at $PY"
  exit 1
fi

echo "TEST will not be loaded. HuBERT stays frozen."
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "$PY" \
  scripts/run_windowed_hubert_3s_dev.py --extract --train --figures \
  2>&1 | tee "$LOGDIR/windowed_hubert_3s_dev.log"

echo "done. Commit results/windowed_dev/audio_3s_hubert/ and push. Do not score TEST."
