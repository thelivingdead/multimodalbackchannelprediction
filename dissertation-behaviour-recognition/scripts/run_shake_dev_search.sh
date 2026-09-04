#!/usr/bin/env bash
# Shake DEV-only jobs on otter95. Do NOT score GOLD TEST.
# Do NOT --force locked results/shake/{cnn,videomae_*}/ dirs.
# No Docker. Pin threads. Calls the three DEV trainers + compare (not a driver .py).
#
#   cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
#   bash scripts/run_shake_dev_search.sh
#
# Skip last-4-block fine-tune on CPU-only (frozen head still runs):
#   SKIP_FT=1 bash scripts/run_shake_dev_search.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/scratch/db01550/venv/bin/python}"
export OMP_NUM_THREADS=1

MANIFEST="${MANIFEST:-results/shake/pseudo_balanced/manifest_40_40.csv}"
OUT="${OUT:-results/shake/dev_search}"

echo "python: $PY"
echo "cwd: $ROOT"
echo "manifest: $MANIFEST"

"$PY" scripts/train_shake_cnn_dev.py \
    --pseudo-labels "$MANIFEST" \
    --out-dir "$OUT/cnn_40_40"

"$PY" scripts/train_videomae_shake_head_dev.py \
    --pseudo-labels "$MANIFEST" \
    --out-dir "$OUT/vmae_frozen_40_40"

if [[ "${SKIP_FT:-0}" != "1" ]]; then
    "$PY" scripts/finetune_videomae_shake_dev.py \
        --pseudo-labels "$MANIFEST" \
        --out-dir "$OUT/vmae_ft4_40_40"
else
    echo "SKIP_FT=1: not running finetune_videomae_shake_dev.py"
fi

"$PY" scripts/compare_shake_dev_search.py
echo "DEV search finished. BEST is DEV-only; do not print a winner TEST F1."
echo "Next: label 10–15 new videos never in the gold 30."
echo "Do not score GOLD TEST."
