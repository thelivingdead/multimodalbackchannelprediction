#!/usr/bin/env bash
# Shake DEV-only search on otter95. Do NOT score GOLD TEST.
# Do NOT write locked dirs (videomae_frozen_head, videomae_finetuned, cnn).
# No Docker. Pin threads.
#
#   cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
#   bash scripts/run_shake_dev_search.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/scratch/db01550/venv/bin/python}"
export OMP_NUM_THREADS=1

echo "python: $PY"
echo "cwd: $ROOT"

"$PY" scripts/run_shake_dev_search.py --run-videomae
echo "DEV search finished. BEST is DEV-only; do not print a winner TEST F1."
echo "Next: label 10–15 new videos never in the gold 30."
echo "Student git-pushes themselves."
