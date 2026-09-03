#!/usr/bin/env bash
# DEV-only nod diagnostic and 1.5 s VideoMAE ablation.
# TEST is never fetched, scored, or written.
# Existing 3 s identity-fixed results are not overwritten.
#
#   cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
#   git pull
#   bash scripts/run_nod_1p5s_diagnostic_otter.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/scratch/db01550/venv/bin/python}"
RGB3="${RGB3:-/scratch/db01550/rgb16_windowed_identity_dev}"
RGB15="${RGB15:-/scratch/db01550/rgb16_windowed_identity_dev_1p5s}"
OUT3="$ROOT/results/windowed_dev/videomae_identity_fixed"
OUT15="$ROOT/results/windowed_dev/videomae_identity_fixed_1p5s"
WIN15="$ROOT/data/windowed_annotations/nod_windows_dev_1p5s.csv"
LOGDIR="${LOGDIR:-$ROOT/logs}"
mkdir -p "$LOGDIR" "$OUT15"

if [[ ! -x "$PY" ]]; then
  echo "STOP: python not found at $PY"
  exit 1
fi

echo "==== 1 drop-bias audit on existing 3 s identity-fixed crops ===="
"$PY" scripts/audit_drop_bias_identity_crops.py \
  --rgb-dir "$RGB3" \
  | tee "$LOGDIR/nod_drop_bias.log"

echo "==== 2 generate DEV 1.5 s windows from human nod events ===="
"$PY" scripts/generate_nod_windows_dev_1p5s.py \
  | tee "$LOGDIR/nod_windows_1p5s.log"

echo "==== 3 fetch identity-fixed 1.5 s RGB crops ===="
if [[ -f "$OUT15/fetch_manifest.csv" ]] && [[ -d "$RGB15" ]]; then
  echo "1.5 s fetch_manifest.csv exists; skipping fetch."
else
  "$PY" scripts/fetch_rgb_windows_identity_dev.py \
    --windows "$WIN15" \
    --out-dir "$RGB15" \
    --summary-json "$OUT15/fetch_summary.json" \
    --min-free-gb 1 \
    | tee "$LOGDIR/identity_fetch_dev_1p5s.log"
fi

echo "==== 4 identity audit for 1.5 s crops ===="
"$PY" scripts/audit_target_person_crops.py \
  --windows "$WIN15" \
  --rgb-dir "$RGB15" \
  --out-dir "$OUT15" \
  | tee "$LOGDIR/identity_audit_dev_1p5s.log"

if [[ ! -f "$OUT15/audit_pass.json" ]]; then
  echo "STOP: 1.5 s audit did not pass. VideoMAE was not started."
  exit 1
fi

echo "==== 5 VideoMAE last two blocks, 1.5 s, threshold 0.5 plus train-fold threshold ===="
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "$PY" \
  scripts/crossval_videomae_identity_1p5s.py \
  --windows "$WIN15" \
  --rgb-dir "$RGB15" \
  --out-root "$OUT15" \
  --hflip \
  --write-train-threshold \
  --run-name last_blocks_unfrozen \
  | tee "$LOGDIR/videomae_1p5s_last_blocks.log"

echo "==== 6 no-flip control (same 1.5 s crops; training flip off) ===="
if [[ -f "$OUT15/last_blocks_no_hflip/metrics.json" ]]; then
  echo "no-flip run already written; skipping."
else
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "$PY" \
    scripts/crossval_videomae_identity_1p5s.py \
    --windows "$WIN15" \
    --rgb-dir "$RGB15" \
    --out-root "$OUT15" \
    --no-hflip \
    --run-name last_blocks_no_hflip \
    | tee "$LOGDIR/videomae_1p5s_no_hflip.log"
fi

echo "==== 7 figures and summary ===="
"$PY" scripts/plot_nod_final_figures.py \
  | tee "$LOGDIR/nod_final_figures.log"
"$PY" scripts/write_nod_final_diagnostic_summary.py \
  | tee "$LOGDIR/nod_final_summary.log"

echo "Done. TEST was not loaded. 3 s identity-fixed metrics were not overwritten."
echo "Push results/windowed_dev/drop_bias_audit, videomae_identity_fixed_1p5s, final_figures."
