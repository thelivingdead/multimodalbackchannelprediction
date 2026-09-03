#!/usr/bin/env bash
# Identity-fixed VideoMAE on DEV 3 s nod windows.
# TEST is never fetched, scored, or written.
#
#   ssh otterdiv
#   cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
#   git pull
#   bash scripts/run_videomae_identity_fixed_otter.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/scratch/db01550/venv/bin/python}"
RGB_DIR="${RGB_DIR:-/scratch/db01550/rgb16_windowed_identity_dev}"
OUT="$ROOT/results/windowed_dev/videomae_identity_fixed"
LOGDIR="${LOGDIR:-$ROOT/logs}"
mkdir -p "$LOGDIR" "$OUT"

if [[ ! -x "$PY" ]]; then
  echo "STOP: python not found at $PY"
  exit 1
fi

echo "==== 1 fetch DEV crops only ===="
if [[ -f "$OUT/fetch_manifest.csv" ]] && [[ -d "$RGB_DIR" ]]; then
  echo "fetch_manifest.csv already exists; skipping fetch. Delete it to redo."
else
  "$PY" scripts/fetch_rgb_windows_identity_dev.py \
    --out-dir "$RGB_DIR" \
    --summary-json "$OUT/fetch_summary.json" \
    --min-free-gb 1 \
    | tee "$LOGDIR/identity_fetch_dev.log"
fi

echo "==== 2 identity audit (must pass before VideoMAE) ===="
"$PY" scripts/audit_target_person_crops.py \
  --rgb-dir "$RGB_DIR" \
  --out-dir "$OUT" \
  | tee "$LOGDIR/identity_audit_dev.log"

if [[ ! -f "$OUT/audit_pass.json" ]]; then
  echo "STOP: audit did not pass. Look at $OUT/crop_audit and $OUT/audit_report.json"
  echo "VideoMAE was not started."
  exit 1
fi

echo "Look at $OUT/crop_audit/*.png before leaving this running overnight."
echo "If any sheet shows the excluded person, delete $OUT/audit_pass.json and stop."

echo "==== 3 frozen VideoMAE, leave one clip out ===="
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "$PY" \
  scripts/crossval_videomae_identity_fixed.py \
  --mode frozen \
  --rgb-dir "$RGB_DIR" \
  --out-root "$OUT" \
  | tee "$LOGDIR/identity_videomae_frozen.log"

echo "==== 4 last 2 blocks unfrozen, same crops and folds ===="
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "$PY" \
  scripts/crossval_videomae_identity_fixed.py \
  --mode last_blocks \
  --rgb-dir "$RGB_DIR" \
  --out-root "$OUT" \
  | tee "$LOGDIR/identity_videomae_last_blocks.log"

echo "==== 5 comparison figure ===="
"$PY" scripts/plot_identity_fixed_comparison.py \
  | tee "$LOGDIR/identity_comparison.log"

echo "Done. TEST was not loaded. Push $OUT when you are back on the Mac."
