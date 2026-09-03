#!/usr/bin/env bash
# 3 s head shake experiments still needed on Otter.
#
# Already finished, do not rerun:
#   results/windowed_shake/baselines_bacc          TEST already scored
#   results/windowed_shake/pose_cnn_loco_dev       leave one clip out
#   results/windowed_shake/videomae_loco_dev       withdrawn, crop defect
#   results/windowed_shake/videomae_loco_dev_ep12  withdrawn, crop defect
#
# Always: crop audit, three pose MIL jobs, identity-fixed VideoMAE on the
# existing 3 s target-person crops (same window ids as nod). TEST is never
# loaded. Pass --with-rgb to also fetch side-aware Haar crops and train a
# second VideoMAE. That fetch is optional because identity-fixed crops
# already exist.
#
#   ssh otter95
#   cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
#   mkdir -p logs
#   bash scripts/run_windowed_shake_3s_otter.sh
#   bash scripts/run_windowed_shake_3s_otter.sh --with-rgb

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/scratch/db01550/venv/bin/python}"
LOGDIR="${LOGDIR:-$ROOT/logs}"
RGB_DIR="${RGB_DIR:-/scratch/db01550/rgb16_windowed}"
IDENTITY_RGB="${IDENTITY_RGB:-/scratch/db01550/rgb16_windowed_identity_dev}"
SIDE_RGB_DIR="${SIDE_RGB_DIR:-/scratch/db01550/rgb16_windowed_sideaware}"
mkdir -p "$LOGDIR"

if [[ ! -x "$PY" ]]; then
  echo "STOP: python not found at $PY"
  echo "Set PY=/path/to/venv/bin/python"
  exit 1
fi

echo "python: $PY"
echo "repo:   $ROOT"

if [[ -d "$RGB_DIR" ]]; then
  echo
  echo "==== 1 crop identity on existing 3 s RGB (shake labels) ===="
  "$PY" scripts/check_windowed_rgb_crop_identity.py \
    --task shake \
    --rgb-dir "$RGB_DIR" \
    --out-dir results/windowed_shake/crop_audit \
    2>&1 | tee "$LOGDIR/shake_crop_identity.log"
else
  echo "==== 1 skip crop audit: $RGB_DIR is missing ===="
fi

echo
echo "==== 2 pose MIL, TRAIN selected, balanced 40/40 bags ===="
if [[ -f results/windowed_shake/pose_mil_balanced40_trainsel/metrics_dev.json ]]; then
  echo "already written; skipping."
else
  echo "This is the shake analogue of the nod 80 bag run, labelled 40 pos / 40 neg"
  echo "on yaw rather than the locked 75/5 z collapse."
  PYTHONUNBUFFERED=1 "$PY" scripts/train_windowed_nod_pose_mil_trainsel.py \
    --task shake \
    2>&1 | tee "$LOGDIR/shake_pose_mil_balanced40_trainsel.log"
fi

echo
echo "==== 3 pose MIL, TRAIN selected, locked 80 clip labels (75/5) ===="
if [[ -f results/windowed_shake/pose_mil_pseudo80_trainsel/metrics_dev.json ]]; then
  echo "already written; skipping."
else
  echo "Direct analogue of nod: same 80 clips, locked shake pseudo_labels.csv."
  PYTHONUNBUFFERED=1 "$PY" scripts/train_windowed_nod_pose_mil_trainsel.py \
    --task shake \
    --labels results/shake/pseudo_labels.csv \
    --out-dir results/windowed_shake/pose_mil_pseudo80_trainsel \
    --checkpoint models/windowed_shake_pose_mil_pseudo80_trainsel.pt \
    2>&1 | tee "$LOGDIR/shake_pose_mil_pseudo80_trainsel.log"
fi

echo
echo "==== 4 pose MIL, DEV selected, balanced 40/40 (comparison only) ===="
if [[ -f results/windowed_shake/pose_mil_balanced40_dev_bacc/metrics_dev.json ]]; then
  echo "already written; skipping."
else
  echo "Epoch and threshold chosen on DEV, so the number is contaminated."
  PYTHONUNBUFFERED=1 "$PY" scripts/train_windowed_nod_pose_mil_dev.py \
    --task shake \
    2>&1 | tee "$LOGDIR/shake_pose_mil_balanced40_dev.log"
fi

echo
echo "==== 5 identity-fixed VideoMAE, shake labels, existing 3 s crops ===="
if [[ -f results/windowed_shake/videomae_identity_fixed/metrics_dev.json ]]; then
  echo "already written; skipping."
elif [[ ! -d "$IDENTITY_RGB" ]]; then
  echo "STOP: missing $IDENTITY_RGB. Nod identity-fixed 3 s crops should be here."
  exit 1
else
  echo "Reuses nod 3 s identity-fixed crops. Same window ids. TEST is not read."
  echo "Unresolved identity crops are allowed up to 25 percent."
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "$PY" \
    scripts/crossval_windowed_videomae_dev.py \
    --task shake \
    --rgb-dir "$IDENTITY_RGB" \
    --out-dir results/windowed_shake/videomae_identity_fixed \
    --epochs 6 \
    --max-missing-frac 0.25 \
    2>&1 | tee "$LOGDIR/shake_videomae_identity_fixed.log"
fi

echo
echo "MIL and identity-fixed VideoMAE finished. Push these folders from Otter:"
echo "  results/windowed_shake/pose_mil_balanced40_trainsel"
echo "  results/windowed_shake/pose_mil_pseudo80_trainsel"
echo "  results/windowed_shake/pose_mil_balanced40_dev_bacc"
echo "  results/windowed_shake/crop_audit"
echo "  results/windowed_shake/videomae_identity_fixed"

if [[ "${1:-}" != "--with-rgb" ]]; then
  echo
  echo "Skipping extra side-aware Haar fetch. Re-run with --with-rgb only if"
  echo "you also want that older cropper. The withdrawn largest-face rows stay."
  exit 0
fi

echo
echo "==== 6 side-aware Haar 3 s crops then VideoMAE leave one clip out ===="
if [[ ! -f results/windowed_shake/rgb16_windowed_sideaware_fetch_summary.json ]]; then
  "$PY" scripts/fetch_rgb_windows_nod3s.py \
    --side-aware \
    --split DEV \
    --out-dir "$SIDE_RGB_DIR" \
    --summary-json results/windowed_shake/rgb16_windowed_sideaware_fetch_summary.json \
    --min-free-gb 2 \
    2>&1 | tee "$LOGDIR/shake_fetch_sideaware.log"
else
  echo "side-aware fetch summary exists; skipping fetch."
fi

"$PY" scripts/check_windowed_rgb_crop_identity.py \
  --task shake \
  --rgb-dir "$SIDE_RGB_DIR" \
  --out-dir results/windowed_shake/crop_audit \
  --tag sideaware \
    2>&1 | tee "$LOGDIR/shake_crop_identity_sideaware.log"

if [[ -f results/windowed_shake/videomae_loco_sideaware/metrics_dev.json ]]; then
  echo "side-aware VideoMAE already written; skipping."
else
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 "$PY" \
    scripts/crossval_windowed_videomae_dev.py \
    --task shake \
    --rgb-dir "$SIDE_RGB_DIR" \
    --out-dir results/windowed_shake/videomae_loco_sideaware \
    --epochs 6 \
    2>&1 | tee "$LOGDIR/shake_videomae_loco_sideaware.log"
fi

echo "Done. TEST was not loaded. Locked shake TEST rule was not rerun."
