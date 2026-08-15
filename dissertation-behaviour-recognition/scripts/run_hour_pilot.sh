#!/usr/bin/env bash
# TWO-STAGE 1-hour lab run. Do not download RealTalk.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
python scripts/00_audit_repository.py
python scripts/01_check_storage.py
python scripts/make_pilot_clips.py
python scripts/02_audit_data.py
python scripts/03_inspect_emoca.py
python scripts/07_extract_features.py --split pilot
python scripts/04_normalize_annotations.py
echo
echo "=============================================================="
echo "STOP. You must annotate now (about 20–40 minutes)."
echo "  1 = clear nod"
echo "  0 = unclear"
echo
echo "  python scripts/annotate_candidates.py"
echo
echo "Then run STAGE B:"
echo "  bash scripts/run_hour_pilot_stage_b.sh"
echo "=============================================================="
