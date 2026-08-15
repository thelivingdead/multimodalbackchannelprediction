#!/usr/bin/env python3
"""Later-phase scripts are blocked until the nod rule pilot is finished."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
name = Path(sys.argv[0]).name

if name.startswith("15") or name.startswith("16") or "videomae" in name:
    raise SystemExit("VideoMAE is blocked until results/pilot_nod_rule_metrics.json exists.")

if not (ROOT / "results" / "pilot_nod_rule_metrics.json").exists():
    raise SystemExit(f"{name}: finish the 1-hour nod rule pilot first (scripts/run_hour_pilot.sh).")

raise SystemExit(f"{name}: not part of today's 1-hour run. See README phase 2+.")
