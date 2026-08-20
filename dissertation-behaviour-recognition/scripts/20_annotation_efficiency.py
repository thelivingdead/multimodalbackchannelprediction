#!/usr/bin/env python3
"""Summarise annotation effort from data/gold/annotation_log.csv.

The 30-window gold set was annotated during a single online-watch pass
(2026-08-15): one ~60 s window per video, one 1/0 decision per window.
Per-video wall-clock timing was not recorded (the annotation_time_s column
is empty), so this script reports what exists and says so explicitly rather
than estimating. Writes results/annotation_efficiency.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import dump_json  # noqa: E402


def main() -> None:
    log = ROOT / "data" / "gold" / "annotation_log.csv"
    if not log.exists():
        raise SystemExit("Missing data/gold/annotation_log.csv")
    df = pd.read_csv(log)
    timing = pd.to_numeric(df.get("annotation_time_s"), errors="coerce").dropna()
    summary = {
        "n_videos": int(len(df)),
        "seconds_watched_total": int(pd.to_numeric(df["video_duration_s"], errors="coerce").fillna(60).sum()),
        "events_recorded_total": int(pd.to_numeric(df["event_count"], errors="coerce").fillna(0).sum()),
        "annotators": sorted(df["annotator"].astype(str).unique()),
        "per_video_timing_recorded": bool(len(df) and len(timing) == len(df)),
        "note": (
            "Per-video annotation wall-clock time was not logged during the online-watch protocol; "
            "no timing estimate is fabricated. One ~60 s window per video, one binary decision per window."
        ),
    }
    dest = ROOT / "results" / "annotation_efficiency.json"
    dump_json(dest, summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
