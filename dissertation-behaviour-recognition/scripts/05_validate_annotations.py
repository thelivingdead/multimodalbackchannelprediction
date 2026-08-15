#!/usr/bin/env python3
"""05 — Validate gold events. --mode pilot requires at least some 1/0 labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotations import load_events, parse_label, validate_events  # noqa: E402
from src.data import default_pilot_dir, list_clip_dirs, read_meta  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="pilot", choices=["pilot", "gold"])
    args = p.parse_args()
    df = load_events(ROOT / "data" / "gold" / "events.csv")
    durs = {}
    for c in list_clip_dirs(default_pilot_dir()):
        m = read_meta(c)
        durs[str(m["video_id"])] = float(m.get("duration_s", 60))
    errors = validate_events(df, durs)
    n1 = sum(parse_label(x) == 1 for x in df.get("label", []))
    n0 = sum(parse_label(x) == 0 for x in df.get("label", []))
    print(f"rows={len(df)} clear_nod(1)={n1} unclear(0)={n0}")
    if errors:
        print("errors:")
        for e in errors[:20]:
            print(" ", e)
        raise SystemExit(1)
    if args.mode == "pilot" and n1 == 0:
        raise SystemExit(
            "No class-1 (clear nod) labels yet.\n"
            "Run: python scripts/annotate_candidates.py\n"
            "Type 1 = clear nod, 0 = unclear."
        )
    print("Labels OK")


if __name__ == "__main__":
    main()
