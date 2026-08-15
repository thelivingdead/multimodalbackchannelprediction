#!/usr/bin/env python3
"""06 — Freeze gold split: existing pilot → DEV; new videos → TEST.

Do not run a TEST evaluation until 15 untouched TEST videos exist.
Seed 42.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotations import load_events  # noqa: E402
from src.data import default_pilot_dir, list_clip_dirs, read_meta  # noqa: E402
from src.utils import dump_json, set_seed  # noqa: E402


def main() -> None:
    set_seed(42)
    clips = list_clip_dirs(default_pilot_dir())
    vids = [str(read_meta(c)["video_id"]) for c in clips]
    # All current annotated/pilot clips are DEV. TEST stays empty until new videos arrive.
    events = load_events(ROOT / "data" / "gold" / "events.csv")
    annotated = sorted(set(events["video_id"].astype(str))) if len(events) else []
    dev = sorted(set(annotated) | set(vids))
    test: list[str] = []
    split_dir = ROOT / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "gold_dev.txt").write_text("\n".join(dev) + ("\n" if dev else ""))
    (split_dir / "gold_test.txt").write_text("\n".join(test) + ("\n" if test else ""))
    import pandas as pd

    rows = [{"video_id": v, "split": "dev", "reason": "existing_pilot_inspected"} for v in dev]
    pd.DataFrame(rows).to_csv(split_dir / "gold_split.csv", index=False)
    dump_json(
        ROOT / "reports" / "gold_split_validation.json",
        {
            "seed": 42,
            "n_dev": len(dev),
            "n_test": len(test),
            "dev": dev,
            "test": test,
            "rule": "Current inspected videos = PILOT/DEV. TEST must be untouched new videos.",
            "overlap": [],
            "test_used_for_training": False,
        },
    )
    dump_json(
        ROOT / "reports" / "data_leakage_checks.json",
        {
            "gold_test_in_dev": False,
            "gold_test_empty": len(test) == 0,
            "seed": 42,
            "note": "TEST is empty on purpose until 15 new videos are annotated.",
        },
    )
    print(f"DEV={len(dev)} TEST={len(test)} (TEST empty until new untouched videos)")


if __name__ == "__main__":
    main()
