#!/usr/bin/env python3
"""04 — Copy/normalise annotations into data/gold/events.csv without changing originals.

Accepts older nod_pipeline labels.csv if present.
Label language: 1 = clear nod, 0 = unclear.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotations import EVENT_COLUMNS, parse_label, save_events  # noqa: E402


def from_legacy(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rows = []
    for _, r in raw.iterrows():
        start = r.get("start_s", r.get("start_time", r.get("nod_start")))
        end = r.get("end_s", r.get("end_time", r.get("nod_end")))
        lab = parse_label(r.get("label", r.get("confidence", 1)))
        if lab is None:
            continue
        rows.append(
            {
                "video_id": r.get("video_id"),
                "conversation_id": r.get("conversation_id", ""),
                "dyad_id": r.get("dyad_id", ""),
                "participant_id": r.get("participant_id", r.get("person", "p0")),
                "start_s": start,
                "end_s": end,
                "label": lab,
                "annotator": r.get("annotator", "imported"),
                "confidence": lab,
                "notes": r.get("notes", "imported_legacy"),
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def main() -> None:
    dest = ROOT / "data" / "gold" / "events.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 20:
        print("Keeping existing", dest)
        return
    candidates = [
        ROOT / "data" / "gold" / "events.csv",
        ROOT.parent / "outputs" / "nod_pipeline" / "labels.csv",
        ROOT.parent / "data" / "labels" / "nod_labels.csv",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 20:
            df = from_legacy(path)
            if len(df):
                save_events(df, dest)
                print("Normalised", path, "→", dest, "n", len(df))
                return
    save_events(pd.DataFrame(columns=EVENT_COLUMNS), dest)
    print("Created empty", dest)
    print("Fill it with scripts/annotate_candidates.py  (type 1 or 0)")


if __name__ == "__main__":
    main()
