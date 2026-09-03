#!/usr/bin/env python3
"""Write DEV 1.5 s nod windows from the existing human event intervals.

Labels come only from data/windowed_annotations/nod_events_windowed.csv.
The pitch rule is not used. TEST event and window files are not read.
The 3 s window CSVs are not overwritten.

    python3 scripts/generate_nod_windows_dev_1p5s.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_protocol import (  # noqa: E402
    CLIP_SEC,
    DEV_SAMPLE_IDS,
    EVENTS_CSV,
    FPS,
    SPLIT_DEV,
    TEST_SAMPLE_IDS,
    WindowedProtocolError,
    atomic_to_csv,
    clip_records,
    iter_window_bounds,
    load_events,
    overlap_seconds,
    refuse_test_id,
    sec_to_rel_frame,
)

OUT = ROOT / "data" / "windowed_annotations" / "nod_windows_dev_1p5s.csv"
WINDOW_SEC = 1.5
STRIDE_SEC = 1.0


def main() -> None:
    events = load_events(EVENTS_CSV, allow_test=False)
    leaked = set(events["sample_id"].astype(str)) & set(TEST_SAMPLE_IDS)
    if leaked:
        raise SystemExit(f"STOP: TEST id in DEV events: {sorted(leaked)}")

    nods_by = {sid: [] for sid in DEV_SAMPLE_IDS}
    for rec in events.itertuples(index=False):
        sid = str(rec.sample_id)
        refuse_test_id(sid)
        if sid not in nods_by:
            raise SystemExit(f"STOP: event for unknown DEV clip {sid}")
        nods_by[sid].append((float(rec.start_sec), float(rec.end_sec)))

    clips = {row["sample_id"]: row for row in clip_records()}
    rows = []
    for sid in DEV_SAMPLE_IDS:
        refuse_test_id(sid)
        clip = clips[sid]
        bounds = iter_window_bounds(
            clip_sec=float(CLIP_SEC),
            window_sec=WINDOW_SEC,
            stride_sec=STRIDE_SEC,
        )
        nods = nods_by[sid]
        for start, end in bounds:
            overlaps = [overlap_seconds(start, end, ns, ne) for ns, ne in nods]
            rows.append(
                {
                    "window_id": f"{sid}_1p5s_w{sec_to_rel_frame(start):05d}",
                    "sample_id": sid,
                    "video_id": clip["video_id"],
                    "person": clip["person"],
                    "split": SPLIT_DEV,
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "start_frame_relative": sec_to_rel_frame(start),
                    "end_frame_relative": sec_to_rel_frame(end),
                    "label": 1 if any(x > 0.0 for x in overlaps) else 0,
                    "n_nods_touched": int(sum(1 for x in overlaps if x > 0.0)),
                    "window_sec": WINDOW_SEC,
                    "stride_sec": STRIDE_SEC,
                    "label_source": "human_nod_events",
                    "fps": FPS,
                }
            )
    frame = pd.DataFrame(rows)
    if set(frame["sample_id"]) & set(TEST_SAMPLE_IDS):
        raise SystemExit("STOP: TEST id in generated 1.5 s windows")
    if (frame["split"] != SPLIT_DEV).any():
        raise SystemExit("STOP: non-DEV split in 1.5 s windows")
    atomic_to_csv(frame, OUT)
    n_pos = int(frame["label"].sum())
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"clips {frame['sample_id'].nunique()}  windows {len(frame)}  "
          f"positive {n_pos}  negative {len(frame) - n_pos}")
    print("TEST was not read. 3 s window files were not written.")


if __name__ == "__main__":
    try:
        main()
    except WindowedProtocolError as exc:
        raise SystemExit(str(exc)) from None
