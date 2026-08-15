#!/usr/bin/env python3
"""Copy filled rows from annotation_sheet.csv into data/gold/events.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotations import EVENT_COLUMNS, empty_events, parse_label, save_events  # noqa: E402
from src.utils import parse_clock  # noqa: E402

SHEET = ROOT / "data" / "gold" / "annotation_sheet.csv"
EVENTS = ROOT / "data" / "gold" / "events.csv"
WATCH = ROOT / "data" / "gold" / "watch_list.csv"
LOG = ROOT / "data" / "gold" / "annotation_log.csv"
SPLITS = ROOT / "data" / "splits"


def listener_person(who: object) -> str:
    """RealTalk: p0 = LEFT, p1 = RIGHT. Use the first word only."""
    token = str(who or "").strip().upper().replace("—", " ").replace("-", " ").split()
    if token and token[0].startswith("RIGHT"):
        return "p1"
    return "p0"


def _row_times(r: pd.Series) -> tuple[float | None, float | None]:
    start_cols = ("nod_start", "annotated_start", "annotated_start_mmss", "annotated_start_s")
    end_cols = ("nod_end", "annotated_end", "annotated_end_mmss", "annotated_end_s")
    a = b = None
    for c in start_cols:
        if c in r.index:
            a = parse_clock(r.get(c))
            if a is not None:
                break
    for c in end_cols:
        if c in r.index:
            b = parse_clock(r.get(c))
            if b is not None:
                break
    return a, b


def _write_splits_from_watch() -> None:
    if not WATCH.exists():
        return
    watch = pd.read_csv(WATCH, dtype=str, keep_default_na=False).dropna(how="all")
    SPLITS.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[str]] = {"dev": [], "test": []}
    rows = []
    for _, r in watch.iterrows():
        vid = str(r.get("video_id") or "").strip()
        split = str(r.get("split") or "").strip().lower()
        if not vid or split not in by_split:
            continue
        by_split[split].append(vid)
        rows.append({"video_id": vid, "split": split, "reason": "watch_list_online_gold"})
    (SPLITS / "planned_dev.txt").write_text("\n".join(by_split["dev"]) + ("\n" if by_split["dev"] else ""))
    (SPLITS / "planned_test.txt").write_text("\n".join(by_split["test"]) + ("\n" if by_split["test"] else ""))
    (SPLITS / "gold_dev.txt").write_text("\n".join(by_split["dev"]) + ("\n" if by_split["dev"] else ""))
    (SPLITS / "gold_test.txt").write_text("\n".join(by_split["test"]) + ("\n" if by_split["test"] else ""))
    pd.DataFrame(rows).to_csv(SPLITS / "gold_split.csv", index=False)
    print(f"Splits DEV={len(by_split['dev'])} TEST={len(by_split['test'])} → {SPLITS}")


def main() -> None:
    if not SHEET.exists():
        raise SystemExit(f"Missing {SHEET}")
    sheet = pd.read_csv(SHEET, dtype=str, keep_default_na=False).dropna(how="all")
    rows: list[dict] = []
    n_skip = 0
    for _, r in sheet.iterrows():
        lab = parse_label(r.get("label"))
        if lab is None:
            n_skip += 1
            print(f"skip {r.get('video_id')}: missing label")
            continue
        a, b = _row_times(r)
        if a is None or b is None:
            n_skip += 1
            print(f"skip {r.get('video_id')}: missing nod_start/nod_end (use 0:55 style)")
            continue
        if b <= a:
            n_skip += 1
            print(f"skip {r.get('video_id')}: end_s <= start_s")
            continue
        vid = str(r.get("video_id") or "").strip()
        row = {c: "" for c in EVENT_COLUMNS}
        row.update(
            {
                "video_id": vid,
                "participant_id": listener_person(r.get("who_to_watch") or r.get("listener_side")),
                "start_s": round(a, 3),
                "end_s": round(b, 3),
                "label": lab,
                "annotator": str(r.get("annotator") or "divya"),
                "confidence": lab,
                "notes": str(r.get("notes") or "online_watch"),
            }
        )
        rows.append(row)
    gold = pd.DataFrame(rows, columns=EVENT_COLUMNS) if rows else empty_events()
    save_events(gold, EVENTS)
    n_add = len(rows)
    log_rows = [
        {
            "video_id": r.video_id,
            "video_duration_s": 60,
            "annotation_time_s": "",
            "annotator": r.annotator,
            "behaviours_annotated": "nod",
            "event_count": 1,
            "notes": r.notes,
        }
        for _, r in gold.iterrows()
    ]
    pd.DataFrame(log_rows).to_csv(LOG, index=False)
    _write_splits_from_watch()
    print(f"Saved {n_add} gold rows → {EVENTS} (skipped {n_skip})")


if __name__ == "__main__":
    main()
