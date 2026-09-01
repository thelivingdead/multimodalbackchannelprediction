#!/usr/bin/env python3
"""Onset audit for Temporal Correspondence of Rule Detections and Annotated Nod Onsets.

Development data only. Does not load or score GOLD TEST. Does not repair
timestamps that fall outside the analysed window.

    python scripts/audit_nod_onsets.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_io import FPS  # noqa: E402
from src.utils import parse_clock  # noqa: E402

EXPERIMENT_NAME = "Temporal Correspondence of Rule Detections and Annotated Nod Onsets"
GOLD_CSV = ROOT / "data" / "gold_annotations.csv"
SHEET_CSV = ROOT / "data" / "gold" / "annotation_sheet.csv"
EVENTS_CSV = ROOT / "data" / "gold" / "events.csv"
OUT_DIR = ROOT / "results" / "temporal_dev"
AUDIT_CSV = OUT_DIR / "onset_audit.csv"

EXPECTED_DEV = [f"gold_{i:03d}" for i in range(1, 16)]
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
TEST_ID_RE = re.compile(r"^gold_(0*(1[6-9]|2[0-9]|30))$")
TEST_MSG = "REFUSING TO LOAD GOLD TEST FOR TEMPORAL CORRESPONDENCE EXPERIMENT"
FORBIDDEN_FLAGS = ("--allow-test", "--force-test", "--include-test")
AUDIT_COLUMNS = [
    "sample_id",
    "split",
    "nod_label",
    "onset_time_s",
    "window_start_s",
    "window_end_s",
    "usable",
    "exclusion_reason",
]


def refuse_forbidden_argv(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    for raw in args:
        token = raw.split("=", 1)[0]
        if token in FORBIDDEN_FLAGS:
            print(TEST_MSG)
            raise SystemExit(TEST_MSG)


def refuse_gold_test(*, sample_id: str | None = None, split: str | None = None) -> None:
    if sample_id is not None:
        sid = str(sample_id)
        if sid in TEST_IDS or TEST_ID_RE.match(sid):
            print(TEST_MSG)
            raise SystemExit(TEST_MSG)
    if split is not None:
        token = str(split).strip().upper().replace("-", "_")
        if token in {"TEST", "GOLD_TEST", "HOLDOUT", "HOLD_OUT"}:
            print(TEST_MSG)
            raise SystemExit(TEST_MSG)


def _sheet_by_video() -> dict[str, dict[str, str]]:
    with SHEET_CSV.open(newline="") as handle:
        return {str(row["video_id"]): dict(row) for row in csv.DictReader(handle)}


def _events_by_video() -> dict[str, dict[str, str]]:
    with EVENTS_CSV.open(newline="") as handle:
        return {str(row["video_id"]): dict(row) for row in csv.DictReader(handle)}


def load_dev_gold_rows() -> list[dict[str, str]]:
    """Read gold_annotations.csv, keeping DEV rows only. TEST rows are skipped, not loaded."""
    refuse_forbidden_argv()
    if not GOLD_CSV.is_file():
        raise SystemExit(f"STOP: missing {GOLD_CSV}")
    kept: list[dict[str, str]] = []
    with GOLD_CSV.open(newline="") as handle:
        for rec in csv.DictReader(handle):
            sid = str(rec["sample_id"])
            split = str(rec["split"]).strip().upper()
            if split == "TEST" or sid in TEST_IDS or TEST_ID_RE.match(sid):
                continue
            refuse_gold_test(sample_id=sid, split=split)
            if split != "DEV":
                raise SystemExit(f"STOP: unexpected split {split!r} for {sid}")
            kept.append(rec)
    ids = [str(r["sample_id"]) for r in kept]
    if ids != EXPECTED_DEV:
        raise SystemExit(f"STOP: DEV ids {ids} != {EXPECTED_DEV}")
    return kept


def audit_dev_onsets() -> tuple[list[dict[str, object]], int, int]:
    """Build the DEV onset audit. TEST onsets are never loaded."""
    gold_rows = load_dev_gold_rows()
    sheet = _sheet_by_video()
    events = _events_by_video()
    audit_rows: list[dict[str, object]] = []
    n_usable = 0
    n_excluded = 0
    for rec in gold_rows:
        sid = str(rec["sample_id"])
        split = str(rec["split"]).strip().upper()
        refuse_gold_test(sample_id=sid, split=split)
        video_id = str(rec["video_id"])
        nod_label = int(rec["label"])
        start_frame = int(rec["start_frame"])
        end_frame = int(rec["end_frame"])
        window_start_s = start_frame / FPS
        window_end_s = end_frame / FPS
        if video_id not in sheet:
            raise SystemExit(f"STOP: {sid} video_id {video_id} missing from {SHEET_CSV}")
        sheet_row = sheet[video_id]
        onset = parse_clock(sheet_row.get("nod_start"))
        if onset is None:
            raise SystemExit(f"STOP: {sid} has no parseable nod_start in {SHEET_CSV}")
        if video_id in events:
            events_onset = float(events[video_id]["start_s"])
            if abs(events_onset - onset) > 1e-6:
                exclusion_reason = (
                    "annotation_sheet nod_start and events.csv start_s disagree; not repaired"
                )
                audit_rows.append(
                    {
                        "sample_id": sid,
                        "split": "DEV",
                        "nod_label": nod_label,
                        "onset_time_s": onset,
                        "window_start_s": window_start_s,
                        "window_end_s": window_end_s,
                        "usable": "no",
                        "exclusion_reason": exclusion_reason,
                    }
                )
                if nod_label == 1:
                    n_excluded += 1
                continue
        inside = window_start_s <= onset <= window_end_s
        if nod_label != 1:
            usable = "no"
            exclusion_reason = "nod_label is 0; not a gold nod onset"
        elif not inside:
            usable = "no"
            exclusion_reason = (
                "annotated onset timestamp lies outside the analysed window "
                "(documented; not repaired)"
            )
            n_excluded += 1
        else:
            usable = "yes"
            exclusion_reason = ""
            n_usable += 1
        audit_rows.append(
            {
                "sample_id": sid,
                "split": "DEV",
                "nod_label": nod_label,
                "onset_time_s": onset,
                "window_start_s": window_start_s,
                "window_end_s": window_end_s,
                "usable": usable,
                "exclusion_reason": exclusion_reason,
            }
        )
    return audit_rows, n_usable, n_excluded


def write_onset_audit(rows: list[dict[str, object]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for row in rows:
            refuse_gold_test(sample_id=str(row["sample_id"]), split=str(row["split"]))
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "nod_label": row["nod_label"],
                    "onset_time_s": f"{float(row['onset_time_s']):.6f}",
                    "window_start_s": f"{float(row['window_start_s']):.6f}",
                    "window_end_s": f"{float(row['window_end_s']):.6f}",
                    "usable": row["usable"],
                    "exclusion_reason": row["exclusion_reason"],
                }
            )
    return AUDIT_CSV


def print_audit_counts(n_usable: int, n_excluded: int) -> None:
    print(f"USABLE_DEV_ONSETS = {n_usable}")
    print(f"EXCLUDED_DEV_ONSETS = {n_excluded}")
    print("TEST_ONSETS_LOADED = 0")


def main() -> None:
    refuse_forbidden_argv()
    rows, n_usable, n_excluded = audit_dev_onsets()
    write_onset_audit(rows)
    print_audit_counts(n_usable, n_excluded)
    print(f"wrote {AUDIT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
