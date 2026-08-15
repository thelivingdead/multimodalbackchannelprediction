#!/usr/bin/env python3
"""Write results/predicted_vs_annotated.csv for the dissertation.

Joins:
  - data/gold/watch_list.csv          (which YouTube clip to open)
  - data/gold/candidates.csv          (rule predictions, if any)
  - data/gold/events.csv              (imported gold)
  - data/gold/annotation_sheet.csv    (online 1/0 sheet)

Predicted times stay blank until real EMOCA exists for that video_id.
Synthetic pilot_* rows are kept but marked source=synthetic (not RealTalk).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotations import parse_label  # noqa: E402
from src.events import Event, iou  # noqa: E402
from src.utils import format_mmss, parse_clock  # noqa: E402

GOLD = ROOT / "data" / "gold"
RES = ROOT / "results"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size < 8:
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.dropna(how="all")


def _watch_clock(w, mmss_key: str, sec_key: str) -> str:
    if w is None:
        return ""
    v = w.get(mmss_key)
    if v is not None and str(v).strip() not in ("", "nan"):
        return str(v)
    return format_mmss(parse_clock(w.get(sec_key)))


def youtube_url(vid: str, t: float | None = None) -> str:
    if len(str(vid)) != 11 or str(vid).startswith("pilot"):
        return ""
    if t is None or (isinstance(t, float) and t != t):
        return f"https://www.youtube.com/watch?v={vid}"
    return f"https://www.youtube.com/watch?v={vid}&t={int(max(0, t))}"


def main() -> None:
    watch = _read(GOLD / "watch_list.csv")
    sheet = _read(GOLD / "annotation_sheet.csv")
    events = _read(GOLD / "events.csv")
    cand = _read(GOLD / "candidates.csv")
    watch_by = {}
    if len(watch):
        for _, r in watch.iterrows():
            watch_by[str(r.video_id)] = r

    rows: list[dict] = []

    pred_events: list[Event] = []
    if len(cand) and {"video_id", "start_s", "end_s"}.issubset(cand.columns):
        for _, r in cand.iterrows():
            try:
                pred_events.append(Event(str(r.video_id), float(r.start_s), float(r.end_s), "nod"))
            except (TypeError, ValueError):
                continue

    gold_events: list[tuple[Event, int]] = []
    if len(events) and {"video_id", "start_s", "end_s", "label"}.issubset(events.columns):
        for _, r in events.iterrows():
            lab = parse_label(r.get("label"))
            if lab is None:
                continue
            try:
                gold_events.append((Event(str(r.video_id), float(r.start_s), float(r.end_s), "nod"), lab))
            except (TypeError, ValueError):
                continue
    seen_gold = {g.video_id for g, _ in gold_events}
    if len(sheet) and {"video_id"}.issubset(sheet.columns):
        for _, r in sheet.iterrows():
            vid = str(r.video_id)
            if vid in seen_gold:
                continue
            lab = parse_label(r.get("label"))
            if lab is None:
                continue
            try:
                a = parse_clock(r.get("nod_start", r.get("annotated_start_s")))
                b = parse_clock(r.get("nod_end", r.get("annotated_end_s")))
            except (TypeError, ValueError):
                continue
            if a is None or b is None:
                continue
            gold_events.append((Event(vid, a, b, "nod"), lab))
            seen_gold.add(vid)

    used_gold: set[int] = set()
    for p in pred_events:
        best_i, best = -1, 0.0
        for i, (g, lab) in enumerate(gold_events):
            if i in used_gold or lab != 1:
                continue
            v = iou(p, g)
            if v > best:
                best, best_i = v, i
        matched = best >= 0.30 and best_i >= 0
        if matched:
            used_gold.add(best_i)
        w = watch_by.get(p.video_id)
        synthetic = str(p.video_id).startswith("pilot")
        rows.append(
            {
                "video_id": p.video_id,
                "youtube_url": youtube_url(p.video_id, p.start_s) if not synthetic else "",
                "who_to_watch": ("" if w is None else w.get("who_to_watch", w.get("listener_side", ""))),
                "watch_from": _watch_clock(w, "watch_from", "watch_start_s"),
                "watch_until": _watch_clock(w, "watch_until", "watch_end_s"),
                "predicted_start": format_mmss(p.start_s),
                "predicted_end": format_mmss(p.end_s),
                "annotated_start": "" if not matched else format_mmss(gold_events[best_i][0].start_s),
                "annotated_end": "" if not matched else format_mmss(gold_events[best_i][0].end_s),
                "label": "" if not matched else gold_events[best_i][1],
                "status": "matched" if matched else "predicted_only",
                "iou": round(best, 3) if matched else "",
                "source": "synthetic_pilot" if synthetic else "realtalk",
            }
        )

    for i, (g, lab) in enumerate(gold_events):
        if i in used_gold:
            continue
        w = watch_by.get(g.video_id)
        synthetic = str(g.video_id).startswith("pilot")
        rows.append(
            {
                "video_id": g.video_id,
                "youtube_url": youtube_url(g.video_id, g.start_s) if not synthetic else "",
                "who_to_watch": ("" if w is None else w.get("who_to_watch", w.get("listener_side", ""))),
                "watch_from": _watch_clock(w, "watch_from", "watch_start_s"),
                "watch_until": _watch_clock(w, "watch_until", "watch_end_s"),
                "predicted_start": "",
                "predicted_end": "",
                "annotated_start": format_mmss(g.start_s),
                "annotated_end": format_mmss(g.end_s),
                "label": lab,
                "status": "annotated_only",
                "iou": "",
                "source": "synthetic_pilot" if synthetic else "realtalk",
            }
        )

    # Videos on the watch list with neither pred nor gold yet
    seen = {str(r["video_id"]) for r in rows}
    for vid, w in watch_by.items():
        if vid in seen:
            continue
        rows.append(
            {
                "video_id": vid,
                "youtube_url": w.get("youtube_url", youtube_url(vid, parse_clock(w.get("watch_from") or w.get("watch_start_s")))),
                "who_to_watch": w.get("who_to_watch", w.get("listener_side", "")),
                "watch_from": _watch_clock(w, "watch_from", "watch_start_s"),
                "watch_until": _watch_clock(w, "watch_until", "watch_end_s"),
                "predicted_start": "",
                "predicted_end": "",
                "annotated_start": "",
                "annotated_end": "",
                "label": "",
                "status": "watch_pending",
                "iou": "",
                "source": "realtalk",
            }
        )

    RES.mkdir(exist_ok=True)
    dest = RES / "predicted_vs_annotated.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    n_pred = sum(1 for r in rows if r["status"] in ("matched", "predicted_only"))
    n_ann = sum(1 for r in rows if r["label"] in (0, 1, "0", "1"))
    print(f"Wrote {dest}")
    print(f"rows={len(rows)} predicted_intervals={n_pred} labelled_intervals={n_ann}")
    print("Watch list:", GOLD / "watch_list.csv")
    print("Fill labels in:", GOLD / "annotation_sheet.csv")


if __name__ == "__main__":
    main()
