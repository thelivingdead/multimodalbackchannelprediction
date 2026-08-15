#!/usr/bin/env python3
"""Manual annotation: type 1 = clear nod, 0 = unclear.

You review rule-proposed intervals (and may add a missed nod).
Times are logged to data/gold/annotation_log.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotations import EVENT_COLUMNS, LOG_COLUMNS, load_events, save_events  # noqa: E402
from src.data import default_pilot_dir, list_clip_dirs, read_meta  # noqa: E402
from src.features import extract_person_pose  # noqa: E402
from src.rules.nod import NodRule  # noqa: E402
from src.utils import load_yaml  # noqa: E402


def propose() -> pd.DataFrame:
    cfg = load_yaml(ROOT / "configs" / "rule_nod.yaml")
    rule = NodRule(
        min_dur=float(cfg["min_dur"]),
        max_dur=float(cfg["max_dur"]),
        min_range_deg=float(cfg["min_range_deg"]),
        min_reversals=int(cfg["min_reversals"]),
        fps=float(cfg["fps"]),
    )
    rows = []
    for c in list_clip_dirs(default_pilot_dir()):
        m = read_meta(c)
        hp = ROOT / "data" / "headpose" / f"{m['video_id']}.csv"
        if hp.exists():
            pose = pd.read_csv(hp)
        else:
            pkl = c / "emoca.pkl"
            if not pkl.exists():
                continue
            pose = extract_person_pose(pkl, str(m.get("listener", "p0")), float(m.get("fps", 25)))
        person = str(m.get("listener", "p0"))
        for e in rule.detect(pose, str(m["video_id"]), person):
            rows.append(
                {
                    "video_id": e.video_id,
                    "participant_id": person,
                    "start_s": round(e.start_s, 3),
                    "end_s": round(e.end_s, 3),
                }
            )
    df = pd.DataFrame(rows)
    dest = ROOT / "data" / "gold" / "candidates.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


def sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return "(no pose samples in this window)"
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1e-6
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    chars = [blocks[min(len(blocks) - 1, int((v - lo) / span * (len(blocks) - 1)))] for v in sampled]
    return "".join(chars)


def preview_interval(vid: str, start_s: float, end_s: float) -> None:
    clip = ROOT / "data" / "working" / "pilot" / vid / "clip.mp4"
    hp = ROOT / "data" / "headpose" / f"{vid}.csv"
    print("  --------------------------------")
    print(f"  CLIP:  {vid}")
    print(f"  PART:  {start_s:.2f}s  →  {end_s:.2f}s   (watch only this slice)")
    if len(vid) == 11 and not vid.startswith("pilot"):
        print(f"  WATCH: https://www.youtube.com/watch?v={vid}&t={int(max(0, start_s))}")
        print("         Jump to that time. Listener: p0=LEFT, p1=RIGHT (see watch_list.csv).")
    if clip.exists() and clip.stat().st_size > 5000:
        print(f"  VIDEO: {clip}")
        print("         Open that file and jump to the start time above.")
    else:
        print("  VIDEO: none (synthetic placeholder — there is nothing to watch)")
        print("         Judge from the pitch trace below.")
    if not hp.exists():
        print("  POSE:  missing head-pose CSV")
        print("  --------------------------------")
        return
    df = pd.read_csv(hp)
    pad = 0.4
    w = df[(df["time_s"] >= start_s - pad) & (df["time_s"] <= end_s + pad)]
    inside = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)]
    if "pitch" not in df.columns or inside.empty:
        print("  POSE:  no pitch in this window")
        print("  --------------------------------")
        return
    rng = float(inside["pitch"].max() - inside["pitch"].min())
    print(f"  PITCH range in window: {rng:.2f} deg")
    print(f"  PITCH trace: {sparkline(inside['pitch'].astype(float).tolist())}")
    print("         A nod is usually a down-then-up (or up-then-down) bump.")
    print("         Flat / tiny wiggle → 0    clear cycle → 1")
    if len(w) and "time_s" in w.columns:
        show = w[["time_s", "pitch"]].copy()
        show["time_s"] = show["time_s"].map(lambda x: f"{x:.2f}")
        show["pitch"] = show["pitch"].map(lambda x: f"{x:.3f}")
        print(show.head(12).to_string(index=False))
        if len(show) > 12:
            print("         ...")
    print("  --------------------------------")
    path = ROOT / "data" / "gold" / "annotation_log.csv"
    df = pd.DataFrame([row], columns=LOG_COLUMNS)
    header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=header, index=False)


def main() -> None:
    print("Classes:  1 = clear nod     0 = unclear")
    print("Also:     a = add a missed nod interval     q = quit\n")
    cand_path = ROOT / "data" / "gold" / "candidates.csv"
    if cand_path.exists() and cand_path.stat().st_size > 10:
        cand = pd.read_csv(cand_path)
    else:
        print("Proposing candidates from the nod rule (you will accept/reject them)...")
        cand = propose()
        print(f"{len(cand)} candidates → {cand_path}")
    if cand.empty:
        raise SystemExit("No candidates. Run feature extraction first.")

    gold_path = ROOT / "data" / "gold" / "events.csv"
    gold = load_events(gold_path)
    done = set()
    if len(gold):
        done = set(zip(gold.video_id.astype(str), gold.start_s.round(3), gold.end_s.round(3)))

    annotator = input("Your name/initials [divya]: ").strip() or "divya"
    t0_all = time.time()
    n_this_video = 0
    current_vid = None
    t_vid = time.time()

    for _, r in cand.iterrows():
        vid = str(r.video_id)
        a, b = float(r.start_s), float(r.end_s)
        key = (vid, round(a, 3), round(b, 3))
        if key in done:
            continue
        if current_vid != vid:
            if current_vid is not None:
                append_log(
                    {
                        "video_id": current_vid,
                        "video_duration_s": 60.0,
                        "annotation_time_s": round(time.time() - t_vid, 1),
                        "annotator": annotator,
                        "behaviours_annotated": "nod",
                        "event_count": n_this_video,
                        "notes": "candidate_review_1_or_0",
                    }
                )
            current_vid = vid
            n_this_video = 0
            t_vid = time.time()
            print(f"\n=== {vid} ===")
        print()
        preview_interval(vid, a, b)
        ans = input("    1=clear nod  0=unclear  a=add  q=quit  > ").strip().lower()
        if ans == "q":
            break
        if ans == "a":
            try:
                na = float(input("    missed start_s: "))
                nb = float(input("    missed end_s: "))
            except ValueError:
                print("    skipped add")
                continue
            row = {c: "" for c in EVENT_COLUMNS}
            row.update(
                {
                    "video_id": vid,
                    "participant_id": str(r.get("participant_id", "p0")),
                    "start_s": na,
                    "end_s": nb,
                    "label": 1,
                    "annotator": annotator,
                    "confidence": 1,
                    "notes": "annotator_added",
                }
            )
            gold = pd.concat([gold, pd.DataFrame([row])], ignore_index=True)
            save_events(gold, gold_path)
            n_this_video += 1
            continue
        if ans not in ("0", "1"):
            print("    please type 0 or 1")
            continue
        row = {c: "" for c in EVENT_COLUMNS}
        row.update(
            {
                "video_id": vid,
                "participant_id": str(r.get("participant_id", "p0")),
                "start_s": a,
                "end_s": b,
                "label": int(ans),
                "annotator": annotator,
                "confidence": int(ans),
                "notes": "candidate_review",
            }
        )
        gold = pd.concat([gold, pd.DataFrame([row])], ignore_index=True)
        save_events(gold, gold_path)
        n_this_video += 1
        done.add(key)

    if current_vid is not None:
        append_log(
            {
                "video_id": current_vid,
                "video_duration_s": 60.0,
                "annotation_time_s": round(time.time() - t_vid, 1),
                "annotator": annotator,
                "behaviours_annotated": "nod",
                "event_count": n_this_video,
                "notes": "candidate_review_1_or_0",
            }
        )
    print(f"\nSaved {gold_path}  ({len(gold)} rows, {time.time()-t0_all:.0f}s session)")
    print("Class 1 rows are CLEAR NODS used as gold positives. Class 0 are UNCLEAR (not positives).")


if __name__ == "__main__":
    main()
