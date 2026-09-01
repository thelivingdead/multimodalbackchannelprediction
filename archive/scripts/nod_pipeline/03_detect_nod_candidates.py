#!/usr/bin/env python3
"""Step 3 — Detect possible nod *intervals* from pitch time series.

Looks for temporal cycles (down→up or up→down→return), not single frames.
Uses Δpitch, velocity, duration, and movement magnitude.

Writes:
  outputs/nod_pipeline/candidates.csv
  data/tiny_subset/<id>/candidates.csv
  outputs/nod_pipeline/label_template.csv   (blank label column for step 4)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_utils import bandpass, format_ts, list_clip_dirs, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_pitch(csv_path: Path, person: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    col = f"{person}_pitch"
    if col not in df.columns:
        return np.array([]), np.array([])
    sub = df[["timestamp", col]].dropna()
    sub = sub[sub[col] != ""]
    return sub["timestamp"].to_numpy(float), sub[col].to_numpy(float)


def find_cycles(
    t: np.ndarray,
    pitch: np.ndarray,
    fps: float,
    min_dur: float = 0.25,
    max_dur: float = 1.4,
    min_range: float = 2.5,
    min_reversals: int = 2,
) -> list[dict]:
    """Zero-crossing / turning-point cycles on band-passed pitch."""
    if t.size < 16:
        return []
    filt = bandpass(pitch, fps, 1.0, 3.0)
    vel = np.gradient(filt, t)
    # turning points: velocity sign changes
    sign = np.sign(vel)
    sign[sign == 0] = 1
    turns = np.where(np.diff(sign) != 0)[0] + 1
    events = []
    i = 0
    while i < len(turns) - 1:
        # take 2–3 turning points as one nod (down-up or up-down-return)
        for span in (2, 3):
            if i + span >= len(turns):
                continue
            a, b = int(turns[i]), int(turns[i + span])
            dur = float(t[b] - t[a])
            if dur < min_dur or dur > max_dur:
                continue
            seg = filt[a : b + 1]
            mag = float(np.ptp(seg))  # peak-to-peak
            if mag < min_range:
                continue
            n_rev = span
            if n_rev < min_reversals:
                continue
            max_vel = float(np.max(np.abs(vel[a : b + 1])))
            score = mag * (n_rev / 3.0) * min(max_vel / 20.0, 1.5)
            events.append(
                {
                    "start_time": round(float(t[a]), 3),
                    "end_time": round(float(t[b]), 3),
                    "duration": round(dur, 3),
                    "pitch_range": round(mag, 3),
                    "max_velocity": round(max_vel, 3),
                    "n_reversals": int(n_rev),
                    "score": round(float(score), 3),
                    "reason": f"cycle {n_rev} reversals, Δpitch={mag:.1f}°, {format_ts(t[a])}–{format_ts(t[b])}",
                }
            )
        i += 1
    # merge heavy overlaps (keep higher score)
    events.sort(key=lambda e: e["start_time"])
    merged: list[dict] = []
    for e in events:
        if merged and e["start_time"] < merged[-1]["end_time"] - 0.05:
            if e["score"] > merged[-1]["score"]:
                merged[-1] = e
        else:
            merged.append(e)
    return merged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default=str(ROOT / "data" / "tiny_subset"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "nod_pipeline"))
    p.add_argument("--min-range", type=float, default=2.5)
    p.add_argument("--min-dur", type=float, default=0.25)
    p.add_argument("--max-dur", type=float, default=1.4)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for clip in list_clip_dirs(Path(args.subset)):
        meta = read_meta(clip)
        fps = float(meta.get("fps", 25))
        pose_csv = clip / "pose.csv"
        if not pose_csv.exists():
            print("skip (run 02 first)", clip.name)
            continue
        persons = list(meta.get("persons", ["p0", "p1"]))
        clip_events = []
        for person in persons:
            t, y = load_pitch(pose_csv, person)
            evs = find_cycles(
                t, y, fps, min_dur=args.min_dur, max_dur=args.max_dur, min_range=args.min_range
            )
            for e in evs:
                row = {
                    "video_id": meta["video_id"],
                    "person": person,
                    **e,
                    "label": "",  # fill in step 4
                }
                all_rows.append(row)
                clip_events.append(row)
            print(f"  {clip.name} {person}: {len(evs)} candidate nods")
        if clip_events:
            pd.DataFrame(clip_events).to_csv(clip / "candidates.csv", index=False)

    if not all_rows:
        print("No candidates. Loosen --min-range or check pitch axis in step 02.")
        pd.DataFrame(
            columns=[
                "video_id",
                "person",
                "start_time",
                "end_time",
                "duration",
                "pitch_range",
                "max_velocity",
                "n_reversals",
                "score",
                "reason",
                "label",
            ]
        ).to_csv(out / "candidates.csv", index=False)
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "candidates.csv", index=False)
    template = df[["video_id", "person", "start_time", "end_time"]].copy()
    template["label"] = ""
    template.to_csv(out / "label_template.csv", index=False)
    print("Wrote", out / "candidates.csv")
    print("Label template →", out / "label_template.csv")
    print("Example intervals:")
    for _, r in df.head(8).iterrows():
        print(
            f"  {r['video_id']} {r['person']}  "
            f"{format_ts(r['start_time'])}–{format_ts(r['end_time'])} = possible nod"
        )


if __name__ == "__main__":
    main()
