#!/usr/bin/env python3
"""Step 2 — Extract frame → timestamp → pitch/yaw/roll and plot pitch vs time.

Reads each clip's emoca.pkl, writes:
  pose.csv
  pitch_p0.png / pitch_p1.png

This is the first technical result: a pitch time series per 1-minute clip.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_utils import (  # noqa: E402
    euler_degrees,
    extract_axis_angle,
    list_clip_dirs,
    load_emoca_pkl,
    read_meta,
)

ROOT = Path(__file__).resolve().parents[2]


def pose_table(pkl_path: Path, fps: float, pitch_axis: int, persons: list[str]) -> list[dict]:
    data = load_emoca_pkl(pkl_path)
    rows = []
    for key, payload in data.items():
        try:
            frame = int(key)
        except (TypeError, ValueError):
            continue
        t = frame / fps
        if not isinstance(payload, dict):
            continue
        row = {"frame": frame, "timestamp": round(t, 4)}
        for person in persons:
            emb = payload.get(person, payload)
            aa = extract_axis_angle(emb)
            if aa is None:
                row[f"{person}_pitch"] = ""
                row[f"{person}_yaw"] = ""
                row[f"{person}_roll"] = ""
                continue
            pitch, yaw, roll = euler_degrees(aa, pitch_axis=pitch_axis)
            row[f"{person}_pitch"] = round(pitch, 4)
            row[f"{person}_yaw"] = round(yaw, 4)
            row[f"{person}_roll"] = round(roll, 4)
        rows.append(row)
    rows.sort(key=lambda r: r["frame"])
    return rows


def plot_pitch(rows: list[dict], person: str, out_png: Path, video_id: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = f"{person}_pitch"
    t, y = [], []
    for r in rows:
        if r.get(key) == "" or r.get(key) is None:
            continue
        t.append(r["timestamp"])
        y.append(float(r[key]))
    if not t:
        return
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(t, y, color="#1f4e5f", lw=1.1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch (deg)")
    ax.set_title(f"{video_id} · {person} · pitch vs time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default=str(ROOT / "data" / "tiny_subset"))
    p.add_argument("--pitch-axis", type=int, default=0, help="0=rx, 1=ry, 2=rz")
    args = p.parse_args()
    clips = list_clip_dirs(Path(args.subset))
    if not clips:
        raise SystemExit(f"No clips in {args.subset}. Run 01_make_tiny_subset.py first.")
    for clip in clips:
        meta = read_meta(clip)
        fps = float(meta.get("fps", 25))
        persons = list(meta.get("persons", ["p0", "p1"]))
        pkl = clip / "emoca.pkl"
        if not pkl.exists():
            print("skip (no emoca)", clip.name)
            continue
        rows = pose_table(pkl, fps, args.pitch_axis, persons)
        csv_path = clip / "pose.csv"
        if rows:
            fields = list(rows[0].keys())
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
        for person in persons:
            plot_pitch(rows, person, clip / f"pitch_{person}.png", meta["video_id"])
        print(f"{clip.name}: {len(rows)} frames → pose.csv + pitch plots")
    print("Done. Inspect data/tiny_subset/*/pitch_p0.png")


if __name__ == "__main__":
    main()
