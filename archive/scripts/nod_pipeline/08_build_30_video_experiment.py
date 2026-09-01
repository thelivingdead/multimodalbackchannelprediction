#!/usr/bin/env python3
"""Build a 30×1-min gold set + extra unlabeled train clips (storage-safe).

Gold labels are written from the *generator's known nod intervals*, not from the
rule detector. That is independent synthetic ground truth for proving the
semi-supervised loop. Replace with human labels.csv when real clips exist.

Layout
------
  data/nod30/gold_XX/     30 clips, 60 s, independent gold
  data/nod30/train_XX/    extra unlabeled clips (pseudo-label later)
  outputs/nod_pipeline/gold_labels.csv
  outputs/nod_pipeline/time_log.txt  (append)
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_utils import CLIP_SECONDS, FPS_DEFAULT, ensure_dir, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def nod_plan(video_index: int, n_nods: int, duration: float) -> list[tuple[float, float]]:
    """Deterministic, video-specific nod intervals (independent of the detector)."""
    rng = np.random.default_rng(1000 + video_index)
    events = []
    t = 3.0 + (video_index % 5) * 0.4
    for k in range(n_nods):
        dur = float(rng.uniform(0.45, 0.85))
        if t + dur >= duration - 2.0:
            break
        events.append((round(t, 3), round(t + dur, 3)))
        t += float(rng.uniform(4.5, 9.0))
    return events


def synthesise_emoca(
    n_frames: int,
    fps: float,
    nods: list[tuple[float, float]],
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    fake = {}
    for fidx in range(n_frames):
        t = fidx / fps
        nod = 0.0
        for a, b in nods:
            if a <= t <= b:
                nod = -0.16 * math.sin(2 * math.pi * 2.2 * (t - a))
                break
        pose_p0 = [
            nod + 0.02 * math.sin(0.25 * t) + float(rng.normal(0, 0.004)),
            0.01 * math.sin(0.1 * t),
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        pose_p1 = [0.01 * math.sin(0.2 * t), 0.0, 0.0, 0.0, 0.0, 0.0]
        fake[fidx] = {"p0": {"pose": pose_p0}, "p1": {"pose": pose_p1}}
    return fake


def write_clip(out_dir: Path, video_id: str, nods: list[tuple[float, float]], duration: float, fps: float, seed: int, role: str) -> None:
    n_frames = int(duration * fps)
    d = ensure_dir(out_dir / video_id)
    (d / "clip.mp4").write_bytes(b"DEMO_NO_VIDEO")
    fake = synthesise_emoca(n_frames, fps, nods, seed)
    with open(d / "emoca.pkl", "wb") as f:
        pickle.dump(fake, f)
    write_json(
        d / "meta.json",
        {
            "video_id": video_id,
            "source": "synthetic_independent_gold" if role == "gold" else "synthetic_unlabeled_train",
            "role": role,
            "fps": fps,
            "duration_s": duration,
            "start_s": 0.0,
            "n_frames": n_frames,
            "persons": ["p0", "p1"],
            "listener": "p0",
            "gold_nods": nods if role == "gold" else [],
        },
    )


def append_time(out: Path, message: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}\n")


def main() -> None:
    t0 = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "data" / "nod30"))
    p.add_argument("--n-gold", type=int, default=30)
    p.add_argument("--n-train", type=int, default=50, help="extra unlabeled 1-min clips")
    p.add_argument("--duration", type=float, default=CLIP_SECONDS)
    p.add_argument("--fps", type=float, default=FPS_DEFAULT)
    args = p.parse_args()

    out = ensure_dir(Path(args.out))
    gold_rows = []
    for i in range(args.n_gold):
        vid = f"gold_{i:02d}"
        n_nods = 4 + (i % 4)  # 4–7 nods
        nods = nod_plan(i, n_nods, args.duration)
        write_clip(out, vid, nods, args.duration, args.fps, seed=i, role="gold")
        for a, b in nods:
            gold_rows.append(
                {
                    "video_id": vid,
                    "person": "p0",
                    "start_time": a,
                    "end_time": b,
                    "label": "nod",
                    "source": "independent_synthetic_generator",
                }
            )
        print(f"{vid}: {len(nods)} gold nods")

    for i in range(args.n_train):
        vid = f"train_{i:02d}"
        nods = nod_plan(10_000 + i, 5 + (i % 3), args.duration)
        write_clip(out, vid, nods, args.duration, args.fps, seed=500 + i, role="train")
        print(f"{vid}: unlabeled (hidden {len(nods)} nods for analysis only)")

    lab_dir = ROOT / "outputs" / "nod_pipeline"
    lab_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(gold_rows).to_csv(lab_dir / "gold_labels.csv", index=False)
    inventory = {
        "n_gold_videos": args.n_gold,
        "n_train_unlabeled": args.n_train,
        "clip_seconds": args.duration,
        "fps": args.fps,
        "gold_nod_events": len(gold_rows),
        "note": (
            "Gold events come from the synthesizer, not the rule detector, "
            "and not from watching video. Replace with human labels for RealTalk."
        ),
    }
    (lab_dir / "inventory.json").write_text(json.dumps(inventory, indent=2))
    elapsed = time.time() - t0
    append_time(lab_dir / "time_log.txt", f"08_build_30_video_experiment.py  {elapsed:.1f}s  gold={args.n_gold} train={args.n_train}")
    print("Wrote", out)
    print("Gold events", len(gold_rows), f"in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
