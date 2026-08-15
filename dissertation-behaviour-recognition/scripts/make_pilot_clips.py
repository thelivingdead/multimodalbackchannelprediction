#!/usr/bin/env python3
"""Build 10 × 1-min synthetic PILOT clips if you have no RealTalk files yet.

Independent nod intervals are stored in meta.json only. They are NOT gold.
You must still annotate 1=clear nod / 0=unclear yourself.

If you already have real 1-min clips + pkls, skip this and copy them into
data/working/pilot/<video_id>/{clip.mp4,emoca.pkl,meta.json}.
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import ensure_dirs  # noqa: E402
from src.storage import assert_can_continue  # noqa: E402
from src.utils import dump_json  # noqa: E402


def nod_plan(i: int, duration: float) -> list[tuple[float, float]]:
    rng = np.random.default_rng(1000 + i)
    events: list[tuple[float, float]] = []
    t = 4.0 + (i % 3) * 0.5
    for _ in range(4 + i % 3):
        dur = float(rng.uniform(0.45, 0.80))
        if t + dur >= duration - 2:
            break
        events.append((round(t, 3), round(t + dur, 3)))
        t += float(rng.uniform(6.0, 10.0))
    return events


def _has_real_clip_mp4(out: Path) -> bool:
    if not out.exists():
        return False
    for d in out.iterdir():
        mp4 = d / "clip.mp4"
        if mp4.exists() and mp4.stat().st_size > 5000:
            return True
    return False


def main() -> None:
    ensure_dirs()
    assert_can_continue()
    out = ROOT / "data" / "working" / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    if _has_real_clip_mp4(out):
        print("Real clip.mp4 files already in", out)
        print("Not overwriting them with synthetic placeholders.")
        return
    print("WARNING: no RealTalk videos found.")
    print("Writing SYNTHETIC placeholders so the pipeline can be tested.")
    print("These are NOT RealTalk. Do not report F1 as a dataset result.")
    print("For real annotation you need clip.mp4 + emoca.pkl per video.")
    fps, duration, n = 25.0, 60.0, 10
    n_frames = int(duration * fps)
    for i in range(n):
        vid = f"pilot_{i:02d}"
        d = out / vid
        d.mkdir(exist_ok=True)
        nods = nod_plan(i, duration)
        rng = np.random.default_rng(i)
        fake = {}
        for fidx in range(n_frames):
            t = fidx / fps
            osc = 0.0
            for a, b in nods:
                if a <= t <= b:
                    osc = -0.16 * math.sin(2 * math.pi * 2.2 * (t - a))
                    break
            pose = [osc + 0.02 * math.sin(0.25 * t) + float(rng.normal(0, 0.004)), 0.01, 0.0, 0.0, 0.0, 0.0]
            fake[fidx] = {"p0": {"pose": pose}, "p1": {"pose": [0.01, 0.0, 0.0, 0, 0, 0]}}
        (d / "clip.mp4").write_bytes(b"NO_REAL_VIDEO")
        with (d / "emoca.pkl").open("wb") as f:
            pickle.dump(fake, f)
        dump_json(
            d / "meta.json",
            {
                "video_id": vid,
                "source": "synthetic_pilot",
                "fps": fps,
                "duration_s": duration,
                "n_frames": n_frames,
                "persons": ["p0", "p1"],
                "listener": "p0",
                "hidden_nods_not_gold": nods,
                "role": "pilot",
            },
        )
        print(vid, "hidden nods (not gold)", len(nods))
    print("Wrote", out)
    print("Next: extract features, then annotate 1/0.")


if __name__ == "__main__":
    main()
