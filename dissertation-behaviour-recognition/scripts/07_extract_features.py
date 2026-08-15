#!/usr/bin/env python3
"""07 — Compact head-pose CSV per video (no frame dumps)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import default_pilot_dir, list_clip_dirs, read_meta  # noqa: E402
from src.features import extract_person_pose  # noqa: E402
from src.storage import assert_can_continue  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="pilot")
    args = p.parse_args()
    assert_can_continue()
    out = ROOT / "data" / "headpose"
    out.mkdir(parents=True, exist_ok=True)
    clips = list_clip_dirs(default_pilot_dir())
    if not clips:
        raise SystemExit("No pilot clips. Run: python scripts/make_pilot_clips.py")
    for c in clips:
        m = read_meta(c)
        pkl = c / "emoca.pkl"
        if not pkl.exists():
            print("skip no pkl", c.name)
            continue
        fps = float(m.get("fps", 25))
        person = str(m.get("listener", "p0"))
        df = extract_person_pose(pkl, person, fps)
        dest = out / f"{m['video_id']}.csv"
        df.to_csv(dest, index=False)
        print(c.name, "rows", len(df), "valid", int(df.valid.sum()), "→", dest.name)
    print("split", args.split)


if __name__ == "__main__":
    main()
