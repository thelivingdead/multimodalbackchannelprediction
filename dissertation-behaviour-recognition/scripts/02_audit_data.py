#!/usr/bin/env python3
"""02 — Audit videos, EMOCA pickles, and gold CSVs that actually exist on disk."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import default_pilot_dir, list_clip_dirs, read_meta  # noqa: E402
from src.utils import dump_json  # noqa: E402


def main() -> None:
    pilot = default_pilot_dir()
    clips = list_clip_dirs(pilot)
    gold = ROOT / "data" / "gold" / "events.csv"
    n_ann = 0
    if gold.exists() and gold.stat().st_size:
        import pandas as pd

        n_ann = int(pd.read_csv(gold).shape[0])
    videos = []
    n_emoca = 0
    dur = 0.0
    for c in clips:
        m = read_meta(c)
        has_pkl = (c / "emoca.pkl").exists()
        n_emoca += int(has_pkl)
        dur += float(m.get("duration_s", 0))
        videos.append(
            {
                "video_id": m.get("video_id", c.name),
                "duration_s": m.get("duration_s"),
                "fps": m.get("fps"),
                "has_emoca": has_pkl,
                "source": m.get("source"),
            }
        )
    report = {
        "n_pilot_clips": len(clips),
        "n_with_emoca": n_emoca,
        "total_duration_s": dur,
        "n_gold_csv_rows": n_ann,
        "fps_documented": 25,
        "resolution": "unknown_until_real_mp4",
        "videos": videos,
        "missing": "RealTalk raw videos are not in this folder unless you copied them.",
        "do_not_download": ["emoca.tar.gz", "full RealTalk shards"],
    }
    dump_json(ROOT / "reports" / "data_audit.json", report)
    print(f"pilot clips={len(clips)} emoca={n_emoca} duration_s={dur:.1f} gold_rows={n_ann}")
    print("Wrote reports/data_audit.json")


if __name__ == "__main__":
    main()
