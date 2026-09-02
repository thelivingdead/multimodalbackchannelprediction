#!/usr/bin/env python3
"""Optional: cut 60 s DEV mp4s for the annotator.

TEST is refused. Gold CSVs are not written. No labels, no models.

  python scripts/prepare_dev_annotation_clips.py

Requires yt-dlp on PATH. Writes data/windowed_annotations/clips/{sample_id}.mp4
with time 0 = gold clip start.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_protocol import CLIPS_DIR, clip_records, refuse_test_id  # noqa: E402


def main() -> None:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("STOP: yt-dlp is not on PATH. Install it, or place 60 s mp4s in data/windowed_annotations/clips/")
    clips = clip_records()
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"DEV clips: {len(clips)}. TEST not loaded.")
    for row in clips:
        sid = row["sample_id"]
        refuse_test_id(sid)
        out = CLIPS_DIR / f"{sid}.mp4"
        if out.is_file() and out.stat().st_size > 1000:
            print(f"skip {sid} (exists)")
            continue
        start = float(row["source_start_sec"])
        end = float(row["source_end_sec"])
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "-f", "mp4/best",
            "--download-sections", f"*{start:.3f}-{end:.3f}",
            "--force-keyframes-at-cuts",
            "-o", str(out),
            row["youtube_url"],
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    print(f"clips dir: {CLIPS_DIR}")


if __name__ == "__main__":
    main()
