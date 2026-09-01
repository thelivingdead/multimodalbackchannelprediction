#!/usr/bin/env python3
"""Dump review frames from a real 1-minute clip and write a blank labels.csv.

No EMOCA pickle required. You watch the video/frames and type nod / non-nod.

Example (on the lab, after the mp4 exists):

  python scripts/nod_pipeline/07_dump_frames_for_annotation.py \
    --video data/realtalk_sample/videos/5hxY5Svr2aM.mp4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to 1-minute mp4/avi")
    p.add_argument("--out", default=str(ROOT / "outputs" / "nod_pipeline" / "manual_gt"))
    p.add_argument("--every", type=float, default=0.5, help="Save one frame every N seconds")
    p.add_argument("--window", type=float, default=0.8, help="Label window length (seconds)")
    args = p.parse_args()

    try:
        import cv2
    except ImportError:
        raise SystemExit("Activate the venv first:  source .venv/bin/activate")

    video = Path(args.video).expanduser().resolve()
    if not video.exists() or video.stat().st_size < 1000:
        raise SystemExit(f"Video missing or empty: {video}\nDownload/trim the mp4 first.")

    vid = video.stem
    out = Path(args.out) / vid
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_total / fps if n_total else 60.0

    step = max(1, int(round(args.every * fps)))
    saved = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / fps
            name = f"t_{t:06.2f}s.jpg"
            cv2.imwrite(str(frames_dir / name), frame)
            saved.append((t, name))
        i += 1
    cap.release()

    # Label rows: one window starting at each saved frame (you fill 'label')
    csv_path = out / "labels.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["video_id", "start_time", "end_time", "frame_file", "label", "notes"],
        )
        w.writeheader()
        for t, name in saved:
            w.writerow(
                {
                    "video_id": vid,
                    "start_time": f"{t:.2f}",
                    "end_time": f"{min(t + args.window, duration):.2f}",
                    "frame_file": f"frames/{name}",
                    "label": "",  # nod  OR  non-nod  OR  other-head-motion
                    "notes": "",
                }
            )

    # Simple HTML so you can click through frames
    html = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>Annotate {vid}</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:1.5rem auto}",
        "img{max-width:100%;border:1px solid #ccc} .row{margin:1.2rem 0;padding:0.8rem;border:1px solid #ddd}",
        "code{background:#f4f4f4;padding:2px 6px}</style>",
        f"<h1>Manual nod labels — {vid}</h1>",
        f"<p>Video duration ~{duration:.1f}s. Fill <code>labels.csv</code>: "
        "<b>nod</b> / <b>non-nod</b> / <b>other-head-motion</b>.</p>",
        "<p>Look at the person who is <em>listening</em> (not speaking). "
        "A nod is a short down-up of the head.</p>",
    ]
    for t, name in saved:
        html.append(
            f"<div class='row'><p><code>{t:.2f}s–{min(t + args.window, duration):.2f}s</code> "
            f"file {name}</p><img src='frames/{name}' alt='{t:.2f}s'/></div>"
        )
    html_path = out / "review.html"
    html_path.write_text("\n".join(html))

    print(f"Frames:     {frames_dir}  ({len(saved)} images)")
    print(f"Labels CSV: {csv_path}")
    print(f"Review:     {html_path}")
    print()
    print("How to annotate:")
    print("  1. Open review.html  (VS Code: right-click → Open Preview / download folder)")
    print("  2. Watch the mp4 at the same times")
    print("  3. Edit labels.csv — put nod or non-nod in the label column")
    print("  4. Leave a row blank only if you have not checked it yet")


if __name__ == "__main__":
    main()
