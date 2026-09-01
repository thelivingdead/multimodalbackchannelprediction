#!/usr/bin/env python3
"""Step 4 — Ground truth from *visual* checking, not pose thresholds.

Copies the candidate template to labels.csv if missing, builds an HTML
review page (frames + timestamps), and accepts filled labels.

Label set (default 2-class):
  nod | non-nod

Optional 3-class (--classes 3):
  nod | other-head-motion | neutral

CSV columns:
  video_id, start_time, end_time, person, label
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_utils import format_ts, list_clip_dirs, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VALID_2 = {"nod", "non-nod"}
VALID_3 = {"nod", "other-head-motion", "neutral"}


def extract_preview_frames(clip_mp4: Path, times: list[float], dest_dir: Path) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except ImportError:
        return []
    cap = cv2.VideoCapture(str(clip_mp4))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    names = []
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        name = f"preview_{i:03d}_{t:.2f}s.jpg"
        cv2.imwrite(str(dest_dir / name), frame)
        names.append(name)
    cap.release()
    return names


def build_html(df: pd.DataFrame, subset: Path, out_html: Path) -> None:
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Nod GT review</title>",
        "<style>body{font-family:sans-serif;max-width:960px;margin:2rem auto}"
        " .card{border:1px solid #ccc;padding:1rem;margin:1rem 0}"
        " img{max-width:220px;margin-right:6px} code{background:#eee;padding:2px 6px}</style>",
        "<h1>Manual ground-truth review</h1>",
        "<p>Watch the clip around each interval. Fill <code>label</code> in "
        "<code>outputs/nod_pipeline/labels.csv</code> with <b>nod</b> or <b>non-nod</b> "
        "(or 3-class: nod / other-head-motion / neutral).</p>",
    ]
    for (vid, person), g in df.groupby(["video_id", "person"]):
        clip = subset / vid / "clip.mp4"
        parts.append(f"<h2>{vid} · {person}</h2>")
        for _, r in g.iterrows():
            mid = 0.5 * (float(r.start_time) + float(r.end_time))
            imgs = []
            if clip.exists() and clip.stat().st_size > 100:
                imgs = extract_preview_frames(
                    clip,
                    [float(r.start_time), mid, float(r.end_time)],
                    subset / vid / "review_frames",
                )
            img_html = "".join(
                f"<img src='{(subset / vid / 'review_frames' / n).as_posix()}'/>" for n in imgs
            )
            parts.append(
                f"<div class='card'><p><code>{vid}</code> {person} "
                f"{format_ts(float(r.start_time))}–{format_ts(float(r.end_time))}</p>"
                f"{img_html}<p>label: <em>(fill in CSV)</em></p></div>"
            )
    out_html.write_text("\n".join(parts))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default=str(ROOT / "data" / "tiny_subset"))
    p.add_argument("--candidates", default=str(ROOT / "outputs" / "nod_pipeline" / "candidates.csv"))
    p.add_argument("--labels", default=str(ROOT / "outputs" / "nod_pipeline" / "labels.csv"))
    p.add_argument("--classes", type=int, default=2, choices=[2, 3])
    p.add_argument("--init-only", action="store_true", help="Create blank labels.csv + HTML, then exit")
    p.add_argument("--validate", action="store_true", help="Check labels.csv is complete")
    p.add_argument("--demo-fill", action="store_true", help="Auto-fill synthetic GT (demo only)")
    args = p.parse_args()

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        raise SystemExit("Run 03_detect_nod_candidates.py first.")
    cand = pd.read_csv(cand_path)
    labels_path = Path(args.labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    if not labels_path.exists() or args.init_only:
        lab = cand[["video_id", "person", "start_time", "end_time"]].copy()
        lab["label"] = ""
        if args.demo_fill:
            # Demo: high-score cycles → nod, rest non-nod
            scores = cand["score"] if "score" in cand.columns else pd.Series([1] * len(cand))
            thr = float(scores.median()) if len(scores) else 0
            lab["label"] = ["nod" if s >= thr else "non-nod" for s in scores]
        lab.to_csv(labels_path, index=False)
        build_html(cand, Path(args.subset), labels_path.parent / "review.html")
        print("Created", labels_path)
        print("Open review page:", labels_path.parent / "review.html")
        print("Edit labels, then re-run with --validate")
        if args.init_only and not args.demo_fill:
            return

    lab = pd.read_csv(labels_path)
    allowed = VALID_3 if args.classes == 3 else VALID_2
    lab["label"] = lab["label"].astype(str).str.strip().str.lower()
    blank = lab[lab["label"].isin(["", "nan", "none"])]
    bad = lab[~lab["label"].isin(allowed | {"", "nan", "none"})]
    if args.validate or args.demo_fill:
        print(f"{len(lab) - len(blank)}/{len(lab)} labelled")
        if len(bad):
            print("Invalid labels:\n", bad)
            raise SystemExit(1)
        if len(blank) and not args.demo_fill:
            print("Still unlabelled:\n", blank.head())
            raise SystemExit("Fill remaining labels before step 05.")
        print("Labels OK →", labels_path)
        print(lab["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
