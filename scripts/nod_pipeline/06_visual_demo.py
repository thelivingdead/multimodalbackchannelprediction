#!/usr/bin/env python3
"""Step 6 — Visual demo: pitch graph + predicted nod intervals (+ video if present).

Writes:
  outputs/nod_pipeline/demo_<video_id>.png   static figure
  outputs/nod_pipeline/demo_<video_id>.mp4   overlay movie when OpenCV + clip exist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_utils import list_clip_dirs, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_pose(clip: Path, person: str):
    df = pd.read_csv(clip / "pose.csv")
    col = f"{person}_pitch"
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df["timestamp"].to_numpy(float), df[col].to_numpy(float)


def intervals_for(df: pd.DataFrame, video_id: str, person: str, pred_col: str):
    sub = df[(df.video_id == video_id) & (df.person == person)]
    out = []
    for _, r in sub.iterrows():
        flag = r.get(pred_col, r.get("label", 0))
        if str(flag) in ("1", "nod", "True"):
            out.append((float(r.start_time), float(r.end_time)))
        elif isinstance(flag, (int, float, np.integer, np.floating)) and int(flag) == 1:
            out.append((float(r.start_time), float(r.end_time)))
    return out


def draw_static(t, pitch, gt, pred, title, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.plot(t, pitch, color="#1f4e5f", lw=1.1, label="pitch")
    for a, b in gt:
        ax.axvspan(a, b, color="#b4532a", alpha=0.25, label="GT nod" if a == gt[0][0] else None)
    for a, b in pred:
        ax.axvspan(a, b, color="#0f6a5c", alpha=0.18, label="pred nod" if a == pred[0][0] else None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch (deg)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_overlay(clip_mp4: Path, t, pitch, pred, out_mp4: Path, fps: float) -> bool:
    try:
        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not clip_mp4.exists() or clip_mp4.stat().st_size < 100:
        return False
    cap = cv2.VideoCapture(str(clip_mp4))
    if not cap.isOpened():
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    graph_h = 180
    writer = cv2.VideoWriter(
        str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h + graph_h)
    )
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = i / src_fps
        fig, ax = plt.subplots(figsize=(w / 100, graph_h / 100), dpi=100)
        ax.plot(t, pitch, color="#1f4e5f", lw=1)
        for a, b in pred:
            ax.axvspan(a, b, color="#0f6a5c", alpha=0.25)
        ax.axvline(now, color="#b4532a", lw=1.5)
        ax.set_xlim(0, max(t[-1], 1))
        ax.set_yticks([])
        ax.set_xlabel("")
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        graph = buf[:, :, :3]
        plt.close(fig)
        graph = cv2.resize(graph, (w, graph_h))
        graph_bgr = cv2.cvtColor(graph, cv2.COLOR_RGB2BGR)
        stacked = np.vstack([frame, graph_bgr])
        writer.write(stacked)
        i += 1
        if i > int(src_fps * 60) + 5:
            break
    writer.release()
    cap.release()
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default=str(ROOT / "data" / "tiny_subset"))
    p.add_argument("--pred", default=str(ROOT / "outputs" / "nod_pipeline" / "predictions.csv"))
    p.add_argument("--labels", default=str(ROOT / "outputs" / "nod_pipeline" / "labels.csv"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "nod_pipeline"))
    p.add_argument("--video-id", default="", help="If empty, demo first test video")
    p.add_argument("--person", default="p0")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pred_path = Path(args.pred)
    if not pred_path.exists():
        # fall back to candidates as "pred"
        pred_path = out / "candidates.csv"
    pred_df = pd.read_csv(pred_path)
    lab_df = pd.read_csv(args.labels) if Path(args.labels).exists() else pred_df

    vid = args.video_id or str(pred_df["video_id"].iloc[0])
    clip = Path(args.subset) / vid
    if not (clip / "pose.csv").exists():
        raise SystemExit(f"Missing pose.csv in {clip}")
    t, pitch = load_pose(clip, args.person)
    pred_col = "pred" if "pred" in pred_df.columns else "label"
    preds = intervals_for(pred_df, vid, args.person, pred_col)
    gts = intervals_for(lab_df, vid, args.person, "label")
    png = out / f"demo_{vid}_{args.person}.png"
    draw_static(t, pitch, gts, preds, f"{vid} {args.person}: pitch + nod intervals", png)
    print("Wrote", png)
    mp4 = out / f"demo_{vid}_{args.person}.mp4"
    if render_overlay(clip / "clip.mp4", t, pitch, preds, mp4, float(read_meta(clip).get("fps", 25))):
        print("Wrote", mp4)
    else:
        print("No overlay mp4 (demo clip or OpenCV missing). Static PNG is enough for the report.")


if __name__ == "__main__":
    main()
