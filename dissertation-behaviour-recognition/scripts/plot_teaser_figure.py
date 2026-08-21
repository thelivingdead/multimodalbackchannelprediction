#!/usr/bin/env python3
"""Clean qualitative figure: two people talking + listener face + pose + who speaks.

Layout is sparse on purpose (no overlapping chips, no text on faces).
Clips are TEST windows where the two heads sit left and right of the frame.

Default pair (table-style, not a webcam stack):
  gold_020  human nod, partner speaks the whole minute, pose 86°
  gold_024  human unclear, partner speaks the whole minute, pose 13.5°

Mac / otter, first run downloads RealTalk annotations (241 MB) plus two
video members (about 0.4–0.7 GB each), then deletes the avis. Later runs
reuse cache/teaser/*_scene_*.npy and finish in seconds.

    # 1) list TEST clips and which look left-right (FAST: annotations only)
    python scripts/plot_teaser_figure.py --scan

    # 2) build the figure (slow only the first time)
    python scripts/plot_teaser_figure.py
    python scripts/plot_teaser_figure.py --fast          # smaller videos
    python scripts/plot_teaser_figure.py --ids gold_020,gold_024

Otter (scratch venv; do not use otter48 for /scratch/db01550):

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    /scratch/db01550/venv/bin/python scripts/plot_teaser_figure.py --scan
    /scratch/db01550/venv/bin/python scripts/plot_teaser_figure.py --fast

Does not train, does not rescore TEST, does not write rgb16 npz.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "teaser"
OUT = ROOT / "figures" / "paper"
INDEX_JSON = ROOT / "results" / "video_shard_index.json"
GOLD_CSV = ROOT / "results" / "gold_dataset_summary.csv"
EVENTS = ROOT / "data" / "gold" / "events.csv"
FEAT = ROOT / "features" / "gold"
FPS = 25.0
TAU = 16.35
N_FRAMES = 5
SHARD_URL = (
    "https://huggingface.co/datasets/scottgeng00/realtalk/resolve/main/videos/{}"
)
ANN_URL = (
    "https://huggingface.co/datasets/scottgeng00/realtalk/resolve/main/"
    "annotations.tar.gz"
)
UA = {"User-Agent": "Mozilla/5.0 (dissertation-teaser)"}

BLUE = "#2f5f8a"
ORANGE = "#d07a1a"
GREEN = "#2f7d4a"
RED = "#b42318"
GREY = "#4b5563"
INK = "#111827"
LINE = "#e5e7eb"

# gold_020: nod, partner talks 100%, large pose, heads on one horizontal line.
# gold_024: unclear, partner talks 100%, pose below τ. Skip gold_016 (webcam look).
DEFAULT_IDS = ["gold_020", "gold_024"]
FAST_IDS = ["gold_021", "gold_024"]  # 395 MB + 545 MB

HEADLINES = {
    "gold_020": "Clear nod while the partner holds the floor",
    "gold_021": "Clear nod (smaller video, faster download)",
    "gold_024": "Unclear — no clear nod",
}


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit("STOP: pip install imageio-ffmpeg") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def resize_sq(rgb: np.ndarray, size: int) -> np.ndarray:
    try:
        from PIL import Image
        try:
            resample = Image.Resampling.BICUBIC
        except AttributeError:
            resample = Image.BICUBIC
        return np.asarray(Image.fromarray(rgb).resize((size, size), resample))
    except ImportError:
        import cv2
        return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def load_gold(sample_id: str) -> dict:
    g = pd.read_csv(GOLD_CSV)
    r = g[g["sample_id"].astype(str) == sample_id].iloc[0]
    return {
        "sample_id": sample_id,
        "video_id": str(r.video_id),
        "person": str(r.person),
        "label": int(r.label),
        "start_frame": int(r.start_frame),
        "end_frame": int(r.end_frame),
    }


def load_event(video_id: str) -> tuple[float, float]:
    ev = pd.read_csv(EVENTS)
    r = ev[ev["video_id"].astype(str) == video_id].iloc[0]
    return float(r.start_s), float(r.end_s)


def load_preds(sample_id: str) -> dict:
    def one(path: Path, pred_col: str, id_col: str = "sample_id") -> int:
        df = pd.read_csv(path)
        if id_col not in df.columns:
            id_col = "clip_id"
        hit = df[df[id_col].astype(str) == sample_id]
        return int(hit.iloc[0][pred_col]) if len(hit) else -1

    return {
        "rule": one(ROOT / "results" / "rule_test_predictions.csv", "pred"),
        "cnn": one(ROOT / "results" / "classifier_test_predictions.csv", "pred"),
        "frozen": one(
            ROOT / "results" / "videomae_frozen_head" / "predictions.csv", "pred"
        ),
        "ft80": one(
            ROOT / "results" / "videomae_finetuned" / "predictions_test.csv",
            "pred", "clip_id",
        ),
    }


def load_pose_x(sample_id: str) -> np.ndarray:
    z = np.load(FEAT / f"{sample_id}.npz", allow_pickle=True)
    return np.asarray(z["rotation_xyz"], dtype=float)[:, 0]


def word(pred: int) -> str:
    if pred == 1:
        return "nod"
    if pred == 0:
        return "unclear"
    return "?"


def choose_indices(lo: int, hi: int, n: int) -> np.ndarray:
    """Stay inside 20–80% of the window so Skin Deep title cards are skipped."""
    span = hi - lo
    a = lo + 0.20 * span
    b = lo + 0.80 * span
    return np.linspace(a, b, n).round().astype(np.int64)


def ensure_annotation_json(video_id: str) -> Path:
    dest = CACHE / "metadata" / f"{video_id}.json"
    if dest.exists():
        return dest
    CACHE.mkdir(parents=True, exist_ok=True)
    tar_path = CACHE / "annotations.tar.gz"
    if not tar_path.exists():
        print("Downloading RealTalk annotations (241 MB)…")
        with requests.get(ANN_URL, headers=UA, stream=True, timeout=(60, 180)) as r:
            r.raise_for_status()
            tmp = tar_path.with_suffix(".part")
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
                        n += len(chunk)
            tmp.rename(tar_path)
    print("Extracting", video_id, "from annotations (one pass)")
    want = f"metadata/{video_id}.json"
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if member.name == want:
                tf.extract(member, path=CACHE)
                break
        else:
            raise SystemExit(f"STOP: {want} not in annotations.tar.gz")
    return dest


def slim_speaker(video_id: str, lo: int, hi: int) -> dict:
    cache_path = CACHE / f"{video_id}_{lo}_{hi}_speaker.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    raw = json.loads(ensure_annotation_json(video_id).read_text())
    n = hi - lo + 1
    speaker = np.empty(n, dtype=object)
    bbox0 = np.full((n, 4), np.nan)
    bbox1 = np.full((n, 4), np.nan)
    for i, f in enumerate(range(lo, hi + 1)):
        rec = raw.get(str(f), {}) or {}
        speaker[i] = rec.get("current_speaker")
        people = rec.get("people") or {}
        for key, arr in (("p0", bbox0), ("p1", bbox1)):
            if key not in people:
                continue
            b = people[key].get("bbox")
            if b is not None:
                arr[i] = np.asarray(b, dtype=float)[:4]
    out = {"speaker": speaker, "bbox0": bbox0, "bbox1": bbox1}
    np.savez(cache_path, **out)
    return out


def _drain(stream, sink: bytearray) -> None:
    while True:
        chunk = stream.read(1 << 16)
        if not chunk:
            break
        sink += chunk


def decode_frames_file(video_path: Path, frame_indices: np.ndarray) -> np.ndarray:
    exe = ffmpeg_exe()
    expr = "+".join(f"eq(n,{int(i)})" for i in frame_indices)
    cmd = [
        exe, "-hide_banner", "-loglevel", "info",
        "-an", "-sn", "-dn", "-i", str(video_path),
        "-vf", f"select='{expr}'", "-vsync", "0",
        "-frames:v", str(len(frame_indices)),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_buf, err_buf = bytearray(), bytearray()
    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_buf))
    t_err = threading.Thread(target=_drain, args=(proc.stderr, err_buf))
    t_out.start()
    t_err.start()
    proc.wait(timeout=600)
    t_out.join()
    t_err.join()
    err = bytes(err_buf).decode("utf-8", "replace")
    import re
    match = re.search(r"Video: .{0,200}?(\d{2,6})x(\d{2,6})", err)
    if not match:
        raise RuntimeError("ffmpeg found no video size: " + err[-400:])
    width, height = int(match.group(1)), int(match.group(2))
    nbytes = width * height * 3
    n_out = len(out_buf) // nbytes
    if n_out < len(frame_indices):
        raise RuntimeError(f"short_decode wanted {len(frame_indices)} got {n_out}")
    frames = np.frombuffer(bytes(out_buf[: n_out * nbytes]), np.uint8)
    return frames.reshape(n_out, height, width, 3).copy()


def fetch_scene_frames(video_id: str, frame_indices: np.ndarray) -> np.ndarray:
    tag = "_".join(str(int(i)) for i in frame_indices)
    npy = CACHE / f"{video_id}_scene_{tag}.npy"
    if npy.exists():
        return np.load(npy)
    index = json.loads(INDEX_JSON.read_text())
    if video_id not in index:
        raise SystemExit(f"STOP: {video_id} missing from video_shard_index.json")
    entry = index[video_id]
    url = SHARD_URL.format(entry["shard"])
    offset, size = int(entry["offset"]), int(entry["size"])
    CACHE.mkdir(parents=True, exist_ok=True)
    avi = CACHE / f"{video_id}.avi"
    if not (avi.exists() and avi.stat().st_size == size):
        print(f"Downloading {video_id} ({size/1e6:.0f} MB) — once, then deleted")
        headers = dict(UA)
        headers["Range"] = f"bytes={offset}-{offset + size - 1}"
        with requests.get(url, headers=headers, stream=True, timeout=(60, 600)) as r:
            if r.status_code != 206:
                raise SystemExit(f"STOP: HTTP {r.status_code} for {video_id} (need 206)")
            tmp = avi.with_suffix(".part")
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
                        n += len(chunk)
                        if n % (80 << 20) < (1 << 20):
                            print(f"  {n/1e6:.0f} / {size/1e6:.0f} MB", flush=True)
            if tmp.stat().st_size != size:
                tmp.unlink(missing_ok=True)
                raise SystemExit("STOP: incomplete video download")
            tmp.rename(avi)
    print("Decoding", len(frame_indices), "frames from", video_id)
    frames = decode_frames_file(avi, frame_indices)
    np.save(npy, frames)
    try:
        avi.unlink()
    except OSError:
        pass
    return frames


def scale_box(xyxy, frame_shape, native=(720, 1280)):
    h, w = frame_shape[:2]
    sx, sy = w / native[1], h / native[0]
    x1, y1, x2, y2 = xyxy
    return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=float)


def square_crop(frame: np.ndarray, xyxy, size: int = 224, expand: float = 1.35):
    if xyxy is None or not np.all(np.isfinite(xyxy)):
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = min(max(bw, bh) * expand, w, h)
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    side_i = int(round(side))
    patch = frame[y0 : y0 + side_i, x0 : x0 + side_i]
    if patch.size == 0:
        return None
    return resize_sq(patch, size)


def runs(mask: np.ndarray):
    m = np.asarray(mask, dtype=bool)
    out = []
    i = 0
    n = len(m)
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        out.append((i / FPS, (j - i) / FPS))
        i = j
    return out


def pred_color(val: int, gold: int) -> str:
    if val < 0:
        return GREY
    return GREEN if val == gold else RED


def scan_test() -> None:
    """Print which TEST clips are left-right. Fast: JSON + pose npz only."""
    g = pd.read_csv(GOLD_CSV)
    g = g[g["split"].astype(str).str.upper() == "TEST"]
    index = json.loads(INDEX_JSON.read_text()) if INDEX_JSON.exists() else {}
    print(f"{'id':10} {'y':2} {'who':3} {'layout':12} {'dx':5} {'dy':5} "
          f"{'pose°':6} {'partner':8} {'MB':5}")
    for r in g.itertuples():
        ensure_annotation_json(str(r.video_id))
        pose = load_pose_x(str(r.sample_id))
        ptp = float(np.nanmax(pose) - np.nanmin(pose))
        data = json.loads((CACHE / "metadata" / f"{r.video_id}.json").read_text())
        lo, hi = int(r.start_frame), int(r.end_frame)
        c0, c1, sp = [], [], []
        for f in range(lo, hi + 1, 12):
            rec = data.get(str(f), {}) or {}
            sp.append(rec.get("current_speaker"))
            people = rec.get("people") or {}
            for key, bucket in (("p0", c0), ("p1", c1)):
                if key in people and people[key].get("bbox"):
                    b = people[key]["bbox"]
                    bucket.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        if c0 and c1:
            m0, m1 = np.median(c0, 0), np.median(c1, 0)
            dx, dy = abs(m0[0] - m1[0]), abs(m0[1] - m1[1])
            layout = "left-right" if dx > 280 and dx > dy * 1.15 else "other"
        else:
            dx = dy = float("nan")
            layout = "missing"
        partner = "p1" if str(r.person) == "p0" else "p0"
        part = 100.0 * np.mean([s == partner for s in sp]) if sp else 0.0
        mb = index.get(str(r.video_id), {}).get("size", 0) / 1e6
        print(f"{r.sample_id:10} {int(r.label):2} {r.person:3} {layout:12} "
              f"{dx:5.0f} {dy:5.0f} {ptp:6.1f} {part:6.0f}%  {mb:5.0f}")
    print("\nSuggested: --ids gold_020,gold_024")
    print("Faster download: --fast   (= gold_021,gold_024)")


def style() -> None:
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 1.0,
    })


def _label_cell(fig, gs, text: str, color=INK) -> None:
    ax = fig.add_subplot(gs)
    ax.set_axis_off()
    if not text:
        return
    ax.text(
        0.92, 0.5, text, ha="right", va="center", fontsize=11,
        color=color, transform=ax.transAxes, linespacing=1.3,
    )


def _blank(fig, gs) -> None:
    ax = fig.add_subplot(gs)
    ax.set_axis_off()


def draw_clip(fig, gs, gold, preds, pose, speaker, frames, frame_idx) -> None:
    """gs is 7 rows x (1 label + N frames). Row 4 is times only — nothing else."""
    listener = gold["person"]
    partner = "p1" if listener == "p0" else "p0"
    lo, hi = gold["start_frame"], gold["end_frame"]
    duration = (hi - lo) / FPS
    t = np.arange(len(pose)) / FPS
    ev_a, ev_b = load_event(gold["video_id"])
    win0 = lo / FPS
    ev0, ev1 = ev_a - win0, ev_b - win0
    gold_nod = gold["label"] == 1
    sp = np.asarray(speaker["speaker"], dtype=object)[: len(pose)]
    bbox_lis = speaker["bbox0"] if listener == "p0" else speaker["bbox1"]
    bbox_par = speaker["bbox1"] if listener == "p0" else speaker["bbox0"]
    ptp = float(np.nanmax(pose) - np.nanmin(pose))
    n = frames.shape[0]
    frame_times = [(int(frame_idx[c]) - lo) / FPS for c in range(n)]

    # ---- clip title ----
    _blank(fig, gs[0, 0])
    axh = fig.add_subplot(gs[0, 1:])
    axh.set_axis_off()
    axh.text(
        0.0, 0.64,
        HEADLINES.get(gold["sample_id"], gold["sample_id"]),
        fontsize=13, fontweight="bold", color=INK, va="center", ha="left",
        transform=axh.transAxes,
    )
    axh.text(
        0.0, 0.18,
        f"{gold['sample_id']}   ·   {gold['video_id']}   ·   TEST   ·   "
        f"listener {listener}   ·   pose range {ptp:.0f}°",
        fontsize=10, color=GREY, va="center", ha="left",
        transform=axh.transAxes,
    )

    # ---- predictions: one column per frame ----
    pred_items = [
        ("Gold", gold["label"]),
        ("Rule", preds["rule"]),
        ("CNN", preds["cnn"]),
        ("Frozen VMAE", preds["frozen"]),
        ("VMAE n=80", preds["ft80"]),
    ]
    _blank(fig, gs[1, 0])
    for col, (name, val) in enumerate(pred_items):
        ax = fig.add_subplot(gs[1, col + 1])
        ax.set_axis_off()
        ax.text(0.5, 0.78, name, ha="center", va="center", fontsize=9, color=GREY,
                transform=ax.transAxes)
        ax.text(
            0.5, 0.28, word(val), ha="center", va="center", fontsize=12,
            fontweight="bold", color=pred_color(val, gold["label"]),
            transform=ax.transAxes,
        )

    # ---- dyad ----
    _label_cell(fig, gs[2, 0], "Both people")
    for col in range(n):
        ax = fig.add_subplot(gs[2, col + 1])
        fr = frames[col]
        ax.imshow(fr)
        ax.set_aspect("equal")
        ax.set_anchor("C")
        fnum = int(frame_idx[col])
        rel = int(np.clip(fnum - lo, 0, len(bbox_lis) - 1))
        for box, color, role in (
            (bbox_lis[rel], BLUE, "L"),
            (bbox_par[rel], ORANGE, "P"),
        ):
            if not np.all(np.isfinite(box)):
                continue
            other = bbox_par[rel] if role == "L" else bbox_lis[rel]
            if np.all(np.isfinite(other)) and np.allclose(box, other, atol=3):
                if role != "L":
                    continue
            x1, y1, x2, y2 = scale_box(box, fr.shape)
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, linewidth=1.6, edgecolor=color,
            ))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(LINE)
            s.set_linewidth(0.8)

    # ---- listener faces (no time text on this row) ----
    _label_cell(fig, gs[3, 0], "Listener", BLUE)
    who_at = []
    for col in range(n):
        ax = fig.add_subplot(gs[3, col + 1])
        fr = frames[col]
        fnum = int(frame_idx[col])
        rel = int(np.clip(fnum - lo, 0, len(bbox_lis) - 1))
        who = sp[rel] if rel < len(sp) else None
        who_at.append(who)
        crop = square_crop(fr, scale_box(bbox_lis[rel], fr.shape))
        if crop is None:
            ax.set_facecolor("#f3f4f6")
        else:
            ax.imshow(crop)
            ax.set_aspect("equal")
            ax.set_anchor("C")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(BLUE)
            s.set_linewidth(1.5)

    # ---- times: own row, so they cannot sit on faces or the plot ----
    _label_cell(fig, gs[4, 0], "Time")
    for col in range(n):
        ax = fig.add_subplot(gs[4, col + 1])
        ax.set_axis_off()
        who = who_at[col]
        speak_c = ORANGE if who == partner else (BLUE if who == listener else GREY)
        ax.text(
            0.5, 0.55, f"{frame_times[col]:.0f} s",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=speak_c, transform=ax.transAxes,
        )

    # ---- pose (y-ticks in col 0; data width matches the five frames) ----
    ax_py = fig.add_subplot(gs[5, 0])
    axr = fig.add_subplot(gs[5, 1:])
    axr.plot(t, pose, color=BLUE, lw=1.7, zorder=3)
    axr.axhline(TAU, color=RED, ls="--", lw=1.15, zorder=2)
    axr.axhline(-TAU, color=RED, ls="--", lw=0.8, alpha=0.4, zorder=2)
    if 0 <= ev0 <= duration:
        axr.axvspan(
            max(0.0, ev0), min(duration, ev1),
            color=GREEN if gold_nod else GREY, alpha=0.15, zorder=0,
        )
    for ts in frame_times:
        axr.axvline(ts, color="#c5cad3", lw=0.9, zorder=1)
    axr.set_xlim(0, duration)
    axr.tick_params(axis="y", length=3, labelleft=False)
    axr.tick_params(axis="x", length=0, labelbottom=False)
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_visible(False)
    axr.grid(axis="y", color=LINE, lw=0.7)
    ax_py.set_ylim(axr.get_ylim())
    ax_py.set_xlim(0, 1)
    ax_py.set_xticks([])
    ax_py.yaxis.tick_right()
    ax_py.tick_params(axis="y", labelsize=9, pad=3)
    ax_py.set_ylabel("Pose (deg)", fontsize=10, labelpad=4)
    for name, spn in ax_py.spines.items():
        spn.set_visible(name == "right")

    # ---- who speaks ----
    _label_cell(fig, gs[6, 0], "Who speaks")
    axs = fig.add_subplot(gs[6, 1:], sharex=axr)
    lis_on = np.array([s == listener for s in sp], dtype=bool)
    par_on = np.array([s == partner for s in sp], dtype=bool)
    for start, dur in runs(par_on):
        axs.broken_barh([(start, dur)], (1.15, 0.7), facecolor=ORANGE, lw=0)
    for start, dur in runs(lis_on):
        axs.broken_barh([(start, dur)], (0.15, 0.7), facecolor=BLUE, lw=0)
    if 0 <= ev0 <= duration:
        axs.axvspan(
            max(0.0, ev0), min(duration, ev1),
            color=GREEN if gold_nod else GREY, alpha=0.15, zorder=0,
        )
    for ts in frame_times:
        axs.axvline(ts, color="#c5cad3", lw=0.9, zorder=1)
    axs.set_xlim(0, duration)
    axs.set_ylim(-0.15, 2.15)
    axs.set_yticks([0.5, 1.5])
    axs.set_yticklabels([])
    axs.tick_params(axis="y", length=0)
    axs.set_xlabel("Time in the 60 s labelled window (seconds)", fontsize=11, labelpad=10)
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.text(
        duration - 1.2, 1.50, "Partner", ha="right", va="center",
        fontsize=9, color="white", fontweight="bold",
    )
    axs.text(
        duration - 1.2, 0.50, "Listener", ha="right", va="center",
        fontsize=9, color=BLUE, fontweight="bold",
    )


def plot_ids(ids: list[str]) -> None:
    style()
    n = N_FRAMES
    fig = plt.figure(figsize=(13.2, 16.4))
    outer = fig.add_gridspec(
        3, 1, height_ratios=[0.14, 1.0, 1.0], hspace=0.20,
        left=0.10, right=0.96, top=0.97, bottom=0.055,
    )
    head = fig.add_subplot(outer[0])
    head.set_axis_off()
    head.set_xlim(0, 1)
    head.set_ylim(0, 1)
    head.text(
        0.04, 0.70,
        "Listener head nod in face-to-face talk   ·   Columbia RealTalk",
        ha="left", va="center", fontsize=16, fontweight="bold", color=INK,
        transform=head.transAxes,
    )
    head.text(
        0.04, 0.22,
        "Each column is one moment in the 60 s window.  "
        "Blue box = listener.  Orange box = partner.\n"
        "Time colour = who is speaking (blue listener, orange partner).  "
        "Blue line = pose.  Dashed red = rule threshold 16.35°.  "
        "Speaker track is official RealTalk TalkNet.",
        ha="left", va="center", fontsize=10, color=GREY,
        transform=head.transAxes, linespacing=1.45,
    )

    ids_axes = (outer[1], outer[2])
    for row, sid in enumerate(ids):
        gold = load_gold(sid)
        preds = load_preds(sid)
        pose = load_pose_x(sid)
        lo, hi = gold["start_frame"], gold["end_frame"]
        speaker = slim_speaker(gold["video_id"], lo, hi)
        frame_idx = choose_indices(lo, hi, n)
        frames = fetch_scene_frames(gold["video_id"], frame_idx)
        inner = ids_axes[row].subgridspec(
            7, n + 1,
            width_ratios=[1.7] + [1.0] * n,
            height_ratios=[0.42, 0.50, 1.30, 1.10, 0.40, 1.10, 0.78],
            hspace=0.55,
            wspace=0.14,
        )
        draw_clip(fig, inner, gold, preds, pose, speaker, frames, frame_idx)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "teaser_backchannel.png"
    jpg = OUT / "teaser_backchannel.jpg"
    fig.savefig(png, dpi=240, facecolor="white")
    fig.savefig(jpg, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", png)
    print("wrote", jpg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scan", action="store_true",
                   help="list TEST clips (fast; no video download)")
    p.add_argument("--fast", action="store_true",
                   help="use smaller videos gold_021 + gold_024")
    p.add_argument("--ids", default="",
                   help="comma-separated sample_ids (default gold_020,gold_024)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.scan:
        scan_test()
        return
    if args.ids.strip():
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    elif args.fast:
        ids = list(FAST_IDS)
    else:
        ids = list(DEFAULT_IDS)
    plot_ids(ids)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
