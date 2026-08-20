#!/usr/bin/env python3
"""Gold-annotation figures only. No detector F1. No EMOCA download. No GPU.

Safe on otter while the Mac is still streaming pose files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting import save_publication_figure  # noqa: E402

FPS = 25.0
OUT = ROOT / "figures" / "gold_visuals"
PERSON_NAME = {"p0": "LEFT (p0)", "p1": "RIGHT (p1)"}
NOD = "#c44e52"
UNCLEAR = "#8c8c8c"
DEV = "#4c72b0"
TEST = "#dd8452"


def load_gold() -> pd.DataFrame:
    csv = ROOT / "data" / "gold_annotations.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df["split"] = df["split"].astype(str).str.upper()
        df["person"] = df["person"].astype(str)
        df["label"] = df["label"].astype(int)
        df["start_s"] = df["start_frame"].astype(int) / FPS
        df["end_s"] = df["end_frame"].astype(int) / FPS
        df["duration_s"] = df["end_s"] - df["start_s"]
        return df
    events = ROOT / "data" / "gold" / "events.csv"
    if not events.exists():
        raise SystemExit("Need data/gold_annotations.csv or data/gold/events.csv")
    ev = pd.read_csv(events)
    ev["label"] = ev["label"].astype(int)
    ev["person"] = ev.get("participant_id", pd.Series(["p0"] * len(ev))).astype(str)
    dev = set()
    tes = set()
    dpath = ROOT / "data" / "splits" / "gold_dev.txt"
    tpath = ROOT / "data" / "splits" / "gold_test.txt"
    if dpath.exists():
        dev = {ln.strip() for ln in dpath.read_text().splitlines() if ln.strip()}
    if tpath.exists():
        tes = {ln.strip() for ln in tpath.read_text().splitlines() if ln.strip()}
    ev["split"] = ["DEV" if v in dev else "TEST" if v in tes else "UNASSIGNED" for v in ev["video_id"].astype(str)]
    ev["sample_id"] = [f"gold_{i:03d}" for i in range(1, len(ev) + 1)]
    ev["start_s"] = ev["start_s"].astype(float)
    ev["end_s"] = ev["end_s"].astype(float)
    ev["duration_s"] = ev["end_s"] - ev["start_s"]
    return ev


def fig_label_counts(df: pd.DataFrame) -> None:
    n1 = int((df.label == 1).sum())
    n0 = int((df.label == 0).sum())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(["clear nod (1)", "unclear (0)"], [n1, n0], color=[NOD, UNCLEAR])
    ax.set_ylabel("clips")
    ax.set_title(f"Gold labels (n={len(df)})")
    ax.set_ylim(0, max(n1, n0) + 3)
    for b, n in zip(bars, (n1, n0)):
        ax.text(b.get_x() + b.get_width() / 2, n + 0.3, str(n), ha="center")
    save_publication_figure(fig, OUT / "label_counts", source="gold labels", force=True)


def fig_by_split(df: pd.DataFrame) -> None:
    rows = []
    for split in ("DEV", "TEST"):
        sub = df[df.split == split]
        rows.append((f"{split}\nclear nod", int((sub.label == 1).sum()), NOD if split == "DEV" else TEST))
        rows.append((f"{split}\nunclear", int((sub.label == 0).sum()), UNCLEAR))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows])
    ax.set_ylabel("clips")
    ax.set_title("Gold labels by split (15 DEV + 15 TEST)")
    save_publication_figure(fig, OUT / "labels_by_split", source="gold labels", force=True)


def fig_by_person(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(2)
    w = 0.35
    for i, (lab, colour, name) in enumerate(((1, NOD, "clear nod"), (0, UNCLEAR, "unclear"))):
        vals = [int(((df.person == p) & (df.label == lab)).sum()) for p in ("p0", "p1")]
        ax.bar(x + (i - 0.5) * w, vals, width=w, label=name, color=colour)
    ax.set_xticks(x)
    ax.set_xticklabels([PERSON_NAME["p0"], PERSON_NAME["p1"]])
    ax.set_ylabel("clips")
    ax.set_title("Gold labels by listener side")
    ax.legend()
    save_publication_figure(fig, OUT / "labels_by_person", source="gold labels", force=True)


def fig_strip(df: pd.DataFrame) -> None:
    df = df.copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    for i, r in df.iterrows():
        colour = NOD if int(r.label) == 1 else UNCLEAR
        ax.barh(i, 1.0, left=0, height=0.7, color=colour, alpha=0.9)
        side = "L" if str(r.person) == "p0" else "R"
        ax.text(1.04, i, f"{r.sample_id}  {r.video_id}  {r.split}  {side}", va="center", fontsize=8)
    ax.set_xlim(0, 2.6)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Gold set: red = clear nod, grey = unclear")
    ax.invert_yaxis()
    save_publication_figure(fig, OUT / "clip_overview", source="gold labels", force=True)


def fig_pose_traces(df: pd.DataFrame) -> None:
    feat = ROOT / "features" / "gold"
    paths = sorted(feat.glob("gold_*.npz")) if feat.exists() else []
    if not paths:
        print("No features/gold/*.npz here — skipping pose traces (that is expected on otter).")
        return
    by_id = {str(r.sample_id): r for r in df.itertuples()}
    n = len(paths)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 2.4 * rows), sharex=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, paths):
        z = np.load(path, allow_pickle=True)
        rot = np.asarray(z["rotation_xyz"], dtype=float)
        sid = path.stem
        r = by_id.get(sid)
        t = np.arange(len(rot)) / FPS
        ax.plot(t, rot[:, 0], lw=0.8, label="rot_x")
        ax.plot(t, rot[:, 1], lw=0.8, label="rot_y")
        ax.plot(t, rot[:, 2], lw=0.8, label="rot_z")
        title = sid
        if r is not None:
            lab = "nod" if int(r.label) == 1 else "unclear"
            title = f"{sid}  {r.split}  {lab}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("deg", fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"EMOCA rotation on extracted gold clips ({n}/30). Not a TEST F1 result.", fontsize=12)
    save_publication_figure(fig, OUT / "pose_traces_extracted", source="features/gold/*.npz", force=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_gold()
    if len(df) != 30:
        print(f"WARNING: expected 30 gold rows, got {len(df)}")
    fig_label_counts(df)
    fig_by_split(df)
    fig_by_person(df)
    fig_strip(df)
    fig_pose_traces(df)
    print("Wrote gold visuals (no detector scores) to", OUT)


if __name__ == "__main__":
    main()
