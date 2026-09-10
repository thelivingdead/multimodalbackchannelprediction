#!/usr/bin/env python3
"""Public README teaser: 3 s yaw rule on one locked TEST shake clip.

Pose only. No face crops. Reads stored TEST predictions; does not rescore.

    python3 scripts/plot_teaser_shake_windowed.py

The 0.654 interval is the 15-clip TEST score from
results/windowed_shake/baselines_bacc/metrics.json (axis y, τ = 4.091°).
That is not the locked 60 s roll rule (z, 11.15°).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paper_figure_style import (  # noqa: E402
    GREEN,
    GREY,
    INK,
    MUTED,
    ORANGE,
    PAPER,
    SIZE_FULL,
    save,
)

METRICS = ROOT / "results" / "windowed_shake" / "baselines_bacc" / "metrics.json"
PRED = ROOT / "results" / "windowed_shake" / "baselines_bacc" / "predictions.csv"
EVENTS = ROOT / "data" / "windowed_annotations" / "shake_events_windowed_test.csv"
POSE = ROOT / "features" / "gold"
OUT = ROOT / "figures" / "paper" / "teaser_shake_windowed"
CLIP = "gold_028"
FPS = 25.0


def main() -> None:
    metrics = json.loads(METRICS.read_text())
    if int(metrics["axis"]) != 1 or str(metrics["axis_name"]) != "y":
        raise SystemExit("STOP: stored shake rule is not yaw (axis y)")
    tau = float(metrics["dev_selected_window_threshold"])
    ba = float(metrics["metrics"]["TEST"]["dev_selected_window_rule"]["balanced_accuracy"])
    boot = metrics["clip_bootstrap"]["TEST"]["dev_selected_window_rule"]["balanced_accuracy"]
    lo = float(boot["ci_lower_95"])
    hi = float(boot["ci_upper_95"])

    pred = pd.read_csv(PRED)
    part = pred[(pred["sample_id"] == CLIP) & (pred["split"] == "TEST")].sort_values("start_sec")
    if part.empty:
        raise SystemExit(f"STOP: no TEST predictions for {CLIP}")
    events = pd.read_csv(EVENTS)
    ev = events[events["sample_id"] == CLIP]
    if ev.empty:
        raise SystemExit(f"STOP: {CLIP} has no TEST shake event")

    pose_path = POSE / f"{CLIP}.npz"
    if not pose_path.exists():
        raise SystemExit(f"STOP: missing {pose_path}")
    yaw = np.asarray(np.load(pose_path)["rotation_xyz"], dtype=float)[:, 1]
    t = np.arange(len(yaw)) / FPS

    fig, axes = plt.subplots(
        2,
        1,
        figsize=SIZE_FULL,
        facecolor=PAPER,
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.18},
    )
    ax_y, ax_s = axes
    ax_y.set_facecolor(PAPER)
    ax_s.set_facecolor(PAPER)

    fired = part[part["dev_selected_pred"] == 1]
    for rec in fired.itertuples(index=False):
        ax_y.axvspan(float(rec.start_sec), float(rec.end_sec), color=GREEN, alpha=0.16, zorder=0)
        ax_s.axvspan(float(rec.start_sec), float(rec.end_sec), color=GREEN, alpha=0.12, zorder=0)
    for rec in ev.itertuples(index=False):
        ax_y.axvspan(float(rec.start_sec), float(rec.end_sec), color=ORANGE, alpha=0.45, zorder=1)
        ax_s.axvspan(float(rec.start_sec), float(rec.end_sec), color=ORANGE, alpha=0.35, zorder=1)

    ax_y.plot(t, yaw, color=INK, lw=1.15, zorder=2)
    ax_y.set_ylabel("Yaw  (°)")
    ax_y.set_title(
        f"Shake from yaw on locked TEST.  {ba:.3f}  [{lo:.3f}, {hi:.3f}]",
        loc="left",
    )
    ax_y.set_xlim(0, 60)
    for side in ("top", "right"):
        ax_y.spines[side].set_visible(False)

    mid = 0.5 * (part["start_sec"] + part["end_sec"])
    colours = [GREEN if int(p) == 1 else GREY for p in part["dev_selected_pred"]]
    ax_s.bar(
        mid,
        part["rule_score"],
        width=1.7,
        color=colours,
        edgecolor=PAPER,
        linewidth=0.3,
        zorder=2,
    )
    ax_s.axhline(tau, color=INK, ls="--", lw=1.0, zorder=3)
    ax_s.text(59.4, tau + 0.35, f"τ = {tau:.3f}°", ha="right", va="bottom", color=INK)
    ax_s.set_ylabel("Window amplitude  (°)")
    ax_s.set_xlabel("Time (s)")
    ax_s.set_ylim(0, max(18.0, float(part["rule_score"].max()) + 2.0))
    for side in ("top", "right"):
        ax_s.spines[side].set_visible(False)

    handles = [
        Line2D([0], [0], color=INK, lw=1.15, label="Yaw (EMOCA y)"),
        Patch(facecolor=ORANGE, alpha=0.55, label="Annotated shake"),
        Patch(facecolor=GREEN, alpha=0.35, label="Rule fires (3 s window)"),
        Line2D([0], [0], color=INK, ls="--", lw=1.0, label=f"DEV-selected threshold, {tau:.3f}°"),
    ]
    ax_y.legend(handles=handles, frameon=False, loc="upper left", ncol=2)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.16)
    fig.text(
        0.11,
        0.04,
        f"{CLIP.replace('_', ' ')}, TEST, watch RIGHT. One 0.4 s annotated shake at 48 s. "
        f"Axis y, τ = {tau:.3f}°. "
        f"The interval is the 15-clip TEST score, not this clip alone. Pose only.",
        color=MUTED,
    )
    save(fig, OUT)
    from PIL import Image

    jpg = OUT.with_suffix(".jpg")
    Image.open(OUT.with_suffix(".png")).convert("RGB").save(jpg, quality=92)
    print(f"wrote {jpg}")


if __name__ == "__main__":
    main()
