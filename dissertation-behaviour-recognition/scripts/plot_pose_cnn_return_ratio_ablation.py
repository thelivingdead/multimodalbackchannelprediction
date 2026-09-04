#!/usr/bin/env python3
"""Compare locked pose CNN vs pose CNN + return-ratio on DEV LOCO.

Reads existing metrics. Does not train. Does not load TEST.

    python3 scripts/plot_pose_cnn_return_ratio_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_split_leakage import assert_unlocked_out_dir  # noqa: E402
from src.paper_figure_style import (  # noqa: E402
    BLUE,
    GREY,
    INK,
    MUTED,
    PAPER,
    SIZE_FULL,
    save,
)

ORIGINAL = ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev" / "metrics_dev.json"
AUGMENTED = (
    ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev_return_ratio" / "metrics_dev.json"
)
OUT = ROOT / "results" / "windowed_nod" / "pose_cnn_return_ratio_ablation"


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"STOP: missing {path}")
    payload = json.loads(path.read_text())
    if payload.get("test_scored"):
        raise SystemExit(f"STOP: {path} scored TEST")
    return payload


def confusion_panel(ax, metrics: dict, title: str) -> None:
    grid = np.array(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        dtype=int,
    )
    ax.set_facecolor(PAPER)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_aspect("equal")
    row_n = grid.sum(axis=1, keepdims=True)
    row_pct = np.divide(grid, np.maximum(row_n, 1))
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=GREY, lw=1.0))
            ax.text(j, i - 0.10, str(grid[i, j]), ha="center", va="center", color=INK)
            ax.text(
                j,
                i + 0.22,
                f"{100 * row_pct[i, j]:.0f}% of row",
                ha="center",
                va="center",
                color=MUTED,
            )
    ax.set_xticks([0, 1], ["Pred. no-nod", "Pred. nod"])
    ax.set_yticks([0, 1], ["Actual no-nod", "Actual nod"])
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title)


def figure_ba(original: dict, augmented: dict, stem: Path) -> None:
    rows = [
        ("Pose CNN", original, GREY),
        ("Pose CNN\n+ return ratio", augmented, BLUE),
    ]
    fig, ax = plt.subplots(figsize=SIZE_FULL, facecolor=PAPER)
    xs = np.arange(len(rows))
    ba, lo, hi, colours = [], [], [], []
    for _label, block, colour in rows:
        metrics = block["at_fixed_threshold_0.5"]
        boot = block["clip_bootstrap_at_0.5"]["balanced_accuracy"]
        ba.append(float(metrics["balanced_accuracy"]))
        lo.append(float(boot["ci_lower_95"]))
        hi.append(float(boot["ci_upper_95"]))
        colours.append(colour)
    ax.axhline(0.5, color=INK, ls="--", lw=1.0, zorder=0)
    ax.bar(xs, ba, color=colours, width=0.45, edgecolor=PAPER, zorder=2)
    ax.errorbar(
        xs,
        ba,
        yerr=[np.array(ba) - np.array(lo), np.array(hi) - np.array(ba)],
        fmt="none",
        ecolor=INK,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(xs, [label for label, _b, _c in rows])
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 0.85)
    ax.set_title("DEV leave-one-clip-out. Same folds. Threshold 0.5.")
    ax.text(1.45, 0.51, "chance  0.500", color=MUTED, ha="right", va="bottom")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.20)
    fig.text(
        0.12,
        0.04,
        "Original CNN is locked. Return-ratio is a 7th channel, broadcast over the 3 s window.\n"
        "Zero crossings were not added. Whiskers are 95% clip-level intervals. TEST was not scored.",
        color=MUTED,
    )
    save(fig, stem)


def figure_confusion(original: dict, augmented: dict, stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=SIZE_FULL, facecolor=PAPER)
    confusion_panel(axes[0], original["at_fixed_threshold_0.5"], "Pose CNN")
    confusion_panel(
        axes[1],
        augmented["at_fixed_threshold_0.5"],
        "Pose CNN + return ratio",
    )
    fig.suptitle("Out-of-fold confusion on DEV. Threshold 0.5.")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.12, wspace=0.35)
    fig.text(
        0.08,
        0.04,
        "DEV only, 15 clips. TEST was not read.",
        color=MUTED,
    )
    save(fig, stem)


def main() -> None:
    out = assert_unlocked_out_dir(OUT)
    original = load_metrics(ORIGINAL)
    augmented = load_metrics(AUGMENTED)
    if not original.get("development_only") or not augmented.get("development_only"):
        raise SystemExit("STOP: both runs must be DEV-only")
    if augmented.get("zero_crossings_used"):
        raise SystemExit("STOP: this ablation must not use zero crossings")
    out.mkdir(parents=True, exist_ok=True)
    figure_ba(original, augmented, out / "figure_ba_comparison")
    figure_confusion(original, augmented, out / "figure_confusion")
    o = original["at_fixed_threshold_0.5"]
    a = augmented["at_fixed_threshold_0.5"]
    o_ci = original["clip_bootstrap_at_0.5"]["balanced_accuracy"]
    a_ci = augmented["clip_bootstrap_at_0.5"]["balanced_accuracy"]
    print(
        "original  BA "
        f"{o['balanced_accuracy']:.3f} [{o_ci['ci_lower_95']:.3f}, {o_ci['ci_upper_95']:.3f}]  "
        f"F1 {o['f1']:.3f}  P {o['precision']:.3f}  R {o['recall']:.3f}  "
        f"PR AUC {original['pr_auc_out_of_fold']:.3f}  "
        f"TP{o['tp']} FP{o['fp']} TN{o['tn']} FN{o['fn']}"
    )
    print(
        "+ return  BA "
        f"{a['balanced_accuracy']:.3f} [{a_ci['ci_lower_95']:.3f}, {a_ci['ci_upper_95']:.3f}]  "
        f"F1 {a['f1']:.3f}  P {a['precision']:.3f}  R {a['recall']:.3f}  "
        f"PR AUC {augmented['pr_auc_out_of_fold']:.3f}  "
        f"TP{a['tp']} FP{a['fp']} TN{a['tn']} FN{a['fn']}"
    )
    delta = a["balanced_accuracy"] - o["balanced_accuracy"]
    print(f"delta BA {delta:+.3f}  (DEV LOCO, not TEST)")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
