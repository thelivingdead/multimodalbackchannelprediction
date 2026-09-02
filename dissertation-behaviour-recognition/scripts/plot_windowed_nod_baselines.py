#!/usr/bin/env python3
"""Publication-style summary of 3 s nod TEST baselines."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "windowed_nod" / "baselines"
METRICS = RESULTS / "metrics.json"
OUT = RESULTS / "test_baselines"

INK = "#1d1d1f"
MUTED = "#5c5c63"
GREY = "#a8a8ad"
ORANGE = "#d97932"
BLUE = "#3178a8"
PAPER = "#fffdf8"


def main() -> None:
    data = json.loads(METRICS.read_text())
    test = data["metrics"]["TEST"]
    names = ["Always\nno", "Always\nyes", "Frozen 60 s\nrule", "DEV-selected\npitch rule"]
    keys = [
        "always_no",
        "always_yes",
        "frozen_60s_threshold_transfer",
        "dev_selected_window_rule",
    ]
    f1 = [float(test[key]["f1"]) for key in keys]
    chosen = test["dev_selected_window_rule"]
    cm = np.asarray(
        [[chosen["tn"], chosen["fp"]], [chosen["fn"], chosen["tp"]]], dtype=int
    )

    fig = plt.figure(figsize=(11.8, 5.5), facecolor=PAPER)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.34)
    ax = fig.add_subplot(grid[0])
    ax_cm = fig.add_subplot(grid[1])
    ax.set_facecolor(PAPER)
    ax_cm.set_facecolor(PAPER)

    colors = [GREY, GREY, ORANGE, BLUE]
    bars = ax.bar(np.arange(len(names)), f1, color=colors, width=0.68)
    ax.set_xticks(np.arange(len(names)), names, fontsize=11)
    ax.set_ylabel("Window F1", fontsize=12)
    ax.set_ylim(0.0, max(0.36, max(f1) + 0.07))
    ax.set_title("Held-out TEST baseline comparison", loc="left", fontsize=15, weight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, f1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=INK,
            weight="bold",
        )
    ax.text(
        0,
        -0.19,
        "435 TEST windows · 69 positive · 3 s duration · 2 s stride",
        transform=ax.transAxes,
        fontsize=10.5,
        color=MUTED,
    )

    image = ax_cm.imshow(cm, cmap="Blues", vmin=0)
    threshold = cm.max() / 2
    for (row, col), value in np.ndenumerate(cm):
        ax_cm.text(
            col,
            row,
            str(value),
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
            color="white" if value > threshold else INK,
        )
    ax_cm.set_xticks([0, 1], ["Predicted no", "Predicted nod"], fontsize=10.5)
    ax_cm.set_yticks([0, 1], ["Actual no", "Actual nod"], fontsize=10.5)
    ax_cm.set_title("DEV-selected pitch rule", fontsize=15, weight="bold")
    ax_cm.set_xlabel(
        f"P {chosen['precision']:.3f}   R {chosen['recall']:.3f}   "
        f"F1 {chosen['f1']:.3f}",
        fontsize=11,
        labelpad=11,
    )
    image.set_clim(0, max(1, int(cm.max())))

    fig.suptitle(
        "Head nod recognition · new sliding-window protocol",
        x=0.06,
        y=1.02,
        ha="left",
        fontsize=18,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.06,
        0.94,
        "Pitch axis x fixed for nod; amplitude threshold selected on human DEV only.",
        fontsize=11,
        color=MUTED,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}.png/.pdf")


if __name__ == "__main__":
    main()
