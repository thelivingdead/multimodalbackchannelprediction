#!/usr/bin/env python3
"""Publication-style summary of the 3 s nod baselines.

Balanced accuracy is the headline metric: at 12-16 percent window prevalence
F1 hides a collapse towards always-yes, while balanced accuracy shows it
against a fixed 0.5 floor. Intervals resample clips, not windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "windowed_nod" / "baselines_bacc"

INK = "#1d1d1f"
MUTED = "#5c5c63"
GREY = "#a8a8ad"
ORANGE = "#d97932"
BLUE = "#3178a8"
PAPER = "#fffdf8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    results = args.results
    data = json.loads((results / "metrics.json").read_text())
    sweep = pd.read_csv(results / "threshold_search_dev.csv")
    out = results / "test_baselines"

    keys = [
        "always_no",
        "always_yes",
        "frozen_60s_threshold_transfer",
        "dev_selected_window_rule",
    ]
    names = [
        "Always\nno",
        "Always\nyes",
        "Frozen 60 s\nrule",
        "DEV-selected\npitch rule",
    ]
    dev = [float(data["metrics"]["DEV"][key]["balanced_accuracy"]) for key in keys]
    test = [float(data["metrics"]["TEST"][key]["balanced_accuracy"]) for key in keys]
    test_f1 = [float(data["metrics"]["TEST"][key]["f1"]) for key in keys]
    names = [f"{name}\nF1 {value:.3f}" for name, value in zip(names, test_f1)]
    ci = {
        split: data["clip_bootstrap"][split]["dev_selected_window_rule"][
            "balanced_accuracy"
        ]
        for split in ("DEV", "TEST")
    }
    chosen = data["metrics"]["TEST"]["dev_selected_window_rule"]
    selected_threshold = float(data["dev_selected_window_threshold"])
    cm = np.asarray(
        [[chosen["tn"], chosen["fp"]], [chosen["fn"], chosen["tp"]]], dtype=int
    )

    fig = plt.figure(figsize=(15.4, 5.4), facecolor=PAPER)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.25, 0.9], wspace=0.32)
    ax = fig.add_subplot(grid[0])
    ax_sweep = fig.add_subplot(grid[1])
    ax_cm = fig.add_subplot(grid[2])
    for panel in (ax, ax_sweep, ax_cm):
        panel.set_facecolor(PAPER)
        panel.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, dev, width=width, color=GREY, label="DEV (selection)")
    ax.bar(x + width / 2, test, width=width, color=BLUE, label="TEST (held out)")
    for offset, values, split in ((-width / 2, dev, "DEV"), (width / 2, test, "TEST")):
        bounds = ci[split]
        lower = float(values[-1] - bounds["ci_lower_95"])
        upper = float(bounds["ci_upper_95"] - values[-1])
        ax.errorbar(
            x[-1] + offset,
            values[-1],
            yerr=[[max(lower, 0.0)], [max(upper, 0.0)]],
            fmt="none",
            ecolor=INK,
            elinewidth=1.4,
            capsize=5,
        )
    # Equal DEV/TEST values (the trivial baselines) get one shared label so the
    # two numbers do not overprint each other.
    for index, (position, dev_value, test_value) in enumerate(zip(x, dev, test)):
        last = index == len(x) - 1
        if abs(dev_value - test_value) < 0.004:
            pairs = [(position, max(dev_value, test_value), 0.0)]
        else:
            pairs = [
                (
                    position - width / 2,
                    dev_value,
                    ci["DEV"]["ci_upper_95"] - dev_value if last else 0.0,
                ),
                (
                    position + width / 2,
                    test_value,
                    ci["TEST"]["ci_upper_95"] - test_value if last else 0.0,
                ),
            ]
        for text_x, value, clearance in pairs:
            ax.text(
                text_x,
                value + max(clearance, 0.0) + 0.011,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=INK,
                weight="bold",
            )
    ax.axhline(
        0.5, color=ORANGE, linewidth=1.6, linestyle="--", label="Chance floor 0.5"
    )
    ax.set_xticks(x, names, fontsize=10.5)
    ax.set_ylabel("Balanced accuracy", fontsize=12)
    ax.set_ylim(0.40, 0.70)
    ax.set_title("Balanced accuracy vs chance", loc="left", fontsize=14, weight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=1)
    ax.text(
        0,
        -0.25,
        f"TEST 95% CI [{ci['TEST']['ci_lower_95']:.3f}, "
        f"{ci['TEST']['ci_upper_95']:.3f}] includes 0.500, so the rule is\n"
        "not distinguishable from chance on TEST.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=INK,
        weight="bold",
    )
    ax.text(
        0,
        -0.37,
        "F1 under each label is TEST F1 · 15 clips per split · 435 windows\n"
        "bars on the selected rule are 95% intervals from resampling clips",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )

    order = np.argsort(sweep["threshold"].to_numpy())
    thresholds = sweep["threshold"].to_numpy()[order]
    ax_sweep.plot(
        thresholds,
        sweep["balanced_accuracy"].to_numpy()[order],
        color=BLUE,
        linewidth=1.9,
        label="Balanced accuracy",
    )
    ax_sweep.plot(
        thresholds,
        sweep["f1"].to_numpy()[order],
        color=ORANGE,
        linewidth=1.5,
        linestyle="-.",
        label="F1",
    )
    ax_sweep.axhline(0.5, color=GREY, linewidth=1.2, linestyle="--")
    ax_sweep.axvline(selected_threshold, color=INK, linewidth=1.2, linestyle=":")
    ax_sweep.annotate(
        f"both criteria peak at {selected_threshold:.2f}$\\degree$",
        xy=(selected_threshold, 0.58),
        xytext=(0.42, 0.9),
        textcoords="axes fraction",
        fontsize=9.5,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1.0},
    )
    ax_sweep.set_xlabel(
        "Pitch amplitude threshold (degrees)\n"
        f"peak-to-peak amplitude scales with window length, so the 60 s\n"
        f"threshold of {float(data['frozen_60s_threshold']):.2f}$\\degree$ "
        f"cannot transfer to 3 s ({selected_threshold:.2f}$\\degree$)",
        fontsize=11,
    )
    ax_sweep.set_ylabel("DEV score", fontsize=12)
    ax_sweep.set_xlim(0, float(np.percentile(thresholds, 96)))
    ax_sweep.set_ylim(0.0, 0.75)
    ax_sweep.set_title("DEV threshold sweep", loc="left", fontsize=14, weight="bold")
    ax_sweep.grid(color="#dddddd", linewidth=0.7)
    ax_sweep.set_axisbelow(True)
    ax_sweep.legend(frameon=False, fontsize=10, loc="center right")

    image = ax_cm.imshow(cm, cmap="Blues", vmin=0)
    for (row, col), value in np.ndenumerate(cm):
        ax_cm.text(
            col,
            row,
            str(value),
            ha="center",
            va="center",
            fontsize=17,
            weight="bold",
            color="white" if value > cm.max() / 2 else INK,
        )
    ax_cm.set_xticks([0, 1], ["Predicted no", "Predicted nod"], fontsize=10)
    ax_cm.set_yticks([0, 1], ["Actual no", "Actual nod"], fontsize=10)
    ax_cm.set_title("TEST, selected rule", loc="left", fontsize=14, weight="bold")
    ax_cm.set_xlabel(
        f"balanced accuracy {chosen['balanced_accuracy']:.3f}\n"
        f"P {chosen['precision']:.3f}   R {chosen['recall']:.3f}   "
        f"F1 {chosen['f1']:.3f}",
        fontsize=10,
        labelpad=11,
    )
    ax_cm.spines[["left", "bottom"]].set_visible(False)
    image.set_clim(0, max(1, int(cm.max())))

    fig.suptitle(
        "Head nod recognition · 3 s sliding-window protocol",
        x=0.055,
        y=1.03,
        ha="left",
        fontsize=17,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.055,
        0.95,
        "Pitch axis fixed for nod. Amplitude threshold selected on human DEV by "
        "balanced accuracy, then applied once to TEST.",
        fontsize=10.5,
        color=MUTED,
    )
    results.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}.png/.pdf")


if __name__ == "__main__":
    main()
