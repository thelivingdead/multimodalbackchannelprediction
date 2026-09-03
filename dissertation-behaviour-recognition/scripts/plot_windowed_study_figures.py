#!/usr/bin/env python3
"""Study-design and results figures for windowed nod and shake.

Reads stored metrics only. Does not score TEST. Does not overwrite locked
3 s identity-fixed metric directories.

    python3 scripts/plot_windowed_study_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paper_figure_style import (  # noqa: E402
    BLUE,
    GREEN,
    GREY,
    INK,
    MUTED,
    ORANGE,
    PALE,
    PAPER,
    SIZE_FULL,
    SIZE_FULL_TALL,
    WITHDRAWN,
    forest,
    save,
)

OUT = ROOT / "results" / "windowed_dev" / "final_figures"


def loadj(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def box(ax, x, y, w, h, title, body="", *, fc=PALE, ec=INK, title_c=INK):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            linewidth=0.9,
            facecolor=fc,
            edgecolor=ec,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(
        x + 0.012,
        y + h - 0.016,
        title,
        transform=ax.transAxes,
        color=title_c,
        fontweight="bold",
        va="top",
        ha="left",
        parse_math=False,
    )
    for i, line in enumerate(body.split("\n") if body else []):
        ax.text(
            x + 0.012,
            y + h - 0.040 - i * 0.022,
            line,
            transform=ax.transAxes,
            color=MUTED,
            va="top",
            ha="left",
            parse_math=False,
        )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            color=INK,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
        )
    )


def figure_study_design(stem: Path) -> None:
    """Gold forks to DEV and TEST. There is no arrow from DEV into TEST."""
    fig = plt.figure(figsize=SIZE_FULL, facecolor=PAPER)
    ax = fig.add_axes((0.03, 0.07, 0.94, 0.88))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Windowed nod and shake recognition: study design", loc="left", pad=4)

    ax.text(0.01, 0.985, "A  Data and splits", fontweight="bold", va="top")
    box(
        ax, 0.35, 0.86, 0.30, 0.10,
        "Gold clips",
        "30 RealTalk conversations\nNod and shake events, watch side",
    )
    box(
        ax, 0.01, 0.68, 0.46, 0.13,
        "DEV, gold 001 to gold 015",
        "Development and leave-one-clip-out.\nThresholds and the 1.5 s run live here.\nDEV-only scripts never read TEST.",
        fc="#e8eef4",
        ec=BLUE,
        title_c=BLUE,
    )
    box(
        ax, 0.53, 0.68, 0.46, 0.13,
        "TEST, gold 016 to gold 030",
        "Locked confirmation only.\nScored after DEV is closed.\nNot used for the 1.5 s run.",
        fc="#f4eee8",
        ec=ORANGE,
        title_c=ORANGE,
    )
    arrow(ax, 0.44, 0.86, 0.24, 0.81)
    arrow(ax, 0.56, 0.86, 0.76, 0.81)

    ax.text(0.01, 0.645, "B  Windowing and representations", fontweight="bold", va="top")
    box(
        ax, 0.01, 0.46, 0.30, 0.16,
        "3 s windows, 2 s stride",
        "Both splits: 15 x 29 = 435.\nNod DEV 52 positive.\nShake DEV 39 positive.",
    )
    box(
        ax, 0.35, 0.46, 0.30, 0.16,
        "1.5 s windows, DEV only",
        "Nod DEV: 885 windows.\n16 RGB frames at 10.7 / s.\nNo TEST file is written.",
        fc="#eef4ee",
        ec=GREEN,
        title_c=GREEN,
    )
    box(
        ax, 0.69, 0.46, 0.30, 0.16,
        "Two representations",
        "Pose: EMOCA xyz\n(pitch 0, yaw 1).\nRGB: annotator-side face.",
    )
    arrow(ax, 0.24, 0.68, 0.16, 0.62)
    arrow(ax, 0.76, 0.68, 0.16, 0.62)
    arrow(ax, 0.24, 0.68, 0.50, 0.62)

    ax.text(0.01, 0.425, "C  Systems", fontweight="bold", va="top")
    box(
        ax, 0.01, 0.29, 0.30, 0.11,
        "Pose systems",
        "Amplitude rule (DEV threshold).\n1-D CNN, leave-one-clip-out.\nMIL on 80 weak TRAIN bags.",
        fc="#e8eef4",
        ec=BLUE,
        title_c=BLUE,
    )
    box(
        ax, 0.35, 0.29, 0.30, 0.11,
        "RGB, identity-fixed",
        "VideoMAE last two blocks.\n3 s 0.562; 1.5 s 0.528.\nTrain-only flip. Seed 42.",
        fc="#eef4ee",
        ec=GREEN,
        title_c=GREEN,
    )
    box(
        ax, 0.69, 0.29, 0.30, 0.11,
        "Withdrawn RGB",
        "Largest-face Haar followed\nthe excluded speaker on 51%\nof DEV nod windows.",
        fc="#f2f2f3",
        ec=WITHDRAWN,
        title_c=WITHDRAWN,
    )
    arrow(ax, 0.16, 0.46, 0.16, 0.40)
    arrow(ax, 0.50, 0.46, 0.50, 0.40)
    arrow(ax, 0.84, 0.46, 0.84, 0.40)

    ax.text(0.01, 0.255, "D  Evaluation protocol", fontweight="bold", va="top")
    box(
        ax, 0.01, 0.04, 0.48, 0.19,
        "Headline metric",
        "Balanced accuracy, chance = 0.500.\nF1 was dropped: the 60 s nod rule had\nTEST F1 0.667 and TEST balanced\naccuracy 0.450.",
    )
    box(
        ax, 0.52, 0.04, 0.47, 0.19,
        "Uncertainty",
        "95% intervals resample 15 clips,\n2000 times. An interval that includes\n0.500 is not distinguished from chance.\nDEV selection and TEST confirmation\nare separate. No path from DEV to TEST.",
    )

    fig.text(
        0.03,
        0.012,
        "Gold forks to DEV and TEST. There is no arrow from DEV into TEST.",
        color=MUTED,
    )
    save(fig, stem)


def _ba_ci_rule(metrics: dict, split: str) -> tuple[float, float, float]:
    block = metrics["metrics"][split]["dev_selected_window_rule"]
    boot = metrics["clip_bootstrap"][split]["dev_selected_window_rule"]["balanced_accuracy"]
    return (
        float(block["balanced_accuracy"]),
        float(boot["ci_lower_95"]),
        float(boot["ci_upper_95"]),
    )


def _ba_ci_loco(metrics: dict) -> tuple[float, float, float]:
    ba = float(metrics["at_fixed_threshold_0.5"]["balanced_accuracy"])
    boot = metrics["clip_bootstrap_at_0.5"]["balanced_accuracy"]
    return ba, float(boot["ci_lower_95"]), float(boot["ci_upper_95"])


def _ba_ci_fixed(metrics: dict) -> tuple[float, float, float]:
    ba = float(metrics["balanced_accuracy"])
    boot = (metrics.get("clip_bootstrap") or {}).get("balanced_accuracy") or {}
    return ba, float(boot.get("ci_lower_95", ba)), float(boot.get("ci_upper_95", ba))


def figure_study_results(stem: Path) -> None:
    nod_rule = loadj(ROOT / "results" / "windowed_nod" / "baselines_bacc" / "metrics.json")
    shake_rule = loadj(ROOT / "results" / "windowed_shake" / "baselines_bacc" / "metrics.json")
    nod_cnn = loadj(ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev" / "metrics_dev.json")
    shake_cnn = loadj(ROOT / "results" / "windowed_shake" / "pose_cnn_loco_dev" / "metrics_dev.json")
    nod_mil = loadj(ROOT / "results" / "windowed_nod" / "pose_mil_pseudo80_trainsel" / "metrics_dev.json")
    frozen = loadj(ROOT / "results" / "windowed_dev" / "videomae_identity_fixed" / "frozen_encoder" / "metrics.json")
    last2 = loadj(ROOT / "results" / "windowed_dev" / "videomae_identity_fixed" / "last_blocks_unfrozen" / "metrics.json")
    one = loadj(ROOT / "results" / "windowed_dev" / "videomae_identity_fixed_1p5s" / "last_blocks_unfrozen" / "metrics.json")
    nod_old = loadj(ROOT / "results" / "windowed_nod" / "videomae_loco_dev" / "metrics_dev.json")
    shake_old = loadj(ROOT / "results" / "windowed_shake" / "videomae_loco_dev" / "metrics_dev.json")

    nod = []
    if nod_rule:
        ba, lo, hi = _ba_ci_rule(nod_rule, "TEST")
        nod.append({"name": "Pitch rule, locked TEST", "ba": ba, "lo": lo, "hi": hi, "colour": ORANGE, "locked": True})
    if nod_cnn:
        ba, lo, hi = _ba_ci_loco(nod_cnn)
        nod.append({"name": "Pose CNN, DEV leave-one-clip-out", "ba": ba, "lo": lo, "hi": hi, "colour": BLUE})
    if nod_mil:
        ba = float(nod_mil["dev_window"]["balanced_accuracy"])
        boot = nod_mil.get("dev_clip_bootstrap", {}).get("balanced_accuracy", {})
        nod.append({
            "name": "Pose MIL, TRAIN-selected, DEV",
            "ba": ba,
            "lo": float(boot.get("ci_lower_95", ba)),
            "hi": float(boot.get("ci_upper_95", ba)),
            "colour": BLUE,
        })
    if frozen:
        ba, lo, hi = _ba_ci_fixed(frozen)
        nod.append({"name": "VideoMAE frozen, identity-fixed, 3 s", "ba": ba, "lo": lo, "hi": hi, "colour": GREEN})
    if last2:
        ba, lo, hi = _ba_ci_fixed(last2)
        nod.append({"name": "VideoMAE last two blocks, 3 s", "ba": ba, "lo": lo, "hi": hi, "colour": GREEN})
    if one:
        ba, lo, hi = _ba_ci_fixed(one)
        nod.append({"name": "VideoMAE last two blocks, 1.5 s", "ba": ba, "lo": lo, "hi": hi, "colour": GREEN})
    if nod_old:
        ba, lo, hi = _ba_ci_loco(nod_old)
        nod.append({"name": "VideoMAE largest-face, withdrawn", "ba": ba, "lo": lo, "hi": hi, "colour": WITHDRAWN})

    shake = []
    if shake_rule:
        ba, lo, hi = _ba_ci_rule(shake_rule, "TEST")
        shake.append({"name": "Yaw rule, locked TEST", "ba": ba, "lo": lo, "hi": hi, "colour": ORANGE, "locked": True})
        ba, lo, hi = _ba_ci_rule(shake_rule, "DEV")
        shake.append({"name": "Yaw rule, DEV (selection only)", "ba": ba, "lo": lo, "hi": hi, "colour": GREY})
    if shake_cnn:
        ba, lo, hi = _ba_ci_loco(shake_cnn)
        shake.append({"name": "Pose CNN, DEV leave-one-clip-out", "ba": ba, "lo": lo, "hi": hi, "colour": BLUE})
    if shake_old:
        ba, lo, hi = _ba_ci_loco(shake_old)
        shake.append({"name": "VideoMAE largest-face, withdrawn", "ba": ba, "lo": lo, "hi": hi, "colour": WITHDRAWN})

    fig, axes = plt.subplots(
        2,
        1,
        figsize=SIZE_FULL_TALL,
        facecolor=PAPER,
        gridspec_kw={"height_ratios": [max(len(shake), 1), max(len(nod), 1)], "hspace": 0.42},
    )
    forest(axes[0], shake, title="A   Shake. Yaw clears chance on locked TEST.")
    forest(axes[1], nod, title="B   Nod. Pitch does not.")
    fig.suptitle("Same protocol, opposite outcomes", x=0.36, ha="left")
    fig.subplots_adjust(left=0.36, right=0.97, top=0.90, bottom=0.14)
    fig.text(
        0.36,
        0.03,
        "Shake yaw rule TEST 0.654 [0.525, 0.794] and pose CNN 0.606 [0.519, 0.680]\n"
        "both exclude 0.500. Nod pitch rule TEST 0.549 [0.480, 0.619] includes it.\n"
        "Diamonds are locked TEST. Circles are DEV. 95% clip-level intervals, 15 clips.",
        color=MUTED,
        va="bottom",
    )
    save(fig, stem)


def main() -> None:
    figure_study_design(OUT / "figure_study_design")
    figure_study_results(OUT / "figure_study_results")


if __name__ == "__main__":
    main()
