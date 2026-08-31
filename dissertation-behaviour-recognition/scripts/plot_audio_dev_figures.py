#!/usr/bin/env python3
"""GOLD DEV audio figures from saved CSVs. Does not score TEST. Does not train.

    cd dissertation-behaviour-recognition
    MPLCONFIGDIR=./.mplconfig OMP_NUM_THREADS=1 \\
        python scripts/plot_audio_dev_figures.py
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "figures" / "paper"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#0072B2"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#999999"
INK = "#111827"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, stem: str) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(PAPER / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"wrote figures/paper/{stem}.png")


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_metric_table(ax, rows: list[list[str]], title: str) -> None:
    ax.axis("off")
    ax.set_title(title, pad=10, fontsize=11)
    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.55)
    n_cols = len(rows[0])
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.6)
        if c == 0:
            cell.set_width(0.28)
            cell.set_text_props(ha="left")
        else:
            cell.set_width(0.72 / (n_cols - 1))
        if r == 0:
            cell.set_facecolor("#003D5B")
            cell.set_text_props(color="white", fontweight="bold", ha="center" if c else "left")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f4f6")



def _bars(ax, labels, f1, p, r, colors, xlabel: str, title: str) -> None:
    y = np.arange(len(labels))
    ax.barh(y, f1, color=colors, height=0.62, zorder=2, edgecolor="none")
    for i in range(len(labels)):
        ax.text(
            1.04,
            y[i],
            f"{f1[i]:.2f}      P {p[i]:.2f}     R {r[i]:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.62)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    despine(ax)


def fig_mfcc_dev() -> None:
    labels = ["Always-nod", "MFCC audio LR", "Frozen RGB + LR", "RGB+audio concat LR"]
    f1 = [0.750, 0.727, 0.857, 0.783]
    p = [0.600, 0.615, 0.750, 0.643]
    r = [1.000, 0.889, 1.000, 1.000]
    colors = [GREY, SKY, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(5.8, 3.2), layout="constrained")
    _bars(
        ax,
        labels,
        f1,
        p,
        r,
        colors,
        "DEV F1",
        "Nod audio  ·  GOLD DEV  n=15  ·  DEV-selected threshold",
    )
    save_fig(fig, "audio_dev_mfcc_f1")


def fig_hubert_dev() -> None:
    labels = ["Always-nod", "Frozen HuBERT + LR", "Frozen RGB + LR", "50/50 HuBERT+RGB"]
    f1 = [0.750, 0.889, 0.818, 0.800]
    p = [0.600, 0.889, 0.692, 0.727]
    r = [1.000, 0.889, 1.000, 0.889]
    colors = [GREY, BLUE, ORANGE, PURPLE]
    fig, ax = plt.subplots(figsize=(5.8, 3.2), layout="constrained")
    _bars(
        ax,
        labels,
        f1,
        p,
        r,
        colors,
        "DEV F1",
        "Nod HuBERT  ·  GOLD DEV  n=15  ·  threshold 0.5",
    )
    save_fig(fig, "audio_dev_hubert_f1")


def fig_audio_confusion() -> None:
    rows = [
        ["Method", "Thr", "TN", "FP", "FN", "TP", "P", "R", "F1"],
        ["Always-nod", "—", "0", "6", "0", "9", "0.60", "1.00", "0.75"],
        ["MFCC audio LR", "DEV 0.30", "1", "5", "1", "8", "0.62", "0.89", "0.73"],
        ["RGB LR (MFCC run)", "DEV 0.55", "3", "3", "0", "9", "0.75", "1.00", "0.86"],
        ["Concat fusion", "DEV 0.20", "1", "5", "0", "9", "0.64", "1.00", "0.78"],
        ["Frozen HuBERT + LR", "0.5", "5", "1", "1", "8", "0.89", "0.89", "0.89"],
        ["RGB LR (HuBERT run)", "0.5", "2", "4", "0", "9", "0.69", "1.00", "0.82"],
        ["50/50 HuBERT+RGB", "0.5", "3", "3", "1", "8", "0.73", "0.89", "0.80"],
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    draw_metric_table(
        ax,
        rows,
        "Nod GOLD DEV  n=15  ·  audio experiments (TEST not scored)",
    )
    save_fig(fig, "audio_dev_confusion")


def main() -> None:
    _style()
    PAPER.mkdir(parents=True, exist_ok=True)
    fig_mfcc_dev()
    fig_hubert_dev()
    fig_audio_confusion()
    print("done ->", PAPER)


if __name__ == "__main__":
    main()
