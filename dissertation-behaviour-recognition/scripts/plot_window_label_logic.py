#!/usr/bin/env python3
"""Nod-window protocol figure. Textbook example only. No TEST, no metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_protocol import example_protocol_labels  # noqa: E402

OUT_DEV = ROOT / "results" / "windowed_dev" / "window_label_logic.png"
OUT_TEST = ROOT / "results" / "windowed_test" / "window_label_logic.png"

INK = "#111111"
MUTED = "#333333"
NOD = "#c45c26"
POS_BG = "#d8efe3"
POS_LINE = "#1f6f4a"
NEG_BG = "#eeeeea"
NEG_LINE = "#7a7670"
PAPER = "#ffffff"
TRACK = "#f4f2ee"

# Layout in data units. Bars live on 0–12 s. Text has its own gutters.
X_LEFT = -4.6
X_RIGHT = 16.4
BAR_H = 0.70
ROW_GAP = 0.55
STATUS_X = 12.55
TIME_X = -0.55


def render(out: Path, title: str, event_word: str = "nod") -> None:
    labels = example_protocol_labels()
    n = len(labels)
    y_bar0 = 5.15
    y_bottom = y_bar0 - (n - 1) * (BAR_H + ROW_GAP)
    fig, ax = plt.subplots(figsize=(14.2, 9.2), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(X_LEFT, X_RIGHT)
    ax.set_ylim(y_bottom - 1.55, 9.8)
    ax.axis("off")

    ax.text(0.0, 9.35, title, fontsize=22, color=INK, fontweight="700")
    ax.text(
        0.0,
        8.88,
        "3 second slice     next slice starts 2 seconds later     1 second overlap",
        fontsize=13,
        color=MUTED,
    )

    # Nod names sit above the time ticks so they do not cover 2 s / 8 s.
    word = event_word.capitalize()
    ax.text(2.6, 8.42, f"{word} A   2.3 to 2.9 s", ha="center", fontsize=13, color=NOD, fontweight="700")
    ax.text(7.55, 8.42, f"{word} B   7.2 to 7.9 s", ha="center", fontsize=13, color=NOD, fontweight="700")

    ax.add_patch(Rectangle((0, 8.05), 12, 0.12, facecolor=TRACK, edgecolor=INK, linewidth=1.0))
    for t in range(0, 13, 2):
        ax.plot([t, t], [8.05, 8.22], color=INK, lw=1.1)
        ax.text(t, 7.72, f"{t} s", ha="center", fontsize=12, color=INK)

    ax.add_patch(Rectangle((2.3, 7.28), 0.6, 0.28, facecolor=NOD, edgecolor=INK, linewidth=0.8))
    ax.add_patch(Rectangle((7.2, 7.28), 0.7, 0.28, facecolor=NOD, edgecolor=INK, linewidth=0.8))

    ax.text(TIME_X, 6.18, "Window", ha="right", fontsize=12, color=MUTED)
    ax.text(6.0, 6.18, "Time on the clip", ha="center", fontsize=12, color=MUTED)
    ax.text(STATUS_X, 6.18, "Label", ha="left", fontsize=12, color=MUTED)

    for i, row in enumerate(labels):
        y = y_bar0 - i * (BAR_H + ROW_GAP)
        ws, we = float(row["start_sec"]), float(row["end_sec"])
        yes = int(row["label"]) == 1
        ax.add_patch(
            FancyBboxPatch(
                (ws, y),
                3.0,
                BAR_H,
                boxstyle="round,pad=0.01,rounding_size=0.06",
                facecolor=POS_BG if yes else NEG_BG,
                edgecolor=POS_LINE if yes else NEG_LINE,
                linewidth=1.5 if yes else 1.0,
            )
        )
        ax.text(
            TIME_X,
            y + BAR_H / 2,
            f"{ws:g} to {we:g} s",
            ha="right",
            va="center",
            fontsize=14,
            color=INK,
            fontweight="700",
        )
        ax.text(
            STATUS_X,
            y + BAR_H / 2,
            "YES" if yes else "no",
            ha="left",
            va="center",
            fontsize=16,
            color=POS_LINE if yes else INK,
            fontweight="700",
        )

    ax.text(
        0.0,
        y_bottom - 0.85,
        f"YES means the 3 s bar overlaps a {event_word}. First 12 s of a 60 s clip.",
        fontsize=13,
        color=INK,
    )
    ax.text(
        0.0,
        y_bottom - 1.25,
        "Full clip has 29 windows: 0 to 3 s, 2 to 5 s, up to 56 to 59 s. Diagram only.",
        fontsize=13,
        color=MUTED,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=PAPER, pad_inches=0.35)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    render(OUT_DEV, "Nod window DEV")


def write_test() -> None:
    render(OUT_TEST, "Nod window TEST")


def write_shake() -> None:
    out = ROOT / "results" / "windowed_shake" / "window_label_logic.png"
    render(out, "Shake window DEV", event_word="shake")


def write_shake_test() -> None:
    out = ROOT / "results" / "windowed_shake_test" / "window_label_logic.png"
    render(out, "Shake window TEST", event_word="shake")


if __name__ == "__main__":
    main()
    write_test()
