#!/usr/bin/env python3
"""Clear TEST sliding-window figures from the annotated events.

TEST only. No DEV overwrite, no metrics. Does not write an overview collage.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_protocol import EVENTS_TEST_CSV, WINDOWS_TEST_CSV  # noqa: E402

OUT_DIR = ROOT / "results" / "windowed_test" / "clip_windows"
EXAMPLE = ROOT / "results" / "windowed_test" / "gold_016_sliding_windows.png"

INK = "#1c1c1c"
MUTED = "#4a4a4a"
NOD = "#c45c26"
POS = "#1f6f4a"
POS_BG = "#d8efe3"
NEG_BG = "#eceae6"
LINE = "#cfc9bf"
PAPER = "#fffdf8"


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = pd.read_csv(EVENTS_TEST_CSV)
    win = pd.read_csv(WINDOWS_TEST_CSV)
    if (win["split"].astype(str).str.upper() != "TEST").any():
        raise SystemExit("STOP: nod_windows_test.csv contains a non-TEST row")
    nums = win["sample_id"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
    if ((nums < 16) | (nums > 30)).any():
        raise SystemExit("STOP: non-TEST sample leaked into nod_windows_test.csv")
    return ev, win


def plot_gold_016(events: pd.DataFrame, windows: pd.DataFrame) -> None:
    sid = "gold_016"
    ev = events.loc[events["sample_id"] == sid].sort_values("start_sec")
    ww = windows.loc[windows["sample_id"] == sid].sort_values("start_sec")
    n_pos = int(ww["label"].sum())

    fig = plt.figure(figsize=(13.2, 9.4), facecolor=PAPER)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 2.6], hspace=0.28)
    ax_t = fig.add_subplot(gs[0])
    ax_w = fig.add_subplot(gs[1])
    ax_t.set_facecolor(PAPER)
    ax_w.set_facecolor(PAPER)

    ax_t.set_xlim(0, 60)
    ax_t.set_ylim(0, 3.2)
    ax_t.axis("off")
    ax_t.set_title(
        "gold_016 — your nods on the 60 s clip",
        loc="left",
        fontsize=18,
        color=INK,
        pad=10,
        fontweight="600",
    )
    ax_t.text(
        0,
        2.85,
        f"{len(ev)} nods marked   ·   {n_pos} of {len(ww)} three-second windows contain a nod",
        fontsize=13,
        color=MUTED,
    )
    ax_t.plot([0, 60], [1.15, 1.15], color=INK, lw=1.6)
    for t in range(0, 61, 5):
        h = 0.22 if t % 10 == 0 else 0.12
        ax_t.plot([t, t], [1.15, 1.15 + h], color=INK, lw=1.0)
        if t % 10 == 0:
            ax_t.text(t, 0.72, f"{t} s", ha="center", fontsize=12, color=INK)

    for i, (_, r) in enumerate(ev.iterrows(), start=1):
        s, e = float(r.start_sec), float(r.end_sec)
        ax_t.axvspan(s, max(e, s + 0.35), ymin=0.38, ymax=0.62, color=NOD, alpha=0.85)
        ax_t.annotate(
            f"Nod {i}\n{s:.1f}–{e:.1f} s",
            xy=((s + e) / 2, 1.42),
            xytext=((s + e) / 2, 2.25),
            ha="center",
            va="bottom",
            fontsize=12,
            color=NOD,
            fontweight="600",
            arrowprops=dict(arrowstyle="-", color=NOD, lw=1.0),
        )

    ax_w.axis("off")
    ax_w.set_xlim(0, 10)
    ax_w.set_ylim(0, 8)
    ax_w.set_title(
        "3-second windows  ·  next window starts 2 seconds later",
        loc="left",
        fontsize=16,
        color=INK,
        pad=8,
        fontweight="600",
    )
    ax_w.text(
        0,
        7.45,
        "YES means that 3 s slice overlaps one of the nods above.",
        fontsize=13,
        color=MUTED,
    )

    cols = 5
    box_w, box_h = 1.78, 0.95
    gap_x, gap_y = 0.16, 0.18
    items = list(ww.itertuples(index=False))
    for idx, r in enumerate(items):
        col = idx % cols
        row = idx // cols
        x = 0.05 + col * (box_w + gap_x)
        y = 6.15 - row * (box_h + gap_y)
        yes = int(r.label) == 1
        ax_w.add_patch(
            FancyBboxPatch(
                (x, y),
                box_w,
                box_h,
                boxstyle="round,pad=0.012,rounding_size=0.08",
                facecolor=POS_BG if yes else NEG_BG,
                edgecolor=POS if yes else LINE,
                linewidth=1.3 if yes else 0.8,
            )
        )
        ax_w.text(
            x + box_w / 2,
            y + 0.58,
            f"{r.start_sec:g}–{r.end_sec:g} s",
            ha="center",
            va="center",
            fontsize=13,
            color=INK,
            fontweight="600",
        )
        ax_w.text(
            x + box_w / 2,
            y + 0.24,
            "YES  nod" if yes else "no nod",
            ha="center",
            va="center",
            fontsize=12,
            color=POS if yes else MUTED,
        )

    EXAMPLE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXAMPLE, dpi=170, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def plot_each_clip(events: pd.DataFrame, windows: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(16, 31):
        sid = f"gold_{i:03d}"
        ev = events.loc[events["sample_id"] == sid].sort_values("start_sec")
        ww = windows.loc[windows["sample_id"] == sid].sort_values("start_sec")
        n_pos = int(ww["label"].sum())
        fig, ax = plt.subplots(figsize=(13.0, 3.6), facecolor=PAPER)
        ax.set_facecolor(PAPER)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 4.0)
        ax.axis("off")
        ax.set_title(
            f"{sid}   ·   {len(ev)} nod{'s' if len(ev) != 1 else ''}   ·   "
            f"{n_pos} of {len(ww)} windows contain a nod",
            loc="left",
            fontsize=16,
            color=INK,
            pad=8,
            fontweight="600",
        )
        ax.plot([0, 60], [1.35, 1.35], color=INK, lw=1.5)
        for t in range(0, 61, 10):
            ax.plot([t, t], [1.35, 1.55], color=INK, lw=1.0)
            ax.text(t, 0.95, f"{t} s", ha="center", fontsize=12, color=INK)

        for j, (_, r) in enumerate(ev.iterrows(), start=1):
            s, e = float(r.start_sec), float(r.end_sec)
            ax.axvspan(s, max(e, s + 0.35), ymin=0.36, ymax=0.52, color=NOD, alpha=0.9)
            y_lab = 2.35 if j % 2 == 0 else 2.05
            ax.text(
                (s + e) / 2,
                y_lab,
                f"Nod {j}\n{s:.1f}–{e:.1f} s",
                ha="center",
                fontsize=11,
                color=NOD,
                fontweight="600",
            )

        for _, r in ww.iterrows():
            s, e = float(r.start_sec), float(r.end_sec)
            yes = int(r.label) == 1
            ax.add_patch(
                Rectangle(
                    (s + 0.06, 2.85),
                    e - s - 0.12,
                    0.7,
                    facecolor=POS_BG if yes else NEG_BG,
                    edgecolor=POS if yes else LINE,
                    linewidth=0.9,
                )
            )
        ax.text(
            0,
            3.68,
            "Each block is one 3 s window (step 2 s). Green edge = YES.",
            fontsize=12,
            color=MUTED,
        )
        fig.savefig(OUT_DIR / f"{sid}.png", dpi=170, bbox_inches="tight", facecolor=PAPER)
        plt.close(fig)


def _write_protocol_figure() -> None:
    spec = importlib.util.spec_from_file_location(
        "plot_window_label_logic",
        ROOT / "scripts" / "plot_window_label_logic.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.write_test()


def main() -> None:
    events, windows = _load()
    plot_gold_016(events, windows)
    plot_each_clip(events, windows)
    _write_protocol_figure()
    print(f"wrote {EXAMPLE.relative_to(ROOT)}")
    print(f"wrote {OUT_DIR.relative_to(ROOT)}/gold_016.png … gold_030.png")
    print("wrote results/windowed_test/window_label_logic.png")


if __name__ == "__main__":
    main()
