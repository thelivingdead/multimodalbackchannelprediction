"""Shared typography and sizes for the windowed publication figures.

Import this module before creating axes. ``apply_style`` runs on import so
every plate uses the same humanist sans, the same point sizes, and one of
two figure sizes: full text width or half width.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

INK = "#1d1d1f"
MUTED = "#5c5c63"
GREY = "#b0b0b4"
ORANGE = "#c46a2d"
BLUE = "#2c5f8a"
GREEN = "#2f6b45"
PAPER = "#ffffff"
PALE = "#f3f3f4"
WITHDRAWN = "#8a8a90"

# Thesis text width ~182 mm. Heights: one panel vs stacked panels.
SIZE_FULL = (7.16, 5.40)
SIZE_FULL_TALL = (7.16, 8.00)
SIZE_HALF = (3.50, 3.50)

_SANS = (
    "Avenir Next",
    "Avenir",
    "Helvetica Neue",
    "TeX Gyre Heros",
    "Nimbus Sans",
    "Liberation Sans",
    "DejaVu Sans",
)


def _first_installed(names: tuple[str, ...]) -> str:
    available = {item.name for item in font_manager.fontManager.ttflist}
    for name in names:
        if name in available:
            return name
    return "DejaVu Sans"


FONT_NAME = _first_installed(_SANS)


def apply_style() -> str:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_NAME, "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "axes.titleweight": "regular",
            "axes.labelweight": "regular",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GREY,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return FONT_NAME


apply_style()


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        facecolor=PAPER,
        bbox_inches="tight",
        pad_inches=0.28,
    )
    fig.savefig(
        stem.with_suffix(".svg"),
        facecolor=PAPER,
        bbox_inches="tight",
        pad_inches=0.28,
    )
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.png')}")


def forest(ax, rows: list[dict], *, title: str) -> None:
    """One forest panel. Points encode the value; bars are not used."""
    n = len(rows)
    ax.set_facecolor(PAPER)
    ax.axvline(0.5, color=INK, lw=1.0, ls="--", zorder=0)
    ax.text(0.5, n - 0.28, "chance  0.500", color=MUTED, ha="center", va="bottom")

    labels = []
    for i, row in enumerate(rows):
        y = n - 1 - i
        labels.append(row["name"])
        colour = row["colour"]
        ax.plot(
            [row["lo"], row["hi"]],
            [y, y],
            color=colour,
            lw=1.8,
            solid_capstyle="butt",
            zorder=2,
        )
        marker = "D" if row.get("locked") else "o"
        ax.plot(
            row["ba"],
            y,
            marker,
            color=colour,
            markersize=7,
            markeredgecolor=PAPER,
            markeredgewidth=0.5,
            zorder=3,
        )
        ax.text(
            0.84,
            y,
            f"{row['ba']:.3f}  [{row['lo']:.3f}, {row['hi']:.3f}]",
            va="center",
            ha="left",
            color=INK,
            clip_on=False,
        )

    ax.set_yticks(list(range(n - 1, -1, -1)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set_xlim(0.30, 1.16)
    ax.set_ylim(-0.55, n - 0.18)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("Balanced accuracy")
    ax.set_title(title, loc="left", pad=6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
