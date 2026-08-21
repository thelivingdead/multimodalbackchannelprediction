#!/usr/bin/env python3
"""System pipeline diagram (training + inference) for the dissertation.

Two panels, research-paper style:
  (A) Offline: rule baseline -> pseudo-labels -> VideoMAE fine-tuning
  (B) Online:  webcam/video -> face crop -> fine-tuned model -> nod event

Pure matplotlib (Agg), 150 dpi, no seaborn. Output:
  figures/pipeline_diagram.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "pipeline_diagram.png"

BLUE = "#4e79a7"
ORANGE = "#f28e2b"
GREEN = "#59a14f"
GREY = "#6b7280"
RED = "#e15759"


def box(ax, x, y, w, h, text, fc=BLUE, fontsize=9.5, tc="white", ec="none"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.2, edgecolor=ec, facecolor=fc, alpha=0.96,
            mutation_aspect=1.0,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize, color=tc,
        linespacing=1.35,
    )


def arrow(ax, x1, y1, x2, y2, color=GREY, style="-|>", lw=1.6, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=13,
            linewidth=lw, color=color, linestyle=ls,
            shrinkA=2, shrinkB=2,
        )
    )


def main() -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(11.5, 7.8), gridspec_kw={"hspace": 0.10}
    )
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ---------------- Panel A: offline training ----------------
    ax = axes[0]
    ax.text(0.012, 0.945, "(A)  Offline: weakly-supervised training",
            fontsize=12.5, fontweight="bold", color="#111827")

    box(ax, 0.02, 0.60, 0.16, 0.22,
        "RealTalk videos\n+ EMOCA/FLAME\nhead pose (25 fps)", fc=GREY)
    box(ax, 0.25, 0.60, 0.15, 0.22,
        "Rule detector\n(pitch amplitude,\nDEV-tuned, frozen)", fc=BLUE)
    box(ax, 0.47, 0.60, 0.15, 0.22,
        "Pseudo-labels\n80–200 clips\n(no manual labels)", fc=BLUE)
    box(ax, 0.69, 0.60, 0.14, 0.22,
        "RGB windows\n16×224×224\nface crops", fc=GREY)
    box(ax, 0.69, 0.14, 0.29, 0.22,
        "VideoMAE-base — last 4 blocks fine-tuned\n"
        "(28.4M / 86.2M params, 1×RTX A4000)", fc=ORANGE)
    box(ax, 0.25, 0.14, 0.15, 0.22,
        "Gold DEV (15)\nearly stopping +\nthreshold 0.45", fc=GREEN)
    box(ax, 0.47, 0.14, 0.15, 0.22,
        "Gold TEST (15)\nscored once\nF1 = 0.818", fc=RED)

    arrow(ax, 0.18, 0.71, 0.25, 0.71)
    arrow(ax, 0.40, 0.71, 0.47, 0.71)
    arrow(ax, 0.62, 0.68, 0.69, 0.32, color=GREY)      # pseudo -> finetune
    arrow(ax, 0.76, 0.60, 0.76, 0.36, color=GREY)      # rgb -> finetune
    arrow(ax, 0.83, 0.25, 0.62, 0.25, color=GREY)      # model -> TEST
    arrow(ax, 0.69, 0.25, 0.62, 0.25, color=GREY)
    arrow(ax, 0.40, 0.25, 0.47, 0.25)
    arrow(ax, 0.325, 0.36, 0.76, 0.14, color=GREEN, ls="--")  # DEV selects

    ax.text(0.325, 0.385, "selects epoch + threshold\n(DEV F1 only)",
            fontsize=8, color=GREEN, ha="center")

    # ---------------- Panel B: online inference ----------------
    ax = axes[1]
    ax.text(0.012, 0.945, "(B)  Online: real-time nod detection (deployment path)",
            fontsize=12.5, fontweight="bold", color="#111827")

    box(ax, 0.02, 0.42, 0.15, 0.24, "Webcam / video\n(OpenCV\nVideoCapture)", fc=GREY)
    box(ax, 0.24, 0.42, 0.16, 0.24,
        "Face detection\n+ crop 224×224\n(Haar / MediaPipe)", fc=BLUE)
    box(ax, 0.47, 0.42, 0.15, 0.24,
        "Sliding window\n16 frames\n(~1–2 s context)", fc=BLUE)
    box(ax, 0.69, 0.42, 0.15, 0.24,
        "Fine-tuned\nVideoMAE head\n(best_model.pt)", fc=ORANGE)
    box(ax, 0.87, 0.42, 0.12, 0.24,
        "p ≥ 0.45 ?\nnod event\n→ UI / agent", fc=GREEN, fontsize=9)

    arrow(ax, 0.17, 0.54, 0.24, 0.54)
    arrow(ax, 0.40, 0.54, 0.47, 0.54)
    arrow(ax, 0.62, 0.54, 0.69, 0.54)
    arrow(ax, 0.84, 0.54, 0.87, 0.54)

    ax.text(0.5, 0.13,
            "No external API calls: torch + transformers + opencv run locally;  "
            "~50 ms per window on GPU.\nPose-only fallback path: EMOCA rotation "
            "→ frozen rule (no GPU required).",
            fontsize=8.5, color=GREY, ha="center", style="italic")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
