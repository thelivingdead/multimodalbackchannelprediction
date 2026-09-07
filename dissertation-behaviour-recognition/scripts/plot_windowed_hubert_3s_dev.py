#!/usr/bin/env python3
"""DEV figures: frozen HuBERT confusion and MFCC vs HuBERT BA. TEST not read."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paper_figure_style import BLUE, INK, PAPER, SIZE_FULL, save  # noqa: E402
from src.windowed_baselines import load_windows  # noqa: E402

OUT = ROOT / "results" / "windowed_dev" / "audio_3s_hubert"
MFCC = ROOT / "results" / "windowed_dev" / "audio_3s" / "metrics.json"
WINDOWS = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
MFCC_FALLBACK = {
    "name": "MFCC + LR",
    "ba": 0.5339927696324563,
    "lo": 0.4702353861629596,
    "hi": 0.6012570264677177,
}


def figure_confusion(out_dir: Path) -> None:
    m = json.loads((out_dir / "metrics.json").read_text())["at_fixed_threshold_0.5"]
    mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=SIZE_FULL, facecolor=PAPER)
    ax.imshow(mat, cmap="Greys")
    for i in range(2):
        for j in range(2):
            n = int(mat[i, j])
            pct = 100.0 * n / mat.sum()
            ax.text(j, i, f"{n}\n({pct:.1f}%)", ha="center", va="center", color=INK)
    ax.set_xticks([0, 1], ["Pred no", "Pred nod"])
    ax.set_yticks([0, 1], ["True no", "True nod"])
    ax.set_title("DEV frozen HuBERT confusion. TEST not scored.")
    fig.tight_layout()
    save(fig, out_dir / "figures" / "figure_hubert_confusion")


def figure_mfcc_vs_hubert(out_dir: Path) -> None:
    hubert = json.loads((out_dir / "metrics.json").read_text())
    rows = [dict(MFCC_FALLBACK)]
    if MFCC.exists():
        audio = json.loads(MFCC.read_text())
        rows[0] = {
            "name": "MFCC + LR",
            "ba": audio["at_fixed_threshold_0.5"]["balanced_accuracy"],
            "lo": audio["clip_bootstrap_at_0.5"]["balanced_accuracy"]["ci_lower_95"],
            "hi": audio["clip_bootstrap_at_0.5"]["balanced_accuracy"]["ci_upper_95"],
        }
    boot = hubert["clip_bootstrap_at_0.5"]["balanced_accuracy"]
    rows.append(
        {
            "name": "HuBERT frozen + LR",
            "ba": hubert["at_fixed_threshold_0.5"]["balanced_accuracy"],
            "lo": boot["ci_lower_95"],
            "hi": boot["ci_upper_95"],
        }
    )
    fig, ax = plt.subplots(figsize=SIZE_FULL, facecolor=PAPER)
    ys = np.arange(len(rows))
    ax.axvline(0.5, color=INK, ls="--", lw=1.0)
    ax.barh(ys, [r["ba"] for r in rows], color=BLUE, height=0.55)
    ax.errorbar(
        [r["ba"] for r in rows],
        ys,
        xerr=[[r["ba"] - r["lo"] for r in rows], [r["hi"] - r["ba"] for r in rows]],
        fmt="none",
        ecolor=INK,
        capsize=2,
    )
    ax.set_yticks(ys, [r["name"] for r in rows])
    ax.set_xlabel("Balanced accuracy")
    ax.set_xlim(0.0, 0.90)
    ax.set_title("DEV MFCC vs frozen HuBERT. Chance 0.500. TEST not scored.")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, out_dir / "figures" / "figure_mfcc_vs_hubert")


def write_figures(out_dir: Path | None = None) -> None:
    out_dir = Path(out_dir or OUT)
    if set(load_windows(WINDOWS, "DEV", DEV_IDS)["sample_id"]) & TEST_IDS:
        raise SystemExit("STOP: TEST id in figure windows")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    if (out_dir / "metrics.json").exists():
        figure_confusion(out_dir)
        figure_mfcc_vs_hubert(out_dir)
    print(f"wrote figures under {out_dir / 'figures'}")


if __name__ == "__main__":
    write_figures()
