#!/usr/bin/env python3
"""Compare identity-fixed VideoMAE against the valid 3 s nod systems.

Uses saved official metrics. Does not plot withdrawn RGB rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_nod_model_inventory import bootstrap_row  # noqa: E402

FIXED = ROOT / "results" / "windowed_dev" / "videomae_identity_fixed"
WIN = ROOT / "results" / "windowed_nod"
INK = "#1d1d1f"
MUTED = "#5c5c63"
ORANGE = "#d97932"
BLUE = "#3178a8"
GREEN = "#1b7f4b"
PALE = "#cdcdd2"
PAPER = "#fffdf8"


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"STOP: missing {path}")
    return json.loads(path.read_text())


def main() -> None:
    base = load(WIN / "baselines_bacc" / "metrics.json")
    loco = load(WIN / "pose_cnn_loco_dev" / "metrics_dev.json")
    rows = [
        {
            "name": "Always yes or always no",
            "detail": "chance floor",
            "bacc": 0.5,
            "split": "any",
            "colour": PALE,
        },
        {
            "name": "Pitch rule",
            "detail": "one threshold, chosen on DEV",
            "bacc": float(
                base["metrics"]["TEST"]["dev_selected_window_rule"][
                    "balanced_accuracy"
                ]
            ),
            "pred": (
                WIN / "baselines_bacc" / "predictions.csv",
                "sample_id", "label", "dev_selected_pred", "split", "TEST",
            ),
            "split": "TEST",
            "colour": ORANGE,
        },
        {
            "name": "Pose CNN, leave one clip out",
            "detail": "human window labels, 14 clips",
            "bacc": float(loco["at_fixed_threshold_0.5"]["balanced_accuracy"]),
            "pred": (
                WIN / "pose_cnn_loco_dev" / "predictions_oof_dev.csv",
                "sample_id", "label", "pred_at_0.5", None, None,
            ),
            "split": "DEV, out of fold",
            "colour": BLUE,
        },
    ]
    for folder, name in (
        ("frozen_encoder", "VideoMAE frozen, identity fixed"),
        ("last_blocks_unfrozen", "VideoMAE last 2 blocks, identity fixed"),
    ):
        path = FIXED / folder / "metrics.json"
        if not path.exists():
            continue
        metrics = load(path)
        rows.append(
            {
                "name": name,
                "detail": "target person crops, threshold 0.5",
                "bacc": float(metrics["balanced_accuracy"]),
                "pred": (
                    FIXED / folder / "oof_predictions.csv",
                    "sample_id", "label", "pred_at_0.5", None, None,
                ),
                "split": "DEV, out of fold",
                "colour": GREEN,
            }
        )
    if len(rows) < 4:
        raise SystemExit("STOP: identity-fixed VideoMAE metrics are not on disk yet")

    for row in rows:
        spec = row.pop("pred", None)
        if spec is None:
            continue
        boot = bootstrap_row(*spec)
        if boot is None:
            raise SystemExit(f"STOP: no interval for {row['name']}")
        if abs(boot["point_estimate"] - row["bacc"]) > 0.002:
            raise SystemExit(
                f"STOP: {row['name']} stored {row['bacc']:.4f} but predictions "
                f"give {boot['point_estimate']:.4f}"
            )
        row["ci"] = boot["balanced_accuracy"]["ci_lower_95"]
        row["ci_high"] = boot["balanced_accuracy"]["ci_upper_95"]

    fig, ax = plt.subplots(figsize=(11.4, 5.2))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.axvline(0.5, color=INK, lw=1.4, ls="--")
    for index, row in enumerate(rows):
        y = -index
        ax.plot(row["bacc"], y, "o", color=row["colour"], markersize=10)
        if "ci" in row:
            ax.plot([row["ci"], row["ci_high"]], [y, y], color=row["colour"], lw=2)
        interval = (
            "" if "ci" not in row
            else f"  [{row['ci']:.3f}, {row['ci_high']:.3f}]"
        )
        ax.text(
            0.28, y,
            f"{row['name']}\n{row['detail']}  {row['split']}",
            va="center", ha="left", fontsize=9.2, color=INK,
        )
        ax.text(
            0.78, y, f"{row['bacc']:.3f}{interval}",
            va="center", ha="left", fontsize=9.2, color=INK,
        )
    ax.set_xlim(0.27, 1.05)
    ax.set_ylim(-len(rows) + 0.4, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("Balanced accuracy (floor 0.500, dashed)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.suptitle(
        "Identity-fixed VideoMAE against the valid 3 s nod systems",
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.text(
        0.01, 0.02,
        "Pitch rule is TEST. Pose CNN and VideoMAE are DEV out of fold. "
        "Withdrawn RGB rows that cropped the excluded person are not shown. "
        "Intervals are 95 percent clip bootstrap.",
        fontsize=8.4, color=MUTED,
    )
    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.02, right=0.99)
    out = FIXED / "model_comparison.png"
    fig.savefig(out, dpi=180, facecolor=PAPER)
    fig.savefig(out.with_suffix(".pdf"), facecolor=PAPER)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
