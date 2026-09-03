#!/usr/bin/env python3
"""Compare identity-fixed VideoMAE against the valid 3 s nod systems.

Additions over v1:
- Paired clip bootstrap for frozen vs unfrozen difference.
- Visual separator between TEST and DEV rows.
Uses saved official metrics. Does not plot withdrawn RGB rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
RNG = np.random.default_rng(42)


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"STOP: missing {path}")
    return json.loads(path.read_text())


def paired_bootstrap_difference(
    csv_a: Path,
    csv_b: Path,
    pred_col_a: str = "pred_at_0.5",
    pred_col_b: str = "pred_at_0.5",
    label_col: str = "label",
    clip_col: str = "sample_id",
    n_boot: int = 2000,
) -> dict:
    """Bootstrap the paired difference in balanced accuracy (B minus A).

    Resamples whole clips (same clips for both systems within each resample)
    so the within-clip correlation structure is preserved.
    Returns point estimate and 95 percent percentile interval.
    """
    def _bacc(labels, preds):
        tp = np.sum((labels == 1) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.5 * (sens + spec)

    # extract clip id from sample_id  e.g. "gold_001_w003" -> "gold_001"
    def _clip(s):
        return "_".join(str(s).split("_")[:2])

    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)
    df_a["_clip"] = df_a[clip_col].apply(_clip)
    df_b["_clip"] = df_b[clip_col].apply(_clip)

    clips = np.array(sorted(set(df_a["_clip"])))

    point_a = _bacc(df_a[label_col].values, df_a[pred_col_a].values)
    point_b = _bacc(df_b[label_col].values, df_b[pred_col_b].values)
    point_diff = point_b - point_a

    diffs = []
    for _ in range(n_boot):
        sampled = RNG.choice(clips, size=len(clips), replace=True)
        mask_a = df_a["_clip"].isin(sampled)
        mask_b = df_b["_clip"].isin(sampled)
        ba = _bacc(df_a.loc[mask_a, label_col].values,
                   df_a.loc[mask_a, pred_col_a].values)
        bb = _bacc(df_b.loc[mask_b, label_col].values,
                   df_b.loc[mask_b, pred_col_b].values)
        diffs.append(bb - ba)

    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": point_diff, "ci_lo": lo, "ci_hi": hi,
            "point_a": point_a, "point_b": point_b}


def main() -> None:
    base = load(WIN / "baselines_bacc" / "metrics.json")
    loco = load(WIN / "pose_cnn_loco_dev" / "metrics_dev.json")

    # rows: chance + TEST rows first, then DEV rows
    rows = [
        {
            "name": "Always yes or always no",
            "detail": "chance floor",
            "bacc": 0.5,
            "split": "any",
            "colour": PALE,
            "section": "test",
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
            "section": "test",
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
            "section": "dev",
        },
    ]

    videomae_csvs = {}
    for folder, name in (
        ("frozen_encoder", "VideoMAE frozen encoder, identity fixed"),
        ("last_blocks_unfrozen", "VideoMAE last 2 blocks unfrozen, identity fixed"),
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
                "section": "dev",
            }
        )
        videomae_csvs[folder] = FIXED / folder / "oof_predictions.csv"

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

    # paired difference: last_blocks minus frozen
    diff_result = None
    if ("frozen_encoder" in videomae_csvs
            and "last_blocks_unfrozen" in videomae_csvs):
        diff_result = paired_bootstrap_difference(
            videomae_csvs["frozen_encoder"],
            videomae_csvs["last_blocks_unfrozen"],
        )

    # --- plot ---
    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.axvline(0.5, color=INK, lw=1.4, ls="--")

    # find where the section changes from test to dev
    separator_y = None
    for i, row in enumerate(rows):
        if i > 0 and row["section"] == "dev" and rows[i - 1]["section"] == "test":
            separator_y = -i + 0.5

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
            f"{row['name']}\n{row['detail']}",
            va="center", ha="left", fontsize=9.2, color=INK,
        )
        # split label shown at right margin
        ax.text(
            0.78, y, f"{row['bacc']:.3f}{interval}",
            va="center", ha="left", fontsize=9.2, color=INK,
        )

    # separator line + section labels
    if separator_y is not None:
        ax.axhline(separator_y, color=MUTED, lw=0.8, ls=":")
        ax.text(0.995, separator_y + 0.35, "TEST", fontsize=8, color=MUTED,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.text(0.995, separator_y - 0.35, "DEV out-of-fold", fontsize=8,
                color=MUTED, ha="right", va="top",
                transform=ax.get_yaxis_transform())

    # paired difference annotation
    if diff_result is not None:
        d = diff_result
        sign = "excludes zero" if d["ci_lo"] > 0 else "includes zero"
        diff_text = (
            f"Paired difference (unfrozen minus frozen): "
            f"{d['point']:+.3f}  [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]  ({sign})"
        )
        fig.text(0.01, 0.095, diff_text, fontsize=8.6, color=INK)

    ax.set_xlim(0.27, 1.05)
    ax.set_ylim(-len(rows) + 0.4, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("Balanced accuracy (floor 0.500, dashed)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.suptitle(
        "Identity-fixed VideoMAE against valid 3 s nod systems",
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.text(
        0.01, 0.02,
        "Pitch rule evaluated on TEST. Pose CNN and VideoMAE evaluated DEV out-of-fold. "
        "Withdrawn RGB rows (wrong-person crop) are excluded. "
        "Intervals are 95 percent clip bootstrap.",
        fontsize=8.4, color=MUTED,
    )
    fig.subplots_adjust(top=0.88, bottom=0.20, left=0.02, right=0.99)
    out = FIXED / "model_comparison.png"
    fig.savefig(out, dpi=180, facecolor=PAPER)
    fig.savefig(out.with_suffix(".pdf"), facecolor=PAPER)
    print(f"wrote {out}")
    if diff_result is not None:
        d = diff_result
        print(f"paired diff (unfrozen - frozen): {d['point']:+.3f} "
              f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]")


if __name__ == "__main__":
    main()
