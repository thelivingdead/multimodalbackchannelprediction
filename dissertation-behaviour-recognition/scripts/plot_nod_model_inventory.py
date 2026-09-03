#!/usr/bin/env python3
"""Audit inventory: every nod system run so far, on one balanced-accuracy axis.

This is a reproducibility artefact for an appendix, not a results figure. It
includes runs that were withdrawn or superseded, which is the point of it.

Balanced accuracy is the only metric available for all of them and its floor
does not move with prevalence, so the 60 s clip protocol and the 3 s window
protocol can share an axis. F1 is printed beside the 60 s rows because that is
how they were originally reported.

Point estimates are read from the saved metrics files. Intervals are recomputed
here by resampling whole clips from the stored per-item predictions, and the
recomputed point estimate is checked against the stored one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clip_metrics import clip_binary_metrics  # noqa: E402
from src.windowed_baselines import clip_bootstrap  # noqa: E402

RES = ROOT / "results"

INK = "#1d1d1f"
MUTED = "#5c5c63"
GREY = "#a8a8ad"
PALE = "#cdcdd2"
ORANGE = "#d97932"
BLUE = "#3178a8"
GREEN = "#1b7f4b"
RED = "#b5342a"
PAPER = "#fffdf8"

STATUS_STYLE = {
    "valid": {"colour": None, "marker": "o", "filled": True},
    "DEV only": {"colour": None, "marker": "o", "filled": False},
    "identity unverified": {"colour": GREY, "marker": "D", "filled": False},
    "superseded": {"colour": GREY, "marker": "s", "filled": False},
    "withdrawn": {"colour": GREY, "marker": "X", "filled": True},
    "trivial": {"colour": PALE, "marker": "o", "filled": True},
}


def bootstrap_row(
    path: Path,
    id_col: str,
    label_col: str,
    pred_col: str,
    split_col: str | None = None,
    split: str | None = None,
) -> dict | None:
    """Clip-resampled balanced accuracy from stored per-item predictions."""
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if split_col and split_col in frame.columns:
        frame = frame[frame[split_col].astype(str).str.upper() == split]
    for column in (id_col, label_col, pred_col):
        if column not in frame.columns:
            raise SystemExit(f"STOP: {path} has no column {column}")
    labels = frame[label_col].to_numpy(dtype=int)
    if len(labels) == 0 or labels.min() == labels.max():
        return None
    ids = frame[id_col].astype(str).to_numpy()
    predictions = frame[pred_col].to_numpy(dtype=int)
    boot = clip_bootstrap(ids, labels, predictions)
    boot["point_estimate"] = float(
        clip_binary_metrics(labels, predictions)["balanced_accuracy"]
    )
    boot["n_items"] = int(len(labels))
    return boot


def dig(payload: dict, *path, default=None):
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"STOP: missing {path}")
    return json.loads(path.read_text())


def build_rows() -> tuple[list[dict], list[dict]]:
    majority = load(RES / "majority_baseline" / "metrics.json")
    rule60 = load(RES / "rule_test_metrics.json")
    cnn60 = load(RES / "classifier_test_metrics.json")
    frozen60 = load(RES / "videomae_frozen_head" / "metrics.json")
    fine60 = load(RES / "videomae_finetuned" / "metrics.json")
    n200 = load(RES / "videomae_finetuned_n200" / "metrics.json")

    clip_rows = [
        {
            "name": "Always positive",
            "detail": "trivial",
            "bacc": dig(majority, "always_positive", "balanced_accuracy"),
            "f1": dig(majority, "always_positive", "f1"),
            "split": "TEST",
            "status": "trivial",
            "colour": PALE,
        },
        {
            "name": "Pose rule, frozen amplitude",
            "pred": (RES / "rule_test_predictions.csv", "sample_id", "label", "pred", None, None),
            "detail": "EMOCA Euler x, τ = 16.35°",
            "bacc": dig(rule60, "balanced_accuracy"),
            "f1": dig(rule60, "f1"),
            "split": "TEST",
            "status": "valid",
            "colour": ORANGE,
        },
        {
            "name": "Pose 1D CNN",
            "pred": (RES / "classifier_test_predictions.csv", "sample_id", "label", "pred", None, None),
            "detail": "Euler + derivatives, 80 pseudo-labels",
            "bacc": dig(cnn60, "balanced_accuracy"),
            "f1": dig(cnn60, "f1"),
            "split": "TEST",
            "status": "valid",
            "colour": BLUE,
        },
        {
            "name": "VideoMAE, frozen encoder",
            "pred": (RES / "videomae_frozen_head" / "predictions.csv", "sample_id", "label", "pred", None, None),
            "detail": "RGB 16-frame crops, linear head",
            "bacc": dig(frozen60, "test_metrics", "balanced_accuracy"),
            "f1": dig(frozen60, "test_metrics", "f1"),
            "split": "TEST",
            "status": "withdrawn",
            "colour": RED,
        },
        {
            "name": "VideoMAE, fine-tuned",
            "pred": (RES / "videomae_finetuned" / "predictions_test.csv", "clip_id", "label", "pred", "split", "TEST"),
            "detail": "last 4 blocks, 80 pseudo-labels",
            "bacc": dig(fine60, "test_metrics", "balanced_accuracy"),
            "f1": dig(fine60, "test_metrics", "f1"),
            "split": "TEST",
            "status": "withdrawn",
            "colour": GREEN,
        },
        {
            "name": "VideoMAE, fine-tuned, n = 200",
            "pred": (RES / "videomae_finetuned_n200" / "predictions_test.csv", "clip_id", "label", "pred", "split", "TEST"),
            "detail": "scaling ablation",
            "bacc": dig(n200, "test_metrics", "balanced_accuracy"),
            "f1": dig(n200, "test_metrics", "f1"),
            "split": "TEST",
            "status": "withdrawn",
            "colour": RED,
        },
    ]

    base = load(RES / "windowed_nod" / "baselines_bacc" / "metrics.json")
    loco_cnn = load(RES / "windowed_nod" / "pose_cnn_loco_dev" / "metrics_dev.json")
    mil = load(
        RES / "windowed_nod" / "pose_mil_pseudo80_dev_bacc" / "metrics_dev.json"
    )
    old_cnn = load(RES / "windowed_nod" / "pose_cnn" / "metrics.json")
    old_mae = load(RES / "windowed_nod" / "videomae_finetuned" / "metrics.json")
    mae6 = load(RES / "windowed_nod" / "videomae_loco_dev" / "metrics_dev.json")
    mae12 = load(RES / "windowed_nod" / "videomae_loco_dev_ep12" / "metrics_dev.json")

    rule_ci = dig(
        base, "clip_bootstrap", "TEST", "dev_selected_window_rule", "balanced_accuracy"
    )
    cnn_ci = dig(loco_cnn, "clip_bootstrap_at_0.5", "balanced_accuracy")

    window_rows = [
        {
            "name": "Always yes or always no",
            "detail": "trivial",
            "bacc": dig(base, "metrics", "TEST", "always_yes", "balanced_accuracy"),
            "split": "TEST",
            "status": "trivial",
            "colour": PALE,
        },
        {
            "name": "60 s threshold, transferred",
            "pred": (RES / "windowed_nod" / "baselines_bacc" / "predictions.csv", "sample_id", "label", "frozen_threshold_pred", "split", "TEST"),
            "detail": "frozen τ applied to 3 s windows",
            "bacc": dig(
                base,
                "metrics",
                "TEST",
                "frozen_60s_threshold_transfer",
                "balanced_accuracy",
            ),
            "split": "TEST",
            "status": "valid",
            "colour": ORANGE,
        },
        {
            "name": "Pitch rule, re-selected at 3 s",
            "pred": (RES / "windowed_nod" / "baselines_bacc" / "predictions.csv", "sample_id", "label", "dev_selected_pred", "split", "TEST"),
            "detail": "one threshold, chosen on DEV",
            "bacc": dig(
                base,
                "metrics",
                "TEST",
                "dev_selected_window_rule",
                "balanced_accuracy",
            ),
            "ci": (rule_ci or {}).get("ci_lower_95"),
            "ci_high": (rule_ci or {}).get("ci_upper_95"),
            "split": "TEST",
            "status": "valid",
            "colour": ORANGE,
        },
        {
            "name": "Pose CNN, leave-one-clip-out",
            "pred": (RES / "windowed_nod" / "pose_cnn_loco_dev" / "predictions_oof_dev.csv", "sample_id", "label", "pred_at_0.5", None, None),
            "detail": "14 clips train, 1 held out, 15 folds",
            "bacc": dig(loco_cnn, "at_fixed_threshold_0.5", "balanced_accuracy"),
            "ci": (cnn_ci or {}).get("ci_lower_95"),
            "ci_high": (cnn_ci or {}).get("ci_upper_95"),
            "split": "DEV, out of fold",
            "status": "valid",
            "colour": BLUE,
        },
        {
            "name": "Pose MIL, 80 pseudo bags",
            "pred": (RES / "windowed_nod" / "pose_mil_pseudo80_dev_bacc" / "predictions_dev.csv", "sample_id", "label", "prediction", "split", "DEV"),
            "detail": "weak supervision, clip level bags",
            "bacc": dig(mil, "dev_window", "balanced_accuracy"),
            "split": "DEV",
            "status": "DEV only",
            "colour": BLUE,
        },
        {
            "name": "Pose CNN trained on DEV",
            "pred": (RES / "windowed_nod" / "pose_cnn" / "predictions.csv", "sample_id", "label", "pred", "split", "TEST"),
            "detail": "trained and selected on the same clips",
            "bacc": dig(old_cnn, "test_window", "balanced_accuracy"),
            "split": "TEST",
            "status": "superseded",
            "colour": GREY,
        },
        {
            "name": "VideoMAE fine-tuned on DEV",
            "pred": (RES / "windowed_nod" / "videomae_finetuned" / "predictions_test.csv", "sample_id", "label", "pred", None, None),
            "detail": "DEV F1 1.000, TEST F1 0.000",
            "bacc": dig(old_mae, "test_window", "balanced_accuracy"),
            "split": "TEST",
            "status": "superseded",
            "colour": GREY,
        },
        {
            "name": "VideoMAE LOCO, 6 epochs",
            "pred": (RES / "windowed_nod" / "videomae_loco_dev" / "predictions_oof_dev.csv", "sample_id", "label", "pred_at_0.5", None, None),
            "detail": "crops showed the excluded person in 51 % of windows",
            "bacc": dig(mae6, "at_fixed_threshold_0.5", "balanced_accuracy"),
            "split": "DEV, out of fold",
            "status": "withdrawn",
            "colour": GREY,
        },
        {
            "name": "VideoMAE LOCO, 12 epochs",
            "pred": (RES / "windowed_nod" / "videomae_loco_dev_ep12" / "predictions_oof_dev.csv", "sample_id", "label", "pred_at_0.5", None, None),
            "detail": "same crops",
            "bacc": dig(mae12, "at_fixed_threshold_0.5", "balanced_accuracy"),
            "split": "DEV, out of fold",
            "status": "withdrawn",
            "colour": GREY,
        },
    ]

    for row in clip_rows + window_rows:
        if row["bacc"] is None:
            raise SystemExit(f"STOP: no balanced accuracy for {row['name']}")
        spec = row.pop("pred", None)
        if spec is None:
            continue
        boot = bootstrap_row(*spec)
        if boot is None:
            print(f"NOTE: no interval for {row['name']} ({spec[0].name})")
            continue
        drift = abs(boot["point_estimate"] - float(row["bacc"]))
        if drift > 0.002:
            raise SystemExit(
                f"STOP: {row['name']} stored balanced accuracy {row['bacc']:.4f} "
                f"but its predictions give {boot['point_estimate']:.4f}"
            )
        row["ci"] = boot["balanced_accuracy"]["ci_lower_95"]
        row["ci_high"] = boot["balanced_accuracy"]["ci_upper_95"]
        row["n_clips"] = boot["n_clips"]
    return clip_rows, window_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RES / "windowed_nod" / "nod_model_inventory")
    args = ap.parse_args()

    clip_rows, window_rows = build_rows()
    blocks = [
        ("60 s clip protocol: one label per clip, 15 TEST clips", clip_rows),
        ("3 s sliding window protocol: 29 windows per clip, 435 windows", window_rows),
    ]

    entries: list[dict] = []
    headers: dict[int, str] = {}
    slot = 0
    for title, rows in blocks:
        headers[slot] = title
        slot += 1
        for row in rows:
            entries.append({**row, "slot": slot})
            slot += 1
        slot += 0.6

    height = 0.42 * slot + 3.4
    fig = plt.figure(figsize=(13.6, height), facecolor=PAPER)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PAPER)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=9)

    lo = min(min(e["bacc"] for e in entries),
             min((e["ci"] for e in entries if e.get("ci") is not None), default=0.5))
    hi = max(max(e["bacc"] for e in entries),
             max((e["ci_high"] for e in entries if e.get("ci") is not None), default=0.5))
    lo = np.floor((lo - 0.02) * 20) / 20
    hi = np.ceil((hi + 0.02) * 20) / 20
    name_x = lo - 0.42 * (hi - lo)
    value_x = hi + 0.03 * (hi - lo)
    status_x = hi + 0.55 * (hi - lo)

    ax.axvline(0.5, color=INK, lw=1.5, ls="--", zorder=2)
    ax.axvspan(0.5, hi, color="#f2f7f3", zorder=0)

    for entry in entries:
        y = -entry["slot"]
        style = STATUS_STYLE[entry["status"]]
        colour = style["colour"] or entry["colour"]
        ax.plot([0.5, entry["bacc"]], [y, y], color=colour, lw=1.6, alpha=0.55, zorder=3)
        ax.scatter(
            entry["bacc"],
            y,
            s=118,
            marker=style["marker"],
            facecolors=colour if style["filled"] else PAPER,
            edgecolors=colour,
            linewidths=1.8,
            zorder=4,
        )
        if entry.get("ci") is not None:
            ax.plot(
                [entry["ci"], entry["ci_high"]],
                [y, y],
                color=colour,
                lw=1.4,
                zorder=3,
            )
            for edge in (entry["ci"], entry["ci_high"]):
                ax.plot([edge, edge], [y - 0.2, y + 0.2], color=colour, lw=1.4, zorder=3)

        label = f"{entry['bacc']:.3f}"
        if entry.get("ci") is not None:
            label += f"  [{entry['ci']:.3f}, {entry['ci_high']:.3f}]"
        if entry.get("f1") is not None:
            label += f"   (F1 {entry['f1']:.2f})"
        ax.text(value_x, y, label, ha="left", va="center", fontsize=8.8, color=INK)
        ax.text(
            status_x,
            y,
            entry["status"],
            ha="left",
            va="center",
            fontsize=8.6,
            color=MUTED if entry["status"] == "valid" else INK,
            style="italic" if entry["status"] == "valid" else "normal",
            fontweight="600" if entry["status"] in ("withdrawn", "superseded") else "normal",
        )
        ax.text(
            name_x,
            y,
            f"{entry['name']}\n{entry['detail']} · {entry['split']}",
            ha="left",
            va="center",
            fontsize=8.8,
            color=INK,
            linespacing=1.35,
        )

    for slot_index, title in headers.items():
        ax.text(
            name_x,
            -slot_index,
            title,
            ha="left",
            va="center",
            fontsize=10.4,
            color=INK,
            fontweight="700",
        )

    ax.set_xlim(name_x - 0.01, status_x + 0.30 * (hi - lo))
    ax.set_ylim(-slot - 0.2, 1.4)
    ax.set_yticks([])
    ticks = [t for t in np.arange(0.1, 1.01, 0.1) if lo - 1e-9 <= t <= hi + 1e-9]
    ax.set_xticks(ticks)
    ax.spines["bottom"].set_bounds(lo, hi)
    ax.set_xlabel("Balanced accuracy (floor 0.500, shown dashed)", fontsize=10.4, color=INK)
    span = ax.get_xlim()
    ax.xaxis.set_label_coords(
        ((lo + hi) / 2 - span[0]) / (span[1] - span[0]), -0.055
    )
    ax.text(
        0.505,
        0.5,
        "better than chance →",
        fontsize=9,
        color=GREEN,
        fontweight="600",
        ha="left",
        va="center",
    )

    handles = [
        Line2D([], [], marker="o", ls="none", color=INK, markersize=8, label="valid"),
        Line2D([], [], marker="o", ls="none", markerfacecolor=PAPER,
               markeredgecolor=INK, markersize=8, label="DEV only, no TEST score"),
        Line2D([], [], marker="s", ls="none", markerfacecolor=PAPER,
               markeredgecolor=GREY, markersize=8, label="superseded, protocol violation"),
        Line2D([], [], marker="X", ls="none", color=GREY, markersize=9,
               label="withdrawn, crop defect measured"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        ncol=2,
        columnspacing=1.4,
    )

    n_systems = sum(1 for e in entries if e["status"] != "trivial")
    scored = [e for e in entries if e.get("ci") is not None]
    n_clear = sum(1 for e in scored if e["ci"] > 0.5)
    verdict = (
        "no interval clears 0.500"
        if n_clear == 0
        else f"{n_clear} of {len(scored)} intervals clear 0.500"
    )
    fig.suptitle(
        f"Audit of all {n_systems} nod systems run so far: {verdict}, and the RGB "
        "rows are not safe to quote",
        fontsize=13.0,
        color=INK,
        fontweight="700",
        x=0.008,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.008,
        0.012,
        "Audit inventory for an appendix: it deliberately includes runs that were withdrawn or superseded. Balanced accuracy is used throughout because its floor stays at 0.500 whatever the class\n"
        "balance, so both protocols share an axis. The 60 s rows also show F1, the metric they were first reported in: the pose rule's F1 of 0.67 beats an always-positive predictor's 0.80 on nothing, "
        "since\nF1 rewards predicting the majority class, and its balanced accuracy is 0.450, below chance.\n"
        "Intervals are 95 % percentile bootstrap over whole clips, 2000 resamples, recomputed here from each system's stored per-item predictions; every point estimate was checked against its saved\n"
        "metrics file. The 60 s protocol resamples only 15 clips, which is why those intervals are three to five times wider than the 3 s ones and why none of them clears 0.500. The frozen\n"
        "encoder is the one interval that excludes 0.500, and it does so from below: on locked TEST that system is reliably worse than chance.\n"
        "Every RGB row is withdrawn, in both protocols, because they share one crop function: the 3 s fetcher imports the 60 s module and calls the same largest-detected-face selection, with no identity\n"
        "check. Audited against the side each clip's annotator was told to watch, that put the excluded person in 51 % of 3 s windows and in 6 of the 15 60 s TEST clips. Two further clips detected no face\n"
        "at all and fell back to a centred crop spanning both people, which is unattributable rather than wrong. The withdrawn 60 s rows include the system first reported at F1 0.82.",
        fontsize=8.6,
        color=MUTED,
        ha="left",
        va="bottom",
        linespacing=1.5,
    )
    fig.subplots_adjust(top=0.9, bottom=0.235, left=0.012, right=0.995)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=200, facecolor=PAPER)
    fig.savefig(args.out.with_suffix(".pdf"), facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {args.out.with_suffix('.png')}")
    print(f"      {args.out.with_suffix('.pdf')}")
    for title, rows in blocks:
        print(f"\n{title}")
        for row in rows:
            f1 = "" if row.get("f1") is None else f"  F1 {row['f1']:.2f}"
            ci = (
                "        -        "
                if row.get("ci") is None
                else f"[{row['ci']:.3f}, {row['ci_high']:.3f}]"
            )
            print(f"  {row['name']:36s} {row['bacc']:.3f}  {ci}  "
                  f"{row['split']:18s} {row['status']}{f1}")


if __name__ == "__main__":
    main()
