#!/usr/bin/env python3
"""VideoMAE result figures for the dissertation.

Reads:
  results/videomae_frozen_head/training_history.csv  (epoch, train_loss, dev_f1, ...)
  results/tables/bootstrap_ci.csv                    (model, ..., f1, f1_ci_lo, f1_ci_hi)

Writes:
  figures/videomae_training_curve.png  (loss + DEV F1 vs epoch, best epoch marked)
  figures/model_comparison_f1.png      (TEST F1 bars with 95% CI error bars)

Run: python3 scripts/plot_videomae_results.py
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MODEL_LABELS = {
    "rule_baseline": "Pose rule",
    "pose_cnn_xyz_deriv": "Pose CNN\n(xyz + deriv)",
    "videomae_frozen_head": "Frozen\nVideoMAE head",
}
MODEL_ORDER = ["rule_baseline", "pose_cnn_xyz_deriv", "videomae_frozen_head"]


def load_training_history():
    path = RESULTS / "videomae_frozen_head" / "training_history.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    loss = [float(r["train_loss"]) for r in rows]
    dev_f1 = [float(r["dev_f1"]) for r in rows]
    return epochs, loss, dev_f1


def best_epoch_from_metrics():
    path = RESULTS / "videomae_frozen_head" / "metrics.json"
    with open(path) as f:
        meta = json.load(f)
    return int(meta["best_epoch"]), float(meta["dev_f1"])


def plot_training_curve():
    epochs, loss, dev_f1 = load_training_history()
    best_epoch, best_dev_f1 = best_epoch_from_metrics()

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()

    ax1.plot(epochs, loss, color="#1f77b4", marker="o", markersize=4,
             linewidth=1.5, label="Training loss")
    ax2.plot(epochs, dev_f1, color="#d62728", marker="s", markersize=4,
             linewidth=1.5, label="DEV F1")

    ax1.axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
    ax2.plot([best_epoch], [best_dev_f1], marker="*", color="#d62728",
             markersize=16, linestyle="none",
             label=f"Early stop (epoch {best_epoch})")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="#1f77b4")
    ax2.set_ylabel("DEV F1", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_xticks(epochs)
    ax2.set_ylim(0.5, 1.0)

    handles = [ln for ln in ax1.get_lines() + ax2.get_lines()
               if not ln.get_label().startswith("_")]
    ax1.legend(handles, [ln.get_label() for ln in handles],
               loc="center right", fontsize=9, framealpha=0.9)

    fig.suptitle("Frozen VideoMAE head: training loss and DEV F1 by epoch",
                 fontsize=11)
    fig.tight_layout()
    out = FIGURES / "videomae_training_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_model_comparison():
    path = RESULTS / "tables" / "bootstrap_ci.csv"
    with open(path, newline="") as f:
        rows = {r["model"]: r for r in csv.DictReader(f)}

    labels, f1s, yerr_lo, yerr_hi = [], [], [], []
    for key in MODEL_ORDER:
        r = rows[key]
        f1 = float(r["f1"])
        lo = float(r["f1_ci_lo"])
        hi = float(r["f1_ci_hi"])
        labels.append(MODEL_LABELS[key])
        f1s.append(f1)
        yerr_lo.append(f1 - lo)
        yerr_hi.append(hi - f1)

    colors = ["#4c78a8", "#59a14f", "#e15759"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars = ax.bar(labels, f1s, color=colors, width=0.55,
                  yerr=[yerr_lo, yerr_hi],
                  error_kw=dict(ecolor="black", elinewidth=1.2, capsize=5))

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{f1:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("TEST F1 (n = 15, scored once)")
    ax.set_title("TEST F1 with 95% bootstrap CIs (1000 resamples, seed 42)",
                 fontsize=10.5)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    out = FIGURES / "model_comparison_f1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    FIGURES.mkdir(exist_ok=True)
    for out in (plot_training_curve(), plot_model_comparison()):
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
