#!/usr/bin/env python3
"""Does more training data help 3 s nod detection? Ordered by clips used.

The pose CNN saw 14 human annotated clips because only the gold clips carry
window labels. The MIL runs saw 80 clips, but labelled once per 60 s clip
by the frozen pitch rule, which cannot say which of the 29 windows holds the
nod. Putting the systems on one axis in order of training clips shows what that
distinction costs, and that quantity does not substitute for localisation.

Values come from the saved metrics files. Intervals are recomputed by
resampling whole clips from the stored per window predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_nod_model_inventory import bootstrap_row  # noqa: E402

RES = ROOT / "results"
WIN = RES / "windowed_nod"
OUT_STEM = WIN / "supervision_comparison_dev"

INK = "#1d1d1f"
MUTED = "#5c5c63"
GREY = "#a8a8ad"
PALE = "#cdcdd2"
ORANGE = "#d97932"
BLUE = "#3178a8"
GREEN = "#1b7f4b"
PAPER = "#fffdf8"


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"STOP: missing {path}\n"
            "       This run lives on otter. Copy it across first, e.g.\n"
            "       scp -r otterdiv:multimodalbackchannelprediction/"
            "dissertation-behaviour-recognition/results/windowed_nod/"
            f"{path.parent.name} \\\n"
            "           dissertation-behaviour-recognition/results/windowed_nod/"
        )
    with path.open() as handle:
        return json.load(handle)


def rows() -> list[dict]:
    base = load(WIN / "baselines_bacc" / "metrics.json")
    loco = load(WIN / "pose_cnn_loco_dev" / "metrics_dev.json")
    mil_dev = load(WIN / "pose_mil_pseudo80_dev_bacc" / "metrics_dev.json")
    mil_train = load(WIN / "pose_mil_pseudo80_trainsel" / "metrics_dev.json")
    teacher = load(RES / "rule_test_metrics.json")

    collected = [
        {
            "name": "Always yes, or always no",
            "supervision": "no training at all",
            "clips": 0,
            "windows": 0,
            "bacc": 0.5,
            "colour": PALE,
            "hollow": False,
            "split": "TEST",
        },
        {
            "name": "Pitch rule, one threshold",
            "supervision": "1 parameter, chosen on DEV",
            "clips": 0,
            "windows": 0,
            "bacc": float(
                base["metrics"]["TEST"]["dev_selected_window_rule"][
                    "balanced_accuracy"
                ]
            ),
            "pred": (
                WIN / "baselines_bacc" / "predictions.csv",
                "sample_id", "label", "dev_selected_pred", "split", "TEST",
            ),
            "colour": ORANGE,
            "hollow": False,
            "split": "TEST",
        },
        {
            "name": "Pose CNN, leave-one-clip-out",
            "supervision": "human window labels",
            "clips": 14,
            "windows": 14 * 29,
            "bacc": float(loco["at_fixed_threshold_0.5"]["balanced_accuracy"]),
            "pred": (
                WIN / "pose_cnn_loco_dev" / "predictions_oof_dev.csv",
                "sample_id", "label", "pred_at_0.5", None, None,
            ),
            "colour": BLUE,
            "hollow": False,
            "split": "DEV, out of fold",
        },
        {
            "name": "Pose MIL, 80 weak bags",
            "supervision": "clip level pseudo labels",
            "clips": int(mil_train["n_train_clips"]),
            "windows": int(mil_train["n_train_instances"]),
            "bacc": float(mil_train["dev_window"]["balanced_accuracy"]),
            "pred": (
                WIN / "pose_mil_pseudo80_trainsel" / "predictions_dev.csv",
                "sample_id", "label", "prediction", "split", "DEV",
            ),
            "oracle": float(mil_train["dev_selected_oracle"]["balanced_accuracy"]),
            "bag": float(mil_train["train_oof_bag_balanced_accuracy"]),
            "teacher": float(teacher["balanced_accuracy"]),
            "colour": GREEN,
            "hollow": False,
            "split": "DEV, scored once",
        },
        {
            "name": "Pose MIL, same bags, DEV selected",
            "supervision": "epoch and threshold tuned on DEV",
            "clips": int(mil_dev["n_train_clips"]),
            "windows": int(mil_dev["n_train_instances"]),
            "bacc": float(mil_dev["dev_window"]["balanced_accuracy"]),
            "pred": (
                WIN / "pose_mil_pseudo80_dev_bacc" / "predictions_dev.csv",
                "sample_id", "label", "prediction", "split", "DEV",
            ),
            "colour": GREY,
            "hollow": True,
            "split": "DEV, contaminated",
        },
    ]
    for row in collected:
        spec = row.pop("pred", None)
        if spec is None:
            continue
        boot = bootstrap_row(*spec)
        if boot is None:
            raise SystemExit(f"STOP: no interval computable for {row['name']}")
        if abs(boot["point_estimate"] - row["bacc"]) > 0.002:
            raise SystemExit(
                f"STOP: {row['name']} stored {row['bacc']:.4f} but its predictions "
                f"give {boot['point_estimate']:.4f}"
            )
        row["ci"] = boot["balanced_accuracy"]["ci_lower_95"]
        row["ci_high"] = boot["balanced_accuracy"]["ci_upper_95"]
    return collected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-stem", type=Path, default=OUT_STEM)
    args = parser.parse_args()
    entries = rows()

    lo = min(min(e["bacc"] for e in entries),
             min(e.get("ci", 1.0) for e in entries)) - 0.02
    hi = max(max(e["bacc"] for e in entries),
             max(e.get("ci_high", 0.0) for e in entries)) + 0.02
    lo, hi = np.floor(lo * 20) / 20, np.ceil(hi * 20) / 20
    name_x = lo - 0.52 * (hi - lo)
    value_x = hi + 0.04 * (hi - lo)
    data_x = hi + 0.52 * (hi - lo)

    fig, ax = plt.subplots(figsize=(13.2, 6.05))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.axvspan(0.5, hi, color="#f2f7f3", zorder=0)
    ax.axvline(0.5, color=INK, lw=1.5, ls="--", zorder=2)

    for index, entry in enumerate(entries):
        y = -index
        colour = entry["colour"]
        if "ci" in entry:
            ax.plot(
                [entry["ci"], entry["ci_high"]], [y, y],
                color=colour, lw=2.0, solid_capstyle="butt", zorder=3,
            )
            for edge in (entry["ci"], entry["ci_high"]):
                ax.plot([edge, edge], [y - 0.13, y + 0.13], color=colour, lw=2.0,
                        zorder=3)
        ax.plot(
            entry["bacc"], y, marker="o", markersize=11,
            markerfacecolor=PAPER if entry["hollow"] else colour,
            markeredgecolor=colour, markeredgewidth=2.0, zorder=5,
        )
        ax.text(
            name_x, y, f"{entry['name']}\n{entry['supervision']} · {entry['split']}",
            ha="left", va="center", fontsize=9.6, color=INK, linespacing=1.5,
        )
        interval = (
            "" if "ci" not in entry
            else f"  [{entry['ci']:.3f}, {entry['ci_high']:.3f}]"
        )
        value_label = f"{entry['bacc']:.3f}{interval}"
        if "oracle" in entry:
            value_label += f"\nDEV tuned oracle: {entry['oracle']:.3f}"
        ax.text(value_x, y, value_label, ha="left", va="center", fontsize=9.4,
                color=INK, linespacing=1.6)
        described = (
            "none" if entry["clips"] == 0
            else f"{entry['clips']} clips, {entry['windows']} windows"
        )
        if "bag" in entry:
            described += f"\nagrees with its teacher: {entry['bag']:.3f}"
        ax.text(data_x, y, described, ha="left", va="center", fontsize=9.4,
                color=MUTED, linespacing=1.6)

    ax.text(
        data_x, 0.95, "training data", ha="left", va="bottom", fontsize=9.4,
        color=INK, fontweight="bold",
    )
    ax.text(
        value_x, 0.95, "balanced accuracy, 95 % CI", ha="left", va="bottom",
        fontsize=9.4, color=INK, fontweight="bold",
    )
    ax.text(0.505, 0.95, "better than chance →", ha="left", va="center",
            fontsize=9.4, color=GREEN, style="italic")

    ax.set_xlim(name_x - 0.01, data_x + 0.50 * (hi - lo))
    ax.set_ylim(-len(entries) + 0.35, 1.45)
    ax.set_yticks([])
    ax.set_xticks([t for t in np.arange(0.35, 0.71, 0.05) if lo <= t <= hi])
    ax.spines["bottom"].set_bounds(lo, hi)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9.4)
    ax.set_xlabel("Balanced accuracy on 435 human 3 s windows (floor 0.500, dashed)",
                  fontsize=10.4, color=INK)
    span = ax.get_xlim()
    ax.xaxis.set_label_coords(((lo + hi) / 2 - span[0]) / (span[1] - span[0]), -0.085)

    mil = next(e for e in entries if "oracle" in e)
    cnn = next(e for e in entries if e["clips"] == 14)
    fig.suptitle(
        f"Weak supervision mimics its teacher at {mil['bag']:.3f} and still cannot "
        "localise the nod\n"
        f"{mil['clips']} weakly labelled clips score {mil['bacc']:.3f} on 3 s windows, "
        f"where {cnn['clips']} human labelled clips score {cnn['bacc']:.3f}",
        fontsize=12.6, fontweight="bold", color=INK, x=0.008, ha="left", y=0.985,
        va="top", linespacing=1.45,
    )
    fig.text(
        0.008, 0.012,
        "Only the 15 gold clips carry human window labels, so the pose CNN trains on 14 at a time. The 80 TRAIN clips carry one label per 60 s clip from the frozen pitch rule, which cannot say which of the\n"
        "29 windows holds the nod, so they are learned as bags by top 2 multiple instance pooling.\n"
        f"The {mil['bag']:.3f} is agreement with that rule, not nod accuracy: those 80 clips have no human label, and the rule itself scores {mil['teacher']:.3f} against human clip labels. Out of fold the model still\n"
        f"learned the teacher, and sweeping the threshold on DEV itself reaches only {mil['oracle']:.3f}. The student learned its lesson and still could not localise. Intervals are 95 % clip bootstrap, 2000\n"
        "resamples, from stored per window predictions. The grey row tuned on the DEV windows it reports.",
        fontsize=8.6, color=MUTED, ha="left", va="bottom", linespacing=1.5,
    )
    fig.subplots_adjust(top=0.835, bottom=0.24, left=0.012, right=0.995)

    out_stem = args.out_stem
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(suffix), dpi=200,
                    facecolor=fig.get_facecolor())
    print(f"wrote {out_stem.with_suffix('.png')}")
    print(f"      {out_stem.with_suffix('.pdf')}")
    print()
    for entry in entries:
        interval = (
            "      -      " if "ci" not in entry
            else f"[{entry['ci']:.3f}, {entry['ci_high']:.3f}]"
        )
        described = "none" if entry["clips"] == 0 else f"{entry['clips']} clips"
        print(f"  {entry['name']:38s} {entry['bacc']:.3f}  {interval}  "
              f"{described:10s} {entry['split']}")


if __name__ == "__main__":
    main()
