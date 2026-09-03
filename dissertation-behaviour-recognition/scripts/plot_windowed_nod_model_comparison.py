#!/usr/bin/env python3
"""Compare the nod pitch rule against the pose CNN on the 3 s DEV windows.

Intervals are percentile bootstrap over whole clips, the unit of independence,
computed from the stored per-window scores of each method.

VideoMAE is excluded: its crops tracked the largest detected face and changed
person in 14 of 15 clips, so those runs are invalid rather than weak. They are
reported separately by plot_windowed_rgb_crop_defect.py.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clip_metrics import clip_binary_metrics  # noqa: E402
from src.windowed_baselines import (  # noqa: E402
    average_precision,
    clip_bootstrap,
    clip_bootstrap_pr_auc,
)

NOD = ROOT / "results" / "windowed_nod"
AXIS_LABEL = {"nod": "pitch", "shake": "yaw"}

INK = "#1d1d1f"
MUTED = "#5c5c63"
GREY = "#a8a8ad"
ORANGE = "#d97932"
BLUE = "#3178a8"
PAPER = "#fffdf8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("nod", "shake"), default="nod")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    task_dir = ROOT / "results" / f"windowed_{args.task}"
    out = args.out or (task_dir / "model_comparison_dev")
    rule_path = task_dir / "baselines_bacc" / "predictions.csv"
    cnn_path = task_dir / "pose_cnn_loco_dev" / "predictions_oof_dev.csv"
    baseline_metrics_path = task_dir / "baselines_bacc" / "metrics.json"
    for path in (rule_path, cnn_path, baseline_metrics_path):
        if not path.exists():
            raise SystemExit(f"STOP: missing {path}")

    baseline_metrics = json.loads(baseline_metrics_path.read_text())
    rule_test = baseline_metrics["metrics"]["TEST"]["dev_selected_window_rule"]
    rule_test_boot = baseline_metrics["clip_bootstrap"]["TEST"][
        "dev_selected_window_rule"
    ]["balanced_accuracy"]

    rule = pd.read_csv(rule_path)
    rule = rule[rule["split"] == "DEV"].reset_index(drop=True)
    cnn = pd.read_csv(cnn_path)
    if len(rule) != len(cnn):
        raise SystemExit(
            f"STOP: {len(rule)} rule rows vs {len(cnn)} CNN rows on DEV"
        )
    merged = rule.merge(
        cnn[["window_id", "oof_probability", "pred_at_0.5"]],
        on="window_id",
        validate="one_to_one",
    )
    if len(merged) != len(rule):
        raise SystemExit("STOP: window_id mismatch between rule and CNN predictions")

    labels = merged["label"].to_numpy(dtype=int)
    clips = merged["sample_id"].to_numpy()
    prevalence = float(labels.mean())

    methods = [
        {
            "name": f"DEV-selected\n{AXIS_LABEL[args.task]} rule",
            "kind": "in-sample",
            "colour": ORANGE,
            "scores": merged["rule_score"].to_numpy(dtype=float),
            "pred": merged["dev_selected_pred"].to_numpy(dtype=int),
        },
        {
            "name": "Pose CNN\nleave-one-clip-out",
            "kind": "out-of-fold",
            "colour": BLUE,
            "scores": merged["oof_probability"].to_numpy(dtype=float),
            "pred": merged["pred_at_0.5"].to_numpy(dtype=int),
        },
    ]
    for entry in methods:
        entry["pr_auc"] = average_precision(labels, entry["scores"])
        entry["pr_boot"] = clip_bootstrap_pr_auc(clips, labels, entry["scores"])
        entry["bacc"] = float(
            clip_binary_metrics(labels, entry["pred"])["balanced_accuracy"]
        )
        entry["bacc_boot"] = clip_bootstrap(clips, labels, entry["pred"])

    rows = [{"name": "Always yes\nor always no", "kind": "trivial", "colour": GREY}]
    rows += methods

    spans_zero = [
        e["pr_boot"]["pr_auc_minus_prevalence"]["ci_lower_95"] <= 0
        for e in methods
    ]
    if all(spans_zero):
        ranking_claim = "neither clears chance on ranking quality"
        panel_one_note = "both intervals contain chance"
    elif not any(spans_zero):
        ranking_claim = "both clear chance on ranking quality"
        panel_one_note = "neither interval contains chance"
    else:
        clear = next(
            e["name"].splitlines()[-1]
            for e, spans in zip(methods, spans_zero)
            if not spans
        )
        ranking_claim = f"only the {clear} clears chance on ranking quality"
        panel_one_note = "one interval contains chance"

    test_low = float(rule_test_boot["ci_lower_95"])
    test_high = float(rule_test_boot["ci_upper_95"])
    test_verdict = (
        "contains chance" if test_low <= 0.5 <= test_high else "excludes chance"
    )

    fig = plt.figure(figsize=(12.4, 7.9), facecolor=PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.26)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    for ax in (ax1, ax2):
        ax.set_facecolor(PAPER)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GREY)
        ax.spines["bottom"].set_color(GREY)
        ax.tick_params(colors=MUTED, labelsize=9)

    x = np.arange(len(rows))
    names = [f"{r['name']}\n({r['kind']})" for r in rows]
    hatches = ["//" if r["kind"] == "in-sample" else "" for r in rows]

    chance = ax1.axhline(prevalence, color=INK, lw=1.4, ls="--", zorder=3)
    ax1.bar(
        0,
        prevalence,
        width=0.58,
        color=GREY,
        edgecolor=PAPER,
        linewidth=1.0,
        zorder=2,
    )
    ax1.text(
        0,
        prevalence + 0.004,
        f"{prevalence:.3f}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=INK,
    )
    for i, entry in enumerate(methods, start=1):
        low = entry["pr_boot"]["pr_auc"]["ci_lower_95"]
        high = entry["pr_boot"]["pr_auc"]["ci_upper_95"]
        ax1.bar(
            i,
            entry["pr_auc"],
            width=0.58,
            color=entry["colour"],
            edgecolor=PAPER,
            linewidth=1.0,
            hatch=hatches[i],
            zorder=2,
        )
        ax1.errorbar(
            i,
            entry["pr_auc"],
            yerr=[[entry["pr_auc"] - low], [high - entry["pr_auc"]]],
            fmt="none",
            ecolor=INK,
            elinewidth=1.5,
            capsize=6,
            capthick=1.5,
            zorder=4,
        )
        ax1.text(
            i + 0.36,
            high,
            f"{entry['pr_auc']:.3f}\n[{low:.3f}, {high:.3f}]",
            ha="left",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax1.legend(
        [chance],
        [f"chance = prevalence {prevalence:.3f}"],
        loc="upper left",
        frameon=False,
        fontsize=9.5,
        labelcolor=INK,
        handlelength=2.4,
    )
    top1 = max(e["pr_boot"]["pr_auc"]["ci_upper_95"] for e in methods)
    ax1.set_xlim(-0.6, len(rows) - 0.05)
    ax1.set_ylim(0, top1 * 1.3)
    ax1.set_ylabel("PR AUC (threshold-free)", fontsize=10.5, color=INK)
    ax1.set_title(
        f"Ranking quality on DEV windows\n{panel_one_note}",
        fontsize=11.5,
        color=INK,
        loc="left",
        fontweight="600",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)

    floor = ax2.axhline(0.5, color=INK, lw=1.4, ls="--", zorder=3)
    ax2.bar(
        0,
        0.5,
        width=0.58,
        color=GREY,
        edgecolor=PAPER,
        linewidth=1.0,
        zorder=2,
    )
    ax2.text(0, 0.5 + 0.004, "0.500", ha="center", va="bottom", fontsize=9.5, color=INK)
    for i, entry in enumerate(methods, start=1):
        low = entry["bacc_boot"]["balanced_accuracy"]["ci_lower_95"]
        high = entry["bacc_boot"]["balanced_accuracy"]["ci_upper_95"]
        ax2.bar(
            i,
            entry["bacc"],
            width=0.58,
            color=entry["colour"],
            edgecolor=PAPER,
            linewidth=1.0,
            hatch=hatches[i],
            zorder=2,
        )
        ax2.errorbar(
            i,
            entry["bacc"],
            yerr=[[entry["bacc"] - low], [high - entry["bacc"]]],
            fmt="none",
            ecolor=INK,
            elinewidth=1.5,
            capsize=6,
            capthick=1.5,
            zorder=4,
        )
        ax2.text(
            i + 0.36,
            high,
            f"{entry['bacc']:.3f}\n[{low:.3f}, {high:.3f}]",
            ha="left",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax2.legend(
        [floor],
        ["floor = 0.500 (any constant predictor)"],
        loc="upper left",
        frameon=False,
        fontsize=9.5,
        labelcolor=INK,
        handlelength=2.4,
    )
    lows = [e["bacc_boot"]["balanced_accuracy"]["ci_lower_95"] for e in methods]
    highs = [e["bacc_boot"]["balanced_accuracy"]["ci_upper_95"] for e in methods]
    ax2.set_xlim(-0.6, len(rows) - 0.05)
    ax2.set_ylim(min(0.44, min(lows) - 0.02), max(highs) + 0.075)
    ax2.set_ylabel("Balanced accuracy", fontsize=10.5, color=INK)
    ax2.set_title(
        "Balanced accuracy on DEV windows\nhatched bar's threshold was fitted here",
        fontsize=11.5,
        color=INK,
        loc="left",
        fontweight="600",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)

    fig.suptitle(
        f"{args.task.capitalize()} at 3 s: the {AXIS_LABEL[args.task]} rule and the "
        f"pose CNN agree, and {ranking_claim}",
        fontsize=13.2,
        color=INK,
        fontweight="700",
        x=0.008,
        ha="left",
        y=0.985,
    )
    margins = ", ".join(
        f"{e['name'].splitlines()[0].lower()} "
        f"{e['pr_boot']['pr_auc_minus_prevalence']['ci_lower_95']:+.3f} to "
        f"{e['pr_boot']['pr_auc_minus_prevalence']['ci_upper_95']:+.3f}"
        for e in methods
    )
    fig.text(
        0.008,
        0.012,
        f"All values are DEV: {len(labels)} windows from {len(set(clips))} clips, "
        f"{int(labels.sum())} positive. Intervals are 95% percentile bootstrap "
        "over whole clips, 2000 resamples.\n"
        f"Intervals for PR AUC minus prevalence: {margins}.\n"
        "A constant predictor's PR AUC equals the prevalence, so the grey bar sits "
        "on the chance line by definition.\n"
        "The rule's threshold was fitted on these same windows, so its balanced "
        "accuracy is optimistic; the held-out TEST figure below the right panel is "
        "the one to quote.\n"
        "VideoMAE is not shown: those crops cropped the excluded person in 51 % of "
        f"windows, so whether RGB carries a {args.task} signal here is unresolved\n"
        "rather than answered.",
        fontsize=8.7,
        color=MUTED,
        ha="left",
        va="bottom",
        linespacing=1.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.34, left=0.075, right=0.985)
    box = ax2.get_position()
    fig.text(
        box.x0,
        0.205,
        "Same rule on held-out TEST: "
        f"{float(rule_test['balanced_accuracy']):.3f} "
        f"[{test_low:.3f}, {test_high:.3f}] — {test_verdict}",
        fontsize=9.2,
        color=INK,
        ha="left",
        va="bottom",
        fontweight="600",
        bbox=dict(
            facecolor="#f3ede1", edgecolor="#ded4c2", boxstyle="round,pad=0.42"
        ),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=PAPER)
    fig.savefig(out.with_suffix(".pdf"), facecolor=PAPER)
    plt.close(fig)

    summary = {
        "protocol": f"windowed_{args.task}_3s_dev_rule_vs_pose_cnn",
        "development_only": True,
        "n_windows": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_clips": int(len(set(clips))),
        "prevalence": prevalence,
        "methods": {
            entry["name"].replace("\n", " "): {
                "selection": entry["kind"],
                "pr_auc": entry["pr_auc"],
                "pr_auc_clip_bootstrap": entry["pr_boot"],
                "balanced_accuracy": entry["bacc"],
                "balanced_accuracy_clip_bootstrap": entry["bacc_boot"],
            }
            for entry in methods
        },
        "videomae_excluded": (
            "crops tracked the largest detected face and changed person in 14 of "
            "15 clips; see results/windowed_nod/crop_audit"
        ),
    }
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote {out.with_suffix('.png')}")
    print(f"      {out.with_suffix('.pdf')}")
    print(f"      {json_path}")
    print(f"prevalence {prevalence:.3f} over {len(labels)} windows "
          f"({int(labels.sum())} positive, {len(set(clips))} clips)")
    for entry in methods:
        pr = entry["pr_boot"]["pr_auc"]
        margin = entry["pr_boot"]["pr_auc_minus_prevalence"]
        bacc = entry["bacc_boot"]["balanced_accuracy"]
        print(f"  {entry['name'].replace(chr(10), ' '):34s} ({entry['kind']})")
        print(f"    PR AUC   {entry['pr_auc']:.3f} "
              f"[{pr['ci_lower_95']:.3f}, {pr['ci_upper_95']:.3f}]")
        print(f"    PR AUC - prevalence {margin['mean']:+.3f} "
              f"[{margin['ci_lower_95']:+.3f}, {margin['ci_upper_95']:+.3f}]")
        print(f"    bal acc  {entry['bacc']:.3f} "
              f"[{bacc['ci_lower_95']:.3f}, {bacc['ci_upper_95']:.3f}]")


if __name__ == "__main__":
    main()
