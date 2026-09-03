#!/usr/bin/env python3
"""Diagram of the 3 s RGB crop defect: where the crop box actually sat.

Every point is a real crop box from results/windowed_nod/crop_audit. The
annotator was told to watch one side of the frame; a box on the other side is
a crop of the person the label explicitly excludes.

Four clips are drawn to show the distinct failure modes, then a per-clip
summary. The VideoMAE runs trained on these crops are tabulated separately by
--write-table rather than plotted, since they measure the defect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "windowed_nod" / "crop_audit"
WATCH_LIST = ROOT / "data" / "gold" / "watch_list.csv"

INK = "#1d1d1f"
MUTED = "#5c5c63"
GREY = "#a8a8ad"
GREEN = "#1b7f4b"
RED = "#b5342a"
PAPER = "#fffdf8"
GOOD_BAND = "#e4f0e7"
BAD_BAND = "#fbeae7"

RUNS = [
    ("nod", "videomae_loco_dev", "Nod, 6 epochs"),
    ("nod", "videomae_loco_dev_ep12", "Nod, 12 epochs"),
    ("shake", "videomae_loco_dev", "Shake, 6 epochs"),
    ("shake", "videomae_loco_dev_ep12", "Shake, 12 epochs"),
]


def load_boxes(task: str) -> pd.DataFrame:
    path = AUDIT / f"crop_boxes_dev_{task}.csv"
    if not path.exists():
        raise SystemExit(f"STOP: missing {path}")
    boxes = pd.read_csv(path)
    watch = pd.read_csv(WATCH_LIST)
    watch["watch_side"] = (
        watch["who_to_watch"].astype(str).str.extract(r"^(LEFT|RIGHT)", expand=False)
    )
    boxes = boxes.merge(
        watch[["video_id", "watch_side"]], on="video_id", how="left", validate="many_to_one"
    )
    if boxes["watch_side"].isna().any():
        raise SystemExit("STOP: a clip has no LEFT/RIGHT watch side")
    if "frame_width" in boxes.columns and (boxes["frame_width"] > 0).all():
        boxes["width"] = boxes["frame_width"]
    else:
        boxes["width"] = int((boxes["crop_x0"] + boxes["crop_side"]).max())
    boxes["midline"] = boxes["width"] / 2.0
    boxes["wrong_half"] = np.where(
        boxes["watch_side"] == "LEFT",
        boxes["crop_centre_x"] >= boxes["midline"],
        boxes["crop_centre_x"] < boxes["midline"],
    ).astype(int)
    return boxes


def write_table(out_path: Path) -> None:
    rows = []
    for task, folder, label in RUNS:
        path = ROOT / "results" / f"windowed_{task}" / folder / "metrics_dev.json"
        if not path.exists():
            raise SystemExit(f"STOP: missing {path}")
        payload = json.loads(path.read_text())
        pr = float(payload["pr_auc_out_of_fold"])
        prevalence = float(payload["prevalence"])
        rows.append(
            {
                "run": label,
                "windows": int(payload["n_windows_scored"]),
                "positives": int(payload["n_positive"]),
                "pr_auc": pr,
                "chance": prevalence,
                "margin": pr - prevalence,
                "balanced_accuracy": float(
                    payload["at_fixed_threshold_0.5"]["balanced_accuracy"]
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_path.with_suffix(".csv"), index=False)
    lines = [
        "| Run | Windows | Positives | PR AUC | Chance | Margin | Balanced accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['windows']} | {r['positives']} | {r['pr_auc']:.3f} | "
            f"{r['chance']:.3f} | {r['margin']:+.3f} | {r['balanced_accuracy']:.3f} |"
        )
    lines.append("")
    lines.append(
        "Withdrawn. Every run was trained and scored on crops that followed the "
        "largest detected face, so they describe the crop defect rather than the "
        "RGB modality. Reported for completeness only."
    )
    out_path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path.with_suffix('.md')}")
    print(f"      {out_path.with_suffix('.csv')}")


def draw_trace(ax, group: pd.DataFrame) -> None:
    side = str(group["watch_side"].iloc[0])
    width = float(group["width"].iloc[0])
    midline = width / 2.0
    good = (0, midline) if side == "LEFT" else (midline, width)
    bad = (midline, width) if side == "LEFT" else (0, midline)
    ax.axhspan(*good, color=GOOD_BAND, zorder=0)
    ax.axhspan(*bad, color=BAD_BAND, zorder=0)
    ax.axhline(midline, color=INK, lw=1.1, zorder=2)

    t = group["start_sec"].to_numpy(dtype=float)
    cx = group["crop_centre_x"].to_numpy(dtype=float)
    wrong = group["wrong_half"].to_numpy(dtype=bool)
    positive = group["label"].to_numpy(dtype=int) == 1
    ax.plot(t, cx, color=GREY, lw=1.0, zorder=3)
    ax.scatter(
        t[~wrong], cx[~wrong], s=26, color=GREEN, zorder=4, label="annotated person's side"
    )
    ax.scatter(
        t[wrong], cx[wrong], s=42, color=RED, marker="X", zorder=5, label="wrong side"
    )
    ax.scatter(
        t[positive],
        cx[positive],
        s=104,
        facecolors="none",
        edgecolors=INK,
        linewidths=1.2,
        zorder=6,
        label="labelled positive",
    )

    n_wrong = int(wrong.sum())
    pct = 100.0 * n_wrong / len(group)
    ax.set_ylim(0, width)
    ax.set_xlim(float(t.min()) - 1.5, float(t.max()) + 1.5)
    ax.set_yticks([0, midline, width])
    ax.set_title(
        f"{group['sample_id'].iloc[0]} — watch {side} — "
        f"{n_wrong}/{len(group)} windows wrong ({pct:.0f}%)",
        fontsize=9.8,
        color=INK,
        loc="left",
        fontweight="600",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("nod", "shake"), default="nod")
    ap.add_argument("--out", type=Path, default=AUDIT / "rgb_crop_defect_dev")
    ap.add_argument(
        "--write-table",
        action="store_true",
        help="also write the withdrawn VideoMAE numbers as markdown and csv",
    )
    args = ap.parse_args()

    boxes = load_boxes(args.task)
    per_clip = (
        boxes.groupby("sample_id")
        .agg(
            watch_side=("watch_side", "first"),
            n_windows=("wrong_half", "size"),
            n_wrong=("wrong_half", "sum"),
            n_outliers=("box_is_outlier", "sum"),
        )
        .reset_index()
    )
    per_clip["pct_wrong"] = 100.0 * per_clip["n_wrong"] / per_clip["n_windows"]

    ranked = per_clip.sort_values("pct_wrong", ascending=False)
    chosen = list(ranked["sample_id"].head(3))
    clean = ranked["sample_id"].iloc[-1]
    if clean not in chosen:
        chosen.append(clean)

    fig = plt.figure(figsize=(14.6, 8.4), facecolor=PAPER)
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1.0, 1.0, 1.06], wspace=0.28, hspace=0.42
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax_sum = fig.add_subplot(gs[:, 2])
    for ax in axes + [ax_sum]:
        ax.set_facecolor(PAPER)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GREY)
        ax.spines["bottom"].set_color(GREY)
        ax.tick_params(colors=MUTED, labelsize=8.5)

    for ax, clip in zip(axes, chosen):
        draw_trace(ax, boxes[boxes["sample_id"] == clip].sort_values("start_sec"))
    axes[0].set_ylabel("crop box centre, x (px)", fontsize=9.5, color=INK)
    axes[2].set_ylabel("crop box centre, x (px)", fontsize=9.5, color=INK)
    axes[2].set_xlabel("window start (s)", fontsize=9.5, color=INK)
    axes[3].set_xlabel("window start (s)", fontsize=9.5, color=INK)

    y = np.arange(len(per_clip))[::-1]
    colours = [GREY if v == 0 else RED for v in per_clip["n_wrong"]]
    ax_sum.barh(
        y,
        per_clip["pct_wrong"],
        height=0.66,
        color=colours,
        edgecolor=PAPER,
        linewidth=1.0,
        zorder=2,
    )
    for pos, row in zip(y, per_clip.itertuples(index=False)):
        text = (
            "none"
            if row.n_wrong == 0
            else f"{row.n_wrong}/{row.n_windows}"
        )
        ax_sum.text(
            row.pct_wrong + 1.6,
            pos,
            text,
            ha="left",
            va="center",
            fontsize=8.4,
            color=INK if row.n_wrong else MUTED,
        )
    ax_sum.set_yticks(y)
    ax_sum.set_yticklabels(
        [
            f"{r.sample_id} ({'L' if r.watch_side == 'LEFT' else 'R'})"
            for r in per_clip.itertuples(index=False)
        ],
        fontsize=8.4,
    )
    ax_sum.set_xlim(0, 118)
    ax_sum.set_xticks([0, 25, 50, 75, 100])
    ax_sum.set_xlabel("windows cropped on the wrong side (%)", fontsize=9.5, color=INK)
    n_wrong_total = int(boxes["wrong_half"].sum())
    n_total = int(len(boxes))
    ax_sum.set_title(
        f"Every DEV clip but one\n{100 * n_wrong_total / n_total:.0f}% of all "
        "windows, the excluded person",
        fontsize=9.8,
        color=INK,
        loc="left",
        fontweight="600",
    )

    handles = [
        Line2D([], [], marker="o", ls="none", color=GREEN, markersize=6,
               label="annotated person's side"),
        Line2D([], [], marker="X", ls="none", color=RED, markersize=8,
               label="excluded person's side"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="none",
               markeredgecolor=INK, markersize=10, label="labelled positive"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.052, 0.952),
        frameon=False,
        fontsize=9.2,
        labelcolor=INK,
        ncol=3,
        columnspacing=1.6,
        handletextpad=0.4,
    )

    n_pos = int((boxes["label"] == 1).sum())
    n_pos_wrong = int(boxes.loc[boxes["label"] == 1, "wrong_half"].sum())
    fig.suptitle(
        "The 3 s RGB crops followed the largest detected face, so half of them "
        "show the person the annotator was told to ignore",
        fontsize=13.0,
        color=INK,
        fontweight="700",
        x=0.008,
        ha="left",
        y=0.982,
    )
    fig.text(
        0.008,
        0.012,
        "Each marker is one real crop box from results/windowed_nod/crop_audit. The green band is the half of the frame the annotator watched; "
        "the pink band is the half they were instructed to ignore.\n"
        f"{n_pos_wrong} of the {n_pos} labelled-positive windows crop the excluded person, so for those windows the label and the pixels describe different people.\n"
        f"{chosen[0]} shows the box sitting on the wrong person for almost the whole clip, which the earlier off-median test barely flagged: a stably wrong box has no outliers. "
        "That test counted 127 unstable windows;\nthe side test counts "
        f"{n_wrong_total}. Side-aware cropping fixes both by rejecting detections outside the green band and holding one box per clip.",
        fontsize=8.7,
        color=MUTED,
        ha="left",
        va="bottom",
        linespacing=1.5,
    )
    fig.subplots_adjust(top=0.845, bottom=0.145, left=0.055, right=0.988)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=200, facecolor=PAPER)
    fig.savefig(args.out.with_suffix(".pdf"), facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {args.out.with_suffix('.png')}")
    print(f"      {args.out.with_suffix('.pdf')}")
    print(f"wrong-side windows: {n_wrong_total}/{n_total} "
          f"({100 * n_wrong_total / n_total:.1f}%)")
    print(f"positives on the wrong side: {n_pos_wrong}/{n_pos}")
    print(f"clips affected: {int((per_clip['n_wrong'] > 0).sum())}/{len(per_clip)}")
    print(f"panels drawn: {', '.join(chosen)}")

    if args.write_table:
        write_table(AUDIT / "videomae_withdrawn_dev")


if __name__ == "__main__":
    main()
