#!/usr/bin/env python3
"""Quieter Overleaf figures: no 60 s labels, no withdrawn scores, no footer notes.

Reads stored metrics only. Does not score TEST.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paper_figure_style import (  # noqa: E402
    BLUE,
    FONT_NAME,
    GREEN,
    GREY,
    INK,
    MUTED,
    ORANGE,
    PALE,
    PAPER,
    SIZE_FULL,
    SIZE_FULL_TALL,
    forest,
    save,
)

OVERLEAF = Path("/Users/divyabisht/Downloads")
OUT = ROOT / "results" / "windowed_dev" / "overleaf_polish"
EVENTS = ROOT / "data" / "windowed_annotations" / "nod_events_windowed.csv"
WINDOWS = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"


def loadj(path: Path) -> dict:
    return json.loads(path.read_text())


def copy_to_overleaf(stem: Path, name: str) -> None:
    dest = OVERLEAF / name
    dest.write_bytes(stem.with_suffix(".png").read_bytes())
    print(f"copied {dest}")


def forest_clean(ax, rows: list[dict], *, title: str) -> None:
    cleaned = []
    for row in rows:
        item = dict(row)
        if item.get("lo") is None:
            item["lo"] = item["ba"]
            item["hi"] = item["ba"]
        cleaned.append(item)
    forest(ax, cleaned, title=title)


def fig_label_counts(stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.4), facecolor=PAPER)
    gestures = ["Nod", "Shake"]
    dev = [52, 39]
    test = [69, 40]
    x = range(len(gestures))
    w = 0.28
    ax.bar([i - w / 2 for i in x], dev, width=w, color=BLUE, label="Development")
    ax.bar([i + w / 2 for i in x], test, width=w, color=ORANGE, label="Test")
    ax.set_xticks(list(x), gestures)
    ax.set_ylabel("Positive windows")
    ax.set_ylim(0, 85)
    ax.legend(frameon=False)
    for i, (a, b) in enumerate(zip(dev, test)):
        ax.text(i - w / 2, a + 1.5, str(a), ha="center", fontsize=8, color=INK)
        ax.text(i + w / 2, b + 1.5, str(b), ha="center", fontsize=8, color=INK)
    ax.set_title("Positive windows on the 3 s protocol")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.16)
    save(fig, stem)


def fig_labels_by_split(stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.4), facecolor=PAPER)
    labels = ["Nod DEV", "Nod TEST", "Shake DEV", "Shake TEST"]
    prev = [0.120, 0.159, 0.090, 0.092]
    colours = [BLUE, ORANGE, BLUE, ORANGE]
    ax.bar(labels, prev, width=0.45, color=colours)
    ax.set_ylabel("Prevalence")
    ax.set_ylim(0, 0.22)
    ax.axhline(0, color=GREY, lw=0.6)
    for i, v in enumerate(prev):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_title("Positive rate by gesture and split")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.18)
    save(fig, stem)


def fig_sliding_windows(stem: Path) -> None:
    events = pd.read_csv(EVENTS)
    windows = pd.read_csv(WINDOWS)
    ev = events[events["sample_id"] == "gold_001"].sort_values("start_sec")
    ww = windows[windows["sample_id"] == "gold_001"].sort_values("start_sec")

    fig = plt.figure(figsize=(7.16, 6.4), facecolor=PAPER)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 2.4], hspace=0.32)
    ax_t = fig.add_subplot(gs[0])
    ax_w = fig.add_subplot(gs[1])
    ax_t.set_facecolor(PAPER)
    ax_w.set_facecolor(PAPER)
    ax_t.axis("off")
    ax_w.axis("off")

    ax_t.set_xlim(0, 59)
    ax_t.set_ylim(0, 3.0)
    ax_t.set_title("gold 001, annotated nods", loc="left", pad=6)
    ax_t.plot([0, 59], [1.05, 1.05], color=INK, lw=1.2)
    for t in range(0, 60, 10):
        ax_t.plot([t, t], [1.05, 1.22], color=INK, lw=0.9)
        ax_t.text(t, 0.68, f"{t} s", ha="center", fontsize=8, color=MUTED)
    for i, r in enumerate(ev.itertuples(index=False), start=1):
        s, e = float(r.start_sec), float(r.end_sec)
        ax_t.axvspan(s, max(e, s + 0.35), ymin=0.38, ymax=0.58, color=ORANGE, alpha=0.9)
        ax_t.text((s + e) / 2, 2.15, f"{s:.1f}–{e:.1f} s", ha="center", fontsize=8, color=ORANGE)

    ax_w.set_xlim(0, 10)
    ax_w.set_ylim(0, 7.6)
    ax_w.set_title("3 s windows, stride 2 s", loc="left", pad=4)
    cols, box_w, box_h, gap_x, gap_y = 5, 1.78, 0.92, 0.16, 0.16
    for idx, r in enumerate(ww.itertuples(index=False)):
        col, row = idx % cols, idx // cols
        x = 0.08 + col * (box_w + gap_x)
        y = 6.35 - row * (box_h + gap_y)
        yes = int(r.label) == 1
        ax_w.add_patch(
            FancyBboxPatch(
                (x, y),
                box_w,
                box_h,
                boxstyle="round,pad=0.01,rounding_size=0.06",
                facecolor="#d8efe3" if yes else "#f3f3f4",
                edgecolor=GREEN if yes else GREY,
                linewidth=1.1 if yes else 0.7,
            )
        )
        ax_w.text(
            x + box_w / 2,
            y + 0.55,
            f"{r.start_sec:g}–{r.end_sec:g} s",
            ha="center",
            fontsize=8,
            color=INK,
        )
        ax_w.text(
            x + box_w / 2,
            y + 0.22,
            "Positive" if yes else "Negative",
            ha="center",
            fontsize=7.5,
            color=GREEN if yes else MUTED,
        )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.04)
    save(fig, stem)


def fig_pseudo_labels(stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4), facecolor=PAPER)
    x = range(2)
    w = 0.28
    pos = [70, 75]
    neg = [10, 5]
    ax.bar([i - w / 2 for i in x], pos, width=w, color=BLUE, label="Positive")
    ax.bar([i + w / 2 for i in x], neg, width=w, color=GREY, label="Negative")
    ax.set_xticks(list(x), ["Nod", "Shake"])
    ax.set_ylabel("Clips")
    ax.set_ylim(0, 90)
    ax.legend(frameon=False)
    for i, (a, b) in enumerate(zip(pos, neg)):
        ax.text(i - w / 2, a + 1.5, str(a), ha="center", fontsize=8)
        ax.text(i + w / 2, b + 1.5, str(b), ha="center", fontsize=8)
    ax.set_title("Clip-level pseudo labels, 80 conversations")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.16)
    save(fig, stem)


FACE = dict(family=FONT_NAME, weight="normal")
FP_TITLE = FontProperties(size=11, **FACE)
FP_LETTER = FontProperties(size=10, **FACE)
FP_HEAD = FontProperties(size=7.5, **FACE)
FP_BOX = FontProperties(size=9, **FACE)
FP_BODY = FontProperties(size=8, **FACE)


def box(ax, x, y, w, h, title, body="", *, fc=PALE, ec=INK, title_c=INK, title_ha="left", body_ha="left"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=0.85,
            facecolor=fc,
            edgecolor=ec,
            transform=ax.transAxes,
            clip_on=False,
            zorder=2,
        )
    )
    tx = x + w / 2 if title_ha == "center" else x + 0.018
    bx = x + w / 2 if body_ha == "center" else x + 0.018
    ax.text(
        tx,
        y + h - 0.036,
        title,
        transform=ax.transAxes,
        color=title_c,
        fontproperties=FP_BOX,
        va="top",
        ha=title_ha,
        parse_math=False,
        zorder=3,
    )
    for i, line in enumerate(body.split("\n") if body else []):
        ax.text(
            bx,
            y + h - 0.068 - i * 0.026,
            line,
            transform=ax.transAxes,
            color=MUTED,
            fontproperties=FP_BODY,
            va="top",
            ha=body_ha,
            parse_math=False,
            zorder=3,
        )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=INK,
            connectionstyle="arc3,rad=0",
            shrinkA=0,
            shrinkB=1.5,
            clip_on=False,
            zorder=1,
        )
    )


def fig_study_design(stem: Path) -> None:
    """Gold forks to DEV and TEST; both splits then share the 3 s protocol."""
    fig = plt.figure(figsize=(7.16, 6.55), facecolor=PAPER)
    ax = fig.add_axes((0.02, 0.03, 0.96, 0.86))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.20,
        1.055,
        "Study design",
        fontproperties=FP_TITLE,
        color=INK,
        va="bottom",
        ha="left",
        clip_on=False,
    )

    def rail(y, letter, name):
        ax.text(0.0, y, letter, fontproperties=FP_LETTER, color=MUTED, va="top", ha="left")
        ax.text(0.0, y - 0.032, name, fontproperties=FP_HEAD, color=MUTED, va="top", ha="left")

    lx, bw, rx = 0.20, 0.375, 0.625
    lc, rc = lx + bw / 2, rx + bw / 2
    span_x, span_w = 0.20, 0.80

    rail(0.978, "A", "Data and splits")
    box(
        ax, 0.40, 0.850, 0.40, 0.110,
        "Thirty annotated clips",
        "RealTalk. Listener nods and head shakes.",
    )
    box(
        ax, lx, 0.675, bw, 0.130,
        "Development, gold 001–015",
        "Thresholds and model selection.\nLeave-one-clip-out where needed.",
        fc="#e8eef4",
        ec=BLUE,
        title_c=BLUE,
    )
    box(
        ax, rx, 0.675, bw, 0.130,
        "Test, gold 016–030",
        "Scored once after development\nchoices are frozen.",
        fc="#f4eee8",
        ec=ORANGE,
        title_c=ORANGE,
    )
    arrow(ax, 0.50, 0.846, lc, 0.807)
    arrow(ax, 0.70, 0.846, rc, 0.807)

    rail(0.638, "B", "Shared protocol")
    box(
        ax, span_x, 0.478, span_w, 0.145,
        "3 s windows, 2 s stride, on both splits",
        "15 × 29 = 435 windows per split. Nod DEV 52 positive; shake DEV 39.\nEMOCA head rotation. RGB face crops of the listener.",
        title_ha="center",
        body_ha="center",
    )
    arrow(ax, lc, 0.675, lc, 0.625)
    arrow(ax, rc, 0.675, rc, 0.625)

    rail(0.445, "C", "Systems")
    box(
        ax, lx, 0.255, bw, 0.165,
        "Pose",
        "Amplitude and return-ratio rules.\n1-D CNN; multiple-instance model.",
        fc="#e8eef4",
        ec=BLUE,
        title_c=BLUE,
    )
    box(
        ax, rx, 0.255, bw, 0.165,
        "Video",
        "VideoMAE on identity-fixed crops.\nDevelopment only; test not opened.",
        fc="#eef4ee",
        ec=GREEN,
        title_c=GREEN,
    )
    arrow(ax, lc, 0.478, lc, 0.422)
    arrow(ax, rc, 0.478, rc, 0.422)

    rail(0.222, "D", "Evaluation")
    box(ax, span_x, 0.018, span_w, 0.185, "", "")
    ax.text(
        lc, 0.150, "Headline metric",
        transform=ax.transAxes, color=INK, fontproperties=FP_BOX, va="top", ha="center", zorder=3,
    )
    ax.text(
        lc, 0.120, "Balanced accuracy.",
        transform=ax.transAxes, color=MUTED, fontproperties=FP_BODY, va="top", ha="center", zorder=3,
    )
    ax.text(
        lc, 0.094, "A constant predictor scores 0.500.",
        transform=ax.transAxes, color=MUTED, fontproperties=FP_BODY, va="top", ha="center", zorder=3,
    )
    ax.text(
        rc, 0.150, "Uncertainty",
        transform=ax.transAxes, color=INK, fontproperties=FP_BOX, va="top", ha="center", zorder=3,
    )
    ax.text(
        rc, 0.120, "95% intervals resample 15 clips.",
        transform=ax.transAxes, color=MUTED, fontproperties=FP_BODY, va="top", ha="center", zorder=3,
    )
    ax.text(
        rc, 0.094, "An interval containing 0.500 is",
        transform=ax.transAxes, color=MUTED, fontproperties=FP_BODY, va="top", ha="center", zorder=3,
    )
    ax.text(
        rc, 0.068, "not distinguished from chance.",
        transform=ax.transAxes, color=MUTED, fontproperties=FP_BODY, va="top", ha="center", zorder=3,
    )
    arrow(ax, lc, 0.255, lc, 0.205)
    arrow(ax, rc, 0.255, rc, 0.205)

    save(fig, stem)


def fig_both_gestures(stem: Path) -> None:
    nod_rule = loadj(ROOT / "results" / "windowed_nod" / "baselines_bacc" / "metrics.json")
    shake_rule = loadj(ROOT / "results" / "windowed_shake" / "baselines_bacc" / "metrics.json")
    nod_cnn = loadj(ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev" / "metrics_dev.json")
    shake_cnn = loadj(ROOT / "results" / "windowed_shake" / "pose_cnn_loco_dev" / "metrics_dev.json")
    rr = loadj(ROOT / "results" / "windowed_test" / "rule_return_ratio_final" / "metrics.json")
    frozen = loadj(ROOT / "results" / "windowed_dev" / "videomae_identity_fixed" / "frozen_encoder" / "metrics.json")
    last2 = loadj(ROOT / "results" / "windowed_dev" / "videomae_identity_fixed" / "last_blocks_unfrozen" / "metrics.json")

    def rule_ci(metrics: dict, split: str) -> tuple[float, float, float]:
        block = metrics["metrics"][split]["dev_selected_window_rule"]
        boot = metrics["clip_bootstrap"][split]["dev_selected_window_rule"]["balanced_accuracy"]
        return float(block["balanced_accuracy"]), float(boot["ci_lower_95"]), float(boot["ci_upper_95"])

    def loco(metrics: dict) -> tuple[float, float, float]:
        ba = float(metrics["at_fixed_threshold_0.5"]["balanced_accuracy"])
        boot = metrics["clip_bootstrap_at_0.5"]["balanced_accuracy"]
        return ba, float(boot["ci_lower_95"]), float(boot["ci_upper_95"])

    def fixed(metrics: dict) -> tuple[float, float, float]:
        ba = float(metrics["balanced_accuracy"])
        boot = metrics["clip_bootstrap"]["balanced_accuracy"]
        return ba, float(boot["ci_lower_95"]), float(boot["ci_upper_95"])

    sba, slo, shi = rule_ci(shake_rule, "TEST")
    sdba, sdlo, sdhi = rule_ci(shake_rule, "DEV")
    scba, sclo, schi = loco(shake_cnn)
    nba, nlo, nhi = rule_ci(nod_rule, "TEST")
    ncba, nclo, nchi = loco(nod_cnn)
    rba = float(rr["TEST"]["metrics"]["balanced_accuracy"])
    rlo = float(rr["TEST"]["clip_bootstrap"]["balanced_accuracy"]["ci_lower_95"])
    rhi = float(rr["TEST"]["clip_bootstrap"]["balanced_accuracy"]["ci_upper_95"])
    fba, flo, fhi = fixed(frozen)
    lba, llo, lhi = fixed(last2)

    shake = [
        {"name": "Yaw rule, test", "ba": sba, "lo": slo, "hi": shi, "colour": ORANGE, "locked": True},
        {"name": "Yaw rule, development", "ba": sdba, "lo": sdlo, "hi": sdhi, "colour": GREY},
        {"name": "Pose CNN, development", "ba": scba, "lo": sclo, "hi": schi, "colour": BLUE},
    ]
    nod = [
        {"name": "Return-ratio rule, test", "ba": rba, "lo": rlo, "hi": rhi, "colour": ORANGE, "locked": True},
        {"name": "Amplitude rule, test", "ba": nba, "lo": nlo, "hi": nhi, "colour": GREY, "locked": True},
        {"name": "Pose CNN, development", "ba": ncba, "lo": nclo, "hi": nchi, "colour": BLUE},
        {"name": "VideoMAE, 3 s, development", "ba": lba, "lo": llo, "hi": lhi, "colour": GREEN},
        {"name": "VideoMAE frozen, development", "ba": fba, "lo": flo, "hi": fhi, "colour": GREEN},
    ]
    fig, axes = plt.subplots(2, 1, figsize=SIZE_FULL_TALL, facecolor=PAPER, gridspec_kw={"hspace": 0.38})
    forest_clean(axes[0], shake, title="A  Head shakes")
    forest_clean(axes[1], nod, title="B  Head nods")
    fig.subplots_adjust(left=0.38, right=0.97, top=0.95, bottom=0.08)
    save(fig, stem)


def fig_rule_dev_test(stem: Path) -> None:
    rr = loadj(ROOT / "results" / "windowed_test" / "rule_return_ratio_final" / "metrics.json")
    labels = ["Development", "Test"]
    ba = [
        float(rr["DEV"]["metrics"]["balanced_accuracy"]),
        float(rr["TEST"]["metrics"]["balanced_accuracy"]),
    ]
    lo = [
        float(rr["DEV"]["clip_bootstrap"]["balanced_accuracy"]["ci_lower_95"]),
        float(rr["TEST"]["clip_bootstrap"]["balanced_accuracy"]["ci_lower_95"]),
    ]
    hi = [
        float(rr["DEV"]["clip_bootstrap"]["balanced_accuracy"]["ci_upper_95"]),
        float(rr["TEST"]["clip_bootstrap"]["balanced_accuracy"]["ci_upper_95"]),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.8), facecolor=PAPER)
    xs = [0, 1]
    ax.axhline(0.5, color=INK, ls="--", lw=1.0, zorder=0)
    ax.bar(xs, ba, color=[GREY, BLUE], width=0.32, zorder=2)
    ax.errorbar(
        xs,
        ba,
        yerr=[[a - b for a, b in zip(ba, lo)], [a - b for a, b in zip(hi, ba)]],
        fmt="none",
        ecolor=INK,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.40, 0.82)
    ax.set_title("Return-ratio rule")
    ax.text(1.42, 0.505, "0.500", color=MUTED, ha="right", va="bottom", fontsize=8)
    for x, v in zip(xs, ba):
        ax.text(x, v + 0.018, f"{v:.3f}", ha="center", fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.96, top=0.88, bottom=0.16)
    save(fig, stem)


def fig_cnn_ablations(stem: Path) -> None:
    locked = loadj(ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev" / "metrics_dev.json")
    rr = loadj(ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev_return_ratio" / "metrics_dev.json")
    scalar = loadj(ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev_scalar_branch" / "metrics_dev.json")
    k11 = loadj(ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev_rf_ablation" / "k11_9_7" / "metrics_dev.json")
    k21 = loadj(ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev_rf_ablation" / "k21_15_13" / "metrics_dev.json")

    def loco(metrics: dict) -> tuple[float, float, float]:
        ba = float(metrics["at_fixed_threshold_0.5"]["balanced_accuracy"])
        boot = metrics["clip_bootstrap_at_0.5"]["balanced_accuracy"]
        return ba, float(boot["ci_lower_95"]), float(boot["ci_upper_95"])

    rows = []
    names = [
        ("Locked network, RF 0.44 s", locked, BLUE),
        ("Return ratio as input", rr, GREY),
        ("Return ratio as scalar", scalar, GREY),
        ("Wider kernels, RF 1.00 s", k11, ORANGE),
        ("Wider kernels, RF 1.88 s", k21, ORANGE),
    ]
    for name, payload, colour in names:
        ba, lo, hi = loco(payload)
        rows.append({"name": name, "ba": ba, "lo": lo, "hi": hi, "colour": colour})
    fig, ax = plt.subplots(figsize=SIZE_FULL, facecolor=PAPER)
    forest_clean(ax, rows, title="Pose network, development")
    fig.subplots_adjust(left=0.38, right=0.97, top=0.88, bottom=0.14)
    save(fig, stem)


def fig_weak_supervision(stem: Path) -> None:
    mil = loadj(ROOT / "results" / "windowed_nod" / "pose_mil_pseudo80_trainsel" / "metrics_dev.json")
    bag = float(mil["train_oof_bag_balanced_accuracy"])
    win = float(mil["dev_window"]["balanced_accuracy"])
    lo = float(mil["dev_clip_bootstrap"]["balanced_accuracy"]["ci_lower_95"])
    hi = float(mil["dev_clip_bootstrap"]["balanced_accuracy"]["ci_upper_95"])
    rows = [
        {
            "name": "Clip-level agreement\nwith the teacher",
            "ba": bag,
            "lo": None,
            "hi": None,
            "colour": BLUE,
            "label": f"{bag:.3f}",
        },
        {
            "name": "Window-level\nlocalisation",
            "ba": win,
            "lo": lo,
            "hi": hi,
            "colour": ORANGE,
            "label": f"{win:.3f}  [{lo:.3f}, {hi:.3f}]",
        },
    ]
    fig, ax = plt.subplots(figsize=(7.16, 3.15), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.axvline(0.5, color=INK, lw=1.0, ls="--", zorder=0)
    n = len(rows)
    for i, row in enumerate(rows):
        y = n - 1 - i
        if row["lo"] is not None:
            ax.plot(
                [row["lo"], row["hi"]],
                [y, y],
                color=row["colour"],
                lw=1.8,
                solid_capstyle="butt",
                zorder=2,
            )
        ax.plot(
            row["ba"],
            y,
            "o",
            color=row["colour"],
            markersize=7,
            markeredgecolor=PAPER,
            markeredgewidth=0.5,
            zorder=3,
        )
        ax.text(0.98, y, row["label"], va="center", ha="left", color=INK, clip_on=False)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([rows[0]["name"], rows[1]["name"]])
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlim(0.38, 1.18)
    ax.set_ylim(-0.55, 1.45)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("Multiple-instance model", loc="left", pad=6)
    ax.text(0.50, 1.32, "chance  0.500", color=MUTED, ha="center", va="bottom")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    fig.subplots_adjust(left=0.28, right=0.78, top=0.84, bottom=0.22)
    save(fig, stem)


def fig_temporal(stem: Path) -> None:
    last2 = loadj(ROOT / "results" / "windowed_dev" / "videomae_identity_fixed" / "last_blocks_unfrozen" / "metrics.json")
    one = loadj(ROOT / "results" / "windowed_dev" / "videomae_identity_fixed_1p5s" / "last_blocks_unfrozen" / "metrics.json")

    def fixed(metrics: dict) -> tuple[float, float, float]:
        ba = float(metrics["balanced_accuracy"])
        boot = metrics["clip_bootstrap"]["balanced_accuracy"]
        return ba, float(boot["ci_lower_95"]), float(boot["ci_upper_95"])

    a, lo, hi = fixed(last2)
    b, blo, bhi = fixed(one)
    rows = [
        {"name": "3.0 s, 5.3 frames/s", "ba": a, "lo": lo, "hi": hi, "colour": GREEN},
        {"name": "1.5 s, 10.7 frames/s", "ba": b, "lo": blo, "hi": bhi, "colour": ORANGE},
    ]
    fig, ax = plt.subplots(figsize=(7.16, 3.4), facecolor=PAPER)
    forest_clean(ax, rows, title="Identity-fixed VideoMAE")
    fig.subplots_adjust(left=0.32, right=0.97, top=0.86, bottom=0.20)
    save(fig, stem)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [
        (fig_label_counts, "label_counts", "label_counts.png"),
        (fig_labels_by_split, "labels_by_split", "labels_by_split.png"),
        (fig_sliding_windows, "sliding_windows", "sliding_windows.png"),
        (fig_pseudo_labels, "pseudo_labels", "pseudo_labels.png"),
        (fig_study_design, "study_design", "study_design.png"),
        (fig_both_gestures, "results_both_gestures", "results_both_gestures.png"),
        (fig_rule_dev_test, "rule_dev_vs_test", "rule_dev_vs_test.png"),
        (fig_cnn_ablations, "cnn_ablations", "cnn_ablations.png"),
        (fig_weak_supervision, "weak_supervision", "weak_supervision.png"),
        (fig_temporal, "temporal_sampling", "temporal_sampling.png"),
    ]
    for fn, stem_name, dest in jobs:
        stem = OUT / stem_name
        fn(stem)
        copy_to_overleaf(stem, dest)


if __name__ == "__main__":
    main()
