#!/usr/bin/env python3
"""Publication figures for the DEV nod diagnostic and 1.5 s ablation.

Reads stored metrics only. Does not score TEST. Does not overwrite the
3 s identity-fixed metrics directories.

    python3 scripts/plot_nod_final_figures.py
"""
from __future__ import annotations

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

from src.windowed_protocol import (  # noqa: E402
    EVENTS_CSV,
    TEST_SAMPLE_IDS,
    load_events,
)

OUT = ROOT / "results" / "windowed_dev" / "final_figures"
FIXED = ROOT / "results" / "windowed_dev" / "videomae_identity_fixed"
FIXED_1P5 = ROOT / "results" / "windowed_dev" / "videomae_identity_fixed_1p5s"
WIN = ROOT / "results" / "windowed_nod"
WINDOWS = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
POSE_DIR = ROOT / "features" / "gold"

INK = "#1d1d1f"
MUTED = "#5c5c63"
GREY = "#b0b0b4"
ORANGE = "#c46a2d"
BLUE = "#2c5f8a"
GREEN = "#2f6b45"
PAPER = "#ffffff"


def loadj(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"STOP: missing {path}")
    return json.loads(path.read_text())


def maybe(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=PAPER)
    fig.savefig(stem.with_suffix(".svg"), facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.png')}")


def ci_of(metrics: dict) -> tuple[float, float, float]:
    ba = float(metrics["balanced_accuracy"])
    boot = metrics.get("clip_bootstrap") or {}
    block = boot.get("balanced_accuracy") or {}
    lo = float(block.get("ci_lower_95", ba))
    hi = float(block.get("ci_upper_95", ba))
    return ba, lo, hi


def figure_a(rows: list[dict], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.axvline(0.5, color=INK, lw=1.0, ls="--")
    for i, row in enumerate(rows):
        y = -i
        ax.plot(row["ba"], y, "o", color=row["colour"], markersize=8)
        ax.plot([row["lo"], row["hi"]], [y, y], color=row["colour"], lw=1.8)
        ax.text(0.18, y, row["name"], va="center", ha="left", fontsize=9, color=INK)
        ax.text(
            0.82, y,
            f"{row['ba']:.3f}  [{row['lo']:.3f}, {row['hi']:.3f}]",
            va="center",
            ha="left",
            fontsize=8.6,
            color=INK,
        )
    ax.set_xlim(0.16, 1.08)
    ax.set_ylim(-len(rows) + 0.45, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("Balanced accuracy (chance = 0.500)")
    ax.set_title("DEV nod detection, identity-fixed RGB and pose systems")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.text(
        0.01, 0.02,
        "Error bars are 95% clip-level bootstrap intervals. "
        "An interval that includes 0.500 is not distinguished from chance.",
        fontsize=8, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, stem)


def figure_b(three: dict, one: dict | None, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    items = [
        ("3.0 s / 16 frames\n(5.3 frames/s)", three),
    ]
    if one is not None:
        items.append(("1.5 s / 16 frames\n(10.7 frames/s)", one))
    xs = np.arange(len(items))
    ba = [it[1]["ba"] for it in items]
    lo = [it[1]["lo"] for it in items]
    hi = [it[1]["hi"] for it in items]
    ax.axhline(0.5, color=INK, lw=1.0, ls="--")
    ax.bar(xs, ba, color=GREEN, width=0.45, edgecolor=PAPER)
    ax.errorbar(xs, ba, yerr=[np.array(ba) - np.array(lo), np.array(hi) - np.array(ba)],
                fmt="none", ecolor=INK, capsize=4)
    ax.set_xticks(xs, [it[0] for it in items])
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.35, 0.85)
    ax.set_title("Temporal sampling ablation, identity-fixed VideoMAE")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if one is None:
        ax.text(0.5, 0.45, "1.5 s result not yet on disk", ha="center", color=MUTED)
    fig.tight_layout()
    save(fig, stem)


def figure_c(matrices: list[tuple[str, dict]], stem: Path) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.0 * len(matrices), 3.6), facecolor=PAPER)
    if len(matrices) == 1:
        axes = [axes]
    for ax, (title, cm) in zip(axes, matrices):
        grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=int)
        ax.imshow(grid, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1], ["Pred. 0", "Pred. 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(title, fontsize=10)
    fig.suptitle("Out-of-fold confusion on DEV", fontsize=12)
    fig.tight_layout()
    save(fig, stem)


def pr_points(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    rec = tp / max(y.sum(), 1)
    prec = tp / np.maximum(tp + fp, 1)
    return rec, prec


def figure_d(curves: list[tuple[str, np.ndarray, np.ndarray]], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    colours = [BLUE, GREEN, ORANGE, GREY]
    for (name, rec, prec), colour in zip(curves, colours):
        ax.plot(rec, prec, color=colour, lw=1.6, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision-recall, identity-fixed VideoMAE")
    ax.legend(frameon=False, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, stem)


def figure_e(stem: Path, rgb_preds: Path, clips: list[str]) -> None:
    if not rgb_preds.exists():
        return
    pred = pd.read_csv(rgb_preds)
    pred["sample_id"] = pred["sample_id"].astype(str)
    if set(pred["sample_id"]) & set(TEST_SAMPLE_IDS):
        raise SystemExit("STOP: TEST id in VideoMAE predictions")
    windows = pd.read_csv(WINDOWS)
    windows["sample_id"] = windows["sample_id"].astype(str)
    events = load_events(EVENTS_CSV, allow_test=False)
    pred = pred.merge(
        windows[["window_id", "start_sec", "end_sec"]],
        on="window_id",
        how="left",
    )
    fig, axes = plt.subplots(len(clips), 1, figsize=(9.4, 2.6 * len(clips)), facecolor=PAPER)
    if len(clips) == 1:
        axes = [axes]
    for ax, sid in zip(axes, clips):
        ax.set_facecolor(PAPER)
        part = pred[pred["sample_id"] == sid].sort_values("start_sec")
        if part.empty:
            ax.set_title(f"{sid}: no predictions")
            continue
        mid = 0.5 * (part["start_sec"] + part["end_sec"])
        prob_col = "oof_probability" if "oof_probability" in part.columns else "pred_at_0.5"
        ax.step(mid, part[prob_col], where="mid", color=GREEN, lw=1.4, label="VideoMAE P(nod)")
        pred_col = "pred_at_0.5" if "pred_at_0.5" in part.columns else "pred"
        pos = part[part[pred_col] == 1]
        for rec in pos.itertuples(index=False):
            ax.axvspan(float(rec.start_sec), float(rec.end_sec), color=GREEN, alpha=0.12)
        for ev in events[events["sample_id"] == sid].itertuples(index=False):
            ax.axvspan(float(ev.start_sec), float(ev.end_sec), color=ORANGE, alpha=0.22, zorder=0)
        pose_path = POSE_DIR / f"{sid}.npz"
        if pose_path.exists():
            with np.load(pose_path, allow_pickle=True) as payload:
                pitch = np.asarray(payload["rotation_xyz"], dtype=float)[:, 0]
            t = np.arange(len(pitch)) / 25.0
            scale = np.nanmax(np.abs(pitch)) or 1.0
            ax.plot(t, 0.15 * pitch / scale, color=MUTED, lw=0.8, alpha=0.8)
        ax.set_xlim(0, 60)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(nod)")
        ax.set_title(sid, loc="left", fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "DEV clip timelines. Orange: annotated nod. Green: VideoMAE positive windows.",
        fontsize=11, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, stem)


def figure_f(rgb_dir: Path | None, manifest_path: Path, stem: Path) -> None:
    if rgb_dir is None or not rgb_dir.exists() or not manifest_path.exists():
        return
    manifest = pd.read_csv(manifest_path)
    resolved = manifest[manifest["crop_status"] == "resolved"].copy()
    if resolved.empty:
        return
    picked = []
    for sid, group in resolved.groupby("sample_id"):
        picked.append(group.iloc[len(group) // 2])
    picked = pd.DataFrame(picked).head(12)
    n = len(picked)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10.4, 2.4 * rows), facecolor=PAPER)
    axes = np.atleast_2d(axes)
    for ax, rec in zip(axes.ravel(), picked.itertuples(index=False)):
        path = rgb_dir / f"{rec.window_id}.npz"
        ax.set_xticks([])
        ax.set_yticks([])
        if path.exists():
            with np.load(path, allow_pickle=True) as payload:
                ax.imshow(payload["rgb"][len(payload["rgb"]) // 2])
                side = str(payload["watch_side"]) if "watch_side" in payload else rec.watch_side
        else:
            ax.text(0.5, 0.5, "missing", ha="center")
            side = rec.watch_side
        ax.set_title(f"{rec.sample_id}  {side}", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Identity-fixed crops. Label is the annotator watch side.", fontsize=11)
    fig.tight_layout()
    save(fig, stem)


def main() -> None:
    frozen = loadj(FIXED / "frozen_encoder" / "metrics.json")
    last2 = loadj(FIXED / "last_blocks_unfrozen" / "metrics.json")
    loco = loadj(WIN / "pose_cnn_loco_dev" / "metrics_dev.json")
    mil = loadj(WIN / "pose_mil_pseudo80_trainsel" / "metrics_dev.json")
    one = maybe(FIXED_1P5 / "last_blocks_unfrozen" / "metrics.json")
    one_thr = maybe(FIXED_1P5 / "last_blocks_unfrozen_train_threshold" / "metrics.json")

    pose_ba = float(loco["at_fixed_threshold_0.5"]["balanced_accuracy"])
    pose_lo = float(loco["clip_bootstrap_at_0.5"]["balanced_accuracy"]["ci_lower_95"])
    pose_hi = float(loco["clip_bootstrap_at_0.5"]["balanced_accuracy"]["ci_upper_95"])
    mil_ba = float(mil["dev_window"]["balanced_accuracy"])
    mil_boot = mil.get("dev_clip_bootstrap", {}).get("balanced_accuracy", {})
    rows = [
        {"name": "Chance", "ba": 0.5, "lo": 0.5, "hi": 0.5, "colour": GREY},
        {"name": "Pose CNN, leave-one-clip-out", "ba": pose_ba, "lo": pose_lo, "hi": pose_hi, "colour": BLUE},
        {
            "name": "Pose MIL, TRAIN-selected",
            "ba": mil_ba,
            "lo": float(mil_boot.get("ci_lower_95", mil_ba)),
            "hi": float(mil_boot.get("ci_upper_95", mil_ba)),
            "colour": BLUE,
        },
    ]
    fba, flo, fhi = ci_of(frozen)
    lba, llo, lhi = ci_of(last2)
    rows.append({"name": "VideoMAE frozen, 3.0 s", "ba": fba, "lo": flo, "hi": fhi, "colour": GREEN})
    rows.append({"name": "VideoMAE last two blocks, 3.0 s", "ba": lba, "lo": llo, "hi": lhi, "colour": GREEN})
    if one is not None:
        ba, lo, hi = ci_of(one)
        rows.append({"name": "VideoMAE last two blocks, 1.5 s", "ba": ba, "lo": lo, "hi": hi, "colour": ORANGE})
    if one_thr is not None:
        ba, lo, hi = ci_of(one_thr)
        rows.append({"name": "VideoMAE 1.5 s, train-fold threshold", "ba": ba, "lo": lo, "hi": hi, "colour": ORANGE})

    OUT.mkdir(parents=True, exist_ok=True)
    figure_a(rows, OUT / "figureA_model_comparison")
    figure_b(
        {"ba": lba, "lo": llo, "hi": lhi},
        None if one is None else dict(zip(("ba", "lo", "hi"), ci_of(one))),
        OUT / "figureB_temporal_sampling",
    )
    cms = [
        ("Frozen, 3.0 s", frozen["confusion"]),
        ("Last two blocks, 3.0 s", last2["confusion"]),
    ]
    if one is not None:
        cms.append(("Last two blocks, 1.5 s", one["confusion"]))
    figure_c(cms, OUT / "figureC_confusion")

    curves = []
    for name, pred_path in (
        ("Frozen, 3.0 s", FIXED / "frozen_encoder" / "oof_predictions.csv"),
        ("Last two blocks, 3.0 s", FIXED / "last_blocks_unfrozen" / "oof_predictions.csv"),
        ("Last two blocks, 1.5 s", FIXED_1P5 / "last_blocks_unfrozen" / "oof_predictions.csv"),
    ):
        if not pred_path.exists():
            continue
        frame = pd.read_csv(pred_path)
        rec, prec = pr_points(frame["label"].to_numpy(), frame["oof_probability"].to_numpy())
        curves.append((name, rec, prec))
    if curves:
        figure_d(curves, OUT / "figureD_pr_curves")

    figure_e(
        OUT / "figureE_clip_timelines",
        FIXED / "last_blocks_unfrozen" / "oof_predictions.csv",
        ["gold_001", "gold_005", "gold_010"],
    )
    rgb = Path("/scratch/db01550/rgb16_windowed_identity_dev")
    figure_f(
        rgb if rgb.exists() else None,
        FIXED / "fetch_manifest.csv",
        OUT / "figureF_identity_crops",
    )


if __name__ == "__main__":
    main()
