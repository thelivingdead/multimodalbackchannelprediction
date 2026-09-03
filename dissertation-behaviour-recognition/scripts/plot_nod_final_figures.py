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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paper_figure_style import (  # noqa: E402
    BLUE,
    GREEN,
    GREY,
    INK,
    MUTED,
    ORANGE,
    PAPER,
    SIZE_FULL,
    SIZE_FULL_TALL,
    SIZE_HALF,
    forest,
    save,
)
from src.windowed_protocol import (  # noqa: E402
    DEV_SAMPLE_IDS,
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
MIDLINE_FRAC = 0.15


def loadj(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"STOP: missing {path}")
    return json.loads(path.read_text())


def maybe(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def ci_of(metrics: dict) -> tuple[float, float, float]:
    ba = float(metrics["balanced_accuracy"])
    boot = metrics.get("clip_bootstrap") or {}
    block = boot.get("balanced_accuracy") or {}
    lo = float(block.get("ci_lower_95", ba))
    hi = float(block.get("ci_upper_95", ba))
    return ba, lo, hi


def figure_a(rows: list[dict], temporal: list[dict], stem: Path) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=SIZE_FULL_TALL,
        facecolor=PAPER,
        gridspec_kw={"height_ratios": [max(len(rows), 1), max(len(temporal), 1)], "hspace": 0.40},
    )
    forest(axes[0], rows, title="A   DEV nod detection. No interval excludes chance.")
    forest(axes[1], temporal, title="B   Same VideoMAE, 3 s vs 1.5 s. Points, not bars.")
    fig.subplots_adjust(left=0.36, right=0.97, top=0.93, bottom=0.12)
    fig.text(
        0.36,
        0.03,
        "Nod only. Chance is the dashed line at 0.500. Error bars are 95% clip-level\n"
        "bootstrap intervals. An interval that includes 0.500 is not distinguished from chance.",
        color=MUTED,
        va="bottom",
    )
    save(fig, stem)


def figure_b(temporal: list[dict], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=SIZE_FULL, facecolor=PAPER)
    forest(ax, temporal, title="Temporal sampling. Identity-fixed VideoMAE.")
    fig.subplots_adjust(left=0.36, right=0.97, top=0.88, bottom=0.18)
    fig.text(
        0.36,
        0.04,
        "Points with 95% clip-level intervals. The y axis is not a bar length.\n"
        "Both intervals include 0.500. The two estimates overlap.",
        color=MUTED,
        va="bottom",
    )
    save(fig, stem)


def figure_c(matrices: list[tuple[str, dict]], stem: Path) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=SIZE_FULL, facecolor=PAPER)
    if len(matrices) == 1:
        axes = [axes]
    for ax, (title, cm) in zip(axes, matrices):
        grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=int)
        row_n = grid.sum(axis=1, keepdims=True)
        row_pct = np.divide(grid, np.maximum(row_n, 1))
        ax.set_facecolor(PAPER)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        ax.set_aspect("equal")
        for i in range(2):
            for j in range(2):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=GREY, lw=1.0))
                ax.text(j, i - 0.10, str(grid[i, j]), ha="center", va="center", color=INK)
                ax.text(
                    j, i + 0.22,
                    f"{100 * row_pct[i, j]:.0f}% of row",
                    ha="center", va="center", color=MUTED,
                )
        ax.set_xticks([0, 1], ["Pred. 0", "Pred. 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.tick_params(length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{title}\nn = {int(grid.sum())}")
    fig.suptitle("Out-of-fold confusion on DEV. Counts and row percentages; no colour scale.")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.10, wspace=0.28)
    save(fig, stem)


def pr_points(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    rec = tp / max(y.sum(), 1)
    prec = tp / np.maximum(tp + fp, 1)
    return rec, prec


def figure_d(curves: list[tuple[str, np.ndarray, np.ndarray, float]], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=SIZE_HALF, facecolor=PAPER)
    ax.set_facecolor(PAPER)
    colours = [BLUE, GREEN, ORANGE, GREY]
    seen_prev: dict[float, str] = {}
    for (name, rec, prec, prev), colour in zip(curves, colours):
        ax.plot(rec, prec, color=colour, lw=1.6, label=name)
        key = round(float(prev), 3)
        if key not in seen_prev:
            seen_prev[key] = name
    drawn = set()
    for prev in seen_prev:
        key = round(prev, 3)
        if key in drawn:
            continue
        drawn.add(key)
        if key >= 0.10:
            ax.axhline(prev, color=GREY, lw=1.0, ls="--", label=f"chance 3 s  ({prev:.3f})")
        else:
            ax.axhline(prev, color=ORANGE, lw=1.0, ls=":", label=f"chance 1.5 s  ({prev:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision-recall, identity-fixed VideoMAE")
    ax.legend(frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.96, top=0.88, bottom=0.16)
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
    fig, axes = plt.subplots(len(clips), 1, figsize=SIZE_FULL_TALL, facecolor=PAPER)
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
        ax.step(mid, part[prob_col], where="mid", color=GREEN, lw=1.5, label="VideoMAE P(nod)")
        pred_col = "pred_at_0.5" if "pred_at_0.5" in part.columns else "pred"
        pos = part[part[pred_col] == 1]
        for rec in pos.itertuples(index=False):
            ax.axvspan(float(rec.start_sec), float(rec.end_sec), color=GREEN, alpha=0.07, zorder=0)
        for ev in events[events["sample_id"] == sid].itertuples(index=False):
            ax.axvspan(float(ev.start_sec), float(ev.end_sec), color=ORANGE, alpha=0.22, zorder=0)
        pose_path = POSE_DIR / f"{sid}.npz"
        if pose_path.exists():
            with np.load(pose_path, allow_pickle=True) as payload:
                pitch = np.asarray(payload["rotation_xyz"], dtype=float)[:, 0]
            t = np.arange(len(pitch)) / 25.0
            scale = np.nanmax(np.abs(pitch)) or 1.0
            ax.plot(t, 0.15 * pitch / scale, color=MUTED, lw=0.9, alpha=0.85, label="pose pitch (scaled)")
        ax.set_xlim(0, 60)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(nod)")
        ax.set_title(sid.replace("gold_", "gold "), loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [
        Line2D([0], [0], color=GREEN, lw=1.5, label="VideoMAE P(nod)"),
        Line2D([0], [0], color=MUTED, lw=0.9, label="pose pitch (scaled)"),
        Patch(facecolor=ORANGE, alpha=0.22, label="annotated nod"),
        Patch(facecolor=GREEN, alpha=0.20, label="predicted positive"),
    ]
    axes[0].legend(handles=handles, frameon=False, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time (s)")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.07, hspace=0.38)
    fig.text(0.10, 0.97, "DEV clip timelines", va="top")
    fig.text(
        0.10, 0.925,
        "The green line is the result: P(nod) stays between 0.4 and 0.6 across each clip,\n"
        "with almost no dynamic range. Orange: annotated nod. Green shading: predicted positive. Grey: pose pitch.",
        color=MUTED,
        va="top",
    )
    save(fig, stem)


def _watch_toward(cx: float, width: float, side: str) -> float:
    mid = width / 2.0
    if str(side).upper() == "LEFT":
        return mid - cx
    return cx - mid


def _pick_contact_sheet(resolved: pd.DataFrame, rgb_dir: Path | None) -> pd.DataFrame:
    """One tile per DEV clip. Prefer the centre furthest onto the watch side.

    Never fall back to a near-midline box when a tighter window exists in the
    manifest. Wrong-half is a different test: centre on the other side of the
    midline. A two-shot on the annotated half still has wrong-half = 0.
    """
    picked = []
    for sid in DEV_SAMPLE_IDS:
        g = resolved[resolved["sample_id"].astype(str) == sid].copy()
        if g.empty:
            picked.append(
                {
                    "sample_id": sid,
                    "window_id": "",
                    "watch_side": "",
                    "crop_centre_x": np.nan,
                    "frame_width": np.nan,
                    "two_shot": False,
                    "missing_clip": True,
                }
            )
            continue
        side = str(g["watch_side"].iloc[0])
        width = g["frame_width"].astype(float)
        cx = g["crop_centre_x"].astype(float)
        toward = [_watch_toward(float(a), float(b), side) for a, b in zip(cx, width)]
        g = g.assign(toward=toward, side_frac=np.array(toward) / width.to_numpy())
        tight = g[g["side_frac"] >= MIDLINE_FRAC]
        pool = tight if not tight.empty else g
        pool = pool.sort_values(["toward", "window_id"], ascending=[False, True])
        rec = pool.iloc[0].copy()
        rec["two_shot"] = bool(tight.empty)
        rec["missing_clip"] = False
        picked.append(rec)
    out = pd.DataFrame(picked)
    print("figure F picks:")
    for rec in out.itertuples(index=False):
        print(
            f"  {rec.sample_id}  {rec.window_id}  cx={rec.crop_centre_x}  "
            f"two_shot={rec.two_shot}  missing={rec.missing_clip}"
        )
    return out


def figure_f_withheld(stem: Path) -> None:
    fig = plt.figure(figsize=SIZE_FULL, facecolor=PAPER)
    fig.text(0.50, 0.62, "Figure F is withheld.", ha="center", va="center")
    fig.text(
        0.50, 0.38,
        "This machine has no identity-fixed RGB crops.\n"
        "The old 12-clip plate with a room shot on gold 004 must not go in.\n"
        "On otter: python3 scripts/plot_nod_final_figures.py",
        ha="center",
        va="center",
        color=MUTED,
    )
    save(fig, stem)


def figure_f(rgb_dir: Path | None, manifest_path: Path, stem: Path) -> None:
    if rgb_dir is None or not rgb_dir.exists() or not manifest_path.exists():
        figure_f_withheld(stem)
        return
    manifest = pd.read_csv(manifest_path)
    resolved = manifest[manifest["crop_status"] == "resolved"].copy()
    if resolved.empty:
        figure_f_withheld(stem)
        return
    picked = _pick_contact_sheet(resolved, rgb_dir)
    cols = 5
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=SIZE_FULL, facecolor=PAPER)
    axes = np.atleast_2d(axes)
    for ax, rec in zip(axes.ravel(), picked.itertuples(index=False)):
        ax.set_xticks([])
        ax.set_yticks([])
        label = str(rec.sample_id).replace("gold_", "gold ")
        if bool(rec.missing_clip) or not rec.window_id:
            ax.text(0.5, 0.5, "no resolved crop", ha="center", va="center", color=MUTED)
            ax.set_title(label)
            continue
        path = rgb_dir / f"{rec.window_id}.npz"
        if path.exists():
            with np.load(path, allow_pickle=True) as payload:
                ax.imshow(payload["rgb"][len(payload["rgb"]) // 2])
                side = str(payload["watch_side"]) if "watch_side" in payload else rec.watch_side
        else:
            ax.text(0.5, 0.5, "crop file missing", ha="center", va="center", color=MUTED)
            side = rec.watch_side
        note = "  two-shot" if bool(rec.two_shot) else ""
        ax.set_title(f"{label}  {side}{note}")
    for ax in axes.ravel()[len(picked):]:
        ax.axis("off")
    fig.suptitle(
        "Dissertation only. All 15 DEV clips. One resolved window per clip,\n"
        "centre furthest onto the annotator side. Wrong-half is a midline test,\n"
        "not a tight-head test. A two-shot on the annotated half still counts as 0.",
    )
    fig.text(
        0.01, 0.01,
        "RealTalk stills stay in the bound dissertation only. Do not put this plate on a website.",
        color=ORANGE,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.08, hspace=0.35, wspace=0.08)
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

    temporal = [
        {"name": "3.0 s / 16 frames  (5.3 frames/s)", "ba": lba, "lo": llo, "hi": lhi, "colour": GREEN},
    ]
    if one is not None:
        ba, lo, hi = ci_of(one)
        temporal.append(
            {"name": "1.5 s / 16 frames  (10.7 frames/s)", "ba": ba, "lo": lo, "hi": hi, "colour": ORANGE}
        )

    OUT.mkdir(parents=True, exist_ok=True)
    figure_a(rows, temporal, OUT / "figureA_model_comparison")
    figure_b(temporal, OUT / "figureB_temporal_sampling")
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
        y = frame["label"].to_numpy()
        rec, prec = pr_points(y, frame["oof_probability"].to_numpy())
        prev = float(y.mean()) if len(y) else 0.0
        curves.append((name, rec, prec, prev))
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
