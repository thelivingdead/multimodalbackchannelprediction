#!/usr/bin/env python3
"""Retention bias of the identity-fixed 3 s cropper on DEV nod windows.

Compares unresolved/failed crops for positive versus negative windows.
Pitch kinematics for dropped versus retained positives are reported only
when features/gold/*.npz are present. TEST is not read.

    python3 scripts/audit_drop_bias_identity_crops.py
    python3 scripts/audit_drop_bias_identity_crops.py \\
        --rgb-dir /scratch/db01550/rgb16_windowed_identity_dev
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

from src.utils import dump_json  # noqa: E402
from src.windowed_protocol import (  # noqa: E402
    DEV_SAMPLE_IDS,
    EVENTS_CSV,
    FPS,
    TEST_SAMPLE_IDS,
    load_events,
)

WINDOWS = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
MANIFEST = ROOT / "results" / "windowed_dev" / "videomae_identity_fixed" / "fetch_manifest.csv"
POSE_DIR = ROOT / "features" / "gold"
OUT = ROOT / "results" / "windowed_dev" / "drop_bias_audit"

INK = "#1d1d1f"
MUTED = "#5c5c63"
GREY = "#8a8a90"
BLUE = "#2c5f8a"
RED = "#9c3d32"
PAPER = "#ffffff"
PITCH_AXIS = 0


def fisher_two_sided(dropped_pos: int, kept_pos: int,
                     dropped_neg: int, kept_neg: int) -> tuple[float, float]:
    table = np.array([[dropped_pos, kept_pos], [dropped_neg, kept_neg]], dtype=int)
    try:
        from scipy.stats import fisher_exact
        odds, p_value = fisher_exact(table, alternative="two-sided")
        return float(odds), float(p_value)
    except ImportError:
        return float("nan"), float("nan")


def mannwhitney(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ImportError:
        return float("nan")


def refuse_test(frame: pd.DataFrame, column: str = "sample_id") -> None:
    leaked = set(frame[column].astype(str)) & set(TEST_SAMPLE_IDS)
    if leaked:
        raise SystemExit(f"STOP: TEST id present: {sorted(leaked)}")


def pitch_metrics(rotation: np.ndarray, i0: int, i1: int) -> dict:
    sl = rotation[max(i0, 0):max(i1, i0 + 1), PITCH_AXIS].astype(float)
    if sl.size < 2:
        return {
            "peak_abs_pitch_velocity": float("nan"),
            "mean_abs_pitch_velocity": float("nan"),
            "max_pitch_excursion": float("nan"),
            "n_pose_frames": int(sl.size),
        }
    vel = np.abs(np.diff(sl)) * FPS
    return {
        "peak_abs_pitch_velocity": float(np.max(vel)),
        "mean_abs_pitch_velocity": float(np.mean(vel)),
        "max_pitch_excursion": float(np.max(sl) - np.min(sl)),
        "n_pose_frames": int(sl.size),
    }


def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=PAPER)
    fig.savefig(stem.with_suffix(".svg"), facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.png')}")


def figure_retention(metrics: dict, stem: Path) -> None:
    labels = ["Positive windows", "Negative windows"]
    values = [
        100.0 * metrics["drop_rate_positive"],
        100.0 * metrics["drop_rate_negative"],
    ]
    dropped = [metrics["dropped_positive"], metrics["dropped_negative"]]
    totals = [metrics["n_positive"], metrics["n_negative"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.2), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    bars = ax.bar(
        labels, values, color=[RED, GREY], width=0.55, edgecolor=PAPER
    )
    for bar, n_drop, n_tot, value in zip(bars, dropped, totals, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.6,
            f"{n_drop}/{n_tot}\n{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax.set_ylabel("Unresolved or failed crops (%)")
    ax.set_ylim(0, max(values) * 1.35 + 4)
    ax.set_title("Crop retention by window class, DEV 3 s")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save_fig(fig, stem)


def figure_motion(motion: pd.DataFrame, stem: Path) -> None:
    if motion.empty or motion["peak_abs_pitch_velocity"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.4), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    groups = ["retained", "dropped"]
    data = [
        motion.loc[motion["retention"] == name, "peak_abs_pitch_velocity"]
        .dropna()
        .to_numpy()
        for name in groups
    ]
    parts = ax.violinplot(data, positions=[1, 2], showextrema=False, widths=0.7)
    for body, colour in zip(parts["bodies"], [BLUE, RED]):
        body.set_facecolor(colour)
        body.set_alpha(0.28)
    ax.boxplot(data, positions=[1, 2], widths=0.22, showfliers=False)
    rng = np.random.default_rng(0)
    for i, values in enumerate(data, start=1):
        jitter = rng.normal(0.0, 0.06, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=18, color=INK, zorder=3)
    ax.set_xticks([1, 2], ["Retained positives", "Dropped positives"])
    ax.set_ylabel("Peak absolute pitch velocity (deg/s)")
    ax.set_title("Pitch motion of positive windows, by crop retention")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save_fig(fig, stem)


def figure_examples(
    dropped: pd.DataFrame,
    events: pd.DataFrame,
    pose_cache: dict,
    rgb_dir: Path | None,
    stem: Path,
) -> None:
    if dropped.empty:
        return
    take = dropped.sort_values("peak_abs_pitch_velocity", ascending=False).head(8)
    n = len(take)
    fig, axes = plt.subplots(n, 2, figsize=(9.2, 1.7 * n), facecolor=PAPER)
    if n == 1:
        axes = np.asarray([axes])
    for row_ax, rec in zip(axes, take.itertuples(index=False)):
        ax_img, ax_tr = row_ax
        ax_img.set_facecolor(PAPER)
        ax_tr.set_facecolor(PAPER)
        shown = False
        if rgb_dir is not None:
            preview_path = rgb_dir / f"{rec.window_id}.npz"
            if preview_path.exists():
                with np.load(preview_path, allow_pickle=True) as payload:
                    if "preview" in payload.files:
                        ax_img.imshow(payload["preview"])
                        shown = True
        if not shown:
            ax_img.text(0.5, 0.5, "crop not stored\n(unresolved)", ha="center", va="center")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(f"{rec.window_id}  {rec.reason}", fontsize=8, loc="left")

        rot = pose_cache.get(str(rec.sample_id))
        if rot is None:
            ax_tr.text(0.5, 0.5, "pose npz absent", ha="center", va="center")
            ax_tr.set_xticks([])
            ax_tr.set_yticks([])
            continue
        i0, i1 = int(rec.start_frame_relative), int(rec.end_frame_relative)
        sl = rot[max(i0, 0):max(i1, i0 + 1), PITCH_AXIS]
        t = np.arange(len(sl)) / FPS + float(rec.start_sec)
        ax_tr.plot(t, sl, color=INK, lw=1.1)
        clip_events = events[events["sample_id"] == rec.sample_id]
        for ev in clip_events.itertuples(index=False):
            if overlap(float(rec.start_sec), float(rec.end_sec),
                       float(ev.start_sec), float(ev.end_sec)):
                ax_tr.axvspan(float(ev.start_sec), float(ev.end_sec),
                              color=RED, alpha=0.18, zorder=0)
        ax_tr.set_xlim(float(rec.start_sec), float(rec.end_sec))
        ax_tr.set_ylabel("pitch (deg)", fontsize=8)
        ax_tr.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax_tr.spines[side].set_visible(False)
    axes[-1, 1].set_xlabel("Time in clip (s)")
    fig.suptitle(
        "Dropped positive windows. Shaded intervals are human nod annotations.",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, stem)


def overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) - max(a0, b0) > 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, default=WINDOWS)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--pose-dir", type=Path, default=POSE_DIR)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows)
    windows["sample_id"] = windows["sample_id"].astype(str)
    windows["window_id"] = windows["window_id"].astype(str)
    refuse_test(windows)
    if set(windows["sample_id"]) - set(DEV_SAMPLE_IDS):
        raise SystemExit("STOP: non-DEV clip in windows file")

    manifest = pd.read_csv(args.manifest)
    manifest["window_id"] = manifest["window_id"].astype(str)
    refuse_test(manifest)
    if "split" in manifest.columns and (manifest["split"].astype(str) != "DEV").any():
        raise SystemExit("STOP: non-DEV row in fetch manifest")

    merged = windows.merge(
        manifest[["window_id", "crop_status", "reason"]],
        on="window_id",
        how="left",
        validate="one_to_one",
    )
    if merged["crop_status"].isna().any():
        missing = merged.loc[merged["crop_status"].isna(), "window_id"].tolist()
        raise SystemExit(f"STOP: windows missing from manifest: {missing[:8]}")
    merged["dropped"] = merged["crop_status"].isin(["unresolved", "failed"])

    n_pos = int((merged["label"] == 1).sum())
    n_neg = int((merged["label"] == 0).sum())
    dropped_pos = int(((merged["label"] == 1) & merged["dropped"]).sum())
    dropped_neg = int(((merged["label"] == 0) & merged["dropped"]).sum())
    kept_pos = n_pos - dropped_pos
    kept_neg = n_neg - dropped_neg
    rate_pos = dropped_pos / n_pos if n_pos else float("nan")
    rate_neg = dropped_neg / n_neg if n_neg else float("nan")
    rel_risk = rate_pos / rate_neg if rate_neg not in (0.0, float("nan")) else float("nan")
    odds, p_value = fisher_two_sided(dropped_pos, kept_pos, dropped_neg, kept_neg)

    events = load_events(EVENTS_CSV, allow_test=False)
    refuse_test(events)

    pose_cache: dict[str, np.ndarray] = {}
    motion_rows = []
    pose_available = False
    for sid in sorted(set(merged["sample_id"])):
        path = args.pose_dir / f"{sid}.npz"
        if not path.exists():
            continue
        pose_available = True
        with np.load(path, allow_pickle=True) as payload:
            pose_cache[sid] = np.asarray(payload["rotation_xyz"], dtype=float)
    positives = merged[merged["label"] == 1].copy()
    for rec in positives.itertuples(index=False):
        rot = pose_cache.get(str(rec.sample_id))
        kinematics = {
            "peak_abs_pitch_velocity": float("nan"),
            "mean_abs_pitch_velocity": float("nan"),
            "max_pitch_excursion": float("nan"),
            "n_pose_frames": 0,
        }
        if rot is not None:
            kinematics = pitch_metrics(
                rot, int(rec.start_frame_relative), int(rec.end_frame_relative)
            )
        motion_rows.append(
            {
                "window_id": rec.window_id,
                "sample_id": rec.sample_id,
                "start_sec": float(rec.start_sec),
                "end_sec": float(rec.end_sec),
                "start_frame_relative": int(rec.start_frame_relative),
                "end_frame_relative": int(rec.end_frame_relative),
                "retention": "dropped" if rec.dropped else "retained",
                "crop_status": rec.crop_status,
                "reason": rec.reason,
                **kinematics,
            }
        )
    motion = pd.DataFrame(motion_rows)
    dropped_m = motion[motion["retention"] == "dropped"]
    retained_m = motion[motion["retention"] == "retained"]

    def summarise(column: str) -> dict:
        a = retained_m[column].dropna().to_numpy()
        b = dropped_m[column].dropna().to_numpy()
        return {
            "retained_median": float(np.median(a)) if a.size else float("nan"),
            "dropped_median": float(np.median(b)) if b.size else float("nan"),
            "mannwhitney_p": mannwhitney(a, b),
        }

    payload = {
        "protocol": "identity_fixed_3s_crop_drop_bias_dev",
        "development_only": True,
        "test_touched": False,
        "n_windows": int(len(merged)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "retained_positive": kept_pos,
        "retained_negative": kept_neg,
        "dropped_positive": dropped_pos,
        "dropped_negative": dropped_neg,
        "drop_rate_positive": rate_pos,
        "drop_rate_negative": rate_neg,
        "relative_risk_drop_positive": rel_risk,
        "fisher_odds_ratio": odds,
        "fisher_p_two_sided": p_value,
        "pose_available": pose_available,
        "pitch_axis": PITCH_AXIS,
        "axis_note": "Channel 0 of rotation_xyz, the axis used by the 3 s nod rule.",
        "motion_comparison": {
            "peak_abs_pitch_velocity": summarise("peak_abs_pitch_velocity"),
            "mean_abs_pitch_velocity": summarise("mean_abs_pitch_velocity"),
            "max_pitch_excursion": summarise("max_pitch_excursion"),
        },
        "inference_note": (
            "A higher drop rate among positives does not by itself show that "
            "nodding caused the cropper to fail."
        ),
    }
    dump_json(out_dir / "drop_bias_metrics.json", payload)
    table = pd.DataFrame(
        [
            {
                "class": "positive",
                "n": n_pos,
                "retained": kept_pos,
                "dropped": dropped_pos,
                "drop_rate": rate_pos,
            },
            {
                "class": "negative",
                "n": n_neg,
                "retained": kept_neg,
                "dropped": dropped_neg,
                "drop_rate": rate_neg,
            },
        ]
    )
    table.to_csv(out_dir / "drop_bias_table.csv", index=False)
    motion.to_csv(out_dir / "positive_window_motion.csv", index=False)

    figure_retention(payload, out_dir / "figure1_crop_retention_bias")
    figure_motion(motion, out_dir / "figure2_pitch_motion_positives")
    figure_examples(
        dropped_m, events, pose_cache, args.rgb_dir,
        out_dir / "figure3_dropped_positive_examples",
    )
    print(json.dumps({
        "n_positive": n_pos,
        "n_negative": n_neg,
        "dropped_positive": dropped_pos,
        "dropped_negative": dropped_neg,
        "drop_rate_positive": round(rate_pos, 4),
        "drop_rate_negative": round(rate_neg, 4),
        "relative_risk": None if np.isnan(rel_risk) else round(rel_risk, 3),
        "fisher_p": None if np.isnan(p_value) else round(p_value, 4),
        "pose_available": pose_available,
    }, indent=2))


if __name__ == "__main__":
    main()
