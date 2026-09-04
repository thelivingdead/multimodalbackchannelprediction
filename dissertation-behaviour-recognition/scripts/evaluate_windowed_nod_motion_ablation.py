#!/usr/bin/env python3
"""DEV-only nod rule ablation: amplitude plus motion structure.

Pitch amplitude cannot tell a nod from a one-way glance or posture shift.
This script adds velocity zero-crossings and a return ratio, selects
thresholds on DEV only, and never reads TEST windows or TEST pose.

    python3 scripts/evaluate_windowed_nod_motion_ablation.py

Writes results/windowed_dev/rule_motion_ablation/. Does not train. Does not
overwrite windowed_nod/baselines_bacc.
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
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_split_leakage import assert_unlocked_out_dir  # noqa: E402
from src.clip_metrics import clip_binary_metrics  # noqa: E402
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
    save,
)
from src.pose_cnn import load_npz  # noqa: E402
from src.utils import dump_json  # noqa: E402
from src.windowed_baselines import (  # noqa: E402
    N_BOOTSTRAP,
    WINDOW_FRAMES,
    average_precision,
    candidate_thresholds,
    clip_bootstrap,
    load_windows,
    rule_score_function,
    selection_key,
)

WINDOWS_DEV = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
GOLD_DIR = ROOT / "features" / "gold"
OUT_DIR = ROOT / "results" / "windowed_dev" / "rule_motion_ablation"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_PREFIXES = tuple(f"gold_{i:03d}" for i in range(16, 31))
AXIS = 0
SAVGOL_WINDOW = 11
SAVGOL_POLY = 2
VELOCITY_DEADBAND = 0.08  # deg / frame; ~2 deg/s at 25 fps. Fixed before the sweep.
EPS = 1e-6
MAX_ZERO_CROSS = 12


def _smooth_pitch(chunk: np.ndarray) -> np.ndarray:
    x = np.asarray(chunk[:, AXIS], dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    if np.isnan(x).all():
        return np.zeros(len(x), dtype=float)
    fill = np.nanmedian(x)
    x = np.where(np.isfinite(x), x, fill)
    if x.size < SAVGOL_WINDOW:
        return x
    return savgol_filter(x, SAVGOL_WINDOW, SAVGOL_POLY)


def zero_crossings(chunk: np.ndarray, deadband: float = VELOCITY_DEADBAND) -> int:
    """Count velocity sign changes after ignoring |v| < deadband."""
    sm = _smooth_pitch(chunk)
    if sm.size < 3:
        return 0
    vel = np.diff(sm)
    signed = np.sign(vel)
    signed[np.abs(vel) < deadband] = 0
    nonzero = signed[signed != 0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] != nonzero[:-1]))


def return_ratio(chunk: np.ndarray) -> float:
    """|end - start| / path length. One-way drift is near 1; a nod is lower."""
    sm = _smooth_pitch(chunk)
    if sm.size < 2:
        return 0.0
    net = float(abs(sm[-1] - sm[0]))
    path = float(np.sum(np.abs(np.diff(sm))))
    return net / (path + EPS)


def extract_features(dev: pd.DataFrame) -> pd.DataFrame:
    rule_score = rule_score_function()
    cache: dict[str, np.ndarray] = {}
    rows = []
    for rec in dev.itertuples(index=False):
        sid = str(rec.sample_id)
        if sid.startswith(TEST_PREFIXES) or sid not in DEV_IDS:
            raise SystemExit(f"STOP: non-DEV sample {sid}")
        if sid not in cache:
            path = GOLD_DIR / f"{sid}.npz"
            if not path.exists():
                raise SystemExit(f"STOP: missing {path}")
            cache[sid] = np.asarray(load_npz(path)["rotation_xyz"], dtype=np.float32)
        start = int(rec.start_frame_relative)
        end = int(rec.end_frame_relative)
        chunk = cache[sid][start:end]
        if chunk.shape != (WINDOW_FRAMES, 3):
            raise SystemExit(f"STOP: {sid} window shape {chunk.shape}")
        rows.append(
            {
                "clip_id": sid,
                "window_id": rec.window_id,
                "window_start": float(rec.start_sec),
                "start_frame_relative": int(rec.start_frame_relative),
                "label": int(rec.label),
                "amplitude": float(rule_score(chunk, AXIS)),
                "zero_crossings": zero_crossings(chunk),
                "return_ratio": return_ratio(chunk),
            }
        )
    frame = pd.DataFrame(rows)
    if set(frame["clip_id"]) != DEV_IDS:
        raise SystemExit("STOP: feature table is not the 15 DEV clips")
    return frame, cache


def _predict(frame: pd.DataFrame, amp_t: float, zc_k: int, rr_t: float) -> np.ndarray:
    return (
        (frame["amplitude"].to_numpy() >= amp_t)
        & (frame["zero_crossings"].to_numpy() >= zc_k)
        & (frame["return_ratio"].to_numpy() <= rr_t)
    ).astype(int)


def _rank_score(frame: pd.DataFrame, amp_t: float, zc_k: int, rr_t: float) -> np.ndarray:
    """Amplitude, zeroed when the selected extra gates fail."""
    score = frame["amplitude"].to_numpy(dtype=float).copy()
    score[_predict(frame, amp_t, zc_k, rr_t) == 0] = 0.0
    return score


def select_rule(
    frame: pd.DataFrame,
    *,
    use_zc: bool,
    use_rr: bool,
) -> dict:
    y = frame["label"].to_numpy(dtype=int)
    amp_grid = candidate_thresholds(frame["amplitude"].to_numpy())
    zc_grid = (
        list(range(0, min(MAX_ZERO_CROSS, int(frame["zero_crossings"].max())) + 1))
        if use_zc
        else [0]
    )
    if use_rr:
        rr_grid = sorted(
            set(
                float(v)
                for v in np.quantile(
                    frame["return_ratio"].to_numpy(),
                    np.linspace(0.05, 1.0, 20),
                )
            )
        )
    else:
        rr_grid = [10.0]
    best = None
    search_rows = []
    for amp_t in amp_grid:
        for zc_k in zc_grid:
            for rr_t in rr_grid:
                pred = _predict(frame, amp_t, zc_k, rr_t)
                metrics = clip_binary_metrics(y, pred)
                row = {
                    "amp_threshold": float(amp_t),
                    "min_zero_crossings": int(zc_k),
                    "max_return_ratio": float(rr_t),
                    **metrics,
                }
                search_rows.append(row)
                key = selection_key(metrics, amp_t, "balanced_accuracy") + (
                    zc_k,
                    -rr_t,
                )
                if best is None or key > best[0]:
                    best = (key, row, pred)
    assert best is not None
    chosen = dict(best[1])
    pred = np.asarray(best[2], dtype=int)
    score = _rank_score(
        frame,
        chosen["amp_threshold"],
        chosen["min_zero_crossings"],
        chosen["max_return_ratio"],
    )
    boot = clip_bootstrap(
        frame["clip_id"].to_numpy(),
        y,
        pred,
        n_resamples=N_BOOTSTRAP,
    )
    return {
        "amp_threshold": chosen["amp_threshold"],
        "min_zero_crossings": chosen["min_zero_crossings"],
        "max_return_ratio": chosen["max_return_ratio"],
        "metrics": {k: chosen[k] for k in (
            "balanced_accuracy", "f1", "precision", "recall", "tp", "fp", "tn", "fn"
        )},
        "pr_auc": average_precision(y, score),
        "clip_bootstrap": boot,
        "prediction": pred,
        "search_n": len(search_rows),
    }


def figure_ba(results: dict[str, dict], stem: Path) -> None:
    order = ["A", "B", "C", "D"]
    labels = {
        "A": "A  amplitude",
        "B": "B  + zero crossings",
        "C": "C  + return ratio",
        "D": "D  both",
    }
    fig, ax = plt.subplots(figsize=SIZE_FULL, facecolor=PAPER)
    xs = np.arange(len(order))
    ba = [results[k]["metrics"]["balanced_accuracy"] for k in order]
    lo = [results[k]["clip_bootstrap"]["balanced_accuracy"]["ci_lower_95"] for k in order]
    hi = [results[k]["clip_bootstrap"]["balanced_accuracy"]["ci_upper_95"] for k in order]
    ax.axhline(0.5, color=INK, ls="--", lw=1.0, zorder=0)
    ax.bar(xs, ba, color=BLUE, width=0.55, edgecolor=PAPER, zorder=2)
    ax.errorbar(
        xs,
        ba,
        yerr=[np.array(ba) - np.array(lo), np.array(hi) - np.array(ba)],
        fmt="none",
        ecolor=INK,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(xs, [labels[k] for k in order])
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 0.85)
    ax.set_title("DEV nod rule. Motion-structure ablation.")
    ax.text(3.45, 0.51, "chance  0.500", color=MUTED, ha="right", va="bottom")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.18)
    fig.text(
        0.12,
        0.04,
        "DEV only, 15 clips. Bars are balanced accuracy; whiskers are 95% clip-level intervals.\n"
        "The y axis starts at 0. Thresholds selected on DEV. TEST was not read.",
        color=MUTED,
    )
    save(fig, stem)


def _diverse_rows(frame: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """One window per clip, then fill if fewer than n clips."""
    picks = []
    seen: set[str] = set()
    for rec in frame.itertuples(index=False):
        sid = str(rec.clip_id)
        if sid in seen:
            continue
        seen.add(sid)
        picks.append(rec)
        if len(picks) >= n:
            break
    if len(picks) < n:
        for rec in frame.itertuples(index=False):
            if rec in picks:
                continue
            picks.append(rec)
            if len(picks) >= n:
                break
    return pd.DataFrame(picks)


def figure_traces(frame: pd.DataFrame, cache: dict[str, np.ndarray], stem: Path) -> None:
    y = frame["label"].to_numpy()
    pred_a = frame["pred_A"].to_numpy()
    nods = frame[(y == 1) & (pred_a == 1)]
    fps = frame[(y == 0) & (pred_a == 1)].sort_values("amplitude", ascending=False)
    if nods.empty or fps.empty:
        nods = frame[frame["label"] == 1]
        fps = frame[frame["label"] == 0].sort_values("amplitude", ascending=False)
    nod_pick = _diverse_rows(nods)
    fp_pick = _diverse_rows(fps)
    fig, axes = plt.subplots(2, 3, figsize=SIZE_FULL_TALL, facecolor=PAPER)
    groups = [(nod_pick, "True nod", GREEN), (fp_pick, "Amplitude false positive", ORANGE)]
    for row, (pick, title, colour) in enumerate(groups):
        for col, rec in enumerate(pick.itertuples(index=False)):
            ax = axes[row, col]
            ax.set_facecolor(PAPER)
            sid = rec.clip_id
            start = int(rec.start_frame_relative)
            chunk = cache[sid][start : start + WINDOW_FRAMES]
            sm = _smooth_pitch(chunk)
            vel = np.diff(sm)
            t = np.arange(len(sm)) / 25.0
            ax.plot(t, sm, color=INK, lw=1.2, label="pitch")
            axr = ax.twinx()
            axr.plot(t[1:], vel, color=colour, lw=0.9, alpha=0.9, label="velocity")
            axr.axhline(0.0, color=MUTED, lw=0.6, ls=":")
            ax.set_title(f"{title}\n{sid.replace('_', ' ')}  {rec.window_start:.0f} s")
            ax.set_xlabel("Time in window (s)")
            if col == 0:
                ax.set_ylabel("Pitch (°)")
            if col == 2:
                axr.set_ylabel("Pitch velocity (°/frame)")
            else:
                axr.set_yticklabels([])
            for spine_ax in (ax, axr):
                for side in ("top",):
                    spine_ax.spines[side].set_visible(False)
    fig.suptitle("Pitch (black) and velocity (colour). Same 3 s window length.")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.08, hspace=0.45, wspace=0.32)
    save(fig, stem)


def figure_scatter(frame: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=SIZE_FULL, facecolor=PAPER)
    ax.set_facecolor(PAPER)
    pos = frame[frame["label"] == 1]
    neg = frame[frame["label"] == 0]
    ax.scatter(
        neg["zero_crossings"],
        neg["return_ratio"],
        s=18,
        c=GREY,
        alpha=0.7,
        label=f"No nod  n={len(neg)}",
        zorder=2,
    )
    ax.scatter(
        pos["zero_crossings"],
        pos["return_ratio"],
        s=22,
        c=ORANGE,
        alpha=0.85,
        label=f"Nod  n={len(pos)}",
        zorder=3,
    )
    ax.set_xlabel("Pitch-velocity zero crossings")
    ax.set_ylabel("Return ratio")
    ax.set_title("DEV windows. Motion structure by human nod label.")
    ax.legend(frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.16)
    fig.text(
        0.12,
        0.04,
        "Return ratio near 1 is one-way drift. Lower values are round trips.\n"
        "Zero crossings count direction reversals after a 0.08°/frame deadband.",
        color=MUTED,
    )
    save(fig, stem)


def decide(results: dict[str, dict]) -> dict:
    a = results["A"]["metrics"]["balanced_accuracy"]
    deltas = {
        k: float(results[k]["metrics"]["balanced_accuracy"] - a)
        for k in ("B", "C", "D")
    }
    clear = {k: deltas[k] >= 0.03 for k in deltas}
    d_minus_c = deltas["D"] - deltas["C"]
    if not any(clear.values()):
        best = max(deltas, key=deltas.get)
        text = (
            f"No clear gain over amplitude-only (A = {a:.3f}). "
            f"Best extra is {best} at {deltas[best]:+.3f}. "
            "Do not retrain CNN or VideoMAE on this."
        )
        return {
            "best_variant": best,
            "cause": None,
            "delta_vs_A": deltas,
            "clear_improvement": False,
            "text": text,
        }
    if clear["C"] and d_minus_c < 0.03 and not clear["B"]:
        best, cause = "C", "return ratio"
    elif clear["B"] and not clear["C"]:
        best, cause = "B", "pitch-velocity zero crossings"
    elif clear["D"] and d_minus_c >= 0.03:
        best, cause = "D", "zero crossings and return ratio together"
    elif clear["C"]:
        best, cause = "C", "return ratio"
    else:
        best, cause = "B", "pitch-velocity zero crossings"
    text = (
        f"{best} clearly improves on A ({a:.3f} → "
        f"{results[best]['metrics']['balanced_accuracy']:.3f}, "
        f"{deltas[best]:+.3f}). The added structure is {cause}."
    )
    return {
        "best_variant": best,
        "cause": cause,
        "delta_vs_A": deltas,
        "clear_improvement": True,
        "text": text,
    }


def main() -> None:
    out = assert_unlocked_out_dir(OUT_DIR)
    if WINDOWS_DEV.name != "nod_windows_dev.csv":
        raise SystemExit("STOP: expected DEV window file")
    dev = load_windows(WINDOWS_DEV, "DEV", DEV_IDS)
    if any(str(s).startswith(TEST_PREFIXES) for s in dev["sample_id"]):
        raise SystemExit("STOP: TEST id in DEV windows")
    frame, cache = extract_features(dev)

    specs = {
        "A": {"use_zc": False, "use_rr": False, "name": "amplitude only"},
        "B": {"use_zc": True, "use_rr": False, "name": "amplitude + zero crossings"},
        "C": {"use_zc": False, "use_rr": True, "name": "amplitude + return ratio"},
        "D": {"use_zc": True, "use_rr": True, "name": "amplitude + zero crossings + return ratio"},
    }
    results = {}
    for key, spec in specs.items():
        print(f"selecting {key} on DEV…")
        chosen = select_rule(frame, use_zc=spec["use_zc"], use_rr=spec["use_rr"])
        frame[f"pred_{key}"] = chosen["prediction"]
        results[key] = {**spec, **{k: v for k, v in chosen.items() if k != "prediction"}}
        print(
            f"  {key} BA {chosen['metrics']['balanced_accuracy']:.3f}  "
            f"F1 {chosen['metrics']['f1']:.3f}  "
            f"P {chosen['metrics']['precision']:.3f}  "
            f"R {chosen['metrics']['recall']:.3f}  "
            f"PR AUC {chosen['pr_auc']:.3f}  "
            f"TP{chosen['metrics']['tp']} FP{chosen['metrics']['fp']} "
            f"TN{chosen['metrics']['tn']} FN{chosen['metrics']['fn']}"
        )

    decision = decide(results)
    out.mkdir(parents=True, exist_ok=True)
    feature_cols = [
        "clip_id",
        "window_start",
        "label",
        "amplitude",
        "zero_crossings",
        "return_ratio",
    ]
    for key in specs:
        part = frame[feature_cols].copy()
        part["prediction"] = frame[f"pred_{key}"]
        part.to_csv(out / f"windows_{key}.csv", index=False)
    frame[feature_cols + [f"pred_{k}" for k in specs]].to_csv(
        out / "windows.csv", index=False
    )

    serialisable = {}
    for key, block in results.items():
        serialisable[key] = {
            "name": block["name"],
            "amp_threshold": block["amp_threshold"],
            "min_zero_crossings": block["min_zero_crossings"],
            "max_return_ratio": block["max_return_ratio"],
            "metrics": block["metrics"],
            "pr_auc": block["pr_auc"],
            "clip_bootstrap": block["clip_bootstrap"],
            "search_n": block["search_n"],
        }
    dump_json(
        out / "metrics.json",
        {
            "protocol": "windowed_nod_3s_motion_ablation_dev",
            "development_only": True,
            "test_scored": False,
            "test_read": False,
            "axis": AXIS,
            "axis_name": "x",
            "savgol_window": SAVGOL_WINDOW,
            "velocity_deadband_deg_per_frame": VELOCITY_DEADBAND,
            "n_windows": int(len(frame)),
            "n_positive": int(frame["label"].sum()),
            "n_clips": 15,
            "variants": serialisable,
            "decision": decision,
        },
    )
    figure_ba(results, out / "figure_ba_comparison")
    figure_traces(frame, cache, out / "figure_nod_vs_fp_traces")
    figure_scatter(frame, out / "figure_zc_vs_return")
    print(decision["text"])
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
