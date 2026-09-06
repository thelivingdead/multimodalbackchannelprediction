#!/usr/bin/env python3
"""DEV figures for the 3 s audio-only nod experiment. TEST is not read."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.audio_io import TARGET_SR, load_wav_mono, resample_mono  # noqa: E402
from src.paper_figure_style import BLUE, GREY, INK, MUTED, PAPER, SIZE_FULL, SIZE_FULL_TALL, save  # noqa: E402
from src.windowed_baselines import load_windows  # noqa: E402

OUT = ROOT / "results" / "windowed_dev" / "audio_3s"
CACHE = ROOT / "data" / "features" / "audio_windowed_dev"
WINDOWS = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
EVENTS = ROOT / "data" / "windowed_annotations" / "nod_events_windowed.csv"
CNN = ROOT / "results" / "windowed_nod" / "pose_cnn_loco_dev" / "metrics_dev.json"
RULE = ROOT / "results" / "windowed_dev" / "rule_motion_ablation" / "metrics.json"
VMAE = (
    ROOT
    / "results"
    / "windowed_dev"
    / "videomae_identity_fixed_1p5s"
    / "last_blocks_no_hflip"
    / "metrics.json"
)
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
EXAMPLE = "gold_001"


def _logmel(y: np.ndarray, sr: int) -> np.ndarray:
    win = max(int(0.025 * sr), 16)
    hop = max(int(0.010 * sr), 8)
    _, _, zxx = signal.stft(
        y.astype(np.float64),
        fs=sr,
        nperseg=win,
        noverlap=win - hop,
        nfft=512,
        boundary=None,
        padded=False,
    )
    mag = np.abs(zxx)
    return np.log(np.maximum(mag, 1e-10))


def figure_a(out_dir: Path) -> None:
    wav = CACHE / f"{EXAMPLE}_clip.wav"
    windows = load_windows(WINDOWS, "DEV", DEV_IDS)
    part = windows[windows["sample_id"] == EXAMPLE]
    events = pd.read_csv(EVENTS)
    ev = events[events["sample_id"].astype(str) == EXAMPLE]
    fig, axes = plt.subplots(3, 1, figsize=SIZE_FULL_TALL, facecolor=PAPER, sharex=True)
    if wav.exists():
        y, sr = load_wav_mono(wav)
        y = resample_mono(y, sr, TARGET_SR)
        t = np.arange(len(y)) / TARGET_SR
        axes[0].plot(t, y, color=INK, lw=0.6)
    else:
        axes[0].text(0.5, 0.5, "clip wav not extracted yet", ha="center", va="center")
    axes[0].set_ylabel("Waveform")
    axes[0].set_title("Figure A. DEV 3 s audio windows. TEST not shown.")
    dur = float(part["end_sec"].max())
    for rec in part.itertuples(index=False):
        colour = BLUE if int(rec.label) == 1 else GREY
        axes[1].axvspan(float(rec.start_sec), float(rec.end_sec), color=colour, alpha=0.35)
    axes[1].set_ylabel("Windows")
    axes[1].set_yticks([])
    for rec in ev.itertuples(index=False):
        axes[2].axvspan(float(rec.start_sec), float(rec.end_sec), color=INK, alpha=0.45)
    axes[2].set_ylabel("Nod events")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_xlim(0.0, max(dur, 3.0))
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, out_dir / "figures" / "figureA_protocol")


def figure_b(out_dir: Path) -> None:
    windows = load_windows(WINDOWS, "DEV", DEV_IDS)
    pos = windows[windows["label"] == 1].iloc[0]
    neg = windows[windows["label"] == 0].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=SIZE_FULL, facecolor=PAPER)
    for col, rec, title in (
        (0, pos, "Positive nod window"),
        (1, neg, "Negative window"),
    ):
        wav = CACHE / f"{rec.sample_id}_clip.wav"
        axes[0, col].set_title(title)
        if not wav.exists():
            axes[0, col].text(0.5, 0.5, "wav missing", ha="center")
            continue
        y, sr = load_wav_mono(wav)
        y = resample_mono(y, sr, TARGET_SR)
        s0 = int(round(float(rec.start_sec) * TARGET_SR))
        s1 = int(round(float(rec.end_sec) * TARGET_SR))
        chunk = y[s0:s1]
        t = np.arange(len(chunk)) / TARGET_SR
        axes[0, col].plot(t, chunk, color=INK, lw=0.6)
        axes[0, col].set_ylabel("Waveform" if col == 0 else "")
        spec = _logmel(chunk, TARGET_SR)
        axes[1, col].imshow(spec, origin="lower", aspect="auto", cmap="gray")
        axes[1, col].set_ylabel("Log STFT" if col == 0 else "")
        axes[1, col].set_xlabel("Time")
    fig.suptitle("Figure B. What the audio model summarises. DEV. TEST not shown.")
    fig.tight_layout()
    save(fig, out_dir / "figures" / "figureB_representation")


def figure_c(out_dir: Path) -> None:
    audio = json.loads((out_dir / "metrics.json").read_text())
    rows = [
        {
            "name": "Audio MFCC LR",
            "ba": audio["at_fixed_threshold_0.5"]["balanced_accuracy"],
            "lo": audio["clip_bootstrap_at_0.5"]["balanced_accuracy"]["ci_lower_95"],
            "hi": audio["clip_bootstrap_at_0.5"]["balanced_accuracy"]["ci_upper_95"],
        }
    ]
    if RULE.exists():
        rule = json.loads(RULE.read_text())["variants"]["C"]
        rows.append(
            {
                "name": "Return-ratio rule",
                "ba": rule["metrics"]["balanced_accuracy"],
                "lo": rule["clip_bootstrap"]["balanced_accuracy"]["ci_lower_95"],
                "hi": rule["clip_bootstrap"]["balanced_accuracy"]["ci_upper_95"],
            }
        )
    if CNN.exists():
        cnn = json.loads(CNN.read_text())
        rows.append(
            {
                "name": "Pose CNN",
                "ba": cnn["at_fixed_threshold_0.5"]["balanced_accuracy"],
                "lo": cnn["clip_bootstrap_at_0.5"]["balanced_accuracy"]["ci_lower_95"],
                "hi": cnn["clip_bootstrap_at_0.5"]["balanced_accuracy"]["ci_upper_95"],
            }
        )
    if VMAE.exists():
        vmae = json.loads(VMAE.read_text())
        rows.append(
            {
                "name": "VideoMAE 1.5 s",
                "ba": vmae["balanced_accuracy"],
                "lo": vmae["clip_bootstrap"]["balanced_accuracy"]["ci_lower_95"],
                "hi": vmae["clip_bootstrap"]["balanced_accuracy"]["ci_upper_95"],
            }
        )
    fig, ax = plt.subplots(figsize=SIZE_FULL, facecolor=PAPER)
    ys = np.arange(len(rows))
    ax.axvline(0.5, color=INK, ls="--", lw=1.0)
    ax.barh(ys, [r["ba"] for r in rows], color=BLUE, height=0.55)
    ax.errorbar(
        [r["ba"] for r in rows],
        ys,
        xerr=[[r["ba"] - r["lo"] for r in rows], [r["hi"] - r["ba"] for r in rows]],
        fmt="none",
        ecolor=INK,
        capsize=2,
    )
    ax.set_yticks(ys, [r["name"] for r in rows])
    ax.set_xlabel("Balanced accuracy")
    ax.set_xlim(0.0, 0.90)
    ax.set_title("Figure C. DEV only. Chance 0.500. TEST not scored.")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, out_dir / "figures" / "figureC_dev_comparison")


def figure_d(out_dir: Path) -> None:
    m = json.loads((out_dir / "metrics.json").read_text())["at_fixed_threshold_0.5"]
    mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=SIZE_FULL, facecolor=PAPER)
    ax.imshow(mat, cmap="Greys")
    for i in range(2):
        for j in range(2):
            n = int(mat[i, j])
            pct = 100.0 * n / mat.sum()
            ax.text(j, i, f"{n}\n({pct:.1f}%)", ha="center", va="center", color=INK)
    ax.set_xticks([0, 1], ["Pred no", "Pred nod"])
    ax.set_yticks([0, 1], ["True no", "True nod"])
    ax.set_title("Figure D. DEV audio confusion. TEST not scored.")
    fig.tight_layout()
    save(fig, out_dir / "figures" / "figureD_confusion")


def write_figures(out_dir: Path | None = None) -> None:
    out_dir = Path(out_dir or OUT)
    if set(load_windows(WINDOWS, "DEV", DEV_IDS)["sample_id"]) & TEST_IDS:
        raise SystemExit("STOP: TEST id in figure windows")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    figure_a(out_dir)
    figure_b(out_dir)
    if (out_dir / "metrics.json").exists():
        figure_c(out_dir)
        figure_d(out_dir)
    print(f"wrote figures under {out_dir / 'figures'}")


if __name__ == "__main__":
    write_figures()
