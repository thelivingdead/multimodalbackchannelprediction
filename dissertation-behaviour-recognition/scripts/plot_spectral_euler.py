#!/usr/bin/env python3
"""Mean power spectrum of EMOCA Euler axes, grouped by gold labels.

Illustration only. Scored systems in this dissertation are amplitude rules
+ 1D CNN + VideoMAE, not an FFT detector. This figure is qualitative
support for *why* the nod rule uses axis x and the shake rule uses axis z.

Needs ``features/gold/gold_*.npz`` with ``rotation_xyz`` (present on Mac
and otter). No GPU.

Otter95::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/plot_spectral_euler.py

Mac::

    cd "/Users/divyabisht/Downloads/Msc Dissertation Divya/dissertation-behaviour-recognition"
    MPLCONFIGDIR=./.mplconfig OMP_NUM_THREADS=1 ../.venv/bin/python \\
        scripts/plot_spectral_euler.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
FEAT = ROOT / "features" / "gold"
OUT = ROOT / "figures" / "paper"
FPS = 25.0
N_FFT = 512
BLUE = "#4e79a7"
ORANGE = "#f28e2b"
GREEN = "#59a14f"
GREY = "#6b7280"
RED = "#e15759"


def load_rot(sample_id: str) -> np.ndarray | None:
    path = FEAT / f"{sample_id}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        if "rotation_xyz" not in z.files:
            return None
        return np.asarray(z["rotation_xyz"], dtype=float)


def resample(x: np.ndarray, n: int = N_FFT) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) == 0:
        return np.zeros(n, dtype=float)
    old = np.linspace(0.0, 1.0, len(x))
    new = np.linspace(0.0, 1.0, n)
    return np.interp(new, old, x)


def mean_spectrum(series: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    specs = []
    for x in series:
        y = resample(x)
        y = y - np.nanmean(y)
        spec = (np.abs(np.fft.rfft(y)) ** 2) / len(y)
        specs.append(spec)
    freq = np.fft.rfftfreq(N_FFT, d=1.0 / FPS)
    return freq, np.mean(np.stack(specs), axis=0)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    jpg = OUT / f"{stem}.jpg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(jpg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", png.relative_to(ROOT))


def main() -> None:
    if not SHAKE_GOLD.exists():
        raise SystemExit(f"STOP: {SHAKE_GOLD} missing")
    gold = pd.read_csv(SHAKE_GOLD)
    need = {"sample_id", "nod_label", "shake_label"}
    if not need <= set(gold.columns):
        raise SystemExit(
            f"STOP: shake gold needs {sorted(need)}; got {list(gold.columns)}"
        )

    missing = []
    nod_yes, nod_no = [], []
    shk_yes, shk_no = [], []
    for r in gold.itertuples():
        sid = str(r.sample_id)
        rot = load_rot(sid)
        if rot is None or rot.ndim != 2 or rot.shape[1] < 3:
            missing.append(sid)
            continue
        if int(r.nod_label) == 1:
            nod_yes.append(rot[:, 0])
        else:
            nod_no.append(rot[:, 0])
        if int(r.shake_label) == 1:
            shk_yes.append(rot[:, 2])
        else:
            shk_no.append(rot[:, 2])

    if missing:
        print(
            f"NOTE: {len(missing)} gold npz missing on this machine "
            f"(script still runnable on otter): {missing[:8]}"
        )
    if len(nod_yes) < 2 or len(nod_no) < 2 or len(shk_yes) < 2 or len(shk_no) < 2:
        raise SystemExit(
            "STOP: not enough gold Euler traces to plot spectra. "
            "On otter, features/gold/gold_001.npz … gold_030.npz must exist. "
            "No figure invented."
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=False)
    fx, px_yes = mean_spectrum(nod_yes)
    _, px_no = mean_spectrum(nod_no)
    axes[0].plot(fx, px_yes, color=GREEN, lw=1.6, label=f"gold nod (n={len(nod_yes)})")
    axes[0].plot(fx, px_no, color=GREY, lw=1.6, label=f"gold not-nod (n={len(nod_no)})")
    axes[0].set_xlim(0, 6)
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("mean power (a.u.)")
    axes[0].set_title("Euler axis x  (nod grouping)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    fz, pz_yes = mean_spectrum(shk_yes)
    _, pz_no = mean_spectrum(shk_no)
    axes[1].plot(fz, pz_yes, color=ORANGE, lw=1.6,
                 label=f"gold shake (n={len(shk_yes)})")
    axes[1].plot(fz, pz_no, color=BLUE, lw=1.6,
                 label=f"gold not-shake (n={len(shk_no)})")
    axes[1].set_xlim(0, 6)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_title("Euler axis z  (shake grouping)")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle(
        "Illustration only — mean power spectrum of EMOCA Euler. "
        "Scored systems are amplitude rules + CNN/VideoMAE, not FFT.",
        fontsize=10, color="#111827",
    )
    fig.text(
        0.01, -0.04,
        "Gold windows only (DEV+TEST, n=30). Not a TEST metric. "
        "Nod rule uses axis x; shake rule uses axis z. EMOCA/FLAME pose is used, not trained.",
        fontsize=8, color=GREY,
    )
    save(fig, "euler_power_spectrum")


if __name__ == "__main__":
    main()
