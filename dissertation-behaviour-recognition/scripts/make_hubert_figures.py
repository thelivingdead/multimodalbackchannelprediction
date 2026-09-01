#!/usr/bin/env python3
"""GOLD DEV HuBERT figures from saved artefacts. Does not score TEST.

Reads only:

    results/hubert_dev/permutation_metrics.csv
    results/hubert_dev/permutation_summary.json
    results/hubert_dev/multimodal_dev_comparison.csv

Does not retrain, re-extract embeddings, or recompute metrics. Missing
inputs abort. GOLD TEST ids (gold_016–gold_030) in any opened file abort.

    cd dissertation-behaviour-recognition
    MPLCONFIGDIR=./.mplconfig OMP_NUM_THREADS=1 \\
        python scripts/make_hubert_figures.py
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "figures" / "paper"
HUBERT_DEV = ROOT / "results" / "hubert_dev"
PERM_CSV = HUBERT_DEV / "permutation_metrics.csv"
PERM_JSON = HUBERT_DEV / "permutation_summary.json"
COMPARE_CSV = HUBERT_DEV / "multimodal_dev_comparison.csv"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe–Ito (same pair as other F1 vs balanced-accuracy panels).
BLUE = "#0072B2"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREY = "#999999"
INK = "#111827"
MUTED = "#4b5563"
F1_COLOR = BLUE
BA_COLOR = ORANGE

TEST_ID_RE = re.compile(r"gold_0*(1[6-9]|2[0-9]|30)(?!\d)", re.IGNORECASE)
TEST_PATH_RE = re.compile(r"(?:^|[/\\])test(?:[/\\]|$)", re.IGNORECASE)
TEST_MSG = "REFUSING TO SCORE GOLD TEST FOR HUBERT DEVELOPMENT EXPERIMENT"

REQUIRED_SUMMARY = (
    "n_permutations",
    "p_F1",
    "p_BA",
    "n_perm_F1_ge_actual",
    "n_perm_BA_ge_actual",
    "permutation_mean_F1",
    "permutation_mean_BA",
    "actual_F1",
    "actual_BA",
)

COMPARE_ROWS = (
    ("Always positive", "Always positive baseline"),
    ("RGB only", "Frozen RGB VideoMAE"),
    ("50/50 HuBERT+RGB probability fusion", "Equal weight fusion"),
    ("HuBERT only", "Frozen HuBERT (mixed audio)"),
)


def abort_test(detail: str) -> None:
    print(TEST_MSG)
    print(detail)
    raise SystemExit(TEST_MSG)


def refuse_test_path(path: Path) -> None:
    resolved = str(path.resolve())
    if TEST_PATH_RE.search(resolved):
        abort_test(f"TEST path refused: {path}")


def require_file(path: Path) -> None:
    refuse_test_path(path)
    if not path.is_file():
        raise SystemExit(f"STOP: required input missing: {path}")


def scan_text_for_test_ids(text: str, path: Path) -> None:
    hit = TEST_ID_RE.search(text)
    if hit:
        abort_test(f"{path} contains TEST identifier {hit.group(0)!r}")


def read_text(path: Path) -> str:
    require_file(path)
    text = path.read_text(encoding="utf-8")
    scan_text_for_test_ids(text, path)
    return text


def load_summary(path: Path) -> dict:
    raw = json.loads(read_text(path))
    if not isinstance(raw, dict):
        raise SystemExit(f"STOP: {path} is not a JSON object")
    missing = [k for k in REQUIRED_SUMMARY if k not in raw]
    if missing:
        raise SystemExit(f"STOP: {path} missing keys: {missing}")
    if raw.get("gold_test_scored") is True:
        abort_test(f"{path} has gold_test_scored=true")
    for flag in ("AUDIO_TEST_SCORED", "FUSION_TEST_SCORED"):
        if str(raw.get(flag, "NO")).strip().upper() not in ("NO", "FALSE", ""):
            abort_test(f"{path} has {flag}={raw.get(flag)!r}")
    split = str(raw.get("split", "DEV")).strip().upper()
    if split == "TEST":
        abort_test(f"{path} split is TEST")
    return raw


def load_permutation_rows(path: Path, n_expected: int) -> tuple[np.ndarray, np.ndarray]:
    text = read_text(path)
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise SystemExit(f"STOP: {path} has no header")
    need = {"F1", "balanced_accuracy"}
    have = set(reader.fieldnames)
    if not need <= have:
        raise SystemExit(f"STOP: {path} missing columns {sorted(need - have)}")
    f1: list[float] = []
    ba: list[float] = []
    for row in reader:
        try:
            f1.append(float(row["F1"]))
            ba.append(float(row["balanced_accuracy"]))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"STOP: {path} has a non-numeric metric row") from exc
    if len(f1) != n_expected:
        raise SystemExit(
            f"STOP: {path} has {len(f1)} data rows; summary n_permutations={n_expected}"
        )
    return np.asarray(f1, dtype=float), np.asarray(ba, dtype=float)


def load_comparison(path: Path) -> list[tuple[str, float, float]]:
    text = read_text(path)
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise SystemExit(f"STOP: {path} has no header")
    need = {"model", "f1", "balanced_accuracy", "split"}
    have = set(reader.fieldnames or [])
    if not need <= have:
        raise SystemExit(f"STOP: {path} missing columns {sorted(need - have)}")
    by_model: dict[str, dict[str, str]] = {}
    for row in reader:
        split = str(row["split"]).strip().upper()
        if split == "TEST":
            abort_test(f"{path} contains a TEST split row")
        if split != "DEV":
            raise SystemExit(f"STOP: {path} row split={row['split']!r} is not DEV")
        by_model[str(row["model"]).strip()] = row
    out: list[tuple[str, float, float]] = []
    for csv_name, label in COMPARE_ROWS:
        if csv_name not in by_model:
            raise SystemExit(f"STOP: {path} missing model {csv_name!r}")
        row = by_model[csv_name]
        try:
            out.append((label, float(row["f1"]), float(row["balanced_accuracy"])))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"STOP: {path} non-numeric metrics for {csv_name!r}") from exc
    return out


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, stem: str) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    png = PAPER / f"{stem}.png"
    pdf = PAPER / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"wrote {png.relative_to(ROOT)}")


def _panel_hist(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    xlabel: str,
    p_value: float,
    n_ge: int,
    n_perm: int,
    mean_v: float,
    observed_v: float,
    title_metric: str,
) -> None:
    ax.hist(values, bins=25, color=SKY, edgecolor="white", linewidth=0.4, zorder=1)
    ax.axvline(
        mean_v,
        color=INK,
        linestyle=":",
        linewidth=1.4,
        label=f"permutation mean = {mean_v:.3f}",
        zorder=3,
    )
    ax.axvline(
        observed_v,
        color=ORANGE,
        linestyle="--",
        linewidth=1.5,
        label=f"observed = {observed_v:.3f}",
        zorder=3,
    )
    lo = float(min(values.min(), mean_v, observed_v))
    hi = float(max(values.max(), mean_v, observed_v))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("permutations")
    ax.set_title(f"{title_metric} p = {p_value:.3f} ({int(n_ge)}/{int(n_perm)})")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)


def fig_permutation_null(summary: dict, f1: np.ndarray, ba: np.ndarray) -> None:
    n_perm = int(summary["n_permutations"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15), layout="constrained", sharex=False, sharey=False)
    _panel_hist(
        axes[0],
        ba,
        xlabel="balanced accuracy",
        p_value=float(summary["p_BA"]),
        n_ge=int(summary["n_perm_BA_ge_actual"]),
        n_perm=n_perm,
        mean_v=float(summary["permutation_mean_BA"]),
        observed_v=float(summary["actual_BA"]),
        title_metric="Balanced accuracy",
    )
    _panel_hist(
        axes[1],
        f1,
        xlabel="F1",
        p_value=float(summary["p_F1"]),
        n_ge=int(summary["n_perm_F1_ge_actual"]),
        n_perm=n_perm,
        mean_v=float(summary["permutation_mean_F1"]),
        observed_v=float(summary["actual_F1"]),
        title_metric="F1",
    )
    fig.suptitle(
        f"Permutation null, {n_perm} shuffled-label refits (GOLD DEV, TEST not scored)",
        fontsize=10,
        color=INK,
    )
    save_fig(fig, "permutation_null")


def fig_acoustic_dev_comparison(rows: list[tuple[str, float, float]]) -> None:
    labels = [r[0] for r in rows]
    f1s = np.asarray([r[1] for r in rows], dtype=float)
    bas = np.asarray([r[2] for r in rows], dtype=float)
    y = np.arange(len(labels))
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.4), layout="constrained")
    ax.barh(
        y - h / 2,
        f1s,
        height=h,
        color=F1_COLOR,
        edgecolor="none",
        label="F1",
        zorder=2,
    )
    ax.barh(
        y + h / 2,
        bas,
        height=h,
        color=BA_COLOR,
        edgecolor="none",
        hatch="///",
        label="balanced accuracy",
        zorder=2,
    )
    for yi, f1, ba in zip(y, f1s, bas):
        ax.text(f1 + 0.018, yi - h / 2, f"{f1:.3f}", va="center", ha="left", fontsize=8, color=INK)
        ax.text(ba + 0.018, yi + h / 2, f"{ba:.3f}", va="center", ha="left", fontsize=8, color=INK)
    ax.axvline(0.5, color=GREY, linestyle="--", linewidth=1.15, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.18)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("score", labelpad=4)
    ax.set_title(
        "GOLD DEV only, n = 15, fixed threshold 0.5  ·  GOLD TEST not scored",
        fontsize=10,
        pad=10,
    )
    # Sit to the left of 0.5, below the axis — not on the title, not on a bar.
    ax.annotate(
        "chance balanced accuracy",
        xy=(0.5, 0.0),
        xycoords=("data", "axes fraction"),
        xytext=(-52, -22),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
        annotation_clip=False,
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    save_fig(fig, "acoustic_dev_comparison")


def main() -> None:
    _style()
    summary = load_summary(PERM_JSON)
    n_perm = int(summary["n_permutations"])
    f1, ba = load_permutation_rows(PERM_CSV, n_perm)
    rows = load_comparison(COMPARE_CSV)
    PAPER.mkdir(parents=True, exist_ok=True)
    fig_permutation_null(summary, f1, ba)
    fig_acoustic_dev_comparison(rows)
    print("done ->", PAPER)
    print("AUDIO_TEST_SCORED = NO")
    print("FUSION_TEST_SCORED = NO")
    print("LOCKED_TEST_RESULTS_MODIFIED = NO")


if __name__ == "__main__":
    main()
