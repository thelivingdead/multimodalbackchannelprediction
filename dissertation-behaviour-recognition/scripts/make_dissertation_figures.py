#!/usr/bin/env python3
"""Dissertation-quality figures from locked artefacts. Does not rescore TEST.

Reads existing json/csv/md-backed numbers and gold Euler npz. Writes PNG
(300 dpi) + PDF under figures/paper/. Never writes into results/*metrics*.json.

    cd dissertation-behaviour-recognition
    MPLCONFIGDIR=./.mplconfig OMP_NUM_THREADS=1 \\
        ../.venv/bin/python scripts/make_dissertation_figures.py
"""
from __future__ import annotations

import json
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RES = ROOT / "results"
FEAT = ROOT / "features" / "gold"
PAPER = ROOT / "figures" / "paper"
GOLD_CSV = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
FPS = 25.0
N_FFT = 512

# Okabe–Ito (colour-blind safe; not red–green only).
BLUE = "#0072B2"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
VERM = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#999999"
INK = "#111827"
MUTED = "#4b5563"

AXIS_NAME = {
    "x": "pitch (nod-like)",
    "y": "yaw (shake-like)",
    "z": "roll (tilt-like)",
}


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
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


def save_fig(fig: plt.Figure, stem: str) -> Path:
    PAPER.mkdir(parents=True, exist_ok=True)
    png = PAPER / f"{stem}.png"
    pdf = PAPER / f"{stem}.pdf"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.22)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"wrote {png.relative_to(ROOT)}  {pdf.name}")
    return png


def skip(name: str, reason: str) -> None:
    print(f"SKIP  {name}: {reason}")


def draw_metric_table(ax, rows: list[list[str]], title: str) -> None:
    ax.axis("off")
    ax.set_title(title, pad=10, fontsize=11)
    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.65)
    n_cols = len(rows[0])
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.6)
        if c == 0:
            cell.set_width(0.40)
            cell.set_text_props(ha="left")
            cell.PAD = 0.08
        else:
            cell.set_width(0.58 / (n_cols - 1))
        if r == 0:
            cell.set_facecolor("#003D5B")
            cell.set_text_props(color="white", fontweight="bold", ha="center" if c else "left")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f4f6")
        else:
            cell.set_facecolor("white")


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# 1. Nod GOLD TEST
# ---------------------------------------------------------------------------
NOD_KEEP = [
    "Rule baseline",
    "Pose CNN xyz_deriv",
    "Frozen VideoMAE head",
    "Fine-tuned VideoMAE (last 4 blocks)",
    "Fine-tuned VideoMAE (last 4 blocks, 200 pseudo)",
]
NOD_LABELS = {
    "Rule baseline": "Pose rule",
    "Pose CNN xyz_deriv": "Pose CNN",
    "Frozen VideoMAE head": "Frozen VideoMAE",
    "Fine-tuned VideoMAE (last 4 blocks)": "Fine-tuned VideoMAE  n=80",
    "Fine-tuned VideoMAE (last 4 blocks, 200 pseudo)": "Fine-tuned VideoMAE  n=200",
}
NOD_COLORS = [BLUE, GREEN, VERM, ORANGE, PURPLE]


def fig_nod_test() -> None:
    path = RES / "tables" / "main_results.csv"
    if not path.exists():
        skip("nod_test_f1", f"missing {path}")
        return
    tab = pd.read_csv(path)
    df = tab[tab["model"].isin(NOD_KEEP)].copy()
    df["model"] = pd.Categorical(df["model"], NOD_KEEP, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)
    labels = [NOD_LABELS[m] for m in df["model"]]
    f1 = df["f1"].to_numpy(float)
    p = df["precision"].to_numpy(float)
    r = df["recall"].to_numpy(float)
    lo = pd.to_numeric(df["f1_ci_lo"], errors="coerce").to_numpy()
    hi = pd.to_numeric(df["f1_ci_hi"], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(5.6, 3.6), layout="constrained")
    y = np.arange(len(df))
    ax.barh(y, f1, color=NOD_COLORS, height=0.62, zorder=2, edgecolor="none")
    xerr = np.vstack([f1 - lo, hi - f1])
    ax.errorbar(
        f1, y, xerr=xerr, fmt="none", ecolor=INK, capsize=3.5, linewidth=1.05, zorder=3
    )
    right = np.nanmax(hi) + 0.04
    for i in range(len(df)):
        ax.text(
            right,
            y[i],
            f"{f1[i]:.2f}      P {p[i]:.2f}     R {r[i]:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color=INK,
            fontfamily="DejaVu Sans",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.62)
    ax.set_xlabel("TEST F1")
    ax.set_title("Nod  ·  GOLD TEST  n=15, scored once")
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    despine(ax)
    save_fig(fig, "nod_test_f1")


def fig_nod_confusion() -> None:
    """Small table-as-figure from locked TEST json (no rescoring)."""
    sources = [
        ("Pose rule", RES / "rule_test_metrics.json", None),
        ("Pose CNN", RES / "classifier_test_metrics.json", None),
        ("Frozen VMAE", RES / "videomae_frozen_head" / "metrics.json", "test_metrics"),
        ("FT VMAE n=80", RES / "videomae_finetuned" / "metrics.json", "test_metrics"),
        ("FT VMAE n=200", RES / "videomae_finetuned_n200" / "metrics.json", "test_metrics"),
    ]
    rows = [["Method", "TN", "FP", "FN", "TP", "P", "R", "F1"]]
    for name, path, key in sources:
        if not path.exists():
            skip("nod_test_confusion", f"missing {path}")
            return
        blob = _load_json(path)
        m = blob[key] if key else blob
        rows.append(
            [
                name,
                str(int(m["tn"])),
                str(int(m["fp"])),
                str(int(m["fn"])),
                str(int(m["tp"])),
                f"{float(m['precision']):.2f}",
                f"{float(m['recall']):.2f}",
                f"{float(m['f1']):.2f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(6.4, 2.75), layout="constrained")
    draw_metric_table(ax, rows, "Nod GOLD TEST  n=15  ·  confusion counts (scored once)")
    save_fig(fig, "nod_test_confusion")


# ---------------------------------------------------------------------------
# 2. Shake GOLD TEST (already-written files; not a new score)
# ---------------------------------------------------------------------------
def fig_shake_test() -> None:
    needed = {
        "rule": RES / "shake" / "rule_test_metrics.json",
        "base": RES / "shake" / "majority_baseline" / "metrics.json",
        "cnn": RES / "shake" / "cnn" / "metrics.json",
        "frz": RES / "shake" / "videomae_frozen_head" / "metrics.json",
        "ft": RES / "shake" / "videomae_finetuned" / "metrics.json",
    }
    missing = [str(p) for p in needed.values() if not p.exists()]
    if missing:
        skip("shake_test_f1", "missing " + ", ".join(missing))
        return
    rule = _load_json(needed["rule"])
    base = _load_json(needed["base"])["always_positive"]
    cnn = _load_json(needed["cnn"])["test_metrics"]
    frz = _load_json(needed["frz"])["test_metrics"]
    ft = _load_json(needed["ft"])["test_metrics"]
    rows = [
        ("Pose rule  (axis z)", rule, GREY),
        ("Always-shake", base, "#bdbdbd"),
        ("Pose CNN  75/5", cnn, BLUE),
        ("Frozen VideoMAE  75/5", frz, VERM),
        ("Fine-tuned VideoMAE  75/5", ft, ORANGE),
    ]
    labels = [r[0] for r in rows]
    f1 = np.array([float(r[1]["f1"]) for r in rows])
    p = np.array([float(r[1]["precision"]) for r in rows])
    rec = np.array([float(r[1]["recall"]) for r in rows])
    colors = [r[2] for r in rows]
    always_f1 = float(base["f1"])

    fig, ax = plt.subplots(figsize=(5.6, 3.6), layout="constrained")
    y = np.arange(len(rows))
    ax.barh(y, f1, color=colors, height=0.62, zorder=2, edgecolor="none")
    ax.axvline(always_f1, color=GREY, ls="--", lw=1.1, zorder=1)
    for i in range(len(rows)):
        ax.text(
            1.04,
            y[i],
            f"{f1[i]:.2f}      P {p[i]:.2f}     R {rec[i]:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.62)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("TEST F1")
    ax.set_title("Head-shake  ·  GOLD TEST  n=15, scored once  ·  rule axis z")
    despine(ax)
    save_fig(fig, "shake_test_f1")


def fig_shake_confusion() -> None:
    needed = [
        ("Pose rule (z)", RES / "shake" / "rule_test_metrics.json", None),
        ("Always-shake", RES / "shake" / "majority_baseline" / "metrics.json", "always_positive"),
        ("CNN 75/5", RES / "shake" / "cnn" / "metrics.json", "test_metrics"),
        ("Frozen VMAE 75/5", RES / "shake" / "videomae_frozen_head" / "metrics.json", "test_metrics"),
        ("FT VMAE 75/5", RES / "shake" / "videomae_finetuned" / "metrics.json", "test_metrics"),
    ]
    rows = [["Method", "TN", "FP", "FN", "TP", "P", "R", "F1"]]
    for name, path, key in needed:
        if not path.exists():
            skip("shake_test_confusion", f"missing {path}")
            return
        blob = _load_json(path)
        m = blob[key] if key else blob
        rows.append(
            [
                name,
                str(int(m["tn"])),
                str(int(m["fp"])),
                str(int(m["fn"])),
                str(int(m["tp"])),
                f"{float(m['precision']):.2f}",
                f"{float(m['recall']):.2f}",
                f"{float(m['f1']):.2f}",
            ]
        )
    fig, ax = plt.subplots(figsize=(6.4, 2.75), layout="constrained")
    draw_metric_table(ax, rows, "Shake GOLD TEST  n=15  ·  confusion counts (scored once)")
    save_fig(fig, "shake_test_confusion")


# ---------------------------------------------------------------------------
# 3. Shake DEV-only (TEST not scored)
# ---------------------------------------------------------------------------
DEV_DISPLAY = [
    (
        "always-shake baseline (DEV)",
        "Always-shake  (baseline)",
        "baseline",
    ),
    ("locked 75/5 pose CNN", "CNN 75/5  (locked)", "collapsed"),
    ("locked 75/5 frozen VideoMAE", "Frozen VideoMAE 75/5  (locked)", "locked"),
    ("locked 75/5 fine-tuned VideoMAE", "FT VideoMAE 75/5  (locked)", "collapsed"),
    ("search:cnn_20_20_highconf", "CNN 20/20 high-conf", "collapsed"),
    ("search:cnn_40_40", "CNN 40/40  (selected on DEV)", "selected"),
    ("search:vmae_frozen_20_20_highconf", "Frozen VideoMAE 20/20 high-conf", "ok"),
    ("search:vmae_frozen_40_40", "Frozen VideoMAE 40/40", "ok"),
    ("search:vmae_ft4_40_40", "FT VideoMAE 40/40", "ok"),
]
KIND_COLOR = {
    "baseline": GREY,
    "collapsed": SKY,
    "locked": BLUE,
    "selected": ORANGE,
    "ok": BLUE,
}


def fig_shake_dev() -> None:
    path = RES / "shake" / "dev_search" / "summary.csv"
    if not path.exists():
        skip("shake_dev_only_f1", f"missing {path}")
        return
    tab = pd.read_csv(path)
    by_sys = {str(r.system): r for r in tab.itertuples()}
    labels, f1s, colors, hatch = [], [], [], []
    for key, label, kind in DEV_DISPLAY:
        if key not in by_sys:
            skip("shake_dev_only_f1", f"missing row {key} in {path}")
            return
        row = by_sys[key]
        labels.append(label)
        f1s.append(float(row.f1))
        colors.append(KIND_COLOR[kind])
        collapsed = str(row.collapse).lower() in {"true", "1"}
        hatch.append("///" if collapsed else None)
    f1s = np.array(f1s)
    always = float(by_sys["always-shake baseline (DEV)"].f1)

    fig, ax = plt.subplots(figsize=(5.6, 4.7), layout="constrained")
    y = np.arange(len(labels))
    bars = ax.barh(y, f1s, color=colors, height=0.66, zorder=2, edgecolor=INK, linewidth=0.35)
    for bar, h in zip(bars, hatch):
        if h:
            bar.set_hatch(h)
            bar.set_edgecolor(MUTED)
    ax.axvline(always, color=GREY, ls="--", lw=1.15, zorder=1)
    for i, val in enumerate(f1s):
        ax.text(1.02, y[i], f"{val:.3f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.28)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("DEV F1")
    ax.set_title("Head-shake  ·  DEV ONLY  n=15  ·  GOLD TEST not scored")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=GREY, edgecolor="none", label="Always-shake baseline (F1 0.80)"),
            Patch(facecolor=SKY, edgecolor=MUTED, hatch="///", label="Collapsed on DEV"),
            Patch(facecolor=BLUE, edgecolor="none", label="Non-collapsed / locked"),
            Patch(facecolor=ORANGE, edgecolor="none", label="Selected on DEV (CNN 40/40)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    despine(ax)
    save_fig(fig, "shake_dev_only_f1")


# ---------------------------------------------------------------------------
# 4. Euler signal illustration (spectrum + traces + DEV axis audit)
# ---------------------------------------------------------------------------
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


def fig_euler_spectrum() -> None:
    if not GOLD_CSV.exists():
        skip("euler_signal_spectrum", f"missing {GOLD_CSV}")
        return
    gold = pd.read_csv(GOLD_CSV)
    buckets: dict[str, list[np.ndarray]] = {
        "nod_yes": [],
        "nod_no": [],
        "shk_yes_x": [],
        "shk_no_x": [],
        "shk_yes_y": [],
        "shk_no_y": [],
        "shk_yes_z": [],
        "shk_no_z": [],
    }
    missing = []
    for r in gold.itertuples():
        sid = str(r.sample_id)
        rot = load_rot(sid)
        if rot is None or rot.ndim != 2 or rot.shape[1] < 3:
            missing.append(sid)
            continue
        if int(r.nod_label) == 1:
            buckets["nod_yes"].append(rot[:, 0])
        else:
            buckets["nod_no"].append(rot[:, 0])
        if int(r.shake_label) == 1:
            buckets["shk_yes_x"].append(rot[:, 0])
            buckets["shk_yes_y"].append(rot[:, 1])
            buckets["shk_yes_z"].append(rot[:, 2])
        else:
            buckets["shk_no_x"].append(rot[:, 0])
            buckets["shk_no_y"].append(rot[:, 1])
            buckets["shk_no_z"].append(rot[:, 2])
    if missing:
        print(f"NOTE: {len(missing)} gold npz missing: {missing[:6]}")
    if min(len(buckets[k]) for k in buckets) < 2:
        skip("euler_signal_spectrum", "not enough gold Euler traces")
        return

    panels = [
        (
            "x  ·  pitch  ·  grouped by nod label",
            buckets["nod_yes"],
            buckets["nod_no"],
            f"gold nod (n={len(buckets['nod_yes'])})",
            f"gold not-nod (n={len(buckets['nod_no'])})",
            BLUE,
            GREY,
        ),
        (
            "y  ·  yaw  ·  grouped by shake label",
            buckets["shk_yes_y"],
            buckets["shk_no_y"],
            f"gold shake (n={len(buckets['shk_yes_y'])})",
            f"gold not-shake (n={len(buckets['shk_no_y'])})",
            ORANGE,
            GREY,
        ),
        (
            "z  ·  roll  ·  grouped by shake label (locked TEST rule used z)",
            buckets["shk_yes_z"],
            buckets["shk_no_z"],
            f"gold shake (n={len(buckets['shk_yes_z'])})",
            f"gold not-shake (n={len(buckets['shk_no_z'])})",
            VERM,
            GREY,
        ),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 7.6), sharex=True)
    fig.subplots_adjust(hspace=0.55, left=0.18, right=0.97, top=0.97, bottom=0.07)
    for ax, (title, yes, no, lab_yes, lab_no, c_yes, c_no) in zip(axes, panels):
        freq, p_yes = mean_spectrum(yes)
        _, p_no = mean_spectrum(no)
        ax.plot(freq, p_yes, color=c_yes, lw=1.7, label=lab_yes)
        ax.plot(freq, p_no, color=c_no, lw=1.5, label=lab_no)
        ax.set_xlim(0, 6)
        ymax = max(float(np.nanmax(p_yes)), float(np.nanmax(p_no)))
        ax.set_ylim(0, ymax * 1.12)
        ax.set_ylabel("Mean power (a.u.)")
        ax.set_title(title, loc="left", fontsize=10, pad=8)
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="#e5e7eb",
            fontsize=8,
        )
        despine(ax)
    axes[-1].set_xlabel("Frequency (Hz)")
    save_fig(fig, "euler_signal_spectrum")


def fig_euler_traces() -> None:
    """One nod-only and one shake-only DEV window. Illustration, not a score."""
    specs = [
        ("gold_009", "Nod-only  (gold_009, DEV)\nnod=1, shake=0"),
        ("gold_004", "Shake-only  (gold_004, DEV)\nnod=0, shake=1"),
    ]
    colors = {"x": BLUE, "y": ORANGE, "z": GREEN}
    fig, axes = plt.subplots(
        3, 2, figsize=(5.6, 5.8), sharex=True, layout="constrained"
    )
    any_missing = False
    for col, (sid, header) in enumerate(specs):
        rot = load_rot(sid)
        if rot is None:
            any_missing = True
            for row in range(3):
                axes[row, col].text(
                    0.5, 0.5, f"{sid}\nnpz missing", ha="center", va="center",
                    transform=axes[row, col].transAxes, fontsize=9, color=MUTED,
                )
            continue
        t = np.arange(len(rot)) / FPS
        for row, ax_key in enumerate("xyz"):
            ax = axes[row, col]
            ax.plot(t, rot[:, row], color=colors[ax_key], lw=1.15)
            ax.axhline(0.0, color="#e5e7eb", lw=0.7, zorder=0)
            despine(ax)
            if col == 0:
                ax.set_ylabel(f"{ax_key}  (°)")
            if row == 0:
                ax.set_title(header, fontsize=10, pad=6)
            if row == 2:
                ax.set_xlabel("Time (s)")
            # Right-side axis name once per row, on the right column.
            if col == 1:
                ax.text(
                    1.03,
                    0.5,
                    AXIS_NAME[ax_key],
                    transform=ax.transAxes,
                    va="center",
                    ha="left",
                    fontsize=8,
                    color=MUTED,
                    rotation=0,
                )
    fig.suptitle("Euler traces  ·  illustration of the signal, not the classifier", fontsize=11, y=1.02)
    if any_missing:
        print("NOTE: at least one gold npz missing for euler_nod_vs_shake_traces")
    save_fig(fig, "euler_nod_vs_shake_traces")


def fig_axis_audit_dev() -> None:
    conc_path = RES / "shake" / "dev_search" / "axis_audit_conclusion.json"
    csv_path = RES / "shake" / "v2" / "axis_audit" / "dev_axis_stats.csv"
    if not conc_path.exists():
        skip("euler_axis_audit_dev", f"missing {conc_path}")
        return
    conc = _load_json(conc_path)
    summary = conc["axis_summary"]
    axes_keys = ["x", "y", "z"]
    shake_pos = [float(summary[a]["mean_rule_shake_pos"]) for a in axes_keys]
    shake_neg = [float(summary[a]["mean_rule_shake_neg"]) for a in axes_keys]
    nod_only = [float(summary[a]["mean_rule_nod_only"]) for a in axes_keys]
    shake_only = [float(summary[a]["mean_rule_shake_only"]) for a in axes_keys]
    xticklabels = ["pitch (x)", "yaw (y)", "roll (z)"]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5.4, 5.9), layout="constrained")
    fig.set_constrained_layout_pads(h_pad=0.10, hspace=0.18)
    x = np.arange(3)
    w = 0.36
    ax0.bar(x - w / 2, shake_neg, w, color=GREY, label="shake−  (n=5)", zorder=2)
    ax0.bar(x + w / 2, shake_pos, w, color=BLUE, label="shake+  (n=10)", zorder=2)
    ax0.set_xticks(x)
    ax0.set_xticklabels(xticklabels, fontsize=9)
    ax0.set_ylabel("Mean rule amplitude (°)")
    ax0.set_title("GOLD DEV  ·  by shake label", fontsize=10)
    ax0.legend(loc="upper right", frameon=False, fontsize=8)
    ax0.set_ylim(0, 48)
    despine(ax0)

    ax1.bar(x - w / 2, shake_only, w, color=ORANGE, label="shake-only  (n=4)", zorder=2)
    ax1.bar(x + w / 2, nod_only, w, color=SKY, label="nod-only  (n=3)", zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xticklabels, fontsize=9)
    ax1.set_ylabel("Mean rule amplitude (°)")
    ax1.set_title("GOLD DEV  ·  exclusive labels (no co-occurring class)", fontsize=10)
    ax1.legend(loc="upper right", frameon=False, fontsize=8)
    ax1.set_ylim(0, 48)
    despine(ax1)
    fig.suptitle("DEV axis audit  ·  illustration of the signal, not the detector", fontsize=11)
    save_fig(fig, "euler_axis_audit_dev")

    if not csv_path.exists():
        skip("euler_axis_strips_dev", f"missing {csv_path}")
        return
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(42)
    panel_title = {
        "x": "x  ·  pitch (nod-like)",
        "y": "y  ·  yaw (geometric shake)",
        "z": "z  ·  roll (locked TEST rule)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 3.6), sharey=True, layout="constrained")
    for i, ax_key in enumerate(axes_keys):
        ax = axes[i]
        for j, (lab, color, name) in enumerate(
            [(0, GREY, "shake−"), (1, BLUE, "shake+")]
        ):
            vals = df.loc[df["shake_label"].astype(int) == lab, f"{ax_key}_rule"].to_numpy(float)
            jitter = rng.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(
                np.full(len(vals), j) + jitter,
                vals,
                s=28,
                color=color,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
                label=name if i == 0 else None,
            )
            ax.hlines(float(np.mean(vals)), j - 0.22, j + 0.22, color=INK, lw=1.7, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["shake−", "shake+"])
        ax.set_title(panel_title[ax_key], fontsize=9)
        if i == 0:
            ax.set_ylabel("Rule amplitude (°)")
        despine(ax)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labs,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        fontsize=9,
        ncol=2,
    )
    fig.suptitle("GOLD DEV per-clip rule amplitude  ·  not a TEST metric", fontsize=11)
    save_fig(fig, "euler_axis_strips_dev")


# ---------------------------------------------------------------------------
# 5. Visual representations (not two sensory modalities)
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc, fs=8.5, tc="white") -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.7,
            edgecolor=INK,
            facecolor=fc,
            alpha=0.95,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=tc,
        linespacing=1.28,
    )


def _arrow(ax, x1, y1, x2, y2) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            color=GREY,
            shrinkA=1,
            shrinkB=1,
        )
    )


def fig_visual_representations() -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.35), layout="constrained")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Two visual representations of the same listener window", fontsize=11, pad=6)
    _box(ax, 0.18, 0.78, 0.64, 0.16, "Columbia RealTalk  ·  ~60 s listener window  ·  25 fps", GREY, fs=8.5)
    _box(ax, 0.04, 0.48, 0.40, 0.18, "EMOCA Euler  x, y, z  (°)\nvisual representation of head pose", BLUE, fs=8)
    _box(ax, 0.56, 0.48, 0.40, 0.18, "RGB face crop  16×224×224\nvisual representation of the same video", ORANGE, fs=8)
    _box(ax, 0.04, 0.18, 0.40, 0.18, "Amplitude rule  /  1D CNN\nwindow label 0/1", BLUE, fs=8)
    _box(ax, 0.56, 0.18, 0.40, 0.18, "VideoMAE-base  (frozen or last 4 blocks)\nwindow label 0/1", ORANGE, fs=8)
    _arrow(ax, 0.50, 0.78, 0.24, 0.66)
    _arrow(ax, 0.50, 0.78, 0.76, 0.66)
    _arrow(ax, 0.24, 0.48, 0.24, 0.36)
    _arrow(ax, 0.76, 0.48, 0.76, 0.36)
    ax.text(
        0.50,
        0.05,
        "Not two sensory modalities.  Not forecasting: clip-level 0/1 on the labelled window.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    save_fig(fig, "visual_representations")


# ---------------------------------------------------------------------------
# 6. Nod qualitative TEST cases (layout fix; predictions from locked CSVs)
# ---------------------------------------------------------------------------
def _idcol(df: pd.DataFrame) -> str:
    for c in ("sample_id", "clip_id"):
        if c in df.columns:
            return c
    raise KeyError(list(df.columns))


def _load_preds(path: Path, name: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cid = _idcol(df)
    df = df.rename(columns={cid: "sample_id"})
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.upper() == "TEST"]
    out = df[["sample_id", "label", "pred"]].copy()
    out["sample_id"] = out["sample_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["pred"] = out["pred"].astype(int)
    return out.rename(columns={"pred": name, "label": "gold"})


def fig_nod_qualitative() -> None:
    cases = [
        ("gold_016", "TP", "Gold nod; pose systems hit"),
        ("gold_017", "FP", "Gold unclear; rule and CNN fire"),
        ("gold_018", "FN", "Gold nod; marked time outside window"),
        ("gold_024", "TN", "Gold unclear; pose systems reject"),
    ]
    parts = [
        _load_preds(RES / "rule_test_predictions.csv", "rule"),
        _load_preds(RES / "classifier_test_predictions.csv", "cnn"),
        _load_preds(RES / "videomae_frozen_head" / "predictions.csv", "frozen"),
        _load_preds(RES / "videomae_finetuned" / "predictions_test.csv", "ft80"),
    ]
    if any(p is None for p in parts):
        skip("nod_qualitative_cases", "missing a locked TEST predictions CSV")
        return
    gold = parts[0][["sample_id", "gold"]].copy()
    out = gold
    for p, col in zip(parts, ["rule", "cnn", "frozen", "ft80"]):
        out = out.merge(p[["sample_id", col]], on="sample_id", how="left")
    by_id = {str(r.sample_id): r for r in out.itertuples()}

    fig, axes = plt.subplots(2, 2, figsize=(5.8, 5.8), sharey=True, layout="constrained")
    for ax, (sid, tag, note) in zip(axes.ravel(), cases):
        x = None
        rot = load_rot(sid)
        if rot is not None:
            x = rot[:, 0]
        r = by_id.get(sid)
        gold_lab = "nod" if r is not None and int(r.gold) == 1 else "unclear"
        if x is None:
            ax.text(0.5, 0.5, f"{sid}\npose npz missing", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color=MUTED)
        else:
            t = np.arange(len(x)) / FPS
            ax.plot(t, x, color=BLUE, lw=1.15)
            ax.fill_between(t, x, 0, color=BLUE, alpha=0.10)
        ax.set_title(f"{sid}   {tag}   gold={gold_lab}\n{note}", fontsize=9, pad=6)
        if ax in (axes[1, 0], axes[1, 1]):
            ax.set_xlabel("Time in watch window (s)")
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel("Rotation x  (°)")
        despine(ax)
        if r is not None:
            ax.text(
                0.02,
                0.97,
                f"rule={int(r.rule)}  CNN={int(r.cnn)}  frozen={int(r.frozen)}  FT80={int(r.ft80)}",
                transform=ax.transAxes,
                fontsize=8,
                color=MUTED,
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.2},
            )
    fig.suptitle("Nod TEST cases  ·  pose axis x  ·  locked CSV predictions", fontsize=11)
    save_fig(fig, "nod_qualitative_cases")


def fig_nod_vs_unclear() -> None:
    specs = [
        ("gold_016", "Gold nod  (TEST)", GREEN),
        ("gold_024", "Gold unclear  (TEST)", GREY),
    ]
    tau_path = RES / "rule_selected_config.json"
    if not tau_path.exists():
        skip("nod_vs_unclear_pitch", f"missing {tau_path}")
        return
    tau = float(_load_json(tau_path)["selected_amplitude_threshold"])
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.2), sharey=True, layout="constrained")
    for i, (ax, (sid, title, color)) in enumerate(zip(axes, specs)):
        rot = load_rot(sid)
        if rot is None:
            ax.set_title(f"{title}\n{sid}", fontsize=10)
            ax.text(0.5, 0.5, "pose npz missing", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color=MUTED)
            despine(ax)
            continue
        x = rot[:, 0]
        t = np.arange(len(x)) / FPS
        pk = float(np.nanmax(x) - np.nanmin(x))
        ax.set_title(f"{title}\n{sid}  ·  ptp ≈ {pk:.1f}°", fontsize=10)
        ax.plot(t, x, color=color, lw=1.2)
        ax.axhline(tau, color=VERM, ls="--", lw=1.0, label=f"rule τ = {tau:.2f}°")
        ax.set_xlabel("Time (s)")
        if i == 0:
            ax.set_ylabel("Rotation x  (°)")
        if i == 1:
            ax.legend(fontsize=8, loc="lower right", frameon=False)
        despine(ax)
    fig.suptitle("What a nod looks like here: EMOCA axis x  (illustration)", fontsize=11)
    save_fig(fig, "nod_vs_unclear_pitch")


# ---------------------------------------------------------------------------
# 7. Nod VideoMAE training curves (DEV F1, not TEST)
# ---------------------------------------------------------------------------
def fig_training_curves() -> None:
    runs = [
        (
            "videomae_frozen_head",
            "Frozen VideoMAE head",
            RES / "videomae_frozen_head" / "training_history.csv",
            RES / "videomae_frozen_head" / "metrics.json",
        ),
        (
            "videomae_finetuned",
            "Fine-tuned VideoMAE (last 4 blocks)",
            RES / "videomae_finetuned" / "training_history.csv",
            RES / "videomae_finetuned" / "metrics.json",
        ),
    ]
    for slug, title, hist, meta_p in runs:
        if not hist.exists() or not meta_p.exists():
            skip(f"nod_{slug}_training", "missing history or metrics json")
            continue
        df = pd.read_csv(hist)
        meta = _load_json(meta_p)
        best_epoch = int(meta["best_epoch"])
        best_dev = float(meta["dev_f1"])
        fig, ax1 = plt.subplots(figsize=(5.4, 3.15), layout="constrained")
        ax2 = ax1.twinx()
        ax2.spines["top"].set_visible(False)
        ln1 = ax1.plot(
            df["epoch"], df["train_loss"], color=BLUE, marker="o", markersize=4,
            lw=1.5, label="Training loss",
        )
        ln2 = ax2.plot(
            df["epoch"], df["dev_f1"], color=ORANGE, marker="s", markersize=4,
            lw=1.5, label="DEV F1",
        )
        ax1.axvline(best_epoch, color=GREY, ls="--", lw=1.0)
        ln3 = ax2.plot(
            [best_epoch], [best_dev], marker="*", color=ORANGE, markersize=14,
            linestyle="none", label=f"Early stop (epoch {best_epoch})",
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training loss", color=BLUE)
        ax2.set_ylabel("DEV F1", color=ORANGE)
        ax1.tick_params(axis="y", labelcolor=BLUE)
        ax2.tick_params(axis="y", labelcolor=ORANGE)
        ax2.set_ylim(0.5, 1.02)
        ax1.set_xticks(list(df["epoch"]))
        lines = ln1 + ln2 + ln3
        ax1.legend(
            lines,
            [ln.get_label() for ln in lines],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
        ax1.set_title(f"{title}\nDEV F1 by epoch  ·  not a TEST result", fontsize=10)
        despine(ax1)
        save_fig(fig, f"nod_{slug}_training")


def main() -> None:
    _style()
    PAPER.mkdir(parents=True, exist_ok=True)
    fig_nod_test()
    fig_nod_confusion()
    fig_shake_test()
    fig_shake_confusion()
    fig_shake_dev()
    fig_euler_spectrum()
    fig_euler_traces()
    fig_axis_audit_dev()
    fig_visual_representations()
    fig_nod_qualitative()
    fig_nod_vs_unclear()
    fig_training_curves()
    print("done ->", PAPER)


if __name__ == "__main__":
    main()
