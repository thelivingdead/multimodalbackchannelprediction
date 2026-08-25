#!/usr/bin/env python3
"""Paper-style dissertation figures from locked TEST artefacts only.

Inspiration (MM-F2F, head-gesture papers): qualitative cases, architecture,
and a metrics table. Nothing here invents BERT/HuBERT/7-class/LMF results.

Reads:
  results/tables/main_results.csv
  results/tables/bootstrap_ci.csv
  results/gold_dataset_summary.csv
  results/rule_test_predictions.csv
  results/classifier_test_predictions.csv
  results/videomae_frozen_head/predictions.csv
  results/videomae_finetuned/predictions_test.csv
  results/videomae_finetuned_n200/predictions_test.csv  (optional)
  features/gold/gold_*.npz  (pose traces; skip if missing)

Writes (300 dpi PNG + JPG) under figures/paper/:
  architecture.png          two-stream system (pose + RGB) as executed
  test_metrics.png          TEST P/R/F1 + 95% CI (n=15)
  test_clip_grid.png        15 TEST clips: gold vs four models
  error_cases.png           pose-x traces for TP / FP / FN / TN
  nod_vs_unclear.png        gold nod vs gold unclear pitch (axis x)

Run:
  python scripts/plot_paper_style_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FEAT = ROOT / "features" / "gold"
OUT = ROOT / "figures" / "paper"
FPS = 25.0

BLUE = "#4e79a7"
ORANGE = "#f28e2b"
GREEN = "#59a14f"
RED = "#e15759"
GREY = "#6b7280"
PURPLE = "#b07aa1"

# Qualitative cases from locked TEST (see results/error_analysis.csv).
CASES = [
    ("gold_016", "TP", "Clear nod, both pose systems hit"),
    ("gold_017", "FP", "Unclear; rule and CNN both false-alarm"),
    ("gold_018", "FN", "Gold nod; marked time outside window"),
    ("gold_024", "TN", "Unclear, correctly rejected"),
]


def _idcol(df: pd.DataFrame) -> str:
    for c in ("sample_id", "clip_id"):
        if c in df.columns:
            return c
    raise KeyError(df.columns)


def load_preds(path: Path, name: str) -> pd.DataFrame:
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


def outcome(gold: int, pred: int) -> str:
    if gold == 1 and pred == 1:
        return "TP"
    if gold == 0 and pred == 1:
        return "FP"
    if gold == 1 and pred == 0:
        return "FN"
    return "TN"


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    jpg = OUT / f"{stem}.jpg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(jpg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", png.relative_to(ROOT))


def box(ax, x, y, w, h, text, fc, fs=8.5, tc="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0.8, edgecolor="#111827", facecolor=fc, alpha=0.95,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
            linewidth=1.4, color=color, shrinkA=1, shrinkB=1,
        )
    )


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "Executed system (binary head-nod, not 7-class fusion)",
            fontsize=12, fontweight="bold", color="#111827")

    box(ax, 0.02, 0.72, 0.18, 0.16, "Columbia RealTalk\nlistener window\n~60 s, 25 fps", GREY)
    box(ax, 0.26, 0.78, 0.20, 0.14, "EMOCA pose\nrotation x,y,z (deg)", BLUE)
    box(ax, 0.26, 0.58, 0.20, 0.14, "RGB face crop\n16×224×224", ORANGE)

    box(ax, 0.52, 0.80, 0.20, 0.12, "Frozen rule\naxis x, τ=16.35°", BLUE, fs=8)
    box(ax, 0.52, 0.64, 0.20, 0.12, "1D CNN on pose\n80 pseudo-labels", BLUE, fs=8)
    box(ax, 0.52, 0.48, 0.20, 0.12, "VideoMAE-base\nfrozen or last 4 blocks", ORANGE, fs=8)

    box(ax, 0.78, 0.70, 0.18, 0.14, "Gold DEV n=15\nepoch + threshold", GREEN)
    box(ax, 0.78, 0.48, 0.18, 0.14, "Gold TEST n=15\nscored once\nclip-level 0/1", RED)

    arrow(ax, 0.20, 0.84, 0.26, 0.85)
    arrow(ax, 0.20, 0.76, 0.26, 0.65)
    arrow(ax, 0.46, 0.85, 0.52, 0.86)
    arrow(ax, 0.46, 0.85, 0.52, 0.70)
    arrow(ax, 0.46, 0.65, 0.52, 0.54)
    arrow(ax, 0.72, 0.86, 0.78, 0.77)
    arrow(ax, 0.72, 0.70, 0.78, 0.77)
    arrow(ax, 0.72, 0.54, 0.78, 0.55)
    arrow(ax, 0.87, 0.70, 0.87, 0.62, color=GREEN)

    ax.text(0.02, 0.08,
            "No BERT, HuBERT, or LMF in this study. TRAIN = frozen-rule pseudo-labels. "
            "TEST never used for tuning. Optical-flow / 7-class diagrams from other papers do not apply.",
            fontsize=8, color=GREY, wrap=True)
    save(fig, "architecture")


def fig_metrics() -> None:
    tab = pd.read_csv(RES / "tables" / "main_results.csv")
    # Drop the raw-xyz CNN row from the graphic (no CI); keep reported systems.
    keep = [
        "Rule baseline",
        "Pose CNN xyz_deriv",
        "Frozen VideoMAE head",
        "Fine-tuned VideoMAE (last 4 blocks)",
        "Fine-tuned VideoMAE (last 4 blocks, 200 pseudo)",
    ]
    short = {
        "Rule baseline": "Pose rule",
        "Pose CNN xyz_deriv": "Pose CNN",
        "Frozen VideoMAE head": "Frozen\nVideoMAE",
        "Fine-tuned VideoMAE (last 4 blocks)": "Fine-tuned\nVideoMAE n=80",
        "Fine-tuned VideoMAE (last 4 blocks, 200 pseudo)": "Fine-tuned\nVideoMAE n=200",
    }
    df = tab[tab["model"].isin(keep)].copy()
    df["model"] = pd.Categorical(df["model"], keep, ordered=True)
    df = df.sort_values("model")

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    x = np.arange(len(df))
    f1 = df["f1"].to_numpy(float)
    lo = pd.to_numeric(df["f1_ci_lo"], errors="coerce").to_numpy()
    hi = pd.to_numeric(df["f1_ci_hi"], errors="coerce").to_numpy()
    yerr = np.vstack([f1 - lo, hi - f1])
    colors = [BLUE, GREEN, RED, ORANGE, PURPLE]
    ax.bar(x, f1, color=colors, width=0.72, zorder=2)
    ax.errorbar(x, f1, yerr=yerr, fmt="none", ecolor="#111827",
                capsize=4, linewidth=1.2, zorder=3)
    for i, row in enumerate(df.itertuples()):
        ax.text(i, min(0.98, float(row.f1) + 0.04),
                f"P {float(row.precision):.2f}  R {float(row.recall):.2f}",
                ha="center", fontsize=8, color="#111827")
    ax.set_xticks(x)
    ax.set_xticklabels([short[m] for m in df["model"]], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("TEST F1  (n = 15, scored once)")
    ax.set_title("Held-out TEST: precision / recall (above bars) and F1 with 95% bootstrap CI")
    ax.axhline(0.67, color=GREY, ls=":", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.0, -0.22,
            "Gold = human 1/0 on the same 15 TEST windows. CIs overlap; 0.82 is the highest point estimate, not a significant win.",
            transform=ax.transAxes, fontsize=8, color=GREY)
    save(fig, "test_metrics")


def assemble_test_table() -> pd.DataFrame:
    gold = pd.read_csv(RES / "gold_dataset_summary.csv")
    gold = gold[gold["split"].astype(str).str.upper() == "TEST"][
        ["sample_id", "video_id", "person", "label"]
    ].copy()
    gold["sample_id"] = gold["sample_id"].astype(str)
    gold["gold"] = gold["label"].astype(int)

    rule = load_preds(RES / "rule_test_predictions.csv", "rule")
    cnn = load_preds(RES / "classifier_test_predictions.csv", "cnn")
    frz = load_preds(RES / "videomae_frozen_head" / "predictions.csv", "frozen")
    ft = load_preds(RES / "videomae_finetuned" / "predictions_test.csv", "ft80")
    n200p = RES / "videomae_finetuned_n200" / "predictions_test.csv"
    parts = [gold, rule[["sample_id", "rule"]], cnn[["sample_id", "cnn"]],
             frz[["sample_id", "frozen"]], ft[["sample_id", "ft80"]]]
    if n200p.exists():
        n200 = load_preds(n200p, "ft200")
        parts.append(n200[["sample_id", "ft200"]])
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="sample_id", how="left")
    return out.sort_values("sample_id")


def fig_clip_grid(df: pd.DataFrame) -> None:
    models = [("gold", "Gold"), ("rule", "Rule"), ("cnn", "CNN"),
              ("frozen", "VMAE\nfrozen"), ("ft80", "VMAE\nn=80")]
    if "ft200" in df.columns:
        models.append(("ft200", "VMAE\nn=200"))
    mat = df[[m[0] for m in models]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    cmap = matplotlib.colors.ListedColormap(["#f3f4f6", "#4e79a7"])
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m[1] for m in models], fontsize=8)
    labels = []
    for r in df.itertuples():
        side = "L" if str(r.person) in ("p0", "LEFT") else "R"
        labels.append(f"{r.sample_id}  {r.video_id}  {side}")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "nod" if mat[i, j] == 1 else "—",
                    ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] == 1 else "#6b7280")
    ax.set_title("TEST windows: human gold vs model predictions (1 = nod)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.0, -0.06,
            "Gold is the human label. Other columns are locked TEST predictions (scored once). n=200 is an ablation, not the RGB headline.",
            transform=ax.transAxes, fontsize=8, color=GREY)
    save(fig, "test_clip_grid")


def load_rot_x(sample_id: str) -> np.ndarray | None:
    path = FEAT / f"{sample_id}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    rot = np.asarray(z["rotation_xyz"], dtype=float)
    return rot[:, 0]


def fig_error_cases(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), sharey=True)
    axes = axes.ravel()
    by_id = {str(r.sample_id): r for r in df.itertuples()}
    missing = 0
    for ax, (sid, tag, note) in zip(axes, CASES):
        x = load_rot_x(sid)
        r = by_id[sid]
        gold = "nod" if int(r.gold) == 1 else "unclear"
        if x is None:
            ax.text(0.5, 0.5, f"{sid}\npose npz not on this machine",
                    ha="center", va="center", transform=ax.transAxes)
            missing += 1
        else:
            t = np.arange(len(x)) / FPS
            ax.plot(t, x, color=BLUE, lw=1.1)
            ax.fill_between(t, x, 0, color=BLUE, alpha=0.12)
        ax.set_title(f"{sid}  gold={gold}  [{tag}]\n{note}", fontsize=9)
        ax.set_xlabel("time in watch window (s)", fontsize=8)
        ax.set_ylabel("rotation x (deg)", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        pred_line = (
            f"rule={int(r.rule)}  CNN={int(r.cnn)}  "
            f"frozen={int(r.frozen)}  FT80={int(r.ft80)}"
        )
        ax.text(0.02, 0.04, pred_line, transform=ax.transAxes, fontsize=7, color=GREY)
    fig.suptitle("Qualitative TEST cases (pose axis x). Predictions are 0/1 from locked CSVs.",
                 fontsize=11)
    save(fig, "error_cases")


def fig_nod_vs_unclear() -> None:
    """Gesture-style panel using real TEST pose, not a synthetic 7-class avatar."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)
    specs = [
        ("gold_016", "Gold nod (TEST)", GREEN),
        ("gold_024", "Gold unclear (TEST)", GREY),
    ]
    for ax, (sid, title, color) in zip(axes, specs):
        x = load_rot_x(sid)
        ax.set_title(title + f"\n{sid}", fontsize=10)
        if x is None:
            ax.text(0.5, 0.5, "pose npz missing", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        t = np.arange(len(x)) / FPS
        ax.plot(t, x, color=color, lw=1.2)
        pk = float(np.nanmax(x) - np.nanmin(x))
        ax.axhline(16.35, color=RED, ls="--", lw=1, label="rule τ = 16.35°")
        ax.text(0.98, 0.92, f"peak-to-peak ≈ {pk:.1f}°", transform=ax.transAxes,
                ha="right", fontsize=8, color="#111827")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("rotation x (deg)")
        ax.legend(fontsize=7, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("What a nod looks like in this study: EMOCA axis x, not optical flow or 7 classes",
                 fontsize=11)
    save(fig, "nod_vs_unclear")


def main() -> None:
    fig_architecture()
    fig_metrics()
    df = assemble_test_table()
    assert len(df) == 15, f"expected 15 TEST rows, got {len(df)}"
    fig_clip_grid(df)
    fig_error_cases(df)
    fig_nod_vs_unclear()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
