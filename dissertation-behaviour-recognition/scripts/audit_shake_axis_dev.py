#!/usr/bin/env python3
"""GOLD DEV-only Euler axis audit for head-shake (does not score TEST).

Plots Euler x, y, z vs time for ≥5 gold-DEV shake+ and all gold-DEV shake−
clips. Infers left-right (yaw-like) vs up-down (nod-like) from traces, not
from the locked TEST F1. Does **not** sort plotted positives by frozen-z
score (that would circularly favour z).

Writes only::

    figures/shake/axis_audit/
    results/shake/dev_balanced/axis_audit.md
    results/shake/dev_balanced/axis_audit_conclusion.json

Does not retune the locked shake rule, does not write TEST metrics, does
not touch nod artefacts. EMOCA/FLAME pose is used, not trained.
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

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from shake_v2_common import load_npz, n_direction_changes, rule_score  # noqa: E402
from src.plotting import FigureLog, save_publication_figure  # noqa: E402

SHEET = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
GOLD_NPZ = ROOT / "features" / "gold"
RGB16 = ROOT / "features" / "rgb16"
FIG_DIR = ROOT / "figures" / "shake" / "axis_audit"
OUT_DIR = ROOT / "results" / "shake" / "dev_balanced"
OUT_DIRS = (OUT_DIR,)
LOCKED_RULE = ROOT / "results" / "shake" / "rule_selected_config.json"
AXIS_NAMES = ("x", "y", "z")
LIT = ("pitch (nod-like)", "yaw (shake-like)", "roll (tilt-like)")
FPS = 25.0


def _load_dev() -> pd.DataFrame:
    df = pd.read_csv(SHEET)
    df["split"] = df["split"].astype(str).str.upper()
    df["shake_label"] = pd.to_numeric(df["shake_label"], errors="coerce")
    df["nod_label"] = pd.to_numeric(df["nod_label"], errors="coerce")
    dev = df[df.split == "DEV"].copy()
    if dev["shake_label"].isna().any():
        raise SystemExit("STOP: unfilled shake_label on DEV")
    n_pos = int((dev.shake_label == 1).sum())
    n_neg = int((dev.shake_label == 0).sum())
    if n_pos < 5 or n_neg < 5:
        raise SystemExit(f"STOP: DEV shake counts {n_pos}/{n_neg}; need ≥5 each")
    return dev


def band_energy(x: np.ndarray, lo: float = 1.0, hi: float = 5.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return 0.0
    x = x - np.nanmean(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), d=1.0 / FPS)
    mask = (freq >= lo) & (freq <= hi)
    return float(spec[mask].sum()) if mask.any() else 0.0


def _scores_for(sid: str) -> tuple[dict, np.ndarray]:
    z = load_npz(GOLD_NPZ / f"{sid}.npz")
    rot = np.asarray(z["rotation_xyz"], dtype=float)
    vid = z["video_id"]
    if hasattr(vid, "item"):
        vid = vid.item()
    row = {
        "sample_id": sid,
        "n_frames": int(len(rot)),
        "video_id": str(vid),
    }
    for ax, name in enumerate(AXIS_NAMES):
        col = rot[:, ax]
        finite = col[np.isfinite(col)]
        row[f"ptp_{name}"] = float(np.ptp(finite)) if finite.size else 0.0
        row[f"std_{name}"] = float(np.nanstd(col))
        row[f"score_{name}"] = float(rule_score(rot, ax))
        row[f"band_1_5hz_{name}"] = band_energy(col)
        row[f"turns_{name}"] = n_direction_changes(col)
    return row, rot


def _pick_clips(dev: pd.DataFrame) -> tuple[list[str], list[str]]:
    """≥5 shake+ (prefer shake-only) and all shake−. Not ranked by frozen z."""
    pos = dev[dev.shake_label == 1].copy()
    neg = dev[dev.shake_label == 0].copy()
    shake_only = pos[pos.nod_label == 0]["sample_id"].astype(str).tolist()
    mixed = pos[pos.nod_label == 1]["sample_id"].astype(str).tolist()
    pos_ids = (shake_only + mixed)[:5]
    if len(pos_ids) < 5:
        pos_ids = pos["sample_id"].astype(str).tolist()[:5]
    neg_ids = neg["sample_id"].astype(str).tolist()
    return pos_ids, neg_ids


def _plot_clip(sid: str, rot: np.ndarray, label: int, nod: int, log: FigureLog) -> None:
    t = np.arange(len(rot)) / FPS
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.4), sharex=True)
    colors = ("#0072B2", "#E69F00", "#009E73")
    for i, (name, lit, col) in enumerate(zip(AXIS_NAMES, LIT, colors)):
        axes[i].plot(t, rot[:, i], color=col, lw=0.9)
        axes[i].set_ylabel(f"{name} (°)\n{lit}")
        axes[i].axhline(0.0, color="0.6", lw=0.4)
    axes[-1].set_xlabel("time (s)  [25 fps; 1500 frames ≈ 60 s gold window]")
    fig.suptitle(f"{sid}  gold DEV  shake={label}  nod={nod}")
    save_publication_figure(
        fig, FIG_DIR / f"{sid}_xyz", log, source="audit_shake_axis_dev.py", force=True
    )


def _plot_summary(scored: pd.DataFrame, log: FigureLog) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True)
    for i, name in enumerate(AXIS_NAMES):
        ax = axes[i]
        for lab, color in ((0, "#D55E00"), (1, "#0072B2")):
            vals = scored.loc[scored.shake_label == lab, f"score_{name}"]
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter(
                np.full(len(vals), lab) + jitter,
                vals,
                c=color,
                s=36,
                zorder=3,
                label=f"shake={lab}",
            )
            ax.hlines(float(vals.mean()), lab - 0.28, lab + 0.28, colors=color, lw=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["shake−", "shake+"])
        ax.set_title(f"rule_score {name}\n{LIT[i]}")
        ax.set_ylabel("amplitude (°)" if i == 0 else "")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("GOLD DEV only — oscillatory amplitude by Euler axis")
    save_publication_figure(
        fig,
        FIG_DIR / "dev_rule_score_by_axis",
        log,
        source="audit_shake_axis_dev.py",
        force=True,
    )


def _plot_exclusive(scored: pd.DataFrame, log: FigureLog) -> None:
    """Shake-only vs nod-only mean rule_score — the cleanest geometric contrast."""
    so = scored[(scored.shake_label == 1) & (scored.nod_label == 0)]
    no = scored[(scored.shake_label == 0) & (scored.nod_label == 1)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(3)
    w = 0.35
    so_m = [float(so[f"score_{a}"].mean()) if len(so) else 0.0 for a in AXIS_NAMES]
    no_m = [float(no[f"score_{a}"].mean()) if len(no) else 0.0 for a in AXIS_NAMES]
    ax.bar(x - w / 2, so_m, w, label=f"shake-only (n={len(so)})", color="#E69F00")
    ax.bar(x + w / 2, no_m, w, label=f"nod-only (n={len(no)})", color="#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{lit}" for a, lit in zip(AXIS_NAMES, LIT)])
    ax.set_ylabel("mean rule_score (°)")
    ax.set_title("GOLD DEV exclusive labels — which axis carries shake vs nod")
    ax.legend(frameon=False)
    save_publication_figure(
        fig,
        FIG_DIR / "dev_nod_only_vs_shake_only",
        log,
        source="audit_shake_axis_dev.py",
        force=True,
    )


def _rgb_note(sample_ids: list[str]) -> str:
    have = [s for s in sample_ids if (RGB16 / f"{s}.npz").exists()]
    if have:
        return (
            f"RGB16 npz present for {len(have)}/{len(sample_ids)} plotted ids "
            f"({', '.join(have)}). Face-crop stills were not opened as video, "
            "so left-right vs up-down was **not** visually confirmed here. "
            "Replay gold YouTube windows on Mac/otter to sanity-check yaw."
        )
    return (
        "No `features/rgb16/*.npz` on this Mac for the plotted gold ids. "
        "YouTube URLs are in `data/gold/shake_annotation_sheet.csv` but were "
        "not fetched. Anatomical left-right vs nod was **not** watched. "
        "On otter, rgb16 crops exist for the locked VideoMAE runs; they were "
        "not re-scored here. Video compare needs otter or Mac playback."
    )


def _axis_stats(scored: pd.DataFrame) -> dict:
    out = {}
    for a in AXIS_NAMES:
        shake_pos = scored.loc[scored.shake_label == 1, f"score_{a}"]
        shake_neg = scored.loc[scored.shake_label == 0, f"score_{a}"]
        nod_pos = scored.loc[scored.nod_label == 1, f"score_{a}"]
        nod_neg = scored.loc[scored.nod_label == 0, f"score_{a}"]
        so = scored[(scored.shake_label == 1) & (scored.nod_label == 0)]
        no = scored[(scored.shake_label == 0) & (scored.nod_label == 1)]
        out[a] = {
            "literature": LIT[AXIS_NAMES.index(a)],
            "mean_rule_shake_pos": float(shake_pos.mean()),
            "mean_rule_shake_neg": float(shake_neg.mean()),
            "mean_rule_nod_pos": float(nod_pos.mean()) if len(nod_pos) else float("nan"),
            "mean_rule_nod_neg": float(nod_neg.mean()) if len(nod_neg) else float("nan"),
            "mean_rule_shake_only": float(so[f"score_{a}"].mean()) if len(so) else float("nan"),
            "mean_rule_nod_only": float(no[f"score_{a}"].mean()) if len(no) else float("nan"),
            "mean_ptp_shake_pos": float(scored.loc[scored.shake_label == 1, f"ptp_{a}"].mean()),
            "mean_ptp_shake_neg": float(scored.loc[scored.shake_label == 0, f"ptp_{a}"].mean()),
            "mean_band_1_5hz_shake_pos": float(
                scored.loc[scored.shake_label == 1, f"band_1_5hz_{a}"].mean()
            ),
            "mean_band_1_5hz_shake_neg": float(
                scored.loc[scored.shake_label == 0, f"band_1_5hz_{a}"].mean()
            ),
            "rule_sep_shake": float(shake_pos.mean() - shake_neg.mean()),
            "band_sep_shake": float(
                scored.loc[scored.shake_label == 1, f"band_1_5hz_{a}"].mean()
                - scored.loc[scored.shake_label == 0, f"band_1_5hz_{a}"].mean()
            ),
        }
    return out


def _write_outputs(
    dev: pd.DataFrame,
    scored: pd.DataFrame,
    pos_ids: list[str],
    neg_ids: list[str],
    rgb_note: str,
    stats: dict,
) -> dict:
    n_pos = int((dev.shake_label == 1).sum())
    n_neg = int((dev.shake_label == 0).sum())
    n_so = int(((dev.shake_label == 1) & (dev.nod_label == 0)).sum())
    n_no = int(((dev.shake_label == 0) & (dev.nod_label == 1)).sum())
    rule_cfg = json.loads(LOCKED_RULE.read_text())
    frozen_ax = str(rule_cfg.get("axis_name", "z"))
    frozen_thr = float(rule_cfg["selected_amplitude_threshold"])

    sep = {a: stats[a]["rule_sep_shake"] for a in AXIS_NAMES}
    so_amp = {a: stats[a]["mean_rule_shake_only"] for a in AXIS_NAMES}
    no_amp = {a: stats[a]["mean_rule_nod_only"] for a in AXIS_NAMES}
    band_sep = {a: stats[a]["band_sep_shake"] for a in AXIS_NAMES}
    geo_shake = max(so_amp, key=lambda a: (so_amp[a], sep[a], band_sep[a]))
    geo_nod = max(no_amp, key=lambda a: no_amp[a])
    z_ok = geo_shake == "z"

    conclusion = {
        "split": "DEV",
        "n_dev": int(len(dev)),
        "n_shake_pos": n_pos,
        "n_shake_neg": n_neg,
        "n_nod_pos": int((dev.nod_label == 1).sum()),
        "n_nod_neg": int((dev.nod_label == 0).sum()),
        "n_shake_only": n_so,
        "n_nod_only": n_no,
        "plotted_shake_pos": pos_ids,
        "plotted_shake_neg": neg_ids,
        "rotation_key": "rotation_xyz",
        "video_comparison": rgb_note,
        "literature_mapping": {
            "x": "pitch (up-down / nod)",
            "y": "yaw (left-right / shake)",
            "z": "roll (tilt)",
        },
        "locked_rule_axis": frozen_ax,
        "locked_rule_threshold_deg": frozen_thr,
        "geometric_shake_axis": geo_shake,
        "geometric_nod_axis": geo_nod,
        "geometric_shake_axis_from_rule_sep": max(sep, key=sep.get),
        "geometric_shake_axis_from_1_5hz_band": max(band_sep, key=band_sep.get),
        "current_z_rule_geometrically_supported": bool(z_ok),
        "pseudo_label_axis": geo_shake,
        "report_also_axis": "y",
        "axis_summary": stats,
        "eyeball_clips": {
            str(r.sample_id): {
                "video_id": str(r.video_id),
                "youtube_url": str(r.youtube_url),
                "who_to_watch": str(r.who_to_watch),
                "watch_from": str(r.watch_from),
                "watch_until": str(r.watch_until),
                "shake_label": int(r.shake_label),
                "nod_label": int(r.nod_label) if pd.notna(r.nod_label) else None,
            }
            for r in dev.itertuples()
            if str(r.sample_id) in set(pos_ids + neg_ids)
        },
        "note": (
            "Axis names are rotation_xyz columns after "
            "Rotation.from_rotvec(pose[:3]).as_euler('xyz', degrees=True). "
            "Locked TEST still used z and is not rewritten. New TRAIN 0/1 "
            f"ranks on geometric **{geo_shake}**, not the frozen z τ cut. "
            "Videos were not on this Mac; YouTube windows are in eyeball_clips."
        ),
    }

    lines = [
        "# Shake axis audit (GOLD DEV only)",
        "",
        "Selection uses **GOLD DEV only**. This file is not a GOLD TEST F1.",
        "",
        "## DEV class counts",
        "",
        f"- gold DEV clips: **{len(dev)}**",
        f"- shake+ (`shake_label=1`): **{n_pos}**",
        f"- shake− (`shake_label=0`): **{n_neg}**",
        f"- nod+ on the same clips: **{int((dev.nod_label == 1).sum())}**",
        f"- shake-only (shake=1, nod=0): **{n_so}**",
        f"- nod-only (shake=0, nod=1): **{n_no}**",
        "",
        "Gold TEST (15 clips) was **not** used to choose the axis.",
        "",
        "## Convention (do not assume names a priori)",
        "",
        "EMOCA/FLAME pose is **used, not trained**. Stored `rotation_xyz` is",
        '`Rotation.from_rotvec(pose[:3]).as_euler("xyz", degrees=True)` →',
        "**x, y, z**. Literature maps these to pitch / yaw / roll, but this",
        "audit decides from DEV traces.",
        "",
        f"Locked **nod** rule used **x** (τ = 16.35°). Locked **shake** rule",
        f"used **{frozen_ax}** (τ = {frozen_thr:.3f}°) by DEV F1, with a note",
        "that yaw was *hypothesised* to be y.",
        "",
        "## Clips plotted (not ranked by frozen z)",
        "",
        "Shake+ (shake-only first, then mixed): " + ", ".join(pos_ids) + ".",
        "",
        "Shake− (all five DEV negatives): " + ", ".join(neg_ids) + ".",
        "",
        rgb_note,
        "",
        "## Mean oscillatory amplitude on DEV (rule_score, degrees)",
        "",
        "| axis | literature | mean shake+ | mean shake− | + minus − | shake-only | nod-only |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for a, lit in zip(AXIS_NAMES, LIT):
        lines.append(
            f"| {a} | {lit} | {stats[a]['mean_rule_shake_pos']:.2f} | "
            f"{stats[a]['mean_rule_shake_neg']:.2f} | {sep[a]:.2f} | "
            f"{stats[a]['mean_rule_shake_only']:.2f} | "
            f"{stats[a]['mean_rule_nod_only']:.2f} |"
        )
    lines += [
        "",
        "1–5 Hz band energy (qualitative, not a detector): largest shake+ minus "
        f"shake− gap on **{max(band_sep, key=band_sep.get)}**.",
        "",
        "## Verdict",
        "",
    ]
    if not z_ok and geo_shake == "y":
        verdict = (
            f"Geometric shake axis on GOLD DEV is **y** (yaw-like). "
            f"Shake-only clips peak on y ({so_amp['y']:.1f}°) not z "
            f"({so_amp['z']:.1f}°). Nod-only clips peak on **{geo_nod}** "
            f"({no_amp[geo_nod]:.1f}°). Locked **z** is roll-like and is "
            "**not** geometrically supported as left-right. "
            "**New pseudo-labels use y.** Locked TEST artefacts stay on z "
            f"(τ={frozen_thr:.3f}°) and are not rewritten."
        )
    elif z_ok:
        verdict = (
            "DEV exclusive-label amplitude is largest on **z**, matching the "
            "frozen shake rule. New pseudo-labels keep z. Locked TEST is not "
            "rescored."
        )
    else:
        verdict = (
            f"Evidence is mixed: largest exclusive-label shake amplitude is "
            f"on **{geo_shake}**, locked rule used **{frozen_ax}**. "
            f"**New pseudo-labels use {geo_shake}.** Locked TEST (z) is "
            "not rewritten."
        )
    lines += [
        verdict,
        "",
        "### YouTube windows (videos were not on this Mac)",
        "",
    ]
    plotted = set(pos_ids + neg_ids)
    for r in dev.itertuples():
        if str(r.sample_id) not in plotted:
            continue
        nod = int(r.nod_label) if pd.notna(r.nod_label) else -1
        lines.append(
            f"- `{r.sample_id}` (`{r.video_id}`) shake={int(r.shake_label)} "
            f"nod={nod}  watch {r.who_to_watch} {r.watch_from}–{r.watch_until}  "
            f"{r.youtube_url}"
        )
    lines += [
        "",
        "Per-clip scores: `axis_audit_dev_scores.csv`. TEST clips were not scored.",
        "",
    ]
    text = json.dumps(conclusion, indent=2) + "\n"
    md = "\n".join(lines) + "\n"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "conclusion.json").write_text(text)
    (FIG_DIR / "README.md").write_text(
        "# Shake axis audit (GOLD DEV only)\n\n"
        "Traces are EMOCA Euler `rotation_xyz`. Geometric shake on these clips is "
        f"**{geo_shake}** (yaw-like). Locked TEST rule used **{frozen_ax}** and "
        "was not rewritten. New TRAIN pseudo-labels use **y**.\n"
        "Narrative: `results/shake/dev_balanced/axis_audit.md`.\n"
    )
    for dest in OUT_DIRS:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "axis_audit.md").write_text(md)
        (dest / "axis_audit_conclusion.json").write_text(text)
        (dest / "conclusion.json").write_text(text)
    print(f"wrote {OUT_DIR / 'axis_audit.md'}")
    print("verdict:", verdict)
    return conclusion


def main() -> None:
    if not SHEET.exists():
        raise SystemExit(f"STOP: missing {SHEET}")
    if not LOCKED_RULE.exists():
        raise SystemExit(f"STOP: missing locked {LOCKED_RULE} (do not invent z/τ)")
    dev = _load_dev()
    print(
        f"DEV shake+ {int((dev.shake_label == 1).sum())} / "
        f"shake− {int((dev.shake_label == 0).sum())}  "
        f"(nod+ {int((dev.nod_label == 1).sum())})"
    )
    rows = []
    rots = {}
    nods = {}
    for r in dev.itertuples():
        sid = str(r.sample_id)
        if not (GOLD_NPZ / f"{sid}.npz").exists():
            raise SystemExit(f"STOP: missing {sid}.npz")
        stats, rot = _scores_for(sid)
        stats["shake_label"] = int(r.shake_label)
        stats["nod_label"] = int(r.nod_label) if pd.notna(r.nod_label) else ""
        rows.append(stats)
        rots[sid] = rot
        nods[sid] = int(r.nod_label) if pd.notna(r.nod_label) else -1
    scored = pd.DataFrame(rows)
    pos_ids, neg_ids = _pick_clips(dev)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = FigureLog()
    for sid in pos_ids + neg_ids:
        lab = int(scored.loc[scored.sample_id == sid, "shake_label"].iloc[0])
        _plot_clip(sid, rots[sid], lab, nods[sid], log)
    _plot_summary(scored, log)
    _plot_exclusive(scored, log)
    scored.sort_values("sample_id").to_csv(
        OUT_DIR / "axis_audit_dev_scores.csv", index=False
    )
    for dest in OUT_DIRS:
        dest.mkdir(parents=True, exist_ok=True)
        scored.sort_values("sample_id").to_csv(
            dest / "axis_audit_dev_scores.csv", index=False
        )
    rgb_note = _rgb_note(pos_ids + neg_ids)
    stats = _axis_stats(scored)
    _write_outputs(dev, scored, pos_ids, neg_ids, rgb_note, stats)
    print(f"figures → {FIG_DIR}")
    print("TEST clips were not used.")


if __name__ == "__main__":
    main()
