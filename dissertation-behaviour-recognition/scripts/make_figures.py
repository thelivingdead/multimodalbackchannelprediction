#!/usr/bin/env python3
"""Regenerate dissertation figures from saved CSVs/JSON. Never invent metrics.

Usage:
  python scripts/make_figures.py --all
  python scripts/make_figures.py --all --force

Missing result files are skipped with an explicit message.
Existing figures are not overwritten unless --force.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotations import gold_nods, load_events, parse_label  # noqa: E402
from src.data import default_pilot_dir, list_clip_dirs, read_meta  # noqa: E402
from src.events import Event, match_pairs  # noqa: E402
from src.paths import ensure_dirs  # noqa: E402
from src.plotting import FigureLog, require_files, save_publication_figure  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for ln in path.read_text().splitlines() if ln.strip())


def fig_split_summary(log: FigureLog, force: bool) -> None:
    name = "figures/dataset/split_summary"
    dev_p = ROOT / "data" / "splits" / "gold_dev.txt"
    tes_p = ROOT / "data" / "splits" / "gold_test.txt"
    ptr_p = ROOT / "data" / "splits" / "pseudo_train_videos.txt"
    gold = load_events(ROOT / "data" / "gold" / "events.csv")
    n_dev = _count_lines(dev_p)
    n_test = _count_lines(tes_p)
    n_ptr = _count_lines(ptr_p)
    clips = list_clip_dirs(default_pilot_dir())
    duration = sum(float(read_meta(c).get("duration_s", 0)) for c in clips)
    n_part = n_dev + n_test  # one listener track per video until dyad metadata exists
    n_dyad = n_part
    if n_dev + n_test + n_ptr == 0 and not clips:
        log.skip(name, "missing data/splits/*.txt and data/working/pilot")
        return
    rows = pd.DataFrame(
        [
            {"quantity": "pseudo-train videos", "value": n_ptr},
            {"quantity": "gold DEV videos", "value": n_dev or len(clips)},
            {"quantity": "gold TEST videos", "value": n_test},
            {"quantity": "unique videos (pilot clips)", "value": len(clips)},
            {"quantity": "gold event rows", "value": int(len(gold))},
            {"quantity": "duration (minutes)", "value": round(duration / 60.0, 2)},
        ]
    )
    rows.to_csv(RES / "split_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.barh(rows["quantity"], rows["value"], color="C0")
    ax.set_xlabel("count (duration bar is minutes)")
    ax.set_title("Selected subset: videos, gold rows, duration")
    save_publication_figure(fig, FIG / "dataset" / "split_summary", log, str(RES / "split_summary.csv"), force)
    _ = n_part, n_dyad


def fig_realtalk_diversity(log: FigureLog, force: bool) -> None:
    name = "figures/dataset/realtalk_participant_diversity"
    clips = []
    for c in list_clip_dirs(default_pilot_dir()):
        mp4 = c / "clip.mp4"
        if mp4.exists() and mp4.stat().st_size > 5000:
            clips.append(c)
    if len(clips) < 2:
        log.skip(name, "need ≥2 real clip.mp4 files in data/working/pilot (placeholders are skipped)")
        return
    try:
        import cv2
    except ImportError:
        log.skip(name, "opencv not installed")
        return
    gold = gold_nods(load_events(ROOT / "data" / "gold" / "events.csv"))
    gold_by = {}
    for e in gold:
        gold_by.setdefault(e.video_id, []).append(e)
    n = min(8, len(clips))
    fig, axes = plt.subplots(n, 4, figsize=(10.5, 2.2 * n))
    if n == 1:
        axes = np.array([axes])
    col_titles = ["listening / other", "nod downward", "nod recovery", "other pose"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=11)
    for i, c in enumerate(clips[:n]):
        m = read_meta(c)
        vid = str(m["video_id"])
        fps = float(m.get("fps", 25))
        dur = float(m.get("duration_s", 60))
        nods = gold_by.get(vid, [])
        times = [dur * 0.15]
        if nods:
            e = nods[0]
            times = [max(0, e.start_s - 0.4), e.start_s + 0.15 * e.duration, e.end_s - 0.1, min(dur, e.end_s + 0.5)]
        else:
            times = [dur * x for x in (0.1, 0.35, 0.6, 0.85)]
        cap = cv2.VideoCapture(str(c / "clip.mp4"))
        for j, t in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if ok:
                ax.imshow(frame[:, :, ::-1])
            else:
                ax.set_facecolor("#ddd")
            if j == 0:
                ax.set_ylabel(vid, fontsize=9)
        cap.release()
    fig.suptitle("Examples of participant and head-motion variability in the selected subset.\nNo demographic attributes are inferred from appearance.", fontsize=12)
    save_publication_figure(fig, FIG / "dataset" / "realtalk_participant_diversity", log, "data/working/pilot/*.mp4 + gold events", force)


def fig_gold_timelines(log: FigureLog, force: bool) -> None:
    gold = gold_nods(load_events(ROOT / "data" / "gold" / "events.csv"))
    if not gold:
        log.skip("figures/annotations/*_gold_timeline", "no class-1 rows in data/gold/events.csv")
        return
    by = {}
    for e in gold:
        by.setdefault(e.video_id, []).append(e)
    clips = {str(read_meta(c)["video_id"]): c for c in list_clip_dirs(default_pilot_dir())}
    n = 0
    for vid, evs in list(by.items())[:8]:
        dur = 60.0
        if vid in clips:
            dur = float(read_meta(clips[vid]).get("duration_s", 60))
        fig, ax = plt.subplots(figsize=(9.5, 1.8))
        ax.set_xlim(0, dur)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("time (s)")
        ax.set_title(f"{vid}: gold clear-nod intervals (class 1)")
        labelled = False
        for e in evs:
            ax.axvspan(e.start_s, e.end_s, color="C3", alpha=0.55, label="gold nod" if not labelled else None)
            labelled = True
        if labelled:
            ax.legend(loc="upper right")
        save_publication_figure(fig, FIG / "annotations" / f"{vid}_gold_timeline", log, "data/gold/events.csv", force)
        n += 1
    if n == 0:
        log.skip("figures/annotations/*_gold_timeline", "no videos to plot")


def fig_annotation_time(log: FigureLog, force: bool) -> None:
    name = "figures/annotations/annotation_time_distribution"
    path = ROOT / "data" / "gold" / "annotation_log.csv"
    if not require_files(log, name, path):
        return
    df = pd.read_csv(path)
    df = df.dropna(how="all")
    if df.empty or "annotation_time_s" not in df.columns:
        log.skip(name, "annotation_log.csv has no timing rows yet")
        return
    df["annotation_time_s"] = pd.to_numeric(df["annotation_time_s"], errors="coerce")
    df = df.dropna(subset=["annotation_time_s"])
    if df.empty:
        log.skip(name, "no numeric annotation_time_s")
        return
    dur = pd.to_numeric(df.get("video_duration_s", 60), errors="coerce").fillna(60)
    df["minutes_per_source_minute"] = (df["annotation_time_s"] / 60.0) / (dur / 60.0)
    out_csv = RES / "annotation_time_results.csv"
    df.to_csv(out_csv, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].bar(df["video_id"].astype(str), df["annotation_time_s"] / 60.0, color="C0")
    axes[0].set_ylabel("annotation time (min)")
    axes[0].set_title("Time spent annotating each video")
    axes[0].tick_params(axis="x", rotation=40)
    axes[1].bar(df["video_id"].astype(str), df["minutes_per_source_minute"], color="C1")
    axes[1].set_ylabel("annotation min / source min")
    axes[1].set_title("Annotation effort ratio")
    axes[1].tick_params(axis="x", rotation=40)
    save_publication_figure(fig, FIG / "annotations" / "annotation_time_distribution", log, str(out_csv), force)


def fig_pitch_traces(log: FigureLog, force: bool) -> None:
    gold = gold_nods(load_events(ROOT / "data" / "gold" / "events.csv"))
    pred_path = RES / "pilot_nod_rule_predictions.csv"
    hp_dir = ROOT / "data" / "headpose"
    if not hp_dir.exists():
        log.skip("figures/rule_baseline/*_pitch_trace", "missing data/headpose/*.csv")
        return
    pred = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    files = sorted(hp_dir.glob("*.csv"))[:5]
    if not files:
        log.skip("figures/rule_baseline/*_pitch_trace", "no head-pose CSVs")
        return
    for f in files:
        vid = f.stem
        df = pd.read_csv(f)
        if "pitch" not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(df["time_s"], df["pitch"], color="C0", lw=1.1, label="pitch (deg)")
        g_lab = p_lab = False
        for e in gold:
            if e.video_id != vid:
                continue
            ax.axvspan(e.start_s, e.end_s, color="C3", alpha=0.28, label="gold nod" if not g_lab else None)
            g_lab = True
        if len(pred):
            sub = pred[pred.video_id.astype(str) == vid]
            for _, r in sub.iterrows():
                ax.axvspan(float(r.start_s), float(r.end_s), color="C2", alpha=0.2, label="predicted nod" if not p_lab else None)
                p_lab = True
        ax.set_xlabel("time (s)")
        ax.set_ylabel("pitch (deg)")
        ax.set_title(f"{vid}: pitch with gold and predicted nod intervals")
        ax.legend(loc="upper right")
        src = f"{f}" + (f"; {pred_path}" if pred_path.exists() else "")
        save_publication_figure(fig, FIG / "rule_baseline" / f"{vid}_pitch_trace", log, src, force)


def fig_rule_metrics(log: FigureLog, force: bool) -> None:
    metrics_p = RES / "pilot_nod_rule_metrics.json"
    if not require_files(log, "figures/rule_baseline/event_metrics", metrics_p):
        return
    obj = json.loads(metrics_p.read_text())
    ev = obj["event"]["iou_0p30"]
    rows = pd.DataFrame(
        [
            {"metric": "Precision", "value": ev["precision"]},
            {"metric": "Recall", "value": ev["recall"]},
            {"metric": "F1", "value": ev["f1"]},
        ]
    )
    csv_p = RES / "rule_metrics.csv"
    if not csv_p.exists():
        rows.assign(method="rule_nod", split=obj.get("split", "pilot_dev")).to_csv(csv_p, index=False)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar(rows["metric"], rows["value"], color=["C0", "C1", "C2"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("event-level score (IoU ≥ 0.30)")
    ax.set_title("Rule-based nod detector (PILOT/DEV)")
    save_publication_figure(fig, FIG / "rule_baseline" / "event_metrics", log, str(metrics_p), force)

    fm = obj.get("frame", {})
    if all(k in fm for k in ("tn", "fp", "fn", "tp")):
        cm = np.array([[fm["tn"], fm["fp"]], [fm["fn"], fm["tp"]]], dtype=int)
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["pred non-nod", "pred nod"])
        ax.set_yticks([0, 1], ["gold non-nod", "gold nod"])
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=14)
        ax.set_title("Frame-level confusion (PILOT/DEV)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        save_publication_figure(fig, FIG / "rule_baseline" / "confusion_matrix", log, str(metrics_p), force)
    else:
        log.skip("figures/rule_baseline/confusion_matrix", "frame tn/fp/fn/tp missing in metrics JSON")

    scores = RES / "pilot_nod_frame_scores.npz"
    if scores.exists():
        z = np.load(scores)
        y_true, y_score = z["y_true"], z["y_score"]
        if len(np.unique(y_true)) < 2:
            log.skip("figures/rule_baseline/precision_recall_curve", "only one class in frame scores")
            return
        from sklearn.metrics import precision_recall_curve

        p, r, _ = precision_recall_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.plot(r, p, color="C0")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Rule detector precision–recall (frames, PILOT/DEV)")
        save_publication_figure(fig, FIG / "rule_baseline" / "precision_recall_curve", log, str(scores), force)
    else:
        log.skip("figures/rule_baseline/precision_recall_curve", f"missing {scores}")


def fig_class_counts(log: FigureLog, force: bool) -> None:
    name = "figures/annotations/class_counts"
    path = ROOT / "data" / "gold" / "events.csv"
    if not require_files(log, name, path):
        return
    df = load_events(path)
    if df.empty:
        log.skip(name, "events.csv is empty — annotate first")
        return
    labs = [parse_label(x) for x in df["label"]]
    n1 = sum(x == 1 for x in labs)
    n0 = sum(x == 0 for x in labs)
    if n1 + n0 == 0:
        log.skip(name, "no 1/0 labels in events.csv yet")
        return
    counts = pd.DataFrame(
        [
            {"label": "1 clear nod", "count": n1},
            {"label": "0 unclear", "count": n0},
        ]
    )
    counts.to_csv(RES / "gold_class_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(counts["label"], counts["count"], color=["C3", "0.65"])
    ax.set_ylabel("annotated intervals")
    ax.set_title("Gold labels (class 1 is the only positive)")
    save_publication_figure(fig, FIG / "annotations" / "class_counts", log, str(path), force)

    if "video_id" not in df.columns:
        return
    per = []
    for vid, sub in df.groupby(df["video_id"].astype(str)):
        labs_v = [parse_label(x) for x in sub["label"]]
        per.append({"video_id": vid, "clear_nod": sum(x == 1 for x in labs_v), "unclear": sum(x == 0 for x in labs_v)})
    per_df = pd.DataFrame(per).sort_values("video_id")
    per_df.to_csv(RES / "gold_class_counts_per_video.csv", index=False)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.7 * len(per_df)), 3.8))
    x = np.arange(len(per_df))
    ax.bar(x, per_df["clear_nod"], label="1 clear nod", color="C3")
    ax.bar(x, per_df["unclear"], bottom=per_df["clear_nod"], label="0 unclear", color="0.65")
    ax.set_xticks(x)
    ax.set_xticklabels(per_df["video_id"].astype(str), rotation=40, ha="right")
    ax.set_ylabel("intervals")
    ax.set_title("Gold class counts per video")
    ax.legend()
    save_publication_figure(fig, FIG / "annotations" / "class_counts_per_video", log, str(path), force)


def fig_grid_heatmap(log: FigureLog, force: bool) -> None:
    name = "figures/rule_baseline/dev_grid_heatmap"
    path = RES / "rule_nod_dev_grid.csv"
    if not require_files(log, name, path):
        return
    df = pd.read_csv(path)
    need = {"min_range_deg", "min_dur", "event_f1_iou0.30"}
    if not need.issubset(df.columns):
        log.skip(name, f"{path} missing columns {need - set(df.columns)}")
        return
    pivot = df.pivot(index="min_range_deg", columns="min_dur", values="event_f1_iou0.30")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(pivot.to_numpy(float), cmap="viridis", vmin=0, vmax=1, origin="upper")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("min_dur (s)")
    ax.set_ylabel("min_range_deg")
    ax.set_title("DEV grid search: event F1 at IoU 0.30")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = float(pivot.to_numpy()[i, j])
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.45 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="F1")
    save_publication_figure(fig, FIG / "rule_baseline" / "dev_grid_heatmap", log, str(path), force)

    if {"precision", "recall"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        sc = ax.scatter(df["recall"], df["precision"], c=df["event_f1_iou0.30"], cmap="viridis", s=70, vmin=0, vmax=1)
        for _, r in df.iterrows():
            ax.annotate(
                f"{r['min_range_deg']}/{r['min_dur']}",
                (r["recall"], r["precision"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("DEV grid: precision vs recall (labelled by range/dur)")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="F1")
        save_publication_figure(fig, FIG / "rule_baseline" / "dev_grid_precision_recall", log, str(path), force)


def fig_per_video_f1(log: FigureLog, force: bool) -> None:
    name = "figures/rule_baseline/per_video_f1"
    path = RES / "pilot_nod_per_video.csv"
    if not require_files(log, name, path):
        return
    df = pd.read_csv(path)
    if df.empty or "f1" not in df.columns:
        log.skip(name, "per-video CSV empty")
        return
    df = df.sort_values("video_id")
    fig, ax = plt.subplots(figsize=(max(6.5, 0.7 * len(df)), 3.8))
    ax.bar(df["video_id"].astype(str), df["f1"], color="C0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("event F1 (IoU 0.30)")
    ax.set_title("Per-video rule F1 (PILOT/DEV)")
    ax.tick_params(axis="x", rotation=40)
    save_publication_figure(fig, FIG / "rule_baseline" / "per_video_f1", log, str(path), force)

    if {"n_gold", "n_pred"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(max(6.5, 0.8 * len(df)), 3.8))
        x = np.arange(len(df))
        w = 0.35
        ax.bar(x - w / 2, df["n_gold"], w, label="gold nods", color="C3")
        ax.bar(x + w / 2, df["n_pred"], w, label="rule detections", color="C2")
        ax.set_xticks(x)
        ax.set_xticklabels(df["video_id"].astype(str), rotation=40, ha="right")
        ax.set_ylabel("count")
        ax.set_title("Gold nods vs rule detections per video")
        ax.legend()
        save_publication_figure(fig, FIG / "rule_baseline" / "gold_vs_pred_counts", log, str(path), force)


def fig_iou_thresholds(log: FigureLog, force: bool) -> None:
    name = "figures/rule_baseline/iou_thresholds"
    path = RES / "pilot_nod_iou_thresholds.csv"
    metrics_p = RES / "pilot_nod_rule_metrics.json"
    if path.exists():
        df = pd.read_csv(path)
        src = str(path)
    elif metrics_p.exists():
        obj = json.loads(metrics_p.read_text())
        rows = []
        for thr, key in ((0.10, "iou_0p10"), (0.30, "iou_0p30"), (0.50, "iou_0p50")):
            m = obj.get("event", {}).get(key)
            if not m:
                continue
            rows.append({"iou_threshold": thr, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
        df = pd.DataFrame(rows)
        src = str(metrics_p)
    else:
        log.skip(name, f"missing {path} and {metrics_p}")
        return
    if df.empty:
        log.skip(name, "no IoU rows")
        return
    x = np.arange(len(df))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - w, df["precision"], w, label="Precision")
    ax.bar(x, df["recall"], w, label="Recall")
    ax.bar(x + w, df["f1"], w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"IoU {t:g}" for t in df["iou_threshold"]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("event-level score")
    ax.set_title("Rule detector vs IoU threshold (PILOT/DEV)")
    ax.legend()
    save_publication_figure(fig, FIG / "rule_baseline" / "iou_thresholds", log, src, force)


def fig_pitch_nod_vs_background(log: FigureLog, force: bool) -> None:
    name = "figures/rule_baseline/pitch_nod_vs_background"
    path = RES / "pilot_nod_pitch_by_gold.csv"
    if not require_files(log, name, path):
        return
    df = pd.read_csv(path)
    if df.empty or "pitch_deg" not in df.columns:
        log.skip(name, "pitch CSV empty")
        return
    nod = df.loc[df["in_gold_nod"] == 1, "pitch_deg"]
    bg = df.loc[df["in_gold_nod"] == 0, "pitch_deg"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].hist(bg, bins=40, alpha=0.7, label="background", color="0.6", density=True)
    axes[0].hist(nod, bins=40, alpha=0.7, label="gold nod", color="C3", density=True)
    axes[0].set_xlabel("pitch (deg)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Pitch during gold nods vs background")
    axes[0].legend()
    axes[1].set_ylabel("pitch (deg)")
    axes[1].set_title("Pitch distribution")
    boxes, names = [], []
    if len(bg.dropna()):
        boxes.append(bg.dropna().to_numpy())
        names.append("background")
    if len(nod.dropna()):
        boxes.append(nod.dropna().to_numpy())
        names.append("gold nod")
    if boxes:
        axes[1].boxplot(boxes)
        axes[1].set_xticklabels(names)
    save_publication_figure(fig, FIG / "rule_baseline" / "pitch_nod_vs_background", log, str(path), force)


def fig_event_durations(log: FigureLog, force: bool) -> None:
    name = "figures/rule_baseline/event_duration_hist"
    path = RES / "pilot_nod_event_durations.csv"
    if not require_files(log, name, path):
        return
    df = pd.read_csv(path)
    if df.empty or "duration_s" not in df.columns:
        log.skip(name, "duration CSV empty")
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for src, color in (("gold", "C3"), ("rule", "C2")):
        sub = df.loc[df["source"] == src, "duration_s"].dropna()
        if len(sub):
            ax.hist(sub, bins=20, alpha=0.55, label=src, color=color)
    ax.set_xlabel("event duration (s)")
    ax.set_ylabel("count")
    ax.set_title("Gold vs rule event durations")
    ax.legend()
    save_publication_figure(fig, FIG / "rule_baseline" / "event_duration_hist", log, str(path), force)


def fig_pose_gold_traces(log: FigureLog, force: bool) -> None:
    gold = gold_nods(load_events(ROOT / "data" / "gold" / "events.csv"))
    hp_dir = ROOT / "data" / "headpose"
    if not hp_dir.exists():
        log.skip("figures/pilot_nod/*_pose_gold", "missing data/headpose/*.csv")
        return
    files = sorted(hp_dir.glob("*.csv"))[:10]
    if not files:
        log.skip("figures/pilot_nod/*_pose_gold", "no head-pose CSVs")
        return
    n = 0
    for f in files:
        vid = f.stem
        df = pd.read_csv(f)
        cols = [c for c in ("pitch", "yaw", "roll") if c in df.columns]
        if not cols:
            continue
        fig, axes = plt.subplots(len(cols), 1, figsize=(10, 2.0 * len(cols) + 0.6), sharex=True)
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            ax.plot(df["time_s"], df[col], lw=1.0, color="C0")
            ax.set_ylabel(f"{col} (deg)")
            labelled = False
            for e in gold:
                if e.video_id != vid:
                    continue
                ax.axvspan(e.start_s, e.end_s, color="C3", alpha=0.25, label="gold nod" if not labelled else None)
                labelled = True
        axes[-1].set_xlabel("time (s)")
        axes[0].set_title(f"{vid} — red = gold clear nod (class 1)")
        if labelled:
            axes[0].legend(loc="upper right")
        save_publication_figure(fig, FIG / "pilot_nod" / f"{vid}_pose_gold", log, str(f), force)
        n += 1
    if n == 0:
        log.skip("figures/pilot_nod/*_pose_gold", "no usable pose columns")


def fig_error_pitch_windows(log: FigureLog, force: bool) -> None:
    pred_path = RES / "pilot_nod_rule_predictions.csv"
    gold = gold_nods(load_events(ROOT / "data" / "gold" / "events.csv"))
    hp_dir = ROOT / "data" / "headpose"
    if not pred_path.exists() or not gold or not hp_dir.exists():
        log.skip(
            "figures/error_analysis/pitch_windows",
            "need gold events, pilot_nod_rule_predictions.csv, and data/headpose",
        )
        return
    pred_df = pd.read_csv(pred_path)
    pred = [
        Event(str(r.video_id), float(r.start_s), float(r.end_s), "nod")
        for _, r in pred_df.iterrows()
    ]
    poses = {f.stem: pd.read_csv(f) for f in hp_dir.glob("*.csv")}
    tps, fps, fns = match_pairs(pred, gold, 0.30)
    panels = [
        ("tp", [p for p, _ in tps[:3]], "C2"),
        ("fp", fps[:3], "C1"),
        ("fn", fns[:3], "C3"),
    ]
    any_ok = False
    for tag, evs, color in panels:
        for i, ev in enumerate(evs):
            df = poses.get(ev.video_id)
            if df is None or "pitch" not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8.5, 2.6))
            ax.plot(df["time_s"], df["pitch"], color="C0", lw=1.0)
            ax.axvspan(ev.start_s, ev.end_s, color=color, alpha=0.35)
            ax.set_xlim(max(0, ev.start_s - 2), ev.end_s + 2)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("pitch (deg)")
            ax.set_title(f"{tag.upper()} {i+1}: {ev.video_id} {ev.start_s:.2f}–{ev.end_s:.2f}s")
            save_publication_figure(
                fig,
                FIG / "error_analysis" / f"{tag}_{i+1}_{ev.video_id}",
                log,
                str(pred_path),
                force,
            )
            any_ok = True
    if not any_ok:
        log.skip("figures/error_analysis/pitch_windows", "no matching pose CSVs for TP/FP/FN events")


def fig_pseudo(log: FigureLog, force: bool) -> None:
    path = ROOT / "data" / "pseudo" / "pseudo_events.csv"
    if not require_files(log, "figures/pseudo_labels/class_distribution", path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    vc = df["label"].astype(str).value_counts()
    ax.bar(vc.index, vc.values, color="C0")
    ax.set_ylabel("count")
    ax.set_title("Pseudo-label class counts")
    save_publication_figure(fig, FIG / "pseudo_labels" / "class_distribution", log, str(path), force)
    if "confidence" in df.columns:
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        ax.hist(pd.to_numeric(df["confidence"], errors="coerce").dropna(), bins=20, color="C0")
        ax.set_xlabel("rule confidence")
        ax.set_ylabel("count")
        ax.set_title("Pseudo-label confidence distribution")
        save_publication_figure(fig, FIG / "pseudo_labels" / "confidence_distribution", log, str(path), force)
    else:
        log.skip("figures/pseudo_labels/confidence_distribution", "no confidence column")
    log.skip(
        "figures/pseudo_labels/pseudo_label_examples",
        "need real clip.mp4 frames + pseudo_events.csv with video_id, start_s, confidence",
    )


def fig_videomae(log: FigureLog, force: bool) -> None:
    hist = list((RES).glob("*_history.csv"))
    if not hist:
        log.skip("figures/videomae/*_training_loss", "missing results/<run_id>_history.csv")
        log.skip("figures/videomae/final_confusion_matrix", "missing results/videomae_gold_test.json")
        return
    for path in hist:
        run_id = path.stem.replace("_history", "")
        df = pd.read_csv(path)
        if "train_loss" in df.columns:
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            ax.plot(df["epoch"], df["train_loss"], marker="o")
            ax.set_xlabel("epoch")
            ax.set_ylabel("training loss")
            ax.set_title(f"{run_id}: training loss")
            save_publication_figure(fig, FIG / "videomae" / f"{run_id}_training_loss", log, str(path), force)
        if "f1" in df.columns:
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            ax.plot(df["epoch"], df["f1"], marker="o", color="C1")
            ax.set_xlabel("epoch")
            ax.set_ylabel("DEV F1")
            ax.set_title(f"{run_id}: development F1")
            save_publication_figure(fig, FIG / "videomae" / f"{run_id}_dev_f1", log, str(path), force)
        if "validation_loss" in df.columns:
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            ax.plot(df["epoch"], df["validation_loss"], marker="o", color="C2")
            ax.set_xlabel("epoch")
            ax.set_ylabel("validation loss")
            ax.set_title(f"{run_id}: validation loss")
            save_publication_figure(fig, FIG / "videomae" / f"{run_id}_validation_loss", log, str(path), force)


def fig_model_comparison(log: FigureLog, force: bool) -> None:
    name = "figures/final_results/model_comparison"
    rows = []
    rule_p = RES / "pilot_nod_rule_metrics.json"
    if rule_p.exists():
        ev = json.loads(rule_p.read_text())["event"]["iou_0p30"]
        rows.append({"method": "Rule-based detector", "precision": ev["precision"], "recall": ev["recall"], "f1": ev["f1"], "source": str(rule_p)})
    for label, path in (
        ("Frozen VideoMAE", RES / "videomae_frozen_gold_test.json"),
        ("Fine-tuned VideoMAE", RES / "videomae_gold_test.json"),
    ):
        if path.exists():
            obj = json.loads(path.read_text())
            ev = obj.get("event", obj)
            rows.append(
                {
                    "method": label,
                    "precision": ev.get("precision", ev.get("event_precision")),
                    "recall": ev.get("recall", ev.get("event_recall")),
                    "f1": ev.get("f1", ev.get("event_f1")),
                    "source": str(path),
                }
            )
    if len(rows) < 1:
        log.skip(name, "no result JSON files yet")
        return
    if len(rows) < 2:
        log.skip(name, "need at least two methods; VideoMAE results not written yet")
        return
    df = pd.DataFrame(rows)
    df.to_csv(RES / "main_model_comparison.csv", index=False)
    x = np.arange(len(df))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - w, df["precision"], w, label="Precision")
    ax.bar(x, df["recall"], w, label="Recall")
    ax.bar(x + w, df["f1"], w, label="F1")
    ax.set_xticks(x, df["method"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("event-level score")
    ax.set_title("Model comparison (accuracy is not the headline metric)")
    ax.legend()
    save_publication_figure(fig, FIG / "final_results" / "model_comparison", log, "results/*.json", force)


def fig_ablations(log: FigureLog, force: bool) -> None:
    mapping = [
        ("figures/ablations/training_data_size", RES / "ablation_data_size.csv"),
        ("figures/ablations/pseudo_confidence_ablation", RES / "ablation_confidence.csv"),
        ("figures/ablations/frozen_vs_finetuned", RES / "ablation_frozen_vs_finetuned.csv"),
    ]
    for name, path in mapping:
        if not path.exists():
            log.skip(name, f"missing {path}")
            continue
        df = pd.read_csv(path)
        ycol = "f1" if "f1" in df.columns else df.columns[-1]
        xcol = df.columns[0]
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.bar(df[xcol].astype(str), df[ycol], color="C0")
        ax.set_ylabel(str(ycol))
        ax.set_ylim(0, 1)
        ax.set_title(path.stem.replace("_", " "))
        stem = name.split("/", 1)[1] if "/" in name else name
        save_publication_figure(fig, FIG / stem, log, str(path), force)


def fig_error_sheets(log: FigureLog, force: bool) -> None:
    fig_error_pitch_windows(log, force)
    for stem in ("true_positives", "false_positives", "false_negatives"):
        log.skip(
            f"figures/error_analysis/{stem}_frames",
            "need real mp4 frames plus matched events; pitch windows are generated separately",
        )
    log.skip(
        "figures/final_results/nod_motion_examples",
        "need real annotated mp4s to sample before/down/bottom/up/after frames",
    )


def _box(ax, x, y, w, h, text, facecolor):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=facecolor, edgecolor="#333", linewidth=1.2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, wrap=True)


def fig_pipeline(log: FigureLog, force: bool) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    steps = [
        (0.2, "RealTalk\nvideo", "#e8f1f8"),
        (2.0, "EMOCA\nhead pose", "#e8f1f8"),
        (3.8, "Rule-based\ndetector", "#fff3cd"),
        (5.6, "Pseudo\nlabels", "#fff3cd"),
        (7.4, "VideoMAE\ntraining", "#d4edda"),
        (9.2, "Gold TEST\nevaluation", "#f8d7da"),
    ]
    for x, text, c in steps:
        _box(ax, x, 1.15, 1.55, 1.2, text, c)
    for x1, x2 in zip([s[0] for s in steps], [s[0] for s in steps][1:]):
        ax.annotate("", xy=(x2, 1.75), xytext=(x1 + 1.55, 1.75), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(0.2, 0.35, "Human annotations = GOLD", color="#a94442", fontsize=10)
    ax.text(3.8, 0.35, "Rule outputs = WEAK / PSEUDO LABELS", color="#8a6d3b", fontsize=10)
    ax.text(7.4, 0.35, "VideoMAE outputs = MODEL PREDICTIONS", color="#3c763d", fontsize=10)
    ax.set_title("Weak-supervision pipeline (no quantitative results in this figure)")
    save_publication_figure(fig, FIG / "final_results" / "pipeline_overview", log, "methodology (no metrics)", force)

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    boxes = [
        (0.2, "RealTalk\nexamples"),
        (2.1, "Head-pose\nsignal"),
        (4.0, "Weak\nlabels"),
        (5.9, "VideoMAE"),
        (7.8, "Head-nod\nprediction"),
    ]
    for x, t in boxes:
        _box(ax, x, 0.7, 1.6, 1.2, t, "#eef5fb")
    for x1, x2 in zip([b[0] for b in boxes], [b[0] for b in boxes][1:]):
        ax.annotate("", xy=(x2, 1.3), xytext=(x1 + 1.6, 1.3), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title("Project overview (no invented scores)")
    save_publication_figure(fig, FIG / "github_overview", log, "methodology schematic", force)


def fig_github_results(log: FigureLog, force: bool) -> None:
    name = "figures/github_results_summary"
    rule_p = RES / "pilot_nod_rule_metrics.json"
    vmae_p = RES / "videomae_gold_test.json"
    if not rule_p.exists() or not vmae_p.exists():
        log.skip(name, f"only generate after both {rule_p.name} and {vmae_p.name} exist")
        return
    rule = json.loads(rule_p.read_text())
    vmae = json.loads(vmae_p.read_text())
    n_vid = rule.get("n_pilot_videos", 0)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = ["Rule F1", "VideoMAE F1", "Gold videos"]
    vals = [rule["event"]["iou_0p30"]["f1"], vmae.get("event_f1", vmae.get("f1", float("nan"))), n_vid]
    ax.bar(labels[:2], vals[:2], color=["C0", "C2"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("event F1")
    ax.set_title(f"Summary (n gold videos = {n_vid})")
    save_publication_figure(fig, FIG / "github_results_summary", log, f"{rule_p}; {vmae_p}", force)


def disk_usage() -> str:
    import subprocess

    r = subprocess.run(["du", "-sh", str(FIG)], capture_output=True, text=True)
    return r.stdout.strip() or str(FIG)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="generate every available figure")
    p.add_argument("--force", action="store_true", help="overwrite existing png/jpg")
    args = p.parse_args()
    if not args.all:
        print("Pass --all to generate figures from saved results.")
        print("Example: python scripts/make_figures.py --all")
        return
    ensure_dirs()
    RES.mkdir(exist_ok=True)
    log = FigureLog()
    fig_split_summary(log, args.force)
    fig_realtalk_diversity(log, args.force)
    fig_gold_timelines(log, args.force)
    fig_class_counts(log, args.force)
    fig_annotation_time(log, args.force)
    fig_pose_gold_traces(log, args.force)
    fig_pitch_traces(log, args.force)
    fig_rule_metrics(log, args.force)
    fig_grid_heatmap(log, args.force)
    fig_per_video_f1(log, args.force)
    fig_iou_thresholds(log, args.force)
    fig_pitch_nod_vs_background(log, args.force)
    fig_event_durations(log, args.force)
    fig_pseudo(log, args.force)
    fig_videomae(log, args.force)
    fig_model_comparison(log, args.force)
    fig_ablations(log, args.force)
    fig_error_sheets(log, args.force)
    fig_pipeline(log, args.force)
    fig_github_results(log, args.force)

    print("\n========== FIGURES GENERATED ==========")
    for stem, src in log.generated:
        print(f"  {stem}\n    source: {src}")
    print("\n========== FIGURES SKIPPED ==========")
    for name, reason in log.skipped:
        print(f"  {name}\n    {reason}")
    print("\nOUTPUT DIRECTORY", FIG)
    print("TOTAL FIGURE DISK USAGE", disk_usage())


if __name__ == "__main__":
    main()
