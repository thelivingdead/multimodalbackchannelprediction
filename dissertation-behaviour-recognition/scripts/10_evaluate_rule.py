#!/usr/bin/env python3
"""10 — Evaluate nod rule on PILOT/DEV. Writes metrics, predictions, diagnostic plots."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotations import gold_nods, load_events  # noqa: E402
from src.data import default_pilot_dir, list_clip_dirs, read_meta  # noqa: E402
from src.events import Event, greedy_match, match_pairs  # noqa: E402
from src.metrics import event_metrics, frame_metrics  # noqa: E402
from src.plotting import save_publication_figure  # noqa: E402
from src.rules.nod import NodRule  # noqa: E402
from src.utils import dump_json, load_yaml  # noqa: E402


def load_rule() -> NodRule:
    path = ROOT / "configs" / "rule_nod_best.yaml"
    cfg = load_yaml(path if path.exists() else ROOT / "configs" / "rule_nod.yaml")
    return NodRule(
        min_dur=float(cfg["min_dur"]),
        max_dur=float(cfg.get("max_dur", 1.4)),
        min_range_deg=float(cfg["min_range_deg"]),
        min_reversals=int(cfg.get("min_reversals", 2)),
        fps=float(cfg.get("fps", 25)),
    )


def frame_labels(pose: pd.DataFrame, events: list[Event], video_id: str) -> np.ndarray:
    y = np.zeros(len(pose), dtype=int)
    t = pose["time_s"].to_numpy(float)
    for e in events:
        if e.video_id != video_id:
            continue
        y |= ((t >= e.start_s) & (t <= e.end_s)).astype(int)
    return y


def diagnostic_plots(poses: dict, gold: list[Event], pred: list[Event], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tps, fps, fns = match_pairs(pred, gold, 0.30)

    def plot_one(ev: Event, tag: str, color: str) -> None:
        if ev.video_id not in poses:
            return
        df = poses[ev.video_id]
        if "pitch" not in df.columns:
            return
        fig, ax = plt.subplots(figsize=(10, 2.8))
        ax.plot(df["time_s"], df["pitch"], color="C0", lw=1.0)
        ax.axvspan(ev.start_s, ev.end_s, color=color, alpha=0.3)
        ax.set_xlim(max(0, ev.start_s - 2), ev.end_s + 2)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("pitch (deg)")
        ax.set_title(f"{tag}: {ev.video_id} {ev.start_s:.2f}–{ev.end_s:.2f}s")
        save_publication_figure(fig, out / f"{tag}_{ev.video_id}_{ev.start_s:.1f}", source="rule eval", force=True)

    if tps:
        plot_one(tps[0][0], "tp_clear", "C2")
        if len(tps) > 1:
            plot_one(tps[-1][0], "tp_subtle", "C2")
    if fps:
        plot_one(fps[0], "fp", "C1")
    if fns:
        plot_one(fns[0], "fn", "C3")
    if gold:
        plot_one(gold[0], "gold_example", "C3")


def write_extra_tables(
    poses: dict[str, pd.DataFrame],
    gold: list[Event],
    pred: list[Event],
    em: dict,
    results: Path,
) -> None:
    iou_rows = []
    for thr, key in ((0.10, "iou_0p10"), (0.30, "iou_0p30"), (0.50, "iou_0p50")):
        if key not in em:
            continue
        m = em[key]
        iou_rows.append(
            {
                "iou_threshold": thr,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
        )
    pd.DataFrame(iou_rows).to_csv(results / "pilot_nod_iou_thresholds.csv", index=False)

    pitch_rows = []
    for vid, df in poses.items():
        if "pitch" not in df.columns:
            continue
        t = df["time_s"].to_numpy(float)
        pitch = df["pitch"].to_numpy(float)
        mask = np.zeros(len(df), dtype=bool)
        for e in gold:
            if e.video_id == vid:
                mask |= (t >= e.start_s) & (t <= e.end_s)
        for pval, is_nod in zip(pitch, mask):
            pitch_rows.append({"video_id": vid, "pitch_deg": float(pval), "in_gold_nod": int(is_nod)})
    if pitch_rows:
        pitch_df = pd.DataFrame(pitch_rows)
        pitch_df.to_csv(results / "pilot_nod_pitch_by_gold.csv", index=False)
        summary = (
            pitch_df.groupby("in_gold_nod")["pitch_deg"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )
        summary["class"] = summary["in_gold_nod"].map({1: "gold_nod", 0: "background"})
        summary.to_csv(results / "pilot_nod_pitch_summary.csv", index=False)

    dur_rows = [{"source": "gold", "video_id": e.video_id, "duration_s": e.duration} for e in gold]
    dur_rows.extend({"source": "rule", "video_id": e.video_id, "duration_s": e.duration} for e in pred)
    pd.DataFrame(dur_rows).to_csv(results / "pilot_nod_event_durations.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--behaviour", default="nod")
    p.add_argument("--split", default="pilot")
    p.parse_args()
    rule = load_rule()
    gold_all = gold_nods(load_events(ROOT / "data" / "gold" / "events.csv"))
    y_true_all, y_pred_all, y_score_all = [], [], []
    pred_events: list[Event] = []
    poses: dict[str, pd.DataFrame] = {}
    pred_rows = []
    n_vid = 0
    duration = 0.0
    for c in list_clip_dirs(default_pilot_dir()):
        m = read_meta(c)
        vid = str(m["video_id"])
        hp = ROOT / "data" / "headpose" / f"{vid}.csv"
        if not hp.exists():
            continue
        pose = pd.read_csv(hp)
        poses[vid] = pose
        person = str(m.get("listener", "p0"))
        pe = rule.detect(pose, vid, person)
        pred_events.extend(pe)
        yt = frame_labels(pose, gold_all, vid)
        yp = frame_labels(pose, pe, vid)
        ys = rule.score_frames(pose, person)
        y_true_all.append(yt)
        y_pred_all.append(yp)
        y_score_all.append(ys[: len(yt)])
        n_vid += 1
        duration += float(m.get("duration_s", pose["time_s"].max()))
        for e in pe:
            pred_rows.append({"video_id": e.video_id, "start_s": e.start_s, "end_s": e.end_s, "label": "nod", "source": "rule"})
    if not y_true_all:
        raise SystemExit("No pose CSVs.")
    yt = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    ys = np.concatenate(y_score_all)
    gold = [g for g in gold_all if g.video_id in poses]
    fm = frame_metrics(yt, yp, ys)
    em = event_metrics(pred_events, gold)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    per_video = []
    for vid in sorted(poses):
        g = [e for e in gold if e.video_id == vid]
        p = [e for e in pred_events if e.video_id == vid]
        tp, fp, fn = greedy_match(p, g, 0.30)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_video.append(
            {
                "video_id": vid,
                "n_gold": len(g),
                "n_pred": len(p),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )
    pd.DataFrame(per_video).to_csv(results / "pilot_nod_per_video.csv", index=False)
    write_extra_tables(poses, gold, pred_events, em, results)
    out = {
        "n_pilot_videos": n_vid,
        "n_gold_clear_nods": len(gold),
        "duration_s": duration,
        "frame": fm,
        "event": em,
        "primary_event_f1_iou_0.30": em["primary_event_f1"],
        "split": "pilot_dev",
        "tuned_on": "pilot_dev",
        "test_touched": False,
    }
    results.mkdir(exist_ok=True)
    dump_json(results / "pilot_nod_rule_metrics.json", out)
    pd.DataFrame(pred_rows).to_csv(results / "pilot_nod_rule_predictions.csv", index=False)
    np.savez_compressed(
        results / "pilot_nod_frame_scores.npz",
        y_true=yt,
        y_pred=yp,
        y_score=ys,
    )
    pd.DataFrame(
        [
            {
                "method": "rule_nod",
                "split": "pilot_dev",
                "frame_precision": fm["precision"],
                "frame_recall": fm["recall"],
                "frame_f1": fm["f1"],
                "pr_auc": fm["pr_auc"],
                "event_precision": em["iou_0p30"]["precision"],
                "event_recall": em["iou_0p30"]["recall"],
                "event_f1": em["iou_0p30"]["f1"],
                "event_iou": 0.30,
                "n_videos": n_vid,
                "n_gold_nods": len(gold),
                "duration_s": duration,
            }
        ]
    ).to_csv(results / "rule_metrics.csv", index=False)
    figdir = ROOT / "figures" / "pilot_nod"
    diagnostic_plots(poses, gold, pred_events, figdir)
    findings = ROOT / "reports" / "pilot_nod_findings.md"
    findings.write_text(
        "\n".join(
            [
                "# Pilot nod rule findings",
                "",
                f"- Videos: {n_vid}",
                f"- Gold clear nods (class 1): {len(gold)}",
                f"- Duration evaluated: {duration:.1f}s",
                f"- Frame precision: {fm['precision']:.3f}",
                f"- Frame recall: {fm['recall']:.3f}",
                f"- Frame F1: {fm['f1']:.3f}",
                f"- PR-AUC: {fm['pr_auc']}",
                f"- Event precision (IoU 0.30): {em['iou_0p30']['precision']:.3f}",
                f"- Event recall (IoU 0.30): {em['iou_0p30']['recall']:.3f}",
                f"- Event F1 (IoU 0.30): {em['iou_0p30']['f1']:.3f}",
                f"- Event F1 (IoU 0.10): {em['iou_0p10']['f1']:.3f}",
                f"- Event F1 (IoU 0.50): {em['iou_0p50']['f1']:.3f}",
                "",
                "Gold positives are **human class 1 (clear nod)** only. Class 0 (unclear) is not a positive.",
                "This split is PILOT/DEV. GOLD TEST was not used.",
                "",
                "Extra tables: `pilot_nod_per_video.csv`, `pilot_nod_iou_thresholds.csv`, `pilot_nod_pitch_summary.csv`.",
                "Figures: `python scripts/make_figures.py --all`",
                "",
                "## Likely failure modes (inspect figures/pilot_nod)",
                "1. False positives: talker emphasis / non-nod pitch jitter passing the cycle test.",
                "2. False negatives: small-amplitude nods below min_range_deg.",
                "3. Boundary mismatch: predicted interval IoU < 0.30 vs human onset/offset.",
                "",
            ]
        )
    )
    print("FRAME", {k: fm[k] for k in ("precision", "recall", "f1", "pr_auc")})
    print("EVENT IoU0.30", em["iou_0p30"])
    print("Wrote", results / "pilot_nod_rule_metrics.json")


if __name__ == "__main__":
    main()
