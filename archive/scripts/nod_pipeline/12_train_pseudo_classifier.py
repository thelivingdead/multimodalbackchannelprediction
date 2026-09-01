#!/usr/bin/env python3
"""Train a nod classifier on TRAIN pseudo-labels; tune on DEV gold; evaluate TEST gold.

Window-level features from FLAME pitch/yaw/roll. Architectures:
  logreg, random forest, MLP
Hyperparameters searched on DEV only. TEST is touched once at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_metrics import greedy_match, prf, window_overlaps_event  # noqa: E402
from pose_utils import read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FEATS = ["mean_pitch", "pitch_range", "max_velocity", "n_reversals", "duration", "yaw_std", "roll_std"]


def window_feats(pose: pd.DataFrame, person: str, t0: float, t1: float) -> dict | None:
    pcol, ycol, rcol = f"{person}_pitch", f"{person}_yaw", f"{person}_roll"
    sub = pose[(pose["timestamp"] >= t0) & (pose["timestamp"] <= t1)].copy()
    for c in (pcol, ycol, rcol):
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    pitch = sub[pcol].dropna().to_numpy(float) if pcol in sub else np.array([])
    if pitch.size < 4:
        return None
    yaw = sub[ycol].dropna().to_numpy(float) if ycol in sub else np.zeros_like(pitch)
    roll = sub[rcol].dropna().to_numpy(float) if rcol in sub else np.zeros_like(pitch)
    vel = np.gradient(pitch)
    sign = np.sign(vel)
    sign[sign == 0] = 1
    return {
        "mean_pitch": float(np.mean(pitch)),
        "pitch_range": float(np.ptp(pitch)),
        "max_velocity": float(np.max(np.abs(vel))),
        "n_reversals": int(np.sum(np.diff(sign) != 0)),
        "duration": float(t1 - t0),
        "yaw_std": float(np.std(yaw)) if yaw.size else 0.0,
        "roll_std": float(np.std(roll)) if roll.size else 0.0,
    }


def load_pose(subset: Path, vid: str) -> pd.DataFrame:
    return pd.read_csv(subset / vid / "pose.csv")


def build_windows(
    subset: Path,
    video_ids: list[str],
    events_by_vid: dict[str, list[tuple[float, float]]],
    person: str,
    win: float,
    hop: float,
    neg_ratio: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for vid in video_ids:
        pose_path = subset / vid / "pose.csv"
        if not pose_path.exists():
            continue
        pose = load_pose(subset, vid)
        dur = float(pose["timestamp"].max())
        ev = events_by_vid.get(vid, [])
        t = 0.0
        cand = []
        while t + win <= dur + 1e-9:
            t1 = t + win
            y = window_overlaps_event(t, t1, ev)
            feats = window_feats(pose, person, t, t1)
            if feats:
                cand.append({"video_id": vid, "person": person, "start_time": t, "end_time": t1, "y": y, **feats})
            t += hop
        pos = [r for r in cand if r["y"] == 1]
        neg = [r for r in cand if r["y"] == 0]
        n_keep = min(len(neg), max(len(pos) * int(neg_ratio), 8))
        if len(neg) > n_keep:
            idx = rng.choice(len(neg), size=n_keep, replace=False)
            neg = [neg[i] for i in idx]
        rows.extend(pos + neg)
    return pd.DataFrame(rows)


def events_from_windows(df: pd.DataFrame, pred: np.ndarray, min_dur: float = 0.3) -> dict[str, list[tuple[float, float]]]:
    """Merge consecutive positive windows into events per video."""
    out: dict[str, list[tuple[float, float]]] = {}
    tmp = df.copy()
    tmp["pred"] = pred
    for vid, g in tmp.groupby("video_id"):
        g = g.sort_values("start_time")
        segs = []
        cur0 = cur1 = None
        for _, r in g.iterrows():
            if int(r.pred) != 1:
                if cur0 is not None:
                    segs.append((cur0, cur1))
                    cur0 = cur1 = None
                continue
            if cur0 is None:
                cur0, cur1 = float(r.start_time), float(r.end_time)
            elif float(r.start_time) <= cur1 + 1e-6:
                cur1 = max(cur1, float(r.end_time))
            else:
                segs.append((cur0, cur1))
                cur0, cur1 = float(r.start_time), float(r.end_time)
        if cur0 is not None:
            segs.append((cur0, cur1))
        out[str(vid)] = [(a, b) for a, b in segs if b - a >= min_dur]
    return out


def event_f1(pred_events, gold_df, video_ids, person="p0") -> dict:
    tp = fp = fn = 0
    gold_sub = gold_df[gold_df.person.astype(str) == person] if "person" in gold_df.columns else gold_df
    for vid in video_ids:
        g = [
            (float(r.start_time), float(r.end_time))
            for _, r in gold_sub[gold_sub.video_id.astype(str) == str(vid)].iterrows()
        ]
        p = pred_events.get(str(vid), [])
        a, b, c = greedy_match(p, g, 0.2)
        tp += a
        fp += b
        fn += c
    return prf(tp, fp, fn)


def model_zoo():
    return [
        (
            "logreg_C0.1",
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
                ]
            ),
        ),
        (
            "logreg_C1",
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)),
                ]
            ),
        ),
        (
            "logreg_C10",
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(C=10.0, class_weight="balanced", max_iter=3000)),
                ]
            ),
        ),
        (
            "rf_d4",
            RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced", random_state=42),
        ),
        (
            "rf_d8",
            RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42),
        ),
        (
            "mlp_32",
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(32,),
                            max_iter=400,
                            random_state=42,
                            early_stopping=True,
                            validation_fraction=0.15,
                        ),
                    ),
                ]
            ),
        ),
        (
            "mlp_64_32",
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            max_iter=500,
                            random_state=42,
                            early_stopping=True,
                            validation_fraction=0.15,
                        ),
                    ),
                ]
            ),
        ),
    ]


def main() -> None:
    t0 = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default=str(ROOT / "data" / "nod30"))
    p.add_argument("--pseudo", default=str(ROOT / "outputs" / "nod_pipeline" / "pseudo_labels_train.csv"))
    p.add_argument("--gold", default=str(ROOT / "outputs" / "nod_pipeline" / "gold_labels.csv"))
    p.add_argument("--splits", default=str(ROOT / "outputs" / "nod_pipeline" / "splits.json"))
    p.add_argument("--win", type=float, default=0.7)
    p.add_argument("--hop", type=float, default=0.2)
    args = p.parse_args()
    subset = Path(args.subset)
    splits = json.loads(Path(args.splits).read_text())
    gold = pd.read_csv(args.gold)
    pseudo = pd.read_csv(args.pseudo)

    def evmap(df, vids):
        m = {}
        sub = df[df.video_id.astype(str).isin(vids)]
        for vid, g in sub.groupby(sub.video_id.astype(str)):
            m[str(vid)] = [(float(r.start_time), float(r.end_time)) for _, r in g.iterrows()]
        return m

    train_ids = splits["train_videos"]
    dev_ids = splits["dev_videos"]
    test_ids = splits["test_videos"]

    train_df = build_windows(subset, train_ids, evmap(pseudo, train_ids), "p0", args.win, args.hop, 3.0, 42)
    dev_df = build_windows(subset, dev_ids, evmap(gold, dev_ids), "p0", args.win, args.hop, 3.0, 42)
    test_df = build_windows(subset, test_ids, evmap(gold, test_ids), "p0", args.win, args.hop, 3.0, 42)
    out = ROOT / "outputs" / "nod_pipeline"
    train_df.to_csv(out / "windows_train_pseudo.csv", index=False)
    print(
        f"windows train={len(train_df)} pos={int(train_df.y.sum())} | "
        f"dev={len(dev_df)} pos={int(dev_df.y.sum())} | test={len(test_df)} pos={int(test_df.y.sum())}"
    )
    if train_df.empty or train_df.y.nunique() < 2:
        raise SystemExit("Not enough pseudo-labelled windows to train.")

    search_rows = []
    best_name, best_f1, best_clf = None, -1.0, None
    Xtr, ytr = train_df[FEATS].to_numpy(), train_df["y"].to_numpy()
    Xdv, ydv = dev_df[FEATS].to_numpy(), dev_df["y"].to_numpy()
    for name, clf in model_zoo():
        t_fit = time.time()
        clf.fit(Xtr, ytr)
        fit_s = time.time() - t_fit
        pred_dv = clf.predict(Xdv)
        p_, r_, f1, _ = precision_recall_fscore_support(ydv, pred_dv, average="binary", zero_division=0)
        ev = events_from_windows(dev_df, pred_dv)
        evm = event_f1(ev, gold, dev_ids)
        loss = None
        inner = clf.named_steps["clf"] if hasattr(clf, "named_steps") else clf
        if hasattr(inner, "loss_curve_") and inner.loss_curve_:
            loss = float(inner.loss_curve_[-1])
        row = {
            "model": name,
            "dev_window_precision": float(p_),
            "dev_window_recall": float(r_),
            "dev_window_f1": float(f1),
            "dev_event_f1": evm["f1"],
            "dev_event_precision": evm["precision"],
            "dev_event_recall": evm["recall"],
            "fit_seconds": fit_s,
            "final_loss": loss,
        }
        search_rows.append(row)
        print(name, "dev window F1", round(f1, 3), "event F1", round(evm["f1"], 3), "loss", loss)
        # select by DEV event F1 (aligned with baseline metric)
        if evm["f1"] > best_f1:
            best_f1, best_name, best_clf = evm["f1"], name, clf

    pd.DataFrame(search_rows).to_csv(out / "hyperparam_search.csv", index=False)
    print("BEST on DEV (event F1):", best_name, best_f1)
    models_dir = ROOT / "outputs" / "nod_pipeline" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, models_dir / "best_pseudo_classifier.joblib")

    # TEST once
    Xte, yte = test_df[FEATS].to_numpy(), test_df["y"].to_numpy()
    pred_te = best_clf.predict(Xte)
    prob_te = best_clf.predict_proba(Xte)[:, 1] if hasattr(best_clf, "predict_proba") else pred_te
    p_, r_, f1, _ = precision_recall_fscore_support(yte, pred_te, average="binary", zero_division=0)
    try:
        ap = float(average_precision_score(yte, prob_te)) if len(np.unique(yte)) > 1 else float("nan")
    except Exception:
        ap = float("nan")
    ev = events_from_windows(test_df, pred_te)
    evm = event_f1(ev, gold, test_ids)
    result = {
        "best_model": best_name,
        "selected_by": "dev_event_f1",
        "test_window_precision": float(p_),
        "test_window_recall": float(r_),
        "test_window_f1": float(f1),
        "test_pr_auc": ap,
        "test_event_precision": evm["precision"],
        "test_event_recall": evm["recall"],
        "test_event_f1": evm["f1"],
        "test_event_tp": evm["tp"],
        "test_event_fp": evm["fp"],
        "test_event_fn": evm["fn"],
        "n_test_windows": int(len(test_df)),
        "n_test_pos_windows": int(yte.sum()),
        "note": "Trained on rule pseudo-labels from TRAIN videos only. TEST gold unused until now.",
    }
    (out / "learned_test_metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame([result]).to_csv(out / "learned_test_metrics.csv", index=False)
    test_df.assign(pred=pred_te, prob=prob_te).to_csv(out / "predictions_test.csv", index=False)
    print("\n=== TEST (held-out gold videos) ===")
    print(json.dumps(result, indent=2))
    elapsed = time.time() - t0
    with (out / "time_log.txt").open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  12_train_pseudo_classifier.py  {elapsed:.1f}s  best={best_name}\n")


if __name__ == "__main__":
    main()
