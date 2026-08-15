#!/usr/bin/env python3
"""Step 5 — Baseline classifier on temporal windows, split by *video*.

Features per candidate interval:
  mean pitch, pitch range, max |velocity|, direction changes,
  duration, yaw std, roll std

Models: logistic regression, SVM, random forest.
Reports precision / recall / F1. Saves metrics + confusion matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_utils import list_clip_dirs, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FEATS = [
    "mean_pitch",
    "pitch_range",
    "max_velocity",
    "n_reversals",
    "duration",
    "yaw_std",
    "roll_std",
]


def window_features(pose_csv: Path, person: str, t0: float, t1: float) -> dict:
    df = pd.read_csv(pose_csv)
    pcol, ycol, rcol = f"{person}_pitch", f"{person}_yaw", f"{person}_roll"
    sub = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)].copy()
    for c in (pcol, ycol, rcol):
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    pitch = sub[pcol].dropna().to_numpy(float) if pcol in sub else np.array([0.0])
    yaw = sub[ycol].dropna().to_numpy(float) if ycol in sub else np.array([0.0])
    roll = sub[rcol].dropna().to_numpy(float) if rcol in sub else np.array([0.0])
    if pitch.size < 2:
        pitch = np.zeros(3)
    vel = np.gradient(pitch)
    sign = np.sign(vel)
    sign[sign == 0] = 1
    n_rev = int(np.sum(np.diff(sign) != 0))
    return {
        "mean_pitch": float(np.mean(pitch)),
        "pitch_range": float(np.ptp(pitch)),
        "max_velocity": float(np.max(np.abs(vel))),
        "n_reversals": n_rev,
        "duration": float(t1 - t0),
        "yaw_std": float(np.std(yaw)) if yaw.size else 0.0,
        "roll_std": float(np.std(roll)) if roll.size else 0.0,
    }


def build_table(subset: Path, labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in labels.iterrows():
        clip = subset / str(r.video_id)
        pose = clip / "pose.csv"
        if not pose.exists():
            continue
        feats = window_features(pose, str(r.person), float(r.start_time), float(r.end_time))
        y = str(r.label).strip().lower()
        y_bin = 1 if y == "nod" else 0
        rows.append({**r.to_dict(), **feats, "y": y_bin})
    return pd.DataFrame(rows)


def video_split(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 0):
    vids = df["video_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(vids)
    n_test = max(1, int(round(len(vids) * test_frac)))
    if n_test >= len(vids):
        n_test = max(1, len(vids) // 3)
    test_vids = set(vids[:n_test])
    train = df[~df["video_id"].isin(test_vids)]
    test = df[df["video_id"].isin(test_vids)]
    if train.empty or test.empty:
        # fallback: last video is test
        test_vids = {vids[-1]}
        train = df[~df["video_id"].isin(test_vids)]
        test = df[df["video_id"].isin(test_vids)]
    return train, test, sorted(test_vids)


def models():
    return {
        "logreg": Pipeline(
            [("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))]
        ),
        "svm": Pipeline(
            [("sc", StandardScaler()), ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True))]
        ),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=0
        ),
    }


def plot_cm(cm, labels, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default=str(ROOT / "data" / "tiny_subset"))
    p.add_argument("--labels", default=str(ROOT / "outputs" / "nod_pipeline" / "labels.csv"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "nod_pipeline"))
    p.add_argument("--model", default="rf", choices=["rf", "svm", "logreg", "all"])
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels)
    labels["label"] = labels["label"].astype(str).str.strip().str.lower()
    labels = labels[labels["label"].isin(["nod", "non-nod", "other-head-motion", "neutral"])]
    if labels.empty:
        raise SystemExit("No filled labels. Run 04 with --demo-fill or edit labels.csv")

    df = build_table(Path(args.subset), labels)
    df.to_csv(out / "window_features.csv", index=False)
    train, test, test_vids = video_split(df)
    print(f"Train videos={train.video_id.nunique()} windows={len(train)}")
    print(f"Test  videos={test_vids} windows={len(test)}")

    names = list(models()) if args.model == "all" else [args.model]
    summary = {}
    for name in names:
        clf = models()[name]
        clf.fit(train[FEATS], train["y"])
        pred = clf.predict(test[FEATS])
        p_, r_, f1, _ = precision_recall_fscore_support(test["y"], pred, average="binary", zero_division=0)
        report = classification_report(
            test["y"], pred, target_names=["non-nod", "nod"], zero_division=0
        )
        cm = confusion_matrix(test["y"], pred, labels=[0, 1])
        plot_cm(cm, ["non-nod", "nod"], out / f"cm_{name}.png", f"{name} (video split)")
        test_out = test.copy()
        test_out["pred"] = pred
        if hasattr(clf, "predict_proba"):
            test_out["prob_nod"] = clf.predict_proba(test[FEATS])[:, 1]
        test_out.to_csv(out / f"predictions_{name}.csv", index=False)
        summary[name] = {
            "precision": round(float(p_), 4),
            "recall": round(float(r_), 4),
            "f1": round(float(f1), 4),
            "n_test": int(len(test)),
            "test_videos": list(test_vids),
        }
        print(f"\n=== {name} ===")
        print(report)
        print("F1 (nod class):", round(float(f1), 3))
        # stash last model preds for demo
        if name == names[-1]:
            test_out.to_csv(out / "predictions.csv", index=False)

    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved", out / "metrics.json")


if __name__ == "__main__":
    main()
