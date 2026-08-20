#!/usr/bin/env python3
"""Per-clip TEST error analysis from the saved prediction tables.

Joins results/rule_test_predictions.csv and results/classifier_test_predictions.csv
(15 TEST clips) into results/error_analysis.csv with per-clip error categories:
correct, shared_fp, shared_fn, rule_only_fp/fn, cnn_only_fp/fn. Reads saved
artifacts only; no model is re-run and no label is changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RES = ROOT / "results"


def category(label: int, rule_pred: int, cnn_pred: int) -> str:
    r_err = rule_pred != label
    c_err = cnn_pred != label
    if not r_err and not c_err:
        return "correct"
    kind = "fp" if label == 0 else "fn"
    if r_err and c_err:
        return f"shared_{kind}"
    return f"rule_only_{kind}" if r_err else f"cnn_only_{kind}"


def main() -> None:
    rule_p = RES / "rule_test_predictions.csv"
    cnn_p = RES / "classifier_test_predictions.csv"
    for p in (rule_p, cnn_p):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run scripts/run_full_experiment.py first.")
    rule = pd.read_csv(rule_p)
    cnn = pd.read_csv(cnn_p)
    df = rule.merge(cnn, on="sample_id", suffixes=("_rule", "_cnn"))
    if (df["label_rule"] != df["label_cnn"]).any():
        raise SystemExit("Label mismatch between rule and CNN prediction tables.")
    df["label"] = df["label_rule"]
    df["error"] = [
        category(int(l), int(r), int(c))
        for l, r, c in zip(df["label"], df["pred_rule"], df["pred_cnn"])
    ]
    out = df[["sample_id", "label", "score", "pred_rule", "prob", "pred_cnn", "error"]].rename(
        columns={"score": "rule_score", "prob": "cnn_probability"}
    )
    dest = RES / "error_analysis.csv"
    out.to_csv(dest, index=False)
    counts = out["error"].value_counts().to_dict()
    print(f"Wrote {dest} ({len(out)} TEST clips)")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")
    both_right = int(counts.get("correct", 0))
    print(f"  both correct: {both_right}/{len(out)}")


if __name__ == "__main__":
    main()
