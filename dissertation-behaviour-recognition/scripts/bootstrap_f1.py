#!/usr/bin/env python3
"""VideoMAE Step 6a: 1000-resample 95% bootstrap CIs for TEST F1.

Every number comes from a **saved predictions CSV** — nothing is recomputed
from models and nothing is invented. For each predictions file that exists
(among the registry below), the observed TEST F1 plus a percentile bootstrap
CI (1000 resamples with replacement, seed 42) is written to
``results/tables/bootstrap_ci.csv``. Missing files yield no row (the results
table prints N/A there).

Registry (model key -> predictions CSV, columns label/pred)::

    rule_baseline        results/rule_test_predictions.csv
    pose_cnn_xyz_deriv   results/classifier_test_predictions.csv
    videomae_frozen_head results/videomae_frozen_head/predictions.csv

Extra files can be added with ``--pred key=path/to.csv``. Lab invocation::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    source ../.venv/bin/activate
    python scripts/bootstrap_f1.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import binary_metrics  # noqa: E402

OUT_CSV = ROOT / "results" / "tables" / "bootstrap_ci.csv"
RESAMPLES = 1000
SEED = 42

REGISTRY = {
    "rule_baseline": ROOT / "results" / "rule_test_predictions.csv",
    "pose_cnn_xyz_deriv": ROOT / "results" / "classifier_test_predictions.csv",
    "videomae_frozen_head": ROOT / "results" / "videomae_frozen_head"
    / "predictions.csv",
}


def bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray,
                 resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    observed = binary_metrics(y_true, y_pred)["f1"]
    # Vectorised resamples; per-draw F1 is 2tp/(2tp+fp+fn) with 0 on a zero
    # denominator — identical to binary_metrics' sklearn f1(zero_division=0).
    idx = rng.integers(0, n, (resamples, n))
    yt = y_true[idx]
    yp = y_pred[idx]
    tp = ((yt == 1) & (yp == 1)).sum(axis=1).astype(float)
    fp = ((yt == 0) & (yp == 1)).sum(axis=1).astype(float)
    fn = ((yt == 1) & (yp == 0)).sum(axis=1).astype(float)
    denom = 2 * tp + fp + fn
    draws = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "n": int(n),
        "resamples": int(resamples),
        "seed": int(seed),
        "f1": float(observed),
        "f1_ci_lo": float(lo),
        "f1_ci_hi": float(hi),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred", action="append", default=[],
                        help="extra key=path/to/predictions.csv entries")
    args = parser.parse_args()

    registry = dict(REGISTRY)
    for item in args.pred:
        if "=" not in item:
            raise SystemExit(f"STOP: --pred {item!r} must be key=path")
        key, path = item.split("=", 1)
        registry[key] = Path(path)

    rows = []
    skipped = []
    for key, path in registry.items():
        if not path.exists():
            skipped.append(f"{key} ({path.relative_to(ROOT) if path.is_absolute() and str(path).startswith(str(ROOT)) else path}: not run)")
            continue
        df = pd.read_csv(path)
        if "label" not in df.columns or "pred" not in df.columns:
            raise SystemExit(
                f"STOP: {path} needs columns label,pred; found "
                f"{list(df.columns)}"
            )
        stats = bootstrap_f1(
            df["label"].to_numpy().astype(int),
            df["pred"].to_numpy().astype(int),
            RESAMPLES,
            SEED,
        )
        rows.append({
            "model": key,
            "predictions_path": str(path.relative_to(ROOT))
            if str(path).startswith(str(ROOT)) else str(path),
            **stats,
        })
        print(f"{key}: F1={stats['f1']:.3f} "
              f"95% CI [{stats['f1_ci_lo']:.3f}, {stats['f1_ci_hi']:.3f}] "
              f"(n={stats['n']}, {RESAMPLES} resamples)")

    if skipped:
        print("no saved predictions yet -> no CI row for: "
              + "; ".join(skipped))
    if not rows:
        raise SystemExit(
            "STOP: no predictions CSVs found at all; nothing to bootstrap."
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
