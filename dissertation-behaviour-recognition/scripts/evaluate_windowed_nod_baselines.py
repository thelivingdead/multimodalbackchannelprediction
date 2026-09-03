#!/usr/bin/env python3
"""Evaluate nod-only baselines on the human 3 s sliding-window protocol.

The pitch axis is fixed by the nod task. A window-level amplitude threshold
is selected on DEV only and then applied once to TEST. This script does not
train a neural model and never writes the locked 60 s result directories.

Selection criterion. At roughly 12-16 percent window prevalence, F1 barely
penalises false positives, so an F1 sweep walks the threshold down until the
rule is close to always-yes. Balanced accuracy is therefore the selection
criterion and the headline metric. DEV PR AUC is reported as a threshold-free
check on how well the score ranks nod windows at all.

Confidence intervals resample whole clips, not windows: 435 windows drawn
from 15 clips are not 435 independent observations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_split_leakage import assert_unlocked_out_dir  # noqa: E402
from src.clip_metrics import always_predict, clip_binary_metrics  # noqa: E402
from src.utils import dump_json  # noqa: E402
from src.windowed_baselines import (  # noqa: E402
    HEADLINE,
    N_BOOTSTRAP,
    WINDOW_FRAMES,
    average_precision,
    candidate_thresholds,
    clip_bootstrap,
    load_windows,
    rule_score_function,
    select_dev_threshold,
)
from src.windowed_baselines import score_windows as _score_windows  # noqa: E402

WINDOWS_DEV = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
WINDOWS_TEST = ROOT / "data" / "windowed_annotations" / "nod_windows_test.csv"
GOLD_DIR = ROOT / "features" / "gold"
RULE_CONFIG = ROOT / "results" / "rule_selected_config.json"
OUT_BY_CRITERION = {
    "balanced_accuracy": ROOT / "results" / "windowed_nod" / "baselines_bacc",
    "f1": ROOT / "results" / "windowed_nod" / "baselines",
}
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}


def score_windows(df: pd.DataFrame, axis: int, rule_score) -> np.ndarray:
    return _score_windows(df, axis, rule_score, GOLD_DIR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--criterion",
        choices=("balanced_accuracy", "f1"),
        default=HEADLINE,
        help="DEV selection criterion, fixed before TEST is scored",
    )
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    out_dir = assert_unlocked_out_dir(
        args.out_dir if args.out_dir is not None else OUT_BY_CRITERION[args.criterion]
    )
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        raise SystemExit(
            f"STOP: {metrics_path} exists. Windowed nod TEST baseline was already scored."
        )

    dev = load_windows(WINDOWS_DEV, "DEV", DEV_IDS)
    test = load_windows(WINDOWS_TEST, "TEST", TEST_IDS)
    if set(dev["sample_id"]) & set(test["sample_id"]):
        raise SystemExit("STOP: DEV/TEST sample overlap")

    config = json.loads(RULE_CONFIG.read_text())
    axis = int(config["chosen_rotation_axis"])
    if axis != 0:
        raise SystemExit(
            f"STOP: nod baseline expected pitch axis x (0), config has axis {axis}"
        )
    frozen_threshold = float(config["selected_amplitude_threshold"])
    rule_score = rule_score_function()
    scores_dev = score_windows(dev, axis, rule_score)
    scores_test = score_windows(test, axis, rule_score)
    y_dev = dev["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)

    selected_threshold, selected_dev, search = select_dev_threshold(
        y_dev, scores_dev, args.criterion
    )
    other = "f1" if args.criterion == "balanced_accuracy" else "balanced_accuracy"
    other_threshold, other_dev, _ = select_dev_threshold(y_dev, scores_dev, other)

    predictions = []
    split_metrics = {}
    bootstrap = {}
    for split, frame, labels, scores in (
        ("DEV", dev, y_dev, scores_dev),
        ("TEST", test, y_test, scores_test),
    ):
        frozen_pred = (scores >= frozen_threshold).astype(int)
        selected_pred = (scores >= selected_threshold).astype(int)
        split_metrics[split] = {
            "always_no": always_predict(labels, 0),
            "always_yes": always_predict(labels, 1),
            "frozen_60s_threshold_transfer": clip_binary_metrics(labels, frozen_pred),
            "dev_selected_window_rule": clip_binary_metrics(labels, selected_pred),
            "pr_auc_rule_score": average_precision(labels, scores),
        }
        bootstrap[split] = {
            "dev_selected_window_rule": clip_bootstrap(
                frame["sample_id"].to_numpy(),
                labels,
                selected_pred,
                n_resamples=args.bootstrap,
            ),
            "always_yes_balanced_accuracy": 0.5,
        }
        part = frame[
            ["window_id", "sample_id", "split", "start_sec", "end_sec", "label"]
        ].copy()
        part["rule_score"] = scores
        part["frozen_threshold_pred"] = frozen_pred
        part["dev_selected_pred"] = selected_pred
        predictions.append(part)

    out_dir.mkdir(parents=True, exist_ok=True)
    search.to_csv(out_dir / "threshold_search_dev.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        out_dir / "predictions.csv", index=False
    )
    dump_json(
        metrics_path,
        {
            "protocol": "windowed_nod_3s_baselines",
            "task": "nod_only",
            "window_sec": 3.0,
            "stride_sec": 2.0,
            "axis": axis,
            "axis_name": "x",
            "axis_meaning": "pitch (up-down / nod-like)",
            "headline_metric": HEADLINE,
            "headline_metric_floor": 0.5,
            "selection_criterion": args.criterion,
            "selection": (
                "Pitch axis fixed for nod. Amplitude threshold selected on human "
                "DEV using balanced accuracy, fixed before TEST was scored, then "
                "applied once to TEST."
            ),
            "criterion_rationale": (
                "F1 is unsuitable as a selection criterion at this window "
                "prevalence: it barely penalises false positives, so an F1 sweep "
                "drifts towards always-yes. Balanced accuracy weights the negative "
                "class equally and exposes that collapse."
            ),
            "frozen_60s_threshold": frozen_threshold,
            "dev_selected_window_threshold": selected_threshold,
            "criterion_comparison": {
                "selected_criterion": args.criterion,
                "selected_threshold": selected_threshold,
                "alternative_criterion": other,
                "alternative_threshold": other_threshold,
                "thresholds_agree": bool(
                    np.isclose(selected_threshold, other_threshold)
                ),
                "alternative_dev_metrics": other_dev,
                "pr_auc_note": (
                    "PR AUC is threshold-free, so it checks whether the amplitude "
                    "score ranks nod windows above non-nod windows at all; it does "
                    "not itself pick a cut."
                ),
            },
            "n_dev": int(len(dev)),
            "n_dev_positive": int(y_dev.sum()),
            "n_test": int(len(test)),
            "n_test_positive": int(y_test.sum()),
            "dev_selected_metrics_check": selected_dev,
            "metrics": split_metrics,
            "clip_bootstrap": bootstrap,
            "bootstrap_note": (
                "Percentile intervals from resampling the 15 clips of a split with "
                "replacement. Window-level resampling would understate uncertainty "
                "because windows within a clip are dependent and overlap by 1 s."
            ),
        },
    )

    dev_sel = split_metrics["DEV"]["dev_selected_window_rule"]
    test_sel = split_metrics["TEST"]["dev_selected_window_rule"]
    test_ci = bootstrap["TEST"]["dev_selected_window_rule"]["balanced_accuracy"]
    print("windowed nod baselines (3 s)")
    print(
        f"DEV {len(dev)} windows ({int(y_dev.sum())} positive); "
        f"TEST {len(test)} ({int(y_test.sum())} positive)"
    )
    print(f"selection criterion: {args.criterion} (fixed before TEST)")
    print(
        f"DEV-selected threshold: {selected_threshold:.4f} degrees "
        f"(F1 criterion would pick {other_threshold:.4f})"
    )
    print(
        f"DEV  balanced accuracy {dev_sel['balanced_accuracy']:.3f}  "
        f"P {dev_sel['precision']:.3f}  R {dev_sel['recall']:.3f}  "
        f"F1 {dev_sel['f1']:.3f}"
    )
    print(
        f"TEST balanced accuracy {test_sel['balanced_accuracy']:.3f} "
        f"[{test_ci['ci_lower_95']:.3f}, {test_ci['ci_upper_95']:.3f}]  "
        f"P {test_sel['precision']:.3f}  R {test_sel['recall']:.3f}  "
        f"F1 {test_sel['f1']:.3f}"
    )
    print(
        f"TEST counts TP{test_sel['tp']} FP{test_sel['fp']} "
        f"TN{test_sel['tn']} FN{test_sel['fn']}; "
        f"always-yes balanced accuracy 0.500"
    )
    print(
        f"DEV PR AUC {split_metrics['DEV']['pr_auc_rule_score']:.3f}  "
        f"TEST PR AUC {split_metrics['TEST']['pr_auc_rule_score']:.3f}"
    )
    print(f"wrote {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
