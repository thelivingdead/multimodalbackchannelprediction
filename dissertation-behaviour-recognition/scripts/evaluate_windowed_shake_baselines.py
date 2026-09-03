#!/usr/bin/env python3
"""Evaluate shake-only baselines on the human 3 s sliding-window protocol.

Unlike the nod baselines, which inherited the pitch axis from the frozen 60 s
configuration, no frozen shake axis exists. The repository deliberately does
not assume that EMOCA rotation channel 0 is anatomical pitch (see the note in
``results/rule_selected_config.json``), so the axis is selected here on shake
DEV together with the amplitude threshold, and the full three-axis DEV table
is reported. That is one extra disclosed DEV decision relative to nod.

Selection criterion is balanced accuracy, fixed before TEST was scored, for
the reason set out in ``src/windowed_baselines``. Intervals resample clips.
"""
from __future__ import annotations

import argparse
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
    AXIS_NAMES,
    CRITERIA,
    HEADLINE,
    N_BOOTSTRAP,
    average_precision,
    clip_bootstrap,
    load_windows,
    rule_score_function,
    score_windows,
    select_dev_threshold,
)

WINDOWS_DEV = ROOT / "data" / "windowed_annotations" / "shake_windows_dev.csv"
WINDOWS_TEST = ROOT / "data" / "windowed_annotations" / "shake_windows_test.csv"
GOLD_DIR = ROOT / "features" / "gold"
DEFAULT_OUT = ROOT / "results" / "windowed_shake" / "baselines_bacc"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
AXES = (0, 1, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--criterion", choices=CRITERIA, default=HEADLINE)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    out_dir = assert_unlocked_out_dir(args.out_dir)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        raise SystemExit(
            f"STOP: {metrics_path} exists. Windowed shake TEST was already scored."
        )

    dev = load_windows(WINDOWS_DEV, "DEV", DEV_IDS)
    test = load_windows(WINDOWS_TEST, "TEST", TEST_IDS)
    if set(dev["sample_id"]) & set(test["sample_id"]):
        raise SystemExit("STOP: DEV/TEST sample overlap")

    rule_score = rule_score_function()
    y_dev = dev["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)

    # ---- axis and threshold selected together, on shake DEV only ----
    dev_cache: dict = {}
    axis_rows: list[dict] = []
    per_axis: dict[int, dict] = {}
    for axis in AXES:
        scores = score_windows(dev, axis, rule_score, GOLD_DIR, dev_cache)
        threshold, metrics, search = select_dev_threshold(
            y_dev, scores, args.criterion
        )
        per_axis[axis] = {
            "scores": scores,
            "threshold": threshold,
            "metrics": metrics,
            "search": search,
        }
        axis_rows.append(
            {
                "axis": axis,
                "axis_name": AXIS_NAMES[axis],
                "dev_threshold": threshold,
                "dev_balanced_accuracy": metrics["balanced_accuracy"],
                "dev_precision": metrics["precision"],
                "dev_recall": metrics["recall"],
                "dev_f1": metrics["f1"],
                "dev_pr_auc": average_precision(y_dev, scores),
            }
        )

    axis_table = pd.DataFrame(axis_rows)
    chosen_axis = int(
        axis_table.sort_values(
            ["dev_balanced_accuracy", "dev_precision"], ascending=False
        ).iloc[0]["axis"]
    )
    chosen = per_axis[chosen_axis]
    selected_threshold = float(chosen["threshold"])
    scores_dev = chosen["scores"]
    scores_test = score_windows(test, chosen_axis, rule_score, GOLD_DIR)

    other = "f1" if args.criterion == "balanced_accuracy" else "balanced_accuracy"
    other_threshold, other_dev, _ = select_dev_threshold(y_dev, scores_dev, other)

    predictions = []
    split_metrics = {}
    bootstrap = {}
    for split, frame, labels, scores in (
        ("DEV", dev, y_dev, scores_dev),
        ("TEST", test, y_test, scores_test),
    ):
        selected_pred = (scores >= selected_threshold).astype(int)
        split_metrics[split] = {
            "always_no": always_predict(labels, 0),
            "always_yes": always_predict(labels, 1),
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
        part["dev_selected_pred"] = selected_pred
        predictions.append(part)

    out_dir.mkdir(parents=True, exist_ok=True)
    axis_table.to_csv(out_dir / "axis_selection_dev.csv", index=False)
    chosen["search"].to_csv(out_dir / "threshold_search_dev.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        out_dir / "predictions.csv", index=False
    )
    dump_json(
        metrics_path,
        {
            "protocol": "windowed_shake_3s_baselines",
            "task": "shake_only",
            "window_sec": 3.0,
            "stride_sec": 2.0,
            "headline_metric": HEADLINE,
            "headline_metric_floor": 0.5,
            "selection_criterion": args.criterion,
            "axis": chosen_axis,
            "axis_name": AXIS_NAMES[chosen_axis],
            "axis_selection": (
                "Axis and amplitude threshold selected together on shake DEV. "
                "No frozen 60 s shake axis exists and the repository does not "
                "assume EMOCA channel 0 is anatomical pitch, so the axis is a "
                "disclosed DEV decision over three candidates rather than an "
                "anatomical assumption. Nod, by contrast, inherited its axis "
                "from the frozen 60 s configuration."
            ),
            "criterion_rationale": (
                "F1 is unsuitable as a selection criterion at roughly 9 percent "
                "window prevalence: it barely penalises false positives, so an F1 "
                "sweep drifts towards always-yes. Balanced accuracy weights the "
                "negative class equally. Criterion fixed before TEST was scored."
            ),
            "dev_selected_window_threshold": selected_threshold,
            "axis_table_dev": axis_rows,
            "criterion_comparison": {
                "selected_criterion": args.criterion,
                "selected_threshold": selected_threshold,
                "alternative_criterion": other,
                "alternative_threshold": other_threshold,
                "thresholds_agree": bool(
                    np.isclose(selected_threshold, other_threshold)
                ),
                "alternative_dev_metrics": other_dev,
            },
            "n_dev": int(len(dev)),
            "n_dev_positive": int(y_dev.sum()),
            "n_test": int(len(test)),
            "n_test_positive": int(y_test.sum()),
            "metrics": split_metrics,
            "clip_bootstrap": bootstrap,
            "bootstrap_note": (
                "Percentile intervals from resampling the 15 clips of a split with "
                "replacement. Windows within a clip overlap by 1 s and are not "
                "independent, so window-level resampling would understate "
                "uncertainty."
            ),
        },
    )

    print("windowed shake baselines (3 s)")
    print(
        f"DEV {len(dev)} windows ({int(y_dev.sum())} positive); "
        f"TEST {len(test)} ({int(y_test.sum())} positive)"
    )
    print(f"selection criterion: {args.criterion} (fixed before TEST)")
    print("\nDEV axis sweep (selection):")
    print(
        axis_table[
            [
                "axis",
                "axis_name",
                "dev_threshold",
                "dev_balanced_accuracy",
                "dev_precision",
                "dev_recall",
                "dev_f1",
                "dev_pr_auc",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(
        f"\nchosen axis {chosen_axis} ({AXIS_NAMES[chosen_axis]}) "
        f"at {selected_threshold:.4f} degrees "
        f"(F1 criterion would pick {other_threshold:.4f})"
    )
    for split in ("DEV", "TEST"):
        result = split_metrics[split]["dev_selected_window_rule"]
        interval = bootstrap[split]["dev_selected_window_rule"]["balanced_accuracy"]
        print(
            f"{split:4} balanced accuracy {result['balanced_accuracy']:.3f} "
            f"[{interval['ci_lower_95']:.3f}, {interval['ci_upper_95']:.3f}]  "
            f"P {result['precision']:.3f} R {result['recall']:.3f} "
            f"F1 {result['f1']:.3f}  "
            f"(TP{result['tp']} FP{result['fp']} TN{result['tn']} FN{result['fn']})"
        )
    print(
        f"always-yes F1: DEV {split_metrics['DEV']['always_yes']['f1']:.3f}  "
        f"TEST {split_metrics['TEST']['always_yes']['f1']:.3f}  "
        "(balanced accuracy 0.500 by construction)"
    )
    print(
        f"PR AUC: DEV {split_metrics['DEV']['pr_auc_rule_score']:.3f}  "
        f"TEST {split_metrics['TEST']['pr_auc_rule_score']:.3f}"
    )
    print(f"wrote {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
