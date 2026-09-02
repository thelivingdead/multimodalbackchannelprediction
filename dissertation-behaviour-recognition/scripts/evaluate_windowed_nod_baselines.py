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
import importlib.util
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
from src.pose_cnn import load_npz  # noqa: E402
from src.utils import dump_json  # noqa: E402

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
WINDOW_FRAMES = 75
HEADLINE = "balanced_accuracy"
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


def _rule_score_function():
    path = ROOT / "scripts" / "run_full_experiment.py"
    spec = importlib.util.spec_from_file_location("run_full_experiment", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise SystemExit(f"STOP: cannot load {path}")
    spec.loader.exec_module(module)
    return module.rule_score


def load_windows(path: Path, split: str, allowed: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {
        "window_id",
        "sample_id",
        "split",
        "start_frame_relative",
        "end_frame_relative",
        "label",
    }
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"STOP: {path.name} missing {sorted(missing)}")
    df["sample_id"] = df["sample_id"].astype(str)
    df["split"] = df["split"].astype(str).str.upper()
    if (df["split"] != split).any():
        raise SystemExit(f"STOP: {path.name} contains a non-{split} row")
    ids = set(df["sample_id"])
    if ids != allowed:
        raise SystemExit(
            f"STOP: {path.name} {split} ids differ: "
            f"extra={sorted(ids - allowed)}, missing={sorted(allowed - ids)}"
        )
    counts = df.groupby("sample_id").size()
    if len(df) != 15 * 29 or not (counts == 29).all():
        raise SystemExit(f"STOP: {path.name} is not 15 clips x 29 windows")
    return df.reset_index(drop=True)


def score_windows(df: pd.DataFrame, axis: int, rule_score) -> np.ndarray:
    cache: dict[str, dict] = {}
    scores: list[float] = []
    for row in df.itertuples(index=False):
        sid = str(row.sample_id)
        if sid not in cache:
            pose_path = GOLD_DIR / f"{sid}.npz"
            if not pose_path.exists():
                raise SystemExit(f"STOP: missing pose file {pose_path}")
            cache[sid] = load_npz(pose_path)
        start = int(row.start_frame_relative)
        end = int(row.end_frame_relative)
        rotation = np.asarray(cache[sid]["rotation_xyz"], dtype=np.float32)
        chunk = rotation[start:end]
        if chunk.shape != (WINDOW_FRAMES, 3):
            raise SystemExit(
                f"STOP: {sid} window {start}:{end} has shape {chunk.shape}"
            )
        scores.append(float(rule_score(chunk, axis)))
    return np.asarray(scores, dtype=float)


def candidate_thresholds(scores: np.ndarray) -> list[float]:
    values = np.unique(np.asarray(scores, dtype=float))
    if not len(values):
        raise SystemExit("STOP: no rule scores to sweep")
    thresholds = [float(np.nextafter(values[0], -np.inf))]
    thresholds.extend(float((a + b) / 2.0) for a, b in zip(values[:-1], values[1:]))
    thresholds.append(float(np.nextafter(values[-1], np.inf)))
    return thresholds


def _selection_key(metrics: dict, threshold: float, criterion: str) -> tuple:
    if criterion == "balanced_accuracy":
        return (
            metrics["balanced_accuracy"],
            metrics["precision"],
            metrics["f1"],
            -threshold,
        )
    return (
        metrics["f1"],
        metrics["balanced_accuracy"],
        metrics["precision"],
        -threshold,
    )


def select_dev_threshold(
    y_dev: np.ndarray,
    scores_dev: np.ndarray,
    criterion: str = HEADLINE,
) -> tuple[float, dict, pd.DataFrame]:
    """Select an amplitude threshold using DEV only.

    ``criterion`` is fixed before TEST is touched. Balanced accuracy is the
    default because F1 is not a safe selection criterion at this prevalence.
    """
    if criterion not in ("balanced_accuracy", "f1"):
        raise SystemExit(f"STOP: unknown selection criterion {criterion!r}")
    y_dev = np.asarray(y_dev, dtype=int)
    scores_dev = np.asarray(scores_dev, dtype=float)
    rows: list[dict] = []
    best = None
    for threshold in candidate_thresholds(scores_dev):
        metrics = clip_binary_metrics(y_dev, (scores_dev >= threshold).astype(int))
        rows.append({"threshold": threshold, **metrics})
        key = _selection_key(metrics, threshold, criterion)
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    assert best is not None
    return float(best[1]), dict(best[2]), pd.DataFrame(rows)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """PR AUC by the step-wise definition. Threshold-free ranking quality."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = y_true[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(1 - hits)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    previous = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - previous) * precision))


def clip_bootstrap(
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Percentile intervals from resampling whole clips with replacement."""
    sample_ids = np.asarray(sample_ids)
    clips = np.unique(sample_ids)
    index_by_clip = {clip: np.flatnonzero(sample_ids == clip) for clip in clips}
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {
        "balanced_accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }
    n_degenerate = 0
    for _ in range(n_resamples):
        drawn = rng.choice(clips, size=len(clips), replace=True)
        idx = np.concatenate([index_by_clip[clip] for clip in drawn])
        labels = y_true[idx]
        if labels.min() == labels.max():
            n_degenerate += 1
            continue
        metrics = clip_binary_metrics(labels, y_pred[idx])
        for name in collected:
            collected[name].append(float(metrics[name]))
    out: dict = {
        "n_resamples": int(n_resamples),
        "n_clips": int(len(clips)),
        "n_skipped_single_class_resamples": int(n_degenerate),
        "resampling_unit": "clip",
    }
    for name, values in collected.items():
        array = np.asarray(values, dtype=float)
        out[name] = {
            "mean": float(array.mean()) if array.size else float("nan"),
            "ci_lower_95": float(np.percentile(array, 2.5)) if array.size else float("nan"),
            "ci_upper_95": float(np.percentile(array, 97.5)) if array.size else float("nan"),
        }
    return out


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
    rule_score = _rule_score_function()
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
