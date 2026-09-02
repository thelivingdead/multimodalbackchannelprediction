#!/usr/bin/env python3
"""Evaluate nod-only baselines on the human 3 s sliding-window protocol.

The pitch axis is fixed by the nod task. A window-level amplitude threshold
is selected on DEV only and then applied once to TEST. This script does not
train a neural model and never writes the locked 60 s result directories.
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
DEFAULT_OUT = ROOT / "results" / "windowed_nod" / "baselines"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
WINDOW_FRAMES = 75


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


def select_dev_threshold(
    y_dev: np.ndarray, scores_dev: np.ndarray
) -> tuple[float, dict, pd.DataFrame]:
    """Select an amplitude threshold using DEV only."""
    y_dev = np.asarray(y_dev, dtype=int)
    scores_dev = np.asarray(scores_dev, dtype=float)
    values = np.unique(scores_dev)
    if not len(values):
        raise SystemExit("STOP: no DEV rule scores")
    thresholds = [float(np.nextafter(values[0], -np.inf))]
    thresholds.extend(float((a + b) / 2.0) for a, b in zip(values[:-1], values[1:]))
    thresholds.append(float(np.nextafter(values[-1], np.inf)))
    rows: list[dict] = []
    best = None
    for threshold in thresholds:
        metrics = clip_binary_metrics(y_dev, (scores_dev >= threshold).astype(int))
        row = {"threshold": threshold, **metrics}
        rows.append(row)
        key = (
            metrics["f1"],
            metrics["balanced_accuracy"],
            metrics["precision"],
            -threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    assert best is not None
    return float(best[1]), dict(best[2]), pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = assert_unlocked_out_dir(args.out_dir)
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
        y_dev, scores_dev
    )
    predictions = []
    split_metrics = {}
    for split, frame, labels, scores in (
        ("DEV", dev, y_dev, scores_dev),
        ("TEST", test, y_test, scores_test),
    ):
        frozen_pred = (scores >= frozen_threshold).astype(int)
        selected_pred = (scores >= selected_threshold).astype(int)
        split_metrics[split] = {
            "always_no": always_predict(labels, 0),
            "always_yes": always_predict(labels, 1),
            "frozen_60s_threshold_transfer": clip_binary_metrics(
                labels, frozen_pred
            ),
            "dev_selected_window_rule": clip_binary_metrics(labels, selected_pred),
        }
        part = frame[
            [
                "window_id",
                "sample_id",
                "split",
                "start_sec",
                "end_sec",
                "label",
            ]
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
            "frozen_60s_threshold": frozen_threshold,
            "dev_selected_window_threshold": selected_threshold,
            "selection": (
                "Pitch axis fixed for nod; amplitude threshold selected on "
                "human DEV window F1 only; TEST applied once."
            ),
            "n_dev": int(len(dev)),
            "n_dev_positive": int(y_dev.sum()),
            "n_test": int(len(test)),
            "n_test_positive": int(y_test.sum()),
            "dev_selected_metrics_check": selected_dev,
            "metrics": split_metrics,
        },
    )

    result = split_metrics["TEST"]["dev_selected_window_rule"]
    print("windowed nod baselines (3 s)")
    print(
        f"DEV {len(dev)} windows ({int(y_dev.sum())} positive); "
        f"TEST {len(test)} ({int(y_test.sum())} positive)"
    )
    print(
        f"DEV-selected pitch-rule threshold: {selected_threshold:.4f} degrees"
    )
    print(
        f"TEST pitch rule P {result['precision']:.3f} "
        f"R {result['recall']:.3f} F1 {result['f1']:.3f} "
        f"(TP{result['tp']} FP{result['fp']} "
        f"TN{result['tn']} FN{result['fn']})"
    )
    print(f"wrote {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
