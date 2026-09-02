from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "evaluate_windowed_nod_baselines.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_windowed_nod_baselines", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_mil_module():
    path = ROOT / "scripts" / "train_windowed_nod_pose_mil_dev.py"
    spec = importlib.util.spec_from_file_location(
        "train_windowed_nod_pose_mil_dev", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_threshold_selection_uses_labels_and_scores_only() -> None:
    module = load_module()
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    threshold, metrics, search = module.select_dev_threshold(labels, scores)
    assert 0.2 < threshold < 0.8
    assert metrics["f1"] == 1.0
    assert len(search) == len(np.unique(scores)) + 1


def test_human_window_splits_are_complete_and_disjoint() -> None:
    module = load_module()
    dev = module.load_windows(module.WINDOWS_DEV, "DEV", module.DEV_IDS)
    test = module.load_windows(module.WINDOWS_TEST, "TEST", module.TEST_IDS)
    assert len(dev) == 15 * 29
    assert len(test) == 15 * 29
    assert set(dev["sample_id"]).isdisjoint(set(test["sample_id"]))


def test_mil_feature_windows_have_fixed_protocol_shape() -> None:
    module = load_mil_module()
    rotation = np.zeros((1500, 3), dtype=np.float32)
    windows = module.feature_windows(rotation)
    assert windows.shape == (29, 75, 6)
    assert np.isfinite(windows).all()


def test_mil_train_bags_use_all_80_pseudo_clips() -> None:
    module = load_mil_module()
    bags, labels, ids = module.load_train_bags()
    assert bags.shape == (80, 29, 75, 6)
    assert labels.shape == (80,)
    assert len(ids) == 80
    assert set(ids).isdisjoint(module.DEV_IDS)
    assert int((labels == 1).sum()) == 70
    assert int((labels == 0).sum()) == 10


def test_mil_development_script_has_no_test_window_input() -> None:
    source = (
        ROOT / "scripts" / "train_windowed_nod_pose_mil_dev.py"
    ).read_text()
    assert "nod_windows_test.csv" not in source
    assert '"test_scored": False' in source
