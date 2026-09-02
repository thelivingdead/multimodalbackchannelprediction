from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

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
    assert metrics["balanced_accuracy"] == 1.0
    assert len(search) == len(np.unique(scores)) + 1


def test_default_criterion_is_balanced_accuracy() -> None:
    module = load_module()
    assert module.HEADLINE == "balanced_accuracy"


def test_each_criterion_maximises_its_own_objective_over_the_sweep() -> None:
    module = load_module()
    rng = np.random.default_rng(0)
    labels = np.asarray([0] * 44 + [1] * 6)
    scores = rng.normal(size=labels.shape) + 0.3 * labels
    for criterion in ("balanced_accuracy", "f1"):
        threshold, metrics, search = module.select_dev_threshold(
            labels, scores, criterion
        )
        assert metrics[criterion] == pytest.approx(search[criterion].max())
        assert threshold in set(search["threshold"])


def test_always_yes_balanced_accuracy_is_exactly_the_floor() -> None:
    module = load_module()
    labels = np.asarray([0] * 44 + [1] * 6)
    always_yes = module.always_predict(labels, 1)
    always_no = module.always_predict(labels, 0)
    assert always_yes["balanced_accuracy"] == pytest.approx(0.5)
    assert always_no["balanced_accuracy"] == pytest.approx(0.5)
    assert always_yes["f1"] > always_no["f1"]


def test_average_precision_matches_hand_computed_value() -> None:
    module = load_module()
    labels = np.asarray([1, 0, 1, 0])
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    # Hits at ranks 1 and 3: (1/1 + 2/3) / 2
    assert module.average_precision(labels, scores) == pytest.approx(
        (1.0 + 2.0 / 3.0) / 2.0
    )
    assert np.isnan(module.average_precision(np.zeros(4, dtype=int), scores))


def test_clip_bootstrap_resamples_clips_not_windows() -> None:
    module = load_module()
    sample_ids = np.repeat([f"gold_{i:03d}" for i in range(1, 6)], 4)
    labels = np.tile([0, 0, 1, 1], 5)
    perfect = labels.copy()
    out = module.clip_bootstrap(sample_ids, labels, perfect, n_resamples=50, seed=0)
    assert out["n_clips"] == 5
    assert out["resampling_unit"] == "clip"
    assert out["balanced_accuracy"]["ci_lower_95"] == pytest.approx(1.0)
    assert out["balanced_accuracy"]["ci_upper_95"] == pytest.approx(1.0)

    noisy = np.tile([0, 1, 1, 0], 5)
    spread = module.clip_bootstrap(sample_ids, labels, noisy, n_resamples=200, seed=0)
    assert 0.0 <= spread["balanced_accuracy"]["ci_lower_95"] <= 1.0
    assert spread["balanced_accuracy"]["mean"] == pytest.approx(0.5)


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


def test_mil_selects_on_balanced_accuracy_by_default() -> None:
    source = (
        ROOT / "scripts" / "train_windowed_nod_pose_mil_dev.py"
    ).read_text()
    assert 'default="balanced_accuracy"' in source
    assert "criterion=args.criterion" in source
