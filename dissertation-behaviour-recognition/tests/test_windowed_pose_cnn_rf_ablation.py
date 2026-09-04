from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "crossval_windowed_pose_cnn_rf_ablation_dev.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "crossval_windowed_pose_cnn_rf_ablation_dev", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_does_not_bind_test_paths() -> None:
    source = SCRIPT.read_text()
    wrapper = (ROOT / "scripts" / "run_pose_cnn_rf_ablation_otter.sh").read_text()
    for text in (source, wrapper):
        assert "nod_windows_test" not in text
        assert "WINDOWS_TEST" not in text
        assert "gold_016" not in text
        assert "gold_030" not in text
    assert "return_ratio_channel=False" in source
    assert "scalar_branch" in source
    assert "evaluate_windowed_final_fusion_search" not in source
    assert "evaluate_windowed_nod_return_ratio_test" not in source


def test_kernels_and_locked_dir_are_listed() -> None:
    module = load_module()
    kernels = [tuple(cfg["kernels"]) for cfg in module.CONFIGS]
    assert kernels == [(5, 5, 3), (11, 9, 7), (21, 15, 13)]
    assert module.WINDOW_FRAMES == 75
    assert module.FPS == 25.0
    assert module.OUT_DIR.name == "pose_cnn_loco_dev_rf_ablation"
    assert module.LOCKED_CNN_DIR.name == "pose_cnn_loco_dev"
    assert module.OUT_DIR.resolve() != module.LOCKED_CNN_DIR.resolve()
    source = SCRIPT.read_text()
    assert "will not overwrite the locked original CNN" in source
    assert str(module.LOCKED_CNN_DIR.as_posix()).endswith(
        "results/windowed_nod/pose_cnn_loco_dev"
    ) or "pose_cnn_loco_dev" in source


def test_script_does_not_import_default_kernels_from_pose_cnn() -> None:
    source = SCRIPT.read_text()
    assert "from src.pose_cnn" not in source
    assert "DEFAULT_KERNELS" in source
    module = load_module()
    assert module.DEFAULT_KERNELS == (5, 5, 3)
    assert [tuple(cfg["kernels"]) for cfg in module.CONFIGS] == [
        (5, 5, 3),
        (11, 9, 7),
        (21, 15, 13),
    ]


def test_receptive_field_formula_and_time_spans() -> None:
    module = load_module()
    assert module.DEFAULT_KERNELS == (5, 5, 3)
    assert module.conv_receptive_field((5, 5, 3)) == 11
    assert module.conv_receptive_field((11, 9, 7)) == 25
    assert module.conv_receptive_field((21, 15, 13)) == 47
    assert module.conv_paddings((5, 5, 3)) == (2, 2, 1)
    assert module.conv_paddings((11, 9, 7)) == (5, 4, 3)
    assert module.conv_paddings((21, 15, 13)) == (10, 7, 6)
    assert abs(module.receptive_field_seconds((5, 5, 3), fps=25.0) - 0.44) < 1e-12
    assert abs(module.receptive_field_seconds((11, 9, 7), fps=25.0) - 1.00) < 1e-12
    assert abs(module.receptive_field_seconds((21, 15, 13), fps=25.0) - 1.88) < 1e-12
    # 128-step / 3 s would make RF 47 ≈ 1.1 s. That is the old clip CNN, not this run.
    assert abs(47 * (3.0 / 128.0) - 1.1) < 0.01
    assert abs(module.receptive_field_seconds((21, 15, 13), fps=25.0) - 1.1) > 0.5


def test_local_build_cnn_takes_kernels() -> None:
    module = load_module()
    sig = inspect.signature(module._build_cnn)
    assert list(sig.parameters) == ["nn", "d", "kernels"]
    assert sig.parameters["kernels"].default is inspect.Parameter.empty


def test_refuses_locked_cnn_dir_and_test_ids() -> None:
    module = load_module()
    try:
        module.assert_ablation_out_dir(module.LOCKED_CNN_DIR)
    except SystemExit as exc:
        assert "locked" in str(exc).lower()
    else:
        raise AssertionError("locked CNN dir must be refused")
    try:
        module.assert_ablation_out_dir(module.LOCKED_CNN_DIR / "nested")
    except SystemExit as exc:
        assert "locked" in str(exc).lower()
    else:
        raise AssertionError("writes inside the locked CNN dir must be refused")
    module.refuse_test_ids(["gold_001", "gold_015"])
    try:
        module.refuse_test_ids(["gold_001", "gold_016"])
    except SystemExit as exc:
        assert "gold_016" in str(exc)
    else:
        raise AssertionError("gold_016 must be refused")
    try:
        module.refuse_test_ids(["gold_030"])
    except SystemExit as exc:
        assert "gold_030" in str(exc)
    else:
        raise AssertionError("gold_030 must be refused")


def test_decide_never_authorises_test() -> None:
    module = load_module()
    weak = module.decide(
        [
            {"name": "k5_5_3", "role": "baseline", "balanced_accuracy": 0.523},
            {"name": "k11_9_7", "role": "proposed", "balanced_accuracy": 0.540},
            {"name": "k21_15_13", "role": "larger", "balanced_accuracy": 0.545},
        ]
    )
    assert weak["clear_improvement"] is False
    assert weak["test_authorised"] is False
    strong = module.decide(
        [
            {"name": "k5_5_3", "role": "baseline", "balanced_accuracy": 0.523},
            {"name": "k11_9_7", "role": "proposed", "balanced_accuracy": 0.560},
            {"name": "k21_15_13", "role": "larger", "balanced_accuracy": 0.570},
        ]
    )
    assert strong["clear_improvement"] is True
    assert strong["test_authorised"] is False
    assert strong["best_larger_kernel"] == "k21_15_13"
