from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "crossval_windowed_pose_cnn_dev.py"
    spec = importlib.util.spec_from_file_location("crossval_windowed_pose_cnn_dev", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_does_not_bind_test_windows() -> None:
    source = (ROOT / "scripts" / "crossval_windowed_pose_cnn_dev.py").read_text()
    assert "nod_windows_test" not in source
    assert "WINDOWS_TEST" not in source


def test_return_ratio_adds_constant_seventh_channel() -> None:
    module = load_module()
    t = np.linspace(0, 2 * np.pi, 75)
    nod = np.column_stack([6.0 * np.sin(t), np.zeros(75), np.zeros(75)])
    feat = module.channels_for_window(nod, return_ratio_channel=True)
    assert feat.shape == (75, 7)
    assert np.allclose(feat[:, 6], feat[0, 6])
    base = module.channels_for_window(nod, return_ratio_channel=False)
    assert base.shape == (75, 6)
    assert np.allclose(base, feat[:, :6])


def test_return_ratio_channel_is_lower_for_oscillation_than_ramp() -> None:
    module = load_module()
    t = np.linspace(0, 2 * np.pi, 75)
    nod = np.column_stack([6.0 * np.sin(t), np.zeros(75), np.zeros(75)])
    drift = np.column_stack([np.linspace(0, 8, 75), np.zeros(75), np.zeros(75)])
    nod_rr = module.channels_for_window(nod, return_ratio_channel=True)[0, 6]
    drift_rr = module.channels_for_window(drift, return_ratio_channel=True)[0, 6]
    assert nod_rr < drift_rr
    assert drift_rr > 0.9
