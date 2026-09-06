from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_audio_3s_does_not_bind_test() -> None:
    train = (ROOT / "scripts" / "run_windowed_audio_3s_dev.py").read_text()
    plot = (ROOT / "scripts" / "plot_windowed_audio_3s_dev.py").read_text()
    sh = (ROOT / "scripts" / "run_windowed_audio_3s_otter.sh").read_text()
    assert "nod_windows_test" not in train
    assert "nod_windows_test" not in plot
    assert "TEST will not be loaded" in sh
    assert "FEATURE_DIM" in train
    assert "windowed_dev" in train and "audio_3s" in train
    assert "evaluate_windowed_late_fusion_logreg_dev" not in train


def test_audio_3s_reuses_existing_mfcc() -> None:
    train = (ROOT / "scripts" / "run_windowed_audio_3s_dev.py").read_text()
    assert "extract_audio_features" in train
    assert "30-D" in train
    assert "TARGET_SR" in train
