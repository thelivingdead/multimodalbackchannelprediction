from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hubert_3s_does_not_bind_test() -> None:
    train = (ROOT / "scripts" / "run_windowed_hubert_3s_dev.py").read_text()
    plot = (ROOT / "scripts" / "plot_windowed_hubert_3s_dev.py").read_text()
    sh = (ROOT / "scripts" / "run_windowed_hubert_3s_otter.sh").read_text()
    assert "nod_windows_test" not in train
    assert "nod_windows_test" not in plot
    assert "TEST will not be loaded" in sh
    assert "facebook/hubert-base-ls960" in train
    assert "requires_grad = False" in train
    assert "fine_tune" in train
    assert "windowed_dev" in train and "audio_3s_hubert" in train
    assert "scale_train_only" in (ROOT / "scripts" / "run_windowed_audio_3s_dev.py").read_text()
    assert "train_loco" in train


def test_hubert_3s_is_frozen_mean_pool() -> None:
    train = (ROOT / "scripts" / "run_windowed_hubert_3s_dev.py").read_text()
    assert "HubertModel.from_pretrained" in train
    assert "mean-pool" in train or "mean-pooled" in train
    assert "Adam" not in train
    assert "train_loco" in train
