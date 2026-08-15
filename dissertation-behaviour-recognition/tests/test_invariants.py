"""IoU, matching, labels, leakage."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotations import parse_label
from src.events import Event, greedy_match, iou, match_pairs
from src.plotting import FigureLog, save_publication_figure


def test_match_pairs_and_greedy() -> None:
    a = Event("v", 1.0, 2.0)
    b = Event("v", 1.2, 2.2)
    extra = Event("v", 8.0, 8.4)
    assert iou(a, b) > 0.5
    tps, fps, fns = match_pairs([a, extra], [b], 0.3)
    assert len(tps) == 1
    assert fps == [extra]
    assert fns == []
    tp, fp, fn = greedy_match([a, extra], [b], 0.3)
    assert (tp, fp, fn) == (1, 1, 0)


def test_parse_label() -> None:
    assert parse_label(1) == 1
    assert parse_label("0") == 0
    assert parse_label("clear") == 1
    assert parse_label("unclear") == 0
    assert parse_label("") is None


def test_split_files_no_overlap() -> None:
    root = Path(__file__).resolve().parents[1]
    dev = root / "data" / "splits" / "gold_dev.txt"
    tes = root / "data" / "splits" / "gold_test.txt"
    if not dev.exists() or not tes.exists():
        return
    d = {x.strip() for x in dev.read_text().splitlines() if x.strip()}
    t = {x.strip() for x in tes.read_text().splitlines() if x.strip()}
    assert d.isdisjoint(t)


def test_save_publication_figure_writes_png_and_jpg(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    stem = tmp_path / "unit_test_fig"
    log = FigureLog()
    ok = save_publication_figure(fig, stem, log, source="test", force=True)
    assert ok
    assert stem.with_suffix(".png").exists()
    assert stem.with_suffix(".jpg").exists()
    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [1, 0])
    ok2 = save_publication_figure(fig2, stem, log, source="test", force=False)
    assert ok2 is False
