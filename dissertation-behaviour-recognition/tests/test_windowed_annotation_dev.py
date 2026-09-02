"""DEV nod-event annotation workflow. No training, no TEST, no window dataset."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.windowed_protocol import (
    CLIP_SEC,
    ENTRY_TEST_CSV,
    EVENTS_CSV,
    FPS,
    ORIGINAL_GOLD_FILES,
    SHAKE_ENTRY_TEST_CSV,
    SHAKE_EVENTS_CSV,
    SHAKE_WINDOWS_DEV_CSV,
    STATUS_CSV,
    WINDOWS_DEV_CSV,
    WINDOWS_TEST_CSV,
    WindowedProtocolError,
    annotation_complete,
    clip_records,
    compile_shake_test_entry_file,
    compile_test_entry_file,
    generate_shake_test_windows,
    generate_test_windows,
    example_protocol_labels,
    events_to_rows,
    iter_window_bounds,
    load_events,
    load_gold,
    load_status,
    replace_clip_events,
    require_test_id,
    sec_to_rel_frame,
    validate_event,
    window_label,
    windows_must_not_be_generated_yet,
)

ROOT = Path(__file__).resolve().parents[1]

GOLD_MD5 = {
    "data/gold_annotations.csv": "463ca404d03cb360c276f4bc28b8fa54",
    "data/gold/events.csv": "2f6d54e287cc21149a4740976bb53545",
    "data/gold/annotation_sheet.csv": "7bbe974accce83d10df0e46b37d6e49b",
}


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_original_gold_files_untouched() -> None:
    for rel, digest in GOLD_MD5.items():
        path = ROOT / rel
        assert path.is_file(), rel
        assert _md5(path) == digest, rel
    for path in ORIGINAL_GOLD_FILES:
        assert path.is_file()


def test_dev_only_fifteen_clips() -> None:
    recs = clip_records()
    ids = [r["sample_id"] for r in recs]
    assert len(ids) == 15
    assert ids == [f"gold_{i:03d}" for i in range(1, 16)]
    gold = load_gold()
    test_ids = set(gold.loc[gold["split"] == "TEST", "sample_id"].astype(str))
    assert test_ids == {f"gold_{i:03d}" for i in range(16, 31)}
    assert test_ids.isdisjoint(set(ids))
    recs = clip_records()
    by_id = {r["sample_id"]: r for r in recs}
    assert by_id["gold_001"]["watch_side"] == "LEFT"
    assert by_id["gold_006"]["watch_side"] == "RIGHT"


def test_status_file_has_only_dev() -> None:
    st = pd.read_csv(STATUS_CSV)
    assert list(st.columns) == ["sample_id", "reviewed", "n_nod_events", "notes"]
    assert list(st["sample_id"]) == [f"gold_{i:03d}" for i in range(1, 16)]


def test_events_file_is_dev_only() -> None:
    ev = pd.read_csv(EVENTS_CSV)
    assert list(ev.columns) == [
        "sample_id",
        "event_id",
        "start_sec",
        "end_sec",
        "start_frame_relative",
        "end_frame_relative",
    ]
    if ev.empty:
        return
    ids = set(ev["sample_id"].astype(str))
    assert ids <= {f"gold_{i:03d}" for i in range(1, 16)}
    assert ids.isdisjoint({f"gold_{i:03d}" for i in range(16, 31)})
    assert (ev["end_sec"] > ev["start_sec"]).all()
    assert (ev["start_sec"] >= 0).all()
    assert (ev["end_sec"] <= 60.0 + 1e-6).all()


def test_relative_frames_25fps() -> None:
    assert FPS == 25.0
    assert sec_to_rel_frame(2.20) == 55
    assert sec_to_rel_frame(2.85) == 71
    assert sec_to_rel_frame(17.40) == 435
    assert sec_to_rel_frame(18.30) == 458


def test_event_bounds() -> None:
    validate_event(0.0, 0.4, clip_sec=CLIP_SEC)
    try:
        validate_event(2.85, 2.20)
        raise AssertionError("expected start < end")
    except WindowedProtocolError:
        pass
    try:
        validate_event(-0.1, 1.0)
        raise AssertionError("expected clip bounds")
    except WindowedProtocolError:
        pass
    try:
        validate_event(59.0, 60.5)
        raise AssertionError("expected clip bounds")
    except WindowedProtocolError:
        pass


def test_zero_events_can_be_reviewed(tmp_path: Path) -> None:
    ev_path = tmp_path / "nod_events_windowed.csv"
    st_path = tmp_path / "annotation_status.csv"
    replace_clip_events(
        "gold_002",
        [],
        events_path=ev_path,
        status_path=st_path,
        reviewed=True,
        notes="no clear nods",
    )
    ev = load_events(ev_path)
    assert ev.empty or "gold_002" not in set(ev["sample_id"].astype(str))
    st = load_status(st_path, sample_ids=["gold_002"])
    row = st.loc[st["sample_id"] == "gold_002"].iloc[0]
    assert bool(row["reviewed"]) is True
    assert int(row["n_nod_events"]) == 0
    assert "no clear nods" in str(row["notes"])


def test_events_persist(tmp_path: Path) -> None:
    ev_path = tmp_path / "nod_events_windowed.csv"
    st_path = tmp_path / "annotation_status.csv"
    replace_clip_events(
        "gold_001",
        [
            {"start_sec": 2.20, "end_sec": 2.85},
            {"start_sec": 17.40, "end_sec": 18.30},
            {"start_sec": 42.10, "end_sec": 43.05},
        ],
        events_path=ev_path,
        status_path=st_path,
    )
    ev = load_events(ev_path)
    assert list(ev["event_id"]) == ["nod_001", "nod_002", "nod_003"]
    assert list(ev["start_frame_relative"]) == [55, 435, 1052]
    assert list(ev["end_frame_relative"]) == [71, 458, 1076]
    again = load_events(ev_path)
    assert len(again) == 3
    st = load_status(st_path, sample_ids=["gold_001"])
    assert int(st.iloc[0]["n_nod_events"]) == 3
    assert bool(st.iloc[0]["reviewed"]) is False


def test_refuse_test_sample(tmp_path: Path) -> None:
    try:
        replace_clip_events(
            "gold_016",
            [{"start_sec": 1.0, "end_sec": 1.4}],
            events_path=tmp_path / "e.csv",
            status_path=tmp_path / "s.csv",
        )
        raise AssertionError("TEST must be refused")
    except WindowedProtocolError as e:
        assert "TEST" in str(e)


def test_window_rule_example() -> None:
    nods = [(2.3, 2.9), (7.2, 7.9)]
    got = {(r["start_sec"], r["end_sec"]): r["label"] for r in example_protocol_labels()}
    assert got[(0.0, 3.0)] == 1
    assert got[(2.0, 5.0)] == 1
    assert got[(4.0, 7.0)] == 0
    assert got[(6.0, 9.0)] == 1
    assert got[(8.0, 11.0)] == 0
    assert window_label(4.0, 7.0, nods) == 0


def test_twenty_nine_windows() -> None:
    wins = iter_window_bounds()
    assert len(wins) == 29
    assert wins[0] == (0.0, 3.0)
    assert wins[1] == (2.0, 5.0)
    assert wins[-1] == (56.0, 59.0)


def test_do_not_generate_windows_before_review(tmp_path: Path) -> None:
    st_path = tmp_path / "annotation_status.csv"
    st_path.write_text("sample_id,reviewed,n_nod_events,notes\ngold_001,false,0,\n")
    try:
        windows_must_not_be_generated_yet(st_path)
        raise AssertionError("must wait for reviewed=true on every DEV clip")
    except WindowedProtocolError:
        pass


def test_dev_windows_file_if_present() -> None:
    if not WINDOWS_DEV_CSV.exists():
        return
    win = pd.read_csv(WINDOWS_DEV_CSV)
    assert set(win["sample_id"].astype(str)) <= {f"gold_{i:03d}" for i in range(1, 16)}
    assert "gold_016" not in set(win["sample_id"].astype(str))
    assert (win["split"].astype(str).str.upper() == "DEV").all()
    assert len(win) == 15 * 29
    g12 = win[win["sample_id"] == "gold_012"]
    if not g12.empty:
        assert int(g12["label"].sum()) == 0


def test_require_test_id() -> None:
    require_test_id("gold_016")
    require_test_id("gold_030")
    try:
        require_test_id("gold_001")
        raise AssertionError("DEV id must not pass require_test_id")
    except WindowedProtocolError:
        pass
    try:
        events_to_rows("gold_016", [{"start_sec": 1.0, "end_sec": 1.4}])
        raise AssertionError("events_to_rows must refuse TEST unless allow_test")
    except WindowedProtocolError:
        pass
    rows = events_to_rows(
        "gold_016",
        [{"start_sec": 1.0, "end_sec": 1.4}],
        allow_test=True,
    )
    assert list(rows["event_id"]) == ["nod_001"]


def test_compile_test_entry_does_not_touch_dev(tmp_path: Path) -> None:
    if not ENTRY_TEST_CSV.exists():
        return
    ev_path = tmp_path / "nod_events_windowed_test.csv"
    st_path = tmp_path / "annotation_status_test.csv"
    ev, st, _log = compile_test_entry_file(
        ENTRY_TEST_CSV,
        events_path=ev_path,
        status_path=st_path,
    )
    ids = set(ev["sample_id"].astype(str))
    assert ids <= {f"gold_{i:03d}" for i in range(16, 31)}
    assert "gold_001" not in ids
    assert set(st["sample_id"].astype(str)) == {f"gold_{i:03d}" for i in range(16, 31)}
    assert bool(st["reviewed"].all())
    if EVENTS_CSV.exists():
        dev_ids = set(pd.read_csv(EVENTS_CSV)["sample_id"].astype(str))
        assert not (dev_ids & ids)


def test_test_windows_file_if_present() -> None:
    if not WINDOWS_TEST_CSV.exists():
        return
    win = pd.read_csv(WINDOWS_TEST_CSV)
    assert set(win["sample_id"].astype(str)) == {f"gold_{i:03d}" for i in range(16, 31)}
    assert "gold_001" not in set(win["sample_id"].astype(str))
    assert (win["split"].astype(str).str.upper() == "TEST").all()
    assert len(win) == 15 * 29


def test_windowed_pose_slice_is_75_frames() -> None:
    import importlib.util

    from src.pose_cnn import load_npz

    spec = importlib.util.spec_from_file_location(
        "train_windowed_nod_pose_cnn",
        ROOT / "scripts" / "train_windowed_nod_pose_cnn.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    z = load_npz(ROOT / "features" / "gold" / "gold_001.npz")
    chunk = mod.slice_rotation(z, 0, 75)
    assert chunk.shape == (mod.WINDOW_FRAMES, 3)
    later = mod.slice_rotation(z, 50, 125)
    assert later.shape == (mod.WINDOW_FRAMES, 3)


def test_generate_test_windows_will_not_overwrite_dev(tmp_path: Path) -> None:
    try:
        generate_test_windows(out_path=WINDOWS_DEV_CSV)
        raise AssertionError("must refuse DEV window path")
    except WindowedProtocolError as e:
        assert "DEV" in str(e)


def test_protocol_figure_exists_after_plot() -> None:
    import importlib.util

    fig = ROOT / "results" / "windowed_dev" / "window_label_logic.png"
    test_fig = ROOT / "results" / "windowed_test" / "window_label_logic.png"
    spec = importlib.util.spec_from_file_location(
        "plot_window_label_logic",
        ROOT / "scripts" / "plot_window_label_logic.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main()
    mod.write_test()
    assert fig.is_file()
    assert fig.stat().st_size > 1000
    assert test_fig.is_file()
    assert test_fig.stat().st_size > 1000


def test_shake_test_entry_is_blank_test_only() -> None:
    assert SHAKE_ENTRY_TEST_CSV.is_file()
    raw = pd.read_csv(SHAKE_ENTRY_TEST_CSV)
    ids = set(raw["sample_id"].astype(str))
    assert ids == {f"gold_{i:03d}" for i in range(16, 31)}
    assert "gold_001" not in ids
    assert len(raw) == 15 * 5


def test_compile_shake_test_does_not_touch_dev(tmp_path: Path) -> None:
    ev, st, _log = compile_shake_test_entry_file(
        SHAKE_ENTRY_TEST_CSV,
        events_path=tmp_path / "shake_events_windowed_test.csv",
        status_path=tmp_path / "annotation_status_shake_test.csv",
    )
    if not ev.empty:
        assert set(ev["sample_id"].astype(str)) <= {f"gold_{i:03d}" for i in range(16, 31)}
        assert "gold_001" not in set(ev["sample_id"].astype(str))
    ids = set(st["sample_id"].astype(str))
    assert ids == {f"gold_{i:03d}" for i in range(16, 31)}
    assert "gold_001" not in ids
    try:
        compile_shake_test_entry_file(
            SHAKE_ENTRY_TEST_CSV,
            events_path=SHAKE_EVENTS_CSV,
            status_path=tmp_path / "annotation_status_shake_test.csv",
        )
        raise AssertionError("must refuse DEV shake events path")
    except WindowedProtocolError:
        pass


def test_generate_shake_test_windows_will_not_overwrite_dev() -> None:
    try:
        generate_shake_test_windows(out_path=SHAKE_WINDOWS_DEV_CSV)
        raise AssertionError("must refuse DEV shake window path")
    except WindowedProtocolError as e:
        assert "DEV" in str(e) or "overwrite" in str(e).lower()
    try:
        generate_shake_test_windows(out_path=WINDOWS_DEV_CSV)
        raise AssertionError("must refuse nod DEV window path")
    except WindowedProtocolError:
        pass
