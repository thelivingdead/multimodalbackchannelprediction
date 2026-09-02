"""Nod-event annotation and the fixed 3 s / 2 s sliding-window rule.

Human work is event annotation only. Window 0/1 labels are derived later
from those events. DEV and TEST events are stored in separate files.
This module does not train or score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .paths import ROOT
from .utils import format_mmss

FPS = 25.0
CLIP_SEC = 60.0
WINDOW_SEC = 3.0
STRIDE_SEC = 2.0  # 1 s overlap
SPLIT_DEV = "DEV"
SPLIT_TEST = "TEST"

GOLD_ANNOTATIONS = ROOT / "data" / "gold_annotations.csv"
GOLD_EVENTS = ROOT / "data" / "gold" / "events.csv"
GOLD_SHEET = ROOT / "data" / "gold" / "annotation_sheet.csv"
WINDOWED_DIR = ROOT / "data" / "windowed_annotations"
EVENTS_CSV = WINDOWED_DIR / "nod_events_windowed.csv"
STATUS_CSV = WINDOWED_DIR / "annotation_status.csv"
ENTRY_CSV = WINDOWED_DIR / "nod_event_entry.csv"
WINDOWS_DEV_CSV = WINDOWED_DIR / "nod_windows_dev.csv"
ENTRY_TEST_CSV = WINDOWED_DIR / "nod_event_entry_test.csv"
EVENTS_TEST_CSV = WINDOWED_DIR / "nod_events_windowed_test.csv"
STATUS_TEST_CSV = WINDOWED_DIR / "annotation_status_test.csv"
WINDOWS_TEST_CSV = WINDOWED_DIR / "nod_windows_test.csv"
TEST_SAMPLE_IDS = [f"gold_{i:03d}" for i in range(16, 31)]
CLIPS_DIR = WINDOWED_DIR / "clips"
MIN_EVENT_SEC = 0.4

EVENT_COLUMNS = [
    "sample_id",
    "event_id",
    "start_sec",
    "end_sec",
    "start_frame_relative",
    "end_frame_relative",
]
STATUS_COLUMNS = ["sample_id", "reviewed", "n_nod_events", "notes"]

NOD_DEFINITION = (
    "A nod is a clear vertical head movement where the head moves down and up, "
    "or up and down, and returns towards its previous position. Annotate clear "
    "nod movements. Do not annotate head shakes, ordinary posture changes, "
    "small ambiguous movements, general head movement caused by speaking, or "
    "movements that cannot confidently be identified as a nod. If several nod "
    "cycles happen continuously with no meaningful pause, store them as one "
    "continuous nod event. If one nod clearly finishes before another starts, "
    "store them as separate events."
)

ORIGINAL_GOLD_FILES = (GOLD_ANNOTATIONS, GOLD_EVENTS, GOLD_SHEET)


def watch_side(who_to_watch: str) -> str:
    token = str(who_to_watch or "").strip().upper()
    if token.startswith("RIGHT"):
        return "RIGHT"
    if token.startswith("LEFT"):
        return "LEFT"
    return ""


class WindowedProtocolError(ValueError):
    """Invalid event, TEST leakage, or incomplete annotation state."""


def refuse_test_id(sample_id: str) -> None:
    sid = str(sample_id).strip()
    if sid.upper().startswith("TEST"):
        raise WindowedProtocolError(f"STOP: TEST id refused: {sid}")
    try:
        n = int(sid.split("_")[-1])
    except (ValueError, IndexError):
        return
    if n >= 16:
        raise WindowedProtocolError(
            f"STOP: {sid} is GOLD TEST. DEV event annotation will not load it."
        )


def require_test_id(sample_id: str) -> None:
    sid = str(sample_id).strip()
    try:
        n = int(sid.split("_")[-1])
    except (ValueError, IndexError) as exc:
        raise WindowedProtocolError(f"STOP: not a gold TEST id: {sid}") from exc
    if n < 16 or n > 30:
        raise WindowedProtocolError(f"STOP: {sid} is not GOLD TEST (gold_016–gold_030).")


def sec_to_rel_frame(sec: float, fps: float = FPS) -> int:
    return int(round(float(sec) * float(fps)))


def parse_clock(value: Any) -> float | None:
    """Parse m:ss, h:mm:ss, or a plain number. Semicolons count as colons."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(";", ":")
    if not text or text.lower() in {"nan", "none"}:
        return None
    if ":" not in text:
        return float(text)
    parts = [float(p) for p in text.split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise WindowedProtocolError(f"cannot parse clock value {value!r}")


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the intersection. Zero if the intervals only touch at an endpoint."""
    return max(0.0, min(float(a_end), float(b_end)) - max(float(a_start), float(b_start)))


def window_label(
    win_start: float,
    win_end: float,
    nods: Iterable[tuple[float, float]],
) -> int:
    """1 if the window has any non-zero overlap with a nod event, else 0."""
    for ns, ne in nods:
        if overlap_seconds(win_start, win_end, ns, ne) > 0.0:
            return 1
    return 0


def iter_window_bounds(
    clip_sec: float = CLIP_SEC,
    window_sec: float = WINDOW_SEC,
    stride_sec: float = STRIDE_SEC,
) -> list[tuple[float, float]]:
    """Inclusive start list: 0-3, 2-5, … while start + window <= clip duration."""
    out: list[tuple[float, float]] = []
    start = 0.0
    while start + window_sec <= clip_sec + 1e-9:
        out.append((start, start + window_sec))
        start += stride_sec
    return out


def validate_event(
    start_sec: float,
    end_sec: float,
    clip_sec: float = CLIP_SEC,
) -> tuple[float, float]:
    s = float(start_sec)
    e = float(end_sec)
    if not (e > s):
        raise WindowedProtocolError(
            f"start_sec must be < end_sec (got {s}, {e})"
        )
    if s < 0.0 or e > clip_sec + 1e-6:
        raise WindowedProtocolError(
            f"event {s:.3f}-{e:.3f}s falls outside 0-{clip_sec:g}s"
        )
    if s > clip_sec:
        raise WindowedProtocolError(f"start_sec {s} is outside the clip")
    return s, e


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "reviewed" in out.columns:
        out["reviewed"] = ["true" if bool(v) else "false" for v in out["reviewed"]]
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)


def load_gold() -> pd.DataFrame:
    df = pd.read_csv(GOLD_ANNOTATIONS)
    need = ["sample_id", "video_id", "start_frame", "end_frame", "person", "split"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise WindowedProtocolError(f"gold_annotations.csv missing {missing}")
    df["sample_id"] = df["sample_id"].astype(str)
    df["split"] = df["split"].astype(str).str.upper()
    df["start_frame"] = df["start_frame"].astype(int)
    df["end_frame"] = df["end_frame"].astype(int)
    return df


def load_dev_clips() -> pd.DataFrame:
    gold = load_gold()
    dev = gold.loc[gold["split"] == SPLIT_DEV].copy()
    if dev.empty:
        raise WindowedProtocolError("STOP: no DEV rows in gold_annotations.csv")
    for sid in dev["sample_id"]:
        refuse_test_id(sid)
    if (gold["split"] == SPLIT_TEST).any() and set(dev["sample_id"]) & set(
        gold.loc[gold["split"] == SPLIT_TEST, "sample_id"]
    ):
        raise WindowedProtocolError("STOP: a sample_id is in both DEV and TEST")
    sheet = _read_csv(GOLD_SHEET)
    if not sheet.empty and "video_id" in sheet.columns:
        keep = [c for c in ("video_id", "youtube_url", "who_to_watch", "watch_from", "watch_until") if c in sheet.columns]
        sheet = sheet[keep].drop_duplicates("video_id")
        dev = dev.merge(sheet, on="video_id", how="left")
    else:
        dev["youtube_url"] = ""
        dev["who_to_watch"] = ""
        dev["watch_from"] = ""
        dev["watch_until"] = ""
    clip_frames = (dev["end_frame"] - dev["start_frame"]).astype(float)
    dev["clip_sec"] = clip_frames / FPS
    dev["source_start_sec"] = dev["start_frame"].astype(float) / FPS
    dev["source_end_sec"] = dev["end_frame"].astype(float) / FPS
    return dev.sort_values("sample_id").reset_index(drop=True)


def clip_records() -> list[dict[str, Any]]:
    rows = []
    for _, r in load_dev_clips().iterrows():
        sid = str(r.sample_id)
        local = CLIPS_DIR / f"{sid}.mp4"
        who = str(r.get("who_to_watch") or "")
        rows.append(
            {
                "sample_id": sid,
                "video_id": str(r.video_id),
                "person": str(r.person),
                "who_to_watch": who,
                "watch_side": watch_side(who),
                "start_frame": int(r.start_frame),
                "end_frame": int(r.end_frame),
                "clip_sec": float(r.clip_sec),
                "source_start_sec": float(r.source_start_sec),
                "source_end_sec": float(r.source_end_sec),
                "source_start_clock": format_mmss(float(r.source_start_sec)),
                "source_end_clock": format_mmss(float(r.source_end_sec)),
                "youtube_url": str(r.get("youtube_url") or f"https://www.youtube.com/watch?v={r.video_id}"),
                "has_local_video": local.is_file() and local.stat().st_size > 1000,
            }
        )
    return rows


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def empty_status(sample_ids: Iterable[str]) -> pd.DataFrame:
    rows = [
        {"sample_id": sid, "reviewed": False, "n_nod_events": 0, "notes": ""}
        for sid in sample_ids
    ]
    return pd.DataFrame(rows, columns=STATUS_COLUMNS)


def _gate_sample_id(sample_id: str, *, allow_test: bool) -> None:
    if allow_test:
        require_test_id(sample_id)
    else:
        refuse_test_id(sample_id)


def load_events(path: Path | None = None, *, allow_test: bool = False) -> pd.DataFrame:
    path = path or EVENTS_CSV
    df = _read_csv(path)
    if df.empty:
        return empty_events()
    for c in EVENT_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df["sample_id"] = df["sample_id"].astype(str)
    for sid in df["sample_id"].unique():
        _gate_sample_id(sid, allow_test=allow_test)
    return df[EVENT_COLUMNS].copy()


def load_status(
    path: Path | None = None,
    sample_ids: Iterable[str] | None = None,
    *,
    allow_test: bool = False,
) -> pd.DataFrame:
    path = path or STATUS_CSV
    ids = [str(s) for s in (sample_ids if sample_ids is not None else [r["sample_id"] for r in clip_records()])]
    for sid in ids:
        _gate_sample_id(sid, allow_test=allow_test)
    df = _read_csv(path)
    if df.empty:
        return empty_status(ids)
    for c in STATUS_COLUMNS:
        if c not in df.columns:
            df[c] = "" if c == "notes" else 0
    df["sample_id"] = df["sample_id"].astype(str)
    for sid in df["sample_id"].unique():
        _gate_sample_id(sid, allow_test=allow_test)
    reviewed = []
    for v in df["reviewed"]:
        reviewed.append(str(v).strip().lower() in {"true", "1", "yes"})
    df["reviewed"] = reviewed
    df["n_nod_events"] = pd.to_numeric(df["n_nod_events"], errors="coerce").fillna(0).astype(int)
    df["notes"] = df["notes"].fillna("").astype(str)
    have = set(df["sample_id"])
    extra = empty_status([i for i in ids if i not in have])
    out = pd.concat([df[STATUS_COLUMNS], extra], ignore_index=True)
    order = {s: i for i, s in enumerate(ids)}
    out = out[out["sample_id"].isin(ids)].copy()
    out["_ord"] = out["sample_id"].map(order)
    return out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def events_to_rows(
    sample_id: str,
    events: list[dict[str, Any]],
    clip_sec: float = CLIP_SEC,
    *,
    allow_test: bool = False,
) -> pd.DataFrame:
    if allow_test:
        require_test_id(sample_id)
    else:
        refuse_test_id(sample_id)
    cleaned: list[tuple[float, float]] = []
    for ev in events:
        s, e = validate_event(ev["start_sec"], ev["end_sec"], clip_sec=clip_sec)
        cleaned.append((s, e))
    cleaned.sort()
    rows = []
    for i, (s, e) in enumerate(cleaned, start=1):
        rows.append(
            {
                "sample_id": sample_id,
                "event_id": f"nod_{i:03d}",
                "start_sec": round(s, 3),
                "end_sec": round(e, 3),
                "start_frame_relative": sec_to_rel_frame(s),
                "end_frame_relative": sec_to_rel_frame(e),
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS) if rows else empty_events()


def replace_clip_events(
    sample_id: str,
    events: list[dict[str, Any]],
    *,
    events_path: Path | None = None,
    status_path: Path | None = None,
    clip_sec: float = CLIP_SEC,
    notes: str | None = None,
    reviewed: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write events for one DEV clip. Updates n_nod_events. Does not imply reviewed."""
    refuse_test_id(sample_id)
    events_path = events_path or EVENTS_CSV
    status_path = status_path or STATUS_CSV
    new_rows = events_to_rows(sample_id, events, clip_sec=clip_sec)
    all_ev = load_events(events_path)
    all_ev = all_ev[all_ev["sample_id"] != sample_id]
    if not new_rows.empty:
        if all_ev.empty:
            all_ev = new_rows.copy()
        else:
            all_ev = pd.concat([all_ev, new_rows], ignore_index=True)
    if all_ev.empty:
        all_ev = empty_events()
    else:
        all_ev = all_ev.sort_values(["sample_id", "start_sec"]).reset_index(drop=True)
    atomic_to_csv(all_ev, events_path)

    ids = list(load_dev_clips()["sample_id"].astype(str)) if status_path == STATUS_CSV else None
    st = load_status(status_path, sample_ids=ids)
    mask = st["sample_id"] == sample_id
    if not mask.any():
        st = pd.concat(
            [st, empty_status([sample_id])],
            ignore_index=True,
        )
        mask = st["sample_id"] == sample_id
    st.loc[mask, "n_nod_events"] = int(len(new_rows))
    if notes is not None:
        st.loc[mask, "notes"] = notes
    if reviewed is not None:
        st.loc[mask, "reviewed"] = bool(reviewed)
    st["reviewed"] = st["reviewed"].astype(bool)
    atomic_to_csv(st, status_path)
    return new_rows, st.loc[st["sample_id"] == sample_id]


def set_reviewed(
    sample_id: str,
    reviewed: bool,
    *,
    notes: str | None = None,
    allow_zero_events: bool = True,
    events_path: Path | None = None,
    status_path: Path | None = None,
) -> pd.DataFrame:
    refuse_test_id(sample_id)
    events_path = events_path or EVENTS_CSV
    status_path = status_path or STATUS_CSV
    ev = load_events(events_path)
    n = int((ev["sample_id"] == sample_id).sum()) if not ev.empty else 0
    if reviewed and n == 0 and not allow_zero_events:
        raise WindowedProtocolError("zero-event reviewed flag refused")
    st = load_status(status_path)
    mask = st["sample_id"] == sample_id
    if not mask.any():
        st = pd.concat([st, empty_status([sample_id])], ignore_index=True)
        mask = st["sample_id"] == sample_id
    st.loc[mask, "reviewed"] = bool(reviewed)
    st.loc[mask, "n_nod_events"] = n
    if notes is not None:
        st.loc[mask, "notes"] = notes
    st["reviewed"] = st["reviewed"].astype(bool)
    atomic_to_csv(st, status_path)
    return st.loc[mask]


def compile_entry_file(
    entry_path: Path | None = None,
    *,
    events_path: Path | None = None,
    status_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Turn nod_event_entry.csv (YouTube clocks or relative seconds) into events + status.

    Does not write 3 s window labels. TEST ids are refused.
    """
    entry_path = entry_path or ENTRY_CSV
    events_path = events_path or EVENTS_CSV
    status_path = status_path or STATUS_CSV
    clips = {c["sample_id"]: c for c in clip_records()}
    raw = pd.read_csv(entry_path)
    need = {"sample_id", "start_sec", "end_sec"}
    missing = need - set(raw.columns)
    if missing:
        raise WindowedProtocolError(f"{entry_path.name} missing {sorted(missing)}")

    notes: dict[str, str] = {sid: "" for sid in clips}
    grouped: dict[str, list[dict[str, float]]] = {sid: [] for sid in clips}
    log: list[str] = []

    for _, row in raw.iterrows():
        sid = str(row["sample_id"]).strip()
        refuse_test_id(sid)
        if sid not in clips:
            raise WindowedProtocolError(f"STOP: {sid} is not a DEV gold clip")
        if pd.isna(row["start_sec"]) and pd.isna(row["end_sec"]):
            continue
        if pd.isna(row["start_sec"]) or pd.isna(row["end_sec"]):
            raise WindowedProtocolError(f"{sid} has only one of start_sec/end_sec filled")
        clock_s = parse_clock(row["start_sec"])
        clock_e = parse_clock(row["end_sec"])
        if clock_s is None or clock_e is None:
            continue
        origin = float(clips[sid]["source_start_sec"])
        clip_sec = float(clips[sid]["clip_sec"])
        raw_start = str(row["start_sec"])
        if ":" in raw_start or ";" in raw_start:
            rel_s = clock_s - origin
            rel_e = clock_e - origin
        else:
            rel_s = clock_s
            rel_e = clock_e
        if rel_e < rel_s:
            raise WindowedProtocolError(
                f"{sid} end before start ({row['start_sec']}, {row['end_sec']})"
            )
        if abs(rel_e - rel_s) < 1e-9:
            rel_e = rel_s + MIN_EVENT_SEC
            notes[sid] = "point time expanded by 0.4 s"
            log.append(f"{sid}: point mark {row['start_sec']} expanded to {MIN_EVENT_SEC:.1f}s")
        if rel_s < 0.0 and rel_s >= -0.51:
            rel_s = 0.0
        if rel_e > clip_sec and rel_e <= clip_sec + 0.51:
            rel_e = clip_sec
        s, e = validate_event(rel_s, rel_e, clip_sec=clip_sec)
        grouped[sid].append({"start_sec": s, "end_sec": e})

    for sid, events in grouped.items():
        replace_clip_events(
            sid,
            events,
            events_path=events_path,
            status_path=status_path,
            clip_sec=float(clips[sid]["clip_sec"]),
            notes=notes[sid],
            reviewed=bool(events),
        )

    st = load_status(status_path, sample_ids=list(clips))
    unfinished = [sid for sid, evs in grouped.items() if not evs]
    for sid in unfinished:
        log.append(f"{sid}: no events yet (not marked reviewed)")
    return load_events(events_path), st, log


def load_test_clip_meta() -> dict[str, dict[str, Any]]:
    gold = load_gold()
    test = gold.loc[gold["split"] == SPLIT_TEST].copy()
    if test.empty:
        raise WindowedProtocolError("STOP: no TEST rows in gold_annotations.csv")
    out: dict[str, dict[str, Any]] = {}
    for _, r in test.iterrows():
        sid = str(r.sample_id)
        require_test_id(sid)
        start_f = int(r.start_frame)
        end_f = int(r.end_frame)
        out[sid] = {
            "sample_id": sid,
            "video_id": str(r.video_id),
            "person": str(r.person),
            "source_start_sec": start_f / FPS,
            "clip_sec": (end_f - start_f) / FPS,
        }
    return out


def compile_test_entry_file(
    entry_path: Path | None = None,
    *,
    events_path: Path | None = None,
    status_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Compile TEST entry clocks into nod_events_windowed_test.csv. Does not touch DEV."""
    entry_path = entry_path or ENTRY_TEST_CSV
    events_path = events_path or EVENTS_TEST_CSV
    status_path = status_path or STATUS_TEST_CSV
    clips = load_test_clip_meta()
    raw = pd.read_csv(entry_path)
    need = {"sample_id", "start_sec", "end_sec"}
    missing = need - set(raw.columns)
    if missing:
        raise WindowedProtocolError(f"{entry_path.name} missing {sorted(missing)}")

    notes: dict[str, str] = {sid: "" for sid in clips}
    grouped: dict[str, list[dict[str, float]]] = {sid: [] for sid in clips}
    log: list[str] = []

    for _, row in raw.iterrows():
        sid = str(row["sample_id"]).strip()
        require_test_id(sid)
        if sid not in clips:
            raise WindowedProtocolError(f"STOP: {sid} is not a TEST gold clip")
        if pd.isna(row["start_sec"]) and pd.isna(row["end_sec"]):
            continue
        if pd.isna(row["start_sec"]) or pd.isna(row["end_sec"]):
            raise WindowedProtocolError(f"{sid} has only one of start_sec/end_sec filled")
        clock_s = parse_clock(row["start_sec"])
        clock_e = parse_clock(row["end_sec"])
        if clock_s is None or clock_e is None:
            continue
        origin = float(clips[sid]["source_start_sec"])
        clip_sec = float(clips[sid]["clip_sec"])
        raw_start = str(row["start_sec"])
        if ":" in raw_start or ";" in raw_start:
            rel_s = clock_s - origin
            rel_e = clock_e - origin
        else:
            rel_s = clock_s
            rel_e = clock_e
        if rel_e < rel_s:
            raise WindowedProtocolError(
                f"{sid} end before start ({row['start_sec']}, {row['end_sec']})"
            )
        if abs(rel_e - rel_s) < 1e-9:
            rel_e = rel_s + MIN_EVENT_SEC
            notes[sid] = "point time expanded by 0.4 s"
            log.append(f"{sid}: point mark {row['start_sec']} expanded to {MIN_EVENT_SEC:.1f}s")
        if rel_s < 0.0 and rel_s >= -0.51:
            rel_s = 0.0
        if rel_e > clip_sec and rel_e <= clip_sec + 0.51:
            rel_e = clip_sec
        s, e = validate_event(rel_s, rel_e, clip_sec=clip_sec)
        grouped[sid].append({"start_sec": s, "end_sec": e})

    parts = []
    status_rows = []
    for sid in [f"gold_{i:03d}" for i in range(16, 31)]:
        evs = grouped.get(sid, [])
        rows = events_to_rows(sid, evs, clip_sec=float(clips[sid]["clip_sec"]), allow_test=True)
        if not rows.empty:
            parts.append(rows)
        status_rows.append(
            {
                "sample_id": sid,
                "reviewed": True,
                "n_nod_events": int(len(rows)),
                "notes": notes.get(sid, "") if evs else "no clear nods",
            }
        )
        if not evs:
            log.append(f"{sid}: no events (reviewed, zero nods)")

    all_ev = pd.concat(parts, ignore_index=True) if parts else empty_events()
    st = pd.DataFrame(status_rows, columns=STATUS_COLUMNS)
    atomic_to_csv(all_ev, events_path)
    atomic_to_csv(st, status_path)
    return all_ev, st, log


def ensure_annotation_files() -> None:
    WINDOWED_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    ids = [str(s) for s in load_dev_clips()["sample_id"]]
    if not EVENTS_CSV.exists():
        atomic_to_csv(empty_events(), EVENTS_CSV)
    else:
        load_events(EVENTS_CSV)
    if not STATUS_CSV.exists():
        atomic_to_csv(empty_status(ids), STATUS_CSV)
    else:
        st = load_status(STATUS_CSV, sample_ids=ids)
        atomic_to_csv(st, STATUS_CSV)


def annotation_complete(status_path: Path | None = None) -> bool:
    st = load_status(status_path)
    return bool(len(st) and bool(st["reviewed"].all()))


def dump_state() -> dict[str, Any]:
    ensure_annotation_files()
    clips = clip_records()
    ev = load_events()
    st = load_status()
    events_by: dict[str, list[dict[str, Any]]] = {c["sample_id"]: [] for c in clips}
    if not ev.empty:
        for _, r in ev.iterrows():
            events_by.setdefault(str(r.sample_id), []).append(
                {
                    "event_id": str(r.event_id),
                    "start_sec": float(r.start_sec),
                    "end_sec": float(r.end_sec),
                    "start_frame_relative": int(r.start_frame_relative),
                    "end_frame_relative": int(r.end_frame_relative),
                }
            )
    status_by = {}
    for _, r in st.iterrows():
        status_by[str(r.sample_id)] = {
            "reviewed": bool(r.reviewed),
            "n_nod_events": int(r.n_nod_events),
            "notes": str(r.notes),
        }
    return {
        "fps": FPS,
        "clip_sec": CLIP_SEC,
        "window_sec": WINDOW_SEC,
        "stride_sec": STRIDE_SEC,
        "overlap_sec": WINDOW_SEC - STRIDE_SEC,
        "n_dev": len(clips),
        "nod_definition": NOD_DEFINITION,
        "clips": clips,
        "events": events_by,
        "status": status_by,
        "n_reviewed": int(sum(1 for v in status_by.values() if v["reviewed"])),
    }


def example_protocol_labels() -> list[dict[str, Any]]:
    """Fixed textbook example used by the protocol figure. Not measured data."""
    nods = [(2.3, 2.9), (7.2, 7.9)]
    windows = [(0.0, 3.0), (2.0, 5.0), (4.0, 7.0), (6.0, 9.0), (8.0, 11.0)]
    rows = []
    for ws, we in windows:
        rows.append(
            {
                "start_sec": ws,
                "end_sec": we,
                "label": window_label(ws, we, nods),
            }
        )
    return rows


def windows_must_not_be_generated_yet(status_path: Path | None = None) -> None:
    if not annotation_complete(status_path):
        raise WindowedProtocolError(
            "STOP: DEV event annotation is not finished. Every row in "
            "annotation_status.csv must have reviewed=true before 3 s "
            "window labels are generated."
        )


def generate_dev_windows(
    *,
    events_path: Path | None = None,
    status_path: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """3 s windows, 2 s stride, DEV only. Label 1 if any overlap with a nod."""
    windows_must_not_be_generated_yet(status_path)
    out_path = out_path or WINDOWS_DEV_CSV
    clips = clip_records()
    ev = load_events(events_path)
    nods_by: dict[str, list[tuple[float, float]]] = {c["sample_id"]: [] for c in clips}
    if not ev.empty:
        for _, r in ev.iterrows():
            sid = str(r.sample_id)
            refuse_test_id(sid)
            if sid not in nods_by:
                raise WindowedProtocolError(f"STOP: event for unknown DEV clip {sid}")
            nods_by[sid].append((float(r.start_sec), float(r.end_sec)))

    rows: list[dict[str, Any]] = []
    for clip in clips:
        sid = clip["sample_id"]
        refuse_test_id(sid)
        nods = nods_by[sid]
        bounds = iter_window_bounds(clip_sec=float(clip["clip_sec"]))
        for ws, we in bounds:
            ov = [overlap_seconds(ws, we, ns, ne) for ns, ne in nods]
            best = max(ov) if ov else 0.0
            touched = sum(1 for x in ov if x > 0.0)
            rows.append(
                {
                    "window_id": f"{sid}_w{sec_to_rel_frame(ws):05d}",
                    "sample_id": sid,
                    "video_id": clip["video_id"],
                    "person": clip["person"],
                    "split": SPLIT_DEV,
                    "start_sec": round(ws, 3),
                    "end_sec": round(we, 3),
                    "start_frame_relative": sec_to_rel_frame(ws),
                    "end_frame_relative": sec_to_rel_frame(we),
                    "label": 1 if best > 0.0 else 0,
                    "n_nods_touched": int(touched),
                }
            )
    win = pd.DataFrame(rows)
    if win.empty:
        raise WindowedProtocolError("STOP: no DEV windows generated")
    if (win["split"] != SPLIT_DEV).any():
        raise WindowedProtocolError("STOP: non-DEV window leaked")
    atomic_to_csv(win, out_path)
    return win


def annotation_complete_test(status_path: Path | None = None) -> bool:
    st = load_status(
        status_path or STATUS_TEST_CSV,
        sample_ids=TEST_SAMPLE_IDS,
        allow_test=True,
    )
    return bool(len(st) == len(TEST_SAMPLE_IDS) and bool(st["reviewed"].all()))


def clip_records_test() -> list[dict[str, Any]]:
    meta = load_test_clip_meta()
    missing = [sid for sid in TEST_SAMPLE_IDS if sid not in meta]
    if missing:
        raise WindowedProtocolError(f"STOP: TEST gold missing {missing}")
    return [meta[sid] for sid in TEST_SAMPLE_IDS]


def generate_test_windows(
    *,
    events_path: Path | None = None,
    status_path: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """3 s windows, 2 s stride, TEST only. Label 1 if any overlap with a nod."""
    if not annotation_complete_test(status_path):
        raise WindowedProtocolError(
            "STOP: TEST event annotation is not finished. Every row in "
            "annotation_status_test.csv must have reviewed=true before 3 s "
            "window labels are generated."
        )
    out_path = out_path or WINDOWS_TEST_CSV
    if out_path.resolve() == WINDOWS_DEV_CSV.resolve():
        raise WindowedProtocolError("STOP: will not overwrite DEV window labels")
    clips = clip_records_test()
    ev = load_events(events_path or EVENTS_TEST_CSV, allow_test=True)
    nods_by: dict[str, list[tuple[float, float]]] = {c["sample_id"]: [] for c in clips}
    if not ev.empty:
        for _, r in ev.iterrows():
            sid = str(r.sample_id)
            require_test_id(sid)
            if sid not in nods_by:
                raise WindowedProtocolError(f"STOP: event for unknown TEST clip {sid}")
            nods_by[sid].append((float(r.start_sec), float(r.end_sec)))

    rows: list[dict[str, Any]] = []
    for clip in clips:
        sid = clip["sample_id"]
        require_test_id(sid)
        nods = nods_by[sid]
        bounds = iter_window_bounds(clip_sec=float(clip["clip_sec"]))
        for ws, we in bounds:
            ov = [overlap_seconds(ws, we, ns, ne) for ns, ne in nods]
            best = max(ov) if ov else 0.0
            touched = sum(1 for x in ov if x > 0.0)
            rows.append(
                {
                    "window_id": f"{sid}_w{sec_to_rel_frame(ws):05d}",
                    "sample_id": sid,
                    "video_id": clip["video_id"],
                    "person": clip["person"],
                    "split": SPLIT_TEST,
                    "start_sec": round(ws, 3),
                    "end_sec": round(we, 3),
                    "start_frame_relative": sec_to_rel_frame(ws),
                    "end_frame_relative": sec_to_rel_frame(we),
                    "label": 1 if best > 0.0 else 0,
                    "n_nods_touched": int(touched),
                }
            )
    win = pd.DataFrame(rows)
    if win.empty:
        raise WindowedProtocolError("STOP: no TEST windows generated")
    if (win["split"] != SPLIT_TEST).any():
        raise WindowedProtocolError("STOP: non-TEST window leaked")
    if set(win["sample_id"].astype(str)) - set(TEST_SAMPLE_IDS):
        raise WindowedProtocolError("STOP: non-TEST sample_id in TEST windows")
    atomic_to_csv(win, out_path)
    return win
