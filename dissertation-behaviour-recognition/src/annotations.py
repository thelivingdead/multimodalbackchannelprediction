"""Gold annotations. Two classes only: 1 = clear nod, 0 = unclear."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .events import Event

LABEL_CLEAR = 1
LABEL_UNCLEAR = 0
LABEL_MAP = {1: "clear_nod", 0: "unclear", "1": "clear_nod", "0": "unclear", "clear": "clear_nod", "unclear": "unclear", "nod": "clear_nod"}


@dataclass(frozen=True)
class GoldEvent:
    video_id: str
    start_s: float
    end_s: float
    label: int  # 1 clear nod, 0 unclear
    annotator: str
    confidence: int
    participant_id: str = "p0"
    conversation_id: str = ""
    dyad_id: str = ""
    notes: str = ""

    def as_event(self) -> Event:
        return Event(self.video_id, self.start_s, self.end_s, "nod" if self.label == 1 else "unclear", self.participant_id)


def parse_label(raw: object) -> int | None:
    if raw is None or (isinstance(raw, float) and np_isnan(raw)):
        return None
    s = str(raw).strip().lower()
    if s in ("", "nan", "none"):
        return None
    if s in ("1", "clear", "clear_nod", "nod", "yes"):
        return LABEL_CLEAR
    if s in ("0", "unclear", "no", "non-nod", "unknown"):
        return LABEL_UNCLEAR
    try:
        v = int(float(s))
        if v in (0, 1):
            return v
    except ValueError:
        return None
    return None


def np_isnan(x: object) -> bool:
    try:
        import math

        return bool(math.isnan(float(x)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


EVENT_COLUMNS = [
    "video_id",
    "conversation_id",
    "dyad_id",
    "participant_id",
    "start_s",
    "end_s",
    "label",
    "annotator",
    "confidence",
    "notes",
]

LOG_COLUMNS = [
    "video_id",
    "video_duration_s",
    "annotation_time_s",
    "annotator",
    "behaviours_annotated",
    "event_count",
    "notes",
]


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return empty_events()
    df = pd.read_csv(path)
    for c in EVENT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df


def save_events(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def gold_nods(df: pd.DataFrame) -> list[Event]:
    """Only class 1 (clear nod) counts as a positive gold event."""
    out: list[Event] = []
    for _, r in df.iterrows():
        lab = parse_label(r.get("label"))
        if lab != LABEL_CLEAR:
            continue
        try:
            a, b = float(r.start_s), float(r.end_s)
        except (TypeError, ValueError):
            continue
        if b <= a:
            continue
        out.append(
            Event(
                str(r.video_id),
                a,
                b,
                "nod",
                str(r.get("participant_id") or "p0"),
            )
        )
    return out


def validate_events(df: pd.DataFrame, clip_duration: dict[str, float] | None = None) -> list[str]:
    errors: list[str] = []
    for i, r in df.iterrows():
        lab = parse_label(r.get("label"))
        if lab is None:
            errors.append(f"row {i}: missing label (use 1=clear nod, 0=unclear)")
            continue
        try:
            a, b = float(r.start_s), float(r.end_s)
        except (TypeError, ValueError):
            errors.append(f"row {i}: bad timestamps")
            continue
        if b <= a:
            errors.append(f"row {i}: end_s must be > start_s")
        vid = str(r.video_id)
        if clip_duration and vid in clip_duration and b > clip_duration[vid] + 0.05:
            errors.append(f"row {i}: {vid} end {b} exceeds duration {clip_duration[vid]}")
    return errors
