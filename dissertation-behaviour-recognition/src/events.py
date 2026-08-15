"""Temporal events and IoU matching. Primary event metric: F1 at IoU 0.30."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    video_id: str
    start_s: float
    end_s: float
    label: str = "nod"
    person: str = "p0"

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def iou(a: Event, b: Event) -> float:
    if a.video_id != b.video_id:
        return 0.0
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = max(a.end_s, b.end_s) - min(a.start_s, b.start_s)
    return inter / union if union > 0 else 0.0


def match_pairs(
    pred: list[Event],
    gold: list[Event],
    iou_thr: float,
) -> tuple[list[tuple[Event, Event]], list[Event], list[Event]]:
    """One-to-one greedy matching. Returns (tp pairs, unmatched pred, unmatched gold)."""
    gold_by: dict[str, list[Event]] = {}
    for g in gold:
        gold_by.setdefault(g.video_id, []).append(g)
    used: set[int] = set()
    pos = {id(g): i for i, g in enumerate(gold)}
    tps: list[tuple[Event, Event]] = []
    fps: list[Event] = []
    for p in pred:
        best_i, best = -1, 0.0
        for g in gold_by.get(p.video_id, []):
            gi = pos[id(g)]
            if gi in used:
                continue
            v = iou(p, g)
            if v > best:
                best, best_i = v, gi
        if best >= iou_thr and best_i >= 0:
            tps.append((p, gold[best_i]))
            used.add(best_i)
        else:
            fps.append(p)
    fns = [g for i, g in enumerate(gold) if i not in used]
    return tps, fps, fns


def greedy_match(
    pred: list[Event],
    gold: list[Event],
    iou_thr: float,
) -> tuple[int, int, int]:
    """One-to-one greedy matching. Returns tp, fp, fn counts."""
    tps, fps, fns = match_pairs(pred, gold, iou_thr)
    return len(tps), len(fps), len(fns)


def merge_positive_frames(
    video_id: str,
    times: list[float],
    positive: list[bool],
    person: str = "p0",
    min_dur: float = 0.20,
) -> list[Event]:
    """Collapse consecutive positive samples into events."""
    events: list[Event] = []
    start: float | None = None
    last: float | None = None
    for t, pos in zip(times, positive):
        if pos:
            if start is None:
                start = t
            last = t
        elif start is not None and last is not None:
            if last - start >= min_dur:
                events.append(Event(video_id, start, last, "nod", person))
            start = last = None
    if start is not None and last is not None and last - start >= min_dur:
        events.append(Event(video_id, start, last, "nod", person))
    return events
