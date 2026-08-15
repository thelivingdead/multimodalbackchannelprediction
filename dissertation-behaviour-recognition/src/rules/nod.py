"""Rule-based head-nod detector on pitch cycles.

A nod is a short vertical oscillation (pitch), typically 0.25–1.4 s,
with at least two direction changes after 1–3 Hz band-pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from ..events import Event


@dataclass
class NodRule:
    name: str = "nod"
    min_dur: float = 0.25
    max_dur: float = 1.40
    min_range_deg: float = 2.5
    min_reversals: int = 2
    band_low_hz: float = 1.0
    band_high_hz: float = 3.0
    fps: float = 25.0

    def detect(self, pose: pd.DataFrame, video_id: str, person: str) -> list[Event]:
        sub = pose[pose["person"] == person].sort_values("time_s")
        if len(sub) < 16:
            return []
        t = sub["time_s"].to_numpy(float)
        pitch = sub["pitch"].to_numpy(float)
        filt = _bandpass(pitch, self.fps, self.band_low_hz, self.band_high_hz)
        vel = np.gradient(filt, t)
        sign = np.sign(vel)
        sign[sign == 0] = 1
        turns = np.where(np.diff(sign) != 0)[0] + 1
        events: list[Event] = []
        i = 0
        while i < len(turns) - 1:
            for span in (2, 3):
                if i + span >= len(turns):
                    continue
                a, b = int(turns[i]), int(turns[i + span])
                dur = float(t[b] - t[a])
                if dur < self.min_dur or dur > self.max_dur:
                    continue
                mag = float(np.ptp(filt[a : b + 1]))
                if mag < self.min_range_deg:
                    continue
                if span < self.min_reversals:
                    continue
                events.append(Event(video_id, float(t[a]), float(t[b]), "nod", person))
            i += 1
        return _nms(events)

    def score_frames(self, pose: pd.DataFrame, person: str) -> np.ndarray:
        """Per-frame score = |band-passed pitch| in degrees (for PR-AUC)."""
        sub = pose[pose["person"] == person].sort_values("time_s")
        pitch = sub["pitch"].to_numpy(float)
        return np.abs(_bandpass(pitch, self.fps, self.band_low_hz, self.band_high_hz))


def _bandpass(x: np.ndarray, fps: float, low: float, high: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < 16:
        return np.zeros_like(x)
    nyq = 0.5 * fps
    high = min(high, nyq * 0.99)
    low = max(low, 1e-3)
    if low >= high:
        return np.zeros_like(x)
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def _nms(events: list[Event]) -> list[Event]:
    events = sorted(events, key=lambda e: (e.start_s, -(e.end_s - e.start_s)))
    kept: list[Event] = []
    for e in events:
        if kept and e.start_s < kept[-1].end_s - 0.05:
            if (e.end_s - e.start_s) > (kept[-1].end_s - kept[-1].start_s):
                kept[-1] = e
            continue
        kept.append(e)
    return kept
