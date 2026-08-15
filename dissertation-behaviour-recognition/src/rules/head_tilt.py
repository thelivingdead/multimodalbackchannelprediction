"""Head-tilt rule (sustained roll). Not scored until gold tilt labels exist."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..events import Event


@dataclass
class HeadTiltRule:
    name: str = "head_tilt"
    min_abs_roll_deg: float = 8.0
    min_dur: float = 0.40

    def detect(self, pose: pd.DataFrame, video_id: str, person: str) -> list[Event]:
        sub = pose[pose["person"] == person].sort_values("time_s")
        if sub.empty:
            return []
        t = sub["time_s"].to_numpy(float)
        roll = sub["roll"].to_numpy(float)
        baseline = float(np.median(roll))
        flag = np.abs(roll - baseline) >= self.min_abs_roll_deg
        events: list[Event] = []
        start = None
        last = None
        for ti, f in zip(t, flag):
            if f:
                if start is None:
                    start = float(ti)
                last = float(ti)
            elif start is not None and last is not None:
                if last - start >= self.min_dur:
                    events.append(Event(video_id, start, last, "head_tilt", person))
                start = last = None
        if start is not None and last is not None and last - start >= self.min_dur:
            events.append(Event(video_id, start, last, "head_tilt", person))
        return events
