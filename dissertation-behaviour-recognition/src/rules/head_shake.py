"""Head-shake rule (yaw). Implemented but not scored until gold shake labels exist."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..events import Event
from .nod import NodRule


@dataclass
class HeadShakeRule:
    name: str = "head_shake"
    inner: NodRule | None = None

    def detect(self, pose: pd.DataFrame, video_id: str, person: str) -> list[Event]:
        # Reuse cycle logic on yaw by temporarily copying yaw into pitch column.
        tmp = pose.copy()
        tmp["pitch"] = tmp["yaw"]
        rule = self.inner or NodRule(min_range_deg=3.0)
        evs = rule.detect(tmp, video_id, person)
        return [Event(e.video_id, e.start_s, e.end_s, "head_shake", e.person) for e in evs]
