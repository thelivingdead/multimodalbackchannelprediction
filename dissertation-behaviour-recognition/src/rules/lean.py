"""Lean forward/back — NOT implemented until translation/depth is VERIFIED."""
from __future__ import annotations

import pandas as pd

from ..events import Event


class LeanRule:
    name = "lean"
    supported = False

    def detect(self, pose: pd.DataFrame, video_id: str, person: str) -> list[Event]:
        return []
