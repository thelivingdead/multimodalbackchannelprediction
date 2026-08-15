"""Eyebrow raise — NOT implemented until an EMOCA expression key is VERIFIED."""
from __future__ import annotations

import pandas as pd

from ..events import Event


class EyebrowRaiseRule:
    name = "eyebrow_raise"
    supported = False

    def detect(self, pose: pd.DataFrame, video_id: str, person: str) -> list[Event]:
        return []
