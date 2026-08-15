"""Shared rule detector interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from ..events import Event


@dataclass(frozen=True)
class RuleConfig:
    name: str
    params: dict[str, Any]


class RuleDetector(Protocol):
    name: str

    def detect(self, pose: pd.DataFrame, video_id: str, person: str) -> list[Event]:
        ...
