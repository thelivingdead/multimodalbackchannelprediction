"""Video / clip inventory helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import ROOT


def list_clip_dirs(subset: Path) -> list[Path]:
    if not subset.exists():
        return []
    return sorted(p for p in subset.iterdir() if p.is_dir() and (p / "meta.json").exists())


def read_meta(clip_dir: Path) -> dict[str, Any]:
    return json.loads((clip_dir / "meta.json").read_text())


def default_pilot_dir() -> Path:
    return ROOT / "data" / "working" / "pilot"
