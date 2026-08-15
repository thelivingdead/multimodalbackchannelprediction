"""Small helpers: YAML, JSON, seeds, subprocess disk."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as f:
        data = yaml.safe_load(f)
    return dict(data or {})


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def format_mmss(seconds: float | None) -> str:
    """YouTube-style clock, e.g. 55s → 0:55, 701s → 11:41."""
    if seconds is None:
        return ""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return ""
    if s != s:  # NaN
        return ""
    s = max(0.0, s)
    m = int(s // 60)
    whole = int(round(s - 60 * m))
    if whole == 60:
        m += 1
        whole = 0
    return f"{m}:{whole:02d}"


def parse_clock(raw: object) -> float | None:
    """Parse 0:55, 11:41, or a raw second count. Returns seconds."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if raw != raw:
            return None
        return float(raw)
    s = str(raw).strip().lower()
    if s in ("", "nan", "none"):
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
            if len(parts) == 3:
                return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        except ValueError:
            return None
        return None
    try:
        return float(s)
    except ValueError:
        return None


def git_commit(root: Path) -> str:
    try:
        import subprocess

        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except OSError:
        return "unknown"
