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
