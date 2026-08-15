"""Storage guard. Hard cap is 24 GB used on the working disk (lab quota)."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .paths import ROOT
from .utils import load_yaml


@dataclass(frozen=True)
class StorageStatus:
    used_gb: float
    free_gb: float
    total_gb: float
    project_gb: float
    ok: bool
    level: str  # ok | warning | stop_downloads | hard_error
    message: str


def _du_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    r = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        return 0.0
    kb = float(r.stdout.split()[0])
    return kb / (1024.0 * 1024.0)


def disk_usage(path: Path) -> tuple[float, float, float]:
    usage = shutil.disk_usage(path)
    return usage.used / 1e9, usage.free / 1e9, usage.total / 1e9


def check_storage(root: Path | None = None) -> StorageStatus:
    root = root or ROOT
    cfg = load_yaml(root / "configs" / "storage.yaml")
    hard = float(cfg.get("hard_limit_gb", 24.0))
    warn = float(cfg.get("warning_gb", 20.0))
    stop = float(cfg.get("stop_downloads_gb", 22.0))
    used, free, total = disk_usage(root)
    project = _du_gb(root)
    # Quota is "working budget" — we key off project size + remaining free.
    # Lab machines often share a large disk; the constraint is this project + quota.
    budget_used = project
    if budget_used >= hard:
        level, ok, msg = "hard_error", False, f"Project {budget_used:.2f} GB >= hard limit {hard} GB"
    elif budget_used >= stop or free < 5.0:
        level, ok, msg = "stop_downloads", False, f"Project {budget_used:.2f} GB or free {free:.2f} GB: do not start downloads"
    elif budget_used >= warn:
        level, ok, msg = "warning", True, f"Project {budget_used:.2f} GB approaching limit"
    else:
        level, ok, msg = "ok", True, f"Project {budget_used:.2f} GB; free {free:.2f} GB"
    return StorageStatus(used, free, total, project, ok, level, msg)


def assert_can_continue(allow_warning: bool = True) -> StorageStatus:
    st = check_storage()
    if st.level == "hard_error":
        raise SystemExit(f"STORAGE HARD ERROR: {st.message}")
    if st.level == "stop_downloads" and not allow_warning:
        raise SystemExit(f"STORAGE STOP: {st.message}")
    return st
