"""Project paths. All scripts import ROOT from here."""
from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_PKG_MARKER = "dissertation-behaviour-recognition/"


def artefact_relpath(path: Path | str) -> str:
    """JSON/CSV path string relative to this package when the file is inside it.

    ``/user/.../dissertation-behaviour-recognition/results/foo.csv`` becomes
    ``results/foo.csv``. Paths outside the package are left unchanged.
    """
    raw = str(path).replace("\\", "/")
    if _PKG_MARKER in raw:
        return raw.split(_PKG_MARKER, 1)[1]
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return raw


def maybe_artefact_relpath(value: str) -> str:
    """Rewrite absolute lab/home paths; leave ordinary strings alone."""
    if value.startswith(("/user/", "/home/")):
        return artefact_relpath(value)
    if value.startswith("/") and _PKG_MARKER in value.replace("\\", "/"):
        return artefact_relpath(value)
    return value


def sanitise_artefact(obj: Any) -> Any:
    """Walk a JSON-like object and rewrite absolute package paths."""
    if isinstance(obj, Path):
        return artefact_relpath(obj)
    if isinstance(obj, str):
        return maybe_artefact_relpath(obj)
    if isinstance(obj, dict):
        return {k: sanitise_artefact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise_artefact(v) for v in obj]
    if isinstance(obj, tuple):
        return type(obj)(sanitise_artefact(v) for v in obj)
    return obj


def json_default(obj: Any) -> str:
    if isinstance(obj, Path):
        return artefact_relpath(obj)
    return str(obj)


def ensure_dirs() -> None:
    # Minimal set used by the executed pipeline. Figure subdirectories are
    # created on demand by src.plotting.save_publication_figure, and the
    # proposal-era staging dirs (data/emoca, data/headpose, data/pseudo,
    # data/context, data/working, data/manifests) are no longer pre-created:
    # scripts/make_figures.py skips gracefully when their inputs are absent.
    for rel in (
        "configs",
        "data/gold",
        "data/splits",
        "results",
        "figures",
        "reports/dissertation_evidence",
        "checkpoints",
        "logs",
        "tests",
    ):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
