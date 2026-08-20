"""Project paths. All scripts import ROOT from here."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
