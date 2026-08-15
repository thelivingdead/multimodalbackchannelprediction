"""Project paths. All scripts import ROOT from here."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_dirs() -> None:
    for rel in (
        "configs",
        "data/gold",
        "data/manifests",
        "data/splits",
        "data/emoca",
        "data/headpose",
        "data/pseudo",
        "data/context",
        "data/working",
        "results",
        "figures/dataset",
        "figures/annotations",
        "figures/rule_baseline",
        "figures/pseudo_labels",
        "figures/videomae",
        "figures/ablations",
        "figures/error_analysis",
        "figures/final_results",
        "figures/pilot_nod",
        "figures/rules/nod",
        "reports/dissertation_evidence",
        "checkpoints",
        "logs",
        "tests",
    ):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
