"""Structured-feature classifier stub. Train only after the nod rule pilot exists."""
from __future__ import annotations


def require_pilot() -> None:
    from .paths import ROOT

    if not (ROOT / "results" / "pilot_nod_rule_metrics.json").exists():
        raise SystemExit("Finish the nod rule pilot before training a pose classifier.")
