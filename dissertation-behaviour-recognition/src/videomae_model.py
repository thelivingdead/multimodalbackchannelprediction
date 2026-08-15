"""VideoMAE is blocked until the nod rule baseline exists."""
from __future__ import annotations


def refuse() -> None:
    raise SystemExit(
        "Do not start VideoMAE until results/pilot_nod_rule_metrics.json exists "
        "and the nod rule pipeline is correct."
    )
