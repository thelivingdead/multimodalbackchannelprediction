#!/usr/bin/env python3
"""VideoMAE evaluation: documentation pointer.

The frozen-head run (see ``scripts/15_train_videomae.py``) was evaluated by
``scripts/train_videomae_head.py`` itself, following the standard protocol —
epoch and threshold selected on the 15 DEV windows, the 15 TEST windows scored
exactly once — and plotted by ``scripts/plot_videomae_results.py``
(``figures/videomae_training_curve.png``). Saved artifacts:

  - ``results/videomae_frozen_head/training_history.csv``
  - ``results/videomae_frozen_head/predictions.csv``
  - ``results/videomae_frozen_head/metrics.json`` (TEST F1 0.57)

There is no separate fine-tuned VideoMAE model to evaluate; fine-tuning was
not run. This script intentionally performs no computation.
"""
from __future__ import annotations


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
