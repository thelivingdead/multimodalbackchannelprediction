#!/usr/bin/env python3
"""Planned experiment, not run: evaluation of the VideoMAE model.

There is no trained VideoMAE model in this repository (see
scripts/15_train_videomae.py and reports/videomae_preflight_lab.md), so there
is nothing to evaluate. When the planned visual experiment is run, evaluation
must follow the same protocol as the pose models: the frozen 15-video TEST
split in data/splits/gold_test.txt is scored exactly once, with all selection
on DEV. No TEST metric for VideoMAE exists or is claimed anywhere in this repo.

This script intentionally performs no computation.
"""
from __future__ import annotations


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
