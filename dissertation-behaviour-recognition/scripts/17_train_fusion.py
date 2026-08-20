#!/usr/bin/env python3
"""Planned experiment, not run: late fusion of VideoMAE and pose features.

Fusion (concatenate the VideoMAE embedding with the pose summary, then a small
MLP; see configs/fusion.yaml) requires the planned VideoMAE experiment first,
which was not run (storage constraint; see reports/videomae_preflight_lab.md).
The submitted study therefore reports pose-only results: the frozen rule
(TEST F1 0.67) and the pseudo-labelled 1D CNN (TEST F1 0.70). No fusion score
exists or is claimed.

This script intentionally performs no computation.
"""
from __future__ import annotations


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
