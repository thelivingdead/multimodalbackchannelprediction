#!/usr/bin/env python3
"""Planned experiment, not run: late fusion of VideoMAE and pose features.

Fusion (concatenate the VideoMAE embedding with the pose summary, then a small
MLP; see ``configs/fusion.yaml``) has not been trained. Frozen VideoMAE
embeddings for the 110 experiment clips now exist (20 August 2026;
``results/videomae_embeddings_meta.json``), so the fusion variant is
technically unblocked, but no fusion model has been trained or scored.

The submitted study reports pose-only results: the frozen rule (TEST F1 0.67)
and the pseudo-labelled 1D CNN (TEST F1 0.70). No fusion score exists or is
claimed.

This script intentionally performs no computation.
"""
from __future__ import annotations


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
