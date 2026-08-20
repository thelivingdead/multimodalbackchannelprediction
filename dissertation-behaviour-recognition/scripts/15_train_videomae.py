#!/usr/bin/env python3
"""VideoMAE frozen-head experiment: documentation pointer.

Status (20 August 2026): a first **frozen-encoder + trained head** run exists.
It was executed not by this numbered script but by:

  - ``scripts/extract_videomae_embeddings.py`` — 16-frame face crops through
    frozen ``MCG-NJU/videomae-base`` (768-d embeddings, 110 clips;
    ``results/videomae_embeddings_meta.json``),
  - ``scripts/train_videomae_head.py`` — small MLP head on the 80 rule
    pseudo-labels, epoch and probability threshold selected on DEV only
    (``results/videomae_frozen_head/``).

Result: DEV F1 0.90 (tuning split); **TEST F1 0.57** (P 0.55, R 0.60;
TP 6, FP 5, TN 0, FN 4) — below the pose 1D CNN (0.70), so it is not a
headline. The submitted dissertation results remain pose-only: frozen rule
TEST F1 0.67 and 1D CNN TEST F1 0.70.

Not run: VideoMAE **fine-tuning** (``configs/videomae_finetune.yaml``) — the
lab quota cannot hold video shards plus a trainable backbone; see
``reports/videomae_preflight_lab.md``.

This script intentionally performs no computation and downloads nothing.
"""
from __future__ import annotations

PREFLIGHT = "reports/videomae_preflight_lab.md"


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
