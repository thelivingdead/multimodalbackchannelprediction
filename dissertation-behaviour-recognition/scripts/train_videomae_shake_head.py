#!/usr/bin/env python3
"""Head-shake frozen VideoMAE MLP head (does not touch nod artefacts).

Locked paths (cannot be omitted by accident):

* gold: ``data/gold/shake_annotation_sheet.csv`` column ``shake_label``
* TRAIN: ``results/shake/pseudo_labels.csv`` (80 clips, 75 pos / 5 neg)
* out-dir: ``results/shake/videomae_frozen_head``
  (checkpoint ``best_model.pt`` there, not ``models/videomae_head.pt``)

Never writes ``results/videomae_frozen_head/``, nod ``results/pseudo_labels.csv``,
or ``data/gold_annotations.csv``. TEST is scored once; the script refuses if
``results/shake/videomae_frozen_head/metrics.json`` exists (do not pass
``--force`` unless that run is formally invalidated).

Must already exist on otter disk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``data/gold/shake_annotation_sheet.csv`` (30 filled ``shake_label``)
* ``results/shake/pseudo_labels.csv``
* ``data/features/videomae/*.npz`` for gold_001–030 and pseudo_00001–00080
  (same embeddings as nod; labels are the task)
* ``results/videomae_embeddings_meta.json`` (read-only provenance)
* leakage PASS: shake pseudo ``video_id`` disjoint from gold DEV/TEST

Otter95 (``/scratch`` venv, **no Docker**; CPU is fine)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/train_videomae_shake_head.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = [
    "--gold-csv", str(ROOT / "data" / "gold" / "shake_annotation_sheet.csv"),
    "--label-col", "shake_label",
    "--pseudo-labels", str(ROOT / "results" / "shake" / "pseudo_labels.csv"),
    "--out-dir", str(ROOT / "results" / "shake" / "videomae_frozen_head"),
]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.argv = [sys.argv[0]] + DEFAULTS + sys.argv[1:]
    from train_videomae_head import main as _main

    _main()


if __name__ == "__main__":
    main()
