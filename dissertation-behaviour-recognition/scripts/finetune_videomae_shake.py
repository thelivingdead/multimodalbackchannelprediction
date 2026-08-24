#!/usr/bin/env python3
"""Head-shake VideoMAE partial fine-tune (does not touch nod artefacts).

Locked paths (cannot be omitted by accident):

* gold: ``data/gold/shake_annotation_sheet.csv`` column ``shake_label``
* TRAIN: ``results/shake/pseudo_labels.csv`` (80 clips, 75 pos / 5 neg)
* out-dir: ``results/shake/videomae_finetuned``

Never writes ``results/videomae_finetuned/``, nod ``results/pseudo_labels.csv``,
or ``data/gold_annotations.csv``. TEST is scored once; the script refuses if
``results/shake/videomae_finetuned/metrics.json`` exists (do not pass
``--force`` unless that run is formally invalidated).

Must already exist on otter disk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``data/gold/shake_annotation_sheet.csv`` (30 filled ``shake_label``)
* ``results/shake/pseudo_labels.csv``
* ``features/rgb16/gold_001.npz`` … ``gold_030.npz`` and
  ``features/rgb16/pseudo_00001.npz`` … ``pseudo_00080.npz``
* leakage PASS: shake pseudo ``video_id`` disjoint from gold DEV/TEST

Otter95 (RTX A4000, ``/scratch`` CUDA venv, **no Docker**)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/finetune_videomae_shake.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = [
    "--gold-csv", str(ROOT / "data" / "gold" / "shake_annotation_sheet.csv"),
    "--label-col", "shake_label",
    "--pseudo-labels", str(ROOT / "results" / "shake" / "pseudo_labels.csv"),
    "--out-dir", str(ROOT / "results" / "shake" / "videomae_finetuned"),
    "--unfreeze-blocks", "4",
]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.argv = [sys.argv[0]] + DEFAULTS + sys.argv[1:]
    from finetune_videomae import main as _main

    _main()


if __name__ == "__main__":
    main()
