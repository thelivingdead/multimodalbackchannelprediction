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

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
SHAKE_PSEUDO = ROOT / "results" / "shake" / "pseudo_labels.csv"
SHAKE_OUT = ROOT / "results" / "shake" / "videomae_frozen_head"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow re-scoring TEST (overwrites metrics.json). Default refuses "
             "if results/shake/videomae_frozen_head/metrics.json exists.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_videomae_head import main as train_head

    # Paths are kwargs, not argv: the nod parser must not see --gold-csv here.
    train_head(
        argv=[],
        gold_csv=SHAKE_GOLD,
        label_col="shake_label",
        pseudo_labels=SHAKE_PSEUDO,
        out_dir=SHAKE_OUT,
        force=args.force,
    )


if __name__ == "__main__":
    main()
