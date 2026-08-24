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

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
SHAKE_PSEUDO = ROOT / "results" / "shake" / "pseudo_labels.csv"
SHAKE_OUT = ROOT / "results" / "shake" / "videomae_finetuned"
UNFREEZE_BLOCKS = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow re-scoring TEST (overwrites metrics.json). Default refuses "
             "if results/shake/videomae_finetuned/metrics.json exists.",
    )
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=UNFREEZE_BLOCKS,
        help="train the last N encoder blocks (+ head); "
             f"default {UNFREEZE_BLOCKS}",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="early-stopping patience on DEV F1",
    )
    parser.add_argument(
        "--flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="horizontal-flip augmentation on TRAIN "
             "(default on; use --no-flip to disable)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from finetune_videomae import main as finetune

    # Paths are kwargs, not argv: do not inject --gold-csv / --label-col.
    finetune(
        argv=[],
        gold_csv=SHAKE_GOLD,
        label_col="shake_label",
        pseudo_labels=SHAKE_PSEUDO,
        out_dir=SHAKE_OUT,
        force=args.force,
        unfreeze_blocks=args.unfreeze_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        flip=args.flip,
    )


if __name__ == "__main__":
    main()
