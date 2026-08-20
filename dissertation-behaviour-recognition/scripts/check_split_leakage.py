#!/usr/bin/env python3
"""Split-leakage gate: fail loudly before any VideoMAE head training.

Asserts, using only committed artefacts (no network, no decoding):

1. gold DEV and gold TEST are disjoint by ``sample_id`` **and** by
   ``video_id`` (no video crosses the DEV/TEST boundary).
2. No pseudo ``sample_id`` collides with any gold ``sample_id``.
3. No pseudo ``video_id`` appears among gold TEST videos (pseudo = TRAIN
   split) — nor among gold DEV videos.
4. Every selected TRAIN/DEV/TEST clip actually has its expected artefacts
   where those artefacts already exist (warn-only: coverage is Step 3/4's
   job, this gate is about leakage).

Exit 0 prints PASS with the counts; any violation exits non-zero listing
every offender. Run this before ``train_videomae_head.py`` — if it fails,
training must not start.

Lab invocation::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    source ../.venv/bin/activate
    python scripts/check_split_leakage.py
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLD_CSV = ROOT / "data" / "gold_annotations.csv"
PSEUDO_DIR = ROOT / "features" / "pseudo"
PSEUDO_LABELS = ROOT / "results" / "pseudo_labels.csv"
RGB16_DIR = ROOT / "features" / "rgb16"
EMB_DIR = ROOT / "data" / "features" / "videomae"


def npz_video_id(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        arr = np.load(io.BytesIO(z.read("video_id.npy")), allow_pickle=True)
    if hasattr(arr, "item"):
        arr = arr.item()
    if isinstance(arr, bytes):
        arr = arr.decode()
    return str(arr).strip()


def main() -> None:
    with GOLD_CSV.open(newline="") as fh:
        gold = list(csv.DictReader(fh))
    dev = [r for r in gold if r["split"].upper() == "DEV"]
    tes = [r for r in gold if r["split"].upper() == "TEST"]
    if not dev or not tes:
        raise SystemExit("STOP: gold CSV must contain both DEV and TEST rows.")

    dev_ids = {r["sample_id"] for r in dev}
    tes_ids = {r["sample_id"] for r in tes}
    dev_vids = {r["video_id"] for r in dev}
    tes_vids = {r["video_id"] for r in tes}

    pseudo_npz = sorted(PSEUDO_DIR.glob("*.npz"))
    pseudo_ids = {p.stem for p in pseudo_npz}
    pseudo_vids = {p.stem: npz_video_id(p) for p in pseudo_npz}
    if PSEUDO_LABELS.exists():
        with PSEUDO_LABELS.open(newline="") as fh:
            labelled = {r["sample_id"] for r in csv.DictReader(fh)}
        if labelled != pseudo_ids:
            print(
                "WARNING: pseudo_labels.csv ids and features/pseudo/*.npz "
                f"differ ({len(labelled)} vs {len(pseudo_ids)}): "
                f"labels-only={sorted(labelled - pseudo_ids)[:5]}, "
                f"npz-only={sorted(pseudo_ids - labelled)[:5]}"
            )

    failures: list[str] = []

    both = sorted(dev_ids & tes_ids)
    if both:
        failures.append(f"sample_id in BOTH DEV and TEST: {both}")
    vids_both = sorted(dev_vids & tes_vids)
    if vids_both:
        failures.append(f"video_id crosses DEV/TEST: {vids_both}")

    id_clash = sorted(pseudo_ids & (dev_ids | tes_ids))
    if id_clash:
        failures.append(f"pseudo sample_id collides with gold: {id_clash}")
    in_test = sorted(v for v in pseudo_vids.values() if v in tes_vids)
    if in_test:
        failures.append(
            f"pseudo video_id present in gold TEST (train->test leak): "
            f"{in_test}"
        )
    in_dev = sorted(v for v in pseudo_vids.values() if v in dev_vids)
    if in_dev:
        failures.append(f"pseudo video_id present in gold DEV: {in_dev}")

    # artefact coverage warnings (not leakage; never fatal here)
    for name, folder in (("rgb16", RGB16_DIR), ("embeddings", EMB_DIR)):
        if folder.exists():
            have = {p.stem for p in folder.glob("*.npz")}
            missing_gold = sorted((dev_ids | tes_ids) - have)
            missing_pseudo = sorted(pseudo_ids - have)
            if missing_gold or missing_pseudo:
                print(
                    f"NOTE: {name}: missing for {len(missing_gold)} gold / "
                    f"{len(missing_pseudo)} pseudo clips (coverage is the "
                    "fetch/extract steps' job)"
                )

    print("---- split-leakage gate ----")
    print(f"gold: {len(dev)} DEV / {len(tes)} TEST; "
          f"{len(dev_vids)} / {len(tes_vids)} unique videos")
    print(f"pseudo: {len(pseudo_ids)} clips, "
          f"{len(set(pseudo_vids.values()))} unique videos")
    if failures:
        raise SystemExit(
            "FAIL: split leakage detected — DO NOT TRAIN:\n- "
            + "\n- ".join(failures)
        )
    print("PASS: DEV/TEST disjoint (ids and videos); no pseudo clip or video "
          "in TEST (or DEV); no id collisions.")


if __name__ == "__main__":
    main()
