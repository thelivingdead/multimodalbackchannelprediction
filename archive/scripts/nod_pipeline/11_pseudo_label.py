#!/usr/bin/env python3
"""Pseudo-label TRAIN (and optionally DEV) with the rule-based detector.

Writes noisy labels used to train the classifier. Gold TEST is never pseudo-labelled
for training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    t0 = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default=str(ROOT / "outputs" / "nod_pipeline" / "candidates.csv"))
    p.add_argument("--splits", default=str(ROOT / "outputs" / "nod_pipeline" / "splits.json"))
    p.add_argument("--include-dev-pseudo", action="store_true", help="also pseudo-label DEV (not recommended if DEV is for HP)")
    args = p.parse_args()

    cand = pd.read_csv(args.candidates)
    splits = json.loads(Path(args.splits).read_text())
    train_set = set(splits["train_videos"])
    if args.include_dev_pseudo:
        train_set |= set(splits["dev_videos"])
    # keep listener p0 (synthetic listener). Drop p1 to reduce speaker noise.
    sub = cand[cand.video_id.astype(str).isin(train_set)].copy()
    sub = sub[sub.person.astype(str) == "p0"]
    sub["label"] = "nod"
    sub["label_source"] = "rule_pseudo"
    out = ROOT / "outputs" / "nod_pipeline" / "pseudo_labels_train.csv"
    sub.to_csv(out, index=False)
    print(f"pseudo nod events: {len(sub)} on {sub.video_id.nunique()} videos")
    print("videos", sorted(sub.video_id.unique())[:12], "...")
    elapsed = time.time() - t0
    with (ROOT / "outputs" / "nod_pipeline" / "time_log.txt").open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  11_pseudo_label.py  {elapsed:.1f}s  n={len(sub)}\n")
    print("Wrote", out)


if __name__ == "__main__":
    main()
