#!/usr/bin/env python3
"""Analyse durations / event counts and freeze video-level splits.

Gold 30 videos → 50% DEV + 50% TEST (15 / 15), seed 42.
Train videos are the unlabeled train_* clips (pseudo-labels later).

Never put a gold video in both DEV and TEST.
Train videos never appear in DEV/TEST.

Also prints a disk budget recommendation (keep derived 1-min clips, not full RealTalk).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_utils import list_clip_dirs, read_meta  # noqa: E402
from event_metrics import events_from_df  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def disk_report(root: Path) -> dict:
    import subprocess

    def du(path: Path) -> str:
        if not path.exists():
            return "0"
        r = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True)
        return r.stdout.split()[0] if r.returncode == 0 else "?"

    df = subprocess.run(["df", "-h", str(root)], capture_output=True, text=True)
    return {
        "project": du(root),
        "nod30": du(root / "data" / "nod30"),
        "tiny_subset": du(root / "data" / "tiny_subset"),
        "hf_cache": du(root / "data" / "hf_cache"),
        "df": df.stdout,
    }


def main() -> None:
    t0 = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default=str(ROOT / "data" / "nod30"))
    p.add_argument("--gold", default=str(ROOT / "outputs" / "nod_pipeline" / "gold_labels.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac-gold", type=float, default=0.5, help="half of gold → test")
    args = p.parse_args()

    clips = list_clip_dirs(Path(args.subset))
    gold = pd.read_csv(args.gold)
    gold_vids = sorted({str(read_meta(c)["video_id"]) for c in clips if read_meta(c).get("role") == "gold"})
    train_vids = sorted({str(read_meta(c)["video_id"]) for c in clips if read_meta(c).get("role") == "train"})
    if not gold_vids:
        # fallback: all non-train
        gold_vids = sorted({c.name for c in clips if c.name.startswith("gold_")})
    rng = np.random.default_rng(args.seed)
    order = np.array(gold_vids)
    rng.shuffle(order)
    n_test = int(round(len(order) * args.test_frac_gold))
    test_vids = sorted(order[:n_test].tolist())
    dev_vids = sorted(order[n_test:].tolist())
    overlap = set(test_vids) & set(dev_vids)
    if overlap:
        raise SystemExit(f"split leak {overlap}")
    if set(train_vids) & set(test_vids + dev_vids):
        raise SystemExit("train overlaps gold split")

    rows = []
    for c in clips:
        m = read_meta(c)
        vid = str(m["video_id"])
        split = "train" if vid in train_vids else "test" if vid in test_vids else "dev" if vid in dev_vids else "unused"
        n_gold = int((gold.video_id.astype(str) == vid).sum()) if split in ("dev", "test") else 0
        rows.append(
            {
                "video_id": vid,
                "split": split,
                "duration_s": float(m.get("duration_s", 60)),
                "n_frames": int(m.get("n_frames", 0)),
                "listener": m.get("listener", "p0"),
                "n_gold_nods": n_gold,
                "role": m.get("role"),
            }
        )
    inv = pd.DataFrame(rows)
    out = ROOT / "outputs" / "nod_pipeline"
    inv.to_csv(out / "split_inventory.csv", index=False)
    summary = {
        "seed": args.seed,
        "protocol": "gold videos 50% test / 50% dev; unlabeled clips = train (pseudo-labels)",
        "n_train_videos": int((inv.split == "train").sum()),
        "n_dev_videos": int((inv.split == "dev").sum()),
        "n_test_videos": int((inv.split == "test").sum()),
        "train_hours": float(inv.loc[inv.split == "train", "duration_s"].sum() / 3600),
        "dev_hours": float(inv.loc[inv.split == "dev", "duration_s"].sum() / 3600),
        "test_hours": float(inv.loc[inv.split == "test", "duration_s"].sum() / 3600),
        "dev_gold_nods": int(gold[gold.video_id.astype(str).isin(dev_vids)].shape[0]),
        "test_gold_nods": int(gold[gold.video_id.astype(str).isin(test_vids)].shape[0]),
        "train_videos": train_vids,
        "dev_videos": dev_vids,
        "test_videos": test_vids,
        "why_not_80_10_10": (
            "Only 30 videos have gold labels. Putting 80% of them in train would leave "
            "~3 test videos, which is too unstable. Gold is reserved for DEV/TEST (15/15). "
            "TRAIN is a larger unlabeled pool labelled by the rule detector."
        ),
        "disk": disk_report(ROOT),
    }
    (out / "splits.json").write_text(json.dumps(summary, indent=2))
    print(inv.groupby("split")[["duration_s", "n_gold_nods"]].agg({"duration_s": ["count", "sum"], "n_gold_nods": "sum"}))
    print("dev videos", len(dev_vids), "test videos", len(test_vids), "train videos", len(train_vids))
    print("disk", summary["disk"])
    elapsed = time.time() - t0
    with (out / "time_log.txt").open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  09_analyse_splits.py  {elapsed:.1f}s\n")
    print("Saved", out / "splits.json")


if __name__ == "__main__":
    main()
