#!/usr/bin/env python3
"""09 — Small grid search for nod rule on PILOT/DEV only. Never uses GOLD TEST."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotations import gold_nods, load_events  # noqa: E402
from src.data import default_pilot_dir, list_clip_dirs, read_meta  # noqa: E402
from src.metrics import event_metrics  # noqa: E402
from src.rules.nod import NodRule  # noqa: E402
from src.utils import dump_json, load_yaml  # noqa: E402


def load_dev_ids() -> set[str]:
    p = ROOT / "data" / "splits" / "gold_dev.txt"
    if p.exists() and p.read_text().strip():
        return {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}
    return {str(read_meta(c)["video_id"]) for c in list_clip_dirs(default_pilot_dir())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--behaviour", default="nod")
    p.add_argument("--split", default="pilot")
    p.parse_args()
    test_ids = set()
    tp = ROOT / "data" / "splits" / "gold_test.txt"
    if tp.exists():
        test_ids = {ln.strip() for ln in tp.read_text().splitlines() if ln.strip()}
    dev_ids = load_dev_ids()
    if test_ids & dev_ids:
        raise SystemExit("LEAK: test ids in dev")
    gold = [e for e in gold_nods(load_events(ROOT / "data" / "gold" / "events.csv")) if e.video_id in dev_ids]
    if not gold:
        raise SystemExit("No class-1 gold nods on DEV. Annotate first.")

    poses = {}
    for c in list_clip_dirs(default_pilot_dir()):
        vid = str(read_meta(c)["video_id"])
        if vid not in dev_ids:
            continue
        hp = ROOT / "data" / "headpose" / f"{vid}.csv"
        if hp.exists():
            poses[vid] = (pd.read_csv(hp), str(read_meta(c).get("listener", "p0")))

    grid = []
    for min_range in (1.5, 2.5, 4.0):
        for min_dur in (0.20, 0.25, 0.35):
            rule = NodRule(min_range_deg=min_range, min_dur=min_dur)
            pred = []
            for vid, (pose, person) in poses.items():
                pred.extend(rule.detect(pose, vid, person))
            m = event_metrics(pred, gold)
            grid.append(
                {
                    "min_range_deg": min_range,
                    "min_dur": min_dur,
                    "event_f1_iou0.30": m["primary_event_f1"],
                    "precision": m["iou_0p30"]["precision"],
                    "recall": m["iou_0p30"]["recall"],
                    "n_pred": len(pred),
                    "n_gold": len(gold),
                }
            )
    df = pd.DataFrame(grid).sort_values("event_f1_iou0.30", ascending=False)
    dest = ROOT / "results" / "rule_nod_dev_grid.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    best = df.iloc[0].to_dict()
    base = load_yaml(ROOT / "configs" / "rule_nod.yaml")
    base["min_range_deg"] = float(best["min_range_deg"])
    base["min_dur"] = float(best["min_dur"])
    base["tuned_on"] = "pilot_dev"
    base["seed"] = 42
    best_path = ROOT / "configs" / "rule_nod_best.yaml"
    best_path.write_text(yaml.safe_dump(base, sort_keys=False))
    dump_json(ROOT / "results" / "rule_nod_dev_best.json", best)
    print(df.to_string(index=False))
    print("Best →", best_path)
    print("Grid CSV →", dest)


if __name__ == "__main__":
    main()
