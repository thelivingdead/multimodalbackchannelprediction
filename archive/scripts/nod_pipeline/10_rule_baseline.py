#!/usr/bin/env python3
"""Rule-based nod detector vs gold labels (event-level IoU). This is the baseline.

Predictions come from pitch cycles (03_detect_nod_candidates), not from a learned model.
Gold must be independent (human or generator), never the detector's own output.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_metrics import events_from_df, greedy_match, prf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def eval_split(pred: pd.DataFrame, gold: pd.DataFrame, video_ids: list[str], person: str = "p0") -> dict:
    tp = fp = fn = 0
    per = []
    for vid in video_ids:
        p = events_from_df(pred, vid, person)
        g = events_from_df(gold, vid, person)
        a, b, c = greedy_match(p, g, iou_thr=0.2)
        tp += a
        fp += b
        fn += c
        m = prf(a, b, c)
        m["video_id"] = vid
        m["n_pred"] = len(p)
        m["n_gold"] = len(g)
        per.append(m)
    overall = prf(tp, fp, fn)
    overall["n_videos"] = len(video_ids)
    overall["n_pred"] = int(tp + fp)
    overall["n_gold"] = int(tp + fn)
    overall["metric"] = "event_iou>=0.2"
    return overall, per


def main() -> None:
    t0 = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default=str(ROOT / "outputs" / "nod_pipeline" / "candidates.csv"))
    p.add_argument("--gold", default=str(ROOT / "outputs" / "nod_pipeline" / "gold_labels.csv"))
    p.add_argument("--splits", default=str(ROOT / "outputs" / "nod_pipeline" / "splits.json"))
    args = p.parse_args()

    pred = pd.read_csv(args.candidates)
    gold = pd.read_csv(args.gold)
    splits = json.loads(Path(args.splits).read_text())
    out = ROOT / "outputs" / "nod_pipeline"
    rows = []
    for name, vids in (("dev", splits["dev_videos"]), ("test", splits["test_videos"])):
        overall, per = eval_split(pred, gold, vids)
        overall["split"] = name
        overall["system"] = "rule_based_pitch_cycles"
        rows.append(overall)
        pd.DataFrame(per).to_csv(out / f"rule_baseline_{name}_per_video.csv", index=False)
        print(f"\n=== RULE BASELINE {name} ===")
        print(json.dumps(overall, indent=2))
    table = pd.DataFrame(rows)
    table.to_csv(out / "rule_baseline_metrics.csv", index=False)
    (out / "rule_baseline_metrics.json").write_text(json.dumps(rows, indent=2))
    elapsed = time.time() - t0
    with (out / "time_log.txt").open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  10_rule_baseline.py  {elapsed:.1f}s\n")
    print("\nSaved", out / "rule_baseline_metrics.csv")
    print("This is current-motion detection, not future-nod forecasting.")


if __name__ == "__main__":
    main()
