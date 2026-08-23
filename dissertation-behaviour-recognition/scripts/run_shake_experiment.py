#!/usr/bin/env python3
"""Head-shake RULE experiment on the same 30 gold windows (DEV 15 / TEST 15).

This script is the amplitude **rule only** (axis + threshold on DEV, TEST
once). The 1D CNN is a **separate** script: ``scripts/train_shake_cnn.py``.
Do not treat this file as the CNN trainer.

Does **not** overwrite nod artefacts in results/rule_*.json.
Does **not** invent shake labels or TEST F1.

You must fill shake_label (0/1) for all 30 rows in
data/gold/shake_annotation_sheet.csv, then:

    python scripts/run_shake_experiment.py

Protocol matches the nod rule: Savitzky–Golay amplitude on one EMOCA Euler
axis, axis + threshold frozen on DEV, TEST scored once. Writes results/shake/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_full_experiment import load_npz, rule_score  # noqa: E402
from src.metrics import binary_metrics  # noqa: E402

SHEET = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
GOLD_NPZ = ROOT / "features" / "gold"
OUT = ROOT / "results" / "shake"


def dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def load_shake_gold() -> pd.DataFrame:
    if not SHEET.exists():
        raise SystemExit(f"STOP: missing {SHEET}")
    df = pd.read_csv(SHEET, dtype=str)
    missing = df["shake_label"].isna() | (df["shake_label"].str.strip() == "")
    if missing.any():
        ids = df.loc[missing, "sample_id"].tolist()
        raise SystemExit(
            "STOP: fill shake_label (0 or 1) for every row before scoring.\n"
            f"  unfilled ({len(ids)}): {', '.join(ids)}\n"
            f"  file: {SHEET}\n"
            "  1 = clear left-right head SHAKE (no). 0 = no clear shake "
            "(nod-only clips are 0 unless a shake is also visible)."
        )
    bad = ~df["shake_label"].str.strip().isin(["0", "1"])
    if bad.any():
        raise SystemExit(
            "STOP: shake_label must be 0 or 1 only. Bad rows: "
            + ", ".join(df.loc[bad, "sample_id"].tolist())
        )
    gold = pd.read_csv(ROOT / "data" / "gold_annotations.csv")
    sh = df[["sample_id", "shake_label"]].copy()
    sh["shake_label"] = sh["shake_label"].astype(int)
    out = gold.merge(sh, on="sample_id", how="left", validate="one_to_one")
    if out["shake_label"].isna().any():
        raise SystemExit("STOP: shake sheet sample_id does not match gold_annotations.csv")
    out = out.copy()
    out["label"] = out["shake_label"].astype(int)
    n_pos = int(out["label"].sum())
    n_neg = int((out["label"] == 0).sum())
    print(f"shake gold: {len(out)} clips, {n_pos} shake / {n_neg} no-shake")
    print(
        "  DEV",
        int(out.loc[out.split == "DEV", "label"].sum()),
        "positives /",
        int((out.split == "DEV").sum()),
    )
    print(
        "  TEST",
        int(out.loc[out.split == "TEST", "label"].sum()),
        "positives /",
        int((out.split == "TEST").sum()),
    )
    if n_pos == 0 or n_neg == 0:
        raise SystemExit(
            "STOP: need at least one shake=1 and one shake=0 in the 30 labels."
        )
    return out


def main() -> None:
    gold = load_shake_gold()
    missing_npz = [
        sid
        for sid in gold["sample_id"]
        if not (GOLD_NPZ / f"{sid}.npz").exists()
    ]
    if missing_npz:
        raise SystemExit(f"STOP: missing pose npz: {missing_npz}")

    OUT.mkdir(parents=True, exist_ok=True)
    dev = gold[gold.split == "DEV"]
    scores = {ax: [] for ax in range(3)}
    labels: list[int] = []
    used: list[str] = []
    for r in dev.itertuples():
        rot = load_npz(GOLD_NPZ / f"{r.sample_id}.npz")["rotation_xyz"]
        labels.append(int(r.label))
        used.append(r.sample_id)
        for ax in range(3):
            scores[ax].append(rule_score(rot, ax))
    y = np.asarray(labels)
    best = None
    search_rows = []
    for ax in range(3):
        s = np.asarray(scores[ax], dtype=float)
        cands = np.unique(np.quantile(s, np.linspace(0.1, 0.9, 17)))
        for thr in cands:
            pred = (s >= thr).astype(int)
            m = binary_metrics(y, pred)
            row = {"axis": ax, "threshold": float(thr), **m}
            search_rows.append(row)
            key = (m["f1"], m["balanced_accuracy"])
            if best is None or key > best[0]:
                best = (key, row)
    assert best is not None
    pd.DataFrame(search_rows).to_csv(OUT / "rule_dev_threshold_search.csv", index=False)
    cfg = {
        "task": "head_shake",
        "chosen_rotation_axis": int(best[1]["axis"]),
        "axis_name": ["x", "y", "z"][int(best[1]["axis"])],
        "smoothing": "savgol_11_2",
        "min_movement_frames": 5,
        "max_movement_frames": 50,
        "selected_amplitude_threshold": float(best[1]["threshold"]),
        "dev_metrics": {
            k: best[1][k]
            for k in ("precision", "recall", "f1", "accuracy", "balanced_accuracy")
        },
        "n_dev_with_features": int(len(labels)),
        "dev_sample_ids": used,
        "note": (
            "Shake rule: same amplitude detector as nod, labels from "
            "shake_annotation_sheet.csv. Axis/threshold frozen on DEV. "
            "Hypothesis: yaw is Euler y, but DEV is allowed to pick x/y/z. "
            "Nod results were not overwritten."
        ),
    }
    dump_json(OUT / "rule_selected_config.json", cfg)
    print(
        f"DEV frozen: axis {cfg['axis_name']} "
        f"thr={cfg['selected_amplitude_threshold']:.3f}°  "
        f"DEV F1={cfg['dev_metrics']['f1']:.3f} (tuning only)"
    )

    def eval_split(split: str) -> dict:
        part = gold[gold.split == split]
        y_t, pred, sids, scs = [], [], [], []
        for r in part.itertuples():
            sc = rule_score(
                load_npz(GOLD_NPZ / f"{r.sample_id}.npz")["rotation_xyz"],
                int(cfg["chosen_rotation_axis"]),
            )
            y_t.append(int(r.label))
            pred.append(int(sc >= cfg["selected_amplitude_threshold"]))
            sids.append(r.sample_id)
            scs.append(sc)
        m = binary_metrics(np.array(y_t), np.array(pred))
        pd.DataFrame(
            {"sample_id": sids, "label": y_t, "score": scs, "pred": pred}
        ).to_csv(OUT / f"rule_{split.lower()}_predictions.csv", index=False)
        return m

    rule_dev = eval_split("DEV")
    rule_test = eval_split("TEST")
    dump_json(OUT / "rule_test_metrics.json", rule_test)
    print("TEST (scored once):", {k: rule_test[k] for k in ("precision", "recall", "f1", "tp", "fp", "tn", "fn")})
    print(f"wrote {OUT}/  (nod results/rule_test_metrics.json untouched)")


if __name__ == "__main__":
    main()
