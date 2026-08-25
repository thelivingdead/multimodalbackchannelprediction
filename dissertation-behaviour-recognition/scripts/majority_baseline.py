#!/usr/bin/env python3
"""Always-positive / always-negative clip baselines from gold CSVs (CPU, no GPU).

Computes TEST P/R/F1 for predicting all 1s and all 0s. Does not invent scores
and does not write locked VideoMAE / CNN / rule TEST directories.

Also (shake only) compares locked VideoMAE TEST *predictions* clip-by-clip so
the identical TP6/FP7/TN1/FN1 tables can be diagnosed as matching *counts*,
not matching clip decisions.

Otter95 (CPU; ``/scratch`` venv, **no Docker**)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/majority_baseline.py

Mac (project venv)::

    cd "/Users/divyabisht/Downloads/Msc Dissertation Divya/dissertation-behaviour-recognition"
    OMP_NUM_THREADS=1 ../.venv/bin/python scripts/majority_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clip_metrics import always_predict, clip_binary_metrics  # noqa: E402
from src.utils import dump_json  # noqa: E402

SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
NOD_GOLD = ROOT / "data" / "gold_annotations.csv"
SHAKE_OUT = ROOT / "results" / "shake" / "majority_baseline"
NOD_OUT = ROOT / "results" / "majority_baseline"
LOCKED = {
    "shake_frozen": ROOT / "results" / "shake" / "videomae_frozen_head",
    "shake_finetuned": ROOT / "results" / "shake" / "videomae_finetuned",
    "shake_rule": ROOT / "results" / "shake" / "rule_test_metrics.json",
    "nod_frozen": ROOT / "results" / "videomae_frozen_head" / "metrics.json",
    "nod_finetuned": ROOT / "results" / "videomae_finetuned" / "metrics.json",
}


def _idcol(df: pd.DataFrame) -> str:
    for c in ("sample_id", "clip_id"):
        if c in df.columns:
            return c
    raise SystemExit(f"STOP: no sample_id/clip_id in columns {list(df.columns)}")


def gold_test(path: Path, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"STOP: gold CSV missing: {path}")
    df = pd.read_csv(path)
    if "split" not in df.columns or label_col not in df.columns:
        raise SystemExit(
            f"STOP: {path} needs split + {label_col}. Columns: {list(df.columns)}"
        )
    df["split"] = df["split"].astype(str).str.upper()
    tes = df[df.split == "TEST"].sort_values("sample_id").copy()
    y = pd.to_numeric(tes[label_col], errors="coerce")
    if y.isna().any():
        bad = tes.loc[y.isna(), "sample_id"].tolist()
        raise SystemExit(f"STOP: empty {label_col} on TEST {bad}")
    tes[label_col] = y.astype(int)
    return tes


def load_test_preds(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cid = _idcol(df)
    df = df.rename(columns={cid: "sample_id"})
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.upper() == "TEST"]
    need = {"sample_id", "pred"}
    if not need <= set(df.columns):
        return None
    out = df[["sample_id", "pred"]].copy()
    if "prob" in df.columns:
        out["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    if "label" in df.columns:
        out["label"] = pd.to_numeric(df["label"], errors="coerce")
    out["sample_id"] = out["sample_id"].astype(str)
    out["pred"] = out["pred"].astype(int)
    return out.sort_values("sample_id")


def read_locked_f1(path: Path) -> float | None:
    if not path.exists():
        return None
    obj = json.loads(path.read_text())
    tm = obj.get("test_metrics") or {}
    if "f1" in tm:
        return float(tm["f1"])
    if "f1" in obj:
        return float(obj["f1"])
    return None


def pack_task(
    *,
    task: str,
    gold_csv: Path,
    label_col: str,
    tes: pd.DataFrame,
    comparisons: dict,
) -> dict:
    y = tes[label_col].to_numpy(int)
    pos = always_predict(y, 1)
    neg = always_predict(y, 0)
    return {
        "task": task,
        "model": "always-constant baseline (not a trained system)",
        "script": Path(__file__).name,
        "gold_csv": str(gold_csv.relative_to(ROOT)),
        "label_col": label_col,
        "split": "TEST",
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "always_positive": pos,
        "always_negative": neg,
        "selection_rule": "no training; TEST labels from gold only",
        "comparisons_read_only": comparisons,
        "note": (
            "Always-positive is the majority class on nod TEST (10/15) and "
            "the *minority* class on shake TEST (7/15). It is still the "
            "relevant RGB collapse check because VideoMAE predicted almost "
            "all 1s."
        ),
    }


def diagnose_shake_videomae(tes: pd.DataFrame, out_dir: Path) -> dict:
    frozen = load_test_preds(
        ROOT / "results" / "shake" / "videomae_frozen_head" / "predictions.csv"
    )
    ft = load_test_preds(
        ROOT / "results" / "shake" / "videomae_finetuned" / "predictions.csv"
    )
    gold = tes[["sample_id", "shake_label"]].rename(
        columns={"shake_label": "gold"}
    )
    gold["sample_id"] = gold["sample_id"].astype(str)
    if frozen is None or ft is None:
        return {
            "predictions_identical": None,
            "note": "one or both VideoMAE predictions.csv missing; skipped",
        }
    a = frozen.rename(columns={"pred": "frozen_pred", "prob": "frozen_prob",
                               "label": "frozen_csv_label"})
    b = ft.rename(columns={"pred": "finetuned_pred", "prob": "finetuned_prob",
                           "label": "finetuned_csv_label"})
    m = gold.merge(a, on="sample_id", how="left").merge(b, on="sample_id", how="left")
    if m["frozen_pred"].isna().any() or m["finetuned_pred"].isna().any():
        missing = m.loc[
            m["frozen_pred"].isna() | m["finetuned_pred"].isna(), "sample_id"
        ].tolist()
        raise SystemExit(f"STOP: VideoMAE TEST preds missing for {missing}")
    pred_same = bool((m["frozen_pred"] == m["finetuned_pred"]).all())
    prob_same = bool(
        np.allclose(
            m["frozen_prob"].to_numpy(float),
            m["finetuned_prob"].to_numpy(float),
            atol=1e-8,
            equal_nan=True,
        )
    )
    y = m["gold"].to_numpy(int)
    frz_m = clip_binary_metrics(y, m["frozen_pred"].to_numpy(int))
    ft_m = clip_binary_metrics(y, m["finetuned_pred"].to_numpy(int))
    counts_same = {k: frz_m[k] for k in ("tp", "fp", "tn", "fn")} == {
        k: ft_m[k] for k in ("tp", "fp", "tn", "fn")
    }
    disagree = m.loc[
        m["frozen_pred"] != m["finetuned_pred"],
        ["sample_id", "gold", "frozen_pred", "frozen_prob",
         "finetuned_pred", "finetuned_prob"],
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    m.to_csv(out_dir / "videomae_test_clip_compare.csv", index=False)
    return {
        "n_test": int(len(m)),
        "pred_labels_identical": pred_same,
        "probabilities_identical": prob_same,
        "confusion_counts_identical": bool(counts_same),
        "n_clips_pred_disagree": int(len(disagree)),
        "disagree_sample_ids": disagree["sample_id"].tolist(),
        "frozen_confusion": {k: frz_m[k] for k in
                             ("precision", "recall", "f1", "tp", "fp", "tn", "fn")},
        "finetuned_confusion": {k: ft_m[k] for k in
                                ("precision", "recall", "f1", "tp", "fp", "tn", "fn")},
        "note": (
            "TEST rows of the two predictions.csv files were compared. "
            "Matching 2×2 *counts* at n=15 does not imply matching clip "
            "decisions or matching probabilities."
        ),
    }


def main() -> None:
    shake = gold_test(SHAKE_GOLD, "shake_label")
    nod = gold_test(NOD_GOLD, "label")

    shake_cmp = {
        "videomae_frozen_test_f1": read_locked_f1(
            LOCKED["shake_frozen"] / "metrics.json"
        ),
        "videomae_finetuned_test_f1": read_locked_f1(
            LOCKED["shake_finetuned"] / "metrics.json"
        ),
        "pose_rule_test_f1": read_locked_f1(LOCKED["shake_rule"]),
    }
    nod_cmp = {
        "videomae_frozen_test_f1": read_locked_f1(LOCKED["nod_frozen"]),
        "videomae_finetuned_test_f1": read_locked_f1(LOCKED["nod_finetuned"]),
    }

    shake_metrics = pack_task(
        task="head_shake",
        gold_csv=SHAKE_GOLD,
        label_col="shake_label",
        tes=shake,
        comparisons=shake_cmp,
    )
    shake_metrics["videomae_locked_test_compare"] = diagnose_shake_videomae(
        shake, SHAKE_OUT
    )
    dump_json(SHAKE_OUT / "metrics.json", shake_metrics)

    nod_metrics = pack_task(
        task="head_nod",
        gold_csv=NOD_GOLD,
        label_col="label",
        tes=nod,
        comparisons=nod_cmp,
    )
    dump_json(NOD_OUT / "metrics.json", nod_metrics)

    def line(name: str, blob: dict) -> None:
        ap = blob["always_positive"]
        an = blob["always_negative"]
        print(
            f"{name} TEST n={blob['n']} ({blob['n_pos']} pos / {blob['n_neg']} neg)\n"
            f"  always-1  P {ap['precision']:.2f}  R {ap['recall']:.2f}  "
            f"F1 {ap['f1']:.2f}  "
            f"(TP{ap['tp']} FP{ap['fp']} TN{ap['tn']} FN{ap['fn']})\n"
            f"  always-0  P {an['precision']:.2f}  R {an['recall']:.2f}  "
            f"F1 {an['f1']:.2f}"
        )

    print("=====================================")
    line("Shake", shake_metrics)
    print(
        "  locked read-only: VideoMAE F1 "
        f"{shake_cmp['videomae_finetuned_test_f1']}  "
        f"pose rule F1 {shake_cmp['pose_rule_test_f1']}"
    )
    cmp = shake_metrics["videomae_locked_test_compare"]
    print(
        "  frozen vs fine-tune TEST: "
        f"pred_identical={cmp.get('pred_labels_identical')}  "
        f"prob_identical={cmp.get('probabilities_identical')}  "
        f"counts_identical={cmp.get('confusion_counts_identical')}  "
        f"n_disagree={cmp.get('n_clips_pred_disagree')}  "
        f"{cmp.get('disagree_sample_ids')}"
    )
    line("Nod", nod_metrics)
    print(f"wrote {SHAKE_OUT / 'metrics.json'}")
    print(f"wrote {NOD_OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
