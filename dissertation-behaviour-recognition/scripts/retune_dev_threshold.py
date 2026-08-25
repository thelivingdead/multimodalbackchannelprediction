#!/usr/bin/env python3
"""New protocol: retune the probability threshold on DEV, score TEST once.

The locked shake VideoMAE runs already swept 13 F1 thresholds and chose 0.5
because DEV F1 was stuck at 0.833 (near always-positive on 10/5). This
script does **not** overwrite those ``metrics.json`` files.

Default criterion is **balanced accuracy** (the scientifically new choice
once F1 is invariant). Optionally ``--criterion f1`` with a fine grid.

Sources, in order:
1. Existing ``predictions.csv`` with DEV + TEST rows and ``prob``
   (fine-tuned shake has this — runnable on Mac).
2. Else ``best_model.pt`` + embeddings (frozen head; otter, pt is gitignored).

Writes e.g. ``results/shake/videomae_finetuned_dev_threshold/``.

Otter95::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/retune_dev_threshold.py

    # frozen head (needs best_model.pt + embeddings on otter):
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/retune_dev_threshold.py \\
        --source-dir results/shake/videomae_frozen_head \\
        --out-dir results/shake/videomae_frozen_head_dev_threshold
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.clip_metrics import clip_binary_metrics  # noqa: E402
from src.utils import dump_json  # noqa: E402

SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
DEFAULT_SRC = ROOT / "results" / "shake" / "videomae_finetuned"
DEFAULT_OUT = ROOT / "results" / "shake" / "videomae_finetuned_dev_threshold"
SEED = 42


def _idcol(df: pd.DataFrame) -> str:
    for c in ("sample_id", "clip_id"):
        if c in df.columns:
            return c
    raise SystemExit(f"STOP: no id column in {list(df.columns)}")


def gold_split(gold_csv: Path, label_col: str, split: str) -> pd.DataFrame:
    g = pd.read_csv(gold_csv)
    g["split"] = g["split"].astype(str).str.upper()
    part = g[g.split == split.upper()].sort_values("sample_id").copy()
    y = pd.to_numeric(part[label_col], errors="coerce")
    if y.isna().any():
        raise SystemExit(f"STOP: empty {label_col} on {split}")
    part[label_col] = y.astype(int)
    part["sample_id"] = part["sample_id"].astype(str)
    return part


def load_pred_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cid = _idcol(df)
    df = df.rename(columns={cid: "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str)
    if "prob" not in df.columns:
        return None
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.upper()
    return df


def sweep_threshold(y: np.ndarray, prob: np.ndarray, criterion: str):
    thr_best, score_best, f1_at, bal_at = 0.5, -1.0, -1.0, -1.0
    rows = []
    grid = np.unique(
        np.concatenate(
            [np.linspace(0.05, 0.95, 91), np.unique(prob)]
        )
    )
    for t in grid:
        m = clip_binary_metrics(y, (prob >= t).astype(int))
        score = float(m[criterion])
        row = {"threshold": float(t), **m}
        rows.append(row)
        better = score > score_best
        tie = (
            score == score_best
            and criterion == "f1"
            and m["balanced_accuracy"] > bal_at
        )
        if better or tie:
            thr_best = float(t)
            score_best = score
            f1_at = float(m["f1"])
            bal_at = float(m["balanced_accuracy"])
    return thr_best, score_best, f1_at, bal_at, pd.DataFrame(rows)


def infer_frozen_probs(source_dir: Path, gold: pd.DataFrame, label_col: str):
    """Rebuild MLP, load best_model.pt, score gold DEV+TEST embeddings."""
    pt = source_dir / "best_model.pt"
    emb_dir = ROOT / "data" / "features" / "videomae"
    if not pt.exists():
        raise SystemExit(
            f"STOP: {pt} missing (gitignored). On otter, where the locked "
            "run wrote best_model.pt, re-run this script with the same "
            "--source-dir. Mac cannot invent frozen-head DEV probabilities."
        )
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit(f"STOP: torch required to load {pt} ({exc})") from exc

    xs, ids, missing = [], [], []
    for sid in gold["sample_id"].astype(str):
        path = emb_dir / f"{sid}.npz"
        if not path.exists():
            missing.append(sid)
            continue
        with np.load(path, allow_pickle=True) as z:
            xs.append(np.asarray(z["embedding"], dtype=np.float32).reshape(-1))
            ids.append(sid)
    if missing:
        raise SystemExit(
            f"STOP: missing embeddings for {missing}. "
            "Run extract_videomae_embeddings.py on otter first."
        )
    X = np.stack(xs)
    dim = int(X.shape[1])
    model = nn.Sequential(
        nn.Linear(dim, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )
    state = torch.load(pt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X)).squeeze(-1).numpy()
    prob = 1 / (1 + np.exp(-logits))
    ymap = dict(zip(gold["sample_id"].astype(str), gold[label_col].astype(int)))
    split_map = dict(zip(gold["sample_id"].astype(str),
                         gold["split"].astype(str).str.upper()))
    return pd.DataFrame(
        {
            "sample_id": ids,
            "prob": prob,
            "label": [ymap[s] for s in ids],
            "split": [split_map[s] for s in ids],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gold-csv", type=Path, default=SHAKE_GOLD)
    parser.add_argument("--label-col", default="shake_label")
    parser.add_argument(
        "--criterion",
        choices=("balanced_accuracy", "f1"),
        default="balanced_accuracy",
        help="DEV selection (default balanced_accuracy: F1 was stuck at 0.833)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import check_split_leakage

    src = args.source_dir if args.source_dir.is_absolute() else ROOT / args.source_dir
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    gold_csv = args.gold_csv if args.gold_csv.is_absolute() else ROOT / args.gold_csv
    src, out, gold_csv = src.resolve(), out.resolve(), gold_csv.resolve()

    check_split_leakage.assert_unlocked_out_dir(out)
    if (out / "metrics.json").exists() and not args.force:
        raise SystemExit(
            f"STOP: {out / 'metrics.json'} already exists — TEST scored once "
            "under this threshold protocol. Do not --force to shop scores."
        )
    if out == src:
        raise SystemExit("STOP: out-dir must be a NEW directory, not the source.")

    gold = pd.read_csv(gold_csv)
    gold["split"] = gold["split"].astype(str).str.upper()
    if args.label_col not in gold.columns:
        raise SystemExit(f"STOP: {gold_csv} has no {args.label_col}")

    preds = load_pred_csv(src / "predictions.csv")
    have_dev = (
        preds is not None
        and "split" in preds.columns
        and (preds["split"] == "DEV").any()
        and (preds["split"] == "TEST").any()
    )
    if have_dev:
        frame = preds
        source_note = "predictions.csv DEV+TEST probabilities (weights not reloaded)"
    else:
        frame = infer_frozen_probs(src, gold, args.label_col)
        source_note = "re-inferred from best_model.pt + embeddings"

    frame = frame.copy()
    if "split" in frame.columns:
        frame["split"] = frame["split"].astype(str).str.upper()
        frame = frame[frame["split"].isin(["DEV", "TEST"])].copy()
    ymap = dict(zip(gold["sample_id"].astype(str),
                    pd.to_numeric(gold[args.label_col], errors="coerce")))
    frame["gold"] = frame["sample_id"].map(ymap)
    if frame["gold"].isna().any():
        bad = frame.loc[frame["gold"].isna(), "sample_id"].tolist()
        raise SystemExit(f"STOP: pred ids not in gold: {bad}")
    frame["gold"] = frame["gold"].astype(int)

    dev = frame[frame["split"] == "DEV"]
    tes = frame[frame["split"] == "TEST"]
    if len(dev) < 3 or len(tes) < 3:
        raise SystemExit(
            f"STOP: need DEV and TEST probs (got {len(dev)}/{len(tes)}). "
            "Frozen predictions.csv is TEST-only — run on otter with "
            "best_model.pt."
        )

    y_dv = dev["gold"].to_numpy(int)
    p_dv = dev["prob"].to_numpy(float)
    y_te = tes["gold"].to_numpy(int)
    p_te = tes["prob"].to_numpy(float)

    thr, score, f1_at, bal_at, sweep = sweep_threshold(
        y_dv, p_dv, args.criterion
    )
    pred_dv = (p_dv >= thr).astype(int)
    pred_te = (p_te >= thr).astype(int)
    dev_m = clip_binary_metrics(y_dv, pred_dv)
    tes_m = clip_binary_metrics(y_te, pred_te)

    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "dev_threshold_search.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": list(dev["sample_id"]) + list(tes["sample_id"]),
            "split": ["DEV"] * len(dev) + ["TEST"] * len(tes),
            "label": np.concatenate([y_dv, y_te]),
            "prob": np.concatenate([p_dv, p_te]),
            "pred": np.concatenate([pred_dv, pred_te]),
        }
    ).to_csv(out / "predictions.csv", index=False)

    src_metrics = {}
    if (src / "metrics.json").exists():
        src_metrics = json.loads((src / "metrics.json").read_text())

    metrics = {
        "task": src_metrics.get("task", "head_shake"),
        "model": "DEV-retuned threshold on frozen probabilities (new protocol)",
        "script": Path(__file__).name,
        "source_dir": str(src),
        "out_dir": str(out),
        "gold_csv": str(gold_csv),
        "label_col": args.label_col,
        "prob_source": source_note,
        "criterion": args.criterion,
        "seed": SEED,
        "dev_n": int(len(y_dv)),
        "test_n": int(len(y_te)),
        "locked_source_threshold": src_metrics.get("dev_probability_threshold"),
        "locked_source_dev_f1": src_metrics.get("dev_f1"),
        "dev_probability_threshold": float(thr),
        "dev_selected_score": float(score),
        "dev_f1_at_selected": float(f1_at),
        "dev_balanced_accuracy_at_selected": float(bal_at),
        "dev_metrics": dev_m,
        "test_metrics": tes_m,
        "selection_rule": (
            f"threshold by DEV {args.criterion} on stored/re-inferred "
            "probabilities; TEST scored once in this new directory"
        ),
        "note": (
            "Locked VideoMAE metrics.json were not overwritten. "
            "0.5 in the original run was already the 13-point F1 winner; "
            "this protocol changes the DEV criterion / grid."
        ),
    }
    dump_json(out / "metrics.json", metrics)
    print(
        f"DEV {args.criterion}={score:.3f}  threshold={thr:.4f}  "
        f"DEV F1={f1_at:.3f}  DEV bal={bal_at:.3f}"
    )
    print(
        f"TEST (once, new dir): P {tes_m['precision']:.2f}  "
        f"R {tes_m['recall']:.2f}  F1 {tes_m['f1']:.2f}  "
        f"(TP{tes_m['tp']} FP{tes_m['fp']} TN{tes_m['tn']} FN{tes_m['fn']})"
    )
    print(f"wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
