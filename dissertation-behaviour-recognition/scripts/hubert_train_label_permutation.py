#!/usr/bin/env python3
"""TRAIN-label permutation test for the frozen HuBERT DEV experiment.

Loads cached HuBERT embeddings from disk. Does not extract embeddings.
Does not score GOLD TEST. Does not retune thresholds. Does not regenerate
pseudo-labels. Does not overwrite hubert_dev_metrics.json or predictions.

    python scripts/hubert_train_label_permutation.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clip_metrics import clip_binary_metrics  # noqa: E402

OUT_DIR = ROOT / "results" / "hubert_dev"
EMB_DIR = OUT_DIR / "embeddings"
PSEUDO_CSV = ROOT / "results" / "pseudo_labels.csv"
GOLD_CSV = ROOT / "data" / "gold_annotations.csv"
ACTUAL_JSON = OUT_DIR / "hubert_dev_metrics.json"
METRICS_CSV = OUT_DIR / "permutation_metrics.csv"
SUMMARY_JSON = OUT_DIR / "permutation_summary.json"

SEED = 42
PCA_DIM = 16
N_PERM = 1000
THRESHOLD = 0.5
EXPECTED_DEV = [f"gold_{i:03d}" for i in range(1, 16)]
TEST_ID_RE = re.compile(r"^gold_(0*(1[6-9]|2[0-9]|30))$")
TEST_MSG = "REFUSING TO SCORE GOLD TEST FOR HUBERT DEVELOPMENT EXPERIMENT"


def refuse_gold_test_id(sample_id: str) -> None:
    sid = str(sample_id)
    if TEST_ID_RE.match(sid) or sid in {f"gold_{i:03d}" for i in range(16, 31)}:
        print(TEST_MSG)
        raise SystemExit(TEST_MSG)


def load_embedding(sample_id: str) -> np.ndarray:
    refuse_gold_test_id(sample_id)
    path = EMB_DIR / f"{sample_id}.npz"
    if not path.is_file():
        raise SystemExit(
            f"STOP: cached embedding missing for {sample_id} at {path}. "
            "HUBERT_EMBEDDINGS_REEXTRACTED = NO (refusing to extract)."
        )
    with np.load(path, allow_pickle=True) as z:
        arr = np.asarray(z["embedding"], dtype=np.float32).reshape(-1)
    if arr.shape != (768,) or np.isnan(arr).any() or np.isinf(arr).any():
        raise SystemExit(f"STOP: bad cached embedding for {sample_id}")
    return arr


def stack_embeddings(ids: list[str]) -> np.ndarray:
    return np.stack([load_embedding(s) for s in ids], axis=0)


def train_ids_and_labels() -> tuple[list[str], np.ndarray]:
    rows = list(csv.DictReader(PSEUDO_CSV.open()))
    ids = [str(r["sample_id"]) for r in rows]
    y = np.asarray([int(r["pseudo_label"]) for r in rows], dtype=int)
    for sid in ids:
        refuse_gold_test_id(sid)
    if len(ids) != 80:
        raise SystemExit(f"STOP: expected 80 TRAIN ids, got {len(ids)}")
    return ids, y


def dev_ids_and_labels() -> tuple[list[str], np.ndarray]:
    gold = list(csv.DictReader(GOLD_CSV.open()))
    by_id = {}
    for rec in gold:
        sid = str(rec["sample_id"])
        split = str(rec["split"]).strip().upper()
        if split == "TEST":
            continue
        if split != "DEV":
            continue
        refuse_gold_test_id(sid)
        by_id[sid] = int(rec["label"])
    ids = list(EXPECTED_DEV)
    if ids != sorted(by_id.keys()):
        raise SystemExit(f"STOP: DEV ids {sorted(by_id.keys())} != {ids}")
    y = np.asarray([by_id[s] for s in ids], dtype=int)
    return ids, y


def make_pipeline(n_pca: int):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca, random_state=SEED)),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def eval_at_05(x_tr, y_tr, x_dv, y_dv, n_pca: int) -> dict:
    pipe = make_pipeline(n_pca)
    pipe.fit(x_tr, y_tr)
    prob = pipe.predict_proba(x_dv)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    return clip_binary_metrics(y_dv, pred)


def pack_row(permutation_id: int, m: dict) -> dict:
    return {
        "permutation_id": permutation_id,
        "F1": m["f1"],
        "balanced_accuracy": m["balanced_accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "TP": m["tp"],
        "FP": m["fp"],
        "TN": m["tn"],
        "FN": m["fn"],
    }


def main() -> None:
    from sklearn.decomposition import PCA  # noqa: F401  # fail fast if missing

    if not ACTUAL_JSON.is_file():
        raise SystemExit(f"STOP: missing {ACTUAL_JSON}")
    actual = json.loads(ACTUAL_JSON.read_text())
    if actual.get("gold_test_scored") is True or int(actual.get("test_n") or 0) != 0:
        raise SystemExit("STOP: actual HuBERT json claims TEST was scored")

    train_ids, y_tr = train_ids_and_labels()
    dev_ids, y_dv = dev_ids_and_labels()
    x_tr = stack_embeddings(train_ids)
    x_dv = stack_embeddings(dev_ids)
    if x_tr.shape != (80, 768) or x_dv.shape != (15, 768):
        raise SystemExit(f"STOP: unexpected shapes {x_tr.shape} {x_dv.shape}")
    n_pca = int(min(PCA_DIM, x_tr.shape[0] - 1, x_tr.shape[1]))
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())

    stored = actual["metrics"]
    replay = eval_at_05(x_tr, y_tr, x_dv, y_dv, n_pca)
    if abs(float(replay["f1"]) - float(stored["f1"])) > 1e-8:
        raise SystemExit(
            "STOP: refit on true TRAIN labels did not match stored HuBERT F1. "
            f"stored={stored['f1']} replay={replay['f1']}. "
            "Existing model was not overwritten."
        )

    rng = np.random.RandomState(SEED)
    rows = []
    f1s = np.empty(N_PERM, dtype=np.float64)
    bas = np.empty(N_PERM, dtype=np.float64)
    for i in range(N_PERM):
        y_perm = rng.permutation(y_tr)
        if int((y_perm == 1).sum()) != n_pos or int((y_perm == 0).sum()) != n_neg:
            raise SystemExit("STOP: permutation did not preserve TRAIN label counts")
        m = eval_at_05(x_tr, y_perm, x_dv, y_dv, n_pca)
        rows.append(pack_row(i, m))
        f1s[i] = m["f1"]
        bas[i] = m["balanced_accuracy"]

    with METRICS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "permutation_id",
                "F1",
                "balanced_accuracy",
                "precision",
                "recall",
                "TP",
                "FP",
                "TN",
                "FN",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    actual_f1 = float(stored["f1"])
    actual_ba = float(stored["balanced_accuracy"])
    p_f1 = (1 + int(np.sum(f1s >= actual_f1))) / (1 + N_PERM)
    p_ba = (1 + int(np.sum(bas >= actual_ba))) / (1 + N_PERM)
    summary = {
        "split": "DEV",
        "gold_test_scored": False,
        "test_n": 0,
        "AUDIO_TEST_SCORED": "NO",
        "FUSION_TEST_SCORED": "NO",
        "LOCKED_TEST_RESULTS_MODIFIED": "NO",
        "HUBERT_EMBEDDINGS_REEXTRACTED": "NO",
        "n_permutations": N_PERM,
        "permutation_seed": SEED,
        "threshold": THRESHOLD,
        "threshold_policy": "fixed 0.5",
        "pca_dim": n_pca,
        "classifier": "StandardScaler + PCA + LogisticRegression(class_weight=balanced, seed=42)",
        "scaler_pca_fitted_on": "TRAIN only",
        "train_n": 80,
        "dev_n": 15,
        "train_label_counts": {"n_pos": n_pos, "n_neg": n_neg},
        "features_fixed": True,
        "dev_labels_fixed": True,
        "pseudo_labels_regenerated": False,
        "actual_F1": actual_f1,
        "actual_BA": actual_ba,
        "actual_precision": float(stored["precision"]),
        "actual_recall": float(stored["recall"]),
        "actual_confusion": {
            "TP": int(stored["tp"]),
            "FP": int(stored["fp"]),
            "TN": int(stored["tn"]),
            "FN": int(stored["fn"]),
        },
        "permutation_mean_F1": float(f1s.mean()),
        "permutation_std_F1": float(f1s.std(ddof=1)),
        "permutation_95th_percentile_F1": float(np.percentile(f1s, 95)),
        "permutation_mean_BA": float(bas.mean()),
        "permutation_std_BA": float(bas.std(ddof=1)),
        "permutation_95th_percentile_BA": float(np.percentile(bas, 95)),
        "p_F1": float(p_f1),
        "p_BA": float(p_ba),
        "n_perm_F1_ge_actual": int(np.sum(f1s >= actual_f1)),
        "n_perm_BA_ge_actual": int(np.sum(bas >= actual_ba)),
        "replay_true_labels_F1": float(replay["f1"]),
        "replay_true_labels_BA": float(replay["balanced_accuracy"]),
        "replay_matches_stored": True,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("wrote", METRICS_CSV)
    print("wrote", SUMMARY_JSON)


if __name__ == "__main__":
    main()
