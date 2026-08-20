#!/usr/bin/env python3
"""VideoMAE Step 5 (otter48): train a small MLP head on frozen embeddings.

Protocol (mirrors the pose CNN in ``src/pose_cnn.py``; TEST is never used
for any selection):

* **TRAIN** = the pseudo clips that have embeddings
  (``data/features/videomae/<sample_id>.npz``, Step 4), labelled by the frozen
  DEV-tuned rule (``results/pseudo_labels.csv``).
* **DEV** = the 15 gold DEV clips (early stopping AND probability threshold
  on DEV F1 only).
* **TEST** = the 15 gold TEST clips, **scored exactly once** with the
  best-on-DEV checkpoint and DEV-chosen threshold.

Model: ``Linear(dim, 64) → ReLU → Dropout(0.2) → Linear(64, 1)``,
``BCEWithLogitsLoss`` with ``pos_weight = neg/pos`` from TRAIN, Adam 1e-3,
batch 16, seed 42. Threshold swept over ``np.linspace(0.2, 0.8, 13)`` (ties
broken by balanced accuracy), exactly as in the pose CNN.

Outputs (commitable, small)::

    results/videomae_frozen_head/metrics.json
    results/videomae_frozen_head/predictions.csv        (TEST rows only)
    results/videomae_frozen_head/training_history.csv
    models/videomae_head.pt                              (gitignored)

TEST-once guard: if ``metrics.json`` already exists the script refuses to
rerun unless ``--force`` is passed, so TEST cannot be silently re-scored.

Run with ``OMP_NUM_THREADS=1`` for deterministic CPU results (same caveat as
the pose CNN). Lab invocation::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    source ../.venv/bin/activate
    OMP_NUM_THREADS=1 python scripts/train_videomae_head.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import binary_metrics  # noqa: E402

GOLD_CSV = ROOT / "data" / "gold_annotations.csv"
PSEUDO_LABELS = ROOT / "results" / "pseudo_labels.csv"
EMB_DIR = ROOT / "data" / "features" / "videomae"
EMB_META = ROOT / "results" / "videomae_embeddings_meta.json"
OUT_DIR = ROOT / "results" / "videomae_frozen_head"
MODEL_PT = ROOT / "models" / "videomae_head.pt"

SEED = 42
EPOCHS = 200
PATIENCE = 8
BATCH = 16
LR = 1e-3
HIDDEN = 64
MIN_FREE_GB = 5.4
MIN_TRAIN = 8
MIN_EVAL = 3


def check_disk(where: str = "") -> None:
    free = shutil.disk_usage(Path.home()).free / 1024**3
    if free < MIN_FREE_GB:
        raise SystemExit(
            f"STOP: free disk on ~ is {free:.2f} GB < {MIN_FREE_GB} GB"
            f"{' at ' + where if where else ''}."
        )


def load_embedding(sample_id: str) -> np.ndarray | None:
    path = EMB_DIR / f"{sample_id}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        return np.asarray(z["embedding"], dtype=np.float32).reshape(-1)


def build_split(sample_ids: list[str], labels: list[int], name: str):
    xs, ys, kept, missing = [], [], [], []
    for sid, y in zip(sample_ids, labels):
        emb = load_embedding(sid)
        if emb is None:
            missing.append(sid)
            continue
        xs.append(emb)
        ys.append(int(y))
        kept.append(sid)
    if missing:
        print(f"NOTE: {name}: {len(missing)} clips have no embedding and are "
              f"excluded: {missing}")
    if not xs:
        return None, None, kept
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="allow re-scoring TEST (overwrites metrics.json)")
    args = parser.parse_args()

    if (OUT_DIR / "metrics.json").exists() and not args.force:
        raise SystemExit(
            f"STOP: {OUT_DIR / 'metrics.json'} already exists — TEST has "
            "already been scored once under this protocol. Pass --force only "
            "if the earlier run is being formally invalidated (record why in "
            "reports/dissertation_evidence/experiment_log.md)."
        )
    for needed in (GOLD_CSV, PSEUDO_LABELS, EMB_META):
        if not needed.exists():
            raise SystemExit(
                f"STOP: {needed} is missing. Run Steps 3-4 (fetch + extract) "
                "first; pseudo labels come from the frozen rule run."
            )
    if not EMB_DIR.exists():
        raise SystemExit(
            f"STOP: {EMB_DIR} does not exist. Run "
            "scripts/extract_videomae_embeddings.py first."
        )
    check_disk("start")

    gold = pd.read_csv(GOLD_CSV)
    gold["split"] = gold["split"].astype(str).str.upper()
    pseudo = pd.read_csv(PSEUDO_LABELS)

    dev = gold[gold.split == "DEV"].sort_values("sample_id")
    tes = gold[gold.split == "TEST"].sort_values("sample_id")

    X_tr, y_tr, train_ids = build_split(
        pseudo["sample_id"].tolist(), pseudo["pseudo_label"].tolist(), "TRAIN"
    )
    X_dv, y_dv, dev_ids = build_split(
        dev["sample_id"].tolist(), dev["label"].tolist(), "DEV"
    )
    X_te, y_te, tes_ids = build_split(
        tes["sample_id"].tolist(), tes["label"].tolist(), "TEST"
    )
    if X_tr is None or len(y_tr) < MIN_TRAIN or len(np.unique(y_tr)) < 2:
        raise SystemExit(
            f"BLOCKED: TRAIN has {0 if X_tr is None else len(y_tr)} usable "
            f"pseudo clips (need >= {MIN_TRAIN} with both classes). No "
            "metrics fabricated; paste this back."
        )
    if X_dv is None or len(y_dv) < MIN_EVAL or X_te is None or len(y_te) < MIN_EVAL:
        raise SystemExit(
            f"BLOCKED: DEV/TEST usable clips "
            f"{0 if X_dv is None else len(y_dv)}/"
            f"{0 if X_te is None else len(y_te)} (need >= {MIN_EVAL} each). "
            "No metrics fabricated; paste this back."
        )

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dim = int(X_tr.shape[1])
    model = nn.Sequential(
        nn.Linear(dim, HIDDEN),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(HIDDEN, 1),
    )
    pos = max(int((y_tr == 1).sum()), 1)
    neg = max(int((y_tr == 0).sum()), 1)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / pos], dtype=torch.float32)
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ds = TensorDataset(
        torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))
    )
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=gen)

    x_dv_t = torch.from_numpy(X_dv)
    history: list[dict] = []
    best: dict | None = None
    best_state = None
    bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            logits = model(x_dv_t).squeeze(-1).numpy()
        prob = 1 / (1 + np.exp(-logits))
        thr_best, f1_best, bal_best = 0.5, -1.0, -1.0
        for t in np.linspace(0.2, 0.8, 13):
            m = binary_metrics(y_dv, (prob >= t).astype(int))
            if m["f1"] > f1_best or (
                m["f1"] == f1_best and m["balanced_accuracy"] > bal_best
            ):
                thr_best, f1_best, bal_best = (
                    float(t), m["f1"], m["balanced_accuracy"],
                )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses) if losses else 0.0),
            "dev_f1": f1_best,
            "dev_balanced_accuracy": bal_best,
            "dev_probability_threshold": thr_best,
        }
        history.append(row)
        print(f"epoch {epoch} loss={row['train_loss']:.4f} DEV F1={f1_best:.3f}")
        if best is None or f1_best > best["dev_f1"]:
            best = {**row}
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is None:
        raise SystemExit("BLOCKED: training produced no epochs; nothing saved.")

    # ---- single TEST scoring with best-on-DEV weights + DEV threshold ----
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(torch.from_numpy(X_te)).squeeze(-1).numpy()
    te_prob = 1 / (1 + np.exp(-te_logits))
    te_pred = (te_prob >= best["dev_probability_threshold"]).astype(int)
    test_metrics = binary_metrics(y_te, te_pred)

    check_disk("write")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_PT)

    meta = json.loads(EMB_META.read_text())
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": tes_ids,
            "label": y_te,
            "prob": te_prob,
            "pred": te_pred,
        }
    ).to_csv(OUT_DIR / "predictions.csv", index=False)
    metrics = {
        "model": "Frozen VideoMAE + MLP head",
        "script": Path(__file__).name,
        "checkpoint": meta.get("checkpoint"),
        "embed_dim": dim,
        "transformers_version": meta.get("transformers_version"),
        "torch_version": torch.__version__,
        "seed": SEED,
        "train_ids": train_ids,
        "train_n": int(len(y_tr)),
        "train_pos": int((y_tr == 1).sum()),
        "train_neg": int((y_tr == 0).sum()),
        "dev_n": int(len(y_dv)),
        "test_n": int(len(y_te)),
        "pos_weight": neg / pos,
        "best_epoch": int(best["epoch"]),
        "dev_f1": float(best["dev_f1"]),
        "dev_balanced_accuracy": float(best["dev_balanced_accuracy"]),
        "dev_probability_threshold": float(best["dev_probability_threshold"]),
        "selection_rule": "epoch + threshold by DEV F1 only; TEST scored once",
        "test_metrics": test_metrics,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"\nwrote {OUT_DIR}/metrics.json, predictions.csv, "
        f"training_history.csv\nTEST (scored once): {test_metrics}"
    )


if __name__ == "__main__":
    main()
