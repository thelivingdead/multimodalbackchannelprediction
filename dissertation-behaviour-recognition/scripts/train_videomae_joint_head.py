#!/usr/bin/env python3
"""Joint two-head MLP on frozen VideoMAE embeddings (CPU; new protocol).

Shared 768-D embedding, two binary heads. TRAIN = inner join of nod and
shake frozen-rule pseudo-labels on the same 80 clip ids. DEV/TEST gold from
``shake_annotation_sheet.csv`` (``nod_label`` + ``shake_label``).

Writes ``results/joint/videomae_frozen_head/``. Needs
``data/features/videomae/*.npz`` (otter).

Otter95 (CPU; ``/scratch`` venv, **no Docker**)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/train_videomae_joint_head.py
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

from src.metrics import binary_metrics  # noqa: E402

SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
NOD_GOLD = ROOT / "data" / "gold_annotations.csv"
NOD_PSEUDO = ROOT / "results" / "pseudo_labels.csv"
SHAKE_PSEUDO = ROOT / "results" / "shake" / "pseudo_labels.csv"
OUT_DIR = ROOT / "results" / "joint" / "videomae_frozen_head"
EMB_DIR = ROOT / "data" / "features" / "videomae"
EMB_META = ROOT / "results" / "videomae_embeddings_meta.json"
SEED = 42
EPOCHS = 200
PATIENCE = 8
BATCH = 16
LR = 1e-3
HIDDEN = 64
MIN_TRAIN = 8
MIN_EVAL = 3


def load_emb(sample_id: str) -> np.ndarray | None:
    path = EMB_DIR / f"{sample_id}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        return np.asarray(z["embedding"], dtype=np.float32).reshape(-1)


def stack_ids(sids, y_n, y_s, name: str):
    xs, yn, ys, kept, missing = [], [], [], [], []
    for sid, a, b in zip(sids, y_n, y_s):
        e = load_emb(str(sid))
        if e is None:
            missing.append(str(sid))
            continue
        xs.append(e)
        yn.append(int(a))
        ys.append(int(b))
        kept.append(str(sid))
    if missing:
        print(f"NOTE: {name}: {len(missing)} embeddings missing: {missing}")
    if not xs:
        return None, None, None, kept
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(yn, dtype=np.int64),
        np.asarray(ys, dtype=np.int64),
        kept,
    )


def best_thr(y, prob):
    thr_best, f1_best, bal_best = 0.5, -1.0, -1.0
    for t in np.linspace(0.2, 0.8, 13):
        m = binary_metrics(y, (prob >= t).astype(int))
        if m["f1"] > f1_best or (
            m["f1"] == f1_best and m["balanced_accuracy"] > bal_best
        ):
            thr_best, f1_best, bal_best = float(t), m["f1"], m["balanced_accuracy"]
    return thr_best, f1_best, bal_best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import check_split_leakage

    out_dir = OUT_DIR
    pt = out_dir / "best_model.pt"
    check_split_leakage.assert_joint_videomae_paths(
        gold_csv=SHAKE_GOLD,
        nod_pseudo=NOD_PSEUDO,
        shake_pseudo=SHAKE_PSEUDO,
        out_dir=out_dir,
        model_pt=pt,
    )
    if (out_dir / "metrics.json").exists() and not args.force:
        raise SystemExit(
            f"STOP: {out_dir / 'metrics.json'} exists — joint TEST scored once."
        )
    for needed in (SHAKE_GOLD, NOD_PSEUDO, SHAKE_PSEUDO, EMB_META):
        if not needed.exists():
            raise SystemExit(f"STOP: missing {needed}")
    if not EMB_DIR.exists():
        raise SystemExit(
            f"STOP: {EMB_DIR} missing. On otter run extract_videomae_embeddings.py."
        )

    print("running nod + shake split-leakage gates…")
    check_split_leakage.run(
        gold_csv=NOD_GOLD if NOD_GOLD.exists() else SHAKE_GOLD,
        pseudo_labels=NOD_PSEUDO,
        labelled_train_only=True,
    )
    check_split_leakage.run(
        gold_csv=SHAKE_GOLD,
        pseudo_labels=SHAKE_PSEUDO,
        labelled_train_only=True,
    )

    gold = pd.read_csv(SHAKE_GOLD)
    gold["split"] = gold["split"].astype(str).str.upper()
    for col in ("nod_label", "shake_label"):
        if col not in gold.columns:
            raise SystemExit(f"STOP: gold missing {col}")

    nod_p = pd.read_csv(NOD_PSEUDO)
    shk_p = pd.read_csv(SHAKE_PSEUDO)
    train = nod_p.merge(
        shk_p[["sample_id", "pseudo_label"]].rename(
            columns={"pseudo_label": "shake_pseudo"}
        ),
        on="sample_id",
        how="inner",
    ).rename(columns={"pseudo_label": "nod_pseudo"})
    if len(train) < MIN_TRAIN:
        raise SystemExit(f"STOP: inner join TRAIN n={len(train)}")

    dev = gold[gold.split == "DEV"].sort_values("sample_id")
    tes = gold[gold.split == "TEST"].sort_values("sample_id")

    X_tr, y_tr_n, y_tr_s, train_ids = stack_ids(
        train["sample_id"], train["nod_pseudo"], train["shake_pseudo"], "TRAIN"
    )
    X_dv, y_dv_n, y_dv_s, dev_ids = stack_ids(
        dev["sample_id"], dev["nod_label"], dev["shake_label"], "DEV"
    )
    X_te, y_te_n, y_te_s, tes_ids = stack_ids(
        tes["sample_id"], tes["nod_label"], tes["shake_label"], "TEST"
    )
    if X_tr is None or len(y_tr_n) < MIN_TRAIN:
        raise SystemExit(
            "BLOCKED: not enough TRAIN embeddings. This script must run on "
            "otter where data/features/videomae/*.npz exist."
        )
    if X_dv is None or X_te is None or len(y_dv_n) < MIN_EVAL or len(y_te_n) < MIN_EVAL:
        raise SystemExit("BLOCKED: DEV/TEST embeddings missing.")
    if len(np.unique(y_tr_n)) < 2 or len(np.unique(y_tr_s)) < 2:
        raise SystemExit("BLOCKED: TRAIN needs both classes on both heads.")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dim = int(X_tr.shape[1])

    class TwoHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(dim, HIDDEN),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.head_nod = nn.Linear(HIDDEN, 1)
            self.head_shake = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            h = self.shared(x)
            return self.head_nod(h).squeeze(-1), self.head_shake(h).squeeze(-1)

    model = TwoHead()

    def pos_w(y):
        pos = max(int((y == 1).sum()), 1)
        neg = max(int((y == 0).sum()), 1)
        return neg / pos

    crit_n = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w(y_tr_n)], dtype=torch.float32)
    )
    crit_s = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w(y_tr_s)], dtype=torch.float32)
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr_n.astype(np.float32)),
        torch.from_numpy(y_tr_s.astype(np.float32)),
    )
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=gen)
    x_dv = torch.from_numpy(X_dv)
    x_te = torch.from_numpy(X_te)

    history = []
    best = None
    best_state = None
    bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yn, ys in loader:
            opt.zero_grad()
            ln, ls = model(xb)
            loss = crit_n(ln, yn) + crit_s(ls, ys)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            ln, ls = model(x_dv)
        pn = 1 / (1 + np.exp(-ln.numpy()))
        ps = 1 / (1 + np.exp(-ls.numpy()))
        thr_n, f1_n, bal_n = best_thr(y_dv_n, pn)
        thr_s, f1_s, bal_s = best_thr(y_dv_s, ps)
        mean_f1 = 0.5 * (f1_n + f1_s)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses) if losses else 0.0),
            "dev_f1_nod": f1_n,
            "dev_f1_shake": f1_s,
            "dev_f1_mean": mean_f1,
            "dev_threshold_nod": thr_n,
            "dev_threshold_shake": thr_s,
        }
        history.append(row)
        print(
            f"epoch {epoch} loss={row['train_loss']:.4f}  "
            f"DEV F1 nod={f1_n:.3f} shake={f1_s:.3f} mean={mean_f1:.3f}"
        )
        if best is None or mean_f1 > best["dev_f1_mean"]:
            best = {**row}
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is None:
        raise SystemExit("BLOCKED: no epochs")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ln_te, ls_te = model(x_te)
        ln_dv, ls_dv = model(x_dv)
    pn_te = 1 / (1 + np.exp(-ln_te.numpy()))
    ps_te = 1 / (1 + np.exp(-ls_te.numpy()))
    pn_dv = 1 / (1 + np.exp(-ln_dv.numpy()))
    ps_dv = 1 / (1 + np.exp(-ls_dv.numpy()))
    thr_n, thr_s = best["dev_threshold_nod"], best["dev_threshold_shake"]
    nod_te = binary_metrics(y_te_n, (pn_te >= thr_n).astype(int))
    shk_te = binary_metrics(y_te_s, (ps_te >= thr_s).astype(int))
    nod_dv = binary_metrics(y_dv_n, (pn_dv >= thr_n).astype(int))
    shk_dv = binary_metrics(y_dv_s, (ps_dv >= thr_s).astype(int))

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, pt)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": tes_ids,
            "nod_label": y_te_n,
            "shake_label": y_te_s,
            "nod_prob": pn_te,
            "shake_prob": ps_te,
            "nod_pred": (pn_te >= thr_n).astype(int),
            "shake_pred": (ps_te >= thr_s).astype(int),
            "split": "TEST",
        }
    ).to_csv(out_dir / "predictions.csv", index=False)
    meta = json.loads(EMB_META.read_text()) if EMB_META.exists() else {}
    metrics = {
        "task": "joint_nod_shake",
        "model": "Frozen VideoMAE embedding + two-head MLP",
        "script": Path(__file__).name,
        "gold_csv": str(SHAKE_GOLD),
        "out_dir": str(out_dir),
        "embed_dim": dim,
        "checkpoint": meta.get("checkpoint"),
        "seed": SEED,
        "train_n": int(len(y_tr_n)),
        "best_epoch": int(best["epoch"]),
        "dev_f1_mean": float(best["dev_f1_mean"]),
        "dev_threshold_nod": float(thr_n),
        "dev_threshold_shake": float(thr_s),
        "dev_metrics_nod": nod_dv,
        "dev_metrics_shake": shk_dv,
        "test_metrics_nod": nod_te,
        "test_metrics_shake": shk_te,
        "selection_rule": (
            "epoch by mean DEV F1; independent DEV thresholds; TEST once"
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"TEST nod: {nod_te}\nTEST shake: {shk_te}\nwrote {out_dir}")


if __name__ == "__main__":
    main()
