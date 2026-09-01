#!/usr/bin/env python3
"""VideoMAE Step 5: train a small MLP head on frozen embeddings.

Protocol (mirrors the pose CNN in ``src/pose_cnn.py``; TEST is never used
for any selection):

* **TRAIN** = the pseudo clips that have embeddings
  (``data/features/videomae/<sample_id>.npz``, Step 4), labelled by the frozen
  DEV-tuned rule (``--pseudo-labels``, default ``results/pseudo_labels.csv``;
  head-shake: ``results/shake/pseudo_labels.csv``).
* **DEV** = the 15 gold DEV clips (``--gold-csv`` / ``--label-col``; nod
  default ``label``, head-shake ``shake_label``) — early stopping AND
  probability threshold on DEV F1 only.
* **TEST** = the 15 gold TEST clips, **scored exactly once** with the
  best-on-DEV checkpoint and DEV-chosen threshold.

Model: ``Linear(dim, 64) → ReLU → Dropout(0.2) → Linear(64, 1)``,
``BCEWithLogitsLoss`` with ``pos_weight = neg/pos`` from TRAIN, Adam 1e-3,
batch 16, seed 42. Threshold swept over ``np.linspace(0.2, 0.8, 13)`` (ties
broken by balanced accuracy), exactly as in the pose CNN.

Outputs (nod defaults)::

    results/videomae_frozen_head/metrics.json
    results/videomae_frozen_head/predictions.csv        (TEST rows only)
    results/videomae_frozen_head/training_history.csv
    models/videomae_head.pt                              (gitignored)

Head-shake writes ``results/shake/videomae_frozen_head/`` only (checkpoint
there, not ``models/videomae_head.pt``). Mixed nod/shake paths abort.

TEST-once guard: if ``metrics.json`` already exists under ``--out-dir`` the
script refuses to rerun unless ``--force`` is passed, so TEST cannot be
silently re-scored. Do not pass ``--force`` by default.

Run with ``OMP_NUM_THREADS=1`` for deterministic CPU results (same caveat as
the pose CNN). Nod (already scored — do not rerun)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    /scratch/db01550/venv/bin/python scripts/train_videomae_head.py

Head-shake on otter95 (``/scratch`` venv, **no Docker**; CPU is fine)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python scripts/train_videomae_head.py \\
        --gold-csv data/gold/shake_annotation_sheet.csv \\
        --label-col shake_label \\
        --pseudo-labels results/shake/pseudo_labels.csv \\
        --out-dir results/shake/videomae_frozen_head
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

from src.balance import (  # noqa: E402
    apply_index_list,
    balance_indices,
    boosted_pos_weight,
)
from src.metrics import (  # noqa: E402
    binary_metrics,
    choose_dev_threshold,
    collapse_diagnostics,
)
from src.utils import dump_json  # noqa: E402

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


def resolve_repo_path(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def gold_y(frame: pd.DataFrame, label_col: str, gold_csv: Path) -> list[int]:
    if label_col not in frame.columns:
        raise SystemExit(
            f"STOP: {gold_csv} has no column {label_col!r}. For head-shake "
            "use --gold-csv data/gold/shake_annotation_sheet.csv "
            "--label-col shake_label."
        )
    y = pd.to_numeric(frame[label_col], errors="coerce")
    if y.isna().any():
        bad = frame.loc[y.isna(), "sample_id"].tolist()
        raise SystemExit(
            f"STOP: empty/non-numeric {label_col} in {gold_csv} for {bad}"
        )
    y = y.astype(int)
    if set(y.unique()) - {0, 1}:
        raise SystemExit(f"STOP: {label_col} must be 0/1 only ({gold_csv}).")
    return y.tolist()


def run_leakage_gate(gold_csv: Path, pseudo_labels: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_split_leakage

    check_split_leakage.run(
        gold_csv=gold_csv,
        pseudo_labels=pseudo_labels,
        labelled_train_only=True,
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


def main(
    argv: list[str] | None = None,
    *,
    gold_csv: Path | str | None = None,
    label_col: str | None = None,
    pseudo_labels: Path | str | None = None,
    out_dir: Path | str | None = None,
    model_pt: Path | str | None = None,
    force: bool | None = None,
    balance_train: str | None = None,
    balance_ratio: float | None = None,
    pos_weight_boost: float | None = None,
    dev_only: bool | None = None,
    score_test: bool | None = None,
    select_dev: str | None = None,
    seed: int | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="allow re-scoring TEST (overwrites metrics.json). "
                             "Ignored for locked dirs under --dev-only.")
    parser.add_argument(
        "--dev-only",
        "--no-test",
        dest="dev_only",
        action="store_true",
        help="select on GOLD DEV only; do not score TEST or write "
             "metrics.json. Writes dev_metrics.json + DEV predictions.csv. "
             "Refuses locked TEST out-dirs.",
    )
    parser.add_argument(
        "--select-dev",
        choices=("f1", "balanced_accuracy"),
        default="f1",
        help="DEV epoch/threshold criterion (use balanced_accuracy for "
             "new shake search).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gold-csv", type=Path, default=GOLD_CSV,
                        help="gold CSV with split + label column (default: "
                             "data/gold_annotations.csv). Head-shake: "
                             "data/gold/shake_annotation_sheet.csv")
    parser.add_argument("--label-col", default="label",
                        help="DEV/TEST label column (default: label = nod). "
                             "Head-shake: shake_label")
    parser.add_argument("--pseudo-labels", type=Path, default=PSEUDO_LABELS,
                        help="CSV of sample_id,pseudo_label (default: "
                             "results/pseudo_labels.csv). Head-shake: "
                             "results/shake/pseudo_labels.csv")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="output directory (default: "
                             "results/videomae_frozen_head). Head-shake MUST "
                             "use results/shake/videomae_frozen_head")
    parser.add_argument("--model-pt", type=Path, default=None,
                        help="checkpoint path (default: models/videomae_head.pt "
                             "for the nod out-dir; <out-dir>/best_model.pt "
                             "otherwise)")
    parser.add_argument(
        "--balance-train",
        choices=("none", "subsample", "oversample"),
        default="none",
        help="rebalance TRAIN embeddings only (gold DEV/TEST untouched). "
             "Shake 75/5 collapse test: subsample into a NEW out-dir.",
    )
    parser.add_argument(
        "--balance-ratio",
        type=float,
        default=1.0,
        help="subsample: majority kept ≈ minority × ratio (default 1.0 = 1:1)",
    )
    parser.add_argument(
        "--pos-weight-boost",
        type=float,
        default=1.0,
        help="further emphasise the minority class in BCE pos_weight",
    )
    args = parser.parse_args(argv)

    gold_csv_path = resolve_repo_path(
        gold_csv if gold_csv is not None else args.gold_csv
    )
    pseudo_labels_path = resolve_repo_path(
        pseudo_labels if pseudo_labels is not None else args.pseudo_labels
    )
    out_dir = resolve_repo_path(out_dir if out_dir is not None else args.out_dir)
    label_col = str(
        args.label_col if label_col is None else label_col
    ).strip()
    if model_pt is not None:
        model_pt = resolve_repo_path(model_pt)
    elif args.model_pt is not None:
        model_pt = resolve_repo_path(args.model_pt)
    elif out_dir == resolve_repo_path(OUT_DIR):
        model_pt = MODEL_PT
    else:
        model_pt = out_dir / "best_model.pt"
    if force is None:
        force = args.force
    if balance_train is None:
        balance_train = args.balance_train
    if balance_ratio is None:
        balance_ratio = args.balance_ratio
    if pos_weight_boost is None:
        pos_weight_boost = args.pos_weight_boost
    if score_test is not None:
        dev_only = not bool(score_test)
    elif dev_only is None:
        dev_only = bool(args.dev_only)
    if select_dev is None:
        select_dev = str(args.select_dev)
    if seed is None:
        seed = int(args.seed)
    balance_train = str(balance_train).strip().lower()

    sys.path.insert(0, str(ROOT / "scripts"))
    import check_split_leakage
    task = check_split_leakage.assert_videomae_task_isolation(
        gold_csv=gold_csv_path,
        label_col=label_col,
        pseudo_labels=pseudo_labels_path,
        out_dir=out_dir,
        model_pt=model_pt,
    )
    print(
        f"task={task}  gold={gold_csv_path}  label_col={label_col}\n"
        f"pseudo={pseudo_labels_path}  out_dir={out_dir}  model_pt={model_pt}"
    )

    if dev_only:
        import check_split_leakage
        check_split_leakage.assert_unlocked_out_dir(out_dir)
        if (out_dir / "metrics.json").exists():
            raise SystemExit(
                f"STOP: --dev-only refuses {out_dir / 'metrics.json'} "
                "(TEST-style artefact). Use a new out-dir under "
                "results/shake/dev_search/."
            )
        already = (out_dir / "dev_metrics.json").exists() or (
            out_dir / "metrics_dev.json"
        ).exists()
        if already and not force:
            raise SystemExit(
                f"STOP: DEV-only metrics already exist under {out_dir}. "
                "Pass --force only to invalidate this DEV-only run "
                "(not to shop GOLD TEST)."
            )
    elif (out_dir / "metrics.json").exists() and not force:
        raise SystemExit(
            f"STOP: {out_dir / 'metrics.json'} already exists — TEST has "
            "already been scored once under this protocol. Pass --force only "
            "if the earlier run is being formally invalidated (record why in "
            "reports/dissertation_evidence/experiment_log.md)."
        )
    for needed in (gold_csv_path, pseudo_labels_path, EMB_META):
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
    print("running the split-leakage gate before anything else…")
    run_leakage_gate(gold_csv_path, pseudo_labels_path)
    check_disk("start")

    gold = pd.read_csv(gold_csv_path)
    gold["split"] = gold["split"].astype(str).str.upper()
    pseudo = pd.read_csv(pseudo_labels_path)
    if "pseudo_label" not in pseudo.columns:
        raise SystemExit(
            f"STOP: {pseudo_labels_path} has no pseudo_label column."
        )

    dev = gold[gold.split == "DEV"].sort_values("sample_id")
    tes = gold[gold.split == "TEST"].sort_values("sample_id")

    X_tr, y_tr, train_ids = build_split(
        pseudo["sample_id"].tolist(), pseudo["pseudo_label"].tolist(), "TRAIN"
    )
    X_dv, y_dv, dev_ids = build_split(
        dev["sample_id"].tolist(), gold_y(dev, label_col, gold_csv_path), "DEV"
    )
    if dev_only:
        print("NOTE: --dev-only: TEST embeddings are not loaded and will not "
              "be scored. Ignore any old TEST F1 from locked folders.")
        X_te, y_te, tes_ids = None, None, []
    else:
        X_te, y_te, tes_ids = build_split(
            tes["sample_id"].tolist(), gold_y(tes, label_col, gold_csv_path), "TEST"
        )
    if X_tr is None or len(y_tr) < MIN_TRAIN or len(np.unique(y_tr)) < 2:
        raise SystemExit(
            f"BLOCKED: TRAIN has {0 if X_tr is None else len(y_tr)} usable "
            f"pseudo clips (need >= {MIN_TRAIN} with both classes). No "
            "metrics fabricated; paste this back."
        )
    train_ids_before_balance = list(train_ids)
    y_tr_before = np.asarray(y_tr, dtype=np.int64)
    if balance_train not in ("none", "off", ""):
        bidx = balance_indices(
            y_tr, mode=balance_train, ratio=float(balance_ratio), seed=SEED
        )
        X_tr = X_tr[bidx]
        y_tr = np.asarray(y_tr, dtype=np.int64)[bidx]
        train_ids = apply_index_list(train_ids, bidx)
        print(
            f"TRAIN balance={balance_train} ratio={balance_ratio}: "
            f"{int((y_tr_before == 1).sum())} pos / "
            f"{int((y_tr_before == 0).sum())} neg → "
            f"{int((y_tr == 1).sum())} pos / {int((y_tr == 0).sum())} neg "
            f"(n={len(y_tr)})"
        )
        if len(y_tr) < MIN_TRAIN or len(np.unique(y_tr)) < 2:
            raise SystemExit(
                f"BLOCKED: after {balance_train} TRAIN has {len(y_tr)} clips "
                f"(need >= {MIN_TRAIN} with both classes)."
            )
    if X_dv is None or len(y_dv) < MIN_EVAL:
        raise SystemExit(
            f"BLOCKED: DEV usable clips "
            f"{0 if X_dv is None else len(y_dv)} (need >= {MIN_EVAL}). "
            "No metrics fabricated; paste this back."
        )
    if not dev_only and (X_te is None or len(y_te) < MIN_EVAL):
        raise SystemExit(
            f"BLOCKED: DEV/TEST usable clips "
            f"{0 if X_dv is None else len(y_dv)}/"
            f"{0 if X_te is None else len(y_te)} (need >= {MIN_EVAL} each). "
            "No metrics fabricated; paste this back."
        )

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    dim = int(X_tr.shape[1])
    model = nn.Sequential(
        nn.Linear(dim, HIDDEN),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(HIDDEN, 1),
    )
    pos = max(int((y_tr == 1).sum()), 1)
    neg = max(int((y_tr == 0).sum()), 1)
    pos_w = boosted_pos_weight(pos, neg, boost=float(pos_weight_boost))
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], dtype=torch.float32)
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ds = TensorDataset(
        torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))
    )
    gen = torch.Generator().manual_seed(int(seed))
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
        thr_best, mm_best = choose_dev_threshold(y_dv, prob, criterion=select_dev)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses) if losses else 0.0),
            "dev_f1": float(mm_best["f1"]),
            "dev_precision": float(mm_best["precision"]),
            "dev_balanced_accuracy": float(mm_best["balanced_accuracy"]),
            "dev_probability_threshold": float(thr_best),
        }
        history.append(row)
        print(
            f"epoch {epoch} loss={row['train_loss']:.4f} "
            f"DEV F1={row['dev_f1']:.3f} bAcc={row['dev_balanced_accuracy']:.3f}"
        )
        if best is None:
            better = True
        elif select_dev in ("balanced_accuracy", "bacc", "bal"):
            better = (
                row["dev_balanced_accuracy"],
                row["dev_precision"],
                row["dev_f1"],
            ) > (
                best["dev_balanced_accuracy"],
                best.get("dev_precision", 0.0),
                best["dev_f1"],
            )
        else:
            better = row["dev_f1"] > best["dev_f1"]
        if better:
            best = {**row}
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is None:
        raise SystemExit("BLOCKED: training produced no epochs; nothing saved.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        dv_logits = model(x_dv_t).squeeze(-1).numpy()
    dv_prob = 1 / (1 + np.exp(-dv_logits))
    dv_pred = (dv_prob >= best["dev_probability_threshold"]).astype(int)
    dev_metrics = binary_metrics(y_dv, dv_pred)
    collapse = collapse_diagnostics(dv_pred, dev_metrics["tn"])

    check_disk("write")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_pt)

    meta = json.loads(EMB_META.read_text())
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    common = {
        "task": task,
        "model": "Frozen VideoMAE + MLP head",
        "script": Path(__file__).name,
        "gold_csv": str(gold_csv_path),
        "label_col": label_col,
        "pseudo_labels": str(pseudo_labels_path),
        "out_dir": str(out_dir),
        "model_pt": str(model_pt),
        "checkpoint": meta.get("checkpoint"),
        "embed_dim": dim,
        "transformers_version": meta.get("transformers_version"),
        "torch_version": torch.__version__,
        "seed": int(seed),
        "train_ids": train_ids,
        "train_n": int(len(y_tr)),
        "train_pos": int((y_tr == 1).sum()),
        "train_neg": int((y_tr == 0).sum()),
        "dev_n": int(len(y_dv)),
        "pos_weight": float(pos_w),
        "pos_weight_boost": float(pos_weight_boost),
        "balance_train": balance_train,
        "balance_ratio": float(balance_ratio),
        "train_n_before_balance": int(len(y_tr_before)),
        "train_pos_before_balance": int((y_tr_before == 1).sum()),
        "train_neg_before_balance": int((y_tr_before == 0).sum()),
        "train_ids_before_balance": train_ids_before_balance,
        "best_epoch": int(best["epoch"]),
        "dev_f1": float(dev_metrics["f1"]),
        "dev_precision": float(dev_metrics["precision"]),
        "dev_recall": float(dev_metrics["recall"]),
        "dev_balanced_accuracy": float(dev_metrics["balanced_accuracy"]),
        "dev_probability_threshold": float(best["dev_probability_threshold"]),
        "dev_metrics": dev_metrics,
        "select_dev": select_dev,
        **collapse,
    }
    dump_json(
        out_dir / "config.json",
        {
            "dev_only": bool(dev_only),
            "seed": int(seed),
            "pseudo_labels": str(pseudo_labels_path),
            "out_dir": str(out_dir),
            "label_col": label_col,
            "balance_train": balance_train,
        },
    )

    if dev_only:
        pd.DataFrame(
            {
                "sample_id": dev_ids,
                "label": y_dv,
                "prob": dv_prob,
                "pred": dv_pred,
                "split": "DEV",
            }
        ).to_csv(out_dir / "predictions.csv", index=False)
        pd.DataFrame(
            {
                "sample_id": dev_ids,
                "label": y_dv,
                "prob": dv_prob,
                "pred": dv_pred,
                "split": "DEV",
            }
        ).to_csv(out_dir / "predictions_dev.csv", index=False)
        payload = {
            **common,
            "selection_rule": (
                f"epoch + threshold by DEV {select_dev}; TEST not scored "
                "(--dev-only / --no-test). Ignore any locked TEST F1."
            ),
            "test_scored": False,
        }
        dump_json(out_dir / "dev_metrics.json", payload)
        dump_json(out_dir / "metrics_dev.json", payload)
        flag = "COLLAPSE" if collapse["collapse"] else "ok"
        print(
            f"\nwrote {out_dir}/metrics_dev.json, predictions_dev.csv (DEV), "
            f"training_history.csv\nDEV F1={dev_metrics['f1']:.3f}  "
            f"P={dev_metrics['precision']:.3f} R={dev_metrics['recall']:.3f}  "
            f"TP{dev_metrics['tp']} FP{dev_metrics['fp']} "
            f"TN{dev_metrics['tn']} FN{dev_metrics['fn']}  "
            f"pred+={collapse['predicted_positive_rate']:.3f} [{flag}]\n"
            "TEST was not scored. Do not use locked TEST numbers for selection."
        )
        return

    # ---- single TEST scoring with best-on-DEV weights + DEV threshold ----
    with torch.no_grad():
        te_logits = model(torch.from_numpy(X_te)).squeeze(-1).numpy()
    te_prob = 1 / (1 + np.exp(-te_logits))
    te_pred = (te_prob >= best["dev_probability_threshold"]).astype(int)
    test_metrics = binary_metrics(y_te, te_pred)
    pd.DataFrame(
        {
            "sample_id": tes_ids,
            "label": y_te,
            "prob": te_prob,
            "pred": te_pred,
        }
    ).to_csv(out_dir / "predictions.csv", index=False)
    metrics = {
        **common,
        "test_n": int(len(y_te)),
        "selection_rule": "epoch + threshold by DEV F1 only; TEST scored once",
        "test_metrics": test_metrics,
    }
    dump_json(out_dir / "metrics.json", metrics)
    print(
        f"\nwrote {out_dir}/metrics.json, predictions.csv, "
        f"training_history.csv\nTEST (scored once): {test_metrics}"
    )


if __name__ == "__main__":
    main()
