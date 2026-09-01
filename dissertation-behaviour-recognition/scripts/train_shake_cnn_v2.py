#!/usr/bin/env python3
"""Head-shake 1D CNN v2 — GOLD DEV only; never writes locked shake/cnn.

TRAIN = a v2 pseudo CSV (rule-ranked, not gold). DEV = gold ``shake_label``.
TEST is **not** scored unless ``--score-test`` (default OFF).

Writes only under ``results/shake/v2/...``.

Otter (CPU torch is fine)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    source ../.venv/bin/activate
    OMP_NUM_THREADS=1 python scripts/train_shake_cnn_v2.py \\
        --pseudo-labels results/shake/v2/pseudo_40_40.csv \\
        --out-dir results/shake/v2/cnn_40_40
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

from shake_v2_common import V2_ROOT, collapse_verdict  # noqa: E402
from src.utils import dump_json  # noqa: E402

FEATURE_MODE = "C"
PATIENCE = 4
SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pseudo-labels", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--score-test",
        action="store_true",
        default=False,
        help="OFF by default. Do not pass this for v2 selection.",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import check_split_leakage
    from src.pose_cnn import _build_cnn, _torch, build_matrix

    try:
        from src.metrics import binary_metrics, collapse_diagnostics
    except Exception:
        from shake_v2_common import clip_binary_metrics as binary_metrics

        def collapse_diagnostics(pred, tn):
            return collapse_verdict({"tn": tn, "tp": 0, "fp": int(np.sum(pred)), "fn": 0, "n": len(pred)})

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir = out_dir.resolve()
    check_split_leakage.assert_unlocked_out_dir(out_dir)
    if not str(out_dir).startswith(str(V2_ROOT.resolve())):
        raise SystemExit(
            f"STOP: v2 CNN must write under {V2_ROOT} (got {out_dir})"
        )
    dest_metrics = out_dir / ("metrics.json" if args.score_test else "metrics_dev.json")
    alt = out_dir / "dev_metrics.json"
    if (dest_metrics.exists() or alt.exists()) and not args.force:
        raise SystemExit(
            f"STOP: DEV metrics already exist under {out_dir}. "
            "Pass --force only to invalidate."
        )

    pl = args.pseudo_labels if args.pseudo_labels.is_absolute() else ROOT / args.pseudo_labels
    check_split_leakage.run(
        gold_csv=SHAKE_GOLD, pseudo_labels=pl, labelled_train_only=True
    )
    check_split_leakage.assert_videomae_task_isolation(
        gold_csv=SHAKE_GOLD,
        label_col="shake_label",
        pseudo_labels=pl,
        out_dir=out_dir,
    )

    gold = pd.read_csv(SHAKE_GOLD)
    gold["split"] = gold["split"].astype(str).str.upper()
    gold["label"] = pd.to_numeric(gold["shake_label"], errors="coerce")
    if gold["label"].isna().any():
        raise SystemExit("STOP: empty shake_label in gold sheet")
    gold["label"] = gold["label"].astype(int)
    pseudo = pd.read_csv(pl)
    if "pseudo_label" not in pseudo.columns:
        raise SystemExit(f"STOP: {pl} has no pseudo_label")

    pseudo_paths = []
    y_tr = []
    for r in pseudo.itertuples():
        p = ROOT / "features" / "pseudo" / f"{r.sample_id}.npz"
        if not p.exists():
            raise SystemExit(f"STOP: missing {p}")
        pseudo_paths.append(p)
        y_tr.append(int(r.pseudo_label))
    y_tr = np.asarray(y_tr, dtype=int)
    if y_tr.min() == y_tr.max() or len(y_tr) < 8:
        raise SystemExit(
            f"STOP: TRAIN needs both classes and n>=8 (got {len(y_tr)} "
            f"pos={int((y_tr == 1).sum())})"
        )

    from train_shake_cnn import _gold_paths

    dev_p, y_dev = _gold_paths(gold, ROOT, "DEV")

    mods = _torch()
    if mods is None:
        raise SystemExit(
            "STOP: torch missing. On otter:\n"
            "  source ../.venv/bin/activate\n"
            "  OMP_NUM_THREADS=1 python scripts/train_shake_cnn_v2.py ..."
        )
    torch, nn, DataLoader, TensorDataset = mods
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Xtr, mean, std = build_matrix(pseudo_paths, FEATURE_MODE)
    Xdv, _, _ = build_matrix(dev_p, FEATURE_MODE, mean, std)
    d = int(Xtr.shape[-1])
    model = _build_cnn(nn, d)
    pos = max(int((y_tr == 1).sum()), 1)
    neg = max(int((y_tr == 0).sum()), 1)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(
        torch.from_numpy(np.transpose(Xtr, (0, 2, 1))),
        torch.from_numpy(y_tr.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    hist: list[dict] = []
    best: dict | None = None
    best_state = None
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(np.transpose(Xdv, (0, 2, 1)))).numpy()
        prob = 1 / (1 + np.exp(-logits))
        thr_best, f1_best, bal_best = 0.5, -1.0, -1.0
        for t in np.linspace(0.2, 0.8, 13):
            mm = binary_metrics(y_dev, (prob >= t).astype(int))
            if mm["f1"] > f1_best or (
                mm["f1"] == f1_best and mm["balanced_accuracy"] > bal_best
            ):
                thr_best, f1_best, bal_best = float(t), mm["f1"], mm["balanced_accuracy"]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses) if losses else 0),
            "dev_f1": f1_best,
            "dev_balanced_accuracy": bal_best,
            "dev_probability_threshold": thr_best,
        }
        hist.append(row)
        print(f"epoch {epoch} loss={row['train_loss']:.4f} DEV F1={f1_best:.3f}")
        if best is None or f1_best > best["dev_f1"]:
            best = {**row}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    assert best is not None and best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        dv = model(torch.from_numpy(np.transpose(Xdv, (0, 2, 1)))).numpy()
    pdv = 1 / (1 + np.exp(-dv))
    thr = float(best["dev_probability_threshold"])
    dv_pred = (pdv >= thr).astype(int)
    dev_m = binary_metrics(y_dev, dv_pred)
    collapse = collapse_diagnostics(dv_pred, dev_m["tn"])

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": [p.stem for p in dev_p],
            "label": y_dev,
            "prob": pdv,
            "pred": dv_pred,
            "split": "DEV",
        }
    ).to_csv(out_dir / "predictions.csv", index=False)
    payload = {
        "task": "head_shake",
        "model": "1D CNN (feature set C = xyz + first differences)",
        "script": "train_shake_cnn_v2.py",
        "feature_set": FEATURE_MODE,
        "seed": int(args.seed),
        "epochs_budget": int(args.epochs),
        "early_stopping_patience": PATIENCE,
        "best_epoch": int(best["epoch"]),
        "dev_probability_threshold": thr,
        "dev_f1": float(dev_m["f1"]),
        "dev_precision": float(dev_m["precision"]),
        "dev_recall": float(dev_m["recall"]),
        "dev_balanced_accuracy": float(dev_m["balanced_accuracy"]),
        "dev_metrics": dev_m,
        "train_n": int(len(y_tr)),
        "train_pos": int((y_tr == 1).sum()),
        "train_neg": int((y_tr == 0).sum()),
        "train_ids": [p.stem for p in pseudo_paths],
        "pseudo_labels": str(pl),
        "test_scored": False,
        "selection_rule": "epoch + threshold by DEV F1 only; TEST not scored",
        "normalization": {"mean": mean.tolist(), "std": std.tolist(), "mode": FEATURE_MODE},
        **collapse,
    }
    if args.score_test:
        raise SystemExit(
            "STOP: v2 CNN refuses --score-test. New holdout annotation is the "
            "next TEST, not a re-score of the locked 15."
        )
    from shake_v2_common import write_dev_json_and_preds

    pred_df = pd.DataFrame(
        {
            "sample_id": [p.stem for p in dev_p],
            "label": y_dev,
            "prob": pdv,
            "pred": dv_pred,
            "split": "DEV",
        }
    )
    write_dev_json_and_preds(out_dir, payload, pred_df)
    dump_json(
        out_dir / "config.json",
        {
            "pseudo_labels": str(pl),
            "out_dir": str(out_dir),
            "seed": int(args.seed),
            "score_test": False,
        },
    )
    flag = "COLLAPSE" if collapse.get("collapse") else "ok"
    print(
        f"DEV F1={dev_m['f1']:.3f} P={dev_m['precision']:.3f} "
        f"R={dev_m['recall']:.3f} TP{dev_m['tp']} FP{dev_m['fp']} "
        f"TN{dev_m['tn']} FN{dev_m['fn']} [{flag}]"
    )
    print("TEST was not scored.")


if __name__ == "__main__":
    main()
