#!/usr/bin/env python3
"""Train the shake 1D CNN on frozen-rule pseudo-labels (standalone).

Mirrors ``scripts/train_pose_cnn.py`` / ``src/pose_cnn.py`` for **head shake**,
not nod. Reuses the same 1D CNN and feature set C (Euler xyz + first
differences — the nod headline set). Writes **only** under ``results/shake/``.

This script does **not** retune or rescore the shake RULE. It reads the
frozen axis + threshold from ``results/shake/rule_selected_config.json`` and
aborts if that file is missing. The official shake-rule TEST F1 (otter:
axis z, 11.150°, F1 0.70) lives in ``results/shake/rule_test_metrics.json``
and is never rewritten here. Nod files such as
``results/classifier_test_metrics.json`` and ``results/pseudo_labels.csv``
are never touched.

Protocol
--------
1. Load the frozen shake rule (axis + threshold). Abort if missing — do
   not invent a new rule TEST score.
2. Pseudo-label ``features/pseudo/pseudo_00001.npz`` … ``pseudo_00080.npz``
   (those 80 ids if they exist) with that rule. Write
   ``results/shake/pseudo_labels.csv``.
3. Leakage gate: every pseudo ``video_id`` must be disjoint from gold DEV
   **and** gold TEST ``video_id``s. Abort on any overlap.
4. Train the 1D CNN on feature set C. Normalisation from TRAIN only.
5. Epoch + probability threshold chosen on gold DEV ``shake_label`` only.
6. Gold TEST scored **once**. Refuse if ``results/shake/cnn/metrics.json``
   already exists unless ``--force`` (invalidation only — do not shop TEST
   scores).

Writes
------
::

    results/shake/pseudo_labels.csv
    results/shake/cnn/metrics.json
    results/shake/cnn/predictions.csv          (TEST rows only)
    results/shake/cnn/training_history.csv

Otter48 — home venv (CPU torch). Do **not** use Docker python.

Must already exist on otter disk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``data/gold/shake_annotation_sheet.csv`` (30 filled ``shake_label`` rows)
* ``data/gold_annotations.csv`` (split / ``video_id`` join)
* ``features/gold/gold_001.npz`` … ``gold_030.npz``
* ``features/pseudo/pseudo_00001.npz`` … ``pseudo_00080.npz``
* ``results/shake/rule_selected_config.json`` (frozen by
  ``scripts/run_shake_experiment.py``). If this json is missing, otter must
  ``git add results/shake/`` from the machine that already scored the rule
  TEST — do **not** re-run the rule TEST.

::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    source ../.venv/bin/activate
    OMP_NUM_THREADS=1 python scripts/train_shake_cnn.py

CPU is fine. Pin threads for the same determinism caveat as the nod CNN.
The rule trainer is a **separate** script (``run_shake_experiment.py``).
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

FEATURE_MODE = "C"  # xyz + first differences; nod headline set
N_PSEUDO = 80
PATIENCE = 4
RULE_CFG_REL = Path("results") / "shake" / "rule_selected_config.json"
CNN_DIR_REL = Path("results") / "shake" / "cnn"


def _as_str(value) -> str:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, OSError):
            pass
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).strip()


def load_npz(path: Path) -> dict:
    """Same key copy as ``src.pose_cnn.load_npz`` (avoid importing torch/scipy here)."""
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def npz_video_id(path: Path) -> str:
    z = load_npz(path)
    if "video_id" not in z:
        raise SystemExit(f"STOP: {path.name} has no video_id; cannot certify no leak")
    return _as_str(z["video_id"])


def load_frozen_shake_rule(work: Path) -> dict:
    path = work / RULE_CFG_REL
    if not path.exists():
        raise SystemExit(
            "STOP: frozen shake rule missing: "
            f"{RULE_CFG_REL.as_posix()}\n"
            "This script will not retune the rule or invent a TEST score.\n"
            "On otter, after the rule TEST has already been scored once:\n"
            "  git add results/shake/\n"
            "  git commit -m \"Record shake-rule TEST (do not rescore).\"\n"
            "Then pull that commit. Do not re-run scripts/run_shake_experiment.py "
            "just to recreate this json."
        )
    cfg = json.loads(path.read_text())
    if "chosen_rotation_axis" not in cfg or "selected_amplitude_threshold" not in cfg:
        raise SystemExit(
            f"STOP: {path} is missing chosen_rotation_axis / "
            "selected_amplitude_threshold. Do not invent them here."
        )
    task = cfg.get("task")
    if task not in (None, "head_shake"):
        raise SystemExit(
            f"STOP: {path} has task={task!r}; expected head_shake. "
            "Do not point this trainer at the nod rule config."
        )
    return {
        "path": str(RULE_CFG_REL.as_posix()),
        "chosen_rotation_axis": int(cfg["chosen_rotation_axis"]),
        "axis_name": str(cfg.get("axis_name") or ["x", "y", "z"][int(cfg["chosen_rotation_axis"])]),
        "selected_amplitude_threshold": float(cfg["selected_amplitude_threshold"]),
    }


def select_pseudo_paths(pseudo_dir: Path) -> list[Path]:
    """Prefer pseudo_00001–00080 when those files exist; do not pull extras."""
    if not pseudo_dir.is_dir():
        raise SystemExit(f"STOP: missing pseudo pose dir: {pseudo_dir}")
    preferred = [pseudo_dir / f"pseudo_{i:05d}.npz" for i in range(1, N_PSEUDO + 1)]
    have = [p for p in preferred if p.exists()]
    if len(have) == N_PSEUDO:
        return have
    if have:
        print(
            f"NOTE: only {len(have)}/{N_PSEUDO} preferred pseudo ids exist; "
            "using those and not substituting later ids"
        )
        return have
    extras = sorted(pseudo_dir.glob("*.npz"))
    if not extras:
        raise SystemExit(f"STOP: no features/pseudo/*.npz under {pseudo_dir}")
    print(
        "NOTE: preferred pseudo_00001–00080 names are absent; "
        f"using the first {min(N_PSEUDO, len(extras))} sorted npz files"
    )
    return extras[:N_PSEUDO]


def assert_no_leakage(pseudo_paths: list[Path], gold: pd.DataFrame) -> list[str]:
    gold_ids = set(gold["sample_id"].astype(str))
    gold_vids = set(
        gold.loc[gold["split"].isin(["DEV", "TEST"]), "video_id"].astype(str)
    )
    leaks: list[str] = []
    vids: list[str] = []
    for p in pseudo_paths:
        sid = p.stem
        vid = npz_video_id(p)
        if sid in gold_ids:
            leaks.append(f"pseudo sample_id collides with gold: {sid}")
        if not vid:
            leaks.append(f"pseudo {sid} has empty video_id")
        elif vid in gold_vids:
            leaks.append(
                f"pseudo {sid} video_id={vid} overlaps gold DEV/TEST"
            )
        vids.append(vid)
    if leaks:
        raise SystemExit(
            "STOP: split leakage — do not train:\n- " + "\n- ".join(leaks)
        )
    return vids


def write_shake_pseudo_labels(
    pseudo_paths: list[Path],
    video_ids: list[str],
    rule: dict,
    dest: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    from run_full_experiment import rule_score

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() == (ROOT / "results" / "pseudo_labels.csv").resolve():
        raise SystemExit("STOP: refusing to overwrite nod results/pseudo_labels.csv")
    axis = int(rule["chosen_rotation_axis"])
    thr = float(rule["selected_amplitude_threshold"])
    rows = []
    for p, vid in zip(pseudo_paths, video_ids):
        sc = float(rule_score(load_npz(p)["rotation_xyz"], axis))
        rows.append(
            {
                "sample_id": p.stem,
                "video_id": vid,
                "rule_score": sc,
                "pseudo_label": int(sc >= thr),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(dest, index=False)
    y = df["pseudo_label"].to_numpy(dtype=int)
    print(
        f"wrote {dest}  ({int((y == 0).sum())} neg / {int((y == 1).sum())} pos)  "
        f"rule axis={rule['axis_name']} thr={thr:.3f}°"
    )
    return y, df


def _gold_paths(gold: pd.DataFrame, work: Path, split: str) -> tuple[list[Path], np.ndarray]:
    part = gold[gold.split == split]
    paths, labels = [], []
    missing = []
    for r in part.itertuples():
        p = work / "features" / "gold" / f"{r.sample_id}.npz"
        if not p.exists():
            missing.append(r.sample_id)
            continue
        paths.append(p)
        labels.append(int(r.label))
    if missing:
        raise SystemExit(f"STOP: missing gold pose npz for {split}: {missing}")
    if len(paths) < 3:
        raise SystemExit(f"STOP: not enough gold {split} features ({len(paths)})")
    return paths, np.asarray(labels, dtype=int)


def train(
    gold: pd.DataFrame,
    work: Path,
    rule: dict,
    pseudo_paths: list[Path],
    y_tr: np.ndarray,
    epochs: int,
    seed: int,
    out_dir: Path | None = None,
    dev_only: bool = False,
    select_dev: str = "f1",
    seq_len: int = 128,
) -> dict:
    from src.clip_metrics import (  # sklearn-free; otter still has sklearn
        choose_dev_threshold,
        clip_binary_metrics as binary_metrics,
        collapse_diagnostics,
    )
    from src.pose_cnn import _build_cnn, _torch, build_matrix
    from src.utils import dump_json

    mods = _torch()
    if mods is None:
        raise SystemExit(
            "STOP: torch is not available in this Python. On otter use the "
            "home venv (CPU is fine):\n"
            "  cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition\n"
            "  source ../.venv/bin/activate\n"
            "  OMP_NUM_THREADS=1 python scripts/train_shake_cnn.py\n"
            "Do not use Docker python. Pseudo labels were written; TEST was not scored."
        )
    torch, nn, DataLoader, TensorDataset = mods
    torch.manual_seed(seed)
    np.random.seed(seed)

    if y_tr.min() == y_tr.max():
        raise SystemExit(
            "STOP: frozen shake rule produced a single class on TRAIN. "
            "Not inventing labels. Check the pseudo-label CSV."
        )

    dev_p, y_dev = _gold_paths(gold, work, "DEV")
    if not dev_only:
        tes_p, y_tes = _gold_paths(gold, work, "TEST")

    Xtr, mean, std = build_matrix(pseudo_paths, FEATURE_MODE, seq_len=seq_len)
    Xdv, _, _ = build_matrix(dev_p, FEATURE_MODE, mean, std, seq_len=seq_len)
    if not dev_only:
        Xte, _, _ = build_matrix(tes_p, FEATURE_MODE, mean, std, seq_len=seq_len)
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
    for epoch in range(1, epochs + 1):
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
        thr_best, mm_best = choose_dev_threshold(y_dev, prob, criterion=select_dev)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses) if losses else 0),
            "dev_f1": float(mm_best["f1"]),
            "dev_precision": float(mm_best["precision"]),
            "dev_balanced_accuracy": float(mm_best["balanced_accuracy"]),
            "dev_probability_threshold": float(thr_best),
        }
        hist.append(row)
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

    if out_dir is None:
        out_dir = work / CNN_DIR_REL
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(out_dir / "training_history.csv", index=False)
    dump_json(
        out_dir / "config.json",
        {
            "dev_only": bool(dev_only),
            "select_dev": select_dev,
            "seq_len": int(seq_len),
            "seed": int(seed),
            "feature_set": FEATURE_MODE,
            "out_dir": str(out_dir),
            "train_n": int(len(y_tr)),
            "train_pos": int((y_tr == 1).sum()),
            "train_neg": int((y_tr == 0).sum()),
            "frozen_rule": rule,
        },
    )
    common = {
        "task": "head_shake",
        "model": "1D CNN (feature set C = xyz + first differences)",
        "script": "train_shake_cnn.py",
        "feature_set": FEATURE_MODE,
        "feature_set_name": "xyz_deriv",
        "input_dimensions": d,
        "seq_len": int(seq_len),
        "seed": int(seed),
        "epochs_budget": int(epochs),
        "early_stopping_patience": PATIENCE,
        "best_epoch": int(best["epoch"]),
        "dev_f1": float(dev_m["f1"]),
        "dev_precision": float(dev_m["precision"]),
        "dev_recall": float(dev_m["recall"]),
        "dev_balanced_accuracy": float(dev_m["balanced_accuracy"]),
        "dev_probability_threshold": thr,
        "dev_metrics": dev_m,
        "frozen_rule": rule,
        "train_n": int(len(y_tr)),
        "train_pos": int((y_tr == 1).sum()),
        "train_neg": int((y_tr == 0).sum()),
        "train_ids": [p.stem for p in pseudo_paths],
        "normalization": {"mean": mean.tolist(), "std": std.tolist(), "mode": FEATURE_MODE},
        **collapse,
    }

    if dev_only:
        pred_df = pd.DataFrame(
            {
                "sample_id": [p.stem for p in dev_p],
                "label": y_dev,
                "prob": pdv,
                "pred": dv_pred,
                "split": "DEV",
            }
        )
        pred_df.to_csv(out_dir / "predictions.csv", index=False)
        pred_df.to_csv(out_dir / "predictions_dev.csv", index=False)
        metrics = {
            **common,
            "select_dev": select_dev,
            "selection_rule": (
                f"epoch + threshold by DEV {select_dev}; TEST not scored "
                "(--dev-only / --no-test). Ignore any locked TEST F1."
            ),
            "test_scored": False,
            "note": (
                "Did not retune or rescore the shake rule TEST. "
                "Nod results/ artefacts were not written."
            ),
        }
        dump_json(out_dir / "dev_metrics.json", metrics)
        dump_json(out_dir / "metrics_dev.json", metrics)
        dump_json(
            out_dir / "config.json",
            {
                "dev_only": True,
                "seed": int(seed),
                "select_dev": select_dev,
                "seq_len": int(seq_len),
                "out_dir": str(out_dir),
                "test_scored": False,
            },
        )
        return metrics

    with torch.no_grad():
        te = model(torch.from_numpy(np.transpose(Xte, (0, 2, 1)))).numpy()
    pte = 1 / (1 + np.exp(-te))
    pred = (pte >= thr).astype(int)
    test_m = binary_metrics(y_tes, pred)
    pd.DataFrame(
        {
            "sample_id": [p.stem for p in tes_p],
            "label": y_tes,
            "prob": pte,
            "pred": pred,
        }
    ).to_csv(out_dir / "predictions.csv", index=False)
    metrics = {
        **common,
        "test_metrics": test_m,
        "selection_rule": "epoch + threshold by DEV F1 only; TEST scored once",
        "note": (
            "Did not retune or rescore the shake rule TEST. "
            "Nod results/ artefacts were not written."
        ),
    }
    dump_json(out_dir / "metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workdir", type=Path, default=ROOT)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--force",
        action="store_true",
        help=(
            "overwrite results/shake/cnn/metrics.json — invalidation only. "
            "Do not pass this to shop TEST scores. Under --dev-only, "
            "overwrites that run's dev_metrics.json in a NEW out-dir."
        ),
    )
    ap.add_argument(
        "--dev-only",
        "--no-test",
        dest="dev_only",
        action="store_true",
        help="select on GOLD DEV only; do not score TEST or write "
             "metrics.json. Requires --out-dir and --pseudo-labels.",
    )
    ap.add_argument(
        "--select-dev",
        choices=("f1", "balanced_accuracy"),
        default="f1",
        help="DEV epoch/threshold criterion. New shake search should use "
             "balanced_accuracy (F1 alone rewards always-shake).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory. Required with --dev-only (must not be the "
             "locked results/shake/cnn/).",
    )
    ap.add_argument(
        "--pseudo-labels",
        type=Path,
        default=None,
        help="existing pseudo CSV (read-only). If set, does not write "
             "results/shake/pseudo_labels.csv.",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="resampled pose length (default 128; 64 is the cheap window variant)",
    )
    args = ap.parse_args(argv)
    work = args.workdir.resolve()

    sys.path.insert(0, str(work / "scripts"))
    import check_split_leakage

    out_dir: Path | None = None
    if args.dev_only:
        if args.out_dir is None:
            raise SystemExit(
                "STOP: --dev-only/--no-test requires --out-dir (a new path "
                "under results/shake/dev_balanced/, never results/shake/cnn/)."
            )
        out_dir = check_split_leakage.assert_unlocked_out_dir(
            args.out_dir if args.out_dir.is_absolute() else work / args.out_dir
        )
        if (out_dir / "metrics.json").exists():
            raise SystemExit(
                f"STOP: --dev-only refuses {out_dir / 'metrics.json'} "
                "(TEST-style). Use a new out-dir."
            )
        already = (out_dir / "dev_metrics.json").exists() or (
            out_dir / "metrics_dev.json"
        ).exists()
        if already and not args.force:
            raise SystemExit(
                f"STOP: DEV-only metrics already exist under {out_dir}."
            )
        if args.pseudo_labels is None:
            raise SystemExit(
                "STOP: --dev-only requires --pseudo-labels pointing at a "
                "new manifest (will not overwrite results/shake/pseudo_labels.csv)."
            )
    else:
        cnn_metrics = work / CNN_DIR_REL / "metrics.json"
        if cnn_metrics.exists() and not args.force:
            raise SystemExit(
                f"STOP: {cnn_metrics} already exists — shake CNN TEST has "
                "already been scored once. Pass --force only if that run is "
                "being formally invalidated (record why). --force is not for "
                "retrying TEST until the number looks better."
            )
        out_dir = work / CNN_DIR_REL

    rule = load_frozen_shake_rule(work)
    from run_shake_experiment import load_shake_gold
    gold = load_shake_gold()
    gold["split"] = gold["split"].astype(str).str.upper()

    locked_pseudo = work / "results" / "shake" / "pseudo_labels.csv"
    if args.pseudo_labels is not None:
        pl = args.pseudo_labels if args.pseudo_labels.is_absolute() else work / args.pseudo_labels
        pl = pl.resolve()
        if not pl.exists():
            raise SystemExit(f"STOP: missing --pseudo-labels {pl}")
        df = pd.read_csv(pl)
        if "pseudo_label" not in df.columns or "sample_id" not in df.columns:
            raise SystemExit(f"STOP: {pl} needs sample_id and pseudo_label")
        pseudo_paths = [
            work / "features" / "pseudo" / f"{sid}.npz"
            for sid in df["sample_id"].astype(str)
        ]
        missing = [p.name for p in pseudo_paths if not p.exists()]
        if missing:
            raise SystemExit(f"STOP: missing pose npz: {missing[:8]}")
        y_tr = df["pseudo_label"].to_numpy(dtype=int)
        print(
            f"using existing {pl} ({int((y_tr == 1).sum())} pos / "
            f"{int((y_tr == 0).sum())} neg); not writing {locked_pseudo}"
        )
        assert_no_leakage(pseudo_paths, gold)
        check_split_leakage.run(
            gold_csv=work / "data" / "gold" / "shake_annotation_sheet.csv",
            pseudo_labels=pl,
            labelled_train_only=True,
        )
    else:
        pseudo_paths = select_pseudo_paths(work / "features" / "pseudo")
        if len(pseudo_paths) < 8:
            raise SystemExit(
                f"STOP: only {len(pseudo_paths)} pseudo clips (need >= 8)"
            )
        video_ids = assert_no_leakage(pseudo_paths, gold)
        y_tr, _ = write_shake_pseudo_labels(
            pseudo_paths, video_ids, rule, locked_pseudo
        )

    if len(pseudo_paths) < 8:
        raise SystemExit(
            f"STOP: only {len(pseudo_paths)} pseudo clips (need >= 8)"
        )

    out = train(
        gold,
        work,
        rule,
        pseudo_paths,
        y_tr,
        epochs=args.epochs,
        seed=args.seed,
        out_dir=out_dir,
        dev_only=bool(args.dev_only),
        select_dev=str(args.select_dev),
        seq_len=int(args.seq_len),
    )
    print("=====================================")
    print("Shake 1D CNN (feature set C = xyz + first differences)")
    print(f"  frozen rule: axis {rule['axis_name']}  thr={rule['selected_amplitude_threshold']:.3f}°")
    print(f"  best epoch (DEV): {out['best_epoch']}   DEV F1: {out['dev_f1']:.3f}")
    if args.dev_only:
        dm = out["dev_metrics"]
        flag = "COLLAPSE" if out.get("collapse") else "ok"
        print(
            f"  DEV P {dm['precision']:.3f}  R {dm['recall']:.3f}  "
            f"F1 {dm['f1']:.3f}  (TP{dm['tp']} FP{dm['fp']} "
            f"TN{dm['tn']} FN{dm['fn']})  [{flag}]"
        )
        print(f"  artefacts: {out_dir}/metrics_dev.json, predictions_dev.csv")
        print("  TEST was not scored.")
        return
    tm = out["test_metrics"]
    print(
        f"  TEST P {tm['precision']:.3f}  R {tm['recall']:.3f}  F1 {tm['f1']:.3f}  "
        f"(TP{tm['tp']} FP{tm['fp']} TN{tm['tn']} FN{tm['fn']})"
    )
    print(
        "  artifacts: results/shake/cnn/metrics.json, "
        "results/shake/cnn/predictions.csv, results/shake/cnn/training_history.csv"
    )
    print("  nod results/classifier_test_metrics.json was not written")


if __name__ == "__main__":
    main()
