#!/usr/bin/env python3
"""Joint two-head VideoMAE: one backbone, nod + shake, same 30 gold videos.

Shared ``MCG-NJU/videomae-base`` encoder (last 4 blocks unfrozen, same recipe
as the single-task fine-tune). Two independent binary heads.

* **TRAIN** = the 80 rgb16 clips whose ids appear in **both**
  ``results/pseudo_labels.csv`` (nod frozen-rule 0/1) and
  ``results/shake/pseudo_labels.csv`` (shake frozen-rule 0/1). Pseudo-labels
  are not gold. The two CSVs share ``pseudo_00001``…``pseudo_00080`` names
  (same RGB windows; different rule targets).
* **DEV/TEST** = ``data/gold/shake_annotation_sheet.csv`` columns
  ``nod_label`` and ``shake_label`` (same 15/15 videos). Early stopping on
  the mean of the two DEV F1s. Independent DEV thresholds per head.
* **TEST** scored once per head.

Writes only ``results/joint/videomae_finetuned/``. Refuses locked nod/shake
VideoMAE dirs. GPU (RTX A4000).

Otter95 (``/scratch`` CUDA venv, **no Docker**)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/finetune_videomae_joint.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

from src.metrics import binary_metrics  # noqa: E402
from src.utils import dump_json  # noqa: E402

SHAKE_GOLD = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
NOD_GOLD = ROOT / "data" / "gold_annotations.csv"
NOD_PSEUDO = ROOT / "results" / "pseudo_labels.csv"
SHAKE_PSEUDO = ROOT / "results" / "shake" / "pseudo_labels.csv"
OUT_DIR = ROOT / "results" / "joint" / "videomae_finetuned"
CHECKPOINT = "MCG-NJU/videomae-base"
SEED = 42
EPOCHS = 15
PATIENCE = 5
BATCH = 8
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
UNFREEZE_BLOCKS = 4
PREPROCESS_TOL = 1e-3
MIN_TRAIN = 8
MIN_EVAL = 3
MAX_MISSING_FRAC = 0.10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--unfreeze-blocks", type=int, default=UNFREEZE_BLOCKS)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--flip", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    import check_split_leakage
    import finetune_videomae as ft

    out_dir = OUT_DIR
    best_pt = out_dir / "best_model.pt"
    check_split_leakage.assert_joint_videomae_paths(
        gold_csv=SHAKE_GOLD,
        nod_pseudo=NOD_PSEUDO,
        shake_pseudo=SHAKE_PSEUDO,
        out_dir=out_dir,
        model_pt=best_pt,
    )
    if (out_dir / "metrics.json").exists() and not args.force:
        raise SystemExit(
            f"STOP: {out_dir / 'metrics.json'} exists — joint TEST already "
            "scored once. Do not --force to shop scores."
        )
    for needed in (SHAKE_GOLD, NOD_PSEUDO, SHAKE_PSEUDO):
        if not needed.exists():
            raise SystemExit(f"STOP: missing {needed}")
    if not ft.RGB16_DIR.exists() or not any(ft.RGB16_DIR.glob("*.npz")):
        raise SystemExit(
            f"STOP: {ft.RGB16_DIR} has no npz. Run fetch_rgb_windows.py on otter."
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
    ft.check_disk("start")

    gold = pd.read_csv(SHAKE_GOLD)
    gold["split"] = gold["split"].astype(str).str.upper()
    for col in ("nod_label", "shake_label"):
        if col not in gold.columns:
            raise SystemExit(
                f"STOP: {SHAKE_GOLD} has no {col}. Joint training needs both "
                "gold columns on the same 30 videos."
            )
    if NOD_GOLD.exists():
        nodg = pd.read_csv(NOD_GOLD)
        merged = gold.merge(
            nodg[["sample_id", "label"]], on="sample_id", how="left"
        )
        mismatch = merged[
            pd.to_numeric(merged["nod_label"], errors="coerce")
            != pd.to_numeric(merged["label"], errors="coerce")
        ]
        if len(mismatch):
            raise SystemExit(
                "STOP: nod_label in shake sheet != label in "
                f"gold_annotations.csv for {mismatch['sample_id'].tolist()}"
            )

    nod_p = pd.read_csv(NOD_PSEUDO)
    shk_p = pd.read_csv(SHAKE_PSEUDO)
    if "pseudo_label" not in nod_p.columns or "pseudo_label" not in shk_p.columns:
        raise SystemExit("STOP: both pseudo CSVs need a pseudo_label column.")
    train = nod_p.merge(
        shk_p[["sample_id", "pseudo_label"]].rename(
            columns={"pseudo_label": "shake_pseudo"}
        ),
        on="sample_id",
        how="inner",
    ).rename(columns={"pseudo_label": "nod_pseudo"})
    if len(train) < MIN_TRAIN:
        raise SystemExit(
            f"STOP: only {len(train)} clips in the inner join of nod and "
            "shake pseudo CSVs (need the same 80 rgb16 ids). Cannot train."
        )
    print(
        f"TRAIN inner join {len(train)} clips  "
        f"nod {int((train.nod_pseudo == 1).sum())} pos / "
        f"{int((train.nod_pseudo == 0).sum())} neg;  "
        f"shake {int((train.shake_pseudo == 1).sum())} pos / "
        f"{int((train.shake_pseudo == 0).sum())} neg"
    )

    dev = gold[gold.split == "DEV"].sort_values("sample_id")
    tes = gold[gold.split == "TEST"].sort_values("sample_id")

    tr_clips, _, train_ids, miss_tr = ft.build_split(
        train["sample_id"].tolist(),
        train["nod_pseudo"].tolist(),
        "TRAIN",
    )
    # rebuild labels aligned to kept ids
    tr_map_n = dict(zip(train["sample_id"].astype(str), train["nod_pseudo"].astype(int)))
    tr_map_s = dict(zip(train["sample_id"].astype(str), train["shake_pseudo"].astype(int)))
    y_tr_n = np.asarray([tr_map_n[s] for s in train_ids], dtype=np.int64)
    y_tr_s = np.asarray([tr_map_s[s] for s in train_ids], dtype=np.int64)

    dv_clips, y_dv_n, dev_ids, miss_dv = ft.build_split(
        dev["sample_id"].tolist(),
        ft.gold_y(dev, "nod_label", SHAKE_GOLD),
        "DEV-nod",
    )
    te_clips, y_te_n, tes_ids, miss_te = ft.build_split(
        tes["sample_id"].tolist(),
        ft.gold_y(tes, "nod_label", SHAKE_GOLD),
        "TEST-nod",
    )
    smap_dv = dict(
        zip(dev["sample_id"].astype(str), ft.gold_y(dev, "shake_label", SHAKE_GOLD))
    )
    smap_te = dict(
        zip(tes["sample_id"].astype(str), ft.gold_y(tes, "shake_label", SHAKE_GOLD))
    )
    y_dv_s = np.asarray([smap_dv[s] for s in dev_ids], dtype=np.int64)
    y_te_s = np.asarray([smap_te[s] for s in tes_ids], dtype=np.int64)

    wanted = len(train) + len(dev) + len(tes)
    missing_all = miss_tr + miss_dv + miss_te
    if wanted and len(missing_all) / wanted > MAX_MISSING_FRAC:
        raise SystemExit(
            f"BLOCKED: {len(missing_all)}/{wanted} clips missing rgb16. "
            f"{missing_all}"
        )
    if len(y_tr_n) < MIN_TRAIN or len(np.unique(y_tr_n)) < 2 or len(np.unique(y_tr_s)) < 2:
        raise SystemExit(
            "BLOCKED: TRAIN needs both classes on both heads after rgb filter."
        )
    if len(y_dv_n) < MIN_EVAL or len(y_te_n) < MIN_EVAL:
        raise SystemExit("BLOCKED: DEV/TEST too small after rgb filter.")

    try:
        import torch
        import transformers
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            VideoMAEConfig,
            VideoMAEForVideoClassification,
            VideoMAEImageProcessor,
            VideoMAEImageProcessorPil,
        )
    except ImportError as exc:
        raise SystemExit(
            f"BLOCKED: transformers/torch import failed ({exc}). "
            "This job needs the otter95 /scratch CUDA venv."
        ) from exc

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("device: cpu (WARNING: joint fine-tune on CPU is a smoke path)")
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"device: cuda ({torch.cuda.get_device_name(0)})")

    def preprocess(rgb_u8, mean_t, std_t):
        x = torch.from_numpy(np.ascontiguousarray(rgb_u8))
        x = x.to(torch.float32).div_(255.0)
        x = (x - mean_t) / std_t
        return x.permute(0, 3, 1, 2).contiguous()

    class ClipDataset(Dataset):
        def __init__(self, clips, y_n, y_s, mean_t, std_t, flip):
            self.clips, self.y_n, self.y_s = clips, y_n, y_s
            self.mean_t, self.std_t, self.flip = mean_t, std_t, flip

        def __len__(self):
            return len(self.y_n)

        def __getitem__(self, i):
            x = preprocess(self.clips[i], self.mean_t, self.std_t)
            if self.flip and torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-1])
            return x, float(self.y_n[i]), float(self.y_s[i])

    class TwoHead(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.videomae = base.videomae
            self.fc_norm = base.fc_norm
            hidden = int(base.config.hidden_size)
            self.head_nod = nn.Linear(hidden, 1)
            self.head_shake = nn.Linear(hidden, 1)

        def encode(self, pixel_values):
            out = self.videomae(pixel_values)
            seq = out.last_hidden_state
            if self.fc_norm is not None:
                return self.fc_norm(seq.mean(1))
            return seq[:, 0]

        def forward(self, pixel_values):
            h = self.encode(pixel_values)
            return (
                self.head_nod(h).squeeze(-1),
                self.head_shake(h).squeeze(-1),
            )

    try:
        processor = VideoMAEImageProcessorPil.from_pretrained(CHECKPOINT)
    except Exception as exc:
        print(f"NOTE: PIL processor failed ({exc}); using VideoMAEImageProcessor")
        processor = VideoMAEImageProcessor.from_pretrained(CHECKPOINT)
    mean_t = torch.tensor(processor.image_mean, dtype=torch.float32)
    std_t = torch.tensor(processor.image_std, dtype=torch.float32)

    frames = [dv_clips[0][i] for i in range(dv_clips[0].shape[0])]
    ref = processor([frames], return_tensors="pt").pixel_values[0]
    mine = preprocess(dv_clips[0], mean_t, std_t)
    diff = (ref.float() - mine).abs().max().item()
    if diff > PREPROCESS_TOL:
        raise SystemExit(f"STOP: preprocess mismatch {diff:.2e}")
    print(f"preprocessing check OK (max abs diff {diff:.2e})")

    config = VideoMAEConfig.from_pretrained(CHECKPOINT)
    config.num_labels = 1
    base = VideoMAEForVideoClassification.from_pretrained(
        CHECKPOINT, config=config, ignore_mismatched_sizes=True
    )
    model = TwoHead(base).to(device)

    layers = model.videomae.encoder.layer
    n_layers = len(layers)
    k = int(args.unfreeze_blocks)
    for p in model.parameters():
        p.requires_grad_(False)
    for module in [model.head_nod, model.head_shake] + (
        [model.fc_norm] if model.fc_norm is not None else []
    ):
        for p in module.parameters():
            p.requires_grad_(True)
    trained_layers = list(layers[n_layers - k:]) if k else []
    for layer in trained_layers:
        for p in layer.parameters():
            p.requires_grad_(True)
    head_params = list(model.head_nod.parameters()) + list(
        model.head_shake.parameters()
    )
    if model.fc_norm is not None:
        head_params += list(model.fc_norm.parameters())
    backbone_params = [p for layer in trained_layers for p in layer.parameters()]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": LR_BACKBONE})
    groups.append({"params": head_params, "lr": LR_HEAD})
    opt = torch.optim.AdamW(groups)
    print(
        f"joint two-head: last {k} blocks + 2 classifiers; "
        f"transformers {transformers.__version__}, torch {torch.__version__}"
    )

    def pos_w(y):
        pos = max(int((y == 1).sum()), 1)
        neg = max(int((y == 0).sum()), 1)
        return neg / pos

    crit_n = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w(y_tr_n)], dtype=torch.float32).to(device)
    )
    crit_s = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w(y_tr_s)], dtype=torch.float32).to(device)
    )

    def predict_logits(clips):
        model.eval()
        ln, ls = [], []
        with torch.no_grad():
            for i in range(0, len(clips), args.batch_size):
                xb = torch.stack(
                    [preprocess(c, mean_t, std_t)
                     for c in clips[i : i + args.batch_size]]
                ).to(device)
                a, b = model(xb)
                ln.append(a.float().cpu().numpy())
                ls.append(b.float().cpu().numpy())
        pn = 1 / (1 + np.exp(-np.concatenate(ln)))
        ps = 1 / (1 + np.exp(-np.concatenate(ls)))
        return pn, ps

    ds = ClipDataset(
        tr_clips, y_tr_n, y_tr_s, mean_t, std_t, flip=args.flip
    )
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=gen)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = []
    best = None
    best_state = None
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yn, ys in loader:
            xb = xb.to(device, non_blocking=True)
            yn = yn.to(device)
            ys = ys.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    ln, ls = model(xb)
                    loss = crit_n(ln, yn) + crit_s(ls, ys)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                ln, ls = model(xb)
                loss = crit_n(ln, yn) + crit_s(ls, ys)
                loss.backward()
                opt.step()
            losses.append(float(loss.item()))
        pn, ps = predict_logits(dv_clips)
        thr_n, f1_n, bal_n = ft.best_threshold(y_dv_n, pn)
        thr_s, f1_s, bal_s = ft.best_threshold(y_dv_s, ps)
        mean_f1 = 0.5 * (f1_n + f1_s)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses) if losses else 0.0),
            "dev_f1_nod": f1_n,
            "dev_f1_shake": f1_s,
            "dev_f1_mean": mean_f1,
            "dev_threshold_nod": thr_n,
            "dev_threshold_shake": thr_s,
            "dev_balanced_accuracy_nod": bal_n,
            "dev_balanced_accuracy_shake": bal_s,
        }
        history.append(row)
        print(
            f"epoch {epoch} loss={row['train_loss']:.4f}  "
            f"DEV F1 nod={f1_n:.3f} shake={f1_s:.3f} mean={mean_f1:.3f}"
        )
        if best is None or mean_f1 > best["dev_f1_mean"]:
            best = {**row}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            ft.check_disk("best_model.pt save")
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, best_pt)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is None:
        raise SystemExit("BLOCKED: joint training produced no epochs.")
    model.load_state_dict(best_state)
    pn_dv, ps_dv = predict_logits(dv_clips)
    pn_tr, ps_tr = predict_logits(tr_clips)
    pn_te, ps_te = predict_logits(te_clips)
    thr_n = best["dev_threshold_nod"]
    thr_s = best["dev_threshold_shake"]
    nod_dev = binary_metrics(y_dv_n, (pn_dv >= thr_n).astype(int))
    shk_dev = binary_metrics(y_dv_s, (ps_dv >= thr_s).astype(int))
    nod_te = binary_metrics(y_te_n, (pn_te >= thr_n).astype(int))
    shk_te = binary_metrics(y_te_s, (ps_te >= thr_s).astype(int))

    ft.check_disk("write")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    rows = []
    for split, ids, pn, ps, yn, ys in (
        ("TRAIN", train_ids, pn_tr, ps_tr, y_tr_n, y_tr_s),
        ("DEV", dev_ids, pn_dv, ps_dv, y_dv_n, y_dv_s),
        ("TEST", tes_ids, pn_te, ps_te, y_te_n, y_te_s),
    ):
        for i, sid in enumerate(ids):
            rows.append(
                {
                    "sample_id": sid,
                    "split": split,
                    "nod_label": int(yn[i]),
                    "shake_label": int(ys[i]),
                    "nod_prob": float(pn[i]),
                    "shake_prob": float(ps[i]),
                    "nod_pred": int(pn[i] >= thr_n),
                    "shake_pred": int(ps[i] >= thr_s),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    metrics = {
        "task": "joint_nod_shake",
        "model": f"VideoMAE two-head fine-tune (last {k} blocks + nod/shake heads)",
        "script": Path(__file__).name,
        "gold_csv": str(SHAKE_GOLD),
        "nod_pseudo": str(NOD_PSEUDO),
        "shake_pseudo": str(SHAKE_PSEUDO),
        "out_dir": str(out_dir),
        "checkpoint": CHECKPOINT,
        "device": device.type,
        "seed": SEED,
        "unfreeze_blocks": k,
        "train_n": int(len(y_tr_n)),
        "train_nod_pos": int((y_tr_n == 1).sum()),
        "train_nod_neg": int((y_tr_n == 0).sum()),
        "train_shake_pos": int((y_tr_s == 1).sum()),
        "train_shake_neg": int((y_tr_s == 0).sum()),
        "best_epoch": int(best["epoch"]),
        "dev_f1_mean": float(best["dev_f1_mean"]),
        "dev_threshold_nod": float(thr_n),
        "dev_threshold_shake": float(thr_s),
        "dev_metrics_nod": nod_dev,
        "dev_metrics_shake": shk_dev,
        "test_metrics_nod": nod_te,
        "test_metrics_shake": shk_te,
        "selection_rule": (
            "epoch by mean(DEV F1 nod, DEV F1 shake); "
            "independent DEV thresholds; TEST scored once per head"
        ),
        "skipped_missing_rgb16": {
            "TRAIN": miss_tr, "DEV": miss_dv, "TEST": miss_te,
        },
    }
    dump_json(out_dir / "metrics.json", metrics)
    print(
        f"\nbest epoch {best['epoch']}  mean DEV F1={best['dev_f1_mean']:.3f}\n"
        f"TEST nod:    {nod_te}\n"
        f"TEST shake:  {shk_te}\n"
        f"wrote {out_dir}/metrics.json"
    )


if __name__ == "__main__":
    main()
