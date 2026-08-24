#!/usr/bin/env python3
"""VideoMAE Step 7 (optional, otter95 GPU): fine-tune VideoMAE on the RGB windows.

Fine-tunes ``MCG-NJU/videomae-base`` directly on the cached 16-frame face-crop
windows (``features/rgb16/<sample_id>.npz``, Step 3 output: uint8 ``rgb``
arrays of shape (16, 224, 224, 3)), replacing the frozen-embedding MLP of
Step 5 (``train_videomae_head.py``). Intended for the RTX A4000 (16 GB) with
a CUDA PyTorch build; it also runs on CPU but that is only a smoke path.

Protocol — identical in shape to Step 5 (TEST is never used for any
selection):

* **TRAIN** = pseudo clips that have an rgb16 npz, labelled by the frozen
  DEV-tuned rule (``--pseudo-labels``, default ``results/pseudo_labels.csv``;
  head-shake: ``results/shake/pseudo_labels.csv``).
* **DEV** = the 15 gold DEV clips (``--gold-csv`` / ``--label-col``, default
  nod ``data/gold_annotations.csv`` ``label``; head-shake:
  ``data/gold/shake_annotation_sheet.csv`` ``shake_label``): early stopping
  (best weights restored) AND the probability threshold, both chosen on DEV
  F1 only (threshold swept over ``np.linspace(0.2, 0.8, 13)``, ties broken
  by balanced accuracy).
* **TEST** = the 15 gold TEST clips, **scored exactly once** with the
  best-on-DEV weights and the DEV-chosen threshold, and reported regardless
  of outcome.

Model: ``VideoMAEForVideoClassification`` with ``num_labels=1``
(``ignore_mismatched_sizes=True`` for the new head). The patch embeddings and
the first ``12 - --unfreeze-blocks`` encoder blocks stay frozen; the last
``--unfreeze-blocks`` blocks (default 4) plus the ``fc_norm``/``classifier``
head are trained. AdamW, lr 1e-5 backbone / 1e-4 head, batch 8, up to 15
epochs, ``BCEWithLogitsLoss`` with ``pos_weight = neg/pos`` from TRAIN, seed
42 everywhere, ``torch.autocast`` + ``GradScaler`` when on CUDA.

Preprocessing replicates ``extract_videomae_embeddings.py`` exactly so that
train and inference match: uint8 (16, 224, 224, 3) → float ``/255`` →
``(x - mean) / std`` with the checkpoint's own ``VideoMAEImageProcessorPil``
constants (ImageNet standard), stacked as (batch, frames, channels, H, W) —
the layout ``VideoMAEPatchEmbeddings`` unpacks (it permutes to
(B, C, T, H, W) internally for the Conv3d). Because the crops are already
224×224 the processor's resize/centre-crop are no-ops, and the manual
pipeline is verified against the processor itself on one clip at startup
(the run aborts on any mismatch). Only augmentation: optional horizontal
flip on TRAIN (``--flip``, default on).

The split-leakage gate of ``scripts/check_split_leakage.py`` runs internally
at startup against the **same** gold CSV and pseudo-label file used for
training; any FAIL aborts before training. HF caches stay pinned to the
gitignored ``.hf_cache/`` inside the repo.

Outputs (nod defaults)::

    results/videomae_finetuned/metrics.json
    results/videomae_finetuned/predictions.csv        (all splits)
    results/videomae_finetuned/training_history.csv
    results/videomae_finetuned/best_model.pt          (gitignored)

Head-shake: pass ``--gold-csv data/gold/shake_annotation_sheet.csv``,
``--label-col shake_label``, ``--pseudo-labels results/shake/pseudo_labels.csv``,
and ``--out-dir results/shake/videomae_finetuned`` (or run
``scripts/finetune_videomae_shake.py``). Mixed nod/shake paths abort. Shake
runs never write ``results/videomae_finetuned/`` or nod gold.

TEST-once guard: if ``metrics.json`` already exists under ``--out-dir`` the
script refuses to rerun unless ``--force`` is passed, so TEST cannot be
silently re-scored. Do not pass ``--force`` by default.
A clip whose rgb16 npz is missing is skipped with a warning and counted; if
more than 10% of wanted clips are missing the run aborts before training.
Free space on ``~`` must stay above 5.4 GB, checked at start and before
every checkpoint save.

Run with ``OMP_NUM_THREADS=1`` (same determinism caveat as the pose CNN and
Step 5). Nod invocation (already scored — do not rerun)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    /scratch/db01550/venv/bin/python scripts/finetune_videomae.py

Head-shake on otter95 (RTX A4000, ``/scratch`` venv, **no Docker**)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python scripts/finetune_videomae.py \\
        --gold-csv data/gold/shake_annotation_sheet.csv \\
        --label-col shake_label \\
        --pseudo-labels results/shake/pseudo_labels.csv \\
        --out-dir results/shake/videomae_finetuned
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
RGB16_DIR = ROOT / "features" / "rgb16"
OUT_DIR = ROOT / "results" / "videomae_finetuned"
BEST_PT = OUT_DIR / "best_model.pt"

# Pin HF caches inside the repo BEFORE transformers is imported (lazy import
# happens in main()); .hf_cache/ is gitignored — same rule as the extract step.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

CHECKPOINT = "MCG-NJU/videomae-base"
SEED = 42
EPOCHS = 15
PATIENCE = 5
BATCH = 8
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
UNFREEZE_BLOCKS = 4  # of the 12 encoder blocks; the first 8 stay frozen
MIN_FREE_GB = 5.4
MAX_MISSING_FRAC = 0.10
MIN_TRAIN = 8
MIN_EVAL = 3
PREPROCESS_TOL = 1e-3  # max |manual - HF processor| allowed on the check clip


def free_gb() -> float:
    return shutil.disk_usage(Path.home()).free / 1024**3


def check_disk(where: str = "") -> None:
    free = free_gb()
    if free < MIN_FREE_GB:
        raise SystemExit(
            f"STOP: free disk on ~ is {free:.2f} GB < {MIN_FREE_GB} GB"
            f"{' at ' + where if where else ''}. Remove partial artefacts "
            "before rerunning."
        )


def load_rgb(sample_id: str) -> np.ndarray | None:
    path = RGB16_DIR / f"{sample_id}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        rgb = z["rgb"]
    if rgb.shape != (16, 224, 224, 3) or rgb.dtype != np.uint8:
        raise SystemExit(
            f"STOP: {path.name} has rgb shape {rgb.shape} dtype {rgb.dtype}, "
            "expected (16, 224, 224, 3) uint8."
        )
    return rgb


def build_split(sample_ids: list[str], labels: list[int], name: str):
    clips, ys, kept, missing = [], [], [], []
    for sid, y in zip(sample_ids, labels):
        rgb = load_rgb(sid)
        if rgb is None:
            missing.append(sid)
            continue
        clips.append(rgb)
        ys.append(int(y))
        kept.append(sid)
    if missing:
        print(f"NOTE: {name}: {len(missing)} clips have no rgb16 npz and are "
              f"excluded: {missing}")
    return clips, np.asarray(ys, dtype=np.int64), kept, missing


def best_threshold(y: np.ndarray, prob: np.ndarray):
    """DEV F1 sweep, ties broken by balanced accuracy (Step 5 rule)."""
    thr_best, f1_best, bal_best = 0.5, -1.0, -1.0
    for t in np.linspace(0.2, 0.8, 13):
        m = binary_metrics(y, (prob >= t).astype(int))
        if m["f1"] > f1_best or (
            m["f1"] == f1_best and m["balanced_accuracy"] > bal_best
        ):
            thr_best, f1_best, bal_best = (
                float(t), m["f1"], m["balanced_accuracy"],
            )
    return thr_best, f1_best, bal_best


def resolve_repo_path(path: Path) -> Path:
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
    """Run scripts/check_split_leakage.py's asserts; SystemExit on any FAIL."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_split_leakage

    check_split_leakage.run(
        gold_csv=gold_csv,
        pseudo_labels=pseudo_labels,
        labelled_train_only=True,
    )


def main(
    argv: list[str] | None = None,
    *,
    gold_csv: Path | str | None = None,
    label_col: str | None = None,
    pseudo_labels: Path | str | None = None,
    out_dir: Path | str | None = None,
    force: bool | None = None,
    unfreeze_blocks: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    patience: int | None = None,
    flip: bool | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="allow re-scoring TEST (overwrites metrics.json)")
    parser.add_argument("--unfreeze-blocks", type=int, default=UNFREEZE_BLOCKS,
                        help="train the last N encoder blocks (+ head); "
                             "patch embeddings and earlier blocks stay frozen "
                             f"(default {UNFREEZE_BLOCKS})")
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="early-stopping patience on DEV F1")
    parser.add_argument("--flip", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="horizontal-flip augmentation on TRAIN "
                             "(default on; use --no-flip to disable)")
    parser.add_argument("--gold-csv", type=Path, default=GOLD_CSV,
                        help="gold CSV with split + label column (default: "
                             "data/gold_annotations.csv). Head-shake: "
                             "data/gold/shake_annotation_sheet.csv")
    parser.add_argument("--label-col", default="label",
                        help="DEV/TEST label column (default: label = nod). "
                             "Head-shake: shake_label")
    parser.add_argument("--pseudo-labels", type=Path, default=PSEUDO_LABELS,
                        help="CSV of sample_id,pseudo_label (default: "
                             "results/pseudo_labels.csv, the 80-clip nod run). "
                             "Head-shake: results/shake/pseudo_labels.csv")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="output directory (default: "
                             "results/videomae_finetuned). Head-shake MUST use "
                             "results/shake/videomae_finetuned so nod TEST is "
                             "never overwritten.")
    args = parser.parse_args(argv)

    gold_csv_path = resolve_repo_path(
        gold_csv if gold_csv is not None else args.gold_csv
    )
    pseudo_labels_path = resolve_repo_path(
        pseudo_labels if pseudo_labels is not None else args.pseudo_labels
    )
    out_dir = resolve_repo_path(out_dir if out_dir is not None else args.out_dir)
    best_pt = out_dir / "best_model.pt"
    label_col = str(
        args.label_col if label_col is None else label_col
    ).strip()
    if force is None:
        force = args.force
    if unfreeze_blocks is None:
        unfreeze_blocks = args.unfreeze_blocks
    if batch_size is None:
        batch_size = args.batch_size
    if epochs is None:
        epochs = args.epochs
    if patience is None:
        patience = args.patience
    if flip is None:
        flip = args.flip

    sys.path.insert(0, str(ROOT / "scripts"))
    import check_split_leakage
    task = check_split_leakage.assert_videomae_task_isolation(
        gold_csv=gold_csv_path,
        label_col=label_col,
        pseudo_labels=pseudo_labels_path,
        out_dir=out_dir,
        model_pt=best_pt,
    )
    print(
        f"task={task}  gold={gold_csv_path}  label_col={label_col}\n"
        f"pseudo={pseudo_labels_path}  out_dir={out_dir}"
    )

    if (out_dir / "metrics.json").exists() and not force:
        raise SystemExit(
            f"STOP: {out_dir / 'metrics.json'} already exists — TEST has "
            "already been scored once under this protocol. Pass --force only "
            "if the earlier run is being formally invalidated (record why in "
            "reports/dissertation_evidence/experiment_log.md)."
        )
    for needed in (gold_csv_path, pseudo_labels_path):
        if not needed.exists():
            raise SystemExit(
                f"STOP: {needed} is missing. Run Steps 3-6 (fetch + extract + "
                "frozen rule) first; pseudo labels come from the rule run."
            )
    if not RGB16_DIR.exists() or not any(RGB16_DIR.glob("*.npz")):
        raise SystemExit(
            f"STOP: {RGB16_DIR} has no npz files. Run "
            "scripts/fetch_rgb_windows.py (Step 3) first."
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

    tr_clips, y_tr, train_ids, miss_tr = build_split(
        pseudo["sample_id"].tolist(), pseudo["pseudo_label"].tolist(), "TRAIN"
    )
    dv_clips, y_dv, dev_ids, miss_dv = build_split(
        dev["sample_id"].tolist(), gold_y(dev, label_col, gold_csv_path), "DEV"
    )
    te_clips, y_te, tes_ids, miss_te = build_split(
        tes["sample_id"].tolist(), gold_y(tes, label_col, gold_csv_path), "TEST"
    )

    wanted = len(pseudo) + len(dev) + len(tes)
    missing_all = miss_tr + miss_dv + miss_te
    if wanted and len(missing_all) / wanted > MAX_MISSING_FRAC:
        raise SystemExit(
            f"BLOCKED: {len(missing_all)}/{wanted} wanted clips "
            f"({100 * len(missing_all) / wanted:.0f}%) have no "
            f"features/rgb16 npz (> {int(MAX_MISSING_FRAC * 100)}%). Run "
            "scripts/fetch_rgb_windows.py for the missing ids first: "
            f"{missing_all}. No metrics fabricated; paste this back."
        )
    if len(y_tr) < MIN_TRAIN or len(np.unique(y_tr)) < 2:
        raise SystemExit(
            f"BLOCKED: TRAIN has {len(y_tr)} usable pseudo clips (need >= "
            f"{MIN_TRAIN} with both classes). No metrics fabricated; paste "
            "this back."
        )
    if len(y_dv) < MIN_EVAL or len(y_te) < MIN_EVAL:
        raise SystemExit(
            f"BLOCKED: DEV/TEST usable clips {len(y_dv)}/{len(y_te)} (need "
            f">= {MIN_EVAL} each). No metrics fabricated; paste this back."
        )

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
            "BLOCKED: transformers/torch import failed "
            f"({exc}). On otter48 install the GPU stack with `pip install "
            "--no-cache-dir torch --index-url "
            "https://download.pytorch.org/whl/cu126` in the existing venv; "
            "transformers must stay at the Step 4 version (5.15.1)."
        ) from exc

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # no-op without CUDA
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"device: cuda ({torch.cuda.get_device_name(0)}) — "
              "AMP autocast + GradScaler enabled")
    else:
        print("device: cpu (WARNING: no CUDA GPU visible; fine-tuning on CPU "
              "is a smoke path only)")

    def preprocess(rgb_u8: np.ndarray, mean_t, std_t):
        """uint8 (16, 224, 224, 3) -> float (16, 3, 224, 224) normalised.

        Identical to the HF processor on these already-224×224 crops:
        rescale 1/255 then (x - mean) / std; resize/centre-crop are no-ops.
        """
        x = torch.from_numpy(np.ascontiguousarray(rgb_u8))
        x = x.to(torch.float32).div_(255.0)
        x = (x - mean_t) / std_t  # broadcast over the channel-last dim
        return x.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

    class ClipDataset(Dataset):
        def __init__(self, clips, labels, mean_t, std_t, flip):
            self.clips, self.labels = clips, labels
            self.mean_t, self.std_t, self.flip = mean_t, std_t, flip

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            x = preprocess(self.clips[i], self.mean_t, self.std_t)
            if self.flip and torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-1])  # horizontal flip (width)
            return x, float(self.labels[i])

    def predict_probs(model, clips, batch_size: int) -> np.ndarray:
        """Eval-mode sigmoid probabilities, no augmentation, fp32."""
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(clips), batch_size):
                xb = torch.stack(
                    [preprocess(c, mean_t, std_t)
                     for c in clips[i : i + batch_size]]
                ).to(device)
                logits = model(pixel_values=xb).logits.squeeze(-1)
                probs.append(1 / (1 + np.exp(-logits.float().cpu().numpy())))
        return np.concatenate(probs) if probs else np.asarray([])

    # ---- model + preprocessing constants from the checkpoint itself ----
    try:
        processor = VideoMAEImageProcessorPil.from_pretrained(CHECKPOINT)
    except Exception as exc:  # PIL backend unavailable -> torchvision backend
        print(f"NOTE: VideoMAEImageProcessorPil load failed ({exc}); using "
              "VideoMAEImageProcessor for the normalisation constants — same "
              "ImageNet mean/std, verified below either way.")
        processor = VideoMAEImageProcessor.from_pretrained(CHECKPOINT)
    proc_name = type(processor).__name__
    mean_t = torch.tensor(processor.image_mean, dtype=torch.float32)
    std_t = torch.tensor(processor.image_std, dtype=torch.float32)

    # Train/inference match is verified, not assumed: the manual pipeline
    # must reproduce the processor's output on a real clip.
    frames = [dv_clips[0][i] for i in range(dv_clips[0].shape[0])]
    ref = processor([frames], return_tensors="pt").pixel_values[0]
    mine = preprocess(dv_clips[0], mean_t, std_t)
    if tuple(ref.shape) != tuple(mine.shape):
        raise SystemExit(
            f"STOP: processor output shape {tuple(ref.shape)} != manual "
            f"{tuple(mine.shape)} — preprocessing mismatch; investigate "
            "before training."
        )
    diff = (ref.float() - mine).abs().max().item()
    if diff > PREPROCESS_TOL:
        raise SystemExit(
            f"STOP: manual preprocessing differs from {proc_name} by "
            f"{diff:.2e} on {dev_ids[0]} (> {PREPROCESS_TOL}) — "
            "train/inference mismatch; investigate before training."
        )
    print(f"preprocessing check vs {proc_name}: max abs diff {diff:.2e} "
          f"on {dev_ids[0]} (OK)")

    config = VideoMAEConfig.from_pretrained(CHECKPOINT)
    config.num_labels = 1
    try:
        model = VideoMAEForVideoClassification.from_pretrained(
            CHECKPOINT, config=config, ignore_mismatched_sizes=True
        )
    except Exception as exc:
        raise SystemExit(
            f"BLOCKED: checkpoint {CHECKPOINT} could not be loaded ({exc}). "
            "Partial cache may exist under .hf_cache/ — remove it before "
            "retrying. Paste this back."
        ) from exc
    check_disk("post-download")
    model.to(device)

    # ---- freeze patch embeddings + first (n - k) blocks; train rest ----
    layers = model.videomae.encoder.layer
    n_layers = len(layers)
    k = unfreeze_blocks
    if not 0 <= k <= n_layers:
        raise SystemExit(
            f"STOP: --unfreeze-blocks {k} out of range 0..{n_layers}."
        )
    for p in model.parameters():
        p.requires_grad_(False)
    head_modules = [model.classifier] + (
        [model.fc_norm] if model.fc_norm is not None else []
    )
    for module in head_modules:
        for p in module.parameters():
            p.requires_grad_(True)
    trained_layers = list(layers[n_layers - k:]) if k else []
    for layer in trained_layers:
        for p in layer.parameters():
            p.requires_grad_(True)

    head_params = [p for m in head_modules for p in m.parameters()]
    backbone_params = [p for layer in trained_layers for p in layer.parameters()]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": LR_BACKBONE})
    groups.append({"params": head_params, "lr": LR_HEAD})
    opt = torch.optim.AdamW(groups)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"{CHECKPOINT}: {n_layers} encoder blocks; training head + last "
          f"{k} block(s) ({n_trainable / 1e6:.1f}M of {n_total / 1e6:.1f}M "
          f"params); transformers {transformers.__version__}, "
          f"torch {torch.__version__}")

    pos = max(int((y_tr == 1).sum()), 1)
    neg = max(int((y_tr == 0).sum()), 1)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / pos], dtype=torch.float32).to(device)
    )
    print(f"TRAIN {len(y_tr)} clips ({pos} pos / {neg} neg), "
          f"pos_weight={neg / pos:.3f}; DEV {len(y_dv)}, TEST {len(y_te)}")

    ds = ClipDataset(tr_clips, y_tr, mean_t, std_t, flip=flip)
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        generator=gen)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history: list[dict] = []
    best: dict | None = None
    best_state = None
    bad = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    logits = model(pixel_values=xb).logits.squeeze(-1)
                    loss = crit(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(pixel_values=xb).logits.squeeze(-1)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
            losses.append(float(loss.item()))

        prob_dv = predict_probs(model, dv_clips, batch_size)
        thr_best, f1_best, bal_best = best_threshold(y_dv, prob_dv)
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
            best_state = {k_: v.detach().cpu().clone()
                          for k_, v in model.state_dict().items()}
            check_disk("best_model.pt save")
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, best_pt)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        raise SystemExit("BLOCKED: training produced no epochs; nothing saved.")

    # ---- single TEST scoring with best-on-DEV weights + DEV threshold ----
    model.load_state_dict(best_state)
    thr = best["dev_probability_threshold"]
    prob_dv = predict_probs(model, dv_clips, batch_size)
    prob_tr = predict_probs(model, tr_clips, batch_size)
    prob_te = predict_probs(model, te_clips, batch_size)
    dev_metrics = binary_metrics(y_dv, (prob_dv >= thr).astype(int))
    test_metrics = binary_metrics(y_te, (prob_te >= thr).astype(int))

    check_disk("write")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame(
        {
            "clip_id": train_ids + dev_ids + tes_ids,
            "prob": np.concatenate([prob_tr, prob_dv, prob_te]),
            "pred": (np.concatenate([prob_tr, prob_dv, prob_te]) >= thr
                     ).astype(int),
            "label": np.concatenate([y_tr, y_dv, y_te]),
            "split": (["TRAIN"] * len(train_ids)
                      + ["DEV"] * len(dev_ids)
                      + ["TEST"] * len(tes_ids)),
        }
    ).to_csv(out_dir / "predictions.csv", index=False)
    metrics = {
        "task": task,
        "model": f"VideoMAE fine-tuned (last {k} blocks + head)",
        "script": Path(__file__).name,
        "gold_csv": str(gold_csv_path),
        "label_col": label_col,
        "pseudo_labels": str(pseudo_labels_path),
        "out_dir": str(out_dir),
        "checkpoint": CHECKPOINT,
        "processor": proc_name,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": device.type,
        "seed": SEED,
        "unfreeze_blocks": k,
        "lr_backbone": LR_BACKBONE,
        "lr_head": LR_HEAD,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "patience": patience,
        "flip_augmentation": bool(flip),
        "train_ids": train_ids,
        "train_n": int(len(y_tr)),
        "train_pos": int((y_tr == 1).sum()),
        "train_neg": int((y_tr == 0).sum()),
        "dev_n": int(len(y_dv)),
        "test_n": int(len(y_te)),
        "skipped_missing_rgb16": {
            "TRAIN": miss_tr, "DEV": miss_dv, "TEST": miss_te,
        },
        "pos_weight": neg / pos,
        "best_epoch": int(best["epoch"]),
        "dev_f1": float(best["dev_f1"]),
        "dev_balanced_accuracy": float(best["dev_balanced_accuracy"]),
        "dev_probability_threshold": float(thr),
        "dev_metrics": dev_metrics,
        "selection_rule": "epoch + threshold by DEV F1 only; TEST scored once",
        "test_metrics": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"\nbest epoch {best['epoch']}  DEV F1={best['dev_f1']:.3f}  "
        f"threshold={thr:.2f}\n"
        f"wrote {out_dir}/metrics.json, predictions.csv, "
        f"training_history.csv (+ best_model.pt, gitignored)\n"
        f"TEST (scored once): {test_metrics}"
    )


if __name__ == "__main__":
    main()
