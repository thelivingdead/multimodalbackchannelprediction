"""1D CNN pose classifier trained on rule pseudo-labels.

This is the learned-model half of the nod experiment, extracted unchanged
from ``scripts/run_full_experiment.py``:

- feature assembly from the committed npz pose clips (feature sets A--D),
- a small 1D CNN over 128-step resampled sequences,
- training on pseudo-labels produced by the frozen DEV-tuned rule,
- epoch and probability-threshold selection on DEV only,
- one TEST evaluation with the best-on-DEV checkpoint.

Selection never uses TEST. Ablation feature set D (rotation + derivatives +
expression) diverged in the recorded run (``loss = nan``); it is written to
``results/ablation_results.csv`` and excluded from reported tables.

Note on numerical reproducibility: CPU training with PyTorch is sensitive to
intra-op thread count. Run with ``OMP_NUM_THREADS=1`` for run-to-run
deterministic results on the same machine; the locked dissertation artifacts
in ``results/`` remain the canonical record of the single TEST scoring.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import binary_metrics
from .utils import dump_json

SEQ_LEN = 128  # 60 s clip CNN only. Windowed 3 s LOCO uses 75 frames at 25 fps.
DEFAULT_KERNELS = (5, 5, 3)
WINDOWED_FPS = 25.0


def conv_receptive_field(kernels) -> int:
    """Stride-1 stacked Conv1d receptive field: 1 + sum(k_i - 1)."""
    ks = tuple(int(k) for k in kernels)
    if not ks:
        raise ValueError("kernels must not be empty")
    return 1 + sum(k - 1 for k in ks)


def conv_paddings(kernels) -> tuple[int, ...]:
    """Same-length padding for odd kernels: padding = kernel // 2."""
    return tuple(int(k) // 2 for k in kernels)


def receptive_field_seconds(kernels, fps: float = WINDOWED_FPS) -> float:
    return conv_receptive_field(kernels) / float(fps)


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def resample_seq(x: np.ndarray, t: int = SEQ_LEN) -> np.ndarray:
    n = len(x)
    if n == 0:
        return np.zeros((t,) + x.shape[1:], dtype=np.float32)
    old = np.linspace(0, 1, n)
    new = np.linspace(0, 1, t)
    if x.ndim == 1:
        return np.interp(new, old, x).astype(np.float32)
    cols = [np.interp(new, old, x[:, j]) for j in range(x.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def build_matrix(
    paths: list[Path],
    mode: str,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    seq_len: int = SEQ_LEN,
):
    xs = []
    t = int(seq_len) if seq_len else SEQ_LEN
    for p in paths:
        z = load_npz(p)
        rot = np.asarray(z["rotation_xyz"], dtype=float)
        drot = np.vstack([np.zeros((1, 3)), np.diff(rot, axis=0)])
        expr = np.asarray(z["expression"], dtype=float)
        if mode == "A":
            feat = rot[:, :1]
        elif mode == "B":
            feat = rot
        elif mode == "C":
            feat = np.concatenate([rot, drot], axis=1)
        else:
            feat = np.concatenate([rot, drot, expr], axis=1)
        feat = resample_seq(feat, t=t)
        xs.append(feat)
    X = np.stack(xs).astype(np.float32)
    if mean is None:
        mean = X.mean(axis=(0, 1))
        std = X.std(axis=(0, 1)) + 1e-6
    X = (X - mean) / std
    return X, mean, std


def _torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        print("torch not available; skip classifier:", exc)
        return None
    return torch, nn, DataLoader, TensorDataset


def _build_cnn(nn, d: int, kernels=None):
    """1D pose CNN. Default kernels (5, 5, 3) match the locked windowed LOCO CNN.

    ``kernels`` is an optional length-3 tuple of odd sizes. Padding is
    kernel // 2 so the temporal length is unchanged. Existing callers that
    omit ``kernels`` keep the original 5, 5, 3 stack.
    """
    resolved = tuple(int(k) for k in (DEFAULT_KERNELS if kernels is None else kernels))
    if len(resolved) != 3:
        raise ValueError(f"expected 3 kernels, got {resolved}")
    if any(k < 1 or k % 2 == 0 for k in resolved):
        raise ValueError(
            f"kernels must be positive odd integers so padding keeps length, got {resolved}"
        )
    pads = conv_paddings(resolved)
    k1, k2, k3 = resolved
    p1, p2, p3 = pads

    class PoseCNN1D(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.kernels = resolved
            self.net = nn.Sequential(
                nn.Conv1d(d, 32, k1, padding=p1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(32, 64, k2, padding=p2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(64, 64, k3, padding=p3),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            h = self.net(x)
            return self.fc(h.squeeze(-1)).squeeze(-1)

    return PoseCNN1D(d)


def _save_jpg(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_pseudo_cnn(
    gold: pd.DataFrame,
    work: Path,
    epochs: int,
    seed: int,
    smoke: bool,
    rule_score_fn,
) -> dict | None:
    """Train the pseudo-labelled 1D CNN. ``rule_score_fn(rotation_xyz, axis)``
    must be the frozen DEV-tuned rule amplitude (see run_full_experiment.rule_score).
    Returns the main (feature set C) result dict, or None if training is not possible.
    """
    mods = _torch()
    if mods is None:
        return None
    torch, nn, DataLoader, TensorDataset = mods
    pseudo = sorted((work / "features" / "pseudo").glob("*.npz"))
    if len(pseudo) < 8:
        print(f"skip classifier: only {len(pseudo)} pseudo clips (need >= 8)")
        return None
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = json.loads((work / "results" / "rule_selected_config.json").read_text())
    axis = int(cfg["chosen_rotation_axis"])
    thr = float(cfg["selected_amplitude_threshold"])
    labels = []
    keep = []
    scores = []
    for p in pseudo:
        sc = rule_score_fn(load_npz(p)["rotation_xyz"], axis)
        labels.append(int(sc >= thr))
        scores.append(sc)
        keep.append(p)
    y_tr = np.asarray(labels)
    pd.DataFrame({"sample_id": [p.stem for p in keep], "rule_score": scores, "pseudo_label": labels}).to_csv(
        work / "results" / "pseudo_labels.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["pseudo 0", "pseudo 1"], [int((y_tr == 0).sum()), int((y_tr == 1).sum())])
    _save_jpg(fig, work / "figures" / "pseudo_label_distribution.jpg")
    print("pseudo labels", int((y_tr == 0).sum()), "neg", int((y_tr == 1).sum()), "pos")

    dev = gold[gold.split == "DEV"]
    tes = gold[gold.split == "TEST"]
    dev_p = [work / "features" / "gold" / f"{s}.npz" for s in dev.sample_id if (work / "features" / "gold" / f"{s}.npz").exists()]
    tes_p = [work / "features" / "gold" / f"{s}.npz" for s in tes.sample_id if (work / "features" / "gold" / f"{s}.npz").exists()]
    if len(dev_p) < 3 or len(tes_p) < 3:
        print("skip classifier: not enough gold features for DEV/TEST")
        return None
    y_dev = np.array([int(gold.loc[gold.sample_id == p.stem, "label"].iloc[0]) for p in dev_p])
    y_tes = np.array([int(gold.loc[gold.sample_id == p.stem, "label"].iloc[0]) for p in tes_p])

    def run_mode(mode: str, do_plots: bool) -> dict:
        Xtr, mean, std = build_matrix(keep, mode)
        Xdv, _, _ = build_matrix(dev_p, mode, mean, std)
        Xte, _, _ = build_matrix(tes_p, mode, mean, std)
        dump_json(work / "models" / "normalization.json", {"mean": mean.tolist(), "std": std.tolist(), "mode": mode})
        d = Xtr.shape[-1]
        model = _build_cnn(nn, d)
        pos = max(int((y_tr == 1).sum()), 1)
        neg = max(int((y_tr == 0).sum()), 1)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ds = TensorDataset(torch.from_numpy(np.transpose(Xtr, (0, 2, 1))), torch.from_numpy(y_tr.astype(np.float32)))
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        hist = []
        best = None
        bad = 0
        (work / "models").mkdir(exist_ok=True)
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
            thr_best, f1_best, bal_best = 0.5, -1.0, -1.0
            for t in np.linspace(0.2, 0.8, 13):
                mm = binary_metrics(y_dev, (prob >= t).astype(int))
                if mm["f1"] > f1_best or (mm["f1"] == f1_best and mm["balanced_accuracy"] > bal_best):
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
                torch.save(model.state_dict(), work / "models" / "best_1dcnn.pt")
                bad = 0
            else:
                bad += 1
                if bad >= 4 and not smoke:
                    break
        model.load_state_dict(torch.load(work / "models" / "best_1dcnn.pt", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            te = model(torch.from_numpy(np.transpose(Xte, (0, 2, 1)))).numpy()
        pte = 1 / (1 + np.exp(-te))
        pred = (pte >= best["dev_probability_threshold"]).astype(int)
        test_m = binary_metrics(y_tes, pred)
        if do_plots:
            pd.DataFrame(hist).to_csv(work / "results" / "training_history.csv", index=False)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist])
            ax.set_title("Training loss")
            _save_jpg(fig, work / "figures" / "training_loss.jpg")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot([h["epoch"] for h in hist], [h["dev_f1"] for h in hist])
            ax.set_title("DEV F1 by epoch")
            _save_jpg(fig, work / "figures" / "dev_f1_by_epoch.jpg")
            fig, ax = plt.subplots(figsize=(4, 3.5))
            cm = np.array([[test_m["tn"], test_m["fp"]], [test_m["fn"], test_m["tp"]]])
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center")
            ax.set_title("Classifier TEST confusion")
            _save_jpg(fig, work / "figures" / "classifier_confusion_matrix.jpg")
            dump_json(work / "results" / "classifier_test_metrics.json", test_m)
            pd.DataFrame({"sample_id": [p.stem for p in tes_p], "label": y_tes, "prob": pte, "pred": pred}).to_csv(
                work / "results" / "classifier_test_predictions.csv", index=False
            )
        return {
            "feature_set": mode,
            "input_dimensions": int(d),
            "best_epoch": int(best["epoch"]),
            "dev_f1": float(best["dev_f1"]),
            "dev_probability_threshold": float(best["dev_probability_threshold"]),
            "test_metrics": test_m,
        }

    main = run_mode("C", do_plots=True)
    abl_rows = []
    for mode, name in (("A", "single_axis"), ("B", "xyz"), ("C", "xyz_deriv"), ("D", "xyz_deriv_expr")):
        out = run_mode(mode, do_plots=False) if mode != "C" else main
        tm = out["test_metrics"]
        abl_rows.append(
            {
                "feature_set": name,
                "input_dimensions": out["input_dimensions"],
                "best_epoch": out["best_epoch"],
                "dev_f1": out["dev_f1"],
                "test_accuracy": tm["accuracy"],
                "test_precision": tm["precision"],
                "test_recall": tm["recall"],
                "test_f1": tm["f1"],
                "test_balanced_accuracy": tm["balanced_accuracy"],
            }
        )
    pd.DataFrame(abl_rows).to_csv(work / "results" / "ablation_results.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([r["feature_set"] for r in abl_rows], [r["test_f1"] for r in abl_rows])
    ax.set_ylabel("TEST F1")
    ax.set_title("Ablation TEST F1")
    _save_jpg(fig, work / "figures" / "ablation_f1.jpg")
    return main
