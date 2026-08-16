#!/usr/bin/env python3
"""End-to-end nod experiment: stream EMOCA → rule baseline → pseudo CNN.

Does not download emoca.tar.gz to disk. Does not invent metrics.
TEST is never used for threshold, axis, epoch, or model selection.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import pickle
import shutil
import sys
import tarfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emoca_loader import (  # noqa: E402
    find_expression_array,
    find_pose_array,
    rotvec_to_euler_deg,
)

FPS = 25.0
EXPR_DIM = 20
SEQ_LEN = 128
MIN_FREE_GB = 3.0
EMOCA_URLS = (
    "https://huggingface.co/datasets/scottgeng00/realtalk/resolve/main/emoca.tar.gz",
    "https://huggingface.co/datasets/scottgeng00/realtalk/resolve/main/emoca/emoca.tar.gz",
)


def disk_free_gb(path: Path) -> float:
    u = shutil.disk_usage(path)
    return u.free / (1024**3)


def stop_if_low_disk(path: Path) -> None:
    free = disk_free_gb(path)
    if free < MIN_FREE_GB:
        raise SystemExit(f"STOP: free disk {free:.2f} GB < {MIN_FREE_GB} GB")


def dump_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def save_jpg(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def load_gold(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["sample_id", "video_id", "start_frame", "end_frame", "person", "split", "label"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"gold CSV missing columns {missing}")
    df["split"] = df["split"].astype(str).str.upper()
    df["person"] = df["person"].astype(str)
    df["label"] = df["label"].astype(int)
    df["start_frame"] = df["start_frame"].astype(int)
    df["end_frame"] = df["end_frame"].astype(int)
    return df


def validate_gold(df: pd.DataFrame, results: Path, figures: Path) -> dict:
    errors = []
    if len(df) != 30:
        errors.append(f"expected 30 gold rows, got {len(df)}")
    n_dev = int((df.split == "DEV").sum())
    n_test = int((df.split == "TEST").sum())
    if n_dev != 15 or n_test != 15:
        errors.append(f"expected 15/15 DEV/TEST, got {n_dev}/{n_test}")
    if set(df.label.unique()) - {0, 1}:
        errors.append("labels must be 0/1")
    if set(df.person.unique()) - {"p0", "p1"}:
        errors.append("person must be p0/p1")
    if (df.end_frame <= df.start_frame).any():
        errors.append("end_frame must be > start_frame")
    if df.sample_id.duplicated().any():
        errors.append("sample_id not unique")
    dev_v = set(df.loc[df.split == "DEV", "video_id"])
    tes_v = set(df.loc[df.split == "TEST", "video_id"])
    overlap = sorted(dev_v & tes_v)
    if overlap:
        print("WARNING: DEV/TEST source video overlap:", overlap)
    if errors:
        raise SystemExit("GOLD VALIDATION FAILED:\n" + "\n".join(errors))
    summary = {
        "gold_total": int(len(df)),
        "dev_total": n_dev,
        "test_total": n_test,
        "dev_positives": int(((df.split == "DEV") & (df.label == 1)).sum()),
        "dev_negatives": int(((df.split == "DEV") & (df.label == 0)).sum()),
        "test_positives": int(((df.split == "TEST") & (df.label == 1)).sum()),
        "test_negatives": int(((df.split == "TEST") & (df.label == 0)).sum()),
        "dev_videos": sorted(dev_v),
        "test_videos": sorted(tes_v),
        "video_overlap": overlap,
    }
    dump_json(results / "gold_dataset_summary.json", summary)
    df.to_csv(results / "gold_dataset_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["clear nod", "unclear"], [int((df.label == 1).sum()), int((df.label == 0).sum())])
    ax.set_title("Gold labels (n=30)")
    save_jpg(fig, figures / "gold_label_distribution.jpg")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(
        ["DEV 1", "DEV 0", "TEST 1", "TEST 0"],
        [
            summary["dev_positives"],
            summary["dev_negatives"],
            summary["test_positives"],
            summary["test_negatives"],
        ],
    )
    ax.set_title("15 DEV / 15 TEST")
    save_jpg(fig, figures / "gold_split_distribution.jpg")
    print(
        f"GOLD OK  DEV+ {summary['dev_positives']} DEV- {summary['dev_negatives']} "
        f"TEST+ {summary['test_positives']} TEST- {summary['test_negatives']}"
    )
    return summary


def _as_frame_dict(obj: object) -> dict[int, object]:
    if not isinstance(obj, dict):
        raise ValueError(f"pickle is {type(obj)}, expected frame dict")
    out: dict[int, object] = {}
    for k, v in obj.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def extract_clip_features(obj: object, person: str, start_f: int, end_f: int) -> dict | None:
    frames = _as_frame_dict(obj)
    if not frames:
        return None
    xs, ys, zs, exps, idxs, valid = [], [], [], [], [], []
    schema_keys: list[str] | None = None
    for fi in range(start_f, end_f):
        rec = frames.get(fi)
        idxs.append(fi)
        if not isinstance(rec, dict):
            xs.append(np.nan)
            ys.append(np.nan)
            zs.append(np.nan)
            exps.append(np.full(EXPR_DIM, np.nan))
            valid.append(False)
            continue
        emb = rec.get(person, rec.get(person.lower()))
        if schema_keys is None and isinstance(emb, dict):
            schema_keys = [str(k) for k in list(emb.keys())[:40]]
        aa = find_pose_array(emb)
        if aa is None or aa.size < 3:
            xs.append(np.nan)
            ys.append(np.nan)
            zs.append(np.nan)
            exps.append(np.full(EXPR_DIM, np.nan))
            valid.append(False)
            continue
        x, y, z = rotvec_to_euler_deg(aa[:3])
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ev = find_expression_array(emb)
        if ev is None:
            exps.append(np.zeros(EXPR_DIM))
        else:
            pad = np.zeros(EXPR_DIM)
            n = min(EXPR_DIM, ev.size)
            pad[:n] = ev[:n]
            exps.append(pad)
        valid.append(True)
    rot = np.stack([xs, ys, zs], axis=1).astype(float)
    valid_a = np.asarray(valid, dtype=bool)
    if valid_a.mean() < 0.05:
        return None
    for c in range(3):
        col = rot[:, c]
        ok = np.isfinite(col)
        if ok.any() and (~ok).any():
            idx = np.arange(len(col))
            col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
            rot[:, c] = col
    return {
        "frames": np.asarray(idxs, dtype=np.int32),
        "rotation_xyz": rot.astype(np.float32),
        "expression": np.stack(exps).astype(np.float32),
        "valid_ratio": float(valid_a.mean()),
        "schema_keys": schema_keys or [],
    }


def save_npz(path: Path, feat: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=feat["frames"],
        rotation_xyz=feat["rotation_xyz"],
        expression=feat["expression"],
        valid_ratio=np.array([feat["valid_ratio"]], dtype=np.float32),
        video_id=np.array(meta["video_id"]),
        person=np.array(meta["person"]),
        sample_id=np.array(meta["sample_id"]),
    )


def video_id_from_member(name: str) -> str | None:
    p = Path(name)
    if p.suffix != ".pkl":
        return None
    stem = p.stem
    if len(stem) >= 8:
        return stem
    return None


def stream_emoca(
    gold: pd.DataFrame,
    work: Path,
    n_pseudo: int,
    smoke: bool,
    local_dir: Path | None,
) -> dict:
    feat_gold = work / "features" / "gold"
    feat_pseudo = work / "features" / "pseudo"
    feat_gold.mkdir(parents=True, exist_ok=True)
    feat_pseudo.mkdir(parents=True, exist_ok=True)
    gold_by_vid: dict[str, list[pd.Series]] = {}
    for _, r in gold.iterrows():
        gold_by_vid.setdefault(str(r.video_id), []).append(r)
    wanted = set(gold_by_vid)
    schema: dict | None = None
    quality_rows = []

    def handle_obj(vid: str, obj: object) -> None:
        nonlocal schema
        if vid in wanted:
            for r in gold_by_vid[vid]:
                dest = feat_gold / f"{r.sample_id}.npz"
                if dest.exists():
                    continue
                feat = extract_clip_features(obj, str(r.person), int(r.start_frame), int(r.end_frame))
                if feat is None:
                    print("NO_POSE", r.sample_id, vid)
                    continue
                if schema is None:
                    schema = {"video_id": vid, "keys": feat["schema_keys"]}
                    print("EMOCA keys:", feat["schema_keys"] or "(pose found without dict keys)")
                    if not feat["schema_keys"] and feat["rotation_xyz"] is None:
                        raise SystemExit("STOP: no identifiable pose in EMOCA embedding")
                save_npz(dest, feat, {"video_id": vid, "person": r.person, "sample_id": r.sample_id})
                quality_rows.append(
                    {"sample_id": r.sample_id, "video_id": vid, "split": r.split, "valid_ratio": feat["valid_ratio"]}
                )
                print(f"gold feature {r.sample_id} valid={feat['valid_ratio']:.2f}")
        elif n_pseudo > 0:
            existing = list(feat_pseudo.glob("pseudo_*.npz"))
            if len(existing) >= n_pseudo:
                return
            if any(p.stem.endswith(vid) for p in existing):
                return
            sid = f"pseudo_{len(existing) + 1:05d}"
            dest = feat_pseudo / f"{sid}.npz"
            frames = _as_frame_dict(obj)
            if not frames:
                return
            lo, hi = min(frames), max(frames)
            start = lo
            end = min(hi + 1, lo + int(60 * FPS))
            feat = extract_clip_features(obj, "p0", start, end)
            if feat is None:
                feat = extract_clip_features(obj, "p1", start, end)
                person = "p1"
            else:
                person = "p0"
            if feat is None:
                return
            save_npz(dest, feat, {"video_id": vid, "person": person, "sample_id": sid})
            print(f"pseudo feature {sid} {vid} valid={feat['valid_ratio']:.2f}")

    if local_dir and local_dir.exists():
        print("Using local EMOCA dir", local_dir)
        for pkl in sorted(local_dir.glob("*.pkl")):
            stop_if_low_disk(work)
            vid = pkl.stem
            with pkl.open("rb") as f:
                obj = pickle.load(f)
            handle_obj(vid, obj)
            del obj
        _write_quality(work, quality_rows)
        return {"schema": schema, "source": str(local_dir)}

    import os

    import requests

    class _ProgressStream:
        """Wrap the HTTP body so a stalled CDN is visible and times out."""

        def __init__(self, raw, stall_s: float = 180.0):
            self.raw = raw
            self.stall_s = stall_s
            self.n = 0
            self.t0 = time.time()
            self.last_print = time.time()
            self.last_byte = time.time()

        def read(self, size: int = -1):
            data = self.raw.read(size)
            now = time.time()
            if data:
                self.n += len(data)
                self.last_byte = now
            elif now - self.last_byte > self.stall_s:
                raise TimeoutError(
                    f"HTTP stream stalled: 0 bytes for {now - self.last_byte:.0f}s after {self.n / 1e6:.1f} MB"
                )
            if now - self.last_print >= 30:
                dt = max(now - self.t0, 1.0)
                print(
                    f"stream bytes={self.n / 1e6:.1f} MB elapsed={dt:.0f}s rate={self.n / 1e6 / dt:.2f} MB/s",
                    flush=True,
                )
                self.last_print = now
            return data

        def readinto(self, buf):
            data = self.read(len(buf))
            n = len(data)
            buf[:n] = data
            return n

        def __getattr__(self, name: str):
            return getattr(self.raw, name)

    headers = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_err = None
    # Connect 60s; abort if no TCP data for 180s (was None = hang forever).
    http_timeout = (60, 180)
    for url in EMOCA_URLS:
        print("Streaming", url, "(not saving archive)")
        try:
            resp = requests.get(url, stream=True, headers=headers, timeout=http_timeout, allow_redirects=True)
            if resp.status_code in (401, 403):
                last_err = f"{url} HTTP {resp.status_code} (need HF_TOKEN for gated dataset)"
                print(last_err)
                continue
            if resp.status_code != 200:
                last_err = f"{url} HTTP {resp.status_code}"
                print(last_err)
                continue
            raw = resp.raw
            raw.decode_content = True
            try:
                sock = raw.fp.raw._sock  # type: ignore[attr-defined]
                sock.settimeout(180)
            except Exception:
                pass
            processed = 0
            with tarfile.open(fileobj=_ProgressStream(raw), mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    vid = video_id_from_member(member.name)
                    if vid is None:
                        continue
                    processed += 1
                    n_got = len(list((work / "features" / "gold").glob("*.npz")))
                    if processed % 10 == 0 or member.size >= 50_000_000:
                        print(
                            f"stream members={processed} gold_npz={n_got}/30 "
                            f"file={member.name} size_mb={member.size / 1e6:.1f}",
                            flush=True,
                        )
                    if smoke and processed > 40 and schema is not None:
                        print("smoke-test: stop after schema + sample members")
                        break
                    need_gold = vid in wanted and not all(
                        (work / "features" / "gold" / f"{r.sample_id}.npz").exists() for r in gold_by_vid[vid]
                    )
                    need_pseudo = n_pseudo > 0 and len(list((work / "features" / "pseudo").glob("*.npz"))) < n_pseudo
                    if not need_gold and not (need_pseudo and vid not in wanted):
                        continue
                    stop_if_low_disk(work)
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue
                    payload = fobj.read()
                    try:
                        obj = pickle.loads(payload)
                    except Exception as exc:
                        print("skip pickle", vid, exc)
                        del payload
                        continue
                    del payload
                    handle_obj(vid, obj)
                    del obj
                    gold_left = [
                        r.sample_id
                        for r in gold.itertuples()
                        if not (work / "features" / "gold" / f"{r.sample_id}.npz").exists()
                    ]
                    if not gold_left and len(list((work / "features" / "pseudo").glob("*.npz"))) >= n_pseudo:
                        print("all requested features extracted; stop stream")
                        break
            _write_quality(work, quality_rows)
            return {"schema": schema, "source": url, "members_seen": processed}
        except Exception as exc:
            last_err = str(exc)
            print("stream failed", url, exc)
    _write_quality(work, quality_rows)
    return {"schema": schema, "error": last_err}


def _write_quality(work: Path, rows: list[dict]) -> None:
    dest = work / "results" / "feature_quality.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(dest, index=False)


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def rule_score(rot: np.ndarray, axis: int, min_frames: int = 5, max_frames: int = 50) -> float:
    x = np.asarray(rot[:, axis], dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 11:
        return 0.0
    win = min(11, x.size if x.size % 2 == 1 else x.size - 1)
    win = max(5, win)
    if win % 2 == 0:
        win -= 1
    sm = savgol_filter(x, win, 2)
    d = np.diff(sm)
    turns = np.where(np.diff(np.sign(d)) != 0)[0] + 1
    best = 0.0
    for i, a in enumerate(turns):
        for b in turns[i + 1 :]:
            span = int(b - a)
            if span < min_frames or span > max_frames:
                continue
            amp = float(abs(sm[int(b)] - sm[int(a)]))
            if amp > best:
                best = amp
    if best == 0.0:
        best = float(np.ptp(sm))
    return best


def choose_axis_and_threshold(gold: pd.DataFrame, work: Path) -> dict:
    dev = gold[gold.split == "DEV"].copy()
    scores = {ax: [] for ax in range(3)}
    labels = []
    used = []
    for r in dev.itertuples():
        p = work / "features" / "gold" / f"{r.sample_id}.npz"
        if not p.exists():
            continue
        rot = load_npz(p)["rotation_xyz"]
        labels.append(int(r.label))
        used.append(r.sample_id)
        for ax in range(3):
            scores[ax].append(rule_score(rot, ax))
    if len(labels) < 3:
        raise SystemExit(
            "STOP: fewer than 3 GOLD DEV feature files. "
            "EMOCA stream did not yield pose for the labelled videos. "
            "No rule TEST F1 will be invented."
        )
    y = np.asarray(labels)
    best = None
    search_rows = []
    for ax in range(3):
        s = np.asarray(scores[ax], dtype=float)
        cands = np.unique(np.quantile(s, np.linspace(0.1, 0.9, 17)))
        for thr in cands:
            pred = (s >= thr).astype(int)
            m = binary_metrics(y, pred)
            row = {"axis": ax, "threshold": float(thr), **m}
            search_rows.append(row)
            key = (m["f1"], m["balanced_accuracy"])
            if best is None or key > best[0]:
                best = (key, row)
    pd.DataFrame(search_rows).to_csv(work / "results" / "rule_dev_threshold_search.csv", index=False)
    assert best is not None
    cfg = {
        "chosen_rotation_axis": int(best[1]["axis"]),
        "axis_name": ["x", "y", "z"][int(best[1]["axis"])],
        "smoothing": "savgol_11_2",
        "min_movement_frames": 5,
        "max_movement_frames": 50,
        "selected_amplitude_threshold": float(best[1]["threshold"]),
        "dev_metrics": {k: best[1][k] for k in ("precision", "recall", "f1", "accuracy", "balanced_accuracy")},
        "n_dev_with_features": int(len(labels)),
        "dev_sample_ids": used,
        "note": "Axis and threshold frozen on DEV only. Physical pitch not assumed.",
    }
    dump_json(work / "results" / "rule_selected_config.json", cfg)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sub = [r for r in search_rows if r["axis"] == cfg["chosen_rotation_axis"]]
    ax.plot([r["threshold"] for r in sub], [r["f1"] for r in sub], marker="o")
    ax.axvline(cfg["selected_amplitude_threshold"], color="red", ls="--")
    ax.set_xlabel("amplitude threshold (deg)")
    ax.set_ylabel("DEV F1")
    ax.set_title("Rule threshold search (DEV)")
    save_jpg(fig, work / "figures" / "rule_dev_threshold_curve.jpg")
    return cfg


def eval_rule(gold: pd.DataFrame, work: Path, cfg: dict, split: str) -> dict:
    part = gold[gold.split == split]
    y, pred, sids, scs = [], [], [], []
    for r in part.itertuples():
        p = work / "features" / "gold" / f"{r.sample_id}.npz"
        if not p.exists():
            continue
        sc = rule_score(load_npz(p)["rotation_xyz"], int(cfg["chosen_rotation_axis"]))
        y.append(int(r.label))
        pred.append(int(sc >= cfg["selected_amplitude_threshold"]))
        sids.append(r.sample_id)
        scs.append(sc)
    if not y:
        raise SystemExit(f"STOP: no {split} gold features to evaluate")
    m = binary_metrics(np.array(y), np.array(pred))
    pd.DataFrame({"sample_id": sids, "label": y, "score": scs, "pred": pred}).to_csv(
        work / "results" / f"rule_{split.lower()}_predictions.csv", index=False
    )
    if split == "TEST":
        dump_json(work / "results" / "rule_test_metrics.json", m)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticklabels(["true 0", "true 1"])
        ax.set_title("Rule TEST confusion")
        save_jpg(fig, work / "figures" / "rule_confusion_matrix.jpg")
    return m


def plot_examples(gold: pd.DataFrame, work: Path) -> None:
    pos = gold[(gold.split == "DEV") & (gold.label == 1)]
    neg = gold[(gold.split == "DEV") & (gold.label == 0)]
    for kind, pool, name in (
        ("positive", pos, "example_positive_rotation.jpg"),
        ("negative", neg, "example_negative_rotation.jpg"),
    ):
        shown = False
        for r in pool.itertuples():
            p = work / "features" / "gold" / f"{r.sample_id}.npz"
            if not p.exists():
                continue
            rot = load_npz(p)["rotation_xyz"]
            t = np.arange(len(rot)) / FPS
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(t, rot[:, 0], label="rot_x")
            ax.plot(t, rot[:, 1], label="rot_y")
            ax.plot(t, rot[:, 2], label="rot_z")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("degrees")
            ax.set_title(f"{kind} {r.sample_id} {r.video_id}")
            ax.legend()
            save_jpg(fig, work / "figures" / name)
            shown = True
            break
        if not shown:
            print("no feature file yet for", name)


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


def build_matrix(paths: list[Path], mode: str, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    xs = []
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
        feat = resample_seq(feat)
        xs.append(feat)
    X = np.stack(xs).astype(np.float32)
    if mean is None:
        mean = X.mean(axis=(0, 1))
        std = X.std(axis=(0, 1)) + 1e-6
    X = (X - mean) / std
    return X, mean, std


def maybe_train_cnn(gold: pd.DataFrame, work: Path, epochs: int, seed: int, smoke: bool) -> dict | None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        print("torch not available; skip classifier:", exc)
        return None
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
        sc = rule_score(load_npz(p)["rotation_xyz"], axis)
        labels.append(int(sc >= thr))
        scores.append(sc)
        keep.append(p)
    y_tr = np.asarray(labels)
    pd.DataFrame({"sample_id": [p.stem for p in keep], "rule_score": scores, "pseudo_label": labels}).to_csv(
        work / "results" / "pseudo_labels.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["pseudo 0", "pseudo 1"], [int((y_tr == 0).sum()), int((y_tr == 1).sum())])
    save_jpg(fig, work / "figures" / "pseudo_label_distribution.jpg")
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

    class CNN(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(d, 32, 5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(32, 64, 5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            h = self.net(x)
            return self.fc(h.squeeze(-1)).squeeze(-1)

    def run_mode(mode: str, do_plots: bool) -> dict:
        Xtr, mean, std = build_matrix(keep, mode)
        Xdv, _, _ = build_matrix(dev_p, mode, mean, std)
        Xte, _, _ = build_matrix(tes_p, mode, mean, std)
        dump_json(work / "models" / "normalization.json", {"mean": mean.tolist(), "std": std.tolist(), "mode": mode})
        d = Xtr.shape[-1]
        model = CNN(d)
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
            save_jpg(fig, work / "figures" / "training_loss.jpg")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot([h["epoch"] for h in hist], [h["dev_f1"] for h in hist])
            ax.set_title("DEV F1 by epoch")
            save_jpg(fig, work / "figures" / "dev_f1_by_epoch.jpg")
            fig, ax = plt.subplots(figsize=(4, 3.5))
            cm = np.array([[test_m["tn"], test_m["fp"]], [test_m["fn"], test_m["tp"]]])
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center")
            ax.set_title("Classifier TEST confusion")
            save_jpg(fig, work / "figures" / "classifier_confusion_matrix.jpg")
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
    save_jpg(fig, work / "figures" / "ablation_f1.jpg")
    return main


def write_final(work: Path, gold_sum: dict, rule_dev: dict, rule_test: dict | None, cnn: dict | None, storage0: dict) -> None:
    rule_test = rule_test or {}
    rows = [
        {
            "method": "Rule baseline",
            "input": "EMOCA head rotation",
            "training_labels": "None",
            "precision": rule_test.get("precision"),
            "recall": rule_test.get("recall"),
            "f1": rule_test.get("f1"),
            "accuracy": rule_test.get("accuracy"),
            "balanced_accuracy": rule_test.get("balanced_accuracy"),
        }
    ]
    if cnn:
        tm = cnn["test_metrics"]
        rows.append(
            {
                "method": "1D CNN",
                "input": "EMOCA temporal features",
                "training_labels": "Pseudo labels",
                "precision": tm["precision"],
                "recall": tm["recall"],
                "f1": tm["f1"],
                "accuracy": tm["accuracy"],
                "balanced_accuracy": tm["balanced_accuracy"],
            }
        )
    pd.DataFrame(rows).to_csv(work / "results" / "model_comparison.csv", index=False)
    if len(rows) >= 1 and rows[0]["f1"] is not None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar([r["method"] for r in rows], [r["f1"] or 0 for r in rows])
        ax.set_ylabel("TEST F1")
        save_jpg(fig, work / "figures" / "model_comparison_f1.jpg")
    n_pseudo = len(list((work / "features" / "pseudo").glob("*.npz")))
    n_gold_f = len(list((work / "features" / "gold").glob("*.npz")))
    summary = {
        "dataset": {**gold_sum, "gold_features": n_gold_f, "train_pseudo_total": n_pseudo},
        "rule_baseline": {"dev_metrics": rule_dev, "test_metrics": rule_test or None},
        "classifier": cnn,
        "storage": {"before": storage0, "after_free_gb": disk_free_gb(work)},
        "limitations": [
            "Small gold set (30 clips).",
            "Pseudo-label noise if the rule is weak.",
            "EMOCA was not trained; only streamed official pickles.",
            "TEST unused for tuning.",
        ],
    }
    dump_json(work / "results" / "final_results_summary.json", summary)
    md = work / "results" / "final_results_summary.md"
    md.write_text(
        "# Final Experiment Results\n\n"
        f"Gold DEV: {gold_sum['dev_total']}\n\nGold TEST: {gold_sum['test_total']}\n\n"
        f"Gold features extracted: {n_gold_f}/30\n\nPseudo TRAIN: {n_pseudo}\n\n"
        f"## Rule baseline\n\nDEV F1: {rule_dev.get('f1')}\n\n"
        f"TEST F1: {rule_test.get('f1')}\n\n"
        f"## Learned model\n\nDEV F1: {None if not cnn else cnn.get('dev_f1')}\n\n"
        f"TEST F1: {None if not cnn else cnn['test_metrics']['f1']}\n\n"
        "Missing values mean EMOCA features were not obtained. They were not invented.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-csv", type=Path, default=ROOT / "data" / "gold_annotations.csv")
    ap.add_argument("--workdir", type=Path, default=ROOT)
    ap.add_argument("--pseudo-clips", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--rule-only", action="store_true", help="Gold rule baseline only; no pseudo-labels or CNN")
    ap.add_argument("--skip-stream", action="store_true", help="Use existing features/gold/*.npz; do not contact Hugging Face")
    ap.add_argument("--local-emoca-dir", type=Path, default=None)
    args = ap.parse_args()
    work = args.workdir.resolve()
    for d in ("results", "figures", "features/gold", "features/pseudo", "models", "logs", "cache"):
        (work / d).mkdir(parents=True, exist_ok=True)
    storage0 = {"free_gb": disk_free_gb(work), "total_gb": shutil.disk_usage(work).total / 1024**3}
    dump_json(work / "results" / "storage_before.json", storage0)
    (work / "logs" / "disk_before.txt").write_text(f"free_gb={storage0['free_gb']}\n")
    stop_if_low_disk(work)
    gold = load_gold(args.gold_csv)
    gold_sum = validate_gold(gold, work / "results", work / "figures")
    n_pseudo = 0 if args.rule_only else (20 if args.smoke_test else args.pseudo_clips)
    epochs = 2 if args.smoke_test else args.epochs
    already = len(list((work / "features" / "gold").glob("*.npz")))
    if args.skip_stream or already >= 30:
        print(f"Skipping EMOCA stream; using {already} existing gold npz files")
        stream_info = {"schema": None, "source": "existing_npz", "gold_npz": already}
    else:
        stream_info = stream_emoca(gold, work, n_pseudo, args.smoke_test, args.local_emoca_dir)
    dump_json(work / "results" / "emoca_stream_status.json", stream_info)
    plot_examples(gold, work)
    n_gold_f = len(list((work / "features" / "gold").glob("*.npz")))
    rule_dev: dict = {}
    rule_test: dict | None = None
    cnn = None
    if n_gold_f == 0:
        print(
            "STOP before metrics: 0 gold EMOCA feature files. "
            "Stream failed or archive not reachable. "
            f"Detail: {stream_info.get('error')}"
        )
    else:
        cfg = choose_axis_and_threshold(gold, work)
        rule_dev = cfg["dev_metrics"]
        print("Rule DEV F1:", rule_dev["f1"])
        rule_test = eval_rule(gold, work, cfg, "TEST")
        print("Rule TEST F1:", rule_test["f1"])
        cnn = None
        if not args.rule_only:
            cnn = maybe_train_cnn(gold, work, epochs, args.seed, args.smoke_test)
    write_final(work, gold_sum, rule_dev, rule_test, cnn, storage0)
    dump_json(work / "results" / "storage_after.json", {"free_gb": disk_free_gb(work)})
    dump_json(
        work / "results" / "experiment_config.json",
        {
            "seed": args.seed,
            "pseudo_clips_requested": n_pseudo,
            "epochs": epochs,
            "smoke_test": args.smoke_test,
            "fps": FPS,
            "gold_csv": str(args.gold_csv),
        },
    )
    print("=====================================")
    print("FINAL EXPERIMENT SUMMARY")
    print("=====================================")
    print("Gold DEV clips:", gold_sum["dev_total"])
    print("Gold TEST clips:", gold_sum["test_total"])
    print("Gold features:", n_gold_f)
    print("Pseudo TRAIN clips:", len(list((work / "features" / "pseudo").glob("*.npz"))))
    print("Rule DEV F1:", rule_dev.get("f1"))
    print("Rule TEST precision:", None if not rule_test else rule_test.get("precision"))
    print("Rule TEST recall:", None if not rule_test else rule_test.get("recall"))
    print("Rule TEST F1:", None if not rule_test else rule_test.get("f1"))
    print("CNN DEV F1:", None if not cnn else cnn.get("dev_f1"))
    print("CNN TEST F1:", None if not cnn else cnn["test_metrics"]["f1"])
    print("Disk free GB:", disk_free_gb(work))
    print("Results:", work / "results")


if __name__ == "__main__":
    main()
