#!/usr/bin/env python3
"""Split-leakage gate: fail loudly before any VideoMAE head training.

Asserts, using only committed artefacts (no network, no decoding):

1. gold DEV and gold TEST are disjoint by ``sample_id`` **and** by
   ``video_id`` (no video crosses the DEV/TEST boundary).
2. No TRAIN pseudo ``sample_id`` collides with any gold ``sample_id``.
3. No TRAIN pseudo ``video_id`` appears among gold TEST videos (pseudo =
   TRAIN split) — nor among gold DEV videos.
4. Every selected TRAIN/DEV/TEST clip actually has its expected artefacts
   where those artefacts already exist (warn-only: coverage is Step 3/4's
   job, this gate is about leakage).

Default (no flags) is the **nod** gate: gold from
``data/gold_annotations.csv``, TRAIN = every ``features/pseudo/*.npz``.
Head-shake VideoMAE must pass the same asserts on shake gold +
``results/shake/pseudo_labels.csv``::

    python scripts/check_split_leakage.py \\
        --gold-csv data/gold/shake_annotation_sheet.csv \\
        --pseudo-labels results/shake/pseudo_labels.csv

Exit 0 prints PASS with the counts; any violation exits non-zero listing
every offender. Run this before ``train_videomae_head.py`` /
``finetune_videomae.py`` — if it fails, training must not start.

Lab invocation (nod defaults)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    source ../.venv/bin/activate
    python scripts/check_split_leakage.py
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLD_CSV = ROOT / "data" / "gold_annotations.csv"
SHAKE_GOLD_CSV = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
PSEUDO_DIR = ROOT / "features" / "pseudo"
PSEUDO_LABELS = ROOT / "results" / "pseudo_labels.csv"
SHAKE_PSEUDO_LABELS = ROOT / "results" / "shake" / "pseudo_labels.csv"
SHAKE_RESULTS = ROOT / "results" / "shake"
RGB16_DIR = ROOT / "features" / "rgb16"
EMB_DIR = ROOT / "data" / "features" / "videomae"
NOD_HEAD_PT = ROOT / "models" / "videomae_head.pt"
NOD_VMAE_OUT_DIRS = (
    ROOT / "results" / "videomae_finetuned",
    ROOT / "results" / "videomae_finetuned_n200",
    ROOT / "results" / "videomae_finetuned_n120",
    ROOT / "results" / "videomae_frozen_head",
)
JOINT_ROOT = ROOT / "results" / "joint"
LOCKED_OUT_DIRS = (
    ROOT / "results" / "videomae_finetuned",
    ROOT / "results" / "videomae_frozen_head",
    ROOT / "results" / "shake" / "videomae_finetuned",
    ROOT / "results" / "shake" / "videomae_frozen_head",
    ROOT / "results" / "shake" / "cnn",
    ROOT / "results" / "shake" / "videomae_finetuned_balanced",
    ROOT / "results" / "shake" / "videomae_frozen_head_balanced",
    ROOT / "results" / "shake" / "videomae_finetuned_dev_threshold",
    ROOT / "results" / "shake" / "fusion_pose_rgb",
    ROOT / "results" / "shake" / "majority_baseline",
    ROOT / "results" / "joint" / "videomae_frozen_head",
    ROOT / "results" / "joint" / "videomae_finetuned",
)
LOCKED_SHAKE_RULE_FILES = (
    ROOT / "results" / "shake" / "rule_test_metrics.json",
    ROOT / "results" / "shake" / "rule_test_predictions.csv",
    ROOT / "results" / "shake" / "rule_selected_config.json",
)


def resolve_repo_path(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def path_under(path: Path | str, parent: Path | str) -> bool:
    try:
        resolve_repo_path(path).relative_to(resolve_repo_path(parent))
        return True
    except ValueError:
        return False


def assert_videomae_task_isolation(
    *,
    gold_csv: Path | str,
    label_col: str,
    pseudo_labels: Path | str,
    out_dir: Path | str,
    model_pt: Path | str | None = None,
) -> str:
    """Return ``head_nod`` or ``head_shake``. Abort if the two tasks mix.

    Shake runs must use ``shake_label``, the shake annotation sheet, shake
    pseudo-labels, and an out-dir under ``results/shake/``. They must never
    write nod VideoMAE artefacts (``results/videomae_finetuned/``,
    ``results/videomae_frozen_head/``, ``models/videomae_head.pt``) or nod
    gold ``data/gold_annotations.csv``.
    """
    gold_csv = resolve_repo_path(gold_csv)
    pseudo_labels = resolve_repo_path(pseudo_labels)
    out_dir = resolve_repo_path(out_dir)
    model_pt_res = (
        resolve_repo_path(model_pt) if model_pt is not None else None
    )
    label_col = str(label_col).strip()

    shake_hits: list[str] = []
    if label_col == "shake_label":
        shake_hits.append("--label-col shake_label")
    if gold_csv == resolve_repo_path(SHAKE_GOLD_CSV):
        shake_hits.append("shake gold CSV")
    if path_under(pseudo_labels, SHAKE_RESULTS):
        shake_hits.append("shake pseudo-labels")
    if path_under(out_dir, SHAKE_RESULTS):
        shake_hits.append("out-dir under results/shake/")

    if shake_hits:
        errors: list[str] = []
        if label_col != "shake_label":
            errors.append(
                f"--label-col {label_col!r} but shake paths were given; "
                "DEV/TEST must use shake_label, not nod label"
            )
        if gold_csv != resolve_repo_path(SHAKE_GOLD_CSV):
            errors.append(
                f"--gold-csv {gold_csv} is not "
                "data/gold/shake_annotation_sheet.csv "
                "(do not train shake against nod gold_annotations.csv label)"
            )
        if not path_under(pseudo_labels, SHAKE_RESULTS):
            errors.append(
                f"--pseudo-labels {pseudo_labels} is not under results/shake/ "
                "(refusing nod results/pseudo_labels.csv)"
            )
        if not path_under(out_dir, SHAKE_RESULTS):
            errors.append(
                f"--out-dir {out_dir} is not under results/shake/ "
                "(refusing to overwrite results/videomae_finetuned/ "
                "or results/videomae_frozen_head/)"
            )
        for blocked in NOD_VMAE_OUT_DIRS:
            if path_under(out_dir, blocked):
                errors.append(
                    "refusing nod out-dir "
                    f"{blocked.relative_to(ROOT).as_posix()}"
                )
        if model_pt_res is not None:
            if model_pt_res == resolve_repo_path(NOD_HEAD_PT):
                errors.append("refusing to overwrite models/videomae_head.pt")
            for blocked in NOD_VMAE_OUT_DIRS:
                if path_under(model_pt_res, blocked):
                    errors.append(
                        "refusing nod checkpoint under "
                        f"{blocked.relative_to(ROOT).as_posix()}"
                    )
                    break
        if errors:
            raise SystemExit(
                "STOP: mixed or unsafe head-shake VideoMAE paths "
                f"({'; '.join(shake_hits)}):\n- "
                + "\n- ".join(errors)
            )
        return "head_shake"

    if path_under(out_dir, SHAKE_RESULTS):
        raise SystemExit(
            "STOP: nod VideoMAE run refuses to write under results/shake/"
        )
    if model_pt_res is not None and path_under(model_pt_res, SHAKE_RESULTS):
        raise SystemExit(
            "STOP: nod VideoMAE run refuses to write a checkpoint under "
            "results/shake/"
        )
    return "head_nod"


def assert_unlocked_out_dir(out_dir: Path | str) -> Path:
    """Abort if ``out_dir`` is a locked TEST artefact directory."""
    out_dir = resolve_repo_path(out_dir)
    for blocked in LOCKED_OUT_DIRS:
        blocked_r = resolve_repo_path(blocked)
        if out_dir == blocked_r or path_under(out_dir, blocked_r):
            raise SystemExit(
                f"STOP: refusing to write locked TEST dir "
                f"{blocked.relative_to(ROOT).as_posix()}. Use a new out-dir "
                "(never --force on locked shake/nod VideoMAE, CNN, or rule)."
            )
    for locked in LOCKED_SHAKE_RULE_FILES:
        if out_dir == resolve_repo_path(locked):
            raise SystemExit(f"STOP: refusing to overwrite locked {locked}")
    return out_dir


def assert_joint_videomae_paths(
    *,
    gold_csv: Path | str,
    nod_pseudo: Path | str,
    shake_pseudo: Path | str,
    out_dir: Path | str,
    model_pt: Path | str | None = None,
) -> str:
    """Joint nod+shake VideoMAE must write only under ``results/joint/``."""
    gold_csv = resolve_repo_path(gold_csv)
    nod_pseudo = resolve_repo_path(nod_pseudo)
    shake_pseudo = resolve_repo_path(shake_pseudo)
    out_dir = assert_unlocked_out_dir(out_dir)
    model_pt_res = (
        resolve_repo_path(model_pt) if model_pt is not None else None
    )
    errors: list[str] = []
    if not path_under(out_dir, JOINT_ROOT):
        errors.append(
            f"--out-dir {out_dir} is not under results/joint/ "
            "(refusing nod and shake locked VideoMAE dirs)"
        )
    if gold_csv != resolve_repo_path(SHAKE_GOLD_CSV):
        errors.append(
            f"--gold-csv {gold_csv} is not "
            "data/gold/shake_annotation_sheet.csv "
            "(joint DEV/TEST needs nod_label and shake_label on the same 30 videos)"
        )
    if path_under(nod_pseudo, SHAKE_RESULTS):
        errors.append(
            f"nod pseudo {nod_pseudo} is under results/shake/ — "
            "expected results/pseudo_labels.csv"
        )
    if not path_under(shake_pseudo, SHAKE_RESULTS):
        errors.append(
            f"shake pseudo {shake_pseudo} is not under results/shake/"
        )
    if model_pt_res is not None:
        if model_pt_res == resolve_repo_path(NOD_HEAD_PT):
            errors.append("refusing to overwrite models/videomae_head.pt")
        if not path_under(model_pt_res, JOINT_ROOT):
            errors.append(
                f"joint checkpoint {model_pt_res} must live under results/joint/"
            )
    if errors:
        raise SystemExit(
            "STOP: joint VideoMAE path check failed:\n- " + "\n- ".join(errors)
        )
    return "joint_nod_shake"


def npz_video_id(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        arr = np.load(io.BytesIO(z.read("video_id.npy")), allow_pickle=True)
    if hasattr(arr, "item"):
        arr = arr.item()
    if isinstance(arr, bytes):
        arr = arr.decode()
    return str(arr).strip()


def _pseudo_train_ids(
    pseudo_labels: Path | None,
    labelled_train_only: bool,
    failures: list[str],
) -> tuple[set[str], dict[str, str]]:
    """TRAIN sample_ids and sample_id → video_id for the leakage asserts."""
    if labelled_train_only:
        if pseudo_labels is None:
            raise SystemExit(
                "STOP: labelled_train_only requires a --pseudo-labels path."
            )
        pl = resolve_repo_path(pseudo_labels)
        if not pl.exists():
            raise SystemExit(f"STOP: {pl} is missing; cannot certify no leak.")
        with pl.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise SystemExit(f"STOP: {pl} has no rows.")
        if "sample_id" not in rows[0]:
            raise SystemExit(f"STOP: {pl} has no sample_id column.")
        ids = {r["sample_id"] for r in rows}
        vids: dict[str, str] = {}
        for r in rows:
            sid = r["sample_id"]
            vid = str(r.get("video_id") or "").strip()
            if vid:
                vids[sid] = vid
                continue
            npz = PSEUDO_DIR / f"{sid}.npz"
            if not npz.exists():
                failures.append(
                    f"pseudo {sid} has no video_id in {pl.name} and no "
                    f"{npz.name}; cannot certify no leak"
                )
                continue
            vids[sid] = npz_video_id(npz)
        return ids, vids

    pseudo_npz = sorted(PSEUDO_DIR.glob("*.npz"))
    ids = {p.stem for p in pseudo_npz}
    vids = {p.stem: npz_video_id(p) for p in pseudo_npz}
    if pseudo_labels is not None and resolve_repo_path(pseudo_labels).exists():
        with resolve_repo_path(pseudo_labels).open(newline="") as fh:
            labelled = {r["sample_id"] for r in csv.DictReader(fh)}
        if labelled != ids:
            print(
                "WARNING: pseudo_labels.csv ids and features/pseudo/*.npz "
                f"differ ({len(labelled)} vs {len(ids)}): "
                f"labels-only={sorted(labelled - ids)[:5]}, "
                f"npz-only={sorted(ids - labelled)[:5]}"
            )
    elif PSEUDO_LABELS.exists():
        with PSEUDO_LABELS.open(newline="") as fh:
            labelled = {r["sample_id"] for r in csv.DictReader(fh)}
        if labelled != ids:
            print(
                "WARNING: pseudo_labels.csv ids and features/pseudo/*.npz "
                f"differ ({len(labelled)} vs {len(ids)}): "
                f"labels-only={sorted(labelled - ids)[:5]}, "
                f"npz-only={sorted(ids - labelled)[:5]}"
            )
    return ids, vids


def run(
    gold_csv: Path | str | None = None,
    pseudo_labels: Path | str | None = None,
    labelled_train_only: bool = False,
) -> None:
    """Run the leakage asserts. ``SystemExit`` on any FAIL.

    Training scripts must call this (not ``main()``) so parent argparse
    flags such as ``--unfreeze-blocks`` are not re-parsed here.
    """
    gold_path = resolve_repo_path(gold_csv or GOLD_CSV)
    pl_path = (
        resolve_repo_path(pseudo_labels) if pseudo_labels is not None else None
    )
    if not gold_path.exists():
        raise SystemExit(f"STOP: gold CSV missing: {gold_path}")

    with gold_path.open(newline="") as fh:
        gold = list(csv.DictReader(fh))
    needed = {"sample_id", "video_id", "split"}
    if not gold or needed - set(gold[0].keys()):
        raise SystemExit(
            f"STOP: {gold_path} must have columns {sorted(needed)}."
        )

    dev = [r for r in gold if r["split"].strip().upper() == "DEV"]
    tes = [r for r in gold if r["split"].strip().upper() == "TEST"]
    if not dev or not tes:
        raise SystemExit("STOP: gold CSV must contain both DEV and TEST rows.")

    dev_ids = {r["sample_id"] for r in dev}
    tes_ids = {r["sample_id"] for r in tes}
    dev_vids = {r["video_id"] for r in dev}
    tes_vids = {r["video_id"] for r in tes}

    failures: list[str] = []
    pseudo_ids, pseudo_vids = _pseudo_train_ids(
        pl_path, labelled_train_only, failures
    )

    both = sorted(dev_ids & tes_ids)
    if both:
        failures.append(f"sample_id in BOTH DEV and TEST: {both}")
    vids_both = sorted(dev_vids & tes_vids)
    if vids_both:
        failures.append(f"video_id crosses DEV/TEST: {vids_both}")

    id_clash = sorted(pseudo_ids & (dev_ids | tes_ids))
    if id_clash:
        failures.append(f"pseudo sample_id collides with gold: {id_clash}")
    in_test = sorted(v for v in pseudo_vids.values() if v in tes_vids)
    if in_test:
        failures.append(
            f"pseudo video_id present in gold TEST (train->test leak): "
            f"{in_test}"
        )
    in_dev = sorted(v for v in pseudo_vids.values() if v in dev_vids)
    if in_dev:
        failures.append(f"pseudo video_id present in gold DEV: {in_dev}")

    for name, folder in (("rgb16", RGB16_DIR), ("embeddings", EMB_DIR)):
        if folder.exists():
            have = {p.stem for p in folder.glob("*.npz")}
            missing_gold = sorted((dev_ids | tes_ids) - have)
            missing_pseudo = sorted(pseudo_ids - have)
            if missing_gold or missing_pseudo:
                print(
                    f"NOTE: {name}: missing for {len(missing_gold)} gold / "
                    f"{len(missing_pseudo)} pseudo clips (coverage is the "
                    "fetch/extract steps' job)"
                )

    print("---- split-leakage gate ----")
    try:
        gold_disp = gold_path.relative_to(ROOT)
    except ValueError:
        gold_disp = gold_path
    print(
        f"gold: {gold_disp}  "
        f"{len(dev)} DEV / {len(tes)} TEST; "
        f"{len(dev_vids)} / {len(tes_vids)} unique videos"
    )
    src = (
        "labelled TRAIN"
        if labelled_train_only
        else "features/pseudo/*.npz"
    )
    print(
        f"pseudo ({src}): {len(pseudo_ids)} clips, "
        f"{len(set(pseudo_vids.values()))} unique videos"
    )
    if failures:
        raise SystemExit(
            "FAIL: split leakage detected — DO NOT TRAIN:\n- "
            + "\n- ".join(failures)
        )
    print(
        "PASS: DEV/TEST disjoint (ids and videos); no pseudo clip or video "
        "in TEST (or DEV); no id collisions."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gold-csv",
        type=Path,
        default=GOLD_CSV,
        help="gold CSV with sample_id, video_id, split "
        "(default: data/gold_annotations.csv)",
    )
    parser.add_argument(
        "--pseudo-labels",
        type=Path,
        default=None,
        help="if set, TRAIN = these labelled ids (and their video_id) "
        "instead of every features/pseudo/*.npz. Required for head-shake.",
    )
    args = parser.parse_args(argv)
    labelled = args.pseudo_labels is not None
    run(
        gold_csv=args.gold_csv,
        pseudo_labels=args.pseudo_labels,
        labelled_train_only=labelled,
    )


if __name__ == "__main__":
    main()
