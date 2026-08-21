#!/usr/bin/env python3
"""Data-scaling Step 8 (otter95): grow the pseudo pool 80 → 200 clips.

Does **not** overwrite the committed 80-clip artefacts
(``results/pseudo_labels.csv``, ``results/videomae_finetuned/``). New labels
go to ``results/pseudo_labels_200.csv``; the original 80 rows are copied
verbatim and the script aborts if they would change.

Stages (resume-safe; skip any whose outputs already exist)::

    1  stream EMOCA pose for 120 new videos into features/pseudo/
       (numbering continues: pseudo_00081 … pseudo_00200)
    2  score the 120 new clips with the FROZEN rule
       (results/rule_selected_config.json) → results/pseudo_labels_200.csv
    3  rebuild results/video_shard_index.json (union of all wanted ids)
    4  fetch rgb16 windows for the 120 new sample_ids

Lab invocation (scratch venv; run under nohup — ~2–3 h, mostly network)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    nohup /scratch/db01550/venv/bin/python scripts/scale_pseudo_pool_200.py \\
        > results/scaling_200.log 2>&1 &
    tail -f results/scaling_200.log

``--from-stage N`` restarts from that stage (1–4). Existing npz files are
never re-downloaded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

GOLD_CSV = ROOT / "data" / "gold_annotations.csv"
PSEUDO_DIR = ROOT / "features" / "pseudo"
PSEUDO_80 = ROOT / "results" / "pseudo_labels.csv"
PSEUDO_200 = ROOT / "results" / "pseudo_labels_200.csv"
RULE_CFG = ROOT / "results" / "rule_selected_config.json"
MIN_FREE_GB = 5.4
TARGET_N = 200


def free_gb() -> float:
    return shutil.disk_usage(Path.home()).free / 1024**3


def check_disk(where: str) -> None:
    free = free_gb()
    print(f"  disk: {free:.2f} GB free on ~  ({where})")
    if free < MIN_FREE_GB:
        raise SystemExit(
            f"STOP: free disk on ~ is {free:.2f} GB < {MIN_FREE_GB} GB at {where}."
        )


def npz_video_id(path: Path) -> str:
    with np.load(path, allow_pickle=True) as z:
        vid = z["video_id"]
    if hasattr(vid, "item"):
        vid = vid.item()
    if isinstance(vid, bytes):
        vid = vid.decode()
    return str(vid).strip()


def stage1_stream() -> None:
    print("STEP 1/4: stream EMOCA pose for new pseudo clips")
    check_disk("stage 1 start")
    have = sorted(PSEUDO_DIR.glob("pseudo_*.npz"))
    print(f"  existing pseudo npz: {len(have)}")
    if len(have) >= TARGET_N:
        print(f"  already have {len(have)} >= {TARGET_N}; skip stream")
        return
    from run_full_experiment import load_gold, stream_emoca

    gold = load_gold(GOLD_CSV)
    gold_vids = set(gold["video_id"].astype(str))
    info = stream_emoca(gold, ROOT, TARGET_N, False, None)
    have = sorted(PSEUDO_DIR.glob("pseudo_*.npz"))
    print(f"  after stream: {len(have)} pseudo npz; stream info={info}")
    if len(have) < TARGET_N:
        raise SystemExit(
            f"BLOCKED: only {len(have)}/{TARGET_N} pseudo npz after stream. "
            f"Detail: {info}. Paste this back; do not invent clips."
        )
    leaked = [p.stem for p in have if npz_video_id(p) in gold_vids]
    if leaked:
        raise SystemExit(
            f"FAIL: new/existing pseudo videos collide with gold: {leaked}. "
            "Do not train."
        )


def stage2_labels() -> None:
    print("STEP 2/4: frozen-rule labels → results/pseudo_labels_200.csv")
    check_disk("stage 2 start")
    if not PSEUDO_80.exists():
        raise SystemExit(f"STOP: {PSEUDO_80} missing; 80-clip labels are canonical.")
    if not RULE_CFG.exists():
        raise SystemExit(f"STOP: {RULE_CFG} missing; cannot label new clips.")
    old = pd.read_csv(PSEUDO_80)
    if len(old) != 80:
        raise SystemExit(f"STOP: {PSEUDO_80} has {len(old)} rows, expected 80.")

    from run_full_experiment import load_npz, rule_score

    cfg = json.loads(RULE_CFG.read_text())
    axis = int(cfg["chosen_rotation_axis"])
    thr = float(cfg["selected_amplitude_threshold"])
    min_f = int(cfg.get("min_movement_frames", 5))
    max_f = int(cfg.get("max_movement_frames", 50))

    rows = old.to_dict("records")
    old_ids = set(old["sample_id"])
    new_rows = []
    for path in sorted(PSEUDO_DIR.glob("pseudo_*.npz")):
        sid = path.stem
        if sid in old_ids:
            continue
        rot = load_npz(path)["rotation_xyz"]
        score = float(rule_score(rot, axis, min_f, max_f))
        new_rows.append({
            "sample_id": sid,
            "rule_score": score,
            "pseudo_label": int(score >= thr),
        })
    if not new_rows and PSEUDO_200.exists():
        chk = pd.read_csv(PSEUDO_200)
        merged_old = chk[chk.sample_id.isin(old_ids)].sort_values("sample_id")
        ref = old.sort_values("sample_id")
        if not merged_old["pseudo_label"].to_numpy().tolist() == \
                ref["pseudo_label"].to_numpy().tolist():
            raise SystemExit("FAIL: existing 80 labels would change. Abort.")
        print(f"  {PSEUDO_200} already exists and original 80 are identical; skip")
        return
    combined = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.sort_values("sample_id").reset_index(drop=True)
    # invariant: original 80 labels unchanged
    chk = combined[combined.sample_id.isin(old_ids)].sort_values("sample_id")
    ref = old.sort_values("sample_id")
    if chk["pseudo_label"].to_numpy().tolist() != ref["pseudo_label"].to_numpy().tolist():
        raise SystemExit("FAIL: original 80 pseudo_label values would change. Abort.")
    if chk["sample_id"].to_numpy().tolist() != ref["sample_id"].to_numpy().tolist():
        raise SystemExit("FAIL: original 80 sample_id set would change. Abort.")
    PSEUDO_200.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PSEUDO_200, index=False)
    print(
        f"  wrote {PSEUDO_200}: {len(combined)} rows "
        f"({len(old)} original + {len(new_rows)} new); "
        f"new pos={(pd.DataFrame(new_rows)['pseudo_label'] == 1).sum() if new_rows else 0} "
        f"neg={(pd.DataFrame(new_rows)['pseudo_label'] == 0).sum() if new_rows else 0}"
    )


def stage3_shard_index() -> None:
    print("STEP 3/4: rebuild video_shard_index.json for all wanted ids")
    check_disk("stage 3 start")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_video_shard_index.py")],
        check=True,
    )


def stage4_rgb() -> None:
    print("STEP 4/4: fetch rgb16 windows for new pseudo clips")
    check_disk("stage 4 start")
    old_ids = set(pd.read_csv(PSEUDO_80)["sample_id"])
    new_ids = [
        p.stem for p in sorted(PSEUDO_DIR.glob("pseudo_*.npz"))
        if p.stem not in old_ids
    ]
    already = [
        sid for sid in new_ids
        if (ROOT / "features" / "rgb16" / f"{sid}.npz").exists()
    ]
    todo = [sid for sid in new_ids if sid not in already]
    print(f"  new pseudo ids: {len(new_ids)}; rgb already: {len(already)}; to fetch: {len(todo)}")
    if not todo:
        print("  all new rgb16 windows exist; skip fetch")
        return
    # chunk --ids so the argv stays reasonable
    chunk = 40
    for i in range(0, len(todo), chunk):
        ids = ",".join(todo[i : i + chunk])
        print(f"  fetch chunk {i // chunk + 1}: {todo[i]} … ({min(chunk, len(todo) - i)} ids)")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "fetch_rgb_windows.py"),
                "--ids",
                ids,
            ],
            check=True,
        )
    check_disk("stage 4 end")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-stage", type=int, default=1, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    stages = {
        1: stage1_stream,
        2: stage2_labels,
        3: stage3_shard_index,
        4: stage4_rgb,
    }
    print(f"scale_pseudo_pool_200 starting from stage {args.from_stage}; "
          f"{free_gb():.2f} GB free on ~")
    for n in range(args.from_stage, 5):
        stages[n]()
    n_npz = len(list(PSEUDO_DIR.glob("pseudo_*.npz")))
    n_lab = len(pd.read_csv(PSEUDO_200)) if PSEUDO_200.exists() else 0
    n_rgb = len(list((ROOT / "features" / "rgb16").glob("pseudo_*.npz")))
    print(
        f"\nDONE. pseudo npz={n_npz}  labels_200={n_lab}  rgb16 pseudo={n_rgb}\n"
        "Next (does NOT touch results/videomae_finetuned/):\n"
        "  /scratch/db01550/venv/bin/python scripts/finetune_videomae.py \\\n"
        "      --pseudo-labels results/pseudo_labels_200.csv \\\n"
        "      --out-dir results/videomae_finetuned_n200"
    )


if __name__ == "__main__":
    main()
