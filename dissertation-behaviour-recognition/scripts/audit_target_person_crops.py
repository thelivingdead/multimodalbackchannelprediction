#!/usr/bin/env python3
"""Visual and automatic identity audit for DEV target-person crops.

Writes contact sheets and crop_audit.csv. Writes audit_pass.json only if
every resolved crop sits on the annotated half. TEST is not read.

    /scratch/db01550/venv/bin/python scripts/audit_target_person_crops.py \\
        --rgb-dir /scratch/db01550/rgb16_windowed_identity_dev
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

from src.crop_target_person import on_wrong_half  # noqa: E402

FIXED = ROOT / "results" / "windowed_dev" / "videomae_identity_fixed"
WINDOWS_DEV = ROOT / "data" / "windowed_annotations" / "nod_windows_dev.csv"
DEV_IDS = {f"gold_{i:03d}" for i in range(1, 16)}
TEST_IDS = {f"gold_{i:03d}" for i in range(16, 31)}
MAX_UNRESOLVED_FRAC = 0.25


def pick_audit_windows(labels: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """About 2 windows per DEV clip, mixed positive and negative, 30 to 50 total."""
    resolved = manifest[manifest["crop_status"] == "resolved"].copy()
    if resolved.empty:
        raise SystemExit("STOP: no resolved crops to audit")
    chosen: list[pd.DataFrame] = []
    for sid, group in resolved.groupby("sample_id"):
        joined = group.merge(
            labels[["window_id", "label"]], on="window_id", how="left"
        )
        pos = joined[joined["label"] == 1]
        neg = joined[joined["label"] == 0]
        take = []
        if not pos.empty:
            take.append(pos.iloc[len(pos) // 2])
        if not neg.empty:
            take.append(neg.iloc[len(neg) // 2])
        if not take:
            take.append(joined.iloc[0])
        chosen.append(pd.DataFrame(take))
    picked = pd.concat(chosen, ignore_index=True)
    if len(picked) < 30:
        extra = resolved[~resolved["window_id"].isin(picked["window_id"])]
        extra = extra.merge(
            labels[["window_id", "label"]], on="window_id", how="left"
        )
        need = min(50 - len(picked), len(extra))
        if need:
            step = max(len(extra) // need, 1)
            picked = pd.concat(
                [picked, extra.iloc[::step].head(need)], ignore_index=True
            )
    return picked.drop_duplicates("window_id").head(50)


def contact_sheet(rgb_dir: Path, rows: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in rows.itertuples(index=False):
        path = rgb_dir / f"{rec.window_id}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as payload:
            preview = payload["preview"]
            crop = payload["rgb"][len(payload["rgb"]) // 2]
            side = str(payload["watch_side"])
            person = str(payload["person"])
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
        axes[0].imshow(preview)
        axes[0].set_title(f"source mid frame, watch {side}")
        axes[1].imshow(crop)
        axes[1].set_title("selected crop")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"{rec.window_id}  {person}  {side}  "
            f"frames {int(rec.start_frame_relative)}-{int(rec.end_frame_relative)}  "
            f"label={int(rec.label) if hasattr(rec, 'label') and pd.notna(rec.label) else '?'}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{rec.window_id}.png", dpi=120)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=FIXED)
    args = parser.parse_args()
    rgb_dir = args.rgb_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(WINDOWS_DEV)
    labels["sample_id"] = labels["sample_id"].astype(str)
    if set(labels["sample_id"]) & TEST_IDS:
        raise SystemExit("STOP: TEST id in DEV window file")
    manifest_path = out_dir / "fetch_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"STOP: missing {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    manifest["window_id"] = manifest["window_id"].astype(str)
    if "split" in manifest.columns and (manifest["split"].astype(str) != "DEV").any():
        raise SystemExit("STOP: fetch manifest contains a non-DEV row")

    n_windows = 15 * 29
    n_resolved = int((manifest["crop_status"] == "resolved").sum())
    n_unresolved = int((manifest["crop_status"] == "unresolved").sum())
    n_failed = int((manifest["crop_status"] == "failed").sum())
    resolved = manifest[manifest["crop_status"] == "resolved"].copy()
    n_wrong = 0
    if not resolved.empty:
        n_wrong = int(
            [
                on_wrong_half(float(r.crop_centre_x), int(r.frame_width), str(r.watch_side))
                for r in resolved.itertuples(index=False)
            ].count(True)
        )
    n_no_face = int(
        manifest["reason"].astype(str).str.contains("only 0 target", na=False).sum()
    ) if "reason" in manifest.columns else 0

    picked = pick_audit_windows(labels, manifest) if n_resolved else pd.DataFrame()
    if not picked.empty:
        contact_sheet(rgb_dir, picked, out_dir / "crop_audit")
        audit_rows = picked.copy()
        audit_rows["target_person"] = audit_rows.get("person", "")
        audit_rows["selected_side"] = audit_rows.get("watch_side", "")
        audit_rows["face_detected"] = (
            audit_rows.get("n_target_detections", pd.Series(dtype=int)).fillna(0) >= 2
        )
        audit_rows["notes"] = audit_rows.get("reason", "")
        keep = [
            c
            for c in (
                "sample_id",
                "window_id",
                "start_frame_relative",
                "end_frame_relative",
                "label",
                "target_person",
                "selected_side",
                "crop_status",
                "face_detected",
                "notes",
            )
            if c in audit_rows.columns
        ]
        audit_rows[keep].to_csv(out_dir / "crop_audit.csv", index=False)

    unresolved_frac = n_unresolved / n_windows if n_windows else 1.0
    automatic_pass = (
        n_resolved > 0
        and n_wrong == 0
        and unresolved_frac <= MAX_UNRESOLVED_FRAC
        and len(picked) >= 30
    )
    report = {
        "n_dev_windows": n_windows,
        "n_successfully_cropped": n_resolved,
        "n_unresolved": n_unresolved,
        "n_failed": n_failed,
        "percent_successfully_cropped": round(100.0 * n_resolved / n_windows, 1),
        "n_no_face_on_target_half": n_no_face,
        "n_resolved_on_wrong_half": n_wrong,
        "n_visually_audited": int(len(picked)),
        "automatic_half_check": "pass" if n_wrong == 0 and n_resolved else "fail",
        "audit_pass": automatic_pass,
        "test_touched": False,
        "stop_training": not automatic_pass,
    }
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if automatic_pass:
        (out_dir / "audit_pass.json").write_text(
            json.dumps(
                {
                    "pass": True,
                    "note": (
                        "Automatic check only: every resolved crop centre is on the "
                        "annotated half, unresolved fraction is at most 25 percent, "
                        "and at least 30 contact sheets were written. Look at "
                        "crop_audit/ before trusting VideoMAE. Delete this file if "
                        "any sheet shows the excluded person."
                    ),
                    **report,
                },
                indent=2,
            )
            + "\n"
        )
    elif (out_dir / "audit_pass.json").exists():
        (out_dir / "audit_pass.json").unlink()

    print(json.dumps(report, indent=2))
    if not automatic_pass:
        raise SystemExit(
            "STOP: identity audit did not pass. Do not train VideoMAE. "
            "See audit_report.json"
        )
    print(f"contact sheets: {out_dir / 'crop_audit'}")
    print("Look at the sheets. If any show the excluded person, delete audit_pass.json.")


if __name__ == "__main__":
    main()
