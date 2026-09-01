#!/usr/bin/env python3
"""Temporal Correspondence of Rule Detections and Annotated Nod Onsets.

Development data only. Reuses the frozen nod rule. Does not load or score
GOLD TEST. Does not retune the amplitude rule. Does not regenerate
pseudo-labels. Does not write locked result files.

    python scripts/evaluate_temporal_correspondence_dev.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_nod_onsets import (  # noqa: E402
    AUDIT_CSV,
    EXPERIMENT_NAME,
    EXPECTED_DEV,
    OUT_DIR,
    TEST_MSG,
    audit_dev_onsets,
    print_audit_counts,
    refuse_forbidden_argv,
    refuse_gold_test,
    write_onset_audit,
)

refuse_forbidden_argv()

from run_full_experiment import load_npz, rule_score  # noqa: E402
from src.audio_io import FPS  # noqa: E402
from src.utils import dump_json  # noqa: E402

RULE_CONFIG = ROOT / "results" / "rule_selected_config.json"
CANDIDATES_CSV = OUT_DIR / "rule_candidates_dev.csv"
MATCHES_CSV = OUT_DIR / "onset_candidate_matches.csv"
CONTROL_CSV = OUT_DIR / "timing_control.csv"
SUMMARY_JSON = OUT_DIR / "temporal_correspondence_summary.json"
GOLD_NPZ_DIR = ROOT / "features" / "gold"

TOLERANCES_S = (0.5, 1.0, 2.0)
CONTROL_EXCLUSION_S = 2.0
N_RANDOM = 1000
SEED = 42
OUTPUT_TEST_ID_RE = re.compile(r"gold_0*(1[6-9]|2[0-9]|30)\b")

CANDIDATE_COLUMNS = [
    "sample_id",
    "candidate_start_s",
    "candidate_end_s",
    "candidate_amplitude",
    "rule_prediction",
]
MATCH_COLUMNS = [
    "sample_id",
    "gold_onset_s",
    "candidate_start_s",
    "candidate_end_s",
    "signed_offset_s",
    "absolute_offset_s",
    "gold_onset_inside_candidate_segment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    return parser.parse_args()


def load_frozen_rule_config() -> dict:
    if not RULE_CONFIG.is_file():
        raise SystemExit(f"STOP: missing frozen rule config {RULE_CONFIG}")
    cfg = json.loads(RULE_CONFIG.read_text())
    axis = int(cfg["chosen_rotation_axis"])
    if axis != 0 or str(cfg.get("axis_name", "x")).lower() != "x":
        raise SystemExit(
            "STOP: frozen nod rule must be rotation axis x; "
            f"found axis={axis} name={cfg.get('axis_name')!r}"
        )
    return {
        "chosen_rotation_axis": axis,
        "axis_name": "x",
        "min_movement_frames": int(cfg["min_movement_frames"]),
        "max_movement_frames": int(cfg["max_movement_frames"]),
        "selected_amplitude_threshold": float(cfg["selected_amplitude_threshold"]),
        "smoothing": str(cfg.get("smoothing", "savgol_11_2")),
    }


def load_dev_pose(sample_id: str) -> np.ndarray:
    refuse_gold_test(sample_id=sample_id)
    path = GOLD_NPZ_DIR / f"{sample_id}.npz"
    if not path.is_file():
        raise SystemExit(f"STOP: missing DEV pose file {path}")
    rot = np.asarray(load_npz(path)["rotation_xyz"], dtype=float)
    if rot.ndim != 2 or rot.shape[1] < 1:
        raise SystemExit(f"STOP: bad rotation_xyz in {path}")
    return rot


def half_cycle_segments(
    rot: np.ndarray,
    axis: int,
    min_frames: int,
    max_frames: int,
) -> list[tuple[int, int, float]]:
    """Same turning-point loop as run_full_experiment.rule_score, with frame indices."""
    x_full = np.asarray(rot[:, axis], dtype=float)
    finite = np.isfinite(x_full)
    orig_idx = np.flatnonzero(finite)
    x = x_full[finite]
    if x.size < 11:
        return []
    win = min(11, x.size if x.size % 2 == 1 else x.size - 1)
    win = max(5, win)
    if win % 2 == 0:
        win -= 1
    sm = savgol_filter(x, win, 2)
    d = np.diff(sm)
    turns = np.where(np.diff(np.sign(d)) != 0)[0] + 1
    segs: list[tuple[int, int, float]] = []
    for i, a in enumerate(turns):
        for b in turns[i + 1 :]:
            span = int(b - a)
            if span < min_frames or span > max_frames:
                continue
            amp = float(abs(sm[int(b)] - sm[int(a)]))
            segs.append((int(orig_idx[int(a)]), int(orig_idx[int(b)]), amp))
    return segs


def union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted((float(a), float(b)) for a, b in intervals)
    merged = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return float(sum(end - start for start, end in merged))


def assert_inside_window(t: float, window_start_s: float, window_end_s: float, what: str) -> None:
    if not (window_start_s <= t <= window_end_s):
        raise SystemExit(
            f"STOP: {what} {t:.6f} lies outside analysed interval "
            f"[{window_start_s:.6f}, {window_end_s:.6f}]"
        )


def match_onset_to_candidates(
    gold_onset_s: float,
    candidates: list[dict[str, float]],
) -> dict[str, float | str] | None:
    """Closest candidate by start time. Used for the annotated path and the control path."""
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda c: (
            abs(float(c["candidate_start_s"]) - gold_onset_s),
            float(c["candidate_start_s"]),
        ),
    )
    start = float(best["candidate_start_s"])
    end = float(best["candidate_end_s"])
    signed = start - gold_onset_s
    inside = "yes" if start <= gold_onset_s <= end else "no"
    return {
        "gold_onset_s": gold_onset_s,
        "candidate_start_s": start,
        "candidate_end_s": end,
        "signed_offset_s": signed,
        "absolute_offset_s": abs(signed),
        "gold_onset_inside_candidate_segment": inside,
    }


def sample_pseudo_onset(
    rng: np.random.Generator,
    window_start_s: float,
    window_end_s: float,
    true_onset_s: float,
    exclusion_s: float,
) -> float:
    lo = max(window_start_s, true_onset_s - exclusion_s)
    hi = min(window_end_s, true_onset_s + exclusion_s)
    parts: list[tuple[float, float]] = []
    if lo > window_start_s:
        parts.append((window_start_s, lo))
    if hi < window_end_s:
        parts.append((hi, window_end_s))
    lengths = [b - a for a, b in parts]
    total = float(sum(lengths))
    if total <= 0.0:
        raise SystemExit("STOP: no valid interval for random timing control")
    u = float(rng.random()) * total
    chosen = parts[-1][1]
    for start, end in parts:
        length = end - start
        if u <= length:
            chosen = start + u
            break
        u -= length
    if abs(chosen - true_onset_s) < exclusion_s:
        raise SystemExit("STOP: random timing control drew inside the excluded onset region")
    assert_inside_window(chosen, window_start_s, window_end_s, "control onset")
    return chosen


def extract_dev_candidates(
    audit_rows: list[dict[str, object]],
    cfg: dict,
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, float]]], dict[str, dict[str, float]]]:
    axis = int(cfg["chosen_rotation_axis"])
    min_frames = int(cfg["min_movement_frames"])
    max_frames = int(cfg["max_movement_frames"])
    threshold = float(cfg["selected_amplitude_threshold"])
    rows: list[dict[str, object]] = []
    by_clip: dict[str, list[dict[str, float]]] = {}
    clip_stats: dict[str, dict[str, float]] = {}
    for rec in audit_rows:
        sid = str(rec["sample_id"])
        refuse_gold_test(sample_id=sid, split=str(rec["split"]))
        window_start_s = float(rec["window_start_s"])
        window_end_s = float(rec["window_end_s"])
        rot = load_dev_pose(sid)
        segs = half_cycle_segments(rot, axis, min_frames, max_frames)
        score = float(rule_score(rot, axis, min_frames=min_frames, max_frames=max_frames))
        if segs:
            best_amp = max(amp for _, _, amp in segs)
            if abs(best_amp - score) > 1e-6:
                raise SystemExit(
                    f"STOP: candidate amplitudes diverge from frozen rule_score for {sid}: "
                    f"{best_amp} vs {score}"
                )
        pred = int(score >= threshold)
        detections = [
            {
                "candidate_start_s": window_start_s + float(a) / FPS,
                "candidate_end_s": window_start_s + float(b) / FPS,
                "candidate_amplitude": amp,
            }
            for a, b, amp in segs
            if amp >= threshold
        ]
        kept: list[dict[str, float]] = []
        for det in detections:
            start = float(det["candidate_start_s"])
            end = float(det["candidate_end_s"])
            assert_inside_window(start, window_start_s, window_end_s, f"{sid} candidate start")
            assert_inside_window(end, window_start_s, window_end_s, f"{sid} candidate end")
            if end < start:
                raise SystemExit(f"STOP: {sid} candidate end before start")
            kept.append(det)
            rows.append(
                {
                    "sample_id": sid,
                    "candidate_start_s": start,
                    "candidate_end_s": end,
                    "candidate_amplitude": float(det["candidate_amplitude"]),
                    "rule_prediction": pred,
                }
            )
        by_clip[sid] = kept
        window_dur = window_end_s - window_start_s
        if window_dur <= 0:
            raise SystemExit(f"STOP: non-positive analysed window for {sid}")
        coverage = union_duration([(d["candidate_start_s"], d["candidate_end_s"]) for d in kept]) / window_dur
        clip_stats[sid] = {
            "n_candidates": float(len(kept)),
            "coverage_fraction": float(coverage),
            "rule_prediction": float(pred),
        }
    return rows, by_clip, clip_stats


def write_candidates(rows: list[dict[str, object]]) -> None:
    with CANDIDATES_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        for row in rows:
            refuse_gold_test(sample_id=str(row["sample_id"]))
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "candidate_start_s": f"{float(row['candidate_start_s']):.6f}",
                    "candidate_end_s": f"{float(row['candidate_end_s']):.6f}",
                    "candidate_amplitude": f"{float(row['candidate_amplitude']):.6f}",
                    "rule_prediction": int(row["rule_prediction"]),
                }
            )


def match_usable_onsets(
    audit_rows: list[dict[str, object]],
    by_clip: dict[str, list[dict[str, float]]],
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    matches: list[dict[str, object]] = []
    usable_ids: list[str] = []
    excluded_ids: list[str] = []
    for rec in audit_rows:
        sid = str(rec["sample_id"])
        refuse_gold_test(sample_id=sid, split=str(rec["split"]))
        if rec["usable"] != "yes":
            if int(rec["nod_label"]) == 1:
                excluded_ids.append(sid)
            continue
        onset = float(rec["onset_time_s"])
        window_start_s = float(rec["window_start_s"])
        window_end_s = float(rec["window_end_s"])
        assert_inside_window(onset, window_start_s, window_end_s, f"{sid} gold onset")
        usable_ids.append(sid)
        hit = match_onset_to_candidates(onset, by_clip.get(sid, []))
        row: dict[str, object] = {
            "sample_id": sid,
            "gold_onset_s": onset,
            "window_start_s": window_start_s,
            "window_end_s": window_end_s,
        }
        if hit is None:
            row.update(
                {
                    "candidate_start_s": "",
                    "candidate_end_s": "",
                    "signed_offset_s": "",
                    "absolute_offset_s": "",
                    "gold_onset_inside_candidate_segment": "no",
                    "matched": False,
                }
            )
        else:
            row.update(hit)
            row["matched"] = True
        matches.append(row)
    return matches, usable_ids, excluded_ids


def write_matches(rows: list[dict[str, object]]) -> None:
    with MATCHES_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_COLUMNS)
        writer.writeheader()
        for row in rows:
            refuse_gold_test(sample_id=str(row["sample_id"]))
            if row["matched"]:
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "gold_onset_s": f"{float(row['gold_onset_s']):.6f}",
                        "candidate_start_s": f"{float(row['candidate_start_s']):.6f}",
                        "candidate_end_s": f"{float(row['candidate_end_s']):.6f}",
                        "signed_offset_s": f"{float(row['signed_offset_s']):.6f}",
                        "absolute_offset_s": f"{float(row['absolute_offset_s']):.6f}",
                        "gold_onset_inside_candidate_segment": row[
                            "gold_onset_inside_candidate_segment"
                        ],
                    }
                )
            else:
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "gold_onset_s": f"{float(row['gold_onset_s']):.6f}",
                        "candidate_start_s": "",
                        "candidate_end_s": "",
                        "signed_offset_s": "",
                        "absolute_offset_s": "",
                        "gold_onset_inside_candidate_segment": "no",
                    }
                )


def run_timing_control(
    match_rows: list[dict[str, object]],
    by_clip: dict[str, list[dict[str, float]]],
) -> tuple[list[float], float]:
    observed = [float(r["absolute_offset_s"]) for r in match_rows if r["matched"]]
    if not observed:
        raise SystemExit("STOP: no matched usable onsets; cannot compute timing control")
    observed_median = float(np.median(np.asarray(observed, dtype=float)))
    rng = np.random.default_rng(SEED)
    null: list[float] = []
    for _ in range(N_RANDOM):
        errors: list[float] = []
        for row in match_rows:
            sid = str(row["sample_id"])
            refuse_gold_test(sample_id=sid)
            if not row["matched"]:
                continue
            pseudo = sample_pseudo_onset(
                rng,
                float(row["window_start_s"]),
                float(row["window_end_s"]),
                float(row["gold_onset_s"]),
                CONTROL_EXCLUSION_S,
            )
            hit = match_onset_to_candidates(pseudo, by_clip[sid])
            if hit is None:
                raise SystemExit(f"STOP: control match missing candidates for {sid}")
            errors.append(float(hit["absolute_offset_s"]))
        if not errors:
            raise SystemExit("STOP: empty control error vector")
        null.append(float(np.median(np.asarray(errors, dtype=float))))
    return null, observed_median


def write_control(null: list[float]) -> None:
    with CONTROL_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "median_absolute_error_s"])
        writer.writeheader()
        for i, value in enumerate(null, start=1):
            writer.writerow(
                {"iteration": i, "median_absolute_error_s": f"{float(value):.6f}"}
            )


def assert_outputs_have_no_test(paths: list[Path]) -> None:
    for path in paths:
        text = path.read_text()
        if OUTPUT_TEST_ID_RE.search(text):
            print(TEST_MSG)
            raise SystemExit(TEST_MSG)


def build_summary(
    cfg: dict,
    match_rows: list[dict[str, object]],
    usable_ids: list[str],
    excluded_ids: list[str],
    clip_stats: dict[str, dict[str, float]],
    null: list[float],
    observed_median: float,
) -> dict:
    n_usable = len(usable_ids)
    matched = [r for r in match_rows if r["matched"]]
    abs_offsets = np.asarray([float(r["absolute_offset_s"]) for r in matched], dtype=float)
    signed_offsets = np.asarray([float(r["signed_offset_s"]) for r in matched], dtype=float)
    hits_05s = int(sum(1 for r in matched if float(r["absolute_offset_s"]) <= TOLERANCES_S[0]))
    hits_10s = int(sum(1 for r in matched if float(r["absolute_offset_s"]) <= TOLERANCES_S[1]))
    hits_20s = int(sum(1 for r in matched if float(r["absolute_offset_s"]) <= TOLERANCES_S[2]))
    counts = [int(clip_stats[sid]["n_candidates"]) for sid in EXPECTED_DEV]
    coverage = [float(clip_stats[sid]["coverage_fraction"]) for sid in EXPECTED_DEV]
    null_arr = np.asarray(null, dtype=float)
    n_le = int(np.sum(null_arr <= observed_median))
    empirical_p = (1.0 + n_le) / (N_RANDOM + 1.0)
    return {
        "experiment_name": EXPERIMENT_NAME,
        "n_usable_onsets": n_usable,
        "hits_05s": hits_05s,
        "hits_10s": hits_10s,
        "hits_20s": hits_20s,
        "rate_05s": float(hits_05s / n_usable) if n_usable else 0.0,
        "rate_10s": float(hits_10s / n_usable) if n_usable else 0.0,
        "rate_20s": float(hits_20s / n_usable) if n_usable else 0.0,
        "mean_signed_offset_s": float(np.mean(signed_offsets)) if matched else None,
        "median_signed_offset_s": float(np.median(signed_offsets)) if matched else None,
        "mean_absolute_offset_s": float(np.mean(abs_offsets)) if matched else None,
        "median_absolute_offset_s": float(np.median(abs_offsets)) if matched else None,
        "onset_inside_segment_count": int(
            sum(1 for r in matched if r["gold_onset_inside_candidate_segment"] == "yes")
        ),
        "candidates_per_clip_mean": float(np.mean(np.asarray(counts, dtype=float))),
        "candidates_per_clip_range": [int(min(counts)), int(max(counts))],
        "candidate_coverage_fraction_mean": float(np.mean(np.asarray(coverage, dtype=float))),
        "random_iterations": N_RANDOM,
        "random_median_absolute_error_mean": float(np.mean(null_arr)),
        "random_median_absolute_error_std": float(np.std(null_arr)),
        "empirical_p": float(empirical_p),
        "control_exclusion_s": CONTROL_EXCLUSION_S,
        "control_exclusion_note": (
            "Pseudo onsets are drawn uniformly from the analysed interval, "
            "excluding any region within 2.0 seconds of that clip's true annotated onset."
        ),
        "correspondence_tolerances_s": list(TOLERANCES_S),
        "random_seed": SEED,
        "usable_onset_ids": usable_ids,
        "excluded_onset_ids": excluded_ids,
        "dev_clip_ids": EXPECTED_DEV,
        "annotation_files": [
            "data/gold_annotations.csv",
            "data/gold/annotation_sheet.csv",
            "data/gold/events.csv",
        ],
        "rule_config_path": "results/rule_selected_config.json",
        "chosen_rotation_axis": cfg["chosen_rotation_axis"],
        "axis_name": cfg["axis_name"],
        "selected_amplitude_threshold": cfg["selected_amplitude_threshold"],
        "min_movement_frames": cfg["min_movement_frames"],
        "max_movement_frames": cfg["max_movement_frames"],
        "fps": FPS,
        "n_matched_onsets": len(matched),
    }


def main() -> None:
    refuse_forbidden_argv()
    parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_rows, n_usable, n_excluded = audit_dev_onsets()
    write_onset_audit(audit_rows)
    print_audit_counts(n_usable, n_excluded)
    cfg = load_frozen_rule_config()
    candidate_rows, by_clip, clip_stats = extract_dev_candidates(audit_rows, cfg)
    write_candidates(candidate_rows)
    match_rows, usable_ids, excluded_ids = match_usable_onsets(audit_rows, by_clip)
    if len(usable_ids) != n_usable:
        raise SystemExit("STOP: usable onset count mismatch between audit and matching")
    write_matches(match_rows)
    null, observed_median = run_timing_control(match_rows, by_clip)
    write_control(null)
    summary = build_summary(
        cfg, match_rows, usable_ids, excluded_ids, clip_stats, null, observed_median
    )
    dump_json(SUMMARY_JSON, summary)
    outputs = [AUDIT_CSV, CANDIDATES_CSV, MATCHES_CSV, CONTROL_CSV, SUMMARY_JSON]
    assert_outputs_have_no_test(outputs)
    print(f"wrote {OUT_DIR.relative_to(ROOT)}")
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
