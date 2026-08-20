#!/usr/bin/env python3
"""VideoMAE Step 6b: assemble results/tables/main_results.csv + .md.

Rows (real values only — every metric is read from a saved artefact, and any
row whose artefact does not exist is written as N/A):

==============  ================================  ==========================
model           metrics source                    predictions source (CI)
==============  ================================  ==========================
Rule baseline   results/rule_test_metrics.json    results/rule_test_predictions.csv
Pose CNN raw    results/ablation_results.csv      *(none saved -> CI N/A)*
                (row feature_set=xyz)
Pose CNN        results/classifier_test_metrics.json  results/classifier_test_predictions.csv
xyz_deriv
Frozen VideoMAE results/videomae_frozen_head/     results/videomae_frozen_head/
head            metrics.json                      predictions.csv
==============  ================================  ==========================

CIs come from ``results/tables/bootstrap_ci.csv`` (scripts/bootstrap_f1.py);
a missing CI row prints N/A. Columns::

    model, input, supervision, train_n, precision, recall, f1, accuracy,
    f1_ci_lo, f1_ci_hi

Lab invocation::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    source ../.venv/bin/activate
    python scripts/make_main_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "tables"
OUT_CSV = OUT_DIR / "main_results.csv"
OUT_MD = OUT_DIR / "main_results.md"

RULE_METRICS = ROOT / "results" / "rule_test_metrics.json"
ABLATION = ROOT / "results" / "ablation_results.csv"
CNN_METRICS = ROOT / "results" / "classifier_test_metrics.json"
VMAE_METRICS = ROOT / "results" / "videomae_frozen_head" / "metrics.json"
PSEUDO_LABELS = ROOT / "results" / "pseudo_labels.csv"
BOOTSTRAP = OUT_DIR / "bootstrap_ci.csv"

NA = "N/A"
COLUMNS = [
    "model", "input", "supervision", "train_n",
    "precision", "recall", "f1", "accuracy", "f1_ci_lo", "f1_ci_hi",
]


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_cis() -> dict[str, dict]:
    if not BOOTSTRAP.exists():
        return {}
    with BOOTSTRAP.open(newline="") as fh:
        return {r["model"]: r for r in csv.DictReader(fh)}


def pseudo_train_n() -> int | None:
    if not PSEUDO_LABELS.exists():
        return None
    with PSEUDO_LABELS.open(newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def metric(m: dict | None, key: str):
    return m[key] if m is not None and key in m else NA


def ci_fields(cis: dict[str, dict], key: str) -> dict:
    row = cis.get(key)
    if row is None:
        return {"f1_ci_lo": NA, "f1_ci_hi": NA}
    return {"f1_ci_lo": float(row["f1_ci_lo"]),
            "f1_ci_hi": float(row["f1_ci_hi"])}


def build_rows() -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    cis = load_cis()
    n_pseudo = pseudo_train_n()
    rows: list[dict] = []

    # --- Rule baseline -------------------------------------------------
    m = load_json(RULE_METRICS)
    if m is None:
        notes.append("Rule baseline: results/rule_test_metrics.json missing -> N/A")
    rows.append({
        "model": "Rule baseline",
        "input": "EMOCA head rotation",
        "supervision": "None (DEV-tuned thresholds)",
        "train_n": NA,
        "precision": metric(m, "precision"),
        "recall": metric(m, "recall"),
        "f1": metric(m, "f1"),
        "accuracy": metric(m, "accuracy"),
        **ci_fields(cis, "rule_baseline"),
    })

    # --- Pose CNN raw (xyz, no derivatives) ----------------------------
    raw = None
    if ABLATION.exists():
        with ABLATION.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["feature_set"] == "xyz":
                    raw = r
    if raw is None:
        notes.append(
            "Pose CNN raw: no feature_set=xyz row in "
            "results/ablation_results.csv -> N/A"
        )
    rows.append({
        "model": "Pose CNN raw",
        "input": "EMOCA pose xyz (128-step resampled)",
        "supervision": f"{n_pseudo if n_pseudo is not None else '80'} rule pseudo-labels",
        "train_n": n_pseudo if n_pseudo is not None else NA,
        "precision": float(raw["test_precision"]) if raw else NA,
        "recall": float(raw["test_recall"]) if raw else NA,
        "f1": float(raw["test_f1"]) if raw else NA,
        "accuracy": float(raw["test_accuracy"]) if raw else NA,
        # ablation runs saved no predictions.csv -> CI unavailable
        "f1_ci_lo": NA,
        "f1_ci_hi": NA,
    })

    # --- Pose CNN xyz_deriv (main pseudo CNN) --------------------------
    m = load_json(CNN_METRICS)
    if m is None:
        notes.append(
            "Pose CNN xyz_deriv: results/classifier_test_metrics.json "
            "missing -> N/A"
        )
    rows.append({
        "model": "Pose CNN xyz_deriv",
        "input": "EMOCA pose xyz + derivatives",
        "supervision": f"{n_pseudo if n_pseudo is not None else '80'} rule pseudo-labels",
        "train_n": n_pseudo if n_pseudo is not None else NA,
        "precision": metric(m, "precision"),
        "recall": metric(m, "recall"),
        "f1": metric(m, "f1"),
        "accuracy": metric(m, "accuracy"),
        **ci_fields(cis, "pose_cnn_xyz_deriv"),
    })

    # --- Frozen VideoMAE head ------------------------------------------
    m = load_json(VMAE_METRICS)
    if m is None:
        notes.append(
            "Frozen VideoMAE head: results/videomae_frozen_head/metrics.json "
            "missing (Steps 3-5 not run) -> N/A row"
        )
    tm = (m or {}).get("test_metrics")
    rows.append({
        "model": "Frozen VideoMAE head",
        "input": f"RGB 16x224x224 face crops ({(m or {}).get('checkpoint', 'VideoMAE')})",
        "supervision": f"{(m or {}).get('train_n', n_pseudo if n_pseudo is not None else 80)} rule pseudo-labels",
        "train_n": (m or {}).get("train_n", NA),
        "precision": metric(tm, "precision"),
        "recall": metric(tm, "recall"),
        "f1": metric(tm, "f1"),
        "accuracy": metric(tm, "accuracy"),
        **ci_fields(cis, "videomae_frozen_head"),
    })
    return rows, notes


def fmt(v) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def main() -> None:
    rows, notes = build_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "|" + "|".join(" --- " for _ in COLUMNS) + "|"
    lines = [
        "# Main results (TEST, scored once per model under the frozen protocol)",
        "",
        "Real values only, read from the saved artefacts listed per row; "
        "`N/A` = not run (or no saved predictions.csv for the CI).",
        "",
        header,
        sep,
    ]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[c]) for c in COLUMNS) + " |")
    lines += [
        "",
        "Sources: Rule baseline <- `results/rule_test_metrics.json`; "
        "Pose CNN raw <- `results/ablation_results.csv` (feature_set `xyz`); "
        "Pose CNN xyz_deriv <- `results/classifier_test_metrics.json`; "
        "Frozen VideoMAE head <- `results/videomae_frozen_head/metrics.json`; "
        "CIs <- `results/tables/bootstrap_ci.csv` "
        "(1000 resamples, seed 42, from saved TEST predictions).",
    ]
    if notes:
        lines += ["", "Row status notes:"] + [f"- {n}" for n in notes]
    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"wrote {OUT_CSV} and {OUT_MD}")
    for r in rows:
        print(f"  {r['model']}: f1={fmt(r['f1'])} "
              f"ci=[{fmt(r['f1_ci_lo'])}, {fmt(r['f1_ci_hi'])}]")
    for n in notes:
        print(f"  NOTE: {n}")


if __name__ == "__main__":
    main()
