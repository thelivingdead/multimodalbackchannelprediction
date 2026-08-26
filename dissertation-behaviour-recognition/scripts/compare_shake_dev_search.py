#!/usr/bin/env python3
"""DEV-only shake comparison table (does not score GOLD TEST).

Reads locked 75/5 DEV numbers already stored in metrics.json (dev_f1 /
dev_metrics). Does **not** re-read TEST for selection. Writes::

    results/shake/dev_search/comparison_dev.md
    results/shake/dev_search/comparison_dev.csv

Best model = highest DEV F1 among runs that are **not** collapsed
(predicted-positive rate > 0.85 or TN=0). Always-shake on DEV (10/5) has
F1 0.80 with TN=0 and is rejected as a trained system.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clip_metrics import always_predict, collapse_diagnostics  # noqa: E402

SHEET = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
SEARCH = ROOT / "results" / "shake" / "dev_search"
LOCKED = ROOT / "results" / "shake"


def _metrics_row(
    name: str,
    *,
    precision: float,
    recall: float,
    f1: float,
    balanced_accuracy: float,
    tp: int | None = None,
    fp: int | None = None,
    tn: int | None = None,
    fn: int | None = None,
    collapse: bool | None = None,
    pred_pos_rate: float | None = None,
    train_pos: int | None = None,
    train_neg: int | None = None,
    path: str = "",
    note: str = "",
    eligible: bool = True,
) -> dict:
    if collapse is None and tn is not None and pred_pos_rate is not None:
        collapse = bool(pred_pos_rate > 0.85 or tn == 0)
    elif collapse is None and tn is not None:
        collapse = bool(tn == 0)
    return {
        "system": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "collapse": bool(collapse) if collapse is not None else "",
        "predicted_positive_rate": pred_pos_rate,
        "train_pos": train_pos,
        "train_neg": train_neg,
        "eligible_for_best": bool(eligible and not collapse),
        "path": path,
        "note": note,
    }


def _from_json(path: Path, name: str) -> dict | None:
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    dm = m.get("dev_metrics") or {}
    tn = dm.get("tn", m.get("tn"))
    pred_rate = m.get("predicted_positive_rate")
    n = None
    if all(k in dm for k in ("tp", "fp", "tn", "fn")):
        n = int(dm["tp"]) + int(dm["fp"]) + int(dm["tn"]) + int(dm["fn"])
        if pred_rate is None and n:
            pred_rate = (int(dm["tp"]) + int(dm["fp"])) / n
    collapse = m.get("collapse")
    if collapse is None:
        if n and pred_rate is not None:
            n_pos = int(round(float(pred_rate) * n))
            dummy_pred = [1] * n_pos + [0] * (n - n_pos)
            collapse = collapse_diagnostics(dummy_pred, int(tn if tn is not None else 0))[
                "collapse"
            ]
        else:
            collapse = bool(tn == 0) if tn is not None else False
    return _metrics_row(
        name,
        precision=float(dm.get("precision", m.get("dev_precision", float("nan")))),
        recall=float(dm.get("recall", m.get("dev_recall", float("nan")))),
        f1=float(dm.get("f1", m.get("dev_f1", float("nan")))),
        balanced_accuracy=float(
            dm.get("balanced_accuracy", m.get("dev_balanced_accuracy", float("nan")))
        ),
        tp=dm.get("tp"),
        fp=dm.get("fp"),
        tn=dm.get("tn"),
        fn=dm.get("fn"),
        collapse=bool(collapse),
        pred_pos_rate=pred_rate,
        train_pos=m.get("train_pos"),
        train_neg=m.get("train_neg"),
        path=str(path.relative_to(ROOT).as_posix()),
        note=m.get("selection_rule", ""),
        eligible=not bool(collapse),
    )


def _scan_search() -> list[dict]:
    rows = []
    if not SEARCH.exists():
        return rows
    for p in sorted(SEARCH.glob("*/dev_metrics.json")):
        name = p.parent.name
        row = _from_json(p, f"search:{name}")
        if row:
            rows.append(row)
    return rows


def _best(rows: list[dict], predicate) -> dict | None:
    cand = [r for r in rows if predicate(r) and r.get("eligible_for_best")]
    if not cand:
        # still report the highest F1 even if collapsed, but not as best
        return None
    return max(cand, key=lambda r: (float(r["f1"]), float(r.get("balanced_accuracy") or 0)))


def main() -> None:
    gold = pd.read_csv(SHEET)
    gold["split"] = gold["split"].astype(str).str.upper()
    dev = gold[gold.split == "DEV"]
    y = pd.to_numeric(dev["shake_label"], errors="coerce").astype(int).to_numpy()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    always1 = always_predict(y, 1)
    always1_coll = collapse_diagnostics([1] * len(y), always1["tn"])

    rows: list[dict] = [
        _metrics_row(
            "always-shake baseline (DEV)",
            precision=always1["precision"],
            recall=always1["recall"],
            f1=always1["f1"],
            balanced_accuracy=always1["balanced_accuracy"],
            tp=always1["tp"],
            fp=always1["fp"],
            tn=always1["tn"],
            fn=always1["fn"],
            collapse=always1_coll["collapse"],
            pred_pos_rate=always1_coll["predicted_positive_rate"],
            path="(no model)",
            note="predict 1 on every DEV clip; F1=0.80 on 10/5 is not a trained win",
            eligible=False,
        )
    ]

    cnn75 = _from_json(LOCKED / "cnn" / "metrics.json", "locked 75/5 pose CNN")
    if cnn75:
        cnn75["note"] = (
            "DEV numbers from locked metrics.json; TEST exists but is "
            "not used for selection"
        )
        rows.append(cnn75)
    fr75 = _from_json(
        LOCKED / "videomae_frozen_head" / "metrics.json",
        "locked 75/5 frozen VideoMAE",
    )
    if fr75:
        # frozen json has no DEV confusion
        fr75["note"] = (
            "DEV F1/balanced acc from locked metrics.json; per-clip DEV "
            "confusion was not stored. TEST not used for selection."
        )
        rows.append(fr75)
    ft75 = _from_json(
        LOCKED / "videomae_finetuned" / "metrics.json",
        "locked 75/5 fine-tuned VideoMAE",
    )
    if ft75:
        ft75["note"] = (
            "DEV numbers from locked metrics.json; TEST exists but is "
            "not used for selection"
        )
        rows.append(ft75)

    search_rows = _scan_search()
    rows.extend(search_rows)

    def is_cnn(r):
        s = r["system"].lower()
        return "cnn" in s and "locked" not in s

    def is_frozen(r):
        s = r["system"].lower()
        return "frozen" in s and "locked" not in s and "highconf" not in s

    def is_ft(r):
        s = r["system"].lower()
        return ("ft4" in s or "finetun" in s) and "locked" not in s and "highconf" not in s

    def is_highconf(r):
        return "highconf" in r["system"].lower() and "locked" not in r["system"].lower()

    bests = {
        "best balanced pose CNN (DEV)": _best(rows, is_cnn),
        "best balanced frozen VideoMAE (DEV)": _best(rows, is_frozen),
        "best balanced fine-tuned VideoMAE (DEV)": _best(rows, is_ft),
        "best high-confidence balanced VideoMAE (DEV)": _best(rows, is_highconf),
    }

    SEARCH.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(SEARCH / "comparison_dev.csv", index=False)

    lines = [
        "# Shake DEV-only comparison",
        "",
        f"GOLD DEV class counts: **{n_pos} shake+ / {n_neg} shake−** (n=15).",
        "Selection uses DEV only. GOLD TEST was **not** scored for this search.",
        "",
        "Collapse rule: predicted-positive rate on DEV **> 0.85** or **TN=0**.",
        "Always-shake on this DEV split has F1 **0.80** (TP10 FP5 TN0 FN0) and is "
        "not a trained system.",
        "",
        "## Headline rows",
        "",
        "| system | P | R | F1 | bal-acc | TP FP TN FN | collapse | train pos/neg | path |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]

    def fmt(r: dict | None, fallback: str) -> None:
        if r is None:
            lines.append(f"| {fallback} | — | — | — | — | — | — | — | not run yet |")
            return
        cm = ""
        if r.get("tp") is not None:
            cm = f"{r['tp']} {r['fp']} {r['tn']} {r['fn']}"
        else:
            cm = "confusion not stored"
        tr = ""
        if r.get("train_pos") is not None:
            tr = f"{r['train_pos']}/{r['train_neg']}"
        lines.append(
            f"| {r['system']} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f} | {r['balanced_accuracy']:.3f} | {cm} | "
            f"{r['collapse']} | {tr} | `{r['path']}` |"
        )

    fmt(rows[0], "always-shake")
    fmt(cnn75, "locked 75/5 CNN")
    fmt(fr75, "locked 75/5 frozen VideoMAE")
    fmt(ft75, "locked 75/5 fine-tuned VideoMAE")
    for label, r in bests.items():
        fmt(r, label)

    lines += [
        "",
        "## All DEV-search runs",
        "",
        "| system | P | R | F1 | collapse | pred+ rate | path |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    if search_rows:
        for r in search_rows:
            rate = r.get("predicted_positive_rate")
            rate_s = f"{rate:.3f}" if isinstance(rate, float) else "—"
            lines.append(
                f"| {r['system']} | {r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['f1']:.3f} | {r['collapse']} | {rate_s} | `{r['path']}` |"
            )
    else:
        lines.append("| *(no `dev_search/*/dev_metrics.json` yet)* | | | | | | |")

    overall = _best(search_rows, lambda r: True)
    if overall is None and search_rows:
        # all collapsed — still name the least-bad by bAcc then F1, but ineligible
        overall_note = (
            "every new run collapsed on DEV; do not treat as a TEST winner"
        )
        overall_pick = max(
            search_rows,
            key=lambda r: (
                float(r.get("balanced_accuracy") or 0),
                float(r["f1"]),
            ),
        )
    else:
        overall_note = "highest DEV F1 among non-collapsed search runs"
        overall_pick = overall

    lines += [
        "",
        "## Window length",
        "",
        "Pose CNN default is 128 resampled steps; one cheap variant uses "
        "`--seq-len 64` (`cnn_*_seq64`). VideoMAE rgb16 stays 16×224×224 "
        "(no second window without a new fetch).",
        "",
        "## One best config (DEV only — not scored on TEST)",
        "",
    ]
    if overall_pick is None:
        lines.append("No new `dev_search/*/dev_metrics.json` yet. Run otter jobs.")
    else:
        lines.append(
            f"**{overall_pick['system']}**  F1={overall_pick['f1']:.3f}  "
            f"P={overall_pick['precision']:.3f}  R={overall_pick['recall']:.3f}  "
            f"bAcc={overall_pick['balanced_accuracy']:.3f}  "
            f"collapse={overall_pick['collapse']}  "
            f"({overall_note})"
        )
        lines.append(f"Path: `{overall_pick['path']}`")
    lines += [
        "",
        "## Do not report a winner TEST F1",
        "",
        "Locked TEST numbers from the 75/5 protocol are already known and are "
        "**not** a selection criterion. Next evaluation should be a **fresh "
        "10–15 clip holdout** from videos that are **not** in the gold 30.",
        "",
        "Footnote (already-known locked TEST, not for selection): shake rule "
        "TEST F1 0.70; 75/5 CNN TEST F1 0.64 with TN=0; 75/5 VideoMAE TEST F1 "
        "0.60. Ignore these when picking the best new run.",
        "",
        "Student git-pushes themselves. Do not score GOLD TEST yet.",
        "",
    ]
    (SEARCH / "comparison_dev.md").write_text("\n".join(lines) + "\n")
    table.to_csv(SEARCH / "summary.csv", index=False)
    best_payload = {
        "dev_n_pos": n_pos,
        "dev_n_neg": n_neg,
        "always_shake_dev_f1": always1["f1"],
        "best_eligible": overall,
        "best_reported": overall_pick,
        "note": overall_note if overall_pick is not None else "no search runs",
        "test_scored": False,
    }
    (SEARCH / "best_config.json").write_text(
        json.dumps(best_payload, indent=2, default=str) + "\n"
    )
    print(f"wrote {SEARCH / 'comparison_dev.md'}")
    print(f"wrote {SEARCH / 'summary.csv'}")
    print(f"DEV always-shake F1={always1['f1']:.3f}  "
          f"TN={always1['tn']}  collapse={always1_coll['collapse']}")
    for label, r in bests.items():
        if r is None:
            print(f"  {label}: not available (or all collapsed)")
        else:
            print(f"  {label}: F1={r['f1']:.3f}  {r['system']}")
    if overall_pick:
        print(f"  overall: {overall_pick['system']} F1={overall_pick['f1']:.3f}")


if __name__ == "__main__":
    main()
