#!/usr/bin/env python3
"""Build video-disjoint balanced shake pseudo-label manifests.

Does **not** overwrite ``results/shake/pseudo_labels.csv`` (locked 75/5).
Does **not** use gold labels as TRAIN targets. Does **not** score GOLD TEST.
Does **not** rewrite ``results/shake/rule_selected_config.json`` (locked z).

After the DEV axis audit: if frozen **z** is geometrically wrong, new
pseudo-labels are ranked on the DEV-chosen axis (yaw-like **y**). The locked
z τ≈11.15° cut is **not** used as a 0/1 assignment here — that cut is the
75-pos / 5-neg collapse. Do **not** switch this script back to frozen-z
0/1: the Mac 80-clip pool has only 5 frozen-z negatives, so 40/40 is
impossible and the collapse is reproduced.

Operational definition (pseudo, not gold)
-----------------------------------------
* **Positives:** highest oscillatory amplitude on the geometric shake axis
  (Savitzky–Golay half-cycle ``rule_score``, same detector as the pose rule).
* **Hard negatives:** remaining clips with **large motion that is not a
  shake** — nod-like (high x), yaw turns (large y range, weak oscillation),
  mid-range conversation motion. Near-zero static clips are used last.

High-confidence = top tail of positives (axis score ≥ 75th percentile) and
confident hard negatives (axis score ≤ 25th percentile), still preferring
nod/turn types.

Writes::

    results/shake/pseudo_balanced/pool_scored.csv
    results/shake/pseudo_balanced/labelling.json
    results/shake/pseudo_balanced/manifest_40_40.csv
    results/shake/pseudo_balanced/manifest_20_20_highconf.csv  (Mac 80-pool)
    results/shake/pseudo_balanced/manifest_80_80.csv           (if enough)
    ... 100/100, 200/200 only if the pose pool is large enough
    results/shake/pseudo_balanced/README.md

Otter95 (pose npz only; CPU)::

    cd ~/multimodalbackchannelprediction/dissertation-behaviour-recognition
    OMP_NUM_THREADS=1 /scratch/db01550/venv/bin/python \\
        scripts/build_shake_balanced_pseudo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_split_leakage  # noqa: E402
from shake_v2_common import (  # noqa: E402
    load_npz,
    max_abs_step,
    n_direction_changes,
    npz_video_id,
    rule_score,
)

SHEET = ROOT / "data" / "gold" / "shake_annotation_sheet.csv"
RULE_CFG = ROOT / "results" / "shake" / "rule_selected_config.json"
AUDIT_CANDIDATES = (
    ROOT / "results" / "shake" / "dev_balanced" / "axis_audit_conclusion.json",
    ROOT / "results" / "shake" / "dev_search" / "axis_audit_conclusion.json",
    ROOT / "results" / "shake" / "v2" / "axis_audit" / "axis_audit_conclusion.json",
)
PSEUDO_DIR = ROOT / "features" / "pseudo"
LOCKED_CSV = ROOT / "results" / "shake" / "pseudo_labels.csv"
OUT = ROOT / "results" / "shake" / "pseudo_balanced"
TARGETS = (40, 80, 100, 200)
STATIC_PTP = 8.0
SEED = 42
AXIS_NAMES = ("x", "y", "z")
DEFAULT_SHAKE_AXIS = "y"
DEFAULT_NOD_AXIS = "x"
STALE_ZCUT = (
    "manifest_MAX_all.csv",
    "manifest_MAX_highconf.csv",
)


def _as_str(value) -> str:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, OSError):
            pass
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).strip()


def gold_block() -> tuple[set[str], set[str]]:
    g = pd.read_csv(SHEET, dtype=str)
    g["split"] = g["split"].astype(str).str.upper()
    part = g[g["split"].isin(["DEV", "TEST"])]
    return set(part["video_id"].astype(str)), set(g["sample_id"].astype(str))


def chosen_axes() -> tuple[str, str, dict]:
    shake_ax, nod_ax = DEFAULT_SHAKE_AXIS, DEFAULT_NOD_AXIS
    audit: dict = {}
    for cand in AUDIT_CANDIDATES:
        if cand.exists():
            audit = json.loads(cand.read_text())
            shake_ax = str(audit.get("geometric_shake_axis") or shake_ax)
            nod_ax = str(audit.get("geometric_nod_axis") or nod_ax)
            break
    if shake_ax not in AXIS_NAMES or nod_ax not in AXIS_NAMES:
        raise SystemExit(f"STOP: bad audit axes shake={shake_ax} nod={nod_ax}")
    return shake_ax, nod_ax, audit


def score_pool() -> pd.DataFrame:
    paths = sorted(PSEUDO_DIR.glob("pseudo_*.npz"))
    if not paths:
        raise SystemExit(f"STOP: no {PSEUDO_DIR}/*.npz — not inventing clips")
    rows = []
    for p in paths:
        z = load_npz(p)
        rot = np.asarray(z["rotation_xyz"], dtype=float)
        vid = _as_str(z["video_id"]) if "video_id" in z else npz_video_id(p)
        ptp = np.ptp(rot, axis=0) if len(rot) else np.zeros(3)
        scores = [float(rule_score(rot, ax)) for ax in range(3)]
        y = np.asarray(rot[:, 1], dtype=float) if len(rot) else np.array([])
        rows.append(
            {
                "sample_id": p.stem,
                "video_id": vid,
                "score_x": scores[0],
                "score_y": scores[1],
                "score_z": scores[2],
                "ptp_x": float(ptp[0]),
                "ptp_y": float(ptp[1]),
                "ptp_z": float(ptp[2]),
                "y_max_step": max_abs_step(y) if y.size else 0.0,
                "y_turns_raw": n_direction_changes(y) if y.size else 0,
                "n_frames": int(len(rot)),
            }
        )
    return pd.DataFrame(rows)


def tag_hard_neg(pool: pd.DataFrame, shake_ax: str, nod_ax: str) -> pd.DataFrame:
    df = pool.copy()
    s_col = f"score_{shake_ax}"
    n_col = f"score_{nod_ax}"
    ptp_s = f"ptp_{shake_ax}"
    y_q20 = float(df[s_col].quantile(0.20))
    y_q50 = float(df[s_col].quantile(0.50))
    x_med = float(df[n_col].median())
    ptp_med = float(df[ptp_s].median())
    osc_frac = df[s_col] / df[ptp_s].clip(lower=1e-6)
    df["static_low_motion"] = (
        df[["ptp_x", "ptp_y", "ptp_z"]].max(axis=1) < STATIC_PTP
    ).astype(int)
    df["neg_type_nod"] = (df[n_col] >= x_med).astype(int)
    df["neg_type_turn"] = ((df[ptp_s] >= ptp_med) & (osc_frac < 0.45)).astype(int)
    df["neg_type_mid"] = ((df[s_col] >= y_q20) & (df[s_col] <= y_q50)).astype(int)
    # nod-like, turn, mid-range conversation, other motion; static last
    df["hard_neg_rank"] = (
        4.0 * df["neg_type_nod"]
        + 3.0 * df["neg_type_turn"]
        + 2.0 * df["neg_type_mid"]
        + 1.0 * (1 - df["static_low_motion"])
        - 5.0 * df["static_low_motion"]
        + 0.01 * df[n_col]
        + 0.005 * df[ptp_s]
    )
    df["rank_pos"] = df[s_col]
    return df


def take_pos_neg(
    pool: pd.DataFrame,
    n_each: int,
    shake_ax: str,
    hi: bool,
) -> pd.DataFrame | None:
    """Rank-based 1:1 draw on the DEV geometric shake axis (not frozen-z τ)."""
    src = pool.copy()
    s_col = f"score_{shake_ax}"
    if hi:
        q75 = float(src[s_col].quantile(0.75))
        q25 = float(src[s_col].quantile(0.25))
        pos_c = src[src[s_col] >= q75].sort_values(s_col, ascending=False)
        neg_c = src[src[s_col] <= q25].sort_values("hard_neg_rank", ascending=False)
        n_avail = min(n_each, len(pos_c), len(neg_c))
        if n_avail < 4 or n_avail < n_each:
            return None
        pos = pos_c.head(n_avail)
        neg = neg_c.head(n_avail)
    else:
        if len(src) < 2 * n_each:
            return None
        ranked = src.sort_values(s_col, ascending=False)
        pos = ranked.head(n_each)
        rest = ranked.iloc[n_each:]
        if len(rest) < n_each:
            return None
        neg = rest.sort_values("hard_neg_rank", ascending=False).head(n_each)
    pos = pos.copy()
    neg = neg.copy()
    pos["pseudo_label"] = 1
    neg["pseudo_label"] = 0
    out = pd.concat([pos, neg], ignore_index=True)
    out["rule_score"] = out[s_col]
    return out.sort_values(["pseudo_label", "sample_id"]).reset_index(drop=True)


def write_csv(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() == LOCKED_CSV.resolve():
        raise SystemExit("STOP: refusing to overwrite results/shake/pseudo_labels.csv")
    cols = [
        "sample_id",
        "video_id",
        "pseudo_label",
        "rule_score",
        "score_x",
        "score_y",
        "score_z",
        "ptp_x",
        "ptp_y",
        "ptp_z",
        "hard_neg_rank",
        "neg_type_nod",
        "neg_type_turn",
        "neg_type_mid",
        "static_low_motion",
        "track",
        "n_each",
        "shake_axis",
    ]
    keep = [c for c in cols if c in df.columns]
    df[keep].to_csv(dest, index=False)


def _record(
    dest: Path,
    df: pd.DataFrame,
    n_each: int,
    hi: bool,
    shake_ax: str,
    written: list[dict],
    alias: str | None = None,
) -> None:
    df = df.copy()
    df["track"] = "highconf" if hi else "all"
    df["n_each"] = int((df.pseudo_label == 1).sum())
    df["shake_axis"] = shake_ax
    write_csv(df, dest)
    if alias:
        write_csv(df, OUT / alias)
    check_split_leakage.run(
        gold_csv=SHEET, pseudo_labels=dest, labelled_train_only=True
    )
    rec = {
        "file": dest.name,
        "requested_n_each": n_each,
        "high_confidence": hi,
        "alias": alias or "",
        "status": "wrote",
        "actual_pos": int((df.pseudo_label == 1).sum()),
        "actual_neg": int((df.pseudo_label == 0).sum()),
        "path": dest.relative_to(ROOT).as_posix(),
        "leakage": "PASS",
        "n_static_neg": int(
            ((df.pseudo_label == 0) & (df.static_low_motion == 1)).sum()
        ),
        "n_hard_neg_nod": int(
            ((df.pseudo_label == 0) & (df.neg_type_nod == 1)).sum()
        ),
        "n_hard_neg_turn": int(
            ((df.pseudo_label == 0) & (df.neg_type_turn == 1)).sum()
        ),
    }
    written.append(rec)
    print(
        f"wrote {dest.name}  {rec['actual_pos']} pos / "
        f"{rec['actual_neg']} neg  leakage PASS"
    )


def main() -> None:
    if not RULE_CFG.exists():
        raise SystemExit(
            f"STOP: frozen shake rule missing ({RULE_CFG}). "
            "Do not invent axis/τ; do not re-run the rule TEST."
        )
    if LOCKED_CSV.exists():
        old = pd.read_csv(LOCKED_CSV)
        print(
            f"locked 75/5 left untouched: {LOCKED_CSV} "
            f"({int((old.pseudo_label == 1).sum())} pos / "
            f"{int((old.pseudo_label == 0).sum())} neg)"
        )
    cfg = json.loads(RULE_CFG.read_text())
    locked_ax = str(cfg.get("axis_name") or "z")
    locked_tau = float(cfg["selected_amplitude_threshold"])
    shake_ax, nod_ax, audit = chosen_axes()
    print(
        f"locked TEST rule: axis {locked_ax} τ={locked_tau:.3f}° (not rewritten)\n"
        f"new TRAIN ranking axis: {shake_ax} (hard-neg nod axis {nod_ax})"
    )
    if not any(p.exists() for p in AUDIT_CANDIDATES):
        print(
            "NOTE: axis_audit_conclusion.json missing; defaulting ranking axis "
            f"to {shake_ax}. Run scripts/audit_shake_axis_dev.py first."
        )
    if shake_ax == "z":
        print(
            "WARNING: audit chose z. That matches the locked TEST rule. "
            "The 80-clip pool still cannot yield 40 frozen-z negatives."
        )

    gold_vids, gold_sids = gold_block()
    pool = score_pool()
    n_npz = len(pool)
    leak = pool["video_id"].isin(gold_vids) | pool["sample_id"].isin(gold_sids)
    if leak.any():
        print(f"dropping {int(leak.sum())} gold DEV/TEST overlapping clips")
        pool = pool[~leak].copy()
    pool = tag_hard_neg(pool, shake_ax, nod_ax)
    s_col = f"score_{shake_ax}"
    pool["rule_score"] = pool[s_col]
    pool["locked_z_pseudo_label"] = (pool["score_z"] >= locked_tau).astype(int)

    OUT.mkdir(parents=True, exist_ok=True)
    # Never glob-delete y-ranking manifests. Only drop leftover frozen-z MAX files.
    for name in STALE_ZCUT:
        p = OUT / name
        if p.exists():
            p.unlink()
            print(f"removed stale frozen-z leftover {name}")
    pool.sort_values("sample_id").to_csv(OUT / "pool_scored.csv", index=False)

    n_locked_pos = int(pool["locked_z_pseudo_label"].sum())
    n_locked_neg = int((pool["locked_z_pseudo_label"] == 0).sum())
    print(
        f"eligible pool {len(pool)} / {n_npz} npz  "
        f"(locked-z 0/1 would be {n_locked_pos} pos / {n_locked_neg} neg — not used)"
    )

    written: list[dict] = []
    for n_each in TARGETS:
        for hi, suffix in ((False, ""), (True, "_highconf")):
            name = f"manifest_{n_each}_{n_each}{suffix}.csv"
            df = take_pos_neg(pool, n_each, shake_ax, hi=hi)
            if df is None:
                leftover = OUT / name
                if leftover.exists():
                    leftover.unlink()
                    print(f"removed incomplete leftover {name}")
                written.append(
                    {
                        "file": name,
                        "requested_n_each": n_each,
                        "high_confidence": hi,
                        "status": "skipped_insufficient_data",
                        "actual_pos": 0,
                        "actual_neg": 0,
                        "path": "",
                        "leakage": "",
                    }
                )
                print(f"SKIP {name}: not enough eligible clips (pool={len(pool)})")
            else:
                n_act = int((df.pseudo_label == 1).sum())
                code = {40: "A", 80: "B", 100: "C", 200: "D"}.get(int(n_each))
                alias = None
                if code and not hi:
                    alias = f"{code}_{n_each}_{n_each}.csv"
                elif code and hi:
                    alias = f"{code}_{n_act}_{n_act}_hi.csv"
                _record(OUT / name, df, n_each, hi, shake_ax, written, alias)

    # Disk-limited highconf (Mac 80 npz → ~20/20 at p75/p25)
    wrote_hc = any(
        r.get("status") == "wrote" and r.get("high_confidence") for r in written
    )
    q75 = float(pool[s_col].quantile(0.75))
    q25 = float(pool[s_col].quantile(0.25))
    hmax = int(min((pool[s_col] >= q75).sum(), (pool[s_col] <= q25).sum()))
    if hmax >= 4 and not wrote_hc:
        hdf = take_pos_neg(pool, hmax, shake_ax, hi=True)
        if hdf is not None:
            dest = OUT / f"manifest_{hmax}_{hmax}_highconf.csv"
            _record(
                dest,
                hdf,
                hmax,
                True,
                shake_ax,
                written,
                alias=f"A_{hmax}_{hmax}_hi.csv",
            )
            print(f"disk-limited highconf size {hmax}/{hmax} (not 40/40)")

    labelling = {
        "pseudo_not_gold": True,
        "ranking_axis": shake_ax,
        "nod_axis_for_hard_neg": nod_ax,
        "locked_test_rule_axis": locked_ax,
        "locked_test_rule_threshold_deg": locked_tau,
        "locked_z_not_used_as_01_cut": True,
        "geometric_shake_axis_from_audit": audit.get("geometric_shake_axis"),
        "current_z_rule_geometrically_supported": audit.get(
            "current_z_rule_geometrically_supported"
        ),
        "n_npz_on_disk": n_npz,
        "n_eligible_after_gold_exclusion": int(len(pool)),
        "locked_z_pos_if_used": n_locked_pos,
        "locked_z_neg_if_used": n_locked_neg,
        "pseudo_00081_to_00200_present": bool(
            (PSEUDO_DIR / "pseudo_00081.npz").exists()
        ),
        "locked_75_5_untouched": str(LOCKED_CSV.as_posix()),
        "seed": SEED,
        "static_ptp_deg": STATIC_PTP,
        "highconf_percentiles": {"pos": 0.75, "neg": 0.25},
        "note": (
            "Labels are rank-based on the DEV geometric axis, not gold and "
            "not the locked z τ cut. Locked TEST still used z and is not scored."
        ),
        "manifests": written,
        "sizes": {str(r.get("alias") or r.get("file")): r for r in written},
    }
    payload = json.dumps(labelling, indent=2) + "\n"
    (OUT / "labelling.json").write_text(payload)
    (OUT / "eligibility.json").write_text(payload)
    pd.DataFrame(written).to_csv(OUT / "manifest_index.csv", index=False)

    lines = [
        "# Shake balanced pseudo-labels (new protocol)",
        "",
        "Locked `results/shake/pseudo_labels.csv` (**75 pos / 5 neg**, frozen **z**) "
        "was **not** overwritten. Labels here are **pseudo**, not gold. GOLD TEST "
        "was not used.",
        "",
        "## Axis",
        "",
        f"- Locked TEST rule (untouched): **{locked_ax}**, τ = {locked_tau:.3f}°",
        f"- New ranking axis (DEV geometry): **{shake_ax}**",
        f"- Hard-negative nod-like axis: **{nod_ax}**",
        "- The locked z 0/1 cut is **not** the TRAIN label here (that is the 75/5 collapse).",
        "",
        "## Eligible pool on this disk",
        "",
        f"- `features/pseudo/*.npz`: **{n_npz}**",
        f"- after dropping gold DEV/TEST `video_id`: **{len(pool)}**",
        "",
        "40/40 needs 80 clips; 80/80 needs 160; 100/100 needs 200. This Mac has the "
        "original 80 pose files unless otter streamed `pseudo_00081.npz`…",
        "",
        "## Hard negatives",
        "",
        "Ranked among the non-positive remainder:",
        "",
        "1. **Nod-like** — high Euler x oscillatory amplitude",
        "2. **Turns** — large shake-axis range with weak oscillation (look-aside)",
        "3. **Mid-range motion** — conversation, not a static head",
        "4. Static (max ptp < 8°) last, never first",
        "",
        "## Manifests",
        "",
        "| file | n/class | highconf | status | leakage |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for r in written:
        lines.append(
            f"| {r.get('file', '')} | {r.get('requested_n_each', '')} | "
            f"{r.get('high_confidence')} | {r.get('status')} | "
            f"{r.get('leakage') or '—'} |"
        )
    lines += [
        "",
        "Leakage: `python scripts/check_split_leakage.py --gold-csv "
        "data/gold/shake_annotation_sheet.csv --pseudo-labels <manifest>`.",
        "No dyad column exists on gold/pseudo; disjointness is by `video_id`.",
        "",
        "Train with `--dev-only` / `--no-test` into `results/shake/dev_balanced/<config>/`.",
        "Do not score GOLD TEST.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"README → {OUT / 'README.md'}")
    print("did not overwrite", LOCKED_CSV)


if __name__ == "__main__":
    main()
