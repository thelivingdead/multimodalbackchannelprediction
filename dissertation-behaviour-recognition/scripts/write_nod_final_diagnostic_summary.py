#!/usr/bin/env python3
"""Write results/windowed_dev/final_diagnostic_summary.md from stored JSON.

DEV only. TEST is not read.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "windowed_dev" / "final_diagnostic_summary.md"
DROP = ROOT / "results" / "windowed_dev" / "drop_bias_audit" / "drop_bias_metrics.json"
FIXED = ROOT / "results" / "windowed_dev" / "videomae_identity_fixed"
FIXED_1P5 = ROOT / "results" / "windowed_dev" / "videomae_identity_fixed_1p5s"


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def ba_line(title: str, metrics: dict | None) -> str:
    if metrics is None:
        return f"- {title}: not yet written"
    ba = metrics["balanced_accuracy"]
    boot = metrics.get("clip_bootstrap", {}).get("balanced_accuracy", {})
    lo = boot.get("ci_lower_95")
    hi = boot.get("ci_upper_95")
    interval = "" if lo is None else f" [{lo:.3f}, {hi:.3f}]"
    cm = metrics.get("confusion", {})
    return (
        f"- {title}: balanced accuracy {ba:.3f}{interval}; "
        f"P {metrics.get('precision', float('nan')):.3f}; "
        f"R {metrics.get('recall', float('nan')):.3f}; "
        f"F1 {metrics.get('f1', float('nan')):.3f}; "
        f"PR AUC {metrics.get('pr_auc', float('nan')):.3f}; "
        f"TP/FP/TN/FN {cm.get('tp')}/{cm.get('fp')}/{cm.get('tn')}/{cm.get('fn')}; "
        f"n+ {metrics.get('n_positive')} n- {metrics.get('n_negative')}"
    )


def main() -> None:
    drop = load(DROP)
    frozen = load(FIXED / "frozen_encoder" / "metrics.json")
    last2 = load(FIXED / "last_blocks_unfrozen" / "metrics.json")
    one = load(FIXED_1P5 / "last_blocks_unfrozen" / "metrics.json")
    one_thr = load(FIXED_1P5 / "last_blocks_unfrozen_train_threshold" / "metrics.json")
    noflip = load(FIXED_1P5 / "last_blocks_no_hflip" / "metrics.json")

    lines = [
        "# DEV nod diagnostic and 1.5 s temporal sampling",
        "",
        "All numbers below are DEV only. TEST was not read or scored.",
        "The 3 s identity-fixed VideoMAE directories were not overwritten.",
        "",
        "## 1. Crop retention bias",
        "",
    ]
    if drop is None:
        lines.append("Drop-bias audit has not been written yet.")
    else:
        lines.extend(
            [
                f"- Positive windows: {drop['n_positive']} total, "
                f"{drop['retained_positive']} retained, {drop['dropped_positive']} dropped "
                f"({100 * drop['drop_rate_positive']:.1f}%).",
                f"- Negative windows: {drop['n_negative']} total, "
                f"{drop['retained_negative']} retained, {drop['dropped_negative']} dropped "
                f"({100 * drop['drop_rate_negative']:.1f}%).",
                f"- Relative risk of dropping a positive window: "
                f"{drop['relative_risk_drop_positive']:.3f}.",
                f"- Fisher exact test (two-sided) p = {drop['fisher_p_two_sided']}.",
                f"- Pose traces available: {drop['pose_available']}.",
                "",
                drop["inference_note"],
                "",
                "Motion comparison (positive windows only) is in "
                "`drop_bias_audit/drop_bias_metrics.json` under `motion_comparison`.",
            ]
        )
    lines.extend(
        [
            "",
            "## 2. Identity-fixed VideoMAE, 3.0 s / 16 frames",
            "",
            ba_line("Frozen encoder", frozen),
            ba_line("Last two blocks unfrozen", last2),
            "",
            "Input: 373 retained of 435 DEV windows; 62 unresolved; 41 retained positives.",
            "",
            "## 3. Temporal sampling, 1.5 s / 16 frames",
            "",
            "Windows are labelled from the existing human nod intervals. "
            "Stride is 1.0 s. Identity-fixed cropper, leave-one-clip-out, "
            "last two blocks unfrozen, seed 42.",
            "",
            ba_line("Fixed threshold 0.5", one),
            ba_line("Training-clip threshold (held-out clip unused)", one_thr),
            "",
            "## 4. Horizontal flip",
            "",
            "The 3.0 s identity-fixed trainer already applied random horizontal "
            "flip to training windows only. The 1.5 s main run matches that protocol. "
            "An additional 1.5 s run without flip is stored separately if present.",
            "",
            ba_line("1.5 s, no training flip", noflip),
            "",
            "Horizontal flip increases transformed training examples. "
            "The number of independent clips remains 15.",
            "",
            "## 5. Figures",
            "",
            "PNG (300 dpi) and SVG files are in `results/windowed_dev/final_figures/` "
            "and `results/windowed_dev/drop_bias_audit/`.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
