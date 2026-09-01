#!/usr/bin/env python3
"""Compare rule baseline vs learned model on the same TEST gold videos + figures."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "nod_pipeline"


def main() -> None:
    t0 = time.time()
    rule = pd.read_csv(OUT / "rule_baseline_metrics.csv")
    learned = json.loads((OUT / "learned_test_metrics.json").read_text())
    search = pd.read_csv(OUT / "hyperparam_search.csv")

    rule_test = rule[rule.split == "test"].iloc[0]
    rows = [
        {
            "system": "A0 rule-based (pitch cycles)",
            "precision": rule_test.precision,
            "recall": rule_test.recall,
            "f1": rule_test.f1,
        },
        {
            "system": f"A1 learned ({learned['best_model']}) trained on pseudo-labels",
            "precision": learned["test_event_precision"],
            "recall": learned["test_event_recall"],
            "f1": learned["test_event_f1"],
        },
    ]
    cmp = pd.DataFrame(rows)
    cmp.to_csv(OUT / "baseline_vs_learned.csv", index=False)
    print(cmp.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(cmp["system"].str.replace(" ", "\n"), cmp["f1"])
    ax.set_ylabel("Event F1 (IoU ≥ 0.2)")
    ax.set_ylim(0, 1)
    ax.set_title("TEST gold: rule baseline vs classifier trained on pseudo-labels")
    fig.tight_layout()
    fig.savefig(OUT / "fig_baseline_vs_learned_f1.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(search["model"], search["dev_event_f1"])
    ax.set_ylabel("DEV event F1")
    ax.set_title("Hyperparameter / architecture search (DEV gold only)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hyperparam_search.png", dpi=160)
    plt.close(fig)

    elapsed = time.time() - t0
    with (OUT / "time_log.txt").open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  13_compare_and_figures.py  {elapsed:.1f}s\n")
    print("Wrote figures in", OUT)


if __name__ == "__main__":
    main()
