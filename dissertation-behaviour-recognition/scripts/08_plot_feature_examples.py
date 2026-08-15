#!/usr/bin/env python3
"""08 — Pitch/yaw/roll plots aligned with gold (class 1) intervals."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotations import gold_nods, load_events  # noqa: E402
from src.data import default_pilot_dir, list_clip_dirs, read_meta  # noqa: E402
from src.plotting import save_publication_figure  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--behaviour", default="nod")
    p.add_argument("--split", default="pilot")
    p.parse_args()
    out = ROOT / "figures" / "pilot_nod"
    out.mkdir(parents=True, exist_ok=True)
    gold = gold_nods(load_events(ROOT / "data" / "gold" / "events.csv"))
    n = 0
    for c in list_clip_dirs(default_pilot_dir()):
        m = read_meta(c)
        vid = str(m["video_id"])
        hp = ROOT / "data" / "headpose" / f"{vid}.csv"
        if not hp.exists():
            continue
        df = pd.read_csv(hp)
        fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True)
        for ax, col, ylab in zip(axes, ("pitch", "yaw", "roll"), ("pitch (deg)", "yaw (deg)", "roll (deg)")):
            if col not in df.columns:
                ax.set_visible(False)
                continue
            ax.plot(df["time_s"], df[col], lw=1.0, color="C0")
            ax.set_ylabel(ylab)
            for e in gold:
                if e.video_id != vid:
                    continue
                ax.axvspan(e.start_s, e.end_s, color="C3", alpha=0.25)
        axes[-1].set_xlabel("time (s)")
        axes[0].set_title(f"{vid} — red = gold clear nod (class 1)")
        save_publication_figure(fig, out / f"{vid}_pose_gold", source=str(hp), force=True)
        n += 1
    print(f"Wrote {n} pose+gold figures in {out}")


if __name__ == "__main__":
    main()
