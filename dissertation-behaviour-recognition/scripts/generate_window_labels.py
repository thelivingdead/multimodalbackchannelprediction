#!/usr/bin/env python3
"""Write DEV 3 s window labels from finished event annotation.

TEST is refused. No training, no metrics.

  python scripts/generate_window_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_protocol import (  # noqa: E402
    WINDOWS_DEV_CSV,
    WindowedProtocolError,
    generate_dev_windows,
)


def main() -> None:
    try:
        win = generate_dev_windows()
    except WindowedProtocolError as exc:
        raise SystemExit(str(exc)) from None
    n_pos = int(win["label"].sum())
    print(f"windows written : {len(win)} -> {WINDOWS_DEV_CSV.relative_to(ROOT)}")
    print(f"clips           : {win['sample_id'].nunique()} DEV")
    print(f"positive        : {n_pos}")
    print(f"negative        : {len(win) - n_pos}")
    print("TEST not loaded. No model was trained.")


if __name__ == "__main__":
    main()
