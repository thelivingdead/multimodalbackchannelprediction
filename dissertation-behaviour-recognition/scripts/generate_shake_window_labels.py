#!/usr/bin/env python3
"""Write DEV 3 s shake window labels from finished event annotation.

Nod files are not written.

  python scripts/generate_shake_window_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_protocol import SHAKE_WINDOWS_DEV_CSV, WindowedProtocolError, generate_shake_dev_windows  # noqa: E402


def main() -> None:
    try:
        win = generate_shake_dev_windows()
    except WindowedProtocolError as exc:
        raise SystemExit(str(exc)) from None
    n_pos = int(win["label"].sum())
    print(f"windows written : {len(win)} -> {SHAKE_WINDOWS_DEV_CSV.relative_to(ROOT)}")
    print(f"clips           : {win['sample_id'].nunique()} DEV")
    print(f"positive        : {n_pos}")
    print(f"negative        : {len(win) - n_pos}")
    print("Nod files were not written. No model was trained.")


if __name__ == "__main__":
    main()
