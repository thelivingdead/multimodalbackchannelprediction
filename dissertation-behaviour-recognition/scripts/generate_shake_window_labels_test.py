#!/usr/bin/env python3
"""Write TEST 3 s shake window labels from finished event annotation.

Nod files and DEV shake windows are not written.

  python scripts/generate_shake_window_labels_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_protocol import (  # noqa: E402
    SHAKE_WINDOWS_TEST_CSV,
    WindowedProtocolError,
    generate_shake_test_windows,
)


def main() -> None:
    try:
        win = generate_shake_test_windows()
    except WindowedProtocolError as exc:
        raise SystemExit(str(exc)) from None
    n_pos = int(win["label"].sum())
    print(f"windows written : {len(win)} -> {SHAKE_WINDOWS_TEST_CSV.relative_to(ROOT)}")
    print(f"clips           : {win['sample_id'].nunique()} TEST")
    print(f"positive        : {n_pos}")
    print(f"negative        : {len(win) - n_pos}")
    print("Nod and DEV shake files were not written. No model was trained.")


if __name__ == "__main__":
    main()
