#!/usr/bin/env python3
"""Compile shake_event_entry_test.csv into official TEST shake times.

Does not write nod files, DEV shake files, or 3 s windows.

  python scripts/import_shake_event_entry_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_protocol import (  # noqa: E402
    SHAKE_EVENTS_TEST_CSV,
    SHAKE_STATUS_TEST_CSV,
    compile_shake_test_entry_file,
)


def main() -> None:
    ev, st, log = compile_shake_test_entry_file()
    print(f"TEST shake events written : {len(ev)} -> {SHAKE_EVENTS_TEST_CSV.relative_to(ROOT)}")
    print(f"clips with shakes         : {ev['sample_id'].nunique() if not ev.empty else 0}")
    print(f"status written            : {SHAKE_STATUS_TEST_CSV.relative_to(ROOT)}")
    print(st.to_string(index=False))
    for line in log:
        print(line)
    print("3 s windows were not generated. Nod and DEV shake files were not written.")


if __name__ == "__main__":
    main()
