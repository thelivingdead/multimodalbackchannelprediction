#!/usr/bin/env python3
"""Compile nod_event_entry_test.csv into official TEST event times.

YouTube clocks (m:ss) become seconds from clip start. Does not write 3 s
windows. Does not touch DEV event or window files.

  python scripts/import_nod_event_entry_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_protocol import (  # noqa: E402
    EVENTS_TEST_CSV,
    STATUS_TEST_CSV,
    compile_test_entry_file,
)


def main() -> None:
    ev, st, log = compile_test_entry_file()
    print(f"TEST events written : {len(ev)} -> {EVENTS_TEST_CSV.relative_to(ROOT)}")
    print(f"clips with nods     : {ev['sample_id'].nunique() if not ev.empty else 0}")
    print(f"status written      : {STATUS_TEST_CSV.relative_to(ROOT)}")
    print(st.to_string(index=False))
    for line in log:
        print(line)
    print("3 s windows were not generated. DEV files were not written.")


if __name__ == "__main__":
    main()
