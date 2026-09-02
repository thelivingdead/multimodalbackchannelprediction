#!/usr/bin/env python3
"""Compile nod_event_entry.csv into official DEV event times.

YouTube clocks (m:ss) become seconds from clip start. Does not write 3 s windows.
TEST is refused.

  python scripts/import_nod_event_entry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_protocol import compile_entry_file  # noqa: E402


def main() -> None:
    ev, st, log = compile_entry_file()
    print(f"events written : {len(ev)}")
    print(f"clips with nods: {ev['sample_id'].nunique() if not ev.empty else 0}")
    print(st.to_string(index=False))
    for line in log:
        print(line)
    print("3 s windows were not generated.")


if __name__ == "__main__":
    main()
