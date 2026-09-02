#!/usr/bin/env python3
"""Compile shake_event_entry.csv into official DEV shake times.

Does not write nod files or 3 s windows.

  python scripts/import_shake_event_entry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_protocol import SHAKE_EVENTS_CSV, SHAKE_STATUS_CSV, compile_shake_entry_file  # noqa: E402


def main() -> None:
    ev, st, log = compile_shake_entry_file()
    print(f"shake events written : {len(ev)} -> {SHAKE_EVENTS_CSV.relative_to(ROOT)}")
    print(f"clips with shakes    : {ev['sample_id'].nunique() if not ev.empty else 0}")
    print(f"status written       : {SHAKE_STATUS_CSV.relative_to(ROOT)}")
    print(st.to_string(index=False))
    for line in log:
        print(line)
    print("3 s windows were not generated. Nod files were not written.")


if __name__ == "__main__":
    main()
