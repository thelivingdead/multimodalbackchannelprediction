#!/usr/bin/env python3
"""01 — Storage check. Hard stop at 24 GB project budget / low free space."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.storage import check_storage  # noqa: E402
from src.utils import dump_json  # noqa: E402


def largest_files(root: Path, n: int = 15) -> list[str]:
    r = subprocess.run(
        ["find", str(root), "-type", "f", "-size", "+1M"],
        capture_output=True,
        text=True,
    )
    paths = [Path(p) for p in r.stdout.splitlines() if p]
    paths.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    lines = []
    for p in paths[:n]:
        mb = p.stat().st_size / 1e6
        lines.append(f"{mb:8.1f} MB  {p.relative_to(root)}")
    return lines


def main() -> None:
    st = check_storage()
    lines = [
        st.message,
        f"project_gb={st.project_gb:.3f}",
        f"free_gb={st.free_gb:.3f}",
        f"level={st.level}",
        "",
        "largest files >1MB:",
        *largest_files(ROOT),
    ]
    text = "\n".join(lines) + "\n"
    (ROOT / "reports" / "storage_report.txt").write_text(text)
    dump_json(
        ROOT / "reports" / "storage_report.json",
        {
            "project_gb": st.project_gb,
            "free_gb": st.free_gb,
            "level": st.level,
            "ok": st.ok,
        },
    )
    print(text)
    if st.level == "hard_error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
