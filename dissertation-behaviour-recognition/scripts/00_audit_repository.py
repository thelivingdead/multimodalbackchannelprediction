#!/usr/bin/env python3
"""00 — Repository tree + what already exists. Does not overwrite old nod_pipeline."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import ensure_dirs  # noqa: E402
from src.storage import check_storage  # noqa: E402
from src.utils import dump_json, git_commit  # noqa: E402


def tree(path: Path, max_depth: int = 3, prefix: str = "") -> list[str]:
    lines: list[str] = []
    skip = {".venv", "__pycache__", ".git", ".hf_cache", "node_modules"}
    try:
        kids = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return lines
    for i, p in enumerate(kids):
        if p.name in skip or p.name.startswith("."):
            continue
        last = i == len(kids) - 1
        branch = "└── " if last else "├── "
        lines.append(prefix + branch + p.name + ("/" if p.is_dir() else ""))
        if p.is_dir() and max_depth > 1:
            ext = "    " if last else "│   "
            lines.extend(tree(p, max_depth - 1, prefix + ext))
    return lines


def main() -> None:
    ensure_dirs()
    st = check_storage()
    parent = ROOT.parent
    existing = {
        "parent_nod_pipeline": str(parent / "scripts" / "nod_pipeline"),
        "parent_nod_pipeline_exists": (parent / "scripts" / "nod_pipeline").exists(),
        "realtalk_nod_forecasting": str(parent / "realtalk_nod_forecasting"),
        "note": "Existing scripts were not overwritten. This package is new and self-contained.",
        "git": git_commit(parent),
        "storage": st.message,
    }
    text = ["dissertation-behaviour-recognition/", *tree(ROOT, 3)]
    (ROOT / "reports" / "repository_audit.txt").write_text("\n".join(text) + "\n\n" + str(existing) + "\n")
    dump_json(ROOT / "reports" / "repository_audit.json", existing)
    print("\n".join(text[:80]))
    print("Wrote reports/repository_audit.txt")
    print(st.message)


if __name__ == "__main__":
    main()
