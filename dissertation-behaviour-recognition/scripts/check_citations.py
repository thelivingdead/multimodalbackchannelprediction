#!/usr/bin/env python3
"""Report BibTeX keys vs citation_register.md. Does not invent metadata."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    bib = (ROOT / "references.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    reg = (ROOT / "reports" / "dissertation_evidence" / "citation_register.md").read_text()
    used = set(re.findall(r"`?([a-z0-9]+20[0-9]{2}[a-z0-9]*)`?", reg.lower()))
    print("bib keys", sorted(keys))
    print("register mentions", sorted(used & {k.lower() for k in keys}))


if __name__ == "__main__":
    main()
