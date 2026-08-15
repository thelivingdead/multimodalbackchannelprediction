#!/usr/bin/env bash
# Print large paths. Never deletes anything.
set -euo pipefail
echo "=== quota / disk ==="
df -h . 2>/dev/null || true
quota -s 2>/dev/null || true
echo
echo "=== this repo (top) ==="
du -sh . data results figures logs checkpoints .venv 2>/dev/null || true
echo
echo "=== files larger than 200 MB (repo + home cache, max 40) ==="
{
  find . -type f -size +200M 2>/dev/null
  find "${HOME}/.cache" -type f -size +200M 2>/dev/null
} | head -40
echo
echo "Nothing was deleted. See LAB_CLEANUP.md before removing anything."
