"""Validate the Elsevier highlights file.

Neurocomputing asks for 3-5 highlights, each at most 85 characters including
spaces. The character limit is enforced by the submission system in a way that
is easy to trip with a long word, so check it here.

Usage:
    python scripts/check_highlights.py [paper/highlights.txt]
"""

from __future__ import annotations

import sys
from pathlib import Path

MAX_CHARS = 85
MIN_N, MAX_N = 3, 5


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/highlights.txt")
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    bullets = [line.strip() for line in lines if line.strip()]
    ok = True
    if not MIN_N <= len(bullets) <= MAX_N:
        print(f"FAIL: {len(bullets)} highlights, expected {MIN_N}-{MAX_N}")
        ok = False
    for i, bullet in enumerate(bullets, start=1):
        n = len(bullet)
        status = "ok" if n <= MAX_CHARS else "TOO LONG"
        if n > MAX_CHARS:
            ok = False
        print(f"{i}: {n:3d}/{MAX_CHARS} {status}  {bullet}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
