"""Report cite keys used in the manuscript that are missing from refs.bib.

Usage:  python scripts/check_citations.py [paper2]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "paper2") / "tex"
CITE = re.compile(r"\\cite[a-z]*\{([^}]*)\}")
ENTRY = re.compile(r"@\w+\{([^,]+),")


def main() -> int:
    text = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "sections").glob("*.tex"))
    text += (ROOT / "main.tex").read_text(encoding="utf-8")
    cited: set[str] = set()
    for m in CITE.finditer(text):
        cited.update(k.strip() for k in m.group(1).split(","))
    have = set(ENTRY.findall((ROOT / "refs.bib").read_text(encoding="utf-8")))
    missing = sorted(cited - have)
    unused = sorted(have - cited)
    print(f"cited {len(cited)}  in bib {len(have)}")
    print("missing:", missing or "none")
    print("unused :", unused or "none")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
