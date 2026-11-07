"""Check that the numbers in a report section come from saved evidence files.

Every decimal number (e.g. 0.43, -1.29, 3.1e-5) and every integer >= 100 (commas allowed) in the report text between
``--start`` and ``--end`` lines must equal, after rounding to the report's precision (decimals, or significant digits
for scientific notation), some number found in the evidence files. Negative report numbers may match either sign of
a stored difference. Prints unmatched numbers and exits non-zero if there are any.

Usage: check_report_numbers.py --report response.md --start "## 1." --end "## 9." --evidence FILE...
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:e-?\d+)?(?![\w])")


def parse(tok: str) -> float:
    return float(tok.replace(",", ""))


def matches(tok: str, pool: list[float]) -> bool:
    v = abs(parse(tok))
    if "e" in tok:
        mant = tok.split("e")[0].lstrip("-")
        sig = len(mant.replace(".", "").lstrip("0"))
        return any(float(f"{abs(p):.{sig - 1}e}") == v for p in pool)
    dec = len(tok.split(".")[1]) if "." in tok else 0
    return any(round(abs(p), dec) == v for p in pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--evidence", required=True, nargs="+", type=Path)
    a = ap.parse_args()
    lines = a.report.read_text().splitlines()
    i = next(k for k, s in enumerate(lines) if s.startswith(a.start))
    j = next(k for k, s in enumerate(lines) if k > i and s.startswith(a.end))
    text = re.sub(r"`[^`]*`", "", "\n".join(lines[i:j]))  # code spans hold names and paths, not results
    pool = [parse(t) for f in a.evidence for t in NUM.findall(f.read_text())]
    checked = [t for t in NUM.findall(text) if "." in t or "e" in t or abs(parse(t)) >= 100]
    missing = [t for t in checked if not matches(t, pool)]
    print(f"{len(checked)} numbers checked, {len(missing)} not found: {missing}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
