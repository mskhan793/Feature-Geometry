"""Repeatability check: compare the chunks two feature caches have in common (same rows, same image names).

Reports per chunk and overall: bitwise equality, max and mean absolute difference. Exits non-zero if a common chunk
covers different rows or names, or if no chunk is in common.

Usage: compare_feature_caches.py --a DIR/ENCODER --b DIR/ENCODER --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    common = sorted({p.name for p in a.a.glob("chunk_*.json")} & {p.name for p in a.b.glob("chunk_*.json")})
    chunks = []
    for name in common:
        ma, mb = (json.loads((d / name).read_text()) for d in (a.a, a.b))
        if (ma["start"], ma["end"], ma["names_sha1"]) != (mb["start"], mb["end"], mb["names_sha1"]):
            sys.exit(f"{name}: the two caches cover different rows")
        xa, xb = (np.load(d / name.replace(".json", ".npy")) for d in (a.a, a.b))
        err = np.abs(xa - xb)
        chunks.append({"chunk": name, "rows": int(len(xa)), "bitwise_equal": bool(np.array_equal(xa, xb)),
                       "max_abs_diff": float(err.max()), "mean_abs_diff": float(err.mean())})
    if not chunks:
        sys.exit("no chunk in common")
    result = {"a": str(a.a), "b": str(a.b), "chunks": chunks, "all_bitwise_equal": all(c["bitwise_equal"] for c in chunks),
              "max_abs_diff": max(c["max_abs_diff"] for c in chunks)}
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
