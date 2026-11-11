"""Compare two sets of verified feature caches on the images they share, matched by image name (caches may be chunked
differently, e.g. the same images encoded on two machines). Read-only.

Reports rows compared, max / mean absolute difference, max and 99th-percentile relative norm difference per image and
the minimum cosine similarity; the settings of both sides (encoder, checkpoint, transform, TF32) must agree.

Usage: compare_caches_by_name.py --names MANIFEST --a MANIFEST=CACHE ... --b MANIFEST=CACHE ... --out result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_lookup import SAME, features_by_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, type=Path, help="parquet whose img column lists the images to compare")
    ap.add_argument("--a", required=True, nargs="+")
    ap.add_argument("--b", required=True, nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    names = pd.read_parquet(a.names, columns=["img"]).img.tolist()
    pair = lambda specs: [tuple(Path(p) for p in s.split("=", 1)) for s in specs]  # noqa: E731
    xa, ma = features_by_name(pair(a.a), names)
    xb, mb = features_by_name(pair(a.b), names)
    differs = {k: (ma.get(k), mb.get(k)) for k in SAME if ma.get(k) != mb.get(k)}
    if differs:
        raise RuntimeError(f"the two sides were made with other settings: {differs}")
    xa, xb = xa.astype(np.float64), xb.astype(np.float64)
    diff = np.abs(xa - xb)
    rel = np.linalg.norm(xa - xb, axis=1) / np.linalg.norm(xa, axis=1)
    cos = (xa * xb).sum(1) / (np.linalg.norm(xa, axis=1) * np.linalg.norm(xb, axis=1))
    result = {"names": str(a.names), "rows": len(names), "a": a.a, "b": a.b,
              "settings": {k: ma.get(k) for k in SAME}, "max_abs_diff": float(diff.max()),
              "mean_abs_diff": float(diff.mean()), "max_rel_norm_diff": float(rel.max()),
              "p99_rel_norm_diff": float(np.quantile(rel, 0.99)), "min_cosine": float(cos.min()),
              "bitwise_equal_rows": int((diff.max(1) == 0).sum())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
