"""Compare the first ``--rows`` manifest rows of a full feature cache with a cache that holds only those rows (e.g. a
CPU re-extraction made with ``extract_features.py --limit-rows``). Both caches are loaded with
``chunk_store.load_cache`` (row coverage, image names and checksums checked) and must share encoder, checkpoint,
preprocessing and TF32 setting. Reports max and mean absolute difference and the largest relative difference of
feature norms.

Usage: compare_cache_prefix.py --manifest M.parquet --full DIR/ENCODER --prefix DIR/ENCODER --rows N --out result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.chunk_store import load_cache
from src.features.feature_lookup import SAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--full", required=True, type=Path)
    ap.add_argument("--prefix", required=True, type=Path)
    ap.add_argument("--rows", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    names = pd.read_parquet(a.manifest, columns=["img"]).img.tolist()
    xa, ma = load_cache(a.full, names)
    xb, mb = load_cache(a.prefix, names[:a.rows])
    differs = {k: (ma.get(k), mb.get(k)) for k in SAME if k != "tf32" and ma.get(k) != mb.get(k)}
    if differs:
        raise SystemExit(f"caches were made with other settings: {differs}")
    xa = xa[:a.rows]
    err = np.abs(xa.astype(np.float64) - xb)
    na, nb = np.linalg.norm(xa, axis=1), np.linalg.norm(xb, axis=1)
    result = {"full": str(a.full), "prefix": str(a.prefix), "rows": a.rows, "tf32": [ma.get("tf32"), mb.get("tf32")],
              "max_abs_diff": float(err.max()), "mean_abs_diff": float(err.mean()),
              "max_rel_norm_diff": float(np.max(np.abs(na - nb) / nb)), "bitwise_equal": bool(np.array_equal(xa, xb))}
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
