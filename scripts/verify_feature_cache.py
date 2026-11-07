"""Check that a feature cache covers every manifest row exactly once, in manifest order.

For every chunk: the JSON identity matches the manifest rows (image-name hash) and encoder, the ``.npy`` checksum
matches, and shape, dtype and finiteness are as recorded. Exits non-zero if any chunk is missing or wrong.

Usage: verify_feature_cache.py --manifest M.parquet --cache-dir DIR/ENCODER --encoder NAME [--chunk-size 65536]
       [--limit-rows N] --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.encoders.frozen_encoders import MODEL_IDS
from src.features import chunk_store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--chunk-size", type=int, default=65536)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    names = pd.read_parquet(a.manifest, columns=["img"])["img"]
    if a.limit_rows:
        names = names.iloc[:a.limit_rows]
    chunks, problems, dims, transforms, rows = [], [], set(), set(), 0
    for k in range(chunk_store.n_chunks(len(names), a.chunk_size)):
        start, end = chunk_store.chunk_bounds(len(names), a.chunk_size, k)
        npy, meta_path = chunk_store.paths(a.cache_dir, k)
        if not meta_path.exists():
            problems.append(f"chunk {k} missing")
            continue
        meta = json.loads(meta_path.read_text())
        identity = {"start": start, "end": end, "names_sha1": chunk_store.names_sha1(names.iloc[start:end].tolist()),
                    "encoder": a.encoder, "model_id": MODEL_IDS[a.encoder], "transform": meta.get("transform")}
        try:
            chunk_store.chunk_done(a.cache_dir, k, identity)
            x = np.load(npy, mmap_mode="r")
            if x.shape != (end - start, meta["dim"]) or x.dtype != np.float32 or not np.isfinite(x).all():
                raise RuntimeError(f"chunk {k}: shape/dtype/finite check failed")
        except RuntimeError as e:
            problems.append(str(e))
            continue
        dims.add(meta["dim"])
        transforms.add(meta["transform"])
        rows += end - start
        chunks.append({k_: meta[k_] for k_ in ("start", "end", "dim", "sha256", "norm_min", "norm_mean", "norm_max",
                                               "images_per_second")})
    if len(dims) > 1 or len(transforms) > 1:
        problems.append(f"chunks disagree: dims {dims}, {len(transforms)} transforms")
    report = {"cache_dir": str(a.cache_dir), "encoder": a.encoder, "manifest_rows": len(names), "rows_covered": rows,
              "chunks_ok": len(chunks), "dims": sorted(dims), "problems": problems, "chunks": chunks}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    ok = not problems and rows == len(names)
    print(("PASS" if ok else "FAIL"), json.dumps({k: v for k, v in report.items() if k != "chunks"}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
