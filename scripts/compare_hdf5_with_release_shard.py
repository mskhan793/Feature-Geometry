"""Read-only check that image bytes in our self-built AquaMonitor HDF5 equal the dataset authors' released images.

For every JPEG in the given released WebDataset shards (Hugging Face ``mikkoim/aquamonitor``, ``images/*.tar``) the
HDF5 position is looked up by name in the HDF5's own ``metadata/img`` (never by a DataFrame index). Raw bytes are
compared by sha256; where they differ, decoded pixels are compared as well, so re-encoding is told apart from a wrong
image. Totals over all shards report coverage: every HDF5 position seen exactly once means the shards and the HDF5
hold the same set of images.

Usage: compare_hdf5_with_release_shard.py --hdf5 H5 --out result.json [--workers N] SHARD.tar [SHARD.tar ...]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import tarfile
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

COUNTS = ("members", "not_in_hdf5", "bytes_identical", "bytes_differ_pixels_identical", "pixels_differ")


def pixels(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"))


POSITION_OF: pd.Series | None = None  # name -> HDF5 position; set in the parent before the worker processes fork


def compare_shard(args: tuple[Path, Path]) -> tuple[dict, np.ndarray]:
    """Compare one shard; return its counts and the HDF5 positions it covered."""
    hdf5_path, shard_path = args
    position_of = POSITION_OF
    counts = dict.fromkeys(COUNTS, 0)
    rows = []
    examples = {"not_in_hdf5": [], "bytes_differ_pixels_identical": [], "pixels_differ": []}
    with h5py.File(hdf5_path, "r") as f, tarfile.open(shard_path) as tar:
        images = f["images"]
        for member in tar:
            if not member.isfile():
                continue
            counts["members"] += 1
            name = Path(member.name).name
            released = tar.extractfile(member).read()
            if name not in position_of.index:
                counts["not_in_hdf5"] += 1
                examples["not_in_hdf5"].append(name)
                continue
            row = int(position_of[name])
            rows.append(row)
            ours = images[row].tobytes()
            if hashlib.sha256(ours).digest() == hashlib.sha256(released).digest():
                counts["bytes_identical"] += 1
            elif np.array_equal(pixels(ours), pixels(released)):
                counts["bytes_differ_pixels_identical"] += 1
                examples["bytes_differ_pixels_identical"].append(name)
            else:
                counts["pixels_differ"] += 1
                examples["pixels_differ"].append(name)
    return {"shard": shard_path.name, **counts, "examples": {k: v[:10] for k, v in examples.items()}}, np.array(rows, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("shards", nargs="+", type=Path)
    a = ap.parse_args()
    global POSITION_OF
    with h5py.File(a.hdf5, "r") as f:
        names = np.array([v.decode() for v in f["metadata/img"][:]], dtype=object)
    assert pd.Index(names).is_unique, "HDF5 image names are not unique"
    POSITION_OF = pd.Series(np.arange(len(names)), index=names)
    seen = np.zeros(len(names), dtype=np.int64)
    per_shard, t0 = [], time.time()
    with mp.get_context("fork").Pool(min(a.workers, len(a.shards))) as pool:
        for counts, rows in pool.imap_unordered(compare_shard, [(a.hdf5, s) for s in a.shards]):
            np.add.at(seen, rows, 1)
            per_shard.append(counts)
            print(json.dumps({k: v for k, v in counts.items() if k != "examples"}), f"{time.time() - t0:.0f} s", flush=True)
    per_shard.sort(key=lambda s: s["shard"])
    totals = {k: sum(s[k] for s in per_shard) for k in COUNTS}
    coverage = {"hdf5_rows": len(names), "rows_seen_once": int((seen == 1).sum()), "rows_never_seen": int((seen == 0).sum()),
                "rows_seen_more_than_once": int((seen > 1).sum())}
    result = {"hdf5": str(a.hdf5), "shards": len(per_shard), "totals": totals, "coverage": coverage, "per_shard": per_shard}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"totals": totals, "coverage": coverage}, indent=1))


if __name__ == "__main__":
    main()
