"""Read-only check that image bytes in our self-built AquaMonitor HDF5 equal the dataset authors' released images.

For every JPEG in one released WebDataset shard (Hugging Face ``mikkoim/aquamonitor``, ``images/*.tar``) the HDF5 row
is found by name through the original classification parquet (row index = HDF5 position, checked by
audit_aquamonitor_hdf5.py) and the row's ``metadata/img`` is confirmed. Raw bytes are compared by sha256; where they
differ, decoded pixels are compared as well, so re-encoding is told apart from a wrong image.

Usage: compare_hdf5_with_release_shard.py --hdf5 H5 --parquet CLASSIF_PARQUET --shard SHARD.tar --out result.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image


def pixels(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"))


def compare(hdf5_path: Path, parquet_path: Path, shard_path: Path) -> dict:
    row_of = pd.Series(pd.read_parquet(parquet_path, columns=["img"]).index, index=pd.read_parquet(parquet_path, columns=["img"])["img"])
    counts = {"members": 0, "not_in_parquet": 0, "name_mismatch_in_hdf5": 0, "bytes_identical": 0,
              "bytes_differ_pixels_identical": 0, "pixels_differ": 0}
    examples = {"not_in_parquet": [], "pixels_differ": [], "bytes_differ_pixels_identical": []}
    rows = []
    with h5py.File(hdf5_path, "r") as f, tarfile.open(shard_path) as tar:
        images, names = f["images"], f["metadata/img"]
        for member in tar:
            if not member.isfile():
                continue
            counts["members"] += 1
            name = Path(member.name).name
            released = tar.extractfile(member).read()
            if name not in row_of.index:
                counts["not_in_parquet"] += 1
                examples["not_in_parquet"].append(name)
                continue
            row = int(row_of[name])
            rows.append(row)
            if names[row].decode() != name:
                counts["name_mismatch_in_hdf5"] += 1
                continue
            ours = images[row].tobytes()
            if hashlib.sha256(ours).digest() == hashlib.sha256(released).digest():
                counts["bytes_identical"] += 1
            elif np.array_equal(pixels(ours), pixels(released)):
                counts["bytes_differ_pixels_identical"] += 1
                examples["bytes_differ_pixels_identical"].append(name)
            else:
                counts["pixels_differ"] += 1
                examples["pixels_differ"].append(name)
    spread = {"row_min": min(rows), "row_max": max(rows), "rows_per_decile": np.bincount(np.array(rows) * 10 // len(row_of), minlength=10).tolist()}
    return {"hdf5": str(hdf5_path), "parquet": str(parquet_path), "shard": str(shard_path), **counts, "hdf5_row_spread": spread,
            "examples": {k: v[:10] for k, v in examples.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, type=Path)
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--shard", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    result = compare(a.hdf5, a.parquet, a.shard)
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
