"""Read-only check that the AquaMonitor image HDF5 is row-aligned with the original authors' classification parquet.

1. Every metadata column stored in the HDF5 (``metadata/<column>``) is compared with the same parquet column for
   all rows; ``labels`` is compared through ``class_info/class_names`` with ``taxon_group``.
2. Image bytes of a fixed spread of rows (every ``--step``-th row plus three blocks of 200 consecutive rows) are
   decoded and their pixel size is compared with the row's ``width`` / ``height``. For each mismatch the rows at
   offsets -2..+2 are tried too, so that a shifted array would be visible as a consistent non-zero offset.

Usage: audit_aquamonitor_hdf5.py --hdf5 H5 --parquet CLASSIF_PARQUET --out audit.json [--step 1000]
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

OFFSETS = (-2, -1, 1, 2)


def as_text(values: np.ndarray) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def compare_column(stored: np.ndarray, expected: pd.Series) -> np.ndarray:
    """Boolean array: row-wise equality of an HDF5 metadata column with the parquet column."""
    if pd.api.types.is_datetime64_any_dtype(expected):
        return pd.to_datetime(as_text(stored)).values == expected.values
    if stored.dtype.kind in "OSU":
        return as_text(stored) == expected.astype(str).values
    return stored == expected.values


def image_size(images: h5py.Dataset, row: int) -> tuple[int, int]:
    with Image.open(io.BytesIO(images[row].tobytes())) as img:
        img.load()  # full decode: a truncated or corrupt JPEG fails here
        return img.size


def audit(hdf5_path: Path, parquet_path: Path, step: int) -> dict:
    df = pd.read_parquet(parquet_path)
    report: dict = {"hdf5": str(hdf5_path), "parquet": str(parquet_path), "parquet_rows": len(df),
                    "parquet_index_is_range": bool(df.index.equals(pd.RangeIndex(len(df))))}
    with h5py.File(hdf5_path, "r") as f:
        images = f["images"]
        report["hdf5_rows"] = int(images.shape[0])
        columns = {}
        for name in sorted(f["metadata"].keys()):
            if name not in df.columns:
                columns[name] = "not in parquet"
                continue
            equal = compare_column(f["metadata"][name][:], df[name])
            bad = np.flatnonzero(~equal)
            columns[name] = {"mismatches": int(bad.size), "first_mismatch_row": int(bad[0]) if bad.size else None}
        report["metadata_columns"] = columns
        report["parquet_columns_not_in_hdf5"] = sorted(set(df.columns) - set(f["metadata"].keys()))
        names = as_text(f["class_info/class_names"][:])
        label_equal = names[f["labels"][:]] == df["taxon_group"].astype(str).values
        report["labels_vs_taxon_group_mismatches"] = int((~label_equal).sum())

        n = len(df)
        rows = sorted(set(range(0, n, step)) | {n - 1} | set(range(200)) | set(range(n // 2, n // 2 + 200)) | set(range(n - 200, n)))
        mismatched, offset_hits, errors = [], {k: 0 for k in OFFSETS}, []
        for row in rows:
            try:
                size = image_size(images, row)
            except Exception as e:  # report, never skip silently
                errors.append({"row": row, "error": repr(e)})
                continue
            if size != (df.at[row, "width"], df.at[row, "height"]):
                mismatched.append(row)
                for k in OFFSETS:
                    if 0 <= row + k < n and size == (df.at[row + k, "width"], df.at[row + k, "height"]):
                        offset_hits[k] += 1
        report["image_check"] = {"rows_checked": len(rows), "decode_errors": errors, "size_mismatches": len(mismatched),
                                 "first_mismatches": mismatched[:20], "mismatches_matching_neighbour_row": offset_hits,
                                 "distinct_sizes_among_checked": int(df.loc[rows, ["width", "height"]].drop_duplicates().shape[0])}
    report["benchmark_rows_fold0"] = int(df["fold0"].isin(["train", "val", "test"]).sum())
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, type=Path)
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--step", type=int, default=1000)
    a = ap.parse_args()
    report = audit(a.hdf5, a.parquet, a.step)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
