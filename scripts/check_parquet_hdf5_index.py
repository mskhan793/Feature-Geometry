"""Read-only: for each AquaMonitor parquet, is its row index a valid HDF5 position for the row's image?

The HDF5 stores the image name of every position (``metadata/img``). For each parquet with an ``img`` column this
reports: whether every image name is in the HDF5; how many rows have ``index == HDF5 position of img`` (safe to
read the HDF5 by index); and how many would read a different image if the index were used.

Usage: check_parquet_hdf5_index.py --hdf5 H5 --out result.json PARQUET [PARQUET ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def check(df: pd.DataFrame, position_of: pd.Series, n_hdf5: int) -> dict:
    pos = position_of.reindex(df["img"].values).values  # NaN where the name is not in the HDF5
    known = ~np.isnan(pos)
    index = df.index.values
    numeric = np.issubdtype(index.dtype, np.integer)
    same = numeric & known & (index == np.nan_to_num(pos, nan=-1))
    in_range = numeric & (index >= 0) & (index < n_hdf5) if numeric else np.zeros(len(df), bool)
    return {
        "rows": len(df),
        "index_type": type(df.index).__name__,
        "index_is_0_to_n_minus_1": bool(df.index.equals(pd.RangeIndex(len(df)))),
        "img_unique": bool(df["img"].is_unique),
        "img_not_in_hdf5": int((~known).sum()),
        "index_equals_hdf5_position": int(same.sum()),
        "index_points_to_other_image": int((in_range & known & ~same).sum()),
        "index_out_of_hdf5_range": int((~in_range).sum()),
        "rows_in_hdf5_order": bool(known.all() and np.all(np.diff(pos) > 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("parquets", nargs="+", type=Path)
    a = ap.parse_args()
    with h5py.File(a.hdf5, "r") as f:
        names = np.array([v.decode() for v in f["metadata/img"][:]], dtype=object)
    assert pd.Index(names).is_unique, "HDF5 image names are not unique"
    position_of = pd.Series(np.arange(len(names)), index=names)
    results = {}
    for path in a.parquets:
        try:
            columns = pq.ParquetFile(path).schema_arrow.names
        except Exception as e:  # recorded, not skipped silently
            results[path.name] = f"not readable as parquet: {e}"
        else:
            results[path.name] = (check(pd.read_parquet(path, columns=["img"]), position_of, len(names))
                                  if "img" in columns else "no img column")
        print(path.name, json.dumps(results[path.name]), flush=True)
    a.out.write_text(json.dumps({"hdf5": str(a.hdf5), "hdf5_rows": len(names), "parquets": results}, indent=1))


if __name__ == "__main__":
    main()
