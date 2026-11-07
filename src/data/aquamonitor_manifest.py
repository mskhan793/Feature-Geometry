"""Image manifest of the official AquaMonitor Classification benchmark (dataset authors' ``aquamonitor-classif`` parquet).

One row per benchmark image, in the original parquet order. Images outside the benchmark (``'0'`` in every fold) are
dropped. ``hdf5_row`` is the image's position in the image HDF5, looked up by name in the HDF5's own
``metadata/img``; a DataFrame index is never used as a position. The manifest is written without an index.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

FOLDS = [f"fold{k}" for k in range(5)]
SPLITS = ("train", "val", "test")
OUTSIDE = "0"
COLUMNS = ["img", "individual", "imaging_run", "taxon_group", *FOLDS, "sample", "site", "lake", "year", "dataset", "camera"]


def hdf5_positions(hdf5_path: str, names: pd.Series) -> np.ndarray:
    """HDF5 position of every image name; raises if a name is missing or the HDF5 names are not unique."""
    with h5py.File(hdf5_path, "r") as f:
        stored = pd.Index([v.decode() for v in f["metadata/img"][:]])
    if not stored.is_unique:
        raise ValueError("HDF5 image names are not unique")
    pos = stored.get_indexer(names)
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} images are not in the HDF5, e.g. {names[pos < 0].iloc[0]}")
    return pos


def build_manifest(classif_parquet: str, hdf5_path: str) -> pd.DataFrame:
    df = pd.read_parquet(classif_parquet, columns=COLUMNS)
    values = set(pd.unique(df[FOLDS].values.ravel()))
    if not values <= {*SPLITS, OUTSIDE}:
        raise ValueError(f"unexpected split values {values - {*SPLITS, OUTSIDE}}")
    outside = df[FOLDS].eq(OUTSIDE)
    if not (outside.all(axis=1) | ~outside.any(axis=1)).all():
        raise ValueError("an image is outside the benchmark in some folds only")
    df = df[~outside.any(axis=1)].reset_index(drop=True)
    if not df["img"].is_unique:
        raise ValueError("duplicate image names")
    df.insert(1, "hdf5_row", hdf5_positions(hdf5_path, df["img"]).astype(np.int64))
    return df
