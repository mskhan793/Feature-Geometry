"""Manifest: benchmark rows only, folds preserved, HDF5 position from the image name (never the DataFrame index)."""

import h5py
import numpy as np
import pandas as pd
import pytest

from src.data.aquamonitor_manifest import FOLDS, build_manifest


def test_manifest_drops_outside_rows_and_keeps_folds(aqua_files):
    m = build_manifest(aqua_files["parquet"], aqua_files["hdf5"])
    source = pd.read_parquet(aqua_files["parquet"])
    kept = source[source["fold0"] != "0"].reset_index(drop=True)
    assert m["img"].tolist() == kept["img"].tolist()  # original order, '0' row removed
    for fold in FOLDS:
        assert m[fold].tolist() == kept[fold].tolist()


def test_hdf5_row_is_looked_up_by_name(aqua_files):
    m = build_manifest(aqua_files["parquet"], aqua_files["hdf5"])
    with h5py.File(aqua_files["hdf5"], "r") as f:
        stored = [v.decode() for v in f["metadata/img"][:]]
    assert [stored[r] for r in m["hdf5_row"]] == m["img"].tolist()
    assert not np.array_equal(m["hdf5_row"].to_numpy(), np.arange(len(m)))  # the fixture HDF5 order differs


def test_image_missing_from_hdf5_raises(aqua_files, tmp_path):
    df = pd.read_parquet(aqua_files["parquet"])
    df.loc[0, "img"] = "not_there.jpg"
    df.to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="not in the HDF5"):
        build_manifest(tmp_path / "bad.parquet", aqua_files["hdf5"])


def test_outside_in_some_folds_only_raises(aqua_files, tmp_path):
    df = pd.read_parquet(aqua_files["parquet"])
    df.loc[0, "fold3"] = "0"
    df.to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="some folds only"):
        build_manifest(tmp_path / "bad.parquet", aqua_files["hdf5"])
