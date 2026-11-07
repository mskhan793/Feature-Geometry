"""Test fixtures: a tiny AquaMonitor-like HDF5 and classification parquet (software tests only, not research data)."""

import io

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

FOLD_VALUES = [  # per image: fold0..fold4; image 5 is outside the benchmark ('0' in every fold)
    ["train", "val", "test", "train", "train"],
    ["train", "val", "test", "train", "train"],
    ["val", "train", "train", "test", "train"],
    ["test", "train", "train", "train", "val"],
    ["train", "test", "val", "train", "train"],
    ["0", "0", "0", "0", "0"],
    ["train", "train", "train", "val", "test"],
]


def jpeg(seed: int, size=(40, 30)) -> bytes:
    rgb = np.random.default_rng(seed).integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def aqua_files(tmp_path):
    """Parquet in authors' order; HDF5 holding the same images in a DIFFERENT order (positions must come from names)."""
    n = len(FOLD_VALUES)
    names = [f"img_{i}.jpg" for i in range(n)]
    parquet = pd.DataFrame({
        "img": names, "individual": [f"spec_{i // 2}" for i in range(n)], "imaging_run": [f"run_{i}" for i in range(n)],
        "taxon_group": ["A", "A", "B", "B", "C", "C", "A"],
        **{f"fold{k}": [row[k] for row in FOLD_VALUES] for k in range(5)},
        "sample": "s", "site": "x", "lake": "l", "year": "2021", "dataset": "bulk21", "camera": 1,
    })
    parquet_path = tmp_path / "classif.parquet"
    parquet.to_parquet(parquet_path)
    order = np.random.default_rng(0).permutation(n)  # HDF5 position p holds image order[p]
    h5_path = tmp_path / "images.h5"
    with h5py.File(h5_path, "w") as f:
        images = f.create_dataset("images", (n,), dtype=h5py.vlen_dtype(np.uint8))
        for p, i in enumerate(order):
            images[p] = np.frombuffer(jpeg(int(i)), dtype=np.uint8)
        f.create_dataset("metadata/img", data=np.array([names[i] for i in order], dtype=object), dtype=h5py.string_dtype())
    return {"parquet": parquet_path, "hdf5": h5_path, "names": names, "order": order}
