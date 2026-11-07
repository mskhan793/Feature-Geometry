"""Hdf5ImageDataset: right image for each name; a wrong position or an unreadable image raises."""

import io

import h5py
import numpy as np
import pytest
from PIL import Image
from torchvision import transforms as T

from src.data.aquamonitor_manifest import build_manifest
from src.data.hdf5_images import Hdf5ImageDataset
from tests.conftest import jpeg


def test_items_are_the_named_images(aqua_files):
    m = build_manifest(aqua_files["parquet"], aqua_files["hdf5"])
    ds = Hdf5ImageDataset(str(aqua_files["hdf5"]), m["img"].tolist(), m["hdf5_row"].tolist(), T.PILToTensor())
    for i, name in enumerate(m["img"]):
        image, index = ds[i]
        source = int(name.split("_")[1].split(".")[0])  # fixture image i was made from seed i
        expected = T.PILToTensor()(Image.open(io.BytesIO(jpeg(source))).convert("RGB"))
        assert index == i and (image == expected).all()


def test_wrong_position_raises(aqua_files):
    m = build_manifest(aqua_files["parquet"], aqua_files["hdf5"])
    rows = m["hdf5_row"].tolist()
    rows[0], rows[1] = rows[1], rows[0]
    ds = Hdf5ImageDataset(str(aqua_files["hdf5"]), m["img"].tolist(), rows, T.PILToTensor())
    with pytest.raises(RuntimeError, match="holds"):
        ds[0]


def test_unreadable_image_raises(aqua_files):
    with h5py.File(aqua_files["hdf5"], "a") as f:
        f["images"][0] = np.frombuffer(b"not a jpeg", dtype=np.uint8)
        name = f["metadata/img"][0].decode()
    ds = Hdf5ImageDataset(str(aqua_files["hdf5"]), [name], [0], T.PILToTensor())
    with pytest.raises(Exception):
        ds[0]
