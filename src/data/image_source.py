"""The image dataset for manifest rows: AquaMonitor rows (``img``, ``hdf5_row``) from the HDF5, or rows with a
relative ``path`` (FIN-Benthic2) from an image folder. Exactly one source must be given."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from torch.utils.data import Dataset

from src.data.file_images import FileImageDataset
from src.data.hdf5_images import Hdf5ImageDataset


def image_dataset(rows: pd.DataFrame, transform: Callable, hdf5: str | None = None,
                  image_root: str | None = None) -> Dataset:
    if (hdf5 is None) == (image_root is None):
        raise ValueError("give exactly one of hdf5 and image_root")
    if hdf5 is not None:
        return Hdf5ImageDataset(hdf5, rows["img"].tolist(), rows["hdf5_row"].tolist(), transform)
    return FileImageDataset(image_root, rows["path"].tolist(), transform)
