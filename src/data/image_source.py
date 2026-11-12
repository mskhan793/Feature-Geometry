"""The image dataset for manifest rows: AquaMonitor rows (``img``, ``hdf5_row``) from the HDF5, rows with a relative
``path`` (FIN-Benthic2) from an image folder, or CIFAR-100 rows (``img`` = benchmark ``source_id``) from the publisher's
pickles. Exactly one source must be given."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from torch.utils.data import Dataset

from src.data.cifar_images import CifarImageDataset
from src.data.file_images import FileImageDataset
from src.data.hdf5_images import Hdf5ImageDataset


def image_dataset(rows: pd.DataFrame, transform: Callable, hdf5: str | None = None,
                  image_root: str | None = None, cifar_dir: str | None = None) -> Dataset:
    if sum(x is not None for x in (hdf5, image_root, cifar_dir)) != 1:
        raise ValueError("give exactly one of hdf5, image_root and cifar_dir")
    if cifar_dir is not None:
        return CifarImageDataset(cifar_dir, rows["img"].tolist(), transform)
    if hdf5 is not None:
        return Hdf5ImageDataset(hdf5, rows["img"].tolist(), rows["hdf5_row"].tolist(), transform)
    return FileImageDataset(image_root, rows["path"].tolist(), transform)
