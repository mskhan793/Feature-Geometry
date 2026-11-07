"""Images read from the AquaMonitor HDF5 (JPEG bytes in ``images``, names in ``metadata/img``)."""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Hdf5ImageDataset(Dataset):
    """Item ``i`` is ``(transform(image), i)`` for the image ``names[i]`` at HDF5 position ``rows[i]``.

    Every read checks that the HDF5 stores ``names[i]`` at that position, and any unreadable image raises: nothing
    is skipped or replaced. The HDF5 file is opened lazily in each DataLoader worker process.
    """

    def __init__(self, hdf5_path: str, names: Sequence[str], rows: Sequence[int], transform: Callable):
        if len(names) != len(rows):
            raise ValueError("names and rows differ in length")
        self.hdf5_path, self.names, self.rows, self.transform = hdf5_path, list(names), np.asarray(rows), transform
        self._file = None

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        row = int(self.rows[i])
        stored = self._file["metadata/img"][row].decode()
        if stored != self.names[i]:
            raise RuntimeError(f"HDF5 position {row} holds {stored}, expected {self.names[i]}")
        with Image.open(io.BytesIO(self._file["images"][row].tobytes())) as img:
            image = img.convert("RGB")
        return self.transform(image), i
