"""CIFAR-100 images from the publisher's python pickles (``train`` / ``test``), addressed by the benchmark manifest's
``source_id`` = ``"<part>[<index>]"`` (array position; the publisher's file names are not unique). Each 3072-byte row is
reshaped to 32 x 32 RGB (channel-major, as documented on the CIFAR page) and returned unchanged as a PIL image; any
resizing is the encoder's own transform."""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

REF = re.compile(r"^(train|test)\[(\d+)\]$")


class CifarImageDataset(Dataset):
    """Item ``i`` is ``(transform(image), i)`` for ``source_ids[i]``; an unknown reference raises."""

    def __init__(self, folder: str | Path, source_ids: Sequence[str], transform: Callable):
        parts = {m.group(1) for m in map(REF.match, source_ids)} if source_ids else set()
        self.arrays = {p: pickle.load(open(Path(folder) / p, "rb"), encoding="latin1")["data"] for p in parts}
        self.refs = [(m.group(1), int(m.group(2))) for m in map(REF.match, source_ids)]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        part, k = self.refs[i]
        image = Image.fromarray(np.asarray(self.arrays[part][k], dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0))
        return self.transform(image), i
