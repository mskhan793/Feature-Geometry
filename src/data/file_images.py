"""Images read from individual files under a root folder (e.g. FIN-Benthic2 ``IDA/Images/<specimen>/<image>``)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class FileImageDataset(Dataset):
    """Item ``i`` is ``(transform(image), i)`` for the file ``root / paths[i]``. A missing or unreadable file raises:
    nothing is skipped or replaced."""

    def __init__(self, root: str | Path, paths: Sequence[str], transform: Callable):
        self.root, self.paths, self.transform = Path(root), list(paths), transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.root / self.paths[i]) as img:
            image = img.convert("RGB")
        return self.transform(image), i
