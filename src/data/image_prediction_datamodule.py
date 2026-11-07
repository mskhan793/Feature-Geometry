"""LightningDataModule that streams one fixed, ordered set of images for ``Trainer.predict``."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


class ImagePredictionDataModule(pl.LightningDataModule):
    """Predict-only: images in dataset order (no shuffling, no augmentation); there is no train/val/test here."""

    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int):
        super().__init__()
        self.dataset, self.batch_size, self.num_workers = dataset, batch_size, num_workers

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=False, pin_memory=True)
