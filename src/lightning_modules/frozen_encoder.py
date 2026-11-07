"""LightningModule that runs a frozen encoder for ``Trainer.predict`` (no training steps, no optimizer)."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn


class FrozenEncoderModule(pl.LightningModule):
    """``predict_step`` returns ``(features as float32 on CPU, dataset indices)`` for one batch."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)

    def predict_step(self, batch, batch_idx):
        images, index = batch
        features = self.encoder(images)
        return features.float().cpu(), index.cpu()
