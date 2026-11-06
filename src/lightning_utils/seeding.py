"""Global seeding for Lightning runs."""

from __future__ import annotations

import pytorch_lightning as pl


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy and torch, and give each dataloader worker its own derived seed.

    Call before building the DataModule and the model. ``workers=True`` is the source repository's setting.
    """
    return pl.seed_everything(seed, workers=True)
