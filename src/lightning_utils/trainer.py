"""The ``pl.Trainer`` with the defaults every experiment shares."""

from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# Source repository defaults: deterministic kernels, at least one epoch, gradient-norm clipping at 1.0, and no
# logger (metrics go to EpochMetricsWriter). Any of them can be overridden through ``overrides``.
DEFAULTS = {
    "min_epochs": 1,
    "gradient_clip_val": 1.0,
    "deterministic": True,
    "logger": False,
}


def build_trainer(
    callbacks: list[Callback],
    max_epochs: int,
    accelerator: str = "auto",
    devices: int | str = 1,
    precision: str = "32-true",
    **overrides,
) -> pl.Trainer:
    """Build a Trainer; ``overrides`` are passed to ``pl.Trainer`` and replace the defaults above."""
    settings = {**DEFAULTS, **overrides}
    return pl.Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        **settings,
    )
