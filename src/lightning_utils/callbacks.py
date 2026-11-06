"""Callbacks shared by every trained LightningModule."""

from __future__ import annotations

import json
from pathlib import Path

from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint


class EpochMetricsWriter(Callback):
    """Rank 0 appends one JSON line per epoch: epoch, step, learning rate per optimizer group and every scalar in
    ``trainer.callback_metrics`` (train and validation values logged with ``self.log``). Works with ``logger=False``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        row = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "lr": [g["lr"] for opt in trainer.optimizers for g in opt.param_groups],
        }
        for key, value in trainer.callback_metrics.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def standard_callbacks(
    run_dir: str | Path,
    monitor: str,
    mode: str = "max",
    save_top_k: int = 1,
    patience: int | None = None,
) -> list[Callback]:
    """Best checkpoint(s) by a validation metric, ``last.ckpt``, per-epoch metrics, and early stopping if ``patience``
    is given. Select models on validation metrics only."""
    run_dir = Path(run_dir)
    callbacks: list[Callback] = [
        ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            dirpath=run_dir,
            filename=f"epoch{{epoch:02d}}_{monitor}{{{monitor}:.4f}}_best",
            save_top_k=save_top_k,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(dirpath=run_dir, save_top_k=0, save_last=True),
        EpochMetricsWriter(run_dir / "epoch_metrics.jsonl"),
    ]
    if patience is not None:
        callbacks.append(EarlyStopping(monitor=monitor, mode=mode, patience=patience))
    return callbacks
