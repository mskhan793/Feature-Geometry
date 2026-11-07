"""Encode manifest rows chunk by chunk with ``Trainer.predict`` and store them with ``chunk_store``."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from src.data.hdf5_images import Hdf5ImageDataset
from src.data.image_prediction_datamodule import ImagePredictionDataModule
from src.features import chunk_store
from src.lightning_modules.frozen_encoder import FrozenEncoderModule


def encode(trainer: pl.Trainer, module: FrozenEncoderModule, dataset: Hdf5ImageDataset, batch_size: int,
           num_workers: int) -> np.ndarray:
    """Features of every dataset item, in dataset order (checked)."""
    batches = trainer.predict(module, datamodule=ImagePredictionDataModule(dataset, batch_size, num_workers),
                              return_predictions=True)
    features = torch.cat([f for f, _ in batches]).numpy()
    index = torch.cat([i for _, i in batches]).numpy()
    if not np.array_equal(index, np.arange(len(dataset))):
        raise RuntimeError("predictions are not in dataset order")
    return features


def extract_chunks(trainer: pl.Trainer, encoder: torch.nn.Module, transform: Callable, info: dict, manifest: pd.DataFrame,
                   hdf5_path: str, cache_dir: Path, chunk_size: int, chunks: Iterable[int], batch_size: int,
                   num_workers: int, log: Callable[[str], None] = print) -> list[dict]:
    module = FrozenEncoderModule(encoder)
    written = []
    for k in chunks:
        start, end = chunk_store.chunk_bounds(len(manifest), chunk_size, k)
        rows = manifest.iloc[start:end]
        identity = {"start": start, "end": end, "names_sha1": chunk_store.names_sha1(rows["img"].tolist()),
                    "encoder": info["encoder"], "model_id": info["model_id"], "transform": info["transform"]}
        if chunk_store.chunk_done(cache_dir, k, identity):
            log(f"chunk {k}: rows {start}-{end} already done")
            continue
        t0 = time.time()
        dataset = Hdf5ImageDataset(hdf5_path, rows["img"].tolist(), rows["hdf5_row"].tolist(), transform)
        features = encode(trainer, module, dataset, batch_size, num_workers).astype(np.float32, copy=False)
        seconds = time.time() - t0
        meta = chunk_store.write_chunk(cache_dir, k, features, identity,
                                       {**info, "seconds": round(seconds, 1), "images_per_second": round(len(rows) / seconds, 1)})
        log(f"chunk {k}: rows {start}-{end}, dim {meta['dim']}, {seconds:.0f} s, {meta['images_per_second']} img/s")
        written.append(meta)
    return written
