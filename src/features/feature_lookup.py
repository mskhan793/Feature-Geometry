"""Features for a list of image names drawn from several verified chunk caches (each paired with the manifest it was
extracted from). Every cache is loaded with ``chunk_store.load_cache`` (coverage, names and checksums checked); all
caches must share encoder, checkpoint, feature, preprocessing and the TF32 setting; every requested name must be found
exactly once. A pair whose cache is one of the earlier vision-language project's BIOSCAN-5M caches
(``frozen_fp32`` folder; the "manifest" is then its ``bioscan_index.parquet``) is read through
src.features.v1_bioscan_cache.v1_block: train and val folders only, shard ids and coverage checked."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.chunk_store import load_cache
from src.features.v1_bioscan_cache import is_v1_cache, v1_block

SAME = ("encoder", "model_id", "feature", "transform", "tf32")


def features_by_name(pairs: Sequence[tuple[Path, Path]], names: Sequence[str]) -> tuple[np.ndarray, dict]:
    blocks, index, first = [], {}, None
    for manifest, cache in pairs:
        if is_v1_cache(cache):
            cached, x, meta = v1_block(manifest, cache)
        else:
            cached = pd.read_parquet(manifest, columns=["img"]).img.tolist()
            x, meta = load_cache(cache, cached)
        if first is None:
            first = meta
        differs = {k: (first.get(k), meta.get(k)) for k in SAME if first.get(k) != meta.get(k)}
        if differs:
            raise RuntimeError(f"{cache} was made with other settings: {differs}")
        offset = sum(len(b) for b in blocks)
        for i, n in enumerate(cached):
            if n in index:
                raise RuntimeError(f"{n} is in more than one cache")
            index[n] = offset + i
        blocks.append(x)
    missing = [n for n in names if n not in index]
    if missing:
        raise RuntimeError(f"{len(missing)} images have no features, e.g. {missing[0]}")
    return np.concatenate(blocks)[[index[n] for n in names]], first
