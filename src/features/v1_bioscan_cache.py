"""Read-only access to the earlier vision-language project's BIOSCAN-5M frozen caches
(``aqua_v1/bioscan/cache_<encoder>/frozen_fp32/<split>/{meta,rows}_*``). Rows of a split are in v1-index order (rows
of ``bioscan_index.parquet`` with that split); every shard's ``img_sha1`` (sha1 of the newline-joined processids),
``n_split`` and contiguous coverage are checked, as v1's own ``load_cache`` does. Only the requested split's folder is
opened; ``v1_block`` opens only ``train`` and ``val``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ids_sha1(ids) -> str:
    return hashlib.sha1("\n".join(map(str, ids)).encode()).hexdigest()


ENCODERS = {"cache_dinov3": "dinov3_l", "cache_bioclip2": "bioclip2"}  # v1 cache folder -> encoder key
READABLE = ("train", "val")


def is_v1_cache(cache: Path) -> bool:
    return (cache / "frozen_fp32").is_dir()


def v1_block(index: Path, cache: Path) -> tuple[list[str], np.ndarray, dict]:
    """Ids and features of the train and val folders (never test or open-world) of one v1 cache, in v1-index order,
    with the settings in the keys of the chunk caches (``encoder`` from the folder name, ``model_id`` = v1 ``model``)."""
    v1 = pd.read_parquet(index, columns=["processid", "split"])
    ids, parts, meta = [], [], None
    for s in READABLE:
        sid = v1.processid[v1.split == s].astype(str).tolist()
        x, meta = load_v1(cache, s, sid)
        ids += sid
        parts.append(x)
    settings = {"encoder": ENCODERS.get(cache.name, cache.name), "model_id": meta["model"], "feature": meta.get("feature"),
                "transform": meta.get("transform"), "tf32": meta.get("tf32"), "source": "v1 cache " + str(cache)}
    return ids, np.concatenate(parts), settings


def load_v1(cache: Path, split: str, ids: list[str]) -> tuple[np.ndarray, dict]:
    folder = cache / "frozen_fp32" / split
    parts, pos, first = [], 0, None
    for m in sorted(folder.glob("meta_*.json")):
        meta = json.loads(m.read_text())
        first = first or meta
        assert meta["n_split"] == len(ids) and meta["img_sha1"] == ids_sha1(ids), f"{m}: does not match the v1 index"
        assert meta["start"] == pos, f"{m}: gap at {pos}"
        parts.append(np.load(m.with_name(m.name.replace("meta_", "rows_").replace(".json", ".npy"))))
        pos = meta["end"]
    assert pos == len(ids), f"cache covers {pos} of {len(ids)} rows"
    return np.concatenate(parts), first
