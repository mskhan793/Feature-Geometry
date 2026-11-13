"""Read-only access to the earlier vision-language project's BIOSCAN-5M frozen caches
(``aqua_v1/bioscan/cache_<encoder>/frozen_fp32/<split>/{meta,rows}_*``). Rows of a split are in v1-index order (rows
of ``bioscan_index.parquet`` with that split); every shard's ``img_sha1`` (sha1 of the newline-joined processids),
``n_split`` and contiguous coverage are checked, as v1's own ``load_cache`` does. Only the requested split's folder is
opened."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def ids_sha1(ids) -> str:
    return hashlib.sha1("\n".join(map(str, ids)).encode()).hexdigest()


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
