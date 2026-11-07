"""Chunked, resumable feature cache: ``<dir>/chunk_NNNNN.npy`` (float32, manifest rows ``start..end``) and
``<dir>/chunk_NNNNN.json`` (identity, shape, sha256, norms).

A chunk is written to temporary files and renamed into place, the JSON last, so a chunk counts as done only when its
JSON exists, its identity (rows, image names, encoder) equals the expected one and the ``.npy`` checksum matches. A
chunk whose JSON has a different identity is an error, never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

IDENTITY = ("start", "end", "names_sha1", "encoder", "model_id", "transform")


def chunk_bounds(n_rows: int, chunk_size: int, k: int) -> tuple[int, int]:
    start = k * chunk_size
    if not 0 <= start < n_rows:
        raise IndexError(f"chunk {k} is outside {n_rows} rows")
    return start, min(start + chunk_size, n_rows)


def n_chunks(n_rows: int, chunk_size: int) -> int:
    return -(-n_rows // chunk_size)


def names_sha1(names: Sequence[str]) -> str:
    return hashlib.sha1("\n".join(names).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def paths(cache_dir: Path, k: int) -> tuple[Path, Path]:
    return cache_dir / f"chunk_{k:05d}.npy", cache_dir / f"chunk_{k:05d}.json"


def chunk_done(cache_dir: Path, k: int, identity: dict) -> bool:
    npy, meta_path = paths(cache_dir, k)
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    differs = {key: (meta.get(key), identity[key]) for key in IDENTITY if meta.get(key) != identity[key]}
    if differs:
        raise RuntimeError(f"{meta_path} was written for other inputs: {differs}")
    if not npy.exists() or file_sha256(npy) != meta["sha256"]:
        raise RuntimeError(f"{npy} is missing or does not match the checksum in {meta_path.name}")
    return True


def load_cache(cache_dir: Path, names: Sequence[str]) -> tuple[np.ndarray, dict]:
    """All chunks of a complete cache, concatenated in row order, after checking that the chunks cover exactly
    ``names`` (rows and image names) and match their checksums. Returns the features and the first chunk's metadata."""
    parts, first, k, end = [], None, 0, 0
    while paths(cache_dir, k)[1].exists():
        meta = json.loads(paths(cache_dir, k)[1].read_text())
        if meta["start"] != end or meta["names_sha1"] != names_sha1(names[meta["start"]:meta["end"]]):
            raise RuntimeError(f"chunk {k} does not hold rows {end}.. of the given names")
        chunk_done(cache_dir, k, {key: meta[key] for key in IDENTITY})
        parts.append(np.load(paths(cache_dir, k)[0]))
        first, k, end = first or meta, k + 1, meta["end"]
    if end != len(names):
        raise RuntimeError(f"cache covers {end} of {len(names)} rows")
    return np.concatenate(parts), first


def write_chunk(cache_dir: Path, k: int, features: np.ndarray, identity: dict, extra: dict) -> dict:
    if features.dtype != np.float32 or features.ndim != 2 or len(features) != identity["end"] - identity["start"]:
        raise ValueError(f"chunk {k}: expected float32 ({identity['end'] - identity['start']}, d), got {features.dtype} {features.shape}")
    finite = bool(np.isfinite(features).all())
    if not finite:
        raise ValueError(f"chunk {k}: non-finite features")
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy, meta_path = paths(cache_dir, k)
    tmp_npy, tmp_meta = npy.with_suffix(".npy.tmp"), meta_path.with_suffix(".json.tmp")
    with open(tmp_npy, "wb") as f:
        np.save(f, features)
        f.flush()
        os.fsync(f.fileno())
    norms = np.linalg.norm(features, axis=1)
    meta = {**identity, **extra, "rows": int(len(features)), "dim": int(features.shape[1]), "dtype": "float32",
            "finite": finite, "norm_min": float(norms.min()), "norm_mean": float(norms.mean()),
            "norm_max": float(norms.max()), "sha256": file_sha256(tmp_npy)}
    tmp_meta.write_text(json.dumps(meta, indent=1))
    os.replace(tmp_npy, npy)
    os.replace(tmp_meta, meta_path)
    return meta
