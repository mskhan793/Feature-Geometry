"""chunk_store: atomic chunks, resume, and refusal of mismatched or corrupted chunks."""

import numpy as np
import pytest

from src.features import chunk_store

IDENTITY = {"start": 0, "end": 3, "names_sha1": chunk_store.names_sha1(["a", "b", "c"]), "encoder": "e",
            "model_id": "m", "transform": "t"}


def test_bounds_cover_rows_once():
    n, size = 10, 4
    rows = [r for k in range(chunk_store.n_chunks(n, size)) for r in range(*chunk_store.chunk_bounds(n, size, k))]
    assert rows == list(range(n))


def test_write_then_done(tmp_path):
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert not chunk_store.chunk_done(tmp_path, 0, IDENTITY)
    chunk_store.write_chunk(tmp_path, 0, x, IDENTITY, {})
    assert chunk_store.chunk_done(tmp_path, 0, IDENTITY)
    np.testing.assert_array_equal(np.load(chunk_store.paths(tmp_path, 0)[0]), x)
    assert not list(tmp_path.glob("*.tmp"))


def test_other_identity_raises(tmp_path):
    chunk_store.write_chunk(tmp_path, 0, np.ones((3, 2), np.float32), IDENTITY, {})
    with pytest.raises(RuntimeError, match="other inputs"):
        chunk_store.chunk_done(tmp_path, 0, {**IDENTITY, "names_sha1": "different"})


def test_corrupted_npy_raises(tmp_path):
    chunk_store.write_chunk(tmp_path, 0, np.ones((3, 2), np.float32), IDENTITY, {})
    np.save(chunk_store.paths(tmp_path, 0)[0], np.zeros((3, 2), np.float32))
    with pytest.raises(RuntimeError, match="checksum"):
        chunk_store.chunk_done(tmp_path, 0, IDENTITY)


def test_partial_write_is_not_done(tmp_path):
    npy, _ = chunk_store.paths(tmp_path, 0)
    np.save(npy, np.ones((3, 2), np.float32))  # data without its JSON, as after an interrupted write
    assert not chunk_store.chunk_done(tmp_path, 0, IDENTITY)


def test_rejects_wrong_shape_or_non_finite(tmp_path):
    with pytest.raises(ValueError):
        chunk_store.write_chunk(tmp_path, 0, np.ones((2, 2), np.float32), IDENTITY, {})
    bad = np.ones((3, 2), np.float32)
    bad[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        chunk_store.write_chunk(tmp_path, 0, bad, IDENTITY, {})
