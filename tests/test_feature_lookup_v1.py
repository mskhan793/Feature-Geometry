"""features_by_name reads a v1-format BIOSCAN-5M cache (train and val folders only) and returns the same features as a
chunk cache holding them; the encoder key comes from the v1 folder name."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import chunk_store
from src.features.feature_lookup import features_by_name


def test_v1_cache_lookup_equals_chunk_cache(tmp_path: Path):
    rng = np.random.default_rng(0)
    train, val, test = [f"A{i}" for i in range(6)], ["V0", "V1"], ["T0"]
    feats = {n: rng.normal(size=5).astype(np.float32) for n in train + val + test}
    pd.DataFrame({"processid": train + val + test, "split": ["train"] * 6 + ["val"] * 2 + ["test"]}).to_parquet(
        tmp_path / "idx.parquet", index=False)
    cache = tmp_path / "cache_dinov3"
    for s, ids in (("train", train), ("val", val)):  # no test folder: opening it would fail
        folder = cache / "frozen_fp32" / s
        folder.mkdir(parents=True)
        np.save(folder / "rows_000.npy", np.stack([feats[n] for n in ids]))
        (folder / "meta_000.json").write_text(json.dumps({
            "start": 0, "end": len(ids), "n_split": len(ids), "model": "m", "transform": "t",
            "img_sha1": hashlib.sha1("\n".join(ids).encode()).hexdigest()}))
    names = ["V1", "A3", "A0", "V0"]
    x, meta = features_by_name([(tmp_path / "idx.parquet", cache)], names)
    assert np.array_equal(x, np.stack([feats[n] for n in names]))
    assert meta["encoder"] == "dinov3_l" and meta["model_id"] == "m"

    pd.DataFrame({"img": train + val}).to_parquet(tmp_path / "m.parquet", index=False)
    ident = {"start": 0, "end": 8, "names_sha1": chunk_store.names_sha1(train + val), "encoder": "dinov3_l",
             "model_id": "m", "transform": "t"}
    chunk_store.write_chunk(tmp_path / "chunks", 0, np.stack([feats[n] for n in train + val]), ident, {"feature": "f"})
    y, _ = features_by_name([(tmp_path / "m.parquet", tmp_path / "chunks")], names)
    assert np.array_equal(x, y)
