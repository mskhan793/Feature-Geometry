"""combination_record.py on fixture caches: a chunk cache and a v1-format cache holding the same features give the
same record checksum; the v1 path opens only the train and val folders; a test id in the manifest fails the record."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from src.features import chunk_store
from tests.test_class_geometry_pilot import ROOT


def record(args):
    return subprocess.run([sys.executable, "scripts/benchmark/combination_record.py", *args], cwd=ROOT, capture_output=True,
                          text=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})


def test_combination_record_chunk_v1_and_test_isolation(tmp_path):
    rng = np.random.default_rng(0)
    train, val, test = [f"A{i}" for i in range(8)], ["V0", "V1", "V2"], ["T0", "T1"]
    std = pd.DataFrame({"source_id": train + val + test, "unit_id": [f"u{i}" for i in range(13)], "available": True,
                        "split": ["train"] * 8 + ["val"] * 3 + ["test"] * 2})
    std.to_parquet(tmp_path / "std.parquet", index=False)
    m = std[std.split != "test"].sample(frac=1, random_state=1).rename(columns={"source_id": "img"})
    m[["img", "unit_id", "split"]].to_parquet(tmp_path / "m.parquet", index=False)
    feats = {n: rng.normal(size=12).astype(np.float32) for n in train + val + test}
    x = np.stack([feats[n] for n in m.img])
    ident = {"start": 0, "end": len(m), "names_sha1": chunk_store.names_sha1(m.img.tolist()), "encoder": "e",
             "model_id": "m", "transform": "t"}
    chunk_store.write_chunk(tmp_path / "cache", 0, x, ident, {"feature": "f", "tf32": False})
    pd.DataFrame({"processid": train + val + test, "split": ["train"] * 8 + ["val"] * 3 + ["test"] * 2}).to_parquet(
        tmp_path / "idx.parquet", index=False)
    for s, ids in (("train", train), ("val", val)):  # no test folder: reading it would fail
        folder = tmp_path / "v1" / "frozen_fp32" / s
        folder.mkdir(parents=True)
        np.save(folder / "rows_000.npy", np.stack([feats[n] for n in ids]))
        (folder / "meta_000.json").write_text(json.dumps({"start": 0, "end": len(ids), "n_split": len(ids), "model": "m",
                                                          "img_sha1": hashlib.sha1("\n".join(ids).encode()).hexdigest(),
                                                          "transform": "t", "feature_dim": 12}))
    common = ["--encoder", "e", "--dataset", "toy", "--manifest", str(tmp_path / "m.parquet"), "--standard",
              str(tmp_path / "std.parquet")]
    a = record(common + ["--chunk", f"{tmp_path / 'm.parquet'}={tmp_path / 'cache'}", "--out", str(tmp_path / "a.json")])
    b = record(common + ["--v1-cache", str(tmp_path / "v1"), "--v1-index", str(tmp_path / "idx.parquet"),
                         "--out", str(tmp_path / "b.json")])
    assert a.returncode == 0 and b.returncode == 0, a.stderr + b.stderr
    ra, rb = (json.loads((tmp_path / f).read_text()) for f in ("a.json", "b.json"))
    assert ra["pass"] and rb["pass"] and ra["rows"] == 11 and ra["units"] == ra["units_expected"] == 11
    assert ra["features_sha256"] == rb["features_sha256"] == hashlib.sha256(x.tobytes()).hexdigest()
    assert ra["rank"]["numerical_rank_1e-12"] == 10  # 11 rows centred -> rank 10 in 12 dimensions

    leak = pd.concat([m, std[std.split == "test"].rename(columns={"source_id": "img"}).assign(split="val")])
    leak[["img", "unit_id", "split"]].to_parquet(tmp_path / "leak.parquet", index=False)
    x2 = np.stack([feats[n] for n in leak.img])
    ident = {"start": 0, "end": len(leak), "names_sha1": chunk_store.names_sha1(leak.img.tolist()), "encoder": "e",
             "model_id": "m", "transform": "t"}
    chunk_store.write_chunk(tmp_path / "cache2", 0, x2, ident, {"feature": "f", "tf32": False})
    c = record(["--encoder", "e", "--dataset", "toy", "--manifest", str(tmp_path / "leak.parquet"), "--standard",
                str(tmp_path / "std.parquet"), "--chunk", f"{tmp_path / 'leak.parquet'}={tmp_path / 'cache2'}",
                "--out", str(tmp_path / "c.json")])
    rc = json.loads((tmp_path / "c.json").read_text())
    assert c.returncode == 1 and not rc["pass"]
    assert not rc["checks"]["no_test_ids"] and not rc["checks"]["ids_equal_standard_trainval"]
