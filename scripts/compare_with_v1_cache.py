"""Equivalence gate: the new extraction path must reproduce an existing Lightning_Aqua_Vision_Language_v1 feature cache.

The v1 cache ``<v1-cache>/frozen_fp32/<split>/rows_000.npy`` holds features of the v1 parquet's ``fold0 == split``
rows in parquet order; its ``meta_000.json`` hash of those image names is checked first. The first ``--n`` of these
images are encoded here (HDF5 position looked up by name) and compared. Passes when max |diff| <= 1e-4 x max |v1|,
the rule of v1 scripts/run_bioscan.sh.

Usage: compare_with_v1_cache.py --encoder NAME --v1-parquet P --v1-cache DIR --split val --hdf5 H5 --out result.json [--n 512]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.aquamonitor_manifest import hdf5_positions
from src.data.hdf5_images import Hdf5ImageDataset
from src.encoders.frozen_encoders import load_encoder
from src.features.extract import encode
from src.lightning_modules.frozen_encoder import FrozenEncoderModule
from src.lightning_utils.trainer import build_trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--v1-parquet", required=True)
    ap.add_argument("--v1-cache", required=True, type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    folder = a.v1_cache / "frozen_fp32" / a.split
    meta = json.loads((folder / "meta_000.json").read_text())
    df = pd.read_parquet(a.v1_parquet, columns=["img", "fold0"])
    names = df.loc[df["fold0"] == a.split, "img"].tolist()[:meta["n_split"]]
    if hashlib.sha1("\n".join(names).encode()).hexdigest() != meta["img_sha1"]:
        raise RuntimeError("v1 parquet rows do not match the v1 cache image order")
    if meta["start"] != 0 or meta["end"] < a.n:
        raise RuntimeError("compare only rows inside the v1 reference shard 0")
    names = names[:a.n]
    ref = np.asarray(np.load(folder / "rows_000.npy", mmap_mode="r")[:a.n])

    encoder, transform, info = load_encoder(a.encoder)
    trainer = build_trainer([], max_epochs=1, accelerator="auto", devices=1, deterministic="warn",
                            enable_progress_bar=False, enable_model_summary=False)
    dataset = Hdf5ImageDataset(a.hdf5, names, hdf5_positions(a.hdf5, pd.Series(names)), transform)
    ours = encode(trainer, FrozenEncoderModule(encoder), dataset, batch_size=128, num_workers=4)
    diff, scale = float(np.abs(ours - ref).max()), float(np.abs(ref).max())
    result = {"encoder": a.encoder, "v1_cache": str(a.v1_cache), "v1_model": meta.get("model"), "v1_transform": meta.get("transform"),
              "new_transform": info["transform"], "rows": a.n, "dim": int(ours.shape[1]), "max_abs_diff": diff,
              "max_abs_v1": scale, "pass": diff <= 1e-4 * scale}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
