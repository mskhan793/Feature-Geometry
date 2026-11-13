"""Equivalence audit of the earlier vision-language project's BIOSCAN-5M frozen caches (v1
``aqua_v1/bioscan/cache_<encoder>/frozen_fp32/<split>/{meta,rows}_*``) against this repository's extractor (task.md
d57a630, C0). Read-only.

v1 rows of a split are in v1-index order (rows of ``bioscan_index.parquet`` with that split); every shard's
``img_sha1`` (sha1 of the newline-joined processids) and coverage are checked, as v1's own ``load_cache`` does. The
sample's processids are looked up by position and compared with the new cache (``features_by_name``).
PREDECLARED criterion (before any result): reuse the v1 cache only if, for every sample row, the relative L2
difference ||a - b|| / ||b|| <= 1e-4 and the cosine >= 0.999999, the checkpoint id and transform recorded in the v1
shards equal the new cache's, and the feature dimension agrees. Otherwise re-extract.

Usage: compare_v1_bioscan_cache.py --v1-index INDEX --v1-cache DIR --split train --sample S.parquet
       --new MANIFEST=CACHE --out FILE.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_lookup import features_by_name
from src.features.v1_bioscan_cache import load_v1

REL_TOL, COS_TOL = 1e-4, 0.999999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-index", required=True, type=Path)
    ap.add_argument("--v1-cache", required=True, type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample", required=True, type=Path)
    ap.add_argument("--new", required=True, nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    ids = pd.read_parquet(a.v1_index, columns=["processid", "split"]).query("split == @a.split").processid.tolist()
    v1, meta = load_v1(a.v1_cache, a.split, ids)
    sample = pd.read_parquet(a.sample).img.tolist()
    pos = pd.Series(range(len(ids)), index=ids).loc[sample].to_numpy()
    old = v1[pos].astype(np.float64)
    new, new_meta = features_by_name([tuple(Path(p) for p in s.split("=", 1)) for s in a.new], sample)
    new = new.astype(np.float64)
    rel = np.linalg.norm(old - new, axis=1) / np.linalg.norm(new, axis=1)
    cos = (old * new).sum(1) / (np.linalg.norm(old, axis=1) * np.linalg.norm(new, axis=1))
    same_settings = {"model_id": meta["model"] == new_meta["model_id"].removeprefix("hf-hub:") or meta["model"] == new_meta["model_id"],
                     "transform": meta["transform"] == new_meta["transform"], "dim": old.shape[1] == new.shape[1]}
    out = {"rows": len(sample), "split": a.split, "v1_cache": str(a.v1_cache), "v1_settings": {k: meta.get(k) for k in ("model", "transform", "library", "torch", "feature_dim")},
           "new_settings": {k: new_meta.get(k) for k in ("model_id", "transform", "library", "torch")},
           "same_settings": same_settings, "max_rel_l2": float(rel.max()), "median_rel_l2": float(np.median(rel)),
           "min_cosine": float(cos.min()), "bitwise_equal_rows": int((old == new).all(1).sum()),
           "criterion": {"rel_l2_max": REL_TOL, "cosine_min": COS_TOL},
           "equivalent": bool(rel.max() <= REL_TOL and cos.min() >= COS_TOL and all(same_settings.values()))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
