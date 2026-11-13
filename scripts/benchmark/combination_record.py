"""Provenance and QA record of one encoder x dataset train+val feature cache (task.md d57a630, C2/C3), read-only.

Features are assembled in extraction-manifest order from either verified chunk caches (``--chunk MANIFEST=CACHE``,
one or more; chunk_store / feature_lookup checks names, coverage, checksums and identical settings) or the earlier
vision-language project's BIOSCAN-5M cache (``--v1-cache``; only the train and val split folders are opened, each
checked one-to-one against the manifest's ids of that split). Recorded and checked:
- identity: manifest sha256, image ids unique, the manifest's ids equal the standardised manifest's available
  train+val ids (five-fold FIN-Benthic2 / AquaMonitor: the selection's ids are a subset of the standardised manifest
  and its specimens equal all specimens there), independent units expected vs present;
- features: shape, float32, all finite, norm range, sha256 of the float32 array in manifest order;
- numerical rank of the covariance and, with ``--weights``, the final-LayerNorm constraint (src/features/null_space);
- encoder, checkpoint, feature, transform and software recorded by the cache.
Exit status 1 if any check fails.

Usage: combination_record.py --encoder E --dataset D --manifest M.parquet --standard STD.parquet
       (--chunk M=CACHE ... | --v1-cache DIR --v1-index INDEX) [--weights W.safetensors --key-prefix P] --out FILE.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file

from src.features.chunk_store import file_sha256
from src.features.feature_lookup import features_by_name
from src.features.null_space import layernorm_constraint, spectrum
from src.features.v1_bioscan_cache import load_v1

FIVE_FOLD = ("finbenthic2", "aquamonitor")
KEEP = ("encoder", "model_id", "model", "feature", "transform", "library", "torch", "tf32", "feature_dim")


def v1_features(cache: Path, index: Path, manifest: pd.DataFrame) -> tuple[np.ndarray, dict]:
    splits = sorted(manifest.split.unique())
    assert set(splits) <= {"train", "val"}, f"only train and val may be read, manifest has {splits}"
    v1 = pd.read_parquet(index, columns=["processid", "split"])
    parts, meta = [], None
    for s in splits:
        ids = v1.processid[v1.split == s].astype(str).tolist()
        mine = manifest.img[manifest.split == s]
        assert len(ids) == len(mine) and set(ids) == set(mine), f"v1 split {s} ids differ from the manifest"
        x, meta = load_v1(cache, s, ids)
        parts.append(pd.DataFrame(x, index=ids).loc[mine])
    return pd.concat(parts).loc[manifest.img].to_numpy(), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--standard", required=True, type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--chunk", nargs="+")
    src.add_argument("--v1-cache", type=Path)
    ap.add_argument("--v1-index", type=Path)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--key-prefix", default="norm")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    m = pd.read_parquet(a.manifest).reset_index(drop=True)
    unit_col = "individual" if a.dataset in FIVE_FOLD else "unit_id"
    std = pd.read_parquet(a.standard, columns=["source_id", "unit_id", "available"]
                          + ([] if a.dataset in FIVE_FOLD else ["split"]))
    checks = {"ids_unique": bool(m.img.is_unique)}
    if a.dataset in FIVE_FOLD:  # train+val over the five folds = every specimen; the selection keeps all of them
        checks["ids_in_standard"] = bool(m.img.isin(std.source_id[std.available]).all())
        expected_units = std.unit_id[std.available].astype(str)
    else:
        tv = std[std.split.isin(["train", "val"]) & std.available]
        checks["ids_equal_standard_trainval"] = bool(len(tv) == len(m) and set(tv.source_id) == set(m.img))
        checks["no_test_ids"] = bool(not m.img.isin(std.source_id[std.split == "test"]).any())
        expected_units = tv.unit_id.astype(str)
    units = m[unit_col].astype(str)
    checks["units_equal_expected"] = bool(set(units) == set(expected_units))

    if a.chunk:
        pairs = [tuple(Path(p) for p in s.split("=", 1)) for s in a.chunk]
        x, meta = features_by_name(pairs, m.img.tolist())
        sources = [{"manifest": str(p), "manifest_sha256": file_sha256(p), "cache": str(c),
                    "chunks": len(list(c.glob("chunk_*.npy")))} for p, c in pairs]
    else:
        x, meta = v1_features(a.v1_cache, a.v1_index, m)
        sources = [{"v1_cache": str(a.v1_cache), "v1_index": str(a.v1_index), "v1_index_sha256": file_sha256(a.v1_index),
                    "splits_read": sorted(m.split.unique())}]
    x = np.ascontiguousarray(x)
    norms = np.linalg.norm(x.astype(np.float64), axis=1)
    checks |= {"rows_equal_manifest": bool(len(x) == len(m)), "float32": x.dtype == np.float32,
               "all_finite": bool(np.isfinite(x).all()), "norms_positive": bool(norms.min() > 0)}
    rank, cov, vec = spectrum(x)
    if a.weights:
        t = load_file(a.weights)
        rank |= {"weights": str(a.weights), "key": f"{a.key_prefix}.weight/bias"}
        rank |= layernorm_constraint(x, t[f"{a.key_prefix}.weight"], t[f"{a.key_prefix}.bias"], cov, vec)
    out = {"encoder": a.encoder, "dataset": a.dataset, "manifest": str(a.manifest),
           "manifest_sha256": file_sha256(a.manifest), "standard": str(a.standard),
           "standard_sha256": file_sha256(a.standard), "rows": int(len(x)), "dim": int(x.shape[1]),
           "units": int(units.nunique()), "units_expected": int(expected_units.nunique()),
           "rows_per_split": m.split.value_counts().to_dict() if "split" in m else None,
           "norm_min": float(norms.min()), "norm_median": float(np.median(norms)), "norm_max": float(norms.max()),
           "features_sha256": hashlib.sha256(x.tobytes()).hexdigest(), "cache_settings": {k: meta.get(k) for k in KEEP if k in meta},
           "sources": sources, "rank": rank, "checks": checks, "pass": all(checks.values())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("encoder", "dataset", "rows", "dim", "units", "units_expected", "pass")}),
          {k: v for k, v in checks.items() if not v} or "all checks pass")
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
