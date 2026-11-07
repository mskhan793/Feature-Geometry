"""Build the AquaMonitor Classification image manifest and its read-only integrity audit.

Writes MANIFEST (parquet, no index; see src/data/aquamonitor_manifest.py) and MANIFEST with suffix ``.audit.json``:
per fold and split the images, specimens and classes; specimens in more than one split of a fold; folds in which
each specimen is test; class count; and the manifest sha256.

Usage: build_aquamonitor_manifest.py --parquet CLASSIF_PARQUET --hdf5 H5 --out MANIFEST.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.aquamonitor_manifest import FOLDS, SPLITS, build_manifest
from src.features.chunk_store import file_sha256


def audit(m) -> dict:
    report = {"images": len(m), "specimens": int(m["individual"].nunique()), "classes": int(m["taxon_group"].nunique()),
              "hdf5_rows_unique": bool(m["hdf5_row"].is_unique), "folds": {}}
    for fold in FOLDS:
        g = m.groupby(fold)
        report["folds"][fold] = {
            "images": g.size().reindex(SPLITS).astype(int).to_dict(),
            "specimens": g["individual"].nunique().reindex(SPLITS).astype(int).to_dict(),
            "classes": g["taxon_group"].nunique().reindex(SPLITS).astype(int).to_dict(),
            "specimens_in_more_than_one_split": int((m.groupby("individual")[fold].nunique() > 1).sum()),
        }
    per_specimen = m.drop_duplicates("individual")[FOLDS].eq("test").sum(axis=1)
    report["specimens_by_number_of_test_folds"] = {int(k): int(v) for k, v in per_specimen.value_counts().sort_index().items()}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    m = build_manifest(a.parquet, a.hdf5)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(a.out, index=False)
    report = {"source_parquet": a.parquet, "source_parquet_sha256": file_sha256(Path(a.parquet)), "hdf5": a.hdf5,
              "manifest": str(a.out), "manifest_sha256": file_sha256(a.out), **audit(m)}
    a.out.with_suffix(".audit.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
