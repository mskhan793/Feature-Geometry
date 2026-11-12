"""Build the standardised manifest of one benchmark dataset (src/data/benchmark_manifests.py), write it as parquet
and its support summary as JSON (sha256 of every source file and of the manifest; per split column: images, units,
classes, eligibility >= 15 train / >= 5 val units, long-tail percentiles, units in several splits).

Usage: build_manifests.py DATASET --out-manifest FILE --out-summary FILE --source NAME=PATH ... [--content FILE]
  finbenthic2: csv=;  aquamonitor: manifest=;  bioscan5m: index=;  df24: train= test= assignment=;  cifar100: folder=;
  food101: meta=;  oxford_pets: annotations=;  fgvc_aircraft: data=;  --content: member<TAB>sha1 table of the image
  archive members (scripts/lumi/hash_archive_members.py) for the last three
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data import benchmark_manifests as bm


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_dir():
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(f.relative_to(path).as_posix().encode())
            h.update(hashlib.sha256(f.read_bytes()).digest())
        return h.hexdigest()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--source", nargs="+", required=True)
    ap.add_argument("--content", type=Path)
    ap.add_argument("--out-manifest", required=True, type=Path)
    ap.add_argument("--out-summary", required=True, type=Path)
    a = ap.parse_args()
    src = {k: Path(v) for k, v in (s.split("=", 1) for s in a.source)}
    build = getattr(bm, a.dataset)
    if a.content:
        content = pd.read_csv(a.content, sep="\t", header=None, names=["member", "sha1"]).set_index("member").sha1
        m = build(*src.values(), content)
    else:
        m = build(*src.values())
    a.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(a.out_manifest, index=False)
    split_cols = [c for c in m.columns if c == "split" or c in bm.FOLDS]
    summary = {"dataset": a.dataset, "version": m.version.iloc[0], "rows": len(m), "units": int(m.unit_id.nunique()),
               "classes": int(m.label.nunique()), "val_origin": sorted(m.val_origin.unique()),
               "derived_seed": sorted(int(x) for x in m.derived_seed.unique()),
               "coarse_labels": int(m.coarse_label.replace("", pd.NA).dropna().nunique()),
               "sources": {k: {"path": str(v), "sha256": sha256(v)} for k, v in src.items()},
               "content_table": {"path": str(a.content), "sha256": sha256(a.content)} if a.content else None,
               "manifest": str(a.out_manifest),
               "manifest_sha256": sha256(a.out_manifest), "support": {c: bm.support(m, c) for c in split_cols}}
    a.out_summary.parent.mkdir(parents=True, exist_ok=True)
    a.out_summary.write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "support"}, indent=1, default=str))
    print(json.dumps(summary["support"][split_cols[0]], indent=1))


if __name__ == "__main__":
    main()
