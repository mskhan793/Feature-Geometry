"""Pack the images of a standardised benchmark manifest from their publisher archives into one HDF5 with the
AquaMonitor layout (``images``: original file bytes; ``metadata/img``: the manifest ``source_id``), in manifest order,
so LUMI reads one file instead of many small ones (as for FIN-Benthic2). Archives are streamed once (.tar.gz) or read
by member (.zip; ``image_ref`` = "<zip name>:<member>"). Every stored image is checked against the bytes read and,
with --content, against the sha1 recorded by scripts/lumi/hash_archive_members.py; a missing image is an error.
Also writes the extraction manifest (img = source_id, hdf5_row, split, label, unit_id) for extract_features.py --hdf5.

``--splits`` keeps only the manifest rows of those splits (e.g. train val: official test images are never packed).

Usage: pack_archive_hdf5.py --manifest DATASET.parquet --archives A.tar.gz|DIR_WITH_ZIPS ... [--content content.tsv]
                            [--splits train val] --out FILE.h5 --out-manifest extract.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.features.chunk_store import file_sha256


def members(archives: list[Path], wanted: set[str]):
    """Yield (image_ref, bytes) for the wanted references."""
    for a in archives:
        if a.is_dir():
            for z in sorted(a.glob("*.zip")):
                with zipfile.ZipFile(z) as zf:
                    for name in zf.namelist():
                        if f"{z.name}:{name}" in wanted:
                            yield f"{z.name}:{name}", zf.read(name)
        else:
            with tarfile.open(a, "r|gz") as tar:
                for m in tar:
                    if m.isfile() and m.name in wanted:
                        yield m.name, tar.extractfile(m).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--archives", required=True, nargs="+", type=Path)
    ap.add_argument("--content", type=Path)
    ap.add_argument("--splits", nargs="*")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--out-manifest", required=True, type=Path)
    a = ap.parse_args()
    m = pd.read_parquet(a.manifest)
    if a.splits:
        m = m[m.split.isin(a.splits)].reset_index(drop=True)
    assert m.source_id.is_unique and m.image_ref.is_unique
    row = {ref: i for i, ref in enumerate(m.image_ref)}
    sha = pd.read_csv(a.content, sep="\t", header=None, names=["ref", "sha1"]).set_index("ref").sha1 if a.content else None
    tmp = a.out.with_suffix(".h5.tmp")
    filled = np.zeros(len(m), dtype=bool)
    with h5py.File(tmp, "w") as h:
        images = h.create_dataset("images", (len(m),), dtype=h5py.vlen_dtype(np.uint8))
        for ref, data in members(a.archives, set(row)):
            if sha is not None and hashlib.sha1(data).hexdigest() != sha[ref]:
                raise RuntimeError(f"{ref}: bytes differ from the content table")
            images[row[ref]] = np.frombuffer(data, dtype=np.uint8)
            filled[row[ref]] = True
        if not filled.all():
            raise RuntimeError(f"{(~filled).sum()} manifest images not found in the archives")
        h.create_dataset("metadata/img", data=m.source_id.to_numpy(dtype=object), dtype=h5py.string_dtype())
    with h5py.File(tmp, "r") as h:  # read back: every image present, names in manifest order
        assert [v.decode() for v in h["metadata/img"][:]] == m.source_id.tolist()
        for ref, data in members(a.archives, set(row)):
            if h["images"][row[ref]].tobytes() != data:
                raise RuntimeError(f"{ref}: stored bytes differ")
    tmp.rename(a.out)
    out = m.rename(columns={"source_id": "img"}).assign(hdf5_row=np.arange(len(m)))
    split_cols = [c for c in out.columns if c == "split" or c.startswith("fold")]
    out[["img", "hdf5_row", "unit_id", "label", *split_cols]].to_parquet(a.out_manifest, index=False)
    report = {"manifest": str(a.manifest), "manifest_sha256": file_sha256(a.manifest), "splits": a.splits, "images": len(m),
              "images_per_split": m.split.value_counts().to_dict(),
              "archives": {str(p): (file_sha256(p) if p.is_file() else "zip folder") for p in a.archives},
              "content_checked": a.content is not None, "hdf5_sha256": file_sha256(a.out), "hdf5_bytes": a.out.stat().st_size,
              "extraction_manifest": str(a.out_manifest)}
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
