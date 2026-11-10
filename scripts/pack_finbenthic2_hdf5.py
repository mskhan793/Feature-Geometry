"""Pack every FIN-Benthic2 image file into one HDF5 with the AquaMonitor layout (``images``: the original PNG bytes,
``metadata/img``: image names), in the order of the frozen five-split CSV, and verify the copy byte for byte. Used
to move the images to LUMI as one file instead of 460,004 small files.

With ``--cohort``, also writes ``<cohort stem>_hdf5.parquet`` next to it: the cohort rows in cohort order with their
``hdf5_row`` (looked up by image name, src.data.aquamonitor_manifest.hdf5_positions), for ``extract_features.py
--hdf5``. A JSON report (image count, sha256 of the CSV and of the HDF5) is written next to the HDF5.

Usage: pack_finbenthic2_hdf5.py --csv 01_finbenthic2_processed_5splits_taxon.csv --images IDA/Images --out FILE.h5
       [--cohort cohort.parquet]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.data.aquamonitor_manifest import hdf5_positions
from src.features.chunk_store import file_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cohort", type=Path)
    a = ap.parse_args()
    df = pd.read_csv(a.csv, usecols=["individual", "img"])
    if not df.img.is_unique:
        raise ValueError("image names are not unique")
    files = [a.images / p for p in df.individual + "/" + df.img]
    if not a.out.exists():
        tmp = a.out.with_suffix(".h5.tmp")
        with h5py.File(tmp, "w") as h:
            images = h.create_dataset("images", (len(df),), dtype=h5py.vlen_dtype(np.uint8))
            for i, f in enumerate(files):
                images[i] = np.frombuffer(f.read_bytes(), dtype=np.uint8)
            h.create_dataset("metadata/img", data=df.img.to_numpy(dtype=object), dtype=h5py.string_dtype())
            h.create_dataset("metadata/individual", data=df.individual.to_numpy(dtype=object), dtype=h5py.string_dtype())
        with h5py.File(tmp, "r") as h:
            for i, f in enumerate(files):
                if h["metadata/img"][i].decode() != df.img[i] or h["images"][i].tobytes() != f.read_bytes():
                    raise RuntimeError(f"copy of {f} differs from the file")
        tmp.rename(a.out)
        report = {"csv": str(a.csv), "csv_sha256": file_sha256(a.csv), "images": len(df),
                  "verified": "every image byte-identical to its file, names in CSV order",
                  "hdf5_sha256": file_sha256(a.out), "hdf5_bytes": a.out.stat().st_size}
        a.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
    if a.cohort:
        cohort = pd.read_parquet(a.cohort)
        cohort["hdf5_row"] = hdf5_positions(str(a.out), cohort.img)
        out = a.cohort.with_name(a.cohort.stem + "_hdf5.parquet")
        cohort.to_parquet(out, index=False)
        print(f"{out}: {len(cohort)} rows with hdf5_row")


if __name__ == "__main__":
    main()
