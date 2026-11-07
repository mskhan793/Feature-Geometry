"""Build the capped camera-pilot cohort (official fold0 train and val only; test is never read into the cohort).

Classes: those meeting the support rule in every camera (``inventory.json`` from camera_inventory.py). Writes
``cohort.parquet`` (manifest rows incl. ``hdf5_row`` and all fold columns) and ``cohort_summary.json`` (rule, seed,
input checksums, per class/split/camera specimen and image counts).

Usage: build_camera_cohort.py --manifest MANIFEST --inventory inventory.json --out DIR
       [--seed 20261107 --train-specimens 30 --val-specimens 15 --frames 3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.camera_cohort import select_cohort
from src.features.chunk_store import file_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--train-specimens", type=int, default=30)
    ap.add_argument("--val-specimens", type=int, default=15)
    ap.add_argument("--frames", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(a.inventory.read_text())
    classes = inventory["classes_eligible_in_every_camera"]
    manifest = pd.read_parquet(a.manifest)
    parts = [select_cohort(manifest, "fold0", split, classes, n, a.frames, a.seed)
             for split, n in (("train", a.train_specimens), ("val", a.val_specimens))]
    cohort = pd.concat(parts, ignore_index=True)
    if set(parts[0].individual) & set(parts[1].individual):
        raise RuntimeError("a specimen is in both train and val")
    if not cohort.img.is_unique:
        raise RuntimeError("duplicate image in cohort")
    cohort.to_parquet(a.out / "cohort.parquet", index=False)
    counts = cohort.groupby(["taxon_group", "fold0", "camera"]).agg(images=("img", "size"), specimens=("individual", "nunique"))
    summary = {
        "manifest": str(a.manifest), "manifest_sha256": file_sha256(a.manifest), "inventory_sha256": file_sha256(a.inventory),
        "rule": "fold0 train/val only; specimens imaged by every camera; per class first N specimens by sha1(seed:individual); "
                "per specimen and camera first F images by sha1(seed:img)",
        "seed": a.seed, "train_specimens_cap": a.train_specimens, "val_specimens_cap": a.val_specimens, "frames_cap": a.frames,
        "classes": classes, "excluded_classes": sorted(set(manifest.taxon_group) - set(classes)),
        "images": len(cohort), "specimens": int(cohort.individual.nunique()),
        "by_split": {s: {"images": int((cohort.fold0 == s).sum()), "specimens": int(cohort[cohort.fold0 == s].individual.nunique())}
                     for s in ("train", "val")},
        "classes_below_cap": {s: sorted(k for k, n in cohort[cohort.fold0 == s].groupby("taxon_group").individual.nunique().items() if n < cap)
                              for s, cap in (("train", a.train_specimens), ("val", a.val_specimens))},
        "cohort_sha256": file_sha256(a.out / "cohort.parquet"),
        "counts": json.loads(counts.reset_index().to_json(orient="records")),
    }
    (a.out / "cohort_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}, indent=1))


if __name__ == "__main__":
    main()
