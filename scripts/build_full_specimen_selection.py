"""Full-specimen image selection for the five-fold geometry study (task.md 92a605a): one global, fold-independent
selection per dataset, so every selected image is encoded once and reused in every fold.

Rule: every eligible specimen of the dataset (all five folds; each specimen is train, val or test in each fold), up to
``--frames`` images per specimen and camera, dealt over imaging runs and evenly spaced in capture order within a run
(src.data.expanded_cohort.select_frames, the rule of the earlier pilots; it depends on a specimen's own images only,
not on folds, seeds or outcomes). Single-camera specimens are kept. Nothing is duplicated.

Datasets:
- ``finbenthic2``: the author's frozen five-split CSV (sha256 checked); ``hdf5_row`` from the packed FIN-Benthic2 HDF5
  (``--hdf5``, looked up by name); ``width``/``height`` read from the image file headers (``--ida``), for the
  crop-size check.
- ``aquamonitor``: the audited Classification manifest (sha256 checked; ``hdf5_row`` already in it).

Checks per fold: no specimen in two partitions, each specimen in test in exactly one fold, class and partition counts.

``--reuse PARQUET`` (repeatable): manifests of verified caches whose features are reused; selected images found there
(by name) are not extracted again. ``--overlap PARQUET`` (repeatable): only counted (e.g. caches from other hardware).

Writes to ``--out``: ``selection.parquet`` (selected image rows, all folds), ``new_images.parquet`` (rows to extract,
in ``hdf5_row`` order), ``per_fold_class_counts.csv`` (independent specimens and selected images per class, fold and
partition) and ``audit.json`` (checks, counts, frames, reuse, feature storage).

Usage: build_full_specimen_selection.py --dataset finbenthic2 --csv CSV --hdf5 H5 --ida IDA --out DIR [--reuse P]
       build_full_specimen_selection.py --dataset aquamonitor --manifest M --out DIR [--overlap P]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.aquamonitor_manifest import hdf5_positions
from src.data.expanded_cohort import select_frames
from src.data.finbenthic2_cohort import finbenthic2_frames
from src.data.full_specimen import FOLDS, PARTS, fold_checks
from src.features.chunk_store import file_sha256

SHA256 = {"finbenthic2": "1c18f058a36ef041bef718197c2debfec0f24607e209e35c0fdaf1f5bc05e257",
          "aquamonitor": "d6e63bb60df5300a6ccf83090b46a019446d3f91c03c69e479345b8bb13570c1"}


def quantiles(s: pd.Series) -> dict:
    return {q: float(s.quantile(p)) for q, p in (("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("max", 1))}


def load(a) -> pd.DataFrame:
    src = a.csv if a.dataset == "finbenthic2" else a.manifest
    if file_sha256(src) != SHA256[a.dataset]:
        raise RuntimeError(f"{src} is not the frozen {a.dataset} file")
    if a.dataset == "aquamonitor":
        return pd.read_parquet(src)
    rows = finbenthic2_frames(src)
    rows.insert(1, "hdf5_row", hdf5_positions(str(a.hdf5), rows.img))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(SHA256))
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--hdf5", type=Path)
    ap.add_argument("--ida", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--reuse", type=Path, action="append", default=[])
    ap.add_argument("--overlap", type=Path, action="append", default=[])
    a = ap.parse_args()
    rows = load(a)
    specimens = fold_checks(rows)
    sel = select_frames(rows, a.frames).sort_values("hdf5_row").reset_index(drop=True)
    if not sel.img.is_unique or sel.individual.nunique() != rows.individual.nunique():
        raise RuntimeError("duplicate image or a specimen without images in the selection")
    if a.dataset == "finbenthic2":
        sizes = [Image.open(a.ida / "Images" / p).size for p in sel.path]
        sel["width"], sel["height"] = [w for w, _ in sizes], [h for _, h in sizes]
    a.out.mkdir(parents=True, exist_ok=True)
    sel.to_parquet(a.out / "selection.parquet", index=False)

    reused = pd.Series(False, index=sel.index)
    for p in a.reuse:
        reused |= sel.img.isin(set(pd.read_parquet(p, columns=["img"]).img))
    sel[~reused].to_parquet(a.out / "new_images.parquet", index=False)

    counts = []
    for f in FOLDS:
        for p in PARTS:
            g = sel[sel[f] == p].groupby("taxon_group")
            counts.append(pd.DataFrame({"fold": f, "partition": p, "specimens": g.individual.nunique(),
                                        "images": g.size()}).reset_index())
    counts = pd.concat(counts)
    counts.to_csv(a.out / "per_fold_class_counts.csv", index=False)
    train = counts[counts.partition == "train"]
    val = counts[counts.partition == "val"]

    per_cam = sel.groupby(["individual", "camera"]).size()
    cams = sel.groupby("individual").camera.nunique()
    audit = {
        "dataset": a.dataset, "source_sha256": SHA256[a.dataset], "rule": {"frames_per_specimen_camera": a.frames},
        "eligible": {"images": len(rows), "specimens": int(rows.individual.nunique()), "classes": int(rows.taxon_group.nunique())},
        "specimens_per_fold": specimens,
        "checks": "per fold no specimen in two partitions; each specimen in test in exactly one fold",
        "selected": {"images": len(sel), "specimens": int(sel.individual.nunique()),
                     "specimens_one_camera": int((cams == 1).sum())},
        "selected_images_per_fold": {f: {p: int((sel[f] == p).sum()) for p in PARTS} for f in FOLDS},
        "train_specimens_per_class_min_median_max": [int(train.specimens.min()), float(train.specimens.median()),
                                                     int(train.specimens.max())],
        "val_specimens_per_class_min_median_max": [int(val.specimens.min()), float(val.specimens.median()),
                                                   int(val.specimens.max())],
        "available_frames_per_specimen_camera": quantiles(rows.groupby(["individual", "camera"]).size()),
        "selected_frames_per_specimen_camera": quantiles(per_cam),
        "specimen_cameras_below_frame_cap": int((per_cam < a.frames).sum()),
        "reuse": {"sources": [str(p) for p in a.reuse], "reused_images": int(reused.sum()),
                  "new_images": int((~reused).sum())},
        "overlap_only": {str(p): int(sel.img.isin(set(pd.read_parquet(p, columns=["img"]).img)).sum()) for p in a.overlap},
        "feature_storage_gb_float32_1024": {"selection": round(len(sel) * 4096 / 1e9, 2),
                                            "new": round(int((~reused).sum()) * 4096 / 1e9, 2)},
    }
    if a.dataset == "finbenthic2":
        audit["image_width_height_median"] = [float(sel.width.median()), float(sel.height.median())]
    (a.out / "audit.json").write_text(json.dumps(audit, indent=1, default=str))
    print(json.dumps(audit, indent=1, default=str))


if __name__ == "__main__":
    main()
