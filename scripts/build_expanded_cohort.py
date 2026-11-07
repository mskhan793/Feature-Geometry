"""Expanded geometry cohort (task.md fb46e1f): audit, selection and resource estimate, before any extraction.

Rule (src.data.expanded_cohort, precommitted): fold0 train, per class up to ``--cap`` specimens with a source-aware
allocation over ``dataset`` (floor ``--floor`` per group); fold0 val, every specimen; up to ``--frames`` frames per
specimen and camera spread over imaging runs and capture time. Test rows are never selected.

Writes to ``--out``:
- ``cohort.parquet``: selected image rows (all manifest columns, incl. hdf5_row and every fold);
- ``specimens.csv``: one row per selected specimen (split, class, dataset, year, site, cameras, frames per camera);
- ``new_images.parquet``: cohort rows whose features are not in ``--reuse-cache`` (matched by image name), to extract;
- ``audit.json``: availability and selection per class and split, source composition (full train vs selected),
  classes reaching 50/100/200 specimens, frame and imaging-run availability, selected frame counts, reuse and the
  resource estimate (new images / ``--rate`` images per second; float32 1024-d features).

Usage: build_expanded_cohort.py --manifest M --reuse-cohort OLD_COHORT --reuse-cache OLD_CACHE --out DIR
       [--cap 200 --floor 20 --frames 10 --seed 20261107 --rate 56.4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.expanded_cohort import select_frames, select_specimens
from src.features.chunk_store import file_sha256, load_cache


def quantiles(s: pd.Series) -> dict:
    return {q: float(s.quantile(p)) for q, p in (("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("max", 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--reuse-cohort", required=True, type=Path)
    ap.add_argument("--reuse-cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--floor", type=int, default=20)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--rate", type=float, default=56.4, help="images/s measured for the camera-pilot GPU extraction")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(a.manifest)
    m = m[m.fold0.isin(["train", "val"])]
    if (m.groupby("individual")[["dataset", "year", "site", "taxon_group", "fold0"]].nunique() > 1).any().any():
        raise RuntimeError("a specimen has more than one dataset/year/site/class/split")

    sel = {"train": select_specimens(m[m.fold0 == "train"], a.cap, a.floor, a.seed),
           "val": select_specimens(m[m.fold0 == "val"], None, a.floor, a.seed)}
    chosen = pd.concat([s.assign(fold0=k) for k, s in sel.items()])
    rows = m[m.individual.isin(chosen.individual)]
    cohort = select_frames(rows, a.frames).sort_values("hdf5_row").reset_index(drop=True)
    if not cohort.img.is_unique or set(cohort.fold0) - {"train", "val"}:
        raise RuntimeError("duplicate image or wrong split in cohort")
    cohort.to_parquet(a.out / "cohort.parquet", index=False)

    frames = cohort.groupby(["individual", "camera"]).size().unstack(fill_value=0)
    frames.columns = [f"frames_camera_{c}" for c in frames.columns]
    meta = rows.groupby("individual")[["year", "site", "lake", "dataset"]].first()
    spec = chosen.set_index("individual").join(meta.drop(columns="dataset")).join(frames)
    spec["cameras"] = (frames > 0).sum(1).reindex(spec.index)
    spec.reset_index().to_csv(a.out / "specimens.csv", index=False)

    old_names = set(pd.read_parquet(a.reuse_cohort, columns=["img"]).img)
    load_cache(a.reuse_cache, pd.read_parquet(a.reuse_cohort, columns=["img"]).img.tolist())  # checks the cache is intact
    reused = cohort.img.isin(old_names)
    cohort[~reused].to_parquet(a.out / "new_images.parquet", index=False)

    avail = m.groupby(["taxon_group", "fold0"]).individual.nunique().unstack(fill_value=0)
    cams = m.groupby("individual").camera.nunique()
    per_class = []
    for c in sorted(avail.index):
        st = sel["train"][sel["train"].taxon_group == c]
        full = m[(m.fold0 == "train") & (m.taxon_group == c)].groupby("individual").dataset.first().value_counts()
        per_class.append({
            "taxon_group": c, "train_available": int(avail.loc[c, "train"]), "train_selected": len(st),
            "val_available": int(avail.loc[c, "val"]), "val_selected": int((sel["val"].taxon_group == c).sum()),
            "train_selected_one_camera": int((cams.reindex(st.individual) == 1).sum()),
            "train_datasets_full": full.sort_index().to_dict(), "train_datasets_selected": st.dataset.value_counts().sort_index().to_dict()})
    per_class = pd.DataFrame(per_class)
    per_class.to_csv(a.out / "per_class_counts.csv", index=False)

    spc = rows.groupby(["individual", "camera"])
    full_train = m[m.fold0 == "train"].groupby("individual").dataset.first().value_counts(normalize=True)
    sel_train = sel["train"].dataset.value_counts(normalize=True)
    new = int((~reused).sum())
    audit = {
        "manifest_sha256": file_sha256(a.manifest), "rule": {"cap": a.cap, "floor": a.floor, "frames": a.frames, "seed": a.seed},
        "classes": len(per_class),
        "classes_train_available_at_least": {k: int((per_class.train_available >= k).sum()) for k in (50, 100, 200)},
        "classes_train_selected_at_least": {k: int((per_class.train_selected >= k).sum()) for k in (50, 100, 200)},
        "specimens_selected": {k: len(s) for k, s in sel.items()},
        "specimens_available": {k: int(m[m.fold0 == k].individual.nunique()) for k in ("train", "val")},
        "selected_one_camera": {k: int((cams.reindex(s.individual) == 1).sum()) for k, s in sel.items()},
        "train_source_share_full": full_train.round(4).to_dict(), "train_source_share_selected": sel_train.round(4).to_dict(),
        "val_source_share": sel["val"].dataset.value_counts(normalize=True).round(4).to_dict(),
        "available_frames_per_specimen_camera": quantiles(spc.size()),
        "available_runs_per_specimen_camera": quantiles(spc.imaging_run.nunique()),
        "available_frames_per_run_camera": quantiles(rows.groupby(["imaging_run", "camera"]).size()),
        "specimen_cameras_with_more_than_one_run": int((spc.imaging_run.nunique() > 1).sum()),
        "selected_frames_per_specimen_camera": quantiles(cohort.groupby(["individual", "camera"]).size()),
        "specimen_cameras_below_frame_cap": int((cohort.groupby(["individual", "camera"]).size() < a.frames).sum()),
        "images": {"cohort": len(cohort), "train": int((cohort.fold0 == "train").sum()), "val": int((cohort.fold0 == "val").sum()),
                   "reused_from_camera_pilot": int(reused.sum()), "new": new},
        "estimate": {"rate_images_per_second": a.rate, "gpu_minutes": round(new / a.rate / 60, 1),
                     "feature_storage_gb": round(new * 1024 * 4 / 1e9, 2)},
    }
    (a.out / "audit.json").write_text(json.dumps(audit, indent=1, default=str))
    print(json.dumps(audit, indent=1, default=str))


if __name__ == "__main__":
    main()
