"""FIN-Benthic2 fold-0 geometry cohort (task.md 366ee62): checks, precommitted selection and resource audit, before any
extraction. Test specimens are never selected.

Checks: the CSV is the frozen author-created five-split file (``--sha256``); fold 0 has the expected train/val/test
specimen counts and no specimen in two partitions or test specimen seen in train/val (src.data.finbenthic2_folds).

Rule (precommitted): fold0 train, per taxon up to ``--cap`` specimens, the first in sha1(f"{seed}:{individual}")
order (no source field exists to stratify on); fold0 val, every specimen; per specimen and camera up to ``--frames``
images, dealt over imaging runs and evenly spaced in capture order within a run (src.data.expanded_cohort, as for
AquaMonitor). Specimens imaged by one camera are kept. Each selected image's pixel size is read from its file header.

Writes to ``--out``: ``cohort.parquet`` (selected image rows incl. ``path``, ``width``, ``height``), ``specimens.csv``,
``per_class_counts.csv`` and ``audit.json`` (checks, availability, selection, frames, workload estimate at ``--rate``
images per second, feature storage, free disk).

Usage: build_finbenthic2_cohort.py --csv 01_finbenthic2_processed_5splits_taxon.csv --ida IDA --out DIR
       [--cap 200 --frames 10 --seed 20261111 --rate 57.6]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.expanded_cohort import select_frames, select_specimens
from src.data.finbenthic2_cohort import finbenthic2_frames
from src.data.finbenthic2_folds import fold_report
from src.features.chunk_store import file_sha256

FROZEN_SHA256 = "1c18f058a36ef041bef718197c2debfec0f24607e209e35c0fdaf1f5bc05e257"
EXPECTED_FOLD0 = {"train": 6165, "val": 1542, "test": 1924}


def quantiles(s: pd.Series) -> dict:
    return {q: float(s.quantile(p)) for q, p in (("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("max", 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--ida", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--rate", type=float, default=57.6, help="images/s of the AquaMonitor expanded-cohort GPU extraction")
    a = ap.parse_args()
    sha = file_sha256(a.csv)
    if sha != FROZEN_SHA256:
        raise RuntimeError(f"{a.csv} is not the frozen five-split CSV (sha256 {sha})")
    frames = finbenthic2_frames(a.csv)
    report = fold_report(pd.read_csv(a.csv))["per_fold"]["fold0"]
    counts = {p: report[p]["specimens"] for p in EXPECTED_FOLD0}
    if counts != EXPECTED_FOLD0 or report["specimens_in_more_than_one_partition"] \
            or report["test_specimens_seen_in_train_or_val"] or report["val_specimens_seen_in_train"]:
        raise RuntimeError(f"fold0 check failed: {counts}, {report}")
    m = frames[frames.fold0.isin(["train", "val"])]

    sel = {"train": select_specimens(m[m.fold0 == "train"], a.cap, 0, a.seed, source=None),
           "val": select_specimens(m[m.fold0 == "val"], None, 0, a.seed, source=None)}
    chosen = pd.concat([s.assign(fold0=k) for k, s in sel.items()])
    rows = m[m.individual.isin(chosen.individual)]
    cohort = select_frames(rows, a.frames).sort_values(["individual", "camera", "img"]).reset_index(drop=True)
    if not cohort.img.is_unique or set(cohort.fold0) - {"train", "val"}:
        raise RuntimeError("duplicate image or wrong split in cohort")
    sizes = [Image.open(a.ida / "Images" / p).size for p in cohort.path]
    cohort["width"], cohort["height"] = [w for w, _ in sizes], [h for _, h in sizes]
    a.out.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(a.out / "cohort.parquet", index=False)

    per_cam = cohort.groupby(["individual", "camera"]).size().unstack(fill_value=0)
    per_cam.columns = [f"frames_camera_{c}" for c in per_cam.columns]
    spec = chosen.set_index("individual").join(per_cam)
    spec["cameras"] = (per_cam > 0).sum(1).reindex(spec.index)
    spec.reset_index().to_csv(a.out / "specimens.csv", index=False)

    avail = m.groupby(["taxon_group", "fold0"]).individual.nunique().unstack(fill_value=0)
    cams = m.groupby("individual").camera.nunique()
    per_class = pd.DataFrame([{
        "taxon_group": c, "train_available": int(avail.loc[c, "train"]),
        "train_selected": int((sel["train"].taxon_group == c).sum()), "val_available": int(avail.loc[c, "val"]),
        "val_selected": int((sel["val"].taxon_group == c).sum()),
        "train_selected_one_camera": int((cams.reindex(sel["train"].individual[sel["train"].taxon_group == c]) == 1).sum())}
        for c in sorted(avail.index)])
    per_class.to_csv(a.out / "per_class_counts.csv", index=False)

    spc = rows.groupby(["individual", "camera"])
    images = len(cohort)
    audit = {
        "csv": str(a.csv), "csv_sha256": sha, "fold0_specimens": counts, "fold0_checks": "no specimen in two partitions, "
        "no test or val specimen seen in train", "rule": {"cap": a.cap, "frames": a.frames, "seed": a.seed},
        "classes": len(per_class),
        "classes_train_available_at_least": {k: int((per_class.train_available >= k).sum()) for k in (15, 50, 100, 200)},
        "classes_val_available_at_least": {k: int((per_class.val_available >= k).sum()) for k in (5, 10, 20)},
        "specimens_selected": {k: len(s) for k, s in sel.items()},
        "selected_one_camera": {k: int((cams.reindex(s.individual) == 1).sum()) for k, s in sel.items()},
        "available_frames_per_specimen_camera": quantiles(spc.size()),
        "available_runs_per_specimen_camera": quantiles(spc.imaging_run.nunique()),
        "specimen_cameras_with_more_than_one_run": int((spc.imaging_run.nunique() > 1).sum()),
        "selected_frames_per_specimen_camera": quantiles(cohort.groupby(["individual", "camera"]).size()),
        "specimen_cameras_below_frame_cap": int((cohort.groupby(["individual", "camera"]).size() < a.frames).sum()),
        "image_width_height_median": [float(cohort.width.median()), float(cohort.height.median())],
        "images": {"cohort": images, "train": int((cohort.fold0 == "train").sum()), "val": int((cohort.fold0 == "val").sum())},
        "estimate": {"rate_images_per_second": a.rate, "gpu_minutes": round(images / a.rate / 60, 1),
                     "feature_storage_gb": round(images * 1024 * 4 / 1e9, 2),
                     "free_disk_gb": round(shutil.disk_usage(a.out).free / 1e9, 1)},
    }
    (a.out / "audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
