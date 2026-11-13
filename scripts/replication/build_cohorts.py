"""Analysis cohorts and class support of the 8 x 4 geometry replication (task.md a82e89e), metadata only.

- Single-protocol datasets: src.data.benchmark_cohort.cohort on each train+val extraction manifest ->
  ``OUT/cohorts/<dataset>.parquet`` (partition column ``official``).
- FIN-Benthic2 / AquaMonitor: the existing full-specimen selections are used unchanged (partition columns
  fold0..fold4); only their support is tabulated.
- Support per dataset and partition (``OUT/support/<dataset>_<partition>.csv``): train / val units per class and
  eligibility under the established rule (>= 15 train and >= 5 val units, as in every earlier geometry script);
  ``OUT/support_summary.json``: classes, eligible classes, units, eligible pairs, conflicting-label units, sha256.

Usage: build_cohorts.py --root FEATURE_GEOMETRY_DATA --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.benchmark_cohort import SPLIT, cohort, support
from src.features.chunk_store import file_sha256

MIN_TRAIN, MIN_VAL = 15, 5
SINGLE = ("bioscan5m", "df24", "food101", "oxford_pets", "cifar100", "fgvc_aircraft")
FIVE_FOLD = {"finbenthic2": "full_specimen/finbenthic2/selection.parquet",
             "aquamonitor": "full_specimen/aquamonitor/selection.parquet"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    (a.out / "cohorts").mkdir(parents=True, exist_ok=True)
    (a.out / "support").mkdir(parents=True, exist_ok=True)
    summary = {"rule": {"min_train": MIN_TRAIN, "min_val": MIN_VAL}, "datasets": {}}
    inputs = {d: (a.root / FIVE_FOLD[d], [f"fold{k}" for k in range(5)], None) for d in FIVE_FOLD}
    for d in SINGLE:
        src = a.root / f"benchmark/extract/{d}_trainval.parquet"
        c, info = cohort(pd.read_parquet(src))
        path = a.out / "cohorts" / f"{d}.parquet"
        c.to_parquet(path, index=False)
        inputs[d] = (path, [SPLIT], {"source": str(src), "source_sha256": file_sha256(src), **info})
    for d, (path, parts, info) in inputs.items():
        c = pd.read_parquet(path, columns=["img", "taxon_group", "individual", *parts])
        rec = {"cohort": str(path), "cohort_sha256": file_sha256(path), "images": len(c),
               "units": int(c.individual.nunique()), **(info or {}), "partitions": {}}
        for p in parts:
            s = support(c, p, MIN_TRAIN, MIN_VAL)
            s.to_csv(a.out / "support" / f"{d}_{p}.csv", index=False)
            e = s[s.eligible]
            rec["partitions"][p] = {"classes_train_or_val": len(s), "classes_eligible": len(e),
                                    "eligible_pairs": len(e) * (len(e) - 1) // 2,
                                    "train_units": int(s.train.sum()), "val_units": int(s.val.sum()),
                                    "train_units_eligible": int(e.train.sum()), "val_units_eligible": int(e.val.sum()),
                                    "eligible_train_units_min_median_max": [int(e.train.min()), float(e.train.median()),
                                                                            int(e.train.max())] if len(e) else None}
        summary["datasets"][d] = rec
    (a.out / "support_summary.json").write_text(json.dumps(summary, indent=1))
    for d, r in summary["datasets"].items():
        print(d, {p: (v["classes_train_or_val"], v["classes_eligible"], v["eligible_pairs"]) for p, v in r["partitions"].items()})


if __name__ == "__main__":
    main()
