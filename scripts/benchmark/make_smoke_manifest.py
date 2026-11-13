"""Fixed smoke-test sample of training units from a standardised benchmark manifest (Stage B, task.md 8538df5): the
first ``--n`` train rows in sha1(f"{seed}:{source_id}") order, one image per unit. Software / reproducibility check
only; not a research partition. Accepts a standardised manifest (``source_id``) or a packed extraction manifest
(``img``, ``hdf5_row``, from pack_archive_hdf5.py); writes the extraction manifest (``img``, label, unit_id, and
``hdf5_row`` when present) and the sample ids.

Usage: make_smoke_manifest.py --manifest DATASET.parquet --n 512 --seed 20261112 --out smoke.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.expanded_cohort import stable_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    m = pd.read_parquet(a.manifest).rename(columns={"source_id": "img"})
    train = m[m.split == "train"].drop_duplicates("unit_id")
    train = train.assign(key=stable_order(train.img, a.seed).values).sort_values("key").head(a.n)
    out = train[[c for c in ("img", "hdf5_row", "label", "unit_id", "dataset") if c in train]].reset_index(drop=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    a.out.with_suffix(".json").write_text(json.dumps({"manifest": str(a.manifest), "n": len(out), "seed": a.seed,
                                                      "classes": int(out.label.nunique()), "img": out.img.tolist()}, indent=1))
    print(len(out), "rows,", out.label.nunique(), "classes")


if __name__ == "__main__":
    main()
