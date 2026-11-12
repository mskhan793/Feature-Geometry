"""Train + validation extraction manifest from a standardised benchmark manifest, for sources read without packing
(CIFAR-100 pickles): ``img`` = source_id, plus split, label and unit_id. Official test rows are never written.

Usage: extraction_manifest.py --manifest DATASET.parquet --out extract.parquet [--splits train val]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    m = pd.read_parquet(a.manifest)
    assert "test" not in a.splits, "official test is not extracted in this phase"
    out = m[m.split.isin(a.splits)].rename(columns={"source_id": "img"})[["img", "split", "label", "unit_id"]]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index(drop=True).to_parquet(a.out, index=False)
    print(a.out, len(out), out.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
