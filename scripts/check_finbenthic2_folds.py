"""Verify and freeze the author-created FIN-Benthic2 five-split CSV (read-only).

Records the CSV's sha256, checks that every image exists under IDA/Images/<individual>/ and that images and labels
match the released Machine_learning_splits.txt, and writes the fold report (src.data.finbenthic2_folds) and the
per-class specimen support.

Writes to ``--out``: ``finbenthic2_fivefold.json`` and ``finbenthic2_fivefold_class_support.csv``. Exits non-zero
if a fold leaks a specimen between partitions or the file does not match the released images and labels.

Usage: check_finbenthic2_folds.py --csv 01_finbenthic2_processed_5splits_taxon.csv --ida IDA_FOLDER --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.finbenthic2_folds import FOLDS, class_support, fold_report


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--ida", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    report = {"protocol": "FIN-Benthic2 five-split CSV created by the author with the taxonomist toolkit "
                          "(StratifiedGroupKFold by specimen on taxon); author-approved, not a dataset-publisher "
                          "split (task 2f6774c, e5f42d9)",
              "csv": str(a.csv), "csv_sha256": sha256(a.csv), **fold_report(df)}
    files = {p.parent.name + "/" + p.name for p in (a.ida / "Images").glob("*/*")}
    report["image_files_missing"] = int(sum(f"{i}/{n}" not in files for i, n in zip(df.individual, df.img)))
    released = pd.read_csv(a.ida / "Machine_learning_splits.txt", sep=" ").rename(
        columns={"tunnus": "individual", "Class": "taxon", "Img_id": "img"})
    m = df[["img", "individual", "taxon"]].merge(released[["img", "individual", "taxon"]], on="img", how="outer",
                                                 suffixes=("", "_released"), indicator=True)
    report["released_images_labels_identical"] = bool(
        (m._merge == "both").all() and (m.individual == m.individual_released).all()
        and (m.taxon == m.taxon_released).all())
    leaks = sum(report["per_fold"][f]["specimens_in_more_than_one_partition"]
                + report["per_fold"][f]["test_specimens_seen_in_train_or_val"] for f in FOLDS)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "finbenthic2_fivefold.json").write_text(json.dumps(report, indent=1))
    class_support(df).to_csv(a.out / "finbenthic2_fivefold_class_support.csv", index=False)
    print(json.dumps(report, indent=1))
    if leaks or report["image_files_missing"] or not report["released_images_labels_identical"]:
        raise SystemExit("integrity check failed")


if __name__ == "__main__":
    main()
