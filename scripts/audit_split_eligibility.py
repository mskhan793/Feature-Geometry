"""Read-only audit of official train/validation/test partitions for candidate geometry benchmarks.

For each dataset the released metadata are read as distributed and summarised: partition sizes, independent units
(specimen, barcode, observation), units that occur in more than one partition, labels present in each partition,
per-class support, and locally available images. Nothing is re-split or written back.

- ``finbenthic2``: the dataset's ``Machine_learning_splits.txt`` and ``Comparison_data_splits.txt`` (10 splits each,
  TRAIN/VAL/TEST, unit = specimen), image files under ``Images/<specimen>/``, and the derived 5-fold CSV for
  comparison with the released splits.
- ``bioscan5m``: ``BIOSCAN_5M_Insect_Dataset_metadata.csv`` (``split`` field; pretrain rows skipped; unit =
  specimen record, grouped by DNA barcode by the authors) and the cropped_256 image folders per split.
- ``df24``: ``DanishFungi2024-train.csv`` and ``DanishFungi2024-pubtest.csv`` (unit = observation).

Usage:
  audit_split_eligibility.py finbenthic2 --ida IDA_FOLDER --fivefold CSV --out DIR
  audit_split_eligibility.py bioscan5m --metadata CSV --images CROPPED_256_DIR --out DIR
  audit_split_eligibility.py df24 --train CSV --test CSV --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

THRESHOLDS = (8, 20, 50, 200)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def support(counts: pd.Series) -> dict:
    """Distribution of independent units per class."""
    return {"classes": int(len(counts)), "min": int(counts.min()), "median": float(counts.median()),
            "max": int(counts.max()), **{f"classes_with_at_least_{t}": int((counts >= t).sum()) for t in THRESHOLDS}}


def partition_table(df: pd.DataFrame, part: str, unit: str, label: str) -> dict:
    g = df.groupby(part)
    return {str(k): {"images": int(len(v)), "units": int(v[unit].nunique()), "classes": int(v[label].nunique())}
            for k, v in g}


def finbenthic2(a) -> dict:
    ida = Path(a.ida)
    out = {"files": {}}
    for name in ("Machine_learning_splits.txt", "Comparison_data_splits.txt"):
        out["files"][name] = sha256(ida / name)
    out["files"][str(a.fivefold)] = sha256(Path(a.fivefold))
    images = {p.parent.name + "/" + p.name for p in (ida / "Images").glob("*/*")}
    result = {}
    for name in ("Machine_learning_splits.txt", "Comparison_data_splits.txt"):
        df = pd.read_csv(ida / name, sep=" ").rename(columns={"tunnus": "specimen", "Class": "taxon", "Img_id": "img"})
        splits = [c for c in df.columns if c.startswith("SPLIT")]
        df[splits] = df[splits].apply(lambda s: s.str.strip())
        per_split = {}
        for s in splits:
            used = df[df[s].notna()]
            spec = used.groupby("specimen")[s].nunique()
            train_taxa = set(used.loc[used[s] == "TRAIN", "taxon"])
            units = used.drop_duplicates(["specimen", s])
            per_split[s] = {
                "partitions": partition_table(used, s, "specimen", "taxon"),
                "images_not_used": int(df[s].isna().sum()),
                "specimens_in_more_than_one_partition": int((spec > 1).sum()),
                "taxa_in_train_missing_from_val": sorted(train_taxa - set(used.loc[used[s] == "VAL", "taxon"])),
                "taxa_in_train_missing_from_test": sorted(train_taxa - set(used.loc[used[s] == "TEST", "taxon"])),
                "specimens_per_taxon": {p: support(units[units[s] == p].groupby("taxon").size())
                                        for p in ("TRAIN", "VAL", "TEST")}}
        test_sets = [set(df.loc[df[s] == "TEST", "specimen"]) for s in splits]
        result[name] = {
            "images": int(len(df)), "specimens": int(df.specimen.nunique()), "taxa": int(df.taxon.nunique()),
            "values": sorted(df[splits].stack().unique().tolist()),
            "specimens_with_more_than_one_taxon": int((df.groupby("specimen").taxon.nunique() > 1).sum()),
            "image_files_missing": int(sum(f"{r.specimen}/{r.img}" not in images for r in df.itertuples())),
            "camera_prefix_counts": df.img.str.split("-").str[0].value_counts().to_dict(),
            "specimens_test_in_some_split": int(len(set().union(*test_sets))),
            "per_split": per_split}
    ml = pd.read_csv(ida / "Machine_learning_splits.txt", sep=" ").rename(columns={"tunnus": "specimen"})
    ml_spec = ml.drop_duplicates("specimen").set_index("specimen")
    five = pd.read_csv(a.fivefold).drop_duplicates("individual").set_index("individual")
    five = five.reindex(ml_spec.index)
    mapping = {"train": "TRAIN", "val": "VAL", "test": "TEST"}
    result["fivefold_vs_released_machine_learning_splits"] = {
        f: {s: round(float((five[f].map(mapping) == ml_spec[s].str.strip()).mean()), 4)
            for s in ml_spec.columns if s.startswith("SPLIT")} for f in [f"fold{k}" for k in range(5)]}
    return out | result


def bioscan5m(a) -> dict:
    cols = ["processid", "species", "genus", "dna_barcode", "dna_bin", "split"]
    parts = []
    for chunk in pd.read_csv(a.metadata, usecols=cols, chunksize=500_000, dtype=str):
        parts.append(chunk[chunk.split != "pretrain"])
    df = pd.concat(parts, ignore_index=True)
    out = {"files": {str(a.metadata): sha256(Path(a.metadata))},
           "records_excluding_pretrain": int(len(df)), "processid_unique": bool(df.processid.is_unique),
           "partitions": {s: {"records": int(len(v)), "with_species": int(v.species.notna().sum()),
                              "species": int(v.species.nunique()), "barcodes": int(v.dna_barcode.nunique()),
                              "bins": int(v.dna_bin.nunique())} for s, v in df.groupby("split")}}
    barcodes = {s: set(v.dna_barcode.dropna()) for s, v in df.groupby("split")}
    names = sorted(barcodes)
    out["barcodes_shared_between_partitions"] = {f"{x}|{y}": len(barcodes[x] & barcodes[y])
                                                 for i, x in enumerate(names) for y in names[i + 1:]}
    closed = df[df.split.isin(["train", "val", "test"])]
    train = closed[closed.split == "train"].groupby("species").size()
    val = closed[closed.split == "val"].groupby("species").size()
    test = closed[closed.split == "test"].groupby("species").size()
    out["closed_world"] = {
        "species_in_val_not_in_train": int(len(set(val.index) - set(train.index))),
        "species_in_test_not_in_train": int(len(set(test.index) - set(train.index))),
        "train_records_per_species": support(train),
        "val_records_per_species": support(val), "test_records_per_species": support(test),
        "species_with_val_and_test": int(len(set(val.index) & set(test.index))),
        **{f"species_with_train_at_least_{t}_and_val": int(((train >= t) & train.index.isin(val.index)).sum())
           for t in THRESHOLDS}}
    out["local_images_per_split"] = {p.name: sum(1 for f in p.rglob("*") if f.is_file())
                                     for p in sorted(Path(a.images).iterdir()) if p.is_dir()}
    return out


def df24(a) -> dict:
    tr, te = pd.read_csv(a.train, low_memory=False), pd.read_csv(a.test, low_memory=False)
    out = {"files": {str(a.train): sha256(Path(a.train)), str(a.test): sha256(Path(a.test))},
           "partition_files": ["train", "pubtest"], "validation_file_released": False}
    for name, d in (("train", tr), ("pubtest", te)):
        out[name] = {"images": int(len(d)), "observations": int(d.observationID.nunique()),
                     "species_labels_present": int(d.species.notna().sum()), "classes": int(d.class_id.nunique()),
                     "observations_per_class": support(d.drop_duplicates("observationID").groupby("class_id").size())}
    out["observations_in_both"] = int(len(set(tr.observationID) & set(te.observationID)))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="dataset", required=True)
    p = sub.add_parser("finbenthic2")
    p.add_argument("--ida", required=True)
    p.add_argument("--fivefold", required=True)
    p = sub.add_parser("bioscan5m")
    p.add_argument("--metadata", required=True)
    p.add_argument("--images", required=True)
    p = sub.add_parser("df24")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    for p in sub.choices.values():
        p.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    result = {"finbenthic2": finbenthic2, "bioscan5m": bioscan5m, "df24": df24}[a.dataset](a)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.dataset}.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
