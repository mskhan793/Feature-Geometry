"""Integrity checks for the author-created FIN-Benthic2 five-split protocol (``fold0``..``fold4`` columns with
train/val/test per image; unit = specimen ``individual``; label = ``taxon``). Read-only: the CSV is never rewritten."""

from __future__ import annotations

import pandas as pd

FOLDS = [f"fold{k}" for k in range(5)]
PARTS = ("train", "val", "test")


def fold_report(df: pd.DataFrame) -> dict:
    """Per fold: counts, leakage within the fold, class support; across folds: overlap of test and val sets."""
    if not df.img.is_unique:
        raise ValueError("image names are not unique")
    per_fold, sets = {}, {}
    classes = set(df.taxon)
    for f in FOLDS:
        values = set(df[f].astype(str))
        if not values <= set(PARTS):
            raise ValueError(f"{f} has values {sorted(values)}")
        spec = {p: set(df.loc[df[f] == p, "individual"]) for p in PARTS}
        sets[f] = spec
        units = df.drop_duplicates(["individual", f])
        per_fold[f] = {
            **{p: {"images": int((df[f] == p).sum()), "specimens": len(spec[p]),
                   "classes": int(df.loc[df[f] == p, "taxon"].nunique())} for p in PARTS},
            "specimens_in_more_than_one_partition": int((df.groupby("individual")[f].nunique() > 1).sum()),
            "test_specimens_seen_in_train_or_val": len(spec["test"] & (spec["train"] | spec["val"])),
            "val_specimens_seen_in_train": len(spec["val"] & spec["train"]),
            "classes_missing": {p: sorted(classes - set(df.loc[df[f] == p, "taxon"])) for p in PARTS},
            "min_specimens_per_class": {p: int(units[units[f] == p].groupby("taxon").size().reindex(
                sorted(classes), fill_value=0).min()) for p in PARTS}}
    pairs = [(x, y) for i, x in enumerate(FOLDS) for y in FOLDS[i + 1:]]
    test_count = pd.Series([s for f in FOLDS for s in sets[f]["test"]]).value_counts().reindex(
        sorted(set(df.individual)), fill_value=0)
    return {"images": len(df), "specimens": int(df.individual.nunique()), "classes": len(classes),
            "specimens_with_more_than_one_taxon": int((df.groupby("individual").taxon.nunique() > 1).sum()),
            "per_fold": per_fold,
            "test_overlap_between_folds": {f"{x}|{y}": len(sets[x]["test"] & sets[y]["test"]) for x, y in pairs},
            "val_overlap_between_folds": {f"{x}|{y}": len(sets[x]["val"] & sets[y]["val"]) for x, y in pairs},
            "specimens_by_times_in_test": test_count.value_counts().sort_index().to_dict()}


def class_support(df: pd.DataFrame) -> pd.DataFrame:
    """Specimens per class in every fold and partition (one row per class)."""
    out = {f"{f}_{p}": df[df[f] == p].groupby("taxon").individual.nunique() for f in FOLDS for p in PARTS}
    return pd.DataFrame(out).fillna(0).astype(int).rename_axis("taxon").reset_index()
