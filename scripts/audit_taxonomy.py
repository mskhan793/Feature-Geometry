"""Taxonomy feasibility audit for class-balanced / taxonomy-informed covariance pooling (gate A2 of task.md f54f534;
metadata only, no features, no validation outcomes).

Per class: the authors' ancestry (src/data/taxonomy.py), rank, specimens in the selection (and, for FIN-Benthic2, the
release table's specimen count, which must be equal). Per fold: train / val specimens per class, eligible targets
(>= 15 train and >= 5 val, as all previous comparisons), donor candidates (>= 15 train; train only) and, for every
eligible target and dimension d, the donor level the predeclared rule picks (src/methods/taxonomic_pooling.py), the
donors and their degrees of freedom. Also lists classes without ancestry, classes sharing one scientific name, and
targets whose genus / family holds no other candidate.

Usage: audit_taxonomy.py --dataset finbenthic2 --selection selection.parquet --fin-table table.csv --out DIR
       audit_taxonomy.py --dataset aquamonitor --selection selection.parquet --aqua-mapping m.csv --aqua-table t.csv --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import taxonomy as tx
from src.methods.taxonomic_pooling import donor_df, taxonomy_donors
from src.statistics.heldout_density import MIN_TRAIN, MIN_VAL

FOLDS = [f"fold{k}" for k in range(5)]


def load_taxonomy(a, labels):
    if a.dataset == "finbenthic2":
        return tx.finbenthic2(a.fin_table, labels)
    return tx.aquamonitor(a.aqua_mapping, a.aqua_table, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["finbenthic2", "aquamonitor"])
    ap.add_argument("--selection", required=True, type=Path)
    ap.add_argument("--fin-table", type=Path)
    ap.add_argument("--aqua-mapping", type=Path)
    ap.add_argument("--aqua-table", type=Path)
    ap.add_argument("--dims", type=int, nargs="+", default=[16, 64, 1023])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sel = pd.read_parquet(a.selection, columns=["individual", "taxon_group", *FOLDS])
    spec = sel.drop_duplicates("individual")
    assert spec.groupby("individual").taxon_group.nunique().max() == 1
    labels = sorted(spec.taxon_group.unique())
    tax = load_taxonomy(a, labels)
    classes = tax.copy()
    classes["specimens"] = spec.taxon_group.value_counts()
    if "table_specimens" in classes:
        classes["specimens_match_table"] = classes.specimens == classes.table_specimens
    classes.to_csv(a.out / "classes.csv")

    donors, per_fold = [], {}
    for f in FOLDS:
        n_tr = spec[spec[f] == "train"].taxon_group.value_counts()
        n_va = spec[spec[f] == "val"].taxon_group.value_counts()
        eligible = sorted(c for c in n_tr.index if n_tr[c] >= MIN_TRAIN and n_va.get(c, 0) >= MIN_VAL)
        candidates = sorted(c for c in n_tr.index if n_tr[c] >= MIN_TRAIN)
        n = n_tr.to_dict()
        per_fold[f] = {"eligible": len(eligible), "candidates": len(candidates),
                       "candidates_not_eligible": sorted(set(candidates) - set(eligible))}
        for d in a.dims:
            for c in eligible:
                level, dn = taxonomy_donors(c, tax, candidates, n, d)
                same = {lv: sorted(o for o in candidates if o != c and pd.notna(tax.loc[c, lv]) and tax.loc[o, lv] == tax.loc[c, lv])
                        for lv in ("genus", "family")}
                donors.append({"fold": f, "dim": d, "taxon_group": c, "rank": tax.loc[c, "rank"], "n_train": int(n[c]),
                               "level": level, "donor_classes": len(dn), "donor_df": donor_df(n, dn),
                               "donors": ";".join(dn) if level != "global" else "",
                               "genus_mates": ";".join(same["genus"]), "family_mates": ";".join(same["family"])})
    donors = pd.DataFrame(donors)
    donors.to_csv(a.out / "donors.csv", index=False)
    by_name = tax.reset_index().groupby("source_name").taxon_group.apply(list)
    summary = {
        "dataset": a.dataset, "classes": len(labels), "rank_counts": tax["rank"].value_counts(dropna=False).to_dict(),
        "classes_without_ancestry": sorted(tax.index[tax["rank"].isna()]),
        "classes_sharing_a_scientific_name": {k: v for k, v in by_name.items() if len(v) > 1},
        "specimen_counts_equal_release_table": bool(classes["specimens_match_table"].all()) if "specimens_match_table" in classes else None,
        "folds": per_fold,
        "level_counts_by_dim": {str(d): donors[donors.dim == d].groupby("fold")["level"].value_counts().unstack(fill_value=0).to_dict("index")
                                for d in a.dims},
        "targets_without_any_genus_or_family_mate": sorted(set(donors[(donors.genus_mates == "") & (donors.family_mates == "")].taxon_group)),
        "min_train": MIN_TRAIN, "min_val": MIN_VAL, "dims": a.dims}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
