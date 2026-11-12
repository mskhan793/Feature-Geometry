"""Cross-check the taxonomy audit (authors' tables, scripts/audit_taxonomy.py) against the v1 repository's parent maps
(Lightning_Aqua_Vision_Language_v1 assets/label_maps/htc_labels/aqua_plus_finb1_finb2_hoye_vocab/
taxon_parent_mappings.json: species_to_genus, genus_to_family, family_to_order; the manuscript's combined vocabulary).
Read-only. For every class and link (species->genus, genus->family, family->order) reports agree / differ / not in v1.
Optionally (--v1-table DATASET=TABLE, a per-image table with a class column 'taxon' and genus / family columns, e.g. the
v1 FIN-Benthic2 hierarchy finb2_taxon.parquet) compares the groupings: which classes share a genus / a family, ignoring
spelling; identical partitions mean the donor pools are the same.

Usage: compare_taxonomy_with_v1.py --classes AUDIT/<dataset>/classes.csv ... --v1 taxon_parent_mappings.json
                                   [--v1-table finbenthic2=finb2_taxon.parquet] --out FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

LINKS = [("species", "genus", "species_to_genus"), ("genus", "family", "genus_to_family"), ("family", "order", "family_to_order")]


def partition(labels: pd.Series) -> set:
    """Classes grouped by a level's name (classes without a name at that level are left out)."""
    return {frozenset(g.index) for _, g in labels.dropna().groupby(labels.dropna())}


def compare_groupings(classes: pd.DataFrame, table: pd.DataFrame) -> dict:
    other = table.drop_duplicates("taxon").set_index("taxon")
    assert set(other.index) == set(classes.index), "class sets differ"
    out = {}
    for level in ("genus", "family"):
        a, b = partition(classes[level]), partition(other.loc[classes.index, level])
        out[level] = {"identical": a == b, "only_authors_table": [sorted(g) for g in a - b], "only_v1": [sorted(g) for g in b - a],
                      "names_differing": sorted({f"{x} / {y}" for x, y in zip(classes[level], other.loc[classes.index, level])
                                                 if pd.notna(x) and pd.notna(y) and x != y})}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", required=True, nargs="+", type=Path)
    ap.add_argument("--v1", required=True, type=Path)
    ap.add_argument("--v1-table", nargs="*", default=[])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    v1 = json.loads(a.v1.read_text())
    rows = []
    for f in a.classes:
        t = pd.read_csv(f)
        for r in t.itertuples():
            for child, parent, key in LINKS:
                c, p = getattr(r, child), getattr(r, parent)
                if pd.isna(c) or pd.isna(p):
                    continue
                v = v1[key].get(c)
                rows.append({"dataset": f.parent.name, "taxon_group": r.taxon_group, "link": f"{child}->{parent}", "child": c,
                             "parent_authors_table": p, "parent_v1": v,
                             "status": "not in v1" if v is None else ("agree" if v == p else "differ")})
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    summary = {"v1_file_sha256": hashlib.sha256(a.v1.read_bytes()).hexdigest(),
               "counts": {f"{d}|{s}": int(v) for (d, s), v in out.groupby(["dataset", "status"]).size().items()},
               "differ": out[out.status == "differ"].to_dict("records"), "not_in_v1": out[out.status == "not in v1"].to_dict("records")}
    for spec in a.v1_table:
        name, path = spec.split("=", 1)
        classes = pd.read_csv(next(f for f in a.classes if f.parent.name == name)).set_index("taxon_group")
        table = pd.read_parquet(path, columns=["taxon", "genus", "family"])
        summary[f"groupings_{name}"] = {"table": path, "table_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                        **compare_groupings(classes, table)}
    a.out.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
