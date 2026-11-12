"""Class-label -> taxonomic ancestry from the dataset authors' own tables only (task.md f54f534). Nothing is inferred
from other sources; a class without an entry gets rank None (global fallback downstream).

- AquaMonitor (release mikkoim/aquamonitor rev 9a30834): class = ``taxon_group``; ``taxon_mapping.csv`` gives its
  scientific ``taxon`` (every taxon_group must map to exactly one taxon); ``taxon_table.csv`` gives
  kingdom..species for ``label == taxon`` (at most one row).
- FIN-Benthic2: class labels of the five-split CSV (e.g. ``Agapetus_sp``) matched to the 'Taxa' column of the release's
  Table of classes (transcribed verbatim by scripts/make_finbenthic2_taxonomy.py) after one fixed normalisation of
  both sides (underscores -> spaces, trailing '.' removed); the match must be one-to-one for every class.

Rank = the lowest filled level among species, genus, family (as the table fills them; e.g. the AquaMonitor table puts
the subfamily Tanypodinae in its genus column and Hydrachnidia in its family column, used as given).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LEVELS = ["species", "genus", "family", "order"]


def _rank(row: pd.Series) -> str | None:
    return next((lv for lv in ("species", "genus", "family") if pd.notna(row.get(lv))), None)


def _normalise(name: str) -> str:
    return name.replace("_", " ").strip().rstrip(".").strip()


def aquamonitor(mapping_csv: Path, table_csv: Path, labels: list[str]) -> pd.DataFrame:
    mapping, table = pd.read_csv(mapping_csv), pd.read_csv(table_csv)
    assert table.label.is_unique
    rows = []
    for c in labels:
        taxa = mapping.loc[mapping.taxon_group == c, "taxon"].unique()
        assert len(taxa) == 1, (c, taxa)
        hit = table[table.label == taxa[0]]
        row = {"taxon_group": c, "source_name": taxa[0]}
        row |= {lv: hit.iloc[0][lv] if len(hit) else None for lv in LEVELS}
        row["note"] = "" if len(hit) else "taxon not in taxon_table"
        rows.append(row)
    out = pd.DataFrame(rows)
    out["rank"] = out.apply(_rank, axis=1)
    return out.set_index("taxon_group")


def finbenthic2(table_csv: Path, labels: list[str]) -> pd.DataFrame:
    table = pd.read_csv(table_csv)
    by_name = {_normalise(t): t for t in table.taxa}
    assert len(by_name) == len(table), "normalisation is not one-to-one on the table"
    keys = {c: _normalise(c) for c in labels}
    assert len(set(keys.values())) == len(labels), "normalisation is not one-to-one on the labels"
    missing = [c for c, k in keys.items() if k not in by_name]
    assert not missing, f"labels without a table row: {missing}"
    rows = []
    for c, k in keys.items():
        hit = table[table.taxa == by_name[k]].iloc[0]
        rows.append({"taxon_group": c, "source_name": hit.taxa, **{lv: hit[lv] if pd.notna(hit[lv]) else None for lv in LEVELS},
                     "table_specimens": int(hit.specimens), "note": ""})
    out = pd.DataFrame(rows)
    out["rank"] = out.apply(_rank, axis=1)
    return out.set_index("taxon_group")


def load(dataset: str, labels: list[str], fin_table: Path | None = None, aqua_mapping: Path | None = None,
         aqua_table: Path | None = None) -> pd.DataFrame:
    if dataset == "finbenthic2":
        return finbenthic2(fin_table, labels)
    if dataset == "aquamonitor":
        return aquamonitor(aqua_mapping, aqua_table, labels)
    raise ValueError(dataset)
