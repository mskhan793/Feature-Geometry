"""Cross-check the taxonomy audit (authors' tables, scripts/audit_taxonomy.py) against the v1 repository's parent maps
(Lightning_Aqua_Vision_Language_v1 assets/label_maps/htc_labels/aqua_plus_finb1_finb2_hoye_vocab/
taxon_parent_mappings.json: species_to_genus, genus_to_family, family_to_order; the manuscript's combined vocabulary).
Read-only. For every class and link (species->genus, genus->family, family->order) reports agree / differ / not in v1.

Usage: compare_taxonomy_with_v1.py --classes AUDIT/<dataset>/classes.csv ... --v1 taxon_parent_mappings.json --out FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

LINKS = [("species", "genus", "species_to_genus"), ("genus", "family", "genus_to_family"), ("family", "order", "family_to_order")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", required=True, nargs="+", type=Path)
    ap.add_argument("--v1", required=True, type=Path)
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
    a.out.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
