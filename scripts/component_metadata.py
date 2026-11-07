"""Do the 2-component tied-mixture groups of each class follow acquisition metadata? (read-only, after
class_geometry_pilot.py)

Per class and split (train components from the train fit; val components predicted by the same train fit), the
adjusted Rand index (ARI) between the component labels and each metadata field (``dataset``, ``year``, ``site``),
plus the class's dataset composition. ARI near 1 = the split reproduces that field; near 0 = unrelated. Only classes
whose specimens span at least two values of a field get an ARI for it.

Also writes per_class_summary.csv: the compact per-class table (geometry, model comparison, verdict, nearest class,
component-metadata ARIs).

Usage: component_metadata.py --cohort cohort.parquet --components specimen_components.csv --per-class per_class.csv --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

FIELDS = ("dataset", "year", "site")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--components", required=True, type=Path)
    ap.add_argument("--per-class", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    cohort = pd.read_parquet(a.cohort)
    meta = cohort.groupby("individual")[list(FIELDS)].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
    if meta.isna().any().any():
        raise RuntimeError("a specimen has more than one value of a metadata field")
    comp = pd.read_csv(a.components).join(meta.astype(str), on="individual")
    verdict = pd.read_csv(a.per_class).set_index("taxon_group")[["verdict", "G2_tied_status"]]
    rows = []
    for (c, split), g in comp.groupby(["taxon_group", "fold0"]):
        row = {"taxon_group": c, "split": split, "specimens": len(g),
               "datasets": json.dumps(g.dataset.value_counts().sort_index().to_dict()),
               "component_sizes": json.dumps(g.g2_tied_component.value_counts().sort_index().to_dict())}
        for f in FIELDS:
            row[f"ari_{f}"] = adjusted_rand_score(g[f], g.g2_tied_component) if g[f].nunique() > 1 else None
        rows.append(row)
    t = pd.DataFrame(rows).join(verdict, on="taxon_group")
    t.to_csv(a.out / "component_metadata.csv", index=False)
    summary = {}
    for split, g in t.groupby("split"):
        summary[split] = {f: {"classes_with_field_varying": int(g[f"ari_{f}"].notna().sum()),
                              "ari_median": float(g[f"ari_{f}"].median()),
                              "classes_ari_at_least_0.5": int((g[f"ari_{f}"] >= 0.5).sum()),
                              "classes_ari_at_least_0.5_list": sorted(g.taxon_group[g[f"ari_{f}"] >= 0.5])} for f in FIELDS}
    datasets = comp[comp.fold0 == "train"].groupby("taxon_group").dataset.nunique()
    summary["train_classes_by_number_of_datasets"] = datasets.value_counts().sort_index().to_dict()
    tr = t[t.split == "train"]
    summary["train_dataset_ari_at_least_0.5_by_verdict"] = {
        v: {"classes": len(g), "dataset_ari_at_least_0.5": int((g.ari_dataset >= 0.5).sum())} for v, g in tr.groupby("verdict")}
    (a.out / "component_metadata.json").write_text(json.dumps(summary, indent=1, default=str))
    cols = ["taxon_group", "n_train", "n_val", "rms_radius", "top1_share", "effective_rank", "participation_ratio",
            "camera_reid", "G2_tied_status", "G2_tied_smallest_component", "G2_tied-G1_full", "G2_tied-G1_full_ci95",
            "G2_full_status", "G2_full-G1_full", "G3_tied_status", "G3_tied-G1_full", "G2_diag-G1_diag",
            "G2_tied_split_ari_val_median", "G2_tied_both_views_same_component_train", "verdict", "nearest_by_dprime_val", "dprime_val"]
    compact = pd.read_csv(a.per_class)[cols]
    for split in ("train", "val"):
        compact = compact.merge(t[t.split == split][["taxon_group", "ari_dataset", "ari_year"]].rename(
            columns={"ari_dataset": f"ari_dataset_{split}", "ari_year": f"ari_year_{split}"}), on="taxon_group", how="left")
    compact.round(3).to_csv(a.out / "per_class_summary.csv", index=False)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
