"""Do the 2-component tied-mixture groups of each class follow acquisition metadata? (read-only, after
class_geometry_pilot.py)

Per class, PCA dimension and split (train components from the train fit; val components predicted by the same train
fit), the adjusted Rand index (ARI) between the component labels and each metadata field (``dataset``, ``year``,
``site``), plus the class's dataset composition. ARI near 1 = the split reproduces that field; near 0 = unrelated.
Only classes whose specimens span at least two values of a field get an ARI for it.

Also writes per_class_summary.csv: the compact per-class table, one row per class and PCA dimension (geometry,
model comparison, verdict, nearest class, component-metadata ARIs).

Usage: component_metadata.py --cohort cohort.parquet --components specimen_components.csv
       --modality per_class_modality.csv --geometry per_class_geometry.csv --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

FIELDS = ("dataset", "year", "site")
GEOMETRY = ["taxon_group", "n", "rms_radius", "top1_share", "effective_rank", "participation_ratio", "effective_rank_n30",
            "camera_reid", "paired_specimens", "frame_rms_within_specimen_median", "nearest_by_dprime_val", "dprime_val"]
MODALITY = ["taxon_group", "pcs", "n_train", "n_val", "G2_tied_status", "G2_tied_smallest_component", "G2_tied-G1_full",
            "G2_tied-G1_full_ci95", "G2_full_status", "G2_full-G1_full", "G3_tied_status", "G3_tied-G1_full",
            "GD_tied-G1_full", "GD_tied-G1_full_ci95", "G2_diag-G1_diag", "G2_tied_split_ari_val_median",
            "G2_tied_both_views_same_component_train", "verdict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--components", required=True, type=Path)
    ap.add_argument("--modality", required=True, type=Path)
    ap.add_argument("--geometry", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    cohort = pd.read_parquet(a.cohort)
    meta = cohort.groupby("individual")[list(FIELDS)].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
    if meta.isna().any().any():
        raise RuntimeError("a specimen has more than one value of a metadata field")
    comp = pd.read_csv(a.components).join(meta.astype(str), on="individual")
    mod = pd.read_csv(a.modality)
    rows = []
    for (c, d, split), g in comp.groupby(["taxon_group", "pcs", "fold0"]):
        row = {"taxon_group": c, "pcs": d, "split": split, "specimens": len(g),
               "datasets": json.dumps(g.dataset.value_counts().sort_index().to_dict()),
               "component_sizes": json.dumps(g.g2_tied_component.value_counts().sort_index().to_dict())}
        for f in FIELDS:
            row[f"ari_{f}"] = adjusted_rand_score(g[f], g.g2_tied_component) if g[f].nunique() > 1 else None
        rows.append(row)
    t = pd.DataFrame(rows).merge(mod[["taxon_group", "pcs", "verdict", "G2_tied_status"]], on=["taxon_group", "pcs"])
    t.to_csv(a.out / "component_metadata.csv", index=False)
    summary = {}
    for (d, split), g in t.groupby(["pcs", "split"]):
        summary[f"{d}_pcs_{split}"] = {f: {"classes_with_field_varying": int(g[f"ari_{f}"].notna().sum()),
                                           "ari_median": float(g[f"ari_{f}"].median()),
                                           "classes_ari_at_least_0.5": int((g[f"ari_{f}"] >= 0.5).sum()),
                                           "classes_ari_at_least_0.5_list": sorted(g.taxon_group[g[f"ari_{f}"] >= 0.5])}
                                       for f in FIELDS}
    for d, g in t[t.split == "train"].groupby("pcs"):
        summary[f"{d}_pcs_train_dataset_ari_at_least_0.5_by_verdict"] = {
            v: {"classes": len(h), "dataset_ari_at_least_0.5": int((h.ari_dataset >= 0.5).sum())} for v, h in g.groupby("verdict")}
    datasets = comp[comp.fold0 == "train"].groupby("taxon_group").dataset.nunique()
    summary["train_classes_by_number_of_datasets"] = datasets.value_counts().sort_index().to_dict()
    (a.out / "component_metadata.json").write_text(json.dumps(summary, indent=1, default=str))
    compact = pd.read_csv(a.geometry)[GEOMETRY].merge(mod[MODALITY], on="taxon_group")
    for split in ("train", "val"):
        compact = compact.merge(t[t.split == split][["taxon_group", "pcs", "ari_dataset", "ari_year"]].rename(
            columns={"ari_dataset": f"ari_dataset_{split}", "ari_year": f"ari_year_{split}"}), on=["taxon_group", "pcs"], how="left")
    compact.round(3).to_csv(a.out / "per_class_summary.csv", index=False)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
