"""Do the 2-component tied-mixture groups of each class follow acquisition metadata? (read-only, after
class_geometry_pilot.py)

Per class, PCA dimension and split (train components from the train fit; val components predicted by the same train
fit):
- for each categorical field in ``--fields`` (default ``dataset year site``; AquaMonitor), the adjusted Rand index
  (ARI) between the component labels and the field; near 1 = the split reproduces that field, near 0 = unrelated.
  Only classes whose specimens span at least two values of a field get an ARI for it;
- for each numeric field in ``--numeric`` (specimen mean of the image values), the AUC of the field for telling
  component 1 from component 0; 0.5 = unrelated, near 0 or 1 = the components differ in that field. ``log_image_area``
  is derived as log(width * height) of each image (FIN-Benthic2 crops: a rough proxy for specimen size).
Also writes per_class_summary.csv: the compact per-class table, one row per class and PCA dimension (geometry,
model comparison, verdict, nearest class, component-metadata scores).

Usage: component_metadata.py --cohort cohort.parquet --components specimen_components.csv
       --modality per_class_modality.csv --geometry per_class_geometry.csv --out DIR [--fields F ...] [--numeric N ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, roc_auc_score

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
    ap.add_argument("--fields", nargs="*", default=["dataset", "year", "site"])
    ap.add_argument("--numeric", nargs="*", default=[])
    a = ap.parse_args()
    cohort = pd.read_parquet(a.cohort)
    if "log_image_area" in a.numeric:
        cohort["log_image_area"] = np.log(cohort.width * cohort.height)
    meta = (cohort.groupby("individual")[a.fields].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None) if a.fields
            else pd.DataFrame(index=pd.Index(cohort.individual.unique(), name="individual")))
    if meta.isna().any().any():
        raise RuntimeError("a specimen has more than one value of a metadata field")
    comp = pd.read_csv(a.components).join(meta.astype(str), on="individual")
    comp = comp.join(cohort.groupby("individual")[a.numeric].mean(), on="individual")
    mod = pd.read_csv(a.modality)
    rows = []
    for (c, d, split), g in comp.groupby(["taxon_group", "pcs", "fold0"]):
        row = {"taxon_group": c, "pcs": d, "split": split, "specimens": len(g),
               "component_sizes": json.dumps(g.g2_tied_component.value_counts().sort_index().to_dict())}
        if "dataset" in a.fields:
            row["datasets"] = json.dumps(g.dataset.value_counts().sort_index().to_dict())
        for f in a.fields:
            row[f"ari_{f}"] = adjusted_rand_score(g[f], g.g2_tied_component) if g[f].nunique() > 1 else None
        for f in a.numeric:
            both = g.g2_tied_component.nunique() == 2
            row[f"auc_{f}"] = roc_auc_score(g.g2_tied_component == 1, g[f]) if both else None
        rows.append(row)
    t = pd.DataFrame(rows).merge(mod[["taxon_group", "pcs", "verdict", "G2_tied_status"]], on=["taxon_group", "pcs"])
    t.to_csv(a.out / "component_metadata.csv", index=False)
    summary = {}
    for (d, split), g in t.groupby(["pcs", "split"]):
        s = {f: {"classes_with_field_varying": int(g[f"ari_{f}"].notna().sum()),
                 "ari_median": float(g[f"ari_{f}"].median()),
                 "classes_ari_at_least_0.5": int((g[f"ari_{f}"] >= 0.5).sum()),
                 "classes_ari_at_least_0.5_list": sorted(g.taxon_group[g[f"ari_{f}"] >= 0.5])} for f in a.fields}
        for f in a.numeric:
            sep = (g[f"auc_{f}"] - 0.5).abs()
            s[f] = {"classes_with_two_components": int(g[f"auc_{f}"].notna().sum()),
                    "abs_auc_minus_half_median": float(sep.median()),
                    "classes_auc_outside_0.2_0.8": int((sep >= 0.3).sum()),
                    "classes_auc_outside_0.2_0.8_list": sorted(g.taxon_group[sep >= 0.3])}
        summary[f"{d}_pcs_{split}"] = s
    if "dataset" in a.fields:
        for d, g in t[t.split == "train"].groupby("pcs"):
            summary[f"{d}_pcs_train_dataset_ari_at_least_0.5_by_verdict"] = {
                v: {"classes": len(h), "dataset_ari_at_least_0.5": int((h.ari_dataset >= 0.5).sum())} for v, h in g.groupby("verdict")}
        datasets = comp[comp.fold0 == "train"].groupby("taxon_group").dataset.nunique()
        summary["train_classes_by_number_of_datasets"] = datasets.value_counts().sort_index().to_dict()
    (a.out / "component_metadata.json").write_text(json.dumps(summary, indent=1, default=str))
    geometry = pd.read_csv(a.geometry)
    compact = geometry[[c for c in GEOMETRY if c in geometry]].merge(mod[[c for c in MODALITY if c in mod]], on="taxon_group")
    scores = [f"ari_{f}" for f in a.fields if f in ("dataset", "year")] + [f"auc_{f}" for f in a.numeric]
    for split in ("train", "val"):
        compact = compact.merge(t[t.split == split][["taxon_group", "pcs", *scores]].rename(
            columns={s: f"{s}_{split}" for s in scores}), on=["taxon_group", "pcs"], how="left")
    compact.round(3).to_csv(a.out / "per_class_summary.csv", index=False)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
