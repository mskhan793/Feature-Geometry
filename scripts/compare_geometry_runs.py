"""Compare the expanded-cohort geometry run with the earlier 30-specimen run (read-only, after both geometry stages
and their component_metadata stages).

1. Verdicts per class: old (8 PCs, 30 train specimens) vs new at each PCA dimension, and the transition counts
   ({old verdict: {new verdict: classes}}).
2. Class pairs: old top-10 closest pairs (d' on val) with their new d' and rank, and the reverse; overlap of the two
   top-10 sets; Spearman correlation of d' (val) over all pairs present in both runs.
3. Stability of class geometry (native space, train): old vs new centroid distance divided by the new class RMS
   radius, with the sampling reference radius * sqrt(1/30 + 1/n_new) for two independent samples of these sizes; old
   vs new top-5 principal-subspace overlap, with the reference overlap between the new first-30 specimens (sha1 order)
   and all new specimens. The old embeddings used 3 hash-chosen frames per camera, the new ones up to 10 spread frames.
4. Within-class effective rank: old (n = 30) vs new on its first 30 specimens and on all.
5. Dataset alignment of the 2-component split (train ARI >= 0.5), old vs new at 8 PCs.

Usage: compare_geometry_runs.py --old OLD_GEOMETRY_DIR --old-cohort C --old-cache DIR --new NEW_GEOMETRY_DIR
       --new-cohort C --new-caches MANIFEST=CACHE ... --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.expanded_cohort import stable_order
from src.features.feature_lookup import features_by_name
from src.features.specimen_means import specimen_camera_means, specimen_means
from src.statistics.class_geometry import principal_axes

SEED, K = 20261107, 5


def train_specimens(cohort_path: Path, pairs) -> tuple[pd.DataFrame, np.ndarray]:
    cohort = pd.read_parquet(cohort_path)
    x, _ = features_by_name(pairs, cohort.img.tolist())
    keys, means, _ = specimen_camera_means(x, cohort)
    sk, s = specimen_means(keys, means)
    tr = (sk.fold0 == "train").values
    return sk[tr].reset_index(drop=True), s[tr]


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = principal_axes(a)[1][:, :K], principal_axes(b)[1][:, :K]
    return float(((ua.T @ ub) ** 2).sum() / K)


def pair_key(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: " | ".join(sorted([r.class_a, r.class_b])), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, type=Path)
    ap.add_argument("--old-cohort", required=True, type=Path)
    ap.add_argument("--old-cache", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--new-cohort", required=True, type=Path)
    ap.add_argument("--new-caches", required=True, nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(a.old / "per_class.csv").set_index("taxon_group")
    new = pd.read_csv(a.new / "per_class_modality.csv")
    verdicts = old[["n_train", "verdict"]].rename(columns={"n_train": "old_n_train", "verdict": "old_verdict_8pcs"})
    for d, g in new.groupby("pcs"):
        verdicts = verdicts.join(g.set_index("taxon_group")[["verdict"]].rename(columns={"verdict": f"new_verdict_{d}pcs"}), how="outer")
    verdicts = verdicts.join(new.groupby("taxon_group").n_train.first().rename("new_n_train"), how="outer")
    verdicts.reset_index().to_csv(a.out / "verdicts.csv", index=False)
    transitions = {f"{d}pcs": pd.crosstab(verdicts.old_verdict_8pcs, verdicts[f"new_verdict_{d}pcs"]).to_dict(orient="index")
                   for d in sorted(new.pcs.unique())}

    po, pn = pd.read_csv(a.old / "class_pairs.csv"), pd.read_csv(a.new / "class_pairs.csv")
    po["pair"], pn["pair"] = pair_key(po), pair_key(pn)
    po["rank"], pn["rank"] = po.dprime_val.rank(method="first"), pn.dprime_val.rank(method="first")
    both = po.merge(pn, on="pair", suffixes=("_old", "_new"))
    cols = ["pair", "dprime_val_old", "dprime_val_new", "dprime_train_old", "dprime_train_new", "rank_old", "rank_new"]
    old_top = both.nsmallest(10, "dprime_val_old")[cols]
    new_top = both.nsmallest(10, "dprime_val_new")[cols]
    pd.concat([old_top.assign(listed_by="old"), new_top.assign(listed_by="new")]).round(3).to_csv(a.out / "closest_pairs_old_vs_new.csv", index=False)

    so, xo = train_specimens(a.old_cohort, [(a.old_cohort, a.old_cache)])
    sn, xn = train_specimens(a.new_cohort, [tuple(Path(p) for p in s.split("=", 1)) for s in a.new_caches])
    rows = []
    for c in sorted(set(so.taxon_group) & set(sn.taxon_group)):
        o, n = xo[(so.taxon_group == c).values], xn[(sn.taxon_group == c).values]
        kn = sn[sn.taxon_group == c]
        first30 = n[np.argsort(stable_order(kn.individual, SEED).values)[:30]]
        radius = float(np.sqrt(((n - n.mean(0)) ** 2).sum(1).mean()))
        shift = float(np.linalg.norm(o.mean(0) - n.mean(0)))
        rows.append({"taxon_group": c, "n_old": len(o), "n_new": len(n), "centroid_shift_over_radius": shift / radius,
                     "sampling_reference": float(np.sqrt(1 / len(o) + 1 / len(n))),
                     "subspace_overlap_old_vs_new": overlap(o, n), "subspace_overlap_new30_vs_new_all": overlap(first30, n)})
    stab = pd.DataFrame(rows)
    stab["shift_over_reference"] = stab.centroid_shift_over_radius / stab.sampling_reference
    stab.round(4).to_csv(a.out / "class_stability.csv", index=False)

    geo_new = pd.read_csv(a.new / "per_class_geometry.csv").set_index("taxon_group")
    er = old[["effective_rank"]].rename(columns={"effective_rank": "old_n30"}).join(
        geo_new[["effective_rank_n30", "effective_rank", "n"]].rename(columns={"effective_rank_n30": "new_n30", "effective_rank": "new_all"}), how="inner")
    er.round(3).reset_index().to_csv(a.out / "effective_rank_old_vs_new.csv", index=False)

    cm_old = pd.read_csv(a.old / "component_metadata.csv")
    cm_new = pd.read_csv(a.new / "component_metadata.csv")
    ds_old = set(cm_old[(cm_old.split == "train") & (cm_old.ari_dataset >= 0.5)].taxon_group)
    ds_new = set(cm_new[(cm_new.split == "train") & (cm_new.pcs == 8) & (cm_new.ari_dataset >= 0.5)].taxon_group)

    result = {
        "verdict_transitions_old8_to_new": transitions,
        "closest_pairs": {"pairs_in_both": len(both), "top10_overlap": len(set(old_top.pair) & set(new_top.pair)),
                          "old_top10_still_top10": sorted(set(old_top.pair) & set(new_top.pair)),
                          "spearman_dprime_val": float(spearmanr(both.dprime_val_old, both.dprime_val_new)[0]),
                          "dprime_val_below_2": {"old": int((both.dprime_val_old < 2).sum()), "new": int((both.dprime_val_new < 2).sum())},
                          "median_dprime_train_minus_val": {"old": float((both.dprime_train_old - both.dprime_val_old).median()),
                                                            "new": float((both.dprime_train_new - both.dprime_val_new).median())}},
        "stability": {"classes": len(stab),
                      "shift_over_reference_median_max": [float(stab.shift_over_reference.median()), float(stab.shift_over_reference.max())],
                      "classes_shift_over_reference_above_2": sorted(stab.taxon_group[stab.shift_over_reference > 2]),
                      "subspace_overlap_old_vs_new_median": float(stab.subspace_overlap_old_vs_new.median()),
                      "subspace_overlap_new30_vs_new_all_median": float(stab.subspace_overlap_new30_vs_new_all.median())},
        "effective_rank_median": {"old_n30": float(er.old_n30.median()), "new_n30": float(er.new_n30.median()), "new_all": float(er.new_all.median())},
        "dataset_aligned_split_train_8pcs": {"old": len(ds_old), "new": len(ds_new), "both": sorted(ds_old & ds_new),
                                             "old_only": sorted(ds_old - ds_new), "new_only": sorted(ds_new - ds_old)},
    }
    (a.out / "compare.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
