"""Headline tables of the 8 x 5 geometry replication (task.md a82e89e + 3b897b0), read-only: one dataset x encoder
table per established headline indicator, from summarise_replication.py (indicators_by_combination.csv) and
separation_table.py (separation_by_combination.csv). Nothing is recomputed.

Cell = mean over partitions (five-fold datasets: the five folds, overlapping val sets, not independent replicates;
other datasets: the single official partition). For contrasts with intervals the cell adds ``(+a/-b)``: partitions
whose 95 % interval lies above / below 0. Settings follow the earlier tasks: mixtures at 8 and 16 PCs; overlap at
64 PCs (the paper's setting); FDA at d = 16; LW per-class at 64 PCs; coupled / pooling at 64 PCs and in native space
(``native`` for rank p - 1 encoders, ``native_all`` for full-rank encoders: the primary native space of each).

Writes OUT/headline_tables.md and OUT/headline_tables.csv (long).

Usage: headline_tables.py --indicators indicators_by_combination.csv --separation separation_by_combination.csv --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ENCODERS = ["dinov3_l", "dinov2_l", "bioclip2", "siglip2", "ijepa_h14"]
DATASETS = ["finbenthic2", "aquamonitor", "bioscan5m", "df24", "food101", "oxford_pets", "cifar100", "fgvc_aircraft"]
FULL_RANK = ("bioclip2", "siglip2")
NATIVE = "native*"  # resolved per encoder: native_all (full rank) or native (rank p - 1)

# (title, analysis, indicator, setting, number format)
TABLES = [
    ("Mixture: G2_tied - G1_full, class-macro held-out log-likelihood, 8 PCs (reliable fits)", "geometry", "G2_tied-G1_full_macro", "pcs8", ".3f"),
    ("Mixture: G2_tied - G1_full, 16 PCs", "geometry", "G2_tied-G1_full_macro", "pcs16", ".3f"),
    ("Mixture: G2_full - G1_full, 8 PCs", "geometry", "G2_full-G1_full_macro", "pcs8", ".3f"),
    ("Classes labelled 'mixture-supported split (exploratory)', 8 PCs", "geometry", "label_mixture-supported split (exploratory)", "pcs8", ".1f"),
    ("Classes analysed at 8 PCs", "geometry", "classes_analysed", "pcs8", ".1f"),
    ("Source-group model GD_tied - G1_full, 8 PCs (AquaMonitor only)", "geometry", "GD_tied-G1_full_macro", "pcs8", ".3f"),
    ("Within-class effective rank at n = 30, class median (native)", "geometry", "within_effective_rank_n30_median", "native", ".1f"),
    ("Within-class top-1 eigenvalue share, class median (native)", "geometry", "within_top1_share_median", "native", ".3f"),
    ("Spearman(d' val, radius separation), native", "geometry", "spearman_dprime_val_vs_radius", "native", ".3f"),
    ("Overlap: Spearman(d' val, spherical Bhattacharyya distance), 64 PCs", "overlap", "spherical_spearman_dprime_vs_distance", "pcs64", ".3f"),
    ("Overlap: Spearman(d' val, full Bhattacharyya distance), 64 PCs (pairs with >= 2d units)", "overlap", "full_spearman_dprime_vs_distance", "pcs64", ".3f"),
    ("Overlap: median covariance-term share, full form, 64 PCs", "overlap", "full_cov_term_share_median", "pcs64", ".3f"),
    ("Overlap: paper vs code-release spherical Spearman, 64 PCs", "overlap", "code_vs_paper_spherical_spearman_bc", "pcs64", ".3f"),
    ("PCA-GBC d = 16: Spearman(spherical distance, d' val)", "fda", "spherical_spearman_distance_vs_dprime_val", "pca16", ".3f"),
    ("FDA-GBC d = 16: Spearman(spherical distance, d' val)", "fda", "spherical_spearman_distance_vs_dprime_val", "fda16", ".3f"),
    ("PCA-GBC d = 16: val / train GBC ratio (optimism > 1)", "fda", "spherical_gbc_val_over_train", "pca16", ".3f"),
    ("FDA-GBC d = 16: val / train GBC ratio (optimism > 1)", "fda", "spherical_gbc_val_over_train", "fda16", ".3f"),
    ("LW per class, 64 PCs: classes where LW has the best held-out fit", "lw", "lw_wins", "pcs64", ".1f"),
    ("LW per class, 64 PCs: classes", "lw", "classes", "pcs64", ".1f"),
    ("Pooled LW alpha (native)", "lw", "pooled_alpha_lw", "native", ".2e"),
    ("Best grid alpha by macro val log-likelihood (native)", "lw", "pooled_best_grid_alpha_macro", "native", ".0e"),
    ("Coupled RSCM - per-class LW, class macro, 64 PCs", "coupled", "coupled_vs_lw_macro", "pca64", ".3f"),
    ("Coupled RSCM - per-class LW, class macro, native", "coupled", "coupled_vs_lw_macro", NATIVE, ".3f"),
    ("Pooling at LW shrinkage - per-class LW, class macro, native", "coupled", "pooling_at_lw_shrink_macro", NATIVE, ".3f"),
    ("Balanced - specimen-weighted pooling, class macro, 16 PCs", "pooling", "balancing_macro", "pca16", ".3f"),
    ("Balanced - specimen-weighted pooling, class macro, 64 PCs", "pooling", "balancing_macro", "pca64", ".3f"),
    ("Balanced pooled covariance - per-class LW, class macro, 64 PCs", "pooling", "pool_bal_vs_lw_macro", "pca64", ".3f"),
    ("Balanced pooled covariance - per-class LW, class macro, native", "pooling", "pool_bal_vs_lw_macro", NATIVE, ".3f"),
]
SEPARATION = [("Median d' (val units, train centroid direction), all eligible pairs", "dprime_median", ".2f"),
              ("Share of pairs with d' < 1", "share_dprime_below_1", ".4f"),
              ("Median nearest-class d'", "nearest_class_dprime_median", ".2f")]


def markdown(t: pd.DataFrame) -> str:
    """Plain markdown table (no tabulate dependency; the pinned environment does not include it)."""
    head = ["dataset", *map(str, t.columns)]
    rows = [[str(i), *map(str, r)] for i, r in zip(t.index, t.values)]
    return "\n".join("| " + " | ".join(r) + " |" for r in [head, ["---"] * len(head), *rows])


def cell(r, fmt: str) -> str:
    s = format(r["mean"], fmt)
    if r["ci"]:
        s += f" (+{int(r['partitions_ci_above_0'])}/-{int(r['partitions_ci_below_0'])})"
    return s


def comparisons(lt: pd.DataFrame) -> list:
    """Counts across the tables above (cells = dataset x encoder; each count's denominator is printed)."""
    def tab(title):
        return lt[lt.table == title].set_index(["dataset", "encoder"])

    out = ["## Comparisons across tables", ""]
    pca, fda = tab("PCA-GBC d = 16: Spearman(spherical distance, d' val)"), tab("FDA-GBC d = 16: Spearman(spherical distance, d' val)")
    j = pca.join(fda, lsuffix="_pca", rsuffix="_fda", how="inner")
    out.append(f"- FDA-GBC agrees better with d' than PCA-GBC (d = 16): {int((j.mean_fda > j.mean_pca).sum())} of {len(j)} cells")
    g2, gd = tab(TABLES[0][0]), tab(TABLES[5][0])
    j = gd.join(g2, lsuffix="_gd", rsuffix="_g2", how="inner")
    out.append(f"- AquaMonitor, 8 PCs: GD_tied gain > G2_tied gain for {int((j.mean_gd > j.mean_g2).sum())} of {len(j)} encoders")
    for title in (TABLES[0][0], TABLES[1][0], "Pooling at LW shrinkage - per-class LW, class macro, native",
                  "Coupled RSCM - per-class LW, class macro, native"):
        t = tab(title)
        up = (t.partitions_ci_above_0 == t.partitions).sum()
        down = (t.partitions_ci_below_0 == t.partitions).sum()
        out.append(f"- {title}: every partition's interval above 0 in {int(up)} of {len(t)} cells, below 0 in {int(down)}")
    for title in ("Median d' (val units, train centroid direction), all eligible pairs",
                  "Within-class effective rank at n = 30, class median (native)"):
        t = tab(title).reset_index()
        low = t.loc[t.groupby("dataset")["mean"].idxmin()]
        out.append(f"- {title}: I-JEPA lowest in {int((low.encoder == 'ijepa_h14').sum())} of {t.dataset.nunique()} datasets")
    return out + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indicators", required=True, type=Path)
    ap.add_argument("--separation", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    ind = pd.read_csv(a.indicators)
    sep = pd.read_csv(a.separation)
    md, long = ["# Geometry replication headline tables", "",
                "Mean over partitions; `(+a/-b)`: partitions with 95 % interval above / below 0 (contrasts only).", ""], []
    for title, analysis, indicator, setting, fmt in TABLES:
        rows = {}
        for e in ENCODERS:
            st = ("native_all" if e in FULL_RANK else "native") if setting == NATIVE else setting
            g = ind[(ind.analysis == analysis) & (ind.indicator == indicator) & (ind.setting == st) & (ind.encoder == e)]
            for r in g.to_dict("records"):
                r["ci"] = analysis in ("coupled", "pooling") or "-" in indicator  # contrasts with intervals
                rows.setdefault(r["dataset"], {})[e] = cell(r, fmt)
                long.append({"table": title, "dataset": r["dataset"], "encoder": e, "setting": st, "mean": r["mean"],
                             "min": r["min"], "max": r["max"], "partitions": r["partitions"],
                             "partitions_ci_above_0": r["partitions_ci_above_0"] if r["ci"] else "",
                             "partitions_ci_below_0": r["partitions_ci_below_0"] if r["ci"] else ""})
        t = pd.DataFrame(rows).T.reindex(index=[d for d in DATASETS if d in rows], columns=ENCODERS).fillna("n/a")
        md += [f"## {title}", "", markdown(t), ""]
    for title, col, fmt in SEPARATION:
        t = sep.pivot(index="dataset", columns="encoder", values=col).reindex(index=DATASETS, columns=ENCODERS)
        md += [f"## {title}", "", markdown(t.map(lambda x: format(x, fmt))), ""]
        for (d, e), v in t.stack().items():
            long.append({"table": title, "dataset": d, "encoder": e, "setting": "native", "mean": v})
    md += comparisons(pd.DataFrame(long))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "headline_tables.md").write_text("\n".join(md))
    pd.DataFrame(long).to_csv(a.out / "headline_tables.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
