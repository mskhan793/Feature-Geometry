"""Summary of the 8 x 5 geometry replication (task.md a82e89e + 3b897b0), read-only: the established indicators of
every analysis output under ROOT/<encoder>/<dataset>/<partition>/<analysis> in one long table, a completion matrix
and per-encoder x dataset aggregates over partitions.

Indicators (the headline quantities of the earlier tasks, unchanged definitions; nothing recomputed from raw data):
- geometry (PCs 8, 16): class-macro held-out log-likelihood differences G2_tied / G2_full / G3_tied / GD_tied minus
  G1_full (mean, 95 % interval, classes, classes with interval above / below 0; primary = reliable fits), the
  per-class label counts, units, classes, PCA variance; native within-class medians (effective rank at n = 30, top-1
  share) and between-class pair counts (d' val < 1, < 2) and Spearman(d' val, radius separation);
- overlap (PCs 64, 8, 16) per covariance form: Spearman(d' val, Bhattacharyya distance), median covariance-term share,
  valid pairs; paper vs code-release spherical Spearman;
- fda (PCA vs FDA, d 8 16 32; spherical form): Spearman(distance, d' val), train-to-val Spearman, val/train GBC ratio
  (optimism), FDA alpha;
- lw (PCs 64, 16): per-class estimator wins (Ledoit-Wolf vs sample covariance), mean held-out log-likelihood of LW,
  sample, diagonal, spherical; pooled LW alpha and the best grid alpha;
- coupled / pooling (each space): contrasts macro mean + interval, micro mean + interval, classes better / interval
  above / below 0, numerical failures;
- acquisition (geometry component_metadata, FIN-Benthic2 / AquaMonitor): per field ARI median and classes with ARI >= 0.5.
A task is "done" if its DONE marker and its summary file exist.

Usage: summarise_replication.py --root RUNS --tasks TASKS.tsv ... --out DIR [--encoders E ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY = {"geometry": "results.json", "overlap": "summary.json", "fda": "summary.json", "lw": "summary.json",
           "coupled": "summary.json", "pooling": "summary.json"}
ANALYSES = list(SUMMARY)


def rows_geometry(r: dict) -> list:
    out = [("units_train", "", r["specimens"]["train"]), ("units_val", "", r["specimens"]["val"]),
           ("classes_eligible", "", r["classes"]), ("classes_insufficient", "", len(r["classes_insufficient_support"]))]
    w = r["within_class_native"]
    out += [("within_effective_rank_n30_median", "native", w["effective_rank_n30"][1]),
            ("within_top1_share_median", "native", w["top1_share"][1]),
            ("within_participation_ratio_median", "native", w["participation_ratio"][1])]
    b = r["between_classes"]
    out += [("pairs", "native", b["pairs"]), ("pairs_dprime_val_below_1", "native", b["dprime_val_below_1"]),
            ("pairs_dprime_val_below_2", "native", b["dprime_val_below_2"]),
            ("spearman_dprime_val_vs_radius", "native", b["spearman_dprime_val_vs_radius_native"])]
    for d, m in r["modality"]["by_pcs"].items():
        s = f"pcs{d}"
        out.append(("pca_explained_variance", s, r["pca_explained_variance"][d]))
        for key, v in m["differences"].items():
            out += [(f"{key}_macro", s, v.get("mean"), *(v.get("ci95") or [np.nan, np.nan])),
                    (f"{key}_classes", s, v.get("classes")), (f"{key}_classes_ci_above_0", s, v.get("classes_ci_above_0")),
                    (f"{key}_classes_ci_below_0", s, v.get("classes_ci_below_0"))]
        for label, n in m["verdicts"].items():
            out.append((f"label_{label}", s, n))
        out.append(("classes_analysed", s, sum(m["verdicts"].values())))
    return out


def rows_metadata(r: dict) -> list:
    out = []
    for key, fields in r.items():
        if not isinstance(fields, dict):
            continue
        for field, v in fields.items():
            if isinstance(v, dict) and "ari_median" in v:
                out += [(f"acq_{field}_ari_median", key, v["ari_median"]),
                        (f"acq_{field}_classes_ari_ge_0.5", key, v.get("classes_ari_at_least_0.5"))]
    return out


def rows_overlap(r: dict) -> list:
    out = [("classes", "", r["classes"]), ("pairs", "", r["pairs"])]
    for d, s in r["by_pcs"].items():
        for k in ("full", "diagonal", "spherical"):
            v = s[k]
            out += [(f"{k}_pairs_valid", f"pcs{d}", v.get("pairs_valid")),
                    (f"{k}_spearman_dprime_vs_distance", f"pcs{d}", v.get("spearman_dprime_val_vs_distance")),
                    (f"{k}_cov_term_share_median", f"pcs{d}", v.get("cov_term_share_median"))]
        out.append(("code_vs_paper_spherical_spearman_bc", f"pcs{d}", s["code_vs_paper_spherical"]["spearman_bc"]))
    return out


def rows_fda(r: dict) -> list:
    out = [("classes_gbc", "", r["classes_gbc"]), ("fda_rank", "", r["fda"]["main"]["rank"]),
           ("fda_alpha", "", r["fda"]["main"]["alpha"]), ("fda_alpha_ledoit_wolf", "", r["fda"]["main"]["alpha_ledoit_wolf"])]
    for arm, v in r["arms"].items():
        if "_" in arm and not arm.startswith("pca"):
            continue  # sensitivity arms (raw, shrink) stay in the per-run files
        s = v["spherical"]
        ratio = s["gbc_val"] / s["gbc_train"] if s.get("gbc_val") is not None and s.get("gbc_train") else np.nan
        out += [("spherical_spearman_distance_vs_dprime_val", arm, s.get("spearman_distance_vs_dprime_val")),
                ("spherical_spearman_distance_train_vs_val", arm, s.get("spearman_distance_train_vs_val")),
                ("spherical_gbc_val_over_train", arm, ratio)]
    return out


def rows_lw(r: dict) -> list:
    ref = r["reference"]
    out = [("pooled_alpha_lw", "native", ref["alpha_lw"]), ("pooled_best_grid_alpha_macro", "native", ref["best_grid_alpha_macro"])]
    for d, v in r["per_class"].items():
        s = f"pcs{d}"
        out += [("classes", s, v["classes"]), ("lw_wins", s, v["best_estimator_counts"].get("lw", 0)),
                ("sample_wins", s, v["best_estimator_counts"].get("sample", 0)),
                ("diagonal_wins", s, v["best_estimator_counts"].get("diagonal", 0)),
                ("spherical_wins", s, v["best_estimator_counts"].get("spherical", 0)),
                *[(f"mean_val_loglik_{k}", s, x) for k, x in v["mean_val_loglik"].items()]]
    return out


def rows_contrasts(r: dict) -> list:
    out = []
    for space, v in r["spaces"].items():
        for key, groups in v["contrasts"].items():
            g = groups["all"]
            if not g.get("classes"):
                continue
            out += [(f"{key}_macro", space, g["macro_mean"], *g["macro_ci95"]), (f"{key}_micro", space, g["micro_mean"], *g["micro_ci95"]),
                    (f"{key}_classes", space, g["classes"]), (f"{key}_classes_better", space, g["classes_better"]),
                    (f"{key}_classes_ci_above_0", space, g["classes_ci_above_0"]),
                    (f"{key}_classes_ci_below_0", space, g["classes_ci_below_0"]),
                    (f"{key}_numerical_failures", space, len(groups["numerical_failures"]))]
    return out


ROWS = {"geometry": rows_geometry, "overlap": rows_overlap, "fda": rows_fda, "lw": rows_lw, "coupled": rows_contrasts,
        "pooling": rows_contrasts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--tasks", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    tasks = pd.concat([pd.read_csv(t, sep="\t") for t in a.tasks]).drop_duplicates(["encoder", "dataset", "partition", "analysis"])
    long, status = [], []
    for t in tasks.itertuples():
        d = a.root / t.encoder / t.dataset / t.partition / t.analysis
        f = d / SUMMARY[t.analysis]
        done = (d / "DONE").exists() and f.exists()
        status.append({"encoder": t.encoder, "dataset": t.dataset, "partition": t.partition, "analysis": t.analysis,
                       "status": "done" if done else ("failed" if d.exists() else "not run")})
        if not done:
            continue
        r = json.loads(f.read_text())
        rows = ROWS[t.analysis](r)
        if t.analysis == "geometry" and (d / "component_metadata.json").exists():
            rows += rows_metadata(json.loads((d / "component_metadata.json").read_text()))
        for row in rows:
            ind, setting, value, *ci = row
            long.append({"encoder": t.encoder, "dataset": t.dataset, "partition": t.partition, "analysis": t.analysis,
                         "indicator": ind, "setting": setting, "value": value,
                         "ci_low": ci[0] if ci else np.nan, "ci_high": ci[1] if ci else np.nan})
    a.out.mkdir(parents=True, exist_ok=True)
    lt, st = pd.DataFrame(long), pd.DataFrame(status)
    lt.to_csv(a.out / "indicators_long.csv", index=False)
    st.to_csv(a.out / "task_status.csv", index=False)
    comp = st.groupby(["encoder", "dataset", "analysis"]).status.agg(
        lambda s: "done" if (s == "done").all() else f"{(s == 'done').sum()}/{len(s)} done").unstack()
    comp.to_csv(a.out / "completion_by_analysis.csv")
    num = lt.assign(value=pd.to_numeric(lt.value, errors="coerce"))
    agg = num.groupby(["encoder", "dataset", "analysis", "indicator", "setting"]).agg(
        partitions=("value", "size"), mean=("value", "mean"), min=("value", "min"), max=("value", "max"),
        partitions_ci_above_0=("ci_low", lambda x: int((x > 0).sum())),
        partitions_ci_below_0=("ci_high", lambda x: int((x < 0).sum()))).reset_index()
    agg.to_csv(a.out / "indicators_by_combination.csv", index=False)
    print(st.status.value_counts().to_dict())
    print(comp.to_string())


if __name__ == "__main__":
    main()
