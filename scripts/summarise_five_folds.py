"""Five-fold train->validation summary of class_geometry_pilot.py + component_metadata.py runs, per dataset, and a
cross-dataset table (task.md 92a605a, output B). Read-only.

Folds share specimens, so the five fold results are correlated repeats, not independent samples: each quantity is
reported per fold and as fold-level mean / min / max, never pooled over folds as if independent. Averages of interval
endpoints over folds are descriptive, not a confidence interval for the fold mean.

Per fold, the model differences are the primary class-macro values (classes where the model and G1_full are reliable)
with the number of classes included and excluded as unreliable; the all-fits values are kept as diagnostics
(``*_allfits``), and the common-classes values compare models on the classes where all of them are reliable
(``common_*``). Per class: the label in each fold (``not analysed`` where the class did not meet the support rule)
and its train / val specimens. Per class pair: d' (val) in each fold and the number of folds in which the pair was
measured. A baseline run (the earlier capped fold-0 pilot) is shown next to fold 0 when its files have the current
format.

``--dataset NAME RUNS_DIR BASELINE_DIR FIELD`` (repeatable): RUNS_DIR holds fold0..fold4 run folders; FIELD is the
component-metadata association shown (``dataset`` = ARI with the source dataset, ``log_image_area`` = AUC of image
size), counted separately for train and val specimens. Writes ``<NAME>/{per_fold.csv, per_class_labels.csv,
class_pairs_by_fold.csv, summary.json}`` and ``comparison.md`` to ``--out``.

Usage: summarise_five_folds.py --dataset FIN-Benthic2 RUNS BASE log_image_area --dataset AquaMonitor RUNS BASE dataset
       --out DIR
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.statistics.class_modality import INCONCLUSIVE, INSUFFICIENT, SUPPORTED, WORSE

FOLDS = [f"fold{k}" for k in range(5)]
KEYS = ["G2_tied-G1_full", "G3_tied-G1_full", "G2_full-G1_full", "G3_full-G1_full", "GD_tied-G1_full"]
LABELS = [SUPPORTED, INCONCLUSIVE, WORSE, INSUFFICIENT]
NOT_ANALYSED = "not analysed"


def run_row(d: Path, field: str) -> dict:
    r = json.loads((d / "results.json").read_text())
    pairs = pd.read_csv(d / "class_pairs.csv")
    row = {"classes": r["classes"], "classes_insufficient": len(r["classes_insufficient_support"]),
           "specimens_train": r["specimens"]["train"], "specimens_val": r["specimens"]["val"],
           "pairs_dprime_val_below_2_share": float((pairs.dprime_val < 2).mean()),
           "pairs_dprime_val_median": float(pairs.dprime_val.median()),
           "effective_rank_median": r["within_class_native"]["effective_rank"][1],
           "effective_rank_n30_median": r["within_class_native"]["effective_rank_n30"][1],
           "effective_rank_n30_classes": r["within_class_n30_classes"]}
    meta = json.loads((d / "component_metadata.json").read_text()) if (d / "component_metadata.json").exists() else {}
    for p, s in r["modality"]["by_pcs"].items():
        n = sum(s["verdicts"].values())
        row[f"pcs{p}_classes_fitted"] = n
        for v in LABELS:
            row[f"pcs{p}_{v}"] = s["verdicts"].get(v, 0)
        row[f"pcs{p}_split_stable"] = s["G2_tied_split_stable_classes"]
        row[f"pcs{p}_both_views_same_component_median"] = s["G2_tied_both_views_same_component_median"]
        row[f"pcs{p}_other_mixture_above_0_classes"] = len(s["other_mixtures_ci_above_0_classes"])
        common = s["common_reliable_classes"]
        row[f"pcs{p}_common_classes"] = common["classes"]
        for k in KEYS:
            v = s["differences"].get(k)
            if v is None:
                continue
            row[f"pcs{p}_{k}_classes_included"] = v["classes"]
            row[f"pcs{p}_{k}_classes_excluded_unreliable"] = len(v["classes_excluded_unreliable"])
            if v["classes"]:
                row[f"pcs{p}_{k}"], row[f"pcs{p}_{k}_ci_lo"], row[f"pcs{p}_{k}_ci_hi"] = v["mean"], *v["ci95"]
            row[f"pcs{p}_{k}_allfits"] = v["all_fits_diagnostic"].get("mean")
            c = common["differences"].get(k, {})
            if c.get("classes"):
                row[f"pcs{p}_common_{k}"] = c["mean"]
        for split in ("train", "val"):
            m = meta.get(f"{p}_pcs_{split}", {}).get(field)
            if m is None:
                continue
            listed = m.get("classes_ari_at_least_0.5_list", m.get("classes_auc_outside_0.2_0.8_list"))
            supported = set(s["verdict_classes"].get(SUPPORTED, []))
            row[f"pcs{p}_{split}_{field}_associated"] = len(listed)
            row[f"pcs{p}_{split}_supported_and_{field}_associated"] = len(supported & set(listed))
    return row


def summarise(name: str, runs: Path, baseline: Path, field: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    per_fold = pd.DataFrame({f: run_row(runs / f, field) for f in FOLDS}).T
    base = json.loads((baseline / "results.json").read_text())
    if "within_class_n30_classes" in base:  # baseline produced by the current code
        per_fold.loc["baseline_fold0_capped"] = pd.Series(run_row(baseline, field))
    per_fold.to_csv(out / "per_fold.csv")
    folds = per_fold.loc[FOLDS].astype(float)
    across = {c: {"mean": float(folds[c].mean()), "min": float(folds[c].min()), "max": float(folds[c].max()),
                  "folds": int(folds[c].notna().sum())} for c in folds.columns}
    ci_above = {c: int((folds[f"{c}_ci_lo"] > 0).sum()) for c in [k.removesuffix("_ci_lo") for k in folds if k.endswith("_ci_lo")]}
    ci_below = {c: int((folds[f"{c}_ci_hi"] < 0).sum()) for c in ci_above}

    labels = []
    for f in FOLDS:
        m = pd.read_csv(runs / f / "per_class_modality.csv")[["taxon_group", "pcs", "verdict", "n_train", "n_val"]]
        labels.append(m.assign(fold=f))
    labels = pd.concat(labels)
    v = labels.pivot_table(index=["taxon_group", "pcs"], columns="fold", values="verdict", aggfunc="first")
    v = v.reindex(columns=FOLDS).fillna(NOT_ANALYSED)
    for col in ("n_train", "n_val"):
        n = labels.pivot_table(index=["taxon_group", "pcs"], columns="fold", values=col, aggfunc="first")
        v = v.join(n.reindex(columns=FOLDS).add_prefix(f"{col}_"))
    v["folds_supported"] = (v[FOLDS] == SUPPORTED).sum(axis=1)
    v["folds_single_gaussian_better"] = (v[FOLDS] == WORSE).sum(axis=1)
    v["folds_analysed"] = (v[FOLDS] != NOT_ANALYSED).sum(axis=1)
    v.reset_index().to_csv(out / "per_class_labels.csv", index=False)

    pairs = []
    for f in FOLDS:
        p = pd.read_csv(runs / f / "class_pairs.csv")
        p["pair"] = [" | ".join(sorted(t)) for t in zip(p.class_a, p.class_b)]
        pairs.append(p.set_index("pair").dprime_val.rename(f))
    pairs = pd.concat(pairs, axis=1)
    complete = pairs.dropna()
    rho = [spearmanr(complete[a], complete[b])[0] for a, b in itertools.combinations(FOLDS, 2)]
    pairs["folds_available"] = pairs[FOLDS].notna().sum(1)
    pairs["mean"], pairs["min"], pairs["max"] = pairs[FOLDS].mean(1), pairs[FOLDS].min(1), pairs[FOLDS].max(1)
    pairs["folds_below_2"] = (pairs[FOLDS] < 2).sum(1)
    pairs.sort_values("mean").to_csv(out / "class_pairs_by_fold.csv")
    nearest = {}
    for f in FOLDS:
        for key, val in pairs[f].dropna().items():
            for c in key.split(" | "):
                if c not in nearest.setdefault(f, {}) or val < nearest[f][c][1]:
                    nearest[f][c] = (key, val)
    classes = sorted(set.union(*[set(n) for n in nearest.values()]))
    same = [max(pd.Series([nearest[f][c][0] for f in FOLDS if c in nearest[f]]).value_counts()) for c in classes]
    in_all = [c for c in classes if all(c in nearest[f] for f in FOLDS)]

    by_d = {str(d): v.xs(d, level="pcs") for d in sorted(v.index.get_level_values("pcs").unique())}
    summary = {
        "dataset": name, "note": "five folds share specimens: fold results are correlated repeats; averaged interval "
        "endpoints are descriptive",
        "across_folds": across, "folds_ci_above_0": ci_above, "folds_ci_below_0": ci_below,
        "classes_supported_in_k_folds": {d: t.folds_supported.value_counts().sort_index().to_dict() for d, t in by_d.items()},
        "classes_supported_in_all_folds": {d: sorted(t.index[t.folds_supported == 5]) for d, t in by_d.items()},
        "classes_analysed_in_k_folds": {d: t.folds_analysed.value_counts().sort_index().to_dict() for d, t in by_d.items()},
        "pairs": {"measured_in_all_folds": len(complete), "measured_in_some_fold": len(pairs),
                  "spearman_dprime_val_between_folds_min_median_max": [float(min(rho)), float(pd.Series(rho).median()),
                                                                       float(max(rho))],
                  "below_2_in_all_folds": int((pairs.folds_below_2 == 5).sum()),
                  "below_2_in_some_fold": int((pairs.folds_below_2 > 0).sum()),
                  "closest_by_mean": json.loads(pairs.nsmallest(10, "mean")[["mean", "min", "max", "folds_below_2",
                                                                             "folds_available"]].reset_index().to_json(orient="records"))},
        "classes_same_nearest_class_in_all_folds": int(sum(s == 5 for c, s in zip(classes, same) if c in in_all)),
        "classes_with_nearest_class_in_all_folds": len(in_all),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs=4, action="append", required=True, metavar=("NAME", "RUNS", "BASELINE", "FIELD"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    s = {n: summarise(n, Path(r), Path(b), f, a.out / n) for n, r, b, f in a.dataset}
    keys = sorted(set.union(*[set(v["across_folds"]) for v in s.values()]))
    lines = ["Fold mean [min, max] over the five folds (correlated repeats). Model differences: primary = classes where the "
             "model and G1_full are reliable (`*_classes_included`, `*_classes_excluded_unreliable`); `*_allfits` = "
             "diagnostic over all fitted comparisons; `common_*` = classes where all models are reliable. `*_ci_lo/hi` "
             "are fold averages of interval endpoints (descriptive).", "",
             "| quantity | " + " | ".join(s) + " |", "|---|" + "---|" * len(s)]
    for k in keys:
        cells = []
        for v in s.values():
            q = v["across_folds"].get(k)
            cells.append("—" if q is None or q["folds"] == 0 else f"{q['mean']:.3f} [{q['min']:.3f}, {q['max']:.3f}]")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "| folds with primary 95 % interval above 0 / below 0 | " + " | ".join(s) + " |", "|---|" + "---|" * len(s)]
    for k in sorted(set.union(*[set(v["folds_ci_above_0"]) for v in s.values()])):
        lines.append(f"| {k} | " + " | ".join(
            f"{v['folds_ci_above_0'].get(k, '—')} / {v['folds_ci_below_0'].get(k, '—')}" for v in s.values()) + " |")
    (a.out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
