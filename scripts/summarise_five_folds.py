"""Five-fold train->validation summary of class_geometry_pilot.py + component_metadata.py runs, per dataset, and a
cross-dataset table (task.md 92a605a, output B). Read-only.

Folds share specimens, so the five fold results are correlated repeats, not independent samples: each quantity is
reported per fold and as fold-level mean / min / max, never pooled over folds as if independent. Per class: verdict in
each fold. Per class pair: d' (val) in each fold. A baseline run (the earlier capped fold-0 pilot) is shown next to
fold 0 of the full-specimen run.

``--dataset NAME RUNS_DIR BASELINE_DIR FIELD`` (repeatable): RUNS_DIR holds fold0..fold4 run folders; FIELD is the
component-metadata association shown (``dataset`` = ARI with the source dataset, ``log_image_area`` = AUC of image
size). Writes ``<NAME>/{per_fold.csv, per_class_verdicts.csv, class_pairs_by_fold.csv, summary.json}`` and
``comparison.md`` to ``--out``.

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

FOLDS = [f"fold{k}" for k in range(5)]
KEYS = ["G2_tied-G1_full", "G3_tied-G1_full", "G2_full-G1_full", "G3_full-G1_full", "GD_tied-G1_full"]
VERDICTS = ["credible multiple groups", "inconclusive", "one Gaussian favoured", "insufficient evidence"]


def run_row(d: Path, field: str) -> dict:
    r = json.loads((d / "results.json").read_text())
    pairs = pd.read_csv(d / "class_pairs.csv")
    row = {"classes": r["classes"], "classes_insufficient": len(r["classes_insufficient_support"]),
           "specimens_train": r["specimens"]["train"], "specimens_val": r["specimens"]["val"],
           "pairs_dprime_val_below_2_share": float((pairs.dprime_val < 2).mean()),
           "pairs_dprime_val_median": float(pairs.dprime_val.median()),
           "effective_rank_median": r["within_class_native"]["effective_rank"][1],
           "effective_rank_n30_median": r["within_class_native"]["effective_rank_n30"][1]}
    meta = json.loads((d / "component_metadata.json").read_text()) if (d / "component_metadata.json").exists() else {}
    for p, s in r["modality"]["by_pcs"].items():
        n = sum(s["verdicts"].values())
        row[f"pcs{p}_classes_fitted"] = n
        for v in VERDICTS:
            row[f"pcs{p}_{v}"] = s["verdicts"].get(v, 0)
        row[f"pcs{p}_split_stable"] = s["G2_tied_split_stable_classes"]
        row[f"pcs{p}_both_views_same_component_median"] = s["G2_tied_both_views_same_component_median"]
        for k in KEYS:
            if k in s["differences"]:
                v = s["differences"][k]
                row[f"pcs{p}_{k}"], row[f"pcs{p}_{k}_ci_lo"], row[f"pcs{p}_{k}_ci_hi"] = v["mean"], *v["ci95"]
        m = meta.get(f"{p}_pcs_train", {}).get(field)
        if m is not None:
            listed = m.get("classes_ari_at_least_0.5_list", m.get("classes_auc_outside_0.2_0.8_list"))
            credible = set(s["verdict_classes"].get("credible multiple groups", []))
            row[f"pcs{p}_{field}_associated"] = len(listed)
            row[f"pcs{p}_credible_and_{field}_associated"] = len(credible & set(listed))
    return row


def summarise(name: str, runs: Path, baseline: Path, field: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    per_fold = pd.DataFrame({f: run_row(runs / f, field) for f in FOLDS}).T
    per_fold.loc["baseline_fold0_capped"] = pd.Series(run_row(baseline, field))
    per_fold.to_csv(out / "per_fold.csv")
    folds = per_fold.loc[FOLDS].astype(float)
    across = {c: {"mean": float(folds[c].mean()), "min": float(folds[c].min()), "max": float(folds[c].max())}
              for c in folds.columns}
    ci_above = {c: int((folds[f"{c}_ci_lo"] > 0).sum()) for c in [k.removesuffix("_ci_lo") for k in folds if k.endswith("_ci_lo")]}
    ci_below = {c: int((folds[f"{c}_ci_hi"] < 0).sum()) for c in ci_above}

    verdicts = []
    for f in FOLDS:
        m = pd.read_csv(runs / f / "per_class_modality.csv")[["taxon_group", "pcs", "verdict"]]
        verdicts.append(m.assign(fold=f))
    v = pd.concat(verdicts).pivot_table(index=["taxon_group", "pcs"], columns="fold", values="verdict", aggfunc="first")
    v["folds_credible"] = (v[FOLDS] == "credible multiple groups").sum(axis=1)
    v["folds_one_gaussian"] = (v[FOLDS] == "one Gaussian favoured").sum(axis=1)
    v.reset_index().to_csv(out / "per_class_verdicts.csv", index=False)

    pairs = []
    for f in FOLDS:
        p = pd.read_csv(runs / f / "class_pairs.csv")
        p["pair"] = [" | ".join(sorted(t)) for t in zip(p.class_a, p.class_b)]
        pairs.append(p.set_index("pair").dprime_val.rename(f))
    pairs = pd.concat(pairs, axis=1)
    complete = pairs.dropna()
    rho = [spearmanr(complete[a], complete[b])[0] for a, b in itertools.combinations(FOLDS, 2)]
    pairs["mean"], pairs["min"], pairs["max"] = pairs[FOLDS].mean(1), pairs[FOLDS].min(1), pairs[FOLDS].max(1)
    pairs["folds_below_2"] = (pairs[FOLDS] < 2).sum(1)
    pairs.sort_values("mean").to_csv(out / "class_pairs_by_fold.csv")
    nearest = {}
    for f in FOLDS:
        p = pairs[f].dropna()
        for key, val in p.items():
            for c in key.split(" | "):
                if c not in nearest.setdefault(f, {}) or val < nearest[f][c][1]:
                    nearest[f][c] = (key, val)
    classes = sorted(set.union(*[set(n) for n in nearest.values()]))
    same_nearest = [max(pd.Series([nearest[f][c][0] for f in FOLDS if c in nearest[f]]).value_counts()) for c in classes]

    summary = {
        "dataset": name, "note": "five folds share specimens: fold results are correlated repeats",
        "across_folds": across, "folds_ci_above_0": ci_above, "folds_ci_below_0": ci_below,
        "classes_credible_in_k_folds": {str(d): v.xs(d, level="pcs").folds_credible.value_counts().sort_index().to_dict()
                                        for d in sorted(v.index.get_level_values("pcs").unique())},
        "classes_credible_in_all_folds": {str(d): sorted(v.xs(d, level="pcs").index[v.xs(d, level="pcs").folds_credible == 5])
                                          for d in sorted(v.index.get_level_values("pcs").unique())},
        "pairs": {"in_all_folds": len(complete), "spearman_dprime_val_between_folds_min_median_max":
                  [float(min(rho)), float(pd.Series(rho).median()), float(max(rho))],
                  "below_2_in_all_folds": int((pairs.folds_below_2 == 5).sum()),
                  "below_2_in_some_fold": int((pairs.folds_below_2 > 0).sum()),
                  "closest_by_mean": json.loads(pairs.nsmallest(10, "mean")[["mean", "min", "max", "folds_below_2"]]
                                                .reset_index().to_json(orient="records"))},
        "classes_same_nearest_class_in_all_folds": int(sum(s == 5 for s in same_nearest)),
        "classes_with_nearest_class": len(classes),
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
    lines = ["| quantity (fold mean [min, max]) | " + " | ".join(s) + " |", "|---|" + "---|" * len(s)]
    for k in keys:
        cells = []
        for v in s.values():
            q = v["across_folds"].get(k)
            cells.append("—" if q is None else f"{q['mean']:.3f} [{q['min']:.3f}, {q['max']:.3f}]")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "| folds with 95 % interval above 0 / below 0 | " + " | ".join(s) + " |", "|---|" + "---|" * len(s)]
    for k in sorted(set.union(*[set(v["folds_ci_above_0"]) for v in s.values()])):
        lines.append(f"| {k} | " + " | ".join(
            f"{v['folds_ci_above_0'].get(k, '—')} / {v['folds_ci_below_0'].get(k, '—')}" for v in s.values()) + " |")
    (a.out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
