"""Five-fold summary of pca_fda_gbc.py runs per dataset and a cross-dataset table (read-only; task.md eeb9707).

Folds share specimens: each indicator is reported per fold and as fold mean / min / max (correlated repeats, never
pooled as independent). Cross-fold stability per arm (spherical train distance): Spearman between every pair of folds
on the class pairs analysed in all five folds, and the number of pairs among the 20 hardest (largest BC) in all five
folds. Per class pair, for the main arms (pca64, pca{d}, fda{d}): fold mean / min / max of the spherical train distance
and coefficient, the val re-estimate and d' (val), and in how many folds the pair was analysed.

``--dataset NAME RUNS_DIR`` (repeatable; RUNS_DIR holds fold0..fold4). Writes ``<NAME>/{indicators.csv,
pairs_main_arms.csv, summary.json}`` and ``comparison.md`` to ``--out``.

Usage: summarise_pca_fda_gbc.py --dataset FIN-Benthic2 RUNS --dataset AquaMonitor RUNS --out DIR
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

FOLDS = [f"fold{k}" for k in range(5)]
TOP = 20


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = v
    return out


def summarise(name: str, runs: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    s = {f: json.loads((runs / f / "summary.json").read_text()) for f in FOLDS}
    rows = {f: {**flat(s[f]["arms"], "arm."), **flat(s[f]["between_arms"], "between."),
                **{f"fda.{k}": v for k, v in flat(s[f]["fda"]["main"]).items() if not k.startswith("eigenvalues")},
                "classes_gbc": s[f]["classes_gbc"], "pairs": s[f]["pairs"]} for f in FOLDS}
    ind = pd.DataFrame(rows).T
    ind.to_csv(out / "indicators.csv")
    across = {c: {"mean": float(ind[c].mean()), "min": float(ind[c].min()), "max": float(ind[c].max()),
                  "folds": int(ind[c].notna().sum())} for c in ind.columns}

    arms = sorted(set.intersection(*[set(s[f]["arms"]) for f in FOLDS]))
    stability, per_pair = {}, []
    for arm in arms:
        t = {f: pd.read_csv(runs / f / f"pairs_{arm}.csv").assign(pair=lambda d: d.class_a + " | " + d.class_b).set_index("pair")
             for f in FOLDS}
        dist = pd.concat({f: t[f].spherical_train_distance.where(t[f].spherical_train_valid) for f in FOLDS}, axis=1)
        complete = dist.dropna()
        rho = [spearmanr(complete[x], complete[y])[0] for x, y in itertools.combinations(FOLDS, 2)]
        hardest = [set(t[f][t[f].spherical_train_valid].nlargest(TOP, "spherical_train_bc").index) for f in FOLDS]
        stability[arm] = {"pairs_in_all_folds": len(complete),
                          "spearman_between_folds_min_median_max": [float(min(rho)), float(pd.Series(rho).median()), float(max(rho))],
                          f"pairs_in_top{TOP}_hardest_in_all_folds": len(set.intersection(*hardest)),
                          f"pairs_in_top{TOP}_hardest_in_any_fold": len(set.union(*hardest))}
        if arm.rstrip("0123456789") in ("pca", "fda"):  # main arms: pca64, pca{d}, fda{d}
            for col in ("spherical_train_distance", "spherical_train_bc", "spherical_val_distance", "dprime_val"):
                v = pd.concat({f: t[f][col] for f in FOLDS}, axis=1)
                per_pair.append(pd.DataFrame({"arm": arm, "quantity": col, "mean": v.mean(1), "min": v.min(1),
                                              "max": v.max(1), "folds": v.notna().sum(1)}))
    pd.concat(per_pair).reset_index().to_csv(out / "pairs_main_arms.csv", index=False)
    summary = {"dataset": name, "note": "five folds share specimens: correlated repeats", "arms": arms,
               "dims_used_per_fold": {f: s[f]["dims_used"] for f in FOLDS},
               "regression_fold0": s["fold0"].get("regression_vs_previous_fold0"),
               "across_folds": across, "cross_fold_stability": stability}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs=2, action="append", required=True, metavar=("NAME", "RUNS"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    s = {n: summarise(n, Path(r), a.out / n) for n, r in a.dataset}
    keys = sorted(set.union(*[set(v["across_folds"]) for v in s.values()]))
    lines = ["Fold mean [min, max] over five correlated folds.", "", "| quantity | " + " | ".join(s) + " |",
             "|---|" + "---|" * len(s)]
    for k in keys:
        cells = [("—" if (q := v["across_folds"].get(k)) is None else f"{q['mean']:.3f} [{q['min']:.3f}, {q['max']:.3f}]")
                 for v in s.values()]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "| cross-fold stability (spherical train distance) | " + " | ".join(s) + " |", "|---|" + "---|" * len(s)]
    for arm in sorted(set.union(*[set(v["cross_fold_stability"]) for v in s.values()])):
        cells = []
        for v in s.values():
            q = v["cross_fold_stability"].get(arm)
            cells.append("—" if q is None else f"Spearman median {q['spearman_between_folds_min_median_max'][1]:.3f}; "
                         f"top-{TOP} in all folds {q[f'pairs_in_top{TOP}_hardest_in_all_folds']}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    (a.out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:5]))


if __name__ == "__main__":
    main()
