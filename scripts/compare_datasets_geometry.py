"""Side-by-side summary of two class_geometry_pilot.py runs on different datasets (read-only).

Both runs use the same unit (one camera-balanced embedding per specimen), statistics and PCA dimensions, so the table
lines up quantities computed the same way; it does not make the datasets comparable in sample support or acquisition.
Fractions are used where class or pair counts differ. Writes ``comparison.json`` and ``comparison.md`` to ``--out``.

Usage: compare_datasets_geometry.py --a GEOMETRY_DIR --a-name NAME --b GEOMETRY_DIR --b-name NAME --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

KEYS = ["G2_tied-G1_full", "G3_tied-G1_full", "G2_full-G1_full", "G3_full-G1_full", "GD_tied-G1_full"]


def summary(d: Path) -> dict:
    r = json.loads((d / "results.json").read_text())
    pairs = pd.read_csv(d / "class_pairs.csv")
    out = {"classes_analysed": r["classes"], "classes_insufficient": len(r["classes_insufficient_support"]),
           "specimens_train": r["specimens"]["train"], "specimens_val": r["specimens"]["val"],
           "train_per_class_min_median_max": r["train_specimens_per_class_min_median_max"],
           "val_per_class_min_median_max": r["val_specimens_per_class_min_median_max"],
           **{f"pca_explained_{k}": v for k, v in r["pca_explained_variance"].items()},
           **{f"within_{k}_median": v[1] for k, v in r["within_class_native"].items()},
           "pairs": len(pairs), "pairs_dprime_val_below_1_share": float((pairs.dprime_val < 1).mean()),
           "pairs_dprime_val_below_2_share": float((pairs.dprime_val < 2).mean()),
           "pairs_dprime_val_median": float(pairs.dprime_val.median()),
           "spearman_dprime_val_vs_radius_native": r["between_classes"]["spearman_dprime_val_vs_radius_native"],
           "pairs_pca2d_above_native_share": r["between_classes"]["pca2d_above_native"] / len(pairs)}
    for d_, s in r["modality"]["by_pcs"].items():
        n = sum(s["verdicts"].values())
        out[f"pcs{d_}_classes_fitted"] = n
        out[f"pcs{d_}_verdicts_share"] = {v: round(c / n, 3) for v, c in s["verdicts"].items()}
        out[f"pcs{d_}_G2_tied_split_stable_share"] = s["G2_tied_split_stable_classes"] / n
        out[f"pcs{d_}_both_views_same_component_median"] = s["G2_tied_both_views_same_component_median"]
        for k in KEYS:
            if s["differences"].get(k, {}).get("classes"):  # primary (reliable) comparison with at least one class
                v = s["differences"][k]
                out[f"pcs{d_}_{k}"] = {"mean": round(v["mean"], 3), "ci95": [round(x, 3) for x in v["ci95"]],
                                       "classes": v["classes"], "share_ci_above_0": round(v["classes_ci_above_0"] / v["classes"], 3),
                                       "share_ci_below_0": round(v["classes_ci_below_0"] / v["classes"], 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--a-name", required=True)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--b-name", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    s = {a.a_name: summary(a.a), a.b_name: summary(a.b)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "comparison.json").write_text(json.dumps(s, indent=1))
    keys = list(dict.fromkeys(list(s[a.a_name]) + list(s[a.b_name])))
    lines = [f"| quantity | {a.a_name} | {a.b_name} |", "|---|---|---|"]
    fmt = lambda v: "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else json.dumps(v))  # noqa: E731
    lines += [f"| {k} | {fmt(s[a.a_name].get(k))} | {fmt(s[a.b_name].get(k))} |" for k in keys]
    (a.out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
