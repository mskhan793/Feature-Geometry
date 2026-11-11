"""How much do the reliability points of the 2026-11-11 methods review affect saved class_geometry_pilot.py outputs?
(read-only; counts from per_class_modality.csv and per_class_geometry.csv of each fold run)

Per run and PCA dimension:
- comparisons (class x mixture) whose mixture is "unreliable" (a component's responsibility mass < d + 1) but still
  enter the class-macro difference to G1_full; the macro mean with and without them;
- "credible multiple groups" verdicts where G2_tied itself is not usable, or where no usable *G2_tied* interval lies
  above 0 (the positive interval came from another mixture);
- G2_tied stability fits that failed on 80 % subsamples (excluded from the median);
- GD_tied fits whose own parameter count exceeds the n * d / 2 budget (eligibility was checked with G1_full's count);
- classes whose "_n30" summary used fewer than 30 specimens.

Usage: audit_mixture_reliability.py --runs DIR [DIR ...] --out result.json
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

MIX = ["G2_tied", "G3_tied", "G2_full", "G3_full"]


def audit(run: Path) -> dict:
    mod = pd.read_csv(run / "per_class_modality.csv")
    geo = pd.read_csv(run / "per_class_geometry.csv")
    out = {"classes_n30_below_30": sorted(geo.taxon_group[geo.n < 30])}
    for d, g in mod.groupby("pcs"):
        s = {}
        for m in MIX:
            have = g[g[f"{m}-G1_full"].notna()]
            ok = have[have[f"{m}_status"] == "ok"]
            s[m] = {"classes_in_macro": len(have), "unreliable_in_macro": int((have[f"{m}_status"] == "unreliable").sum()),
                    "macro_mean_as_reported": float(have[f"{m}-G1_full"].mean()) if len(have) else None,
                    "macro_mean_ok_only": float(ok[f"{m}-G1_full"].mean()) if len(ok) else None}
        cred = g[g.verdict == "credible multiple groups"]
        g2_ci_above = cred["G2_tied-G1_full_ci95"].map(lambda v: ast.literal_eval(v)[0] > 0 if isinstance(v, str) else False)
        s["credible"] = len(cred)
        s["credible_G2_tied_not_ok"] = int((cred.G2_tied_status != "ok").sum())
        s["credible_without_G2_tied_interval_above_0"] = sorted(cred.taxon_group[~g2_ci_above | (cred.G2_tied_status != "ok")])
        s["classes_with_stability_failures"] = int((g.G2_tied_split_boot_failures > 0).sum())
        s["stability_failures_max"] = int(g.G2_tied_split_boot_failures.max())
        if "GD_tied_params" in g:
            s["GD_tied_over_budget"] = sorted(g.taxon_group[g.GD_tied_params > g.n_train * d / 2])
        out[f"pcs{d}"] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    result = {str(r): audit(r) for r in a.runs}
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
