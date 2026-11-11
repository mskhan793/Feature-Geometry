"""Class-macro summaries with reliability filtering (software test, small handmade table)."""

import numpy as np
import pandas as pd

from src.statistics.class_macro import compare, macro_difference, reliable
from src.statistics.specimen_geometry import within_class


def table():
    t = pd.DataFrame({"taxon_group": ["A", "B", "C"], "G1_full_status": ["ok", "ok", "ok"],
                      "G2_tied_status": ["ok", "unreliable", "ok"], "G3_tied_status": ["ok", "ok", "unreliable"],
                      "G2_tied-G1_full": [0.2, -3.0, 0.4], "G2_tied-G1_full_ci95": [[0.1, 0.3], [-4, -2], [-0.1, 0.9]],
                      "G3_tied-G1_full": [0.5, 0.1, -2.0], "G3_tied-G1_full_ci95": [[0.2, 0.8], [-0.1, 0.3], [-3, -1]]})
    boots = {c: {"G2_tied-G1_full": np.full(10, v2), "G3_tied-G1_full": np.full(10, v3)}
             for c, v2, v3 in zip(t.taxon_group, t["G2_tied-G1_full"], t["G3_tied-G1_full"])}
    return t.set_index("taxon_group"), boots


def test_primary_summary_excludes_unreliable_and_keeps_all_fits_as_diagnostic():
    t, boots = table()
    r = compare(t, boots, "G2_tied", "G1_full", ["A", "B", "C"])
    assert r["classes"] == 2 and r["classes_excluded_unreliable"] == ["B"] and np.isclose(r["mean"], 0.3)
    assert r["classes_ci_above_0"] == 1 and r["classes_fitted"] == 3
    assert r["all_fits_diagnostic"]["classes"] == 3 and np.isclose(r["all_fits_diagnostic"]["mean"], -0.8)


def test_common_reliable_classes_compare_models_on_the_same_classes():
    t, boots = table()
    common = reliable(t, ["A", "B", "C"], ["G1_full", "G2_tied", "G3_tied"])
    assert common == ["A"]
    assert macro_difference(t, boots, "G3_tied-G1_full", common)["mean"] == 0.5
    assert macro_difference(t, boots, "G3_tied-G1_full", []) == {"classes": 0}


def test_fixed_size_summary_only_for_classes_with_enough_specimens():
    rng = np.random.default_rng(0)
    n = {"A": 40, "B": 20}
    sp = pd.DataFrame({"taxon_group": [c for c in n for _ in range(n[c])],
                       "individual": [f"{c}{i}" for c in n for i in range(n[c])]})
    cam = pd.concat([sp.assign(camera=k) for k in (0, 1)], ignore_index=True).assign(row=lambda d: np.arange(len(d)))
    geo = within_class(["A", "B"], sp, rng.normal(size=(60, 50)), cam, rng.normal(size=(120, 50)), np.ones(120), 0, 30)
    geo = geo.set_index("taxon_group")
    assert np.isfinite(geo.loc["A", "effective_rank_n30"]) and np.isnan(geo.loc["B", "effective_rank_n30"])
    assert np.isfinite(geo.loc["B", "effective_rank"])
