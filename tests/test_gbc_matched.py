"""Matched-pair GBC (task.md e4d291d), software tests on tiny fixtures: (a) agreement with the earlier pca_fda_gbc
summaries, (b) ordered vs unordered pair factor 2, (c) identical pair sets for numerator / denominator and across
matched arms, (d) the leakage gate rejects test units and wrong classes, (e) near-zero GBC gives no ratio."""

import itertools

import numpy as np
import pandas as pd
import pytest

from scripts.gbc_matched_comparison import leakage_gate
from scripts.pca_fda_gbc import indicators
from src.methods.bhattacharyya import gbc_score
from src.statistics import gbc_matched as g


def pairs(bc_train, bc_val, val_valid=None, classes="abcd", n_val=12):
    rows = []
    for i, (a, b) in enumerate(itertools.combinations(classes, 2)):
        r = {"class_a": a, "class_b": b, "n_train_a": 30, "n_train_b": 30, "n_val_a": n_val, "n_val_b": n_val,
             "dprime_val": 1.0 + i, "dprime_train": 1.0 + i, "dprime_defined": True, "code_spherical_bc": bc_train[i] * 1.1}
        for k in g.KINDS:
            for s, bc in (("train", bc_train[i]), ("val", bc_val[i])):
                ok = True if s == "train" else (True if val_valid is None else val_valid[i])
                r.update({f"{k}_{s}_valid": ok, f"{k}_{s}_bc": bc, f"{k}_{s}_distance": -np.log(bc),
                          f"{k}_{s}_mean_term": -np.log(bc) * 0.9, f"{k}_{s}_cov_term": -np.log(bc) * 0.1})
        rows.append(r)
    return pd.DataFrame(rows)


BT = [0.30, 0.10, 0.05, 0.20, 0.02, 0.01]
BV = [0.35, 0.12, 0.04, 0.30, 0.03, 0.02]


def test_a_matches_earlier_summary_definitions():
    p = pairs(BT, BV, val_valid=[True, True, False, True, True, True])
    old = indicators(p)
    for k in g.KINDS:
        new = g.train_val(p, k)
        assert new["gbc_train_all"] == pytest.approx(old[k]["gbc_train"])
        assert new["gbc_val_common"] == pytest.approx(old[k]["gbc_val"])


def test_b_unordered_sum_times_two_equals_ordered_paper_sum():
    m = np.eye(4)
    for (i, j), v in zip(itertools.combinations(range(4), 2), BT):
        m[i, j] = m[j, i] = v
    assert g.gbc(BT) == pytest.approx(gbc_score(m))


def test_c_same_pairs_in_train_and_val_and_across_arms():
    p = pairs(BT, BV, val_valid=[True, True, False, True, True, True])
    r = g.train_val(p, "spherical")
    keep = [0, 1, 3, 4, 5]
    assert r["pairs_common"] == 5 and r["gbc_train_all"] == pytest.approx(g.gbc(BT))
    assert r["gbc_train_common"] == pytest.approx(g.gbc([BT[i] for i in keep]))
    assert r["gbc_val_common"] == pytest.approx(g.gbc([BV[i] for i in keep]))
    other = pairs(BV, BT, val_valid=[True, False, True, True, True, True])
    m = g.matched(p, other.sample(frac=1, random_state=0), "spherical")  # row order must not matter
    both = [0, 3, 4, 5]
    assert m["pairs_matched"] == 4
    assert m["gbc_train_pca"] == pytest.approx(g.gbc([BT[i] for i in both]))
    assert m["gbc_val_fda"] == pytest.approx(g.gbc([BT[i] for i in both]))
    assert g.matched(p, pairs(BT[:3], BV[:3], classes="abc"), "spherical")["status"].startswith("not eligible")


def test_d_leakage_gate_rejects_test_units_and_wrong_classes():
    classes = pd.DataFrame({"taxon_group": ["a", "b", "c"], "n_train": [20, 30, 4], "n_val": [6, 7, 1],
                            "in_gbc_pairs": [True, True, False]})
    support = pd.DataFrame({"taxon_group": ["a", "b", "c"], "train": [20, 30, 4], "val": [6, 7, 1],
                            "eligible": [True, True, False]})
    summary = {"cohort_rows": {"test_rows_dropped": 9}}
    assert leakage_gate(classes, support, summary)["classes"] == 2
    with pytest.raises(AssertionError, match="unit counts"):  # e.g. test units counted as train
        leakage_gate(classes.assign(n_train=[25, 30, 4]), support, summary)
    with pytest.raises(AssertionError, match="eligible"):
        leakage_gate(classes.assign(in_gbc_pairs=[True, True, True]), support, summary)


def test_e_near_zero_gbc_gives_no_ratio():
    p = pairs([1e-12] * 6, [1e-9] * 6)
    r = g.train_val(p, "spherical")
    assert np.isnan(r["ratio_val_over_train"]) and r["ratio_status"].startswith("train mean BC <")
    assert r["mean_bc_val_minus_train"] == pytest.approx(1e-9 - 1e-12)
    none = g.train_val(pairs(BT, BV, val_valid=[False] * 6), "spherical")
    assert none["pairs_common"] == 0 and none["ratio_status"] == "no common pairs"
    assert g.train_val(pairs(BT, BV), "spherical")["ratio_val_over_train"] == pytest.approx(np.mean(BV) / np.mean(BT))
