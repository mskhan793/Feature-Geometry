"""Class-balanced / taxonomy-informed pooling: pools of within-class covariances, weights, leave-target-out, donor levels
with mixed ranks, the df rule, the nearest-mean control, the taxonomy mapping, and an end-to-end run of
scripts/balanced_taxonomic_pooling.py + summarise_balanced_taxonomic.py (software fixtures only)."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.data import taxonomy as tx
from src.methods.taxonomic_pooling import donor_df, nearest_donors, pool, taxonomy_donors
from tests.test_class_geometry_pilot import run, write_cache

TAX = pd.DataFrame({"taxon_group": ["A", "B", "C", "D", "E"], "rank": ["species", "species", "genus", "family", None],
                    "species": ["G1 a", "G1 b", None, None, None], "genus": ["G1", "G1", "G2", None, None],
                    "family": ["F1", "F1", "F1", "F2", None], "order": ["O", "O", "O", "O", None]}).set_index("taxon_group")


def test_pools_are_within_class_and_weighted_as_declared():
    rng = np.random.default_rng(0)
    x = {c: rng.normal(size=(n, 3)) + 10 * k for k, (c, n) in enumerate([("A", 10), ("B", 30)])}
    scm = {c: np.cov(v, rowvar=False) for c, v in x.items()}
    n = {"A": 10, "B": 30}
    assert np.allclose(pool(scm, n, ["A", "B"], "specimen"), 0.25 * scm["A"] + 0.75 * scm["B"])
    assert np.allclose(pool(scm, n, ["A", "B"], "balanced"), 0.5 * (scm["A"] + scm["B"]))
    grand = np.cov(np.vstack(list(x.values())), rowvar=False)  # includes the 10-unit mean shift
    assert np.trace(grand) > 10 * np.trace(pool(scm, n, ["A", "B"], "balanced"))
    with pytest.raises(ValueError):
        pool(scm, n, ["A"], "capped")


def test_donor_levels_mixed_ranks_and_df_rule():
    n = {"A": 40, "B": 80, "C": 300, "D": 22, "E": 50}
    cands = list(n)
    assert taxonomy_donors("A", TAX, cands, n, 8) == ("genus", ["B"])       # species -> genus mate
    assert taxonomy_donors("C", TAX, cands, n, 8) == ("family", ["A", "B"])  # genus rank -> family, target left out
    assert taxonomy_donors("D", TAX, cands, n, 8)[0] == "global"             # family rank -> no level
    assert taxonomy_donors("E", TAX, cands, n, 8) == ("global", ["A", "B", "C", "D"])  # no ancestry
    assert taxonomy_donors("Z", TAX, cands + ["Z"], {**n, "Z": 20}, 8)[0] == "global"  # not in the table
    assert taxonomy_donors("B", TAX, cands, n, 39) == ("genus", ["A"])       # df 39 >= 39
    assert taxonomy_donors("B", TAX, cands, n, 40) == ("family", ["A", "C"])  # df 39 < 40 -> family (each class once)
    assert taxonomy_donors("A", TAX, ["A", "C", "D"], n, 8) == ("family", ["C"])  # genus mate not a candidate
    assert donor_df(n, ["A", "C"]) == 39 + 299


def test_nearest_mean_control():
    means = {c: np.array([float(k)]) for k, c in enumerate("ABCDE")}
    n = {c: 10 for c in "ABCDE"}
    assert nearest_donors("C", means, list("ABCDE"), 1, n, 5) == ["B"]  # tie B / D: candidate order (stable sort)
    assert nearest_donors("A", means, list("ABCDE"), 1, n, 5) == ["B"]
    assert nearest_donors("A", means, list("ABCDE"), 1, n, 15) == ["B", "C"]  # extended until df 18 >= 15


def test_taxonomy_mapping(tmp_path):
    pd.DataFrame({"taxa": ["Ab cd", "Ef sp.", "Fam"], "species": ["Ab cd", None, None], "genus": ["Ab", "Ef", None],
                  "family": ["F", "F", "Fam"], "order": ["O", "O", "O"], "specimens": [1, 2, 3]}).to_csv(tmp_path / "t.csv", index=False)
    t = tx.finbenthic2(tmp_path / "t.csv", ["Ab_cd", "Ef_sp", "Fam"])
    assert t["rank"].to_dict() == {"Ab_cd": "species", "Ef_sp": "genus", "Fam": "family"} and t.loc["Ef_sp", "table_specimens"] == 2
    with pytest.raises(AssertionError):
        tx.finbenthic2(tmp_path / "t.csv", ["Ab_cd", "Unknown_sp"])
    pd.DataFrame({"taxon": ["X y", "X y", "Z"], "taxon_group": ["X y", "X y adult", "Z"]}).to_csv(tmp_path / "m.csv", index=False)
    pd.DataFrame({"kingdom": ["A"], "phylum": ["P"], "class": ["C"], "order": ["O"], "family": ["F"], "genus": ["X"],
                  "species": ["X y"], "label": ["X y"]}).to_csv(tmp_path / "tt.csv", index=False)
    a = tx.aquamonitor(tmp_path / "m.csv", tmp_path / "tt.csv", ["X y", "X y adult", "Z"])
    assert a.loc["X y adult", "genus"] == "X" and a.loc["Z", "rank"] is None and a.loc["Z", "note"] == "taxon not in taxon_table"


def test_runs_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    rows, feats, p = [], [], 30
    names = {"G1_alpha": 40, "G1_beta": 80, "G2_sp": 300, "Fam2": 22, "Small_sp": 10}
    for c, n in names.items():
        centre = rng.normal(0, 3, p)
        for s in range(n + 8):
            base = centre + rng.normal(0, 1, p) * (1 + (c == "G2_sp"))
            for cam in (1, 2):
                rows.append({"img": f"{c}{s}_{cam}", "individual": f"{c}{s:03d}", "taxon_group": c,
                             "fold0": "train" if s < n else "val", "camera": cam, "hdf5_row": len(rows)})
                feats.append(base + rng.normal(0, 0.1, p))
    caches = [write_cache(tmp_path, "c1", rows, feats)]
    test_rows = pd.DataFrame(rows[:4]).assign(img=lambda d: "test_" + d.img, individual="T1", fold0="test")
    pd.concat([pd.DataFrame(rows), test_rows]).to_parquet(tmp_path / "cohort.parquet", index=False)
    pd.DataFrame({"taxa": ["G1 alpha", "G1 beta", "G2 sp.", "Fam2", "Small sp."], "species": ["G1 alpha", "G1 beta", None, None, None],
                  "genus": ["G1", "G1", "G2", None, "Sm"], "family": ["F1", "F1", "F1", "Fam2", "F1"], "order": ["O"] * 5,
                  "specimens": [48, 88, 308, 30, 18]}).to_csv(tmp_path / "table.csv", index=False)
    args = ["--cohort", str(tmp_path / "cohort.parquet"), "--caches", *caches, "--fold", "fold0", "--spaces", "pca8", "pca4", "native"]
    out, cpl = tmp_path / "out", tmp_path / "cpl"
    run(["scripts/balanced_taxonomic_pooling.py", "--dataset", "finbenthic2", "--fin-table", str(tmp_path / "table.csv"),
         *args, "--out", str(out)])
    run(["scripts/coupled_rscm_vs_lw.py", *args, "--out", str(cpl)])
    s = json.loads((out / "summary.json").read_text())
    assert s["cohort_rows"]["test_rows_dropped"] == 4 and s["eligible_classes"] == 4
    r8 = s["spaces"]["pca8"]
    assert r8["level_counts"] == {"genus": 2, "family": 1, "global": 1} and r8["candidates"] == 4
    assert all(not v["numerical_failures"] for k, v in r8["contrasts"].items() if k != "taxonomy_vs_nearest")
    assert r8["contrasts"]["taxonomy_vs_nearest"]["all"]["classes"] == 3  # defined only where taxonomy was used
    per = pd.read_csv(out / "perclass.csv").set_index(["space", "taxon_group"])
    assert per.loc[("pca8", "G2_sp"), "level"] == "family" and per.loc[("pca8", "G2_sp"), "donors"] == "G1_alpha;G1_beta"
    assert per.loc[("pca8", "Fam2"), "taxonomy_mean"] == 0  # fallback: taxonomic pool = global leave-out pool
    ref = pd.read_csv(cpl / "perclass.csv").set_index(["space", "taxon_group"]).loc[per.index]
    assert (per.val_loglik_lw == ref.val_loglik_lw).all() and (per.val_loglik_coupled == ref.val_loglik_coupled).all()
    assert (per.val_loglik_pool_spec == ref.val_loglik_pooling_lw_shrink).all()
    spec = pd.read_csv(out / "perspecimen.csv")
    assert np.allclose(spec.d_balancing, spec.loglik_pool_bal - spec.loglik_pool_spec)
    assert not set(spec.individual) & {r["individual"] for r in rows if r["fold0"] == "train"}
    runs, cps = tmp_path / "runs", tmp_path / "cps"
    for k in range(5):
        shutil.copytree(out, runs / f"fold{k}")
        shutil.copytree(cpl, cps / f"fold{k}")
    run(["scripts/summarise_balanced_taxonomic.py", "--dataset", "Toy", str(runs), str(cps), "--out", str(tmp_path / "sum")])
    toy = tmp_path / "sum" / "Toy"
    repro = json.loads((toy / "reproduction.json").read_text())
    assert repro["matched"] == repro["rows_here"] == 60 and repro["coupled_vs_lw_ci95_equal"]
    assert all(v == 0 for k, v in repro.items() if k.startswith("max_abs_diff"))
    levels = pd.read_csv(toy / "levels.csv")
    assert len(levels) == 15 and (levels[levels.space == "pca8"].genus == 2).all()
    groups = pd.read_csv(toy / "groups.csv")
    assert groups[(groups.contrast == "taxonomy") & (groups.group == "fallback")].class_diff_mean.fillna(0).eq(0).all()
    assert len(pd.read_csv(toy / "paths.csv")) == 12 and (tmp_path / "sum" / "summary.md").exists()
