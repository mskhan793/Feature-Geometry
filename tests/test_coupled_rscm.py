"""Coupled RSCM wrapper against the authors' unmodified code, Eq. (4) special cases, parameter conventions, and an
end-to-end run of scripts/coupled_rscm_vs_lw.py + summarise_coupled_rscm.py (software fixtures only)."""

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.methods.coupled_rscm import (SOURCE, SOURCE_SHA256, adapted_module, convergence_residuals, coupled_covariance,
                                      fit_coupled, original_module)
from src.statistics.covariance_fit import gaussian_loglik, gaussian_parts, lw_alpha, lw_class_covariance, shrunk
from tests.test_class_geometry_pilot import run, write_cache


def classes_data(sizes, p, seed=0):
    """Heavy-ish tailed classes with different covariances (t-like scale mixture), for software checks only."""
    rng = np.random.default_rng(seed)
    x, y = [], []
    for k, n in enumerate(sizes):
        a = rng.normal(size=(p, p)) * (0.3 + 0.2 * k)
        z = rng.normal(size=(n, p)) @ (a + np.eye(p)) / np.sqrt(rng.chisquare(8, (n, 1)) / 8)
        x.append(z + 3 * k)
        y += [f"c{k}"] * n
    return np.vstack(x), np.array(y)


def test_vendored_file_is_the_pinned_authors_file():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    assert (SOURCE.parent / "LICENSE").read_text().startswith("MIT License")


def test_adapted_code_equals_original_for_four_classes_and_original_fails_otherwise():
    x, y = classes_data([10, 20, 30, 45], 6)
    classes = sorted(set(y))
    a, b = fit_coupled(x, y, classes, original_module()), fit_coupled(x, y, classes, adapted_module())
    assert np.array_equal(a.alpha, b.alpha) and np.array_equal(a.beta, b.beta)
    assert all(np.array_equal(s, t) for s, t in zip(a.sigmas, b.sigmas))
    # the output-only patch: coefficients exposed by the adapted code, residuals of the returned fixed point
    assert np.isnan(a.residual_alpha).all() and a.iterations is None and b.iterations >= 1
    assert (b.residual_alpha < 1e-6).all() and (b.residual_beta < 1e-6).all()
    x5, y5 = classes_data([10, 20, 30, 45, 12], 6)
    with pytest.raises(ValueError):  # the hard-coded reshape(4) is why the adaptation is needed
        fit_coupled(x5, y5, sorted(set(y5)), original_module())
    assert len(fit_coupled(x5, y5, sorted(set(y5))).sigmas) == 5


def test_fit_follows_equation_4_with_unbiased_and_pooled_scm():
    x, y = classes_data([8, 20, 30, 45, 60], 12)  # first class has n < p
    classes = sorted(set(y))
    f = fit_coupled(x, y, classes)
    n = np.array([(y == c).sum() for c in classes])
    assert np.array_equal(f.n, n)
    for k, c in enumerate(classes):
        assert np.allclose(f.scm[k], np.cov(x[y == c], rowvar=False))  # unbiased (n - 1)
        assert np.allclose(f.sigmas[k], coupled_covariance(f.scm[k], f.pooled, f.alpha[k], f.beta[k]))
        s = f.sigmas[k]
        assert np.allclose(s, s.T) and (f.alpha[k] == 1 or np.linalg.eigvalsh(s)[0] > 0)
    assert np.allclose(f.pooled, sum(nk * s for nk, s in zip(n, f.scm)) / n.sum())
    assert ((0 <= f.alpha) & (f.alpha <= 1) & (0 <= f.beta) & (f.beta <= 1)).all()
    assert ((1 <= f.gamma) & (f.gamma <= 12)).all() and (f.kappa >= -2 / 14).all()


def test_special_cases_and_conventions():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(5, 5)), rng.normal(size=(5, 5))
    s_k, s_p = a @ a.T, b @ b.T
    assert np.allclose(coupled_covariance(s_k, s_p, 1, 1), s_k)                       # class only
    assert np.allclose(coupled_covariance(s_k, s_p, 1, 0), s_p)                       # pooled only
    assert np.allclose(coupled_covariance(s_k, s_p, 0, 1), np.trace(s_k) / 5 * np.eye(5))  # identity
    for al, be in [(0.3, 0.7), (0.9, 0.1)]:
        assert np.trace(coupled_covariance(s_k, s_p, al, be)) == pytest.approx(np.trace(be * s_k + (1 - be) * s_p))
    # LW alpha is the shrinkage; the coupled alpha is the weight kept: beta = 1, alpha = 1 - alpha_LW gives our LW form
    z = rng.normal(size=(30, 5)) * np.arange(1, 6)
    cov, a_lw = lw_class_covariance(z)
    assert a_lw == lw_alpha(z - z.mean(0))
    assert np.allclose(cov, shrunk(np.cov(z, rowvar=False, bias=True), a_lw))
    assert np.allclose(coupled_covariance(np.cov(z, rowvar=False, bias=True), s_p, 1 - a_lw, 1), cov)


def test_gaussian_parts_and_residuals():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(5, 5))
    cov, mean, x = a @ a.T + np.eye(5), rng.normal(size=5), rng.normal(size=(6, 5))
    maha, logdet = gaussian_parts(x, mean, cov)
    assert np.allclose(maha, np.einsum("ij,jk,ik->i", x - mean, np.linalg.inv(cov), x - mean))
    assert logdet == pytest.approx(np.linalg.slogdet(cov)[1])
    assert np.allclose(gaussian_loglik(x, mean, cov), -0.5 * (maha + logdet + 5 * np.log(2 * np.pi)))
    c = {"22": np.array([1.0]), "21": np.array([0.0]), "20": np.array([1.0]), "02": np.array([1.0]), "11": np.array([0.0]),
         "10": np.array([-1.0]), "01": np.array([-0.6])}
    ra, rb = convergence_residuals(c, np.array([0.5]), np.array([0.3]))
    # (7): alpha = -(1/2)(0.3*0 - 1)/(0.09 + 0 + 1) = 0.5/1.09; (8): beta = -(1/2)(0 + 0 - 0.6)/(0.25 + 1) = 0.24
    assert ra[0] == pytest.approx(abs(0.5 / 1.09 - 0.5)) and rb[0] == pytest.approx(abs(0.24 - 0.3))


def test_comparison_runs_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    rows, feats = [], []
    p = 30
    sizes = {"A": 40, "B": 80, "C": 300, "D": 22, "E": 10}  # train specimens; E below the minimum; D <= p natively
    for c, n in sizes.items():
        centre = rng.normal(0, 3, p)
        for s in range(n + 8):
            split = "train" if s < n else "val"
            base = centre + rng.normal(0, 1, p) * (1 + (c == "C"))
            for cam in (1, 2):
                rows.append({"img": f"{c}{s}_{cam}", "individual": f"{c}{s:03d}", "taxon_group": c, "fold0": split,
                             "camera": cam, "hdf5_row": len(rows)})
                feats.append(base + rng.normal(0, 0.1, p))
    caches = [write_cache(tmp_path, "c1", rows, feats)]
    test_rows = pd.DataFrame(rows[:4]).assign(img=lambda d: "test_" + d.img, individual="T1", fold0="test")
    pd.concat([pd.DataFrame(rows), test_rows]).to_parquet(tmp_path / "cohort.parquet", index=False)
    out, lw = tmp_path / "out", tmp_path / "lw"
    args = ["--cohort", str(tmp_path / "cohort.parquet"), "--caches", *caches, "--fold", "fold0"]
    run(["scripts/coupled_rscm_vs_lw.py", *args, "--out", str(out), "--spaces", "pca8", "pca4", "native"])
    run(["scripts/lw_fit_diagnostics.py", *args, "--out", str(lw), "--frames", "2", "--caps", "15", "--pcs", "8", "4"])
    s = json.loads((out / "summary.json").read_text())
    assert s["cohort_rows"]["test_rows_dropped"] == 4 and s["eligible_classes"] == 4
    assert s["excluded_classes"] == {"E": {"train": 10, "val": 8}} and list(s["spaces"]) == ["pca8", "pca4", "native"]
    r8, rn = s["spaces"]["pca8"], s["spaces"]["native"]
    for r in (r8, rn):
        assert r["four_class_original_vs_adapted"] == {"run": True, "classes": ["A", "B", "C", "D"],
                                                       "max_abs_diff_alpha_beta_sigmas": 0.0}
        assert r["contrasts"]["coupled_vs_lw"]["all"]["classes"] == 4 and not r["contrasts"]["coupled_vs_lw"]["numerical_failures"]
        assert np.isfinite(r["residual_alpha_max"]) and r["pooled"]["rank"] == r["dim"]
    assert r8["contrasts"]["coupled_vs_lw"]["15-63"]["classes"] == 2 and r8["classes_n_le_p"] == 0
    assert rn["dim"] == p and rn["classes_n_le_p"] == 1 and rn["contrasts"]["coupled_vs_lw"]["n<=p"]["classes"] == 1
    per = pd.read_csv(out / "perclass.csv")
    assert len(per) == 3 * 4
    for m in ("lw", "coupled", "lw_unbiased", "coupled_no_pooling", "pooling_lw_shrink"):
        assert (per[f"status_{m}"] == "ok").all()
    assert per.set_index(["space", "taxon_group"]).loc[("native", "D"), "status_class_scm"] == "not defined (n <= d)"
    parts = per.coupled_vs_lw_maha_part + per.coupled_vs_lw_logdet_part
    assert np.allclose(parts, per.coupled_vs_lw_mean)
    spec = pd.read_csv(out / "perspecimen.csv")
    assert len(spec) == 3 * 4 * 8 and spec[["loglik_lw", "loglik_coupled"]].notna().all().all()  # paired support
    assert not set(spec.individual) & {r["individual"] for r in rows if r["fold0"] == "train"}
    assert np.allclose(spec.d_coupled_vs_lw, spec.loglik_coupled - spec.loglik_lw)
    # the 2 x 2 decomposition and the normalisation arm add up exactly
    both = spec.d_pooling_at_lw_shrink + spec.d_shrink_at_coupled_pooling
    assert np.allclose(both, spec.d_shrink_at_no_pooling + spec.d_pooling_at_coupled_shrink)
    assert np.allclose(spec.d_normalisation + both, spec.d_coupled_vs_lw)
    # the LW reference reproduces lw_fit_diagnostics.py exactly in the PCA spaces; summariser over five identical folds
    runs, lws = tmp_path / "runs", tmp_path / "lws"
    for k in range(5):
        shutil.copytree(out, runs / f"fold{k}")
        shutil.copytree(lw, lws / f"fold{k}")
    run(["scripts/summarise_coupled_rscm.py", "--dataset", "Toy", str(runs), str(lws), "--out", str(tmp_path / "sum")])
    toy = tmp_path / "sum" / "Toy"
    repro = json.loads((toy / "reproduction.json").read_text())
    assert repro["matched"] == repro["class_folds_here"] == repro["class_folds_reference"] == 40
    assert repro["max_abs_diff_val_loglik_lw"] == 0 and repro["max_abs_diff_alpha_lw"] == 0 and repro["n_train_equal"]
    folds = pd.read_csv(toy / "folds.csv")
    assert len(folds) == 5 * 3 * 6 and folds.groupby(["space", "contrast"]).macro_mean.nunique().eq(1).all()
    groups = pd.read_csv(toy / "groups.csv")
    assert len(groups) == 3 * 6 * 5 and groups[(groups.space == "native") & (groups.group == "n<=p")].class_folds.eq(5).all()
    corners = pd.read_csv(toy / "corners.csv")
    assert len(corners) == 9 and corners.class_scm_defined.sum() == 20 + 20 + 15
    assert len(pd.read_csv(toy / "examples.csv")) == 15 and (toy / "examples_by_fold.csv").exists()
    assert set(json.loads((toy / "resources.json").read_text())["fold0"]) >= {"total_seconds", "native_seconds"}
    # regression check against a first-version (pcs / diff_mean) layout of the same PCA results
    v1 = tmp_path / "v1" / "toy" / "fold0"
    v1.mkdir(parents=True)
    first = per[per.space != "native"].assign(pcs=lambda t: t.dim)
    first.rename(columns={"coupled_vs_lw_mean": "diff_mean", "coupled_vs_lw_ci95": "diff_ci95"}).to_csv(v1 / "perclass.csv", index=False)
    (tmp_path / "v2" / "toy").mkdir(parents=True)
    shutil.copytree(out, tmp_path / "v2" / "toy" / "fold0")
    run(["scripts/check_coupled_regression.py", "--old", str(tmp_path / "v1"), "--new", str(tmp_path / "v2")])
