"""Shared protocol for held-out class-conditional Gaussian density comparisons (coupled_rscm_vs_lw.py,
balanced_taxonomic_pooling.py).

- Fold data: specimen embedding = mean of its camera means over all selected frames; test rows dropped before the
  feature lookup; train and val specimens unique and disjoint (asserted); eligible classes = >= MIN_TRAIN train and
  >= MIN_VAL val specimens (section 8 D3 of the Ledoit-Wolf diagnostics).
- Spaces: ``pcaN`` = first N components of a PCA fitted on the fold's train specimens; ``native`` / ``native_alt`` =
  the original coordinates without the last / first one (the encoder's final LayerNorm makes the full coordinates
  rank-deficient by one; scripts/audit_feature_null_space.py); ``native_all`` = all original coordinates, for
  encoders without that constraint (BioCLIP 2, SigLIP 2: full-rank features, task.md a82e89e).
- Scoring: -Mahalanobis/2 and -log|Sigma|/2 per val specimen; a covariance whose smallest eigenvalue is
  <= d * float64 eps * its largest (numpy's matrix-rank tolerance) is "numerically singular" and not scored.
- Paired contrasts: per class mean / median / stratified-bootstrap 95 % interval of the per-specimen difference and
  its two parts; class-macro (equal class weights) and specimen-micro (val-count weights) summaries over a class list,
  with the same bootstrap draws for every contrast (conditional on the train fits).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.features.feature_lookup import features_by_name
from src.features.specimen_means import specimen_camera_means, specimen_means
from src.statistics.class_macro import ci95
from src.statistics.covariance_fit import gaussian_parts

MIN_TRAIN, MIN_VAL = 15, 5
STRATA = [(15, 64, "15-63"), (64, 256, "64-255"), (256, np.inf, ">=256")]
N_BOOT = 2000


def stratum(n: int) -> str:
    return next(name for lo, hi, name in STRATA if lo <= n < hi)


def numerics(cov: np.ndarray) -> dict:
    eig = np.linalg.eigvalsh((cov + cov.T) / 2)
    return {"asymmetry": float(np.abs(cov - cov.T).max()), "min_eig": float(eig[0]),
            "condition": float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")}


def score(z: np.ndarray, mu: np.ndarray, cov: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, str]:
    """-maha/2 and -log|cov|/2 per row (NaN with a status if the covariance is undefined, numerically singular or not
    positive definite)."""
    nan = np.full(len(z), np.nan)
    if cov is None:
        return nan, nan, "not defined (n <= d)"
    eig = np.linalg.eigvalsh((cov + cov.T) / 2)
    if eig[0] <= eig[-1] * cov.shape[0] * np.finfo(np.float64).eps:
        return nan, nan, "failed: numerically singular"
    try:
        maha, logdet = gaussian_parts(z, mu, cov)
    except np.linalg.LinAlgError:
        return nan, nan, "failed: not positive definite"
    return -0.5 * maha, np.full(len(z), -0.5 * logdet), "ok"


@dataclass
class FoldData:
    fold: str
    encoder: str
    rows: dict                # in_file, train_val_used, test_rows_dropped
    sk_tr: pd.DataFrame       # train specimen keys (individual, taxon_group, ...)
    x_tr: np.ndarray
    sk_va: pd.DataFrame
    x_va: np.ndarray
    eligible: list
    excluded: dict
    n_tr: pd.Series

    @property
    def y_tr(self) -> np.ndarray:
        return self.sk_tr.taxon_group.to_numpy()

    @property
    def y_va(self) -> np.ndarray:
        return self.sk_va.taxon_group.to_numpy()


def load_fold(cohort_path: Path, caches: list[str], fold: str) -> FoldData:
    cohort = pd.read_parquet(cohort_path)
    rows_in = len(cohort)
    cohort = cohort[cohort[fold].isin(["train", "val"])].reset_index(drop=True)  # test rows dropped before lookup
    x, meta = features_by_name([tuple(Path(p) for p in s.split("=", 1)) for s in caches], cohort.img.tolist())
    sets = {}
    for split in ("train", "val"):
        rows = cohort[fold] == split
        keys, cm, _ = specimen_camera_means(x[rows.values], cohort[rows].reset_index(drop=True), fold)
        sets[split] = specimen_means(keys, cm, fold)
    (sk_tr, x_tr), (sk_va, x_va) = sets["train"], sets["val"]
    assert sk_tr.individual.is_unique and sk_va.individual.is_unique
    assert not set(sk_tr.individual) & set(sk_va.individual), "train and val share specimens"
    n_tr, n_va = sk_tr.taxon_group.value_counts(), sk_va.taxon_group.value_counts()
    eligible = sorted(c for c in n_tr.index if n_tr[c] >= MIN_TRAIN and n_va.get(c, 0) >= MIN_VAL)
    excluded = {c: {"train": int(n_tr.get(c, 0)), "val": int(n_va.get(c, 0))}
                for c in sorted(set(n_tr.index) | set(n_va.index)) if c not in eligible}
    return FoldData(fold, meta["encoder"], {"in_file": rows_in, "train_val_used": len(cohort),
                                            "test_rows_dropped": rows_in - len(cohort)},
                    sk_tr, x_tr, sk_va, x_va, eligible, excluded, n_tr)


def spaces_from(names: list[str], x_tr: np.ndarray, x_va: np.ndarray):
    """(name, dim, train, val) per space; PCA fitted once on the train specimens for all ``pcaN``."""
    dims = [int(s.removeprefix("pca")) for s in names if s.startswith("pca")]
    pca = PCA(n_components=max(dims), svd_solver="full").fit(x_tr) if dims else None
    pt, pv = (pca.transform(x_tr), pca.transform(x_va)) if pca else (None, None)
    for s in names:
        if s == "native_all":
            yield s, x_tr.shape[1], np.asarray(x_tr, np.float64), np.asarray(x_va, np.float64)
        elif s in ("native", "native_alt"):
            keep = slice(None, -1) if s == "native" else slice(1, None)
            yield s, x_tr.shape[1] - 1, np.asarray(x_tr[:, keep], np.float64), np.asarray(x_va[:, keep], np.float64)
        else:
            d = int(s.removeprefix("pca"))
            yield s, d, pt[:, :d], pv[:, :d]


def loglik(parts: tuple, d: int) -> np.ndarray:
    """Gaussian log-likelihood per row from ``score`` parts."""
    return parts[0] + parts[1] - 0.5 * d * np.log(2 * np.pi)


def contrast_columns(key: str, a: tuple, b: tuple, d: int, draws: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Per-class columns of the paired contrast ``a - b`` (each a (maha_part, logdet_part, status) from ``score``),
    the bootstrap means of the per-specimen difference, and that difference (NaN unless both are scored)."""
    ok = a[2] == b[2] == "ok"
    diff = loglik(a, d) - loglik(b, d)
    boot = diff[draws].mean(1) if ok else np.full(len(draws), np.nan)
    cols = {f"{key}_mean": float(diff.mean()) if ok else np.nan,
            f"{key}_median": float(np.median(diff)) if ok else np.nan,
            f"{key}_ci95": ci95(boot) if ok else [np.nan, np.nan],
            f"{key}_maha_part": float((a[0] - b[0]).mean()) if ok else np.nan,
            f"{key}_logdet_part": float((a[1] - b[1]).mean()) if ok else np.nan}
    return cols, boot, diff


def class_groups() -> dict:
    """Standard class groups: all, train-size strata, n_c <= d and n_c > d (predicates on a per-class row)."""
    return {"all": lambda r: True, **{s: (lambda r, s=s: r.stratum == s) for _, _, s in STRATA},
            "n<=p": lambda r: r.n_le_p, "n>p": lambda r: not r.n_le_p}


def full_coordinate_rank(x_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    """Rank (matrix-rank tolerance) and extreme eigenvalues of the within-class covariance of all coordinates."""
    resid = np.asarray(x_tr, np.float64) - np.stack([x_tr[y_tr == c].mean(0) for c in y_tr])
    eig = np.linalg.eigvalsh(resid.T @ resid / len(resid))
    return {"dim": int(x_tr.shape[1]), "rank": int((eig > eig[-1] * len(eig) * np.finfo(np.float64).eps).sum()),
            "smallest_eigenvalues": eig[:3].tolist(), "largest_eigenvalue": float(eig[-1])}


def summarise(per: pd.DataFrame, boots: np.ndarray, classes: list, key: str) -> dict:
    """Macro / micro of contrast ``key`` over ``classes`` (rows of ``per`` and columns of ``boots`` in that order)."""
    if not classes:
        return {"classes": 0}
    t = per.set_index("taxon_group").loc[classes]
    cols = [list(per.taxon_group).index(c) for c in classes]
    w = t.n_val.to_numpy() / t.n_val.sum()
    return {"classes": len(classes), "val_specimens": int(t.n_val.sum()),
            "macro_mean": float(t[f"{key}_mean"].mean()), "macro_ci95": ci95(boots[:, cols].mean(1)),
            "micro_mean": float((t[f"{key}_mean"] * w).sum()), "micro_ci95": ci95(boots[:, cols] @ w),
            "macro_maha_part": float(t[f"{key}_maha_part"].mean()), "macro_logdet_part": float(t[f"{key}_logdet_part"].mean()),
            "class_diff_median": float(t[f"{key}_mean"].median()), "classes_better": int((t[f"{key}_mean"] > 0).sum()),
            "classes_ci_above_0": int(sum(v[0] > 0 for v in t[f"{key}_ci95"])),
            "classes_ci_below_0": int(sum(v[1] < 0 for v in t[f"{key}_ci95"]))}


def summarise_contrasts(per: pd.DataFrame, boots: dict, contrasts: dict, groups: dict) -> dict:
    """For every contrast: ``summarise`` over each class group (a predicate on a ``per`` row) restricted to classes in
    which both estimators were scored, plus the list of numerical failures."""
    out = {}
    sub = per.set_index("taxon_group")
    for key, (a, b) in contrasts.items():
        good = [r.taxon_group for r in per.itertuples() if getattr(r, f"status_{a}") == getattr(r, f"status_{b}") == "ok"]
        out[key] = {g: summarise(per, boots[key], [c for c in good if f(sub.loc[c])], key) for g, f in groups.items()}
        out[key]["numerical_failures"] = per.loc[~per.taxon_group.isin(good), ["taxon_group", f"status_{a}", f"status_{b}"]].to_dict("records")
    return out
