"""One Gaussian or several, per class, in a low-dimensional projection (fixed, predeclared model set).

Models are sklearn ``GaussianMixture`` fits (``fit_mixture``): 5 k-means and 5 random-specimen initialisations plus, for
K > 1, one start per principal direction of the class (equal-count split along it); the
fit with the highest training likelihood bound is kept (k-means alone can lock onto a split along high-variance noise
directions and miss a narrow but well-separated split). A model is *eligible* only if its free parameters are at most
half the training scalars (``n * d / 2``); a fitted mixture is *unreliable* if a component's effective size (sum of
responsibilities) is below ``d + 1``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

MODELS = {  # name: (components, covariance type); diag models are diagnostics only
    "G1_full": (1, "full"), "G2_full": (2, "full"), "G3_full": (3, "full"), "G2_tied": (2, "tied"),
    "G3_tied": (3, "tied"), "G1_diag": (1, "diag"), "G2_diag": (2, "diag"),
}


def axis_split_starts(z: np.ndarray, k: int, covariance_type: str, reg: float) -> list[dict]:
    """Complete initial parameters (weights, means, precisions) from splitting the rows into k equal-count groups
    along each principal direction of ``z``; independent of any random initialisation."""
    _, _, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
    eye = np.eye(z.shape[1])
    starts = []
    for v in vt:
        groups = [z[g] for g in np.array_split(np.argsort(z @ v, kind="stable"), k)]
        covs = [np.cov(g, rowvar=False, bias=True) + reg * eye for g in groups]
        if covariance_type == "tied":
            prec = np.linalg.inv(sum(len(g) * c for g, c in zip(groups, covs)) / len(z))
        elif covariance_type == "diag":
            prec = np.stack([1 / np.diag(c) for c in covs])
        else:
            prec = np.stack([np.linalg.inv(c) for c in covs])
        starts.append({"weights_init": np.array([len(g) for g in groups]) / len(z),
                       "means_init": np.stack([g.mean(0) for g in groups]), "precisions_init": prec})
    return starts


def fit_mixture(z: np.ndarray, k: int, covariance_type: str, reg: float, seed: int) -> GaussianMixture:
    common = dict(n_components=k, covariance_type=covariance_type, reg_covar=reg, max_iter=1000, random_state=seed)
    candidates = [GaussianMixture(n_init=5, init_params=init, **common) for init in ("kmeans", "random_from_data")]
    if k > 1:
        candidates += [GaussianMixture(n_init=1, **s, **common) for s in axis_split_starts(z, k, covariance_type, reg)]
    fits = []
    for gm in candidates:
        gm.fit(z)
        if gm.converged_:
            fits.append(gm)
    if not fits:
        raise RuntimeError(f"K={k} {covariance_type} mixture did not converge")
    return max(fits, key=lambda g: g.lower_bound_)


def free_parameters(k: int, d: int, covariance_type: str) -> int:
    cov = {"full": k * d * (d + 1) // 2, "tied": d * (d + 1) // 2, "diag": k * d}[covariance_type]
    return k * d + (k - 1) + cov


def eligible(k: int, d: int, covariance_type: str, n: int) -> bool:
    return free_parameters(k, d, covariance_type) <= n * d / 2


def covariances(gm) -> list[np.ndarray]:
    c = gm.covariances_
    if gm.covariance_type == "tied":
        return [c]
    if gm.covariance_type == "diag":
        return [np.diag(v) for v in c]
    return list(c)


def condition_number(gm) -> float:
    return float(max(np.linalg.cond(c) for c in covariances(gm)))


def smallest_component(gm, z: np.ndarray) -> float:
    return float(gm.predict_proba(z).sum(0).min())


def split_stability(z_train: np.ndarray, z_val: np.ndarray, k: int, covariance_type: str, reg: float, seed: int,
                    n_boot: int) -> dict:
    """Median adjusted Rand index between the full-train fit's component labels and those of fits on 80 % train
    subsamples drawn without replacement (bootstrap duplicates let a component collapse onto repeated rows), on val
    rows (held out) and on train rows. Val is never used for fitting."""
    ref = fit_mixture(z_train, k, covariance_type, reg, seed)
    ref_val, ref_train = ref.predict(z_val), ref.predict(z_train)
    rng = np.random.default_rng(seed)
    ari_val, ari_train, failures = [], [], 0
    for _ in range(n_boot):
        idx = rng.choice(len(z_train), size=int(0.8 * len(z_train)), replace=False)
        try:
            gm = fit_mixture(z_train[idx], k, covariance_type, reg, seed)
        except (RuntimeError, ValueError):
            failures += 1
            continue
        ari_val.append(adjusted_rand_score(ref_val, gm.predict(z_val)))
        ari_train.append(adjusted_rand_score(ref_train, gm.predict(z_train)))
    return {"ari_val_median": float(np.median(ari_val)), "ari_train_median": float(np.median(ari_train)),
            "val_components_used": int(len(np.unique(ref_val))), "boot_failures": failures}


class SourceGaussians:
    """Sensitivity model: one Gaussian per source group (e.g. ``dataset``) with a shared covariance, mixed with the
    train group proportions; scored unconditionally (the group of a scored row is not used). Groups with fewer than
    ``min_group`` train rows are pooled into one group; if that pool is still too small it joins the largest group."""

    def __init__(self, reg: float, min_group: int = 5):
        self.reg, self.min_group = reg, min_group

    def fit(self, z: np.ndarray, groups: np.ndarray) -> SourceGaussians:
        groups = np.asarray(groups, dtype=object)
        counts = {g: int((groups == g).sum()) for g in np.unique(groups)}
        small = [g for g, n in counts.items() if n < self.min_group]
        merged = np.where(np.isin(groups, small), "_pooled", groups)
        if 0 < (merged == "_pooled").sum() < self.min_group:
            merged = np.where(merged == "_pooled", max(counts, key=counts.get), merged)
        self.groups_ = sorted(np.unique(merged))
        self.weights_ = np.array([np.mean(merged == g) for g in self.groups_])
        self.means_ = np.stack([z[merged == g].mean(0) for g in self.groups_])
        resid = np.concatenate([z[merged == g] - m for g, m in zip(self.groups_, self.means_)])
        self.covariance_ = resid.T @ resid / len(z) + self.reg * np.eye(z.shape[1])
        return self

    def n_parameters(self) -> int:
        k, d = self.means_.shape
        return k * d + (k - 1) + d * (d + 1) // 2

    def score_samples(self, z: np.ndarray) -> np.ndarray:
        from scipy.special import logsumexp
        from scipy.stats import multivariate_normal
        logs = [np.log(w) + np.atleast_1d(multivariate_normal(m, self.covariance_).logpdf(z))
                for w, m in zip(self.weights_, self.means_)]
        return logsumexp(np.stack(logs), axis=0)
