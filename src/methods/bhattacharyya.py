"""Gaussian Bhattacharyya distance and coefficient between two class Gaussians (Pándy et al., *Transferability
Estimation using Bhattacharyya Class Separability*, CVPR 2022, arXiv:2111.12780v3; source note
docs/bhattacharyya_source_note.md).

Paper (Sect. 3.2, Eq. 2, p. 4), for class Gaussians N(mu_i, S_i), N(mu_j, S_j) and S = (S_i + S_j) / 2:
    D_B = 1/8 (mu_i - mu_j)^T S^-1 (mu_i - mu_j)  +  1/2 ln( |S| / sqrt(|S_i| |S_j|) )
    BC  = exp(-D_B)            (1 = identical Gaussians, near 0 = little overlap; D_B has the opposite orientation)
    GBC = - sum over ordered pairs i != j of BC(c_i, c_j)          (Eq. 3; each unordered pair counted twice)
Class covariances are unbiased (divided by n - 1, Sect. 3.2, p. 3). ``class_covariance`` gives the three forms the paper
compares (App. B): full, diagonal, spherical. The paper does not define the spherical variance; here it is the mean of
the per-dimension unbiased variances, s_c = trace(S_c) / d, so S_c = s_c I and Eq. 2 gives d/2 ln(s / sqrt(s_i s_j)).

``stable_transfer_*``: an exact numpy port of the authors' later code release (google-research
``stable_transfer/transferability/gbc.py`` @ 5b09c22, ECCV 2022 paper), kept only as a labelled reproduction. It
differs from Eq. 2: population variance (divided by n), variances floored at 1e-4, and for spherical Gaussians a
covariance term without the factor d. Computations keep the input dtype, as the original does.
"""

from __future__ import annotations

import numpy as np

KINDS = ("full", "diagonal", "spherical")


def class_covariance(x: np.ndarray, kind: str) -> np.ndarray:
    """Unbiased class covariance of the rows of ``x`` (n x d) in the requested form, as a d x d matrix."""
    if kind == "full":
        return np.cov(x, rowvar=False, ddof=1)
    var = x.var(axis=0, ddof=1)
    if kind == "diagonal":
        return np.diag(var)
    if kind == "spherical":
        return np.eye(x.shape[1]) * var.mean()
    raise ValueError(f"unknown covariance form {kind}")


def bhattacharyya_terms(mu_i: np.ndarray, mu_j: np.ndarray, cov_i: np.ndarray, cov_j: np.ndarray) -> tuple[float, float]:
    """The two terms of Eq. 2: mean separation (1/8 Mahalanobis under the average covariance) and covariance shape
    (1/2 ln |S| / sqrt(|S_i||S_j|), >= 0, zero only for equal covariances). Raises if a covariance is not positive
    definite."""
    avg = (cov_i + cov_j) / 2
    logdets = []
    for c in (avg, cov_i, cov_j):
        sign, logdet = np.linalg.slogdet(c)
        if sign <= 0:
            raise np.linalg.LinAlgError("covariance is not positive definite")
        logdets.append(logdet)
    delta = mu_i - mu_j
    mean_term = float(delta @ np.linalg.solve(avg, delta) / 8)
    cov_term = float(0.5 * (logdets[0] - 0.5 * (logdets[1] + logdets[2])))
    return mean_term, cov_term


def gbc_score(bc: np.ndarray) -> float:
    """Eq. 3 from a symmetric matrix of pairwise coefficients: minus the sum over ordered pairs i != j."""
    return float(-(bc.sum() - np.trace(bc)))


def stable_transfer_class_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``compute_per_class_mean_and_variance``: mean and population variance, floored at 1e-4."""
    return x.mean(axis=0), np.maximum(x.var(axis=0), x.dtype.type(1e-4))


def stable_transfer_distance(mu1, mu2, var1, var2, gaussian_type: str):
    """Port of ``get_bhattacharyya_distance`` + ``compute_bhattacharyya_distance`` (diagonal or spherical)."""
    if gaussian_type == "spherical":
        var1, var2 = var1.mean(), var2.mean()
    avg = (var1 + var2) / 2
    first = np.sum((mu1 - mu2) ** 2 / avg) / 8
    second = np.sum(np.log(avg)) - 0.5 * np.sum(np.log(var1)) - 0.5 * np.sum(np.log(var2))
    return first + 0.5 * second


def stable_transfer_gbc(features: np.ndarray, labels: np.ndarray, gaussian_type: str) -> float:
    """Port of ``get_gbc_score``: minus the sum over ordered class pairs of exp(-distance)."""
    assert gaussian_type in ("diagonal", "spherical")
    classes = list(dict.fromkeys(labels.tolist()))  # tf.unique keeps first-appearance order
    stats = {c: stable_transfer_class_stats(features[labels == c]) for c in classes}
    per_class = []
    for c1 in classes:
        (mu1, var1) = stats[c1]
        terms = [np.exp(-stable_transfer_distance(mu1, stats[c2][0], var1, stats[c2][1], gaussian_type))
                 for c2 in classes if c2 != c1]
        per_class.append(np.sum(terms))
    return float(-np.sum(per_class))
