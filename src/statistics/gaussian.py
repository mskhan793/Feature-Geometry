"""Small, reproducible utilities for Gaussian-feature experiments."""

from __future__ import annotations

import numpy as np


def estimate_mean_covariance(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate feature mean and unbiased covariance; rows are independent samples."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("features must have shape (n_samples >= 2, d >= 1)")
    return x.mean(axis=0), np.atleast_2d(np.cov(x, rowvar=False, ddof=1))


def sample_gaussian(
    mean: np.ndarray,
    covariance: np.ndarray,
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample numeric feature embeddings, not images, from a Gaussian model."""
    mu = np.asarray(mean, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if mu.ndim != 1 or cov.shape != (mu.size, mu.size):
        raise ValueError("mean and covariance dimensions do not agree")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return np.random.default_rng(seed).multivariate_normal(
        mu, cov, size=n_samples, check_valid="raise"
    )
