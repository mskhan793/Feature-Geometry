"""Sanity checks for the first Gaussian feature-replay experiment."""

import numpy as np
import pytest

from src.statistics.gaussian import estimate_mean_covariance, sample_gaussian


def test_empirical_statistics_converge_to_known_gaussian():
    mean = np.array([5.0, -2.0])
    cov = np.array([[2.0, 0.7], [0.7, 1.5]])
    features = sample_gaussian(mean, cov, n_samples=50_000, seed=42)
    observed_mean, observed_cov = estimate_mean_covariance(features)
    np.testing.assert_allclose(observed_mean, mean, atol=0.04)
    np.testing.assert_allclose(observed_cov, cov, atol=0.08)


def test_reproducible_sampling():
    mean = np.array([0.0, 0.0])
    cov = np.eye(2)
    np.testing.assert_array_equal(
        sample_gaussian(mean, cov, 20, seed=7),
        sample_gaussian(mean, cov, 20, seed=7),
    )


def test_reject_invalid_shape():
    with pytest.raises(ValueError):
        sample_gaussian(np.zeros(2), np.eye(3), 10)
