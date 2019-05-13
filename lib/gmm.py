import tensorflow as tf
import numpy as np
from sklearn import mixture

def gmm_sample(n_samples):
    # Test sample generation from mixture.sample_gaussian where covariance
    # is diagonal, spherical and full
 
    n_features, n_samples = 2, 300
    axis = 1
    mu = rng.randint(10) * rng.rand(n_features)
    cv = (rng.rand(n_features) + 1.0) ** 2
 
    samples = mixture.gmm._sample_gaussian(
        mu, cv, covariance_type='diag', n_samples=n_samples)
 
    assert_true(np.allclose(samples.mean(axis), mu, atol=1.3))
    assert_true(np.allclose(samples.var(axis), cv, atol=1.5))
 
    # the same for spherical covariances
    cv = (rng.rand() + 1.0) ** 2
    samples = mixture.gmm._sample_gaussian(
        mu, cv, covariance_type='spherical', n_samples=n_samples)
 
    assert_true(np.allclose(samples.mean(axis), mu, atol=1.5))
    assert_true(np.allclose(
        samples.var(axis), np.repeat(cv, n_features), atol=1.5))
 
    # and for full covariances
    A = rng.randn(n_features, n_features)
    cv = np.dot(A.T, A) + np.eye(n_features)
    samples = mixture.gmm._sample_gaussian(
        mu, cv, covariance_type='full', n_samples=n_samples)
    assert_true(np.allclose(samples.mean(axis), mu, atol=1.3))
    assert_true(np.allclose(np.cov(samples), cv, atol=2.5))
 
    # Numerical stability check: in SciPy 0.12.0 at least, eigh may return
    # tiny negative values in its second return value.
    x = mixture.gmm._sample_gaussian(
        [0, 0], [[4, 3], [1, .1]], covariance_type='full', random_state=42)
    assert_true(np.isfinite(x).all())