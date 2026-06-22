import numpy as np
import pytest
from bo import (
    rbf_kernel, gp_posterior,
    acquisition_ucb, acquisition_ei,
    maximize_acquisition, bo_step,
    TARGET_FUNCTIONS,
)


# rbf_kernel

def test_rbf_kernel_shape():
    K = rbf_kernel(np.linspace(0, 1, 5), np.linspace(0, 1, 8),
                   length_scale=0.3, signal_var=1.0)
    assert K.shape == (5, 8)


def test_rbf_kernel_symmetric():
    X = np.linspace(0, 1, 10)
    K = rbf_kernel(X, X, length_scale=0.3, signal_var=1.0)
    np.testing.assert_allclose(K, K.T)


def test_rbf_kernel_diagonal_equals_signal_var():
    X = np.linspace(0, 1, 5)
    K = rbf_kernel(X, X, length_scale=0.3, signal_var=2.5)
    np.testing.assert_allclose(np.diag(K), 2.5)


def test_rbf_kernel_positive_semidefinite():
    X = np.linspace(0, 1, 10)
    K = rbf_kernel(X, X, length_scale=0.3, signal_var=1.0)
    assert np.all(np.linalg.eigvalsh(K) > -1e-10)


# gp_posterior

def test_gp_posterior_shapes():
    mu, sigma = gp_posterior(
        np.array([0.1, 0.5, 0.9]), np.array([0.0, 1.0, 0.0]),
        np.linspace(0, 1, 50), length_scale=0.3, signal_var=1.0, noise_var=1e-4,
    )
    assert mu.shape == (50,)
    assert sigma.shape == (50,)


def test_gp_posterior_sigma_nonnegative():
    _, sigma = gp_posterior(
        np.array([0.2, 0.6]), np.array([1.0, -1.0]),
        np.linspace(0, 1, 100), length_scale=0.3, signal_var=1.0, noise_var=1e-6,
    )
    assert np.all(sigma >= 0)


def test_gp_posterior_interpolates():
    # Using very low noise, posterior mean should be close to training values
    X_train = np.array([0.2, 0.5, 0.8])
    y_train = np.array([1.0, 0.0, -1.0])
    mu, _ = gp_posterior(X_train, y_train, X_train,
                          length_scale=0.3, signal_var=1.0, noise_var=1e-8)
    np.testing.assert_allclose(mu, y_train, atol=1e-4)


def test_gp_posterior_sigma_small_at_training_points():
    X_train = np.array([0.3, 0.7])
    _, sigma = gp_posterior(X_train, np.array([1.0, -1.0]), X_train,
                             length_scale=0.3, signal_var=1.0, noise_var=1e-8)
    assert np.all(sigma < 0.01)


# acquisition_ucb

def test_acquisition_ucb_values():
    result = acquisition_ucb(np.array([1.0, 2.0, 3.0]),
                             np.array([0.5, 0.5, 0.5]), kappa=2.0)
    np.testing.assert_allclose(result, [2.0, 3.0, 4.0])


def test_acquisition_ucb_zero_sigma_returns_mu():
    mu = np.array([1.0, 2.0])
    np.testing.assert_allclose(acquisition_ucb(mu, np.zeros(2), kappa=3.0), mu)


# acquisition_ei

def test_acquisition_ei_nonnegative():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(100)
    sigma = np.abs(rng.standard_normal(100)) + 0.01
    assert np.all(acquisition_ei(mu, sigma, y_best=0.5) >= 0)


def test_acquisition_ei_zero_when_no_uncertainty():
    np.testing.assert_allclose(
        acquisition_ei(np.array([1.0, 2.0]), np.zeros(2), y_best=0.0), 0.0,
    )


def test_acquisition_ei_positive_above_best():
    ei = acquisition_ei(np.array([2.0]), np.array([0.1]), y_best=1.0, xi=0.0)
    assert ei[0] > 0


# maximize_acquisition

def test_maximize_acquisition_returns_grid_point():
    X_grid = np.linspace(0, 1, 100)
    mu = np.sin(2 * np.pi * X_grid)
    next_x, acq = maximize_acquisition(
        X_grid, mu, np.ones_like(mu) * 0.1,
        acq_type='ucb', kappa=1.0, xi=0.01, y_best=0.0,
    )
    assert next_x in X_grid
    assert acq.shape == (100,)


def test_maximize_acquisition_invalid_type_raises():
    X_grid = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        maximize_acquisition(X_grid, np.zeros(10), np.ones(10),
                             acq_type='bad', kappa=1.0, xi=0.01, y_best=0.0)


# bo_step

def test_bo_step_output_shape():
    X_grid = np.linspace(0, 1, 50)
    result = bo_step(
        np.array([0.1, 0.5, 0.9]), np.array([0.5, 1.0, 0.3]), X_grid,
        length_scale=0.3, signal_var=1.0, noise_var=1e-4,
        acq_type='ucb', kappa=2.0, xi=0.01,
    )
    assert set(result.keys()) == {'mu', 'sigma', 'acq', 'next_x'}
    assert len(result['mu']) == 50
    assert isinstance(result['next_x'], float)


def test_bo_step_next_x_in_grid():
    X_grid = np.linspace(0, 1, 50)
    result = bo_step(
        np.array([0.5]), np.array([1.0]), X_grid,
        length_scale=0.3, signal_var=1.0, noise_var=1e-4,
        acq_type='ei', kappa=2.0, xi=0.01,
    )
    assert result['next_x'] in X_grid


# target functions

@pytest.mark.parametrize("fn", TARGET_FUNCTIONS.values())
def test_target_returns_finite_float(fn):
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = fn(x)
        assert isinstance(result, float)
        assert np.isfinite(result)
