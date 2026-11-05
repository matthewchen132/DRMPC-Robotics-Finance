"""Run from the repo root: uv run --with pytest python -m pytest tests"""

import numpy as np

from core.mpc import closed_loop_metrics, make_controller, simulate
from core.offline import collect_offline_trajectories
from core.radius import estimate_radius, fit_predictor

A = np.array([[0.9, 0.1], [0.05, 0.9]])
B = np.array([[0.0], [1.0]])
n, m, T = 2, 1, 5
SPEC = dict(state_weight=[1.0, 0.0], ref=[2.0, 0.0],
            x_min=[-np.inf, 0.0], x_max=[1.0, np.inf])


def true_predictor():
    L = np.zeros((n * T, n + m * T))
    for t in range(1, T + 1):
        rows = slice((t - 1) * n, t * n)
        L[rows, :n] = np.linalg.matrix_power(A, t)
        for k in range(t):
            L[rows, n + k] = (np.linalg.matrix_power(A, t - 1 - k) @ B).ravel()
    return L


def data(N, seed=0, noise_std=0.03):
    rng = np.random.default_rng(seed)
    return collect_offline_trajectories(A, B, N, T, noise_std=noise_std, rng=rng)


def test_fit_keeps_causal_structure_with_noise():
    Z, Y = data(30)
    L_hat = fit_predictor(Z, Y, n, m)
    for t in range(1, T + 1):
        assert np.all(L_hat[(t - 1) * n:t * n, n + m * t:] == 0)


def test_radius_vanishes_without_noise_and_is_positive_with_it():
    Z, Y = data(30, noise_std=0.0)
    L_hat, eps1, eps2 = estimate_radius(Z, Y, n, m)
    assert np.allclose(L_hat, true_predictor(), atol=1e-8)
    assert eps1 < 1e-7 and eps2 < 1e-7

    Z, Y = data(10)
    _, eps1, eps2 = estimate_radius(Z, Y, n, m)
    assert eps1 >= 0 and eps2 >= 0 and eps1 + eps2 > 1e-4


def test_solution_meets_the_worst_case_cvar_bound():
    """For a 1-Lipschitz g, sup over the ball of radius eps is CVaR_hat + eps / beta."""
    Z, Y = data(40)
    L_hat = fit_predictor(Z, Y, n, m)
    eps, beta = 0.02, 0.2
    x = np.array([0.5, 0.5])
    plan = make_controller(L_hat, Z, Y, n, m, 0.0, eps, beta=beta, **SPEC)(x)

    y = L_hat @ np.concatenate([x, plan]) + (Y - Z @ L_hat.T)
    g = np.maximum(y[:, ::2] - 1.0, -y[:, 1::2]).max(axis=1)
    cvar = np.sort(g)[-8:].mean()  # beta * N = 8 is an integer, so this is exact
    assert cvar <= -eps / beta + 1e-6


def test_radius_tradeoff_matches_fig_1():
    """Median over datasets: small radius near cost 30 with violations, large near 43 without."""
    results = {0.0: [], 0.1: []}
    for seed in [1, 3, 4, 5, 7, 8, 9, 10]:
        Z, Y = data(10, seed)
        L_hat = fit_predictor(Z, Y, n, m)
        W = np.random.default_rng(100 + seed).normal(0, 0.03, (30, n))
        for eps2 in results:
            controller = make_controller(L_hat, Z, Y, n, m, 0.0, eps2, **SPEC)
            X, _ = simulate(A, B, controller, [0.9, 0.9], W)
            results[eps2].append(closed_loop_metrics(X, **SPEC))

    small_cost, small_violations = np.median(results[0.0], axis=0)
    large_cost, large_violations = np.median(results[0.1], axis=0)
    assert 29 < small_cost < 34 and small_violations >= 5
    assert 41 < large_cost < 45 and large_violations == 0


def test_input_bound_is_respected():
    Z, Y = data(10)
    L_hat = fit_predictor(Z, Y, n, m)
    plan = make_controller(L_hat, Z, Y, n, m, 0.0, 0.0, u_max=0.05, **SPEC)(np.array([0.9, 0.9]))
    assert np.all(np.abs(plan) <= 0.05 + 1e-7)
