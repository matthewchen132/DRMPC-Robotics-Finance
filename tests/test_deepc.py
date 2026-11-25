"""Run from the repo root: uv run --with pytest python -m pytest tests"""

import numpy as np

from core.deepc import data_matrices, fundamental_lemma_rank, make_controller
from core.drmpc import simulate
from core.generate_rng import trial_rngs
from core.offline import collect_offline_trajectories

A = np.array([[0.9, 0.1], [0.05, 0.9]])
B = np.array([[0.0], [1.0]])
n, m, T = 2, 1, 5
SPEC = dict(state_weight=[1.0, 0.0], ref=[2.0, 0.0], x_min=[-np.inf, 0.0], x_max=[1.0, np.inf])


def true_predictor():
    L = np.zeros((n * T, n + m * T))
    for t in range(1, T + 1):
        rows = slice((t - 1) * n, t * n)
        L[rows, :n] = np.linalg.matrix_power(A, t)
        for k in range(t):
            L[rows, n + k] = (np.linalg.matrix_power(A, t - 1 - k) @ B).ravel()
    return L


def data(N, noise_std):
    rng, _ = trial_rngs(0, 0)
    return collect_offline_trajectories(A, B, N, T, noise_std=noise_std, rng=rng)


def test_fundamental_lemma_any_g_predicts_exactly_without_noise():
    """Noise-free: every g matching (x_0, u) gives the true prediction, not just one."""
    Z, Y = data(20, noise_std=0.0)
    rank, needed = fundamental_lemma_rank(Z)
    assert rank == needed
    X0, Uf, Yf = data_matrices(Z, Y, n)
    z = np.random.default_rng(1).normal(size=n + m * T)
    g = np.linalg.lstsq(Z.T, z, rcond=None)[0]
    null_direction = np.linalg.svd(Z.T)[2][-1]  # Z.T @ null_direction = 0
    for g_any in (g, g + 5 * null_direction):
        assert np.allclose(Z.T @ g_any, z)
        assert np.allclose(Yf @ g_any, true_predictor() @ z)


def test_minimum_norm_g_is_least_squares_identification():
    """With noise, the smallest-norm g reproduces the unstructured least-squares predictor."""
    Z, Y = data(20, noise_std=0.03)
    _, _, Yf = data_matrices(Z, Y, n)
    L_ls = np.linalg.lstsq(Z, Y, rcond=None)[0].T
    z = np.random.default_rng(1).normal(size=n + m * T)
    assert np.allclose(Yf @ np.linalg.pinv(Z.T) @ z, L_ls @ z)


def test_controller_respects_input_bound_in_closed_loop():
    Z, Y = data(20, noise_std=0.03)
    controller = make_controller(Z, Y, n, lambda_g=1.0, u_max=0.5, **SPEC)
    plan = controller(np.array([0.9, 0.9]))
    assert plan.shape == (m * T,)
    _, U = simulate(A, B, controller, [0.9, 0.9], np.zeros((10, n)))
    assert np.all(np.abs(U) <= 0.5 + 1e-6)
