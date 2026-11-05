"""Multi-step predictor identification and Algorithm 1 of Micheli et al. (2022)."""

import cvxpy as cp
import numpy as np


def fit_predictor(Z, Y, n, m):
    """Least-squares T-step predictor (eq. 13) with the causal structure of L-bar.

    Each block row t only regresses on x_0 and u_0, ..., u_{t-1}; every other
    entry of L_hat stays zero.

    Z: (N, n + m*T) stacked [x_0, u_0, ..., u_{T-1}], as from offline.py.
    Y: (N, n*T) stacked [x_1, ..., x_T].
    Returns L_hat: (n*T, n + m*T).
    """
    T = Y.shape[1] // n
    L_hat = np.zeros((n * T, Z.shape[1]))
    for t in range(1, T + 1):
        causal = n + m * t
        rows = slice((t - 1) * n, t * n)
        L_hat[rows, :causal] = np.linalg.lstsq(Z[:, :causal], Y[:, rows], rcond=None)[0].T
    return L_hat


def estimate_radius(Z, Y, n, m):
    """Algorithm 1: leave-one-out estimate of the predictor and radius parameters.

    Returns (L_hat, eps1, eps2), where L_hat is the mean of the leave-one-out
    predictors and the radius is eps(z) = eps1 * mean_i ||z - z^i|| + eps2 (eq. 8).
    """
    N = len(Z)
    predictors, V, E = [], np.zeros(N), np.zeros(N)
    for out in range(N):
        keep = np.arange(N) != out
        L_out = fit_predictor(Z[keep], Y[keep], n, m)
        residuals = Y[keep] - Z[keep] @ L_out.T
        predictions = L_out @ Z[out] + residuals
        predictors.append(L_out)
        V[out] = np.linalg.norm(Z[out] - Z[keep], axis=1).sum() / (N - 1)
        E[out] = np.linalg.norm(Y[out] - predictions, axis=1).sum() / (N**2 - N)

    eps = cp.Variable(2, nonneg=True)
    problem = cp.Problem(cp.Minimize(cp.norm1(eps[0] * V + eps[1] - E)))
    problem.solve(solver=cp.CLARABEL, canon_backend=cp.SCIPY_CANON_BACKEND)
    if eps.value is None:
        raise RuntimeError(f"Radius fit failed: {problem.status}")
    eps1, eps2 = np.maximum(eps.value, 0.0).tolist()  # the solver can return -1e-12
    return np.mean(predictors, axis=0), eps1, eps2
