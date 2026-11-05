"""Online DRMPC of Micheli et al. (2022): the Theorem 8 program in closed loop.

The problem spec is shared by make_controller and closed_loop_metrics, so a
notebook can define it once and pass it to both as **spec:

    state_weight, ref: cost h(y) = sum_t sum_j state_weight[j] * |x_t[j] - ref[j]|
    x_min, x_max:      per-state bounds, collected into one joint CVaR constraint
                       g(y) = max_k d_k y + f_k. Use +/-np.inf for "no bound".

Spec that reproduces the paper's figures:

    dict(state_weight=[1, 0], ref=[2, 0], x_min=[-np.inf, 0], x_max=[1, np.inf])

This differs from the paper's text on purpose. Eq. 11 prints a reference of 1, which
gives closed-loop costs about 10x below Figs. 1-2. Eq. 12 prints +y_(2), which has
no lower limit on the state, so the cost cannot saturate as in Fig. 1; the prose
says y_(2) must be positive. The spec above is inferred from the figures.
"""

import cvxpy as cp
import numpy as np


def make_controller(L_hat, Z, Y, n, m, eps1, eps2, state_weight, ref, x_min, x_max,
                    beta=0.2, slack_weight=1e6, u_max=None):
    """Compile the DR FHOC problem once; return controller(x) -> input plan (m*T,).

    eps1 = eps2 = 0 gives the sample-average controller (SAAMPC, eq. 5).
    u_max is an optional symmetric input bound, |u| <= u_max.

    controller raises RuntimeError if the solve fails. This usually means the closed
    loop diverged: with no input bound, a negative u_0 -> x_1 coefficient in L_hat can
    drive the state past 1e3, even at N=40. A rare plain solver failure at an ordinary
    state is also possible. Sweeps should catch the error and count the run as failed.
    """
    N, nT = Y.shape
    T = nT // n
    residuals = Y - Z @ L_hat.T  # eq. 3

    x0 = cp.Parameter(n)
    u = cp.Variable(m * T)
    t = cp.Variable()
    q = cp.Variable(N, nonneg=True)
    slack = cp.Variable(nonneg=True)

    z = cp.hstack([x0, u])
    predictions = np.ones((N, 1)) @ cp.reshape(L_hat @ z, (1, nT), order="F") + residuals  # eq. 4

    # ||a_j|| is the same for every sign pattern of the weighted 1-norm.
    weight = np.tile(np.asarray(state_weight, float), T)
    lam = np.linalg.norm(weight)
    # Every piece of g is a single state coordinate, so theta = max_k ||d_k|| = 1.
    D, f = _constraint_pieces(x_min, x_max, n, T)

    radius = eps2
    if eps1 > 0:
        z_copies = np.ones((N, 1)) @ cp.reshape(z, (1, Z.shape[1]), order="F")
        radius = radius + eps1 * cp.sum(cp.norm(z_copies - Z, axis=1)) / N

    cost = cp.sum(cp.abs(predictions - np.tile(ref, T)) @ weight) / N
    pieces = predictions @ D.T + f  # (N, K)
    constraints = [
        radius + cp.sum(q) / N <= t * beta + slack,
        cp.reshape(q, (N, 1), order="F") @ np.ones((1, len(f))) >= pieces + t,
    ]
    if u_max is not None:
        constraints.append(cp.abs(u) <= u_max)

    problem = cp.Problem(cp.Minimize(lam * radius + cost + slack_weight * slack), constraints)

    def controller(x):
        x0.value = x
        try:
            problem.solve(solver=cp.CLARABEL, canon_backend=cp.SCIPY_CANON_BACKEND)
        except cp.error.SolverError as e:
            raise RuntimeError(f"MPC solve failed at x={x}") from e
        if u.value is None:
            raise RuntimeError(f"MPC solve failed at x={x}: {problem.status}")
        return u.value

    return controller


def simulate(A, B, controller, x0, W):
    """Run the closed loop under the disturbance sequence W: (steps, n).

    Passing W in lets controllers be compared on the same noise realization.
    Returns X: (steps + 1, n) and U: (steps, m).
    """
    m = B.shape[1]
    X, U = [np.asarray(x0, float)], []
    for w in W:
        U.append(controller(X[-1])[:m])  # receding horizon: apply only u_0
        X.append(A @ X[-1] + B @ U[-1] + w)
    return np.array(X), np.array(U)


def closed_loop_metrics(X, state_weight, ref, x_min, x_max):
    """Return (cost, violation count) over x_1, ..., x_steps; x_0 is not controlled."""
    X = X[1:]
    cost = np.sum(np.abs(X - ref) @ np.asarray(state_weight, float))
    violations = np.sum(np.any((X > x_max) | (X < x_min), axis=1))
    return float(cost), int(violations)


def _constraint_pieces(x_min, x_max, n, T):
    """Write the finite state bounds over the horizon as rows of D y + f <= 0."""
    rows, offsets = [], []
    for t in range(T):
        for j in range(n):
            e = np.zeros(n * T)
            e[t * n + j] = 1.0
            if np.isfinite(x_max[j]):
                rows.append(e)
                offsets.append(-x_max[j])
            if np.isfinite(x_min[j]):
                rows.append(-e)
                offsets.append(x_min[j])
    return np.array(rows), np.array(offsets)
