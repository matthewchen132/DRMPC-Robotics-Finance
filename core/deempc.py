"""DeePC of Coulson, Lygeros & Dörfler (2019), https://arxiv.org/pdf/1811.05890, on DRMPC's data.

Data matrix. The paper slides windows over one long trajectory (a Hankel matrix). Here each
offline trajectory from collect_offline_trajectories is one column of the Hankel matrix,
so DeePC and DRMPC learn from exactly the same samples, and the unstable cart-pole never
has to be driven open loop for hundreds of steps.

Initial condition. The paper's past blocks [U_p; Y_p] g = [u_ini; y_ini] exist to pin down
the unmeasured state. With y = full state, the state is measured, so they reduce to
X0 g = x_t: each column's starting state x_0 plays the role of the past window.

    X0 g = x_t      (n rows)    initial condition, replaces U_p, Y_p
    Uf g = u        (m*T rows)  future inputs
    Yf g = y        (n*T rows)  predicted future states x_1..x_T

Cost and bounds use the same spec as drmpc.make_controller, with the CVaR constraint
replaced by hard state bounds plus one slack, so the methods differ only in how they predict.
"""

import cvxpy as cp
import numpy as np


def data_matrices(Z, Y, n):
    """Split DRMPC data Z = [x_0, u], Y = [x_1..x_T] (one row per trajectory) into X0, Uf, Yf."""
    return Z[:, :n].T, Z[:, n:].T, Y.T


def fundamental_lemma_rank(Z):
    """Rank of [X0; Uf]. The lemma needs n + m*T (every (x_0, u) direction covered)."""
    return np.linalg.matrix_rank(Z.T), Z.shape[1]


def make_controller(Z, Y, n, lambda_g, state_weight, ref, x_min, x_max,
                    slack_weight=1e6, u_max=None):
    """Compile the regularized DeePC problem once; return controller(x) -> input plan (m*T,).

    lambda_g weights ||g||_1 (paper eq. 8). With noisy data, g can steer the prediction along
    noise directions that are not real dynamics; the penalty limits how far it can.
    """
    X0, Uf, Yf = data_matrices(Z, Y, n)
    T = Yf.shape[0] // n

    x0 = cp.Parameter(n)
    g = cp.Variable(X0.shape[1])
    slack = cp.Variable(nonneg=True)
    u, y = Uf @ g, Yf @ g

    weight = np.tile(np.asarray(state_weight, float), T)
    cost = cp.abs(y - np.tile(ref, T)) @ weight
    hi, lo = np.tile(np.asarray(x_max, float), T), np.tile(np.asarray(x_min, float), T)
    upper, lower = np.isfinite(hi), np.isfinite(lo)

    constraints = [X0 @ g == x0, y[upper] <= hi[upper] + slack, y[lower] >= lo[lower] - slack]
    if u_max is not None:
        constraints.append(cp.abs(u) <= u_max)

    problem = cp.Problem(cp.Minimize(cost + lambda_g * cp.norm1(g) + slack_weight * slack), constraints)

    def controller(x):
        x0.value = x
        try:
            problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError as e:
            raise RuntimeError(f"DeePC solve failed at x={x}") from e
        if g.value is None:
            raise RuntimeError(f"DeePC solve failed at x={x}: {problem.status}")
        return Uf @ g.value

    return controller