import numpy as np

def collect_offline_trajectories(A, B, num_trajectories, T,
                                 noise_std=0.03, action_std=0.5, x0_std=0.5, rng=None):    
    """
    Collect offline trajectories from a linear dynamical system defined by matrices A and B with additive Gaussian noise.

    Parameters:
    A (np.ndarray): State transition matrix.
    B (np.ndarray): Control input matrix.
    num_trajectories (int): Number of trajectories to collect.
    T (int): Length of each trajectory.
    noise_std (float): Standard deviation of the Gaussian noise added to the state transitions.
    action_std (float): Standard deviation of the Gaussian noise added to the actions.
    x0_std (float): Standard deviation of the Gaussian noise added to the initial states.
    rng (np.random.Generator): Random number generator.

    Returns:
    tuple: A tuple representing (Z, Y) representing the collected trajectories, where Z contains the initial states and actions, 
            and Y contains the resulting states after applying the dynamics.
    """
    if rng is None:
        rng = np.random.default_rng()

    n,m = B.shape

    Z = np.zeros((num_trajectories, n + m * T))
    Y = np.zeros((num_trajectories, n * T))

    for i in range(num_trajectories):
        x = rng.normal(0.0, x0_std, size=n)              # x_0 ~ N(0, 0.5^2 I)
        u = rng.normal(0.0, action_std, size=(T, m))     # u_k ~ N(0, 0.5^2)
        w = rng.normal(0.0, noise_std, size=(T, n))      # w_k ~ N(0, 0.03^2 I)

        Z[i, :n] = x
        Z[i, n:] = u.ravel()

        for k in range(T):
            x = A @ x + B @ u[k] + w[k]                  # noise is part of the dynamics
            Y[i, k * n:(k + 1) * n] = x                  # store x_1 ... x_T
    
    return Z, Y