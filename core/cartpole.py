"""Gymnasium's CartPole-v1, linearized about upright and discretized with its Euler step.

State [x, x_dot, theta, theta_dot], input u = horizontal force on the cart (N).
Feed (A, B) to collect_offline_trajectories and simulate like any other linear system.
"""

import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


def cartpole():
    """Return (A, B) of x_{k+1} = A x_k + B u_k, exact for the env's dynamics at theta = 0."""
    env = CartPoleEnv()
    M, mp, half_len, g, tau = env.total_mass, env.masspole, env.length, env.gravity, env.tau
    d = half_len * (4 / 3 - mp / M)
    th_th, th_u = g / d, -1 / (M * d)  # theta_acc = th_th * theta + th_u * u
    x_th, x_u = -mp * half_len * th_th / M, 1 / M - mp * half_len * th_u / M
    Ac = np.array([[0, 1, 0, 0], [0, 0, x_th, 0], [0, 0, 0, 1], [0, 0, th_th, 0]])
    Bc = np.array([[0], [x_u], [0], [th_u]])
    return np.eye(4) + tau * Ac, tau * Bc
