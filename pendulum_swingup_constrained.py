"""
Starter code for the problem "Cart-pole swing-up with limited actuation".

Autonomous Systems Lab (ASL), Stanford University
"""

from functools import partial

from animations import animate_cartpole

import cvxpy as cvx

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm


@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def affinize(f, s, u):
    """Affinize the function `f(s, u)` around `(s, u)`.

    Arguments
    ---------
    f : callable
        A nonlinear function with call signature `f(s, u)`.
    s : numpy.ndarray
        The state (1-D).
    u : numpy.ndarray
        The control input (1-D).

    Returns
    -------
    A : jax.numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `s`.
    B : jax.numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `u`.
    c : jax.numpy.ndarray
        The offset term in the first-order Taylor expansion of `f` at `(s, u)`
        that sums all vector terms strictly dependent on the nominal point
        `(s, u)` alone.
    """
    # PART (b) ################################################################
    # INSTRUCTIONS: Use JAX to affinize `f` around `(s, u)` in two lines.
    # raise NotImplementedError()
    A = jax.jacobian(f, argnums=0)(s, u)
    B = jax.jacobian(f, argnums=1)(s, u)
    c = f(s, u) - A @ s - B @ u
    # END PART (b) ############################################################
    return A, B, c


def solve_swingup_scp(f, s0, s_goal, N, P, Q, R, u_max, ρ, eps, max_iters):
    """Solve the cart-pole swing-up problem via SCP.

    Arguments
    ---------
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    N : int
        The time horizon of the LQR cost function.
    P : numpy.ndarray
        The terminal state cost matrix (2-D).
    Q : numpy.ndarray
        The state stage cost matrix (2-D).
    R : numpy.ndarray
        The control stage cost matrix (2-D).
    u_max : float
        The bound defining the control set `[-u_max, u_max]`.
    ρ : float
        Trust region radius.
    eps : float
        Termination threshold for SCP.
    max_iters : int
        Maximum number of SCP iterations.

    Returns
    -------
    s : numpy.ndarray
        A 2-D array where `s[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u : numpy.ndarray
        A 2-D array where `u[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    J : numpy.ndarray
        A 1-D array where `J[i]` is the SCP sub-problem cost after the i-th
        iteration, for `i = 0, 1, ..., (iteration when convergence occured)`
    """
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize dynamically feasible nominal trajectories
    u = np.zeros((N, m))
    s = np.zeros((N + 1, n))
    s[0] = s0
    for k in range(N):
        s[k+1] = fd(s[k], u[k])

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    J = np.zeros(max_iters + 1)
    J[0] = np.inf
    for i in (prog_bar := tqdm(range(max_iters))):
        s, u, J[i + 1] = scp_iteration(f, s0, s_goal, s, u, N,
                                       P, Q, R, u_max, ρ)
        dJ = np.abs(J[i + 1] - J[i])
        prog_bar.set_postfix({'objective change': '{:.5f}'.format(dJ)})
        if dJ < eps:
            converged = True
            print('SCP converged after {} iterations with objective change {}'.format(i, dJ))
            break
    if not converged:
        raise RuntimeError('SCP did not converge!')
    J = J[1:i+1]
    return s, u, J


def scp_iteration(f, s0, s_goal, s_prev, u_prev, N, P, Q, R, u_max, ρ):
    """Solve a single SCP sub-problem for the cart-pole swing-up problem.

    Arguments
    ---------
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    s_prev : numpy.ndarray
        The state trajectory around which the problem is convexified (2-D).
    u_prev : numpy.ndarray
        The control trajectory around which the problem is convexified (2-D).
    N : int
        The time horizon of the LQR cost function.
    P : numpy.ndarray
        The terminal state cost matrix (2-D).
    Q : numpy.ndarray
        The state stage cost matrix (2-D).
    R : numpy.ndarray
        The control stage cost matrix (2-D).
    u_max : float
        The bound defining the control set `[-u_max, u_max]`.
    ρ : float
        Trust region radius.

    Returns
    -------
    s : numpy.ndarray
        A 2-D array where `s[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u : numpy.ndarray
        A 2-D array where `u[k]` is the open-loop state at time step `k`,
        for `k = 0, 1, ..., N-1`
    J : float
        The SCP sub-problem cost.
    """
    A, B, c = affinize(f, s_prev[:-1], u_prev)
    A, B, c = np.array(A), np.array(B), np.array(c)
    n = Q.shape[0]
    m = R.shape[0]
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    # PART (c) ################################################################
    # INSTRUCTIONS: Construct the convex SCP sub-problem.
    objective = 0.
    constraints = []
    constraints += [s_cvx[0] == s0]
    for t in range(N):
        objective += cvx.quad_form(s_cvx[t] - s_goal, Q) + cvx.quad_form(u_cvx[t], R)
        constraints += [s_cvx[t + 1] == A[t] @ s_cvx[t] + B[t] @ u_cvx[t] + c[t],
                        cvx.norm(s_cvx[t] - s_prev[t], 'inf') <= ρ,
                        cvx.norm(u_cvx[t] - u_prev[t], 'inf') <= ρ
                        ]
        # constraints += [cvx.abs(u_cvx[t]) <= u_max]
    objective += cvx.quad_form(s_cvx[-1] - s_goal, P)
    # raise NotImplementedError()
    # END PART (c) ############################################################

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()
    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)
    s = s_cvx.value
    u = u_cvx.value
    J = prob.objective.value
    print(J)
    return s, u, J

def pendulum(s, u):
    g = 10.
    m = 1.
    l = 1.

    noise = np.random.normal(scale=1.0)

    theta, dtheta = s
    ds = jnp.array([
        dtheta,
        (3. * g / (2. * l)) * jnp.sin(theta) + 3.0 / (m * l ** 2) * u[0] + noise
    ])

    return ds

def twod_quad(s, u):
    m = 0.027
    g = 10.0
    Iyy = 1.4e-5
    dt = 0.008
    l = 0.025
    x, dx, z, dz, theta, dtheta = s
    ds = jnp.array([
        dx,
        1 / m * (u[0] + u[1]) * jnp.sin(theta),
        dz,
        1 / m * (u[0] + u[1]) * jnp.cos(theta) - g,
        dtheta,
        1 / 2 / Iyy * (u[1] - u[0]) * l
    ])

    return ds

# def dot_states(states):
#     dot_states = np.hstack([states[1],
#                             1 / m * np.multiply(np.sum(action), np.sin(states[4])),
#                             states[3],
#                             1 / m * np.multiply(np.sum(action), np.cos(states[4])) - g,
#                             states[5],
#                             1 / 2 / Iyy * (action[1] - action[0]) * 0.025
#                             ])
#     return dot_states

def discretize(f, dt):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""

    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator


# Define constants
n = 2                                # state dimension
m = 1                                # control dimension
s_goal = np.array([np.pi, 0])  # desired upright pendulum state
s0 = np.array([0, 0])          # initial downright pendulum state
dt = 0.05                         # discrete time resolution
T = 10.                              # total simulation time
# P = 1e3*np.eye(n)                    # terminal state cost matrix
# Q = np.diag([1e-2, 1., 1e-3, 1e-3])  # state cost matrix
# R = 1e-3*np.eye(m)                   # control cost matrix
Q = np.diag(np.array([1., 0.1]))
R = 1e-2*np.eye(m)
P = Q
ρ = 2.                               # trust region parameter
u_max = 2.                           # control effort bound
eps = 5e-3                           # convergence tolerance
max_iters = 200                      # maximum number of SCP iterations
animate = False                      # flag for animation

# Initialize the discrete-time dynamics
fd = jax.jit(discretize(pendulum, dt))

# Solve the swing-up problem with SCP
t = np.arange(0., T + dt, dt)
N = t.size - 1
s, u, J = solve_swingup_scp(fd, s0, s_goal, N, P, Q, R, u_max, ρ,
                            eps, max_iters)

# Simulate open-loop control
for k in range(N):
    s[k+1] = fd(s[k], u[k])

# Plot state and control trajectories
fig, ax = plt.subplots(1, n + m, dpi=150, figsize=(15, 2))
plt.subplots_adjust(wspace=0.45)
labels_s = (r'$\theta(t)$', r'$\dot{\theta}(t)$')
labels_u = (r'$u(t)$',)
for i in range(n):
    ax[i].plot(t, s[:, i])
    ax[i].axhline(s_goal[i], linestyle='--', color='tab:orange')
    ax[i].set_xlabel(r'$t$')
    ax[i].set_ylabel(labels_s[i])
for i in range(m):
    ax[n + i].plot(t[:-1], u[:, i])
    ax[n + i].axhline(u_max, linestyle='--', color='tab:orange')
    ax[n + i].axhline(-u_max, linestyle='--', color='tab:orange')
    ax[n + i].set_xlabel(r'$t$')
    ax[n + i].set_ylabel(labels_u[i])
plt.savefig('pendulum_swingup_constrained.png',
            bbox_inches='tight')

# Plot cost history over SCP iterations
fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 5))
ax.semilogy(J)
ax.set_xlabel(r'SCP iteration $i$')
ax.set_ylabel(r'SCP cost $J(\bar{x}^{(i)}, \bar{u}^{(i)})$')
plt.savefig('pendulum_swingup_constrained_cost.png',
            bbox_inches='tight')
plt.show()

# Animate the solution
if animate:
    fig, ani = animate_cartpole(t, s[:, 0], s[:, 1])
    ani.save('cartpole_swingup_constrained.mp4', writer='ffmpeg')
    plt.show()
