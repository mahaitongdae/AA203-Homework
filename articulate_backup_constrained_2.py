"""
Starter code for the problem "Cart-pole swing-up with limited actuation".

Autonomous Systems Lab (ASL), Stanford University
"""
import os
from functools import partial

from animations import animate_cartpole

import cvxpy as cvx

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm

from datetime import datetime

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


def solve_swingup_scp(f, s0, s_goal, N, P, Q, R, u_max, x_max, ρ, eps, max_iters,
                      initial_guess_s = None,
                      initial_guess_u = None):
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
    if initial_guess_s is None and initial_guess_u is None:
        u = np.zeros((N, m))
        s = np.zeros((N + 1, n))
    else:
        u = initial_guess_u
        s = initial_guess_s
    s[0] = s0
    for k in range(N):
        s[k+1] = fd(s[k], u[k])

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    J = np.zeros(max_iters + 1)
    J[0] = np.inf
    for i in (prog_bar := tqdm(range(max_iters))):
        s, u, J[i + 1] = scp_iteration(f, s0, s_goal, s, u, N,
                                       P, Q, R, u_max, x_max, ρ)
        dJ = np.abs(J[i + 1] - J[i])
        prog_bar.set_postfix({'objective change': '{:.5f}'.format(dJ)})
        prog_bar.set_postfix({'current objective': '{:.5f}'.format(J[i + 1])})
        if dJ < eps:
            converged = True
            print('SCP converged after {} iterations with objective change {}'.format(i, dJ))
            break
    if not converged:
        raise RuntimeError('SCP did not converge!')
    J = J[1:i+1]
    return s, u, J


def scp_iteration(f, s0, s_goal, s_prev, u_prev, N, P, Q, R, u_max, x_max, ρ):
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
        # if t > 0:
        #     objective += cvx.quad_form(u_cvx[t] - u_cvx[t-1], 10 * R)
        constraints += [s_cvx[t + 1] == A[t] @ s_cvx[t] + B[t] @ u_cvx[t] + c[t],
                        cvx.norm(s_cvx[t] - s_prev[t], 'inf') <= ρ,
                        cvx.norm(u_cvx[t] - u_prev[t], 'inf') <= ρ,
                        cvx.abs(u_cvx[t]) <= u_max,
                        # cvx.abs(s_cvx[t][3]) <= np.pi / 3,
                        cvx.abs(s_cvx[t][2]) <= np.pi / 4,
                        # s_cvx[t][0] >= 0,
                        # s_cvx[t][1] >= 0,
                        cvx.abs(s_cvx[t][4]) <= 0.6,
                        cvx.abs(s_cvx[t][5]) <= np.pi / 6,
                        ]
    objective += cvx.quad_form(s_cvx[-1] - s_goal, P)
    # raise NotImplementedError()
    # END PART (c) ############################################################

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve(solver=cvx.OSQP, verbose=False, max_iter=int(1e6))
    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)
    s = s_cvx.value
    u = u_cvx.value
    J = prob.objective.value
    # print(J)
    return s, u, J


def cartpole(s, u):
    """Compute the cart-pole state derivative."""
    mp = 0.1     # pendulum mass
    mc = 1.0     # cart mass
    L = 0.5      # pendulum length
    g = 9.81    # gravitational acceleration

    x, θ, dx, dθ = s
    sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)
    # h = mc + mp*(sinθ**2)
    # ds = jnp.array([
    #     dx,
    #     dθ,
    #     (mp*sinθ*(L*(dθ**2) + g*cosθ) + u[0]) / h,
    #     -((mc + mp)*g*sinθ + mp*L*(dθ**2)*sinθ*cosθ + u[0]*cosθ) / (h*L)
    # ])
    h = (u[0] + mp * L * dθ ** 2 * sinθ) / (mp + mc)
    ds = jnp.array([
        dx,
        dθ,
        h - mp * L * dθ * cosθ / (mc + mp),
        (g * sinθ - h * cosθ) / (L * (4./3. - mp * cosθ ** 2 / (mc + mp)))
    ])
    return ds

def articulate_one_trailer(s, u):

    # L = 4.9276
    d1 = 15.8496
    # delta_max = np.pi / 6
    R = 8.5349

    x, y, th0, dth = s
    v, delta = u

    ds = jnp.array([
        v * jnp.cos(th0),
        v * jnp.sin(th0),
        v * delta / R,
        -1 * v * (d1 * delta + jnp.sin(dth) * R) / (R * d1),
    ])

    return ds

def articulate_one_trailer_integrator(s, u):

    l = 4.9276       #tractor length
    R = 8.5349       #turning radius
    d1 = 15.8496           # trailer length

    x, y, th0, dth, v, delta = s
    acc, delta_rate = u
    normalized_steer = jnp.tan(delta) * R / l

    ds = jnp.array([
        v * jnp.cos(th0),
        v * jnp.sin(th0),
        v * normalized_steer / R,
        -1 * v * (d1 * normalized_steer + jnp.sin(dth) * R) / (R * d1),
        acc,
        delta_rate
    ])

    return ds

def discretize(f, dt):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""

    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator

if __name__ == '__main__':

    # Define constants
    n = 6                                # state dimension
    m = 2                                # control dimension
    s_goal = np.array([0, 0, 0, 0, 0, 0])  # desired upright pendulum state
    s0 = np.array([2, 0.5, 0, 0, 0, 0])          # initial downright pendulum state
    dt = 0.05                             # discrete time resolution
    T = 12.0                               # total simulation time
    Q = 1e-3 * np.diag([1., 1., 10., 10., 0, 0])  # state cost matrix
    P = np.diag([100, 100, 5000, 5000, 0, 0])                  # terminal state cost matrix

    R = 1e-2*np.eye(m)                   # control cost matrix
    # Q = np.diag(np.array([10., 10., 2., 2.]))
    # R = 1e-2*np.eye(m)
    # P = 1e2*np.eye(n)
    ρ = 0.5                              # trust region parameter
    u_max = np.array([1.0, np.pi / 24])                        # control effort bound
    x_max = np.array([np.inf, np.inf, np.inf, np.pi / 4])
    eps = 5e-2                          # convergence tolerance
    max_iters = 300                      # maximum number of SCP iterations
    animate = True                      # flag for animation

    initial_guess_dir = None # "2024-07-08_14-31-24"

    if initial_guess_dir is not None:
        s_init = np.load(os.path.join(f"./logs/{initial_guess_dir}", "s.npy"))
        u_init = np.load(os.path.join(f"./logs/{initial_guess_dir}", "u.npy"))

    # Initialize the discrete-time dynamics
    fd = jax.jit(discretize(articulate_one_trailer_integrator, dt))

    # Solve the swing-up problem with SCP
    t = np.arange(0., T + dt, dt)
    N = t.size - 1
    if initial_guess_dir is not None:
        s, u, J = solve_swingup_scp(fd, s0, s_goal, N, P, Q, R, u_max, x_max, ρ,
                                eps, max_iters, initial_guess_s=s_init,
                                    initial_guess_u=u_init)
    else:
        s, u, J = solve_swingup_scp(fd, s0, s_goal, N, P, Q, R, u_max, x_max, ρ,
                                    eps, max_iters)

    # Simulate open-loop control
    for k in range(N):
        s[k+1] = fd(s[k], u[k])

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = f'./logs/{current_time}'
    os.makedirs(file_path, exist_ok=True)

    # Plot state and control trajectories
    fig, ax = plt.subplots(1, n + m, dpi=150, figsize=(15, 2))
    plt.subplots_adjust(wspace=0.45)
    labels_s = (r'$x(t)$', r'$y(t)$', r'$\theta_0{x}(t)$', r'$\theta_0 - \theta_1(t)$', r'v(t)', 'delta(t)')
    labels_u = (r'$a(t)$',r'$ddelta(t)$')
    for i in range(n):
        ax[i].plot(t, s[:, i])
        ax[i].axhline(s_goal[i], linestyle='--', color='tab:orange')
        ax[i].set_xlabel(r'$t$')
        ax[i].set_title(labels_s[i])
    for i in range(m):
        ax[n + i].plot(t[:-1], u[:, i])
        ax[n + i].axhline(u_max[i], linestyle='--', color='tab:orange')
        ax[n + i].axhline(-u_max[i], linestyle='--', color='tab:orange')
        ax[n + i].set_xlabel(r'$t$')
        ax[n + i].set_title(labels_u[i])
    plt.savefig(os.path.join(file_path, 'vehicle_backup_constrained.png'),
                bbox_inches='tight')

    # Plot cost history over SCP iterations
    # fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 5))
    # ax.semilogy(J)
    # ax.set_xlabel(r'SCP iteration $i$')
    # ax.set_ylabel(r'SCP cost $J(\bar{x}^{(i)}, \bar{u}^{(i)})$')
    # plt.savefig('cartpole_swingup_constrained_cost.png',
    #             bbox_inches='tight')
    # plt.show()


    np.save(file_path + '/s.npy', s)
    np.save(file_path + '/u.npy', u)


    # Animate the solution
    if animate:
        from vehicle_render import Renderer
        renderer = Renderer(vehicle_length=4.9276, trailer_length= 15.8496)
        for i in range(len(s)):
            renderer.set_state(s[i])
            renderer.render()

        # fig, ani = animate_cartpole(t, s[:, 0], s[:, 1])
        # ani.save('cartpole_swingup_constrained.mp4', writer='ffmpeg')
        # plt.show()
