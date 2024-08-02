"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    OCP example for lane keeping problem in a circle road

    [Method]
    Model predictive control

"""
import numpy as np
from  casadi import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from datetime import datetime
from tqdm import tqdm
import os

class Solver(object):
    """
    NLP solver for nonlinear model predictive control with Casadi.
    """
    def __init__(self):

        self._sol_dic = {'ipopt.print_level': 1, 'ipopt.sb': 'yes', 'print_time': 0}
        self.state_dim = 6
        self.action_dim = 2
        self.u_max = [1.0, 1.0]

    def dynamics(self,s, a):
        l = 4.9276  # tractor length
        R = 8.5349  # turning radius
        d1 = 15.8496  # trailer length
        dt = 0.05

        x, y, th0, dth, v, delta = s[0], s[1], s[2], s[3], s[4], s[5]
        acc, delta_rate = 2 * a[0], np.pi / 12 * a[1]
        normalized_steer = tan(delta) * R / l
        return vertcat(x + dt * v * cos(th0),
        y + dt * v * sin(th0),
        th0 + dt * v * normalized_steer / R,
        dth - dt * v * (d1 * normalized_steer + sin(dth) * R) / (R * d1),
        v + dt * acc,
        delta + dt * delta_rate)

    def cost(self, s, a):

        x, y, th0, dth, v, delta = s [0], s [1], s [2], s [3], s [4], s [5]
        acc, delta_rate = a [0], a [1]

        return 1e-3 * (x ** 2 + y ** 2 + 10 * th0 ** 2 + 10 * dth ** 2) + 1e-2 * (acc ** 2 + delta_rate ** 2)

    def terminal_cost(self, s, a):

        x, y, th0, dth, v, delta = s [0], s [1], s [2], s [3], s [4], s [5]
        acc, delta_rate = a [0], a [1]

        return 100 * x ** 2 + 100 * y ** 2 + 1000 * th0 ** 2 + 1000 * dth ** 2

    def single_solve(self, x_init, predictive_steps):
        r = self.solve(x_init, predictive_steps)
        return self.extract_solution(r, predictive_steps)

    def solve(self, x_init, predict_steps):
        """
        Solver of nonlinear MPC

        Parameters
        ----------
        x_init: list
            input state for MPC.
        predict_steps: int
            steps of predict horizon.

        Returns
        ----------
        state: np.array     shape: [predict_steps+1, state_dimension]
            state trajectory of MPC in the whole predict horizon.
        control: np.array   shape: [predict_steps, control_dimension]
            control signal of MPC in the whole predict horizon.
        """
        x = SX.sym('x', self.state_dim)
        u = SX.sym('u', self.action_dim)

        # Create solver instance
        self.F = Function("F", [x, u], [self.dynamics(x, u)])

        # Create empty NLP
        w = []
        lbw = []
        ubw = []
        lbg = []
        ubg = []
        G = []
        J = 0

        # Initial conditions
        Xk = MX.sym('X0', self.state_dim)
        w += [Xk]
        lbw += x_init
        ubw += x_init

        for k in range(1, predict_steps + 1):
            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, self.action_dim)
            w += [Uk]
            lbw += [ -1 * u for u in self.u_max]
            ubw += self.u_max

            Fk = self.F(Xk, Uk)
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.state_dim)

            # Dynamic Constriants
            G += [Fk - Xk]
            lbg += [0.0] * self.state_dim
            ubg += [0.0] * self.state_dim
            w += [Xk]
            # if self.tire_model == 'Fiala':
            lbw += [-inf] * (self.state_dim - 3) + [- np.pi / 2, - 2.0, - np.pi/6]
            ubw += [inf] * (self.state_dim - 3) + [np.pi / 2, 2.0, np.pi/6]
            # else:
            #     lbw += [-inf, -20, -pi, -20, -inf]
            #     ubw += [inf, 20, pi, 20, inf]
            if k != predict_steps:
                F_cost = Function('F_cost', [x, u], [self.cost(x, u)])
                J += F_cost(w[k * 2], w[k * 2 - 1])
            else:
                # terminal cost
                T_cost = Function('F_cost', [x, u], [self.terminal_cost(x, u)])
                J += T_cost(w [k * 2], w [k * 2 - 1])


        # Create NLP solver
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)

        # Solve NLP
        r = S(lbx=lbw, ubx=ubw, x0=0, lbg=lbg, ubg=ubg)
        # print(r['x'])

        return r

    def check_feasible(self, r):

        feasible = r['g'].full().flatten().max()
        if feasible > 1e-6:
            print("not feasible")
            return False
        terminal = r['f'].full()[0][0]
        if terminal > 20:
            print(terminal)
            return False
        return True
    def extract_solution(self, r, predict_steps):
        state_all = np.array(r ['x'])
        state = np.zeros([predict_steps, self.state_dim])
        control = np.zeros([predict_steps, self.action_dim])
        nt = self.state_dim + self.action_dim  # total variable per step

        # save trajectories
        for i in range(predict_steps):
            state [i] = state_all [nt * i: nt * i + self.state_dim].reshape(-1)
            control [i] = state_all [nt * i + self.state_dim: nt * i + self.state_dim + self.action_dim].reshape(-1)
        return state, control

    def generate_dataset(self, grid_size=3):
        x = np.linspace(2, 5, grid_size)
        y = np.linspace(0, 3, grid_size)
        th0 = np.linspace(-np.pi / 12, np.pi / 12, grid_size)
        # Create the grid
        X, Y, TH0 = np.meshgrid(x, y, th0, indexing='ij')

        grid_x = X.ravel()
        grid_y = Y.ravel()
        grid_th0 = TH0.ravel()
        grid_dth = -1 * grid_th0
        grid_v = np.zeros_like(grid_x)
        grid_delta = np.zeros_like(grid_x)

        init_states = np.vstack([grid_x, grid_y, grid_th0, grid_dth, grid_v, grid_delta]).T
        total_init_nums = len(init_states)
        feasible_points = 0
        feasible_initials = []
        states = []
        controls = []

        for x_init in tqdm(init_states):
            casadi_sol = self.solve(x_init.tolist(), predict_steps=240)
            feasible = self.check_feasible(casadi_sol)
            if feasible:
                feasible_points += 1
                state, control = self.extract_solution(casadi_sol, predict_steps=240)
                feasible_initials.append(np.ones([1, ]))
                states.append(state)
                controls.append(control)
            else:
                feasible_initials.append(np.zeros([1, ]))
                print(x_init)

        states = np.vstack(states)
        controls = np.vstack(controls)
        feasible_initials = np.vstack(feasible_initials)

        self.create_dataset_and_save(states, controls, init_states, feasible_initials, feasible_points / total_init_nums, grid_size)

        print(f"feasible rate: {feasible_points / total_init_nums}")


    def create_dataset_and_save(self, states, actions, x_init, feasible_initials, rate, size):
        import torch
        dataset = SupervisedParkingDataset(states, actions)
        feasibility_dataset = InitialStateFeasibilityDataset(x_init, feasible_initials)
        # Get current date and time
        now = datetime.now()

        # Format date and time
        formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs('./datas/{}'.format(formatted_now), exist_ok=True)
        torch.save(dataset, './datas/{}'.format(formatted_now) + f'/{str(size)}_{rate:.3f}_{len(states)}.pt')
        # np.save('./datas/{}'.format(formatted_now) + "/init.npy", initials)
        torch.save(feasibility_dataset, './datas/{}'.format(formatted_now) + f'/feasibility.pt')

class SupervisedParkingDataset(Dataset):

    def __init__(self, states, controls):
        self.states = states
        self.controls = controls

    def __len__(self):
        assert len(self.states) == len(self.controls)
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.controls[idx]


class InitialStateFeasibilityDataset(Dataset):

    def __init__(self, states, feasible):
        self.states = states
        self.feasible = feasible

    def __len__(self):
        assert len(self.states) == len(self.feasible)
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.feasible[idx]


def try_openloop_solver():
    from articulate_fh import ArticulateParkingInfiniteHorizon
    env = ArticulateParkingInfiniteHorizon(render_mode='human')
    x_init = [ 2.   ,       0.5    ,     -0.26179939 , 0.26179939 , 0.       ,   0.        ]
    env.reset(options={'state': np.array(x_init)})
    solver = Solver()
    state, control = solver.single_solve(x_init=x_init, predictive_steps=240)
    from vehicle_render import Renderer
    renderer = Renderer(vehicle_length=4.9276, trailer_length=15.8496)
    for i in range(len(state)):
        renderer.set_state(state[i])
        renderer.render()


if __name__ == '__main__':
    # try_openloop_solver()
    solver = Solver()
    solver.generate_dataset(grid_size=15)