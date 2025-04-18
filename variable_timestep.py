import casadi as ca

# Parameters
N = 10                       # Horizon length
x0_val = 0.0                 # Initial state
x_goal = 5.0                 # Goal state
u_min, u_max = -1.0, 1.0     # Control limits
dt_min, dt_max = 0.01, 1.0   # Time step bounds

# Variables
x = ca.MX.sym("x", N + 1)    # States x[0] to x[N]
u = ca.MX.sym("u", N)        # Controls u[0] to u[N-1]
dt = ca.MX.sym("dt")         # Time step (scalar)

# Objective and constraints
cost = 0
g = []

# Initial condition
g.append(x[0] - x0_val)

# Dynamics and cost
for k in range(N):
    x_next = x[k] + dt * u[k]
    g.append(x[k + 1] - x_next)
    cost += (u[k])**2         # Quadratic control cost

# Terminal cost (distance to goal)
cost += 100 * (x[N] - x_goal)**2

# Optimization variables and bounds
opt_vars = ca.vertcat(x, u, dt)

# Variable bounds
lbx = [-ca.inf] * (N + 1) + [u_min] * N + [dt_min]
ubx = [ ca.inf] * (N + 1) + [u_max] * N + [dt_max]

# Constraint bounds (equality constraints)
lbg = [0] * (N + 1)
ubg = [0] * (N + 1)

# Solver setup
nlp = {"x": opt_vars, "f": cost, "g": ca.vertcat(*g)}
solver = ca.nlpsol("solver", "ipopt", nlp)

# Solve
sol = solver(x0=[0]*(2*N+2), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
opt = sol["x"].full().flatten()

# Extract solutions
x_opt = opt[0:N+1]
u_opt = opt[N+1:2*N+1]
dt_opt = opt[-1]

# Print result
print("Optimal states:", x_opt)
print("Optimal controls:", u_opt)
print("Optimal time step:", dt_opt)
print("Total time:", dt_opt * N)