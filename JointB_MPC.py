# This file to try to run MPC on the joint B linear actuator.
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.optimize import minimize
from casadi import *

# Time parameters
dT = 0.05  # Time step (seconds)
N = 20  # Prediction horizon (number of steps)

# Circuit constants
R = 1e6  # Resistance (Ohms)
Ra = 35.0  # Resistance of armature (Ohms)
La = 1.2e-3  # Inductance of motor (Henries)
L = 10e-3  # Inductance
C = 100e-6  # Capacitance (Farads)
J = 6e-7  # Inertia (kg*m^2)
b_damp = 1e-5  # Viscous damping (N*m*s/rad)
Kt = 0.032  # Torque constant (N*m/A)
Ke = 0.0435  # Back EMF constant (V*s/rad)
Vin = 36.0  # Input voltage (Volts)

# Add position to state variables
il = MX.sym("il")
vc = MX.sym("vc")
ila = MX.sym("ila")
w = MX.sym("w")
theta = MX.sym("theta") # NEW STATE
x = vertcat(il, vc, ila, w, theta) # Now a 5x1 vector

# Control input
u1 = MX.sym("u1")
u2 = MX.sym("u2")
u = vertcat(u1, u2)

A_mat = vertcat(
    horzcat(0, -1/L, 0, 0, 0),
    horzcat(1/C, -1/(R*C), -1/C, 0, 0),
    horzcat(0, -1/La, -Ra/La, -Ke/La, 0),
    horzcat(0, 0, Kt/J, -b_damp/J, 0),
    horzcat(0, 0, 0, 1, 0)  # d(theta)/dt = w
)

B_mat = vertcat(
    horzcat(0, 0, 0, 0, 0),
    horzcat(0, 0, 2/C, 0, 0),
    horzcat(0, 2/La, 0, 0, 0),
    horzcat(0, 0, 0, 0, 0),
    horzcat(0, 0, 0, 0, 0)
)

C_mat = vertcat(Vin/L, 0, 0, 0, 0)

# Update the continuous dynamics function
f = Function('f', [x, u], [A_mat @ x + B_mat @ x * u2 + C_mat * u1])

k1 = f(x, u)
k2 = f(x + dT/2 * k1, u)
k3 = f(x + dT/2 * k2, u)
k4 = f(x + dT * k3, u)

x_next = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

# Discrete time dynamics function
F_discrete = Function('F_discrete', [x, u], [x_next])

# Optimization
opti = Opti()

X = opti.variable(5, N+1)  # State trajectory
U = opti.variable(2, N)    # Control trajectory

# Parameter for the initial state (so we can update it in our real-time control loop)
x_initial = opti.parameter(5, 1)
opti.set_value(x_initial, np.zeros((5, 1)))  # Initial state

# Constraints
opti.subject_to(X[:, 0] == x_initial)  # Initial state constraint


# Add a weight for position (index 4) - let's make it the highest priority
Q = np.diag([1.0, 1.0, 5.0, 5.0, 20.0])  
R = np.diag([0.1, 0.1])  

# New physically valid target: Stop moving at an angle of 3.14 radians (180 degrees)
target_angle = np.pi 
x_target = vertcat(0, 0, 0, 0, target_angle)

cost = 0

for k in range(N):
    opti.subject_to(X[:, k + 1] == F_discrete(X[:, k], U[:, k]))  # System dynamics constraints
    
    # Bounding of duty cycle
    opti.subject_to(opti.bounded(0, U[0, k], 1))  # Control input constraints
    opti.subject_to(opti.bounded(0, U[1, k], 1))  # Control input constraints

    # Cost function
    state_error = X[:, k] - x_target
    cost += state_error.T @ Q @ state_error + U[:, k].T @ R @ U[:, k]

# Terminal cost
state_error_terminal = X[:, N] - x_target
cost += state_error_terminal.T @ Q @ state_error_terminal

opti.minimize(cost)

opts = {"ipopt.print_level": 0, "print_time": 0}
opti.solver("ipopt", opts)

try:
    sol = opti.solve()
    print(sol.stats()["iter_count"])
    X_result = sol.value(X)
    U_result = sol.value(U)
    for k in range(N):
        print(f"Time step {k}: Control input = {sol.value(U[:, k])}, State = {sol.value(X[:, k])}")
except RuntimeError:
    print("Solver failed to converge. Catching debug values... 🪲")
    X_debug = opti.debug.value(X)
    U_debug = opti.debug.value(U)

    # Print the state trajectory to see where it went wrong
    print("\n--- Failed State Trajectory (X) ---")
    # Rounding makes the matrix much easier to read in the terminal
    print(np.round(X_debug, 3))




