# Cascade Architecture: Outer Loop MPC
# Generates current and voltage setpoints for an inner SMC loop.
import numpy as np
from casadi import *

# Time parameters
dT = 0.05  # Time step (seconds) runs much slower than SMC
N = 20     # Prediction horizon (number of steps)

# Circuit & Motor constants
Ra = 35.0    # Resistance of armature (Ohms)
J = 6e-7     # Inertia (kg*m^2)
b_damp = 4e-5 # Viscous damping (N*m*s/rad)
Kt = 0.032   # Torque constant (N*m/A)
Ke = 0.0435  # Back EMF constant (V*s/rad)
Vin = 36.0   # Input battery voltage (Volts)

# Target constraints
target_rpm = 4000.0
target_w = target_rpm * (2 * np.pi / 60.0) # Convert RPM to rad/s

# 1. NEW STATE VECTOR (Mechanical Only)
w = MX.sym("w")
theta = MX.sym("theta")
x = vertcat(w, theta) # Reduced to a 2x1 vector

# 2. NEW CONTROL INPUTS (Setpoints for the SMC)
i_la_ref = MX.sym("i_la_ref")
vc_ref = MX.sym("vc_ref")
u = vertcat(i_la_ref, vc_ref)

# 3. CONTINUOUS DYNAMICS (Assuming electrical loop settles instantly)
w_dot = (Kt/J) * i_la_ref - (b_damp/J) * w
theta_dot = w
x_dot = vertcat(w_dot, theta_dot)

f = Function('f', [x, u], [x_dot])

# RK4 Integration
k1 = f(x, u)
k2 = f(x + dT/2 * k1, u)
k3 = f(x + dT/2 * k2, u)
k4 = f(x + dT * k3, u)
x_next = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

# Discrete time dynamics function
F_discrete = Function('F_discrete', [x, u], [x_next])

# Optimization
opti = Opti()

X = opti.variable(2, N+1)  # State trajectory
U = opti.variable(2, N)    # Control trajectory (references)

# Initial state: 0 RPM, 0 radians
x_initial = opti.parameter(2, 1)
opti.set_value(x_initial, [0.0, 0.0])  

opti.subject_to(X[:, 0] == x_initial)  

# Weights
Q_w = 100.0  # High priority on hitting target velocity
R_i = 1.0    # Small penalty to minimize extreme current usage
R_v = 0.1    # Small penalty to minimize high voltage targets

cost = 0

for k in range(N):
    opti.subject_to(X[:, k + 1] == F_discrete(X[:, k], U[:, k])) 
    
    w_k = X[0, k]
    i_ref_k = U[0, k]
    vc_ref_k = U[1, k]
    
    # SMC REACHABILITY CONSTRAINTS
    # The capacitor voltage MUST be high enough to overcome resistive drop and back-EMF.
    # We add a 1.0V margin to ensure robust sliding mode control.
    # We use two linear constraints to act as an absolute value bounding box.
    margin = 1.0
    opti.subject_to(vc_ref_k >= (Ra * i_ref_k + Ke * w_k) + margin)
    opti.subject_to(vc_ref_k >= -(Ra * i_ref_k + Ke * w_k) + margin)

    # Physical hardware limits
    opti.subject_to(opti.bounded(-1.5, i_ref_k, 1.5)) # Max current roughly Vin/Ra
    opti.subject_to(opti.bounded(0, vc_ref_k, Vin))   # Cannot ask for more voltage than battery has

    # Cost function (Ignore theta since we only care about velocity for now)
    velocity_error = w_k - target_w
    cost += Q_w * (velocity_error**2) + R_i * (i_ref_k**2) + R_v * (vc_ref_k**2)

# Terminal cost
velocity_error_terminal = X[0, N] - target_w
cost += Q_w * (velocity_error_terminal**2)

opti.minimize(cost)

opts = {"ipopt.print_level": 0, "print_time": 0}
opti.solver("ipopt", opts)

try:
    sol = opti.solve()
    print("Solver converged! 🚀")
    print(f"Targeting a constant {target_rpm} RPM ({target_w:.3f} rad/s)\n")
    
    for k in range(N):
        w_val = sol.value(X[0, k])
        i_ref_val = sol.value(U[0, k])
        vc_ref_val = sol.value(U[1, k])
        
        # Convert rad/s back to RPM for readable console output
        rpm_val = w_val * (60.0 / (2 * np.pi))
        
        print(f"Step {k:02d}: Speed = {rpm_val:05.2f} RPM | SMC Setpoints -> Target Current: {i_ref_val:5.2f} A, Target Voltage: {vc_ref_val:5.2f} V")
        
except RuntimeError:
    print("Solver failed to converge. Catching debug values... 🪲")
    X_debug = opti.debug.value(X)
    print("\n--- Failed State Trajectory (X) ---")
    print(np.round(X_debug, 3))