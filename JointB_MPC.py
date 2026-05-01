import numpy as np
import matplotlib.pyplot as plt
from casadi import *

# ==========================================
# 1. PARAMETERS & INITIALIZATION
# ==========================================
dt_outer = 0.05      # MPC loop runs at 20 Hz
dt_inner = 0.00001   # SMC loop runs at 100 kHz 
sim_time = 1.25       
steps_outer = int(sim_time / dt_outer)
steps_inner_per_outer = int(dt_outer / dt_inner)
N = 10  

# Circuit & Motor constants
R = 1e6       
Ra = 35.0     
La = 1.2e-3   
L = 10e-3     
C = 100e-6    
J = 6e-7      
b_damp = 4e-5 
Kt = 0.032    
Ke = 0.0435   
Vin = 36.0    

target_rpm = 1000.0
target_w = target_rpm * (2 * np.pi / 60.0) 

# ==========================================
# 2. INNER LOOP: SMC CONTROLLERS
# ==========================================
def smc_buck_duty_cycle(v_target, v_c, i_l, i_la):
    lam, K, Phi = 500.0, 1.0, 500.0  
    S2 = -(1/C)*i_l + (1/(R*C))*v_c + (1/C)*i_la + lam*(v_target - v_c)
    H_smc = (v_c / (L*C)) + (1/(R*C) - lam) * ((1/C)*i_l - (1/(R*C))*v_c - (1/C)*i_la)
    g_smc = -Vin / (L*C)
    u_eq = -H_smc / g_smc
    u1_raw = u_eq + K * np.clip(S2 / Phi, -1.0, 1.0)
    return np.clip(u1_raw, 0.0, 1.0)

def smc_hbridge_switch(i_target, i_la, i_integral):
    """ Integral SMC for H-Bridge to prevent deadband issues """
    Ki = 5000.0  
    S1 = (i_target - i_la) + Ki * i_integral
    return 1.0 if S1 > 0 else 0.0

# ==========================================
# 3. OUTER LOOP: MPC SETUP (CasADi)
# ==========================================
w_sym = MX.sym("w")
theta_sym = MX.sym("theta")
x_mpc = vertcat(w_sym, theta_sym)

i_la_ref = MX.sym("i_la_ref")
vc_ref = MX.sym("vc_ref")
u_mpc = vertcat(i_la_ref, vc_ref)

w_dot = (Kt/J) * i_la_ref - (b_damp/J) * w_sym
theta_dot = w_sym
f_mpc = Function('f_mpc', [x_mpc, u_mpc], [vertcat(w_dot, theta_dot)])

k1 = f_mpc(x_mpc, u_mpc)
k2 = f_mpc(x_mpc + dt_outer/2 * k1, u_mpc)
k3 = f_mpc(x_mpc + dt_outer/2 * k2, u_mpc)
k4 = f_mpc(x_mpc + dt_outer * k3, u_mpc)
F_discrete = Function('F_discrete', [x_mpc, u_mpc], [x_mpc + dt_outer/6 * (k1 + 2*k2 + 2*k3 + k4)])

opti = Opti()
X = opti.variable(2, N+1)
U = opti.variable(2, N)

# ---> NEW: Create a Slack Variable to soften the constraints <---
S_v = opti.variable(1, N)
opti.subject_to(S_v >= 0) # Slack cannot be negative

x_initial = opti.parameter(2, 1)
opti.subject_to(X[:, 0] == x_initial)
u_prev = opti.parameter(2, 1)

Q_w = 100.0  
R_i = 1.0    
R_v = 0.1    

max_delta_i = 0.5  
max_delta_v = 5.0  

for k in range(N):
    opti.subject_to(X[:, k + 1] == F_discrete(X[:, k], U[:, k])) 
    w_k = X[0, k]
    i_ref_k = U[0, k]
    vc_ref_k = U[1, k]
    
    margin = 1.0
    
    # ---> THE FIX: Subtract the slack variable to soften the boundary <---
    opti.subject_to(vc_ref_k >= (Ra * i_ref_k + Ke * w_k) + margin - S_v[k])
    opti.subject_to(vc_ref_k >= -(Ra * i_ref_k + Ke * w_k) + margin - S_v[k])
    
    opti.subject_to(opti.bounded(-1.5, i_ref_k, 1.5)) 
    opti.subject_to(opti.bounded(0, vc_ref_k, Vin))   

    if k == 0:
        delta_i = i_ref_k - u_prev[0]
        delta_v = vc_ref_k - u_prev[1]
    else:
        delta_i = i_ref_k - U[0, k-1]
        delta_v = vc_ref_k - U[1, k-1]
        
    opti.subject_to(opti.bounded(-max_delta_i, delta_i, max_delta_i))
    opti.subject_to(opti.bounded(-max_delta_v, delta_v, max_delta_v))

    # ---> THE FIX: Add a MASSIVE penalty for using the slack <---
    # This ensures the solver only breaks the reachability rule if absolutely necessary to survive.
    cost = Q_w * ((w_k - target_w)**2) + R_i * (i_ref_k**2) + R_v * (vc_ref_k**2) + 10000.0 * (S_v[k]**2)
    opti.minimize(cost)

opts = {
    "ipopt.print_level": 0, 
    "print_time": 0, 
    "ipopt.sb": "yes",
    "ipopt.tol": 1e-5,               
    "ipopt.acceptable_tol": 1e-4,    
    "ipopt.constr_viol_tol": 1e-4    
}
opti.solver("ipopt", opts)

# ==========================================
# 4. FULL PLANT DYNAMICS 
# ==========================================
def plant_derivatives(state, u1, u2):
    il, vc, ila, w, theta = state
    il_dot = -vc/L + (Vin/L)*u1
    vc_dot = il/C - vc/(R*C) - ila/C
    ila_dot = -vc/La - (Ra/La)*ila - (Ke/La)*w + (2*vc/La)*u2
    w_dot = (Kt/J)*ila - (b_damp/J)*w
    theta_dot = w
    return np.array([il_dot, vc_dot, ila_dot, w_dot, theta_dot])

def rk4_step_plant(state, u1, u2, dt):
    k1 = plant_derivatives(state, u1, u2)
    k2 = plant_derivatives(state + dt/2 * k1, u1, u2)
    k3 = plant_derivatives(state + dt/2 * k2, u1, u2)
    k4 = plant_derivatives(state + dt * k3, u1, u2)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# ==========================================
# 5. SIMULATION LOOP (Cascade Architecture)
# ==========================================
current_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 

time_log, w_log, ila_log, vc_log = [], [], [], []
w_ref_log, ila_ref_log, vc_ref_log = [], [], []
u1_log, u2_log = [], []

print(f"Starting Cascade Simulation. Targeting {target_rpm} RPM...")

t = 0.0
# Initialize fallback targets just in case step 0 fails
target_i, target_v = 0.0, 0.0 
i_error_integral = 0.0  

for step in range(steps_outer):
    # --- OUTER LOOP: RUN MPC ---
    opti.set_value(x_initial, [current_state[3], current_state[4]]) 
    opti.set_value(u_prev, [target_i, target_v]) # ---> NEW: Update control memory <---
    
    try:
        sol = opti.solve()
        target_i = sol.value(U[0, 0])
        target_v = sol.value(U[1, 0])
        
        opti.set_initial(X, sol.value(X))
        opti.set_initial(U, sol.value(U))
        
    except RuntimeError:
        print(f"\n--- MPC Failed at t={t:.2f}s! 🪲 ---")
        opti.debug.show_infeasibilities() 
        print("Using previous valid targets to ride through the failure.")

# --- INNER LOOP: RUN SMC & PLANT ---
    for _ in range(steps_inner_per_outer):
        il, vc, ila, w, theta = current_state
        
        i_error_integral += (target_i - ila) * dt_inner
        
        # ---> THE FIX: Anti-Windup Clamp <---
        # Limits the maximum integral effect to a safe boundary
        i_error_integral = np.clip(i_error_integral, -0.01, 0.01) 
        
        u1_duty = smc_buck_duty_cycle(target_v, vc, il, ila)
        u2_switch = smc_hbridge_switch(target_i, ila, i_error_integral)
        
        current_state = rk4_step_plant(current_state, u1_duty, u2_switch, dt_inner)
        t += dt_inner
        
        # Log data
        time_log.append(t); w_log.append(current_state[3] * (60 / (2*np.pi)))
        ila_log.append(current_state[2]); vc_log.append(current_state[1])
        w_ref_log.append(target_rpm); ila_ref_log.append(target_i); vc_ref_log.append(target_v)
        u1_log.append(u1_duty); u2_log.append(u2_switch)

print("Simulation Complete! Generating plots...")

# ==========================================
# 6. PLOTTING
# ==========================================
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axs[0].plot(time_log, w_log, label='Actual RPM', color='blue')
axs[0].plot(time_log, w_ref_log, '--', label='Target RPM', color='black')
axs[0].set_ylabel('Speed (RPM)'); axs[0].legend(); axs[0].grid()

axs[1].plot(time_log, ila_log, label='Actual Current (ila)', color='orange')
axs[1].plot(time_log, ila_ref_log, '--', label='MPC Target Current', color='black')
axs[1].set_ylabel('Current (A)'); axs[1].legend(); axs[1].grid()

axs[2].plot(time_log, vc_log, label='Capacitor Voltage (vc)', color='green')
axs[2].plot(time_log, vc_ref_log, '--', label='MPC Target Voltage', color='black')
axs[2].set_ylabel('Voltage (V)'); axs[2].legend(); axs[2].grid()

axs[3].plot(time_log, u1_log, label='Buck Duty Cycle (u1)', alpha=0.7)
axs[3].plot(time_log, u2_log, label='H-Bridge State (u2)', alpha=0.5, linewidth=0.5)
axs[3].set_ylabel('Switch Inputs'); axs[3].set_xlabel('Time (s)')
axs[3].legend(loc='upper right'); axs[3].grid()

plt.tight_layout()
plt.show()