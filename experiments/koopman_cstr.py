
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import os
import time
import sys
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ddt.sim.cstr import CSTRPlant

# ==================== Plotting Style Setup ====================
plt.style.use('seaborn-v0_8-whitegrid')

# ----------------- Adjustable Parameters -----------------
Q_STATE_DIAG = [100.0, 10.0]  # Weights for [C_A, T]
R_CONTROL_DIAG = [0.1, 0.001] # Weights for [q, Q_c]
LAMBDA_REG = 1e-3
FORGETTING_FACTOR = 0.995
H_P = 15  # Prediction Horizon
H_C = 5   # Control Horizon
MAX_DEVIATION_FROM_HISTORICAL_PERCENT = 0.10
MIN_ABS_DEVIATION_ALLOWANCE = np.array([0.1, 5.0]) # [L/s, kJ/s]
SIM_STEPS = 200
TRAIN_STEPS = 1000

# ----------------- Data Generation -----------------
print("Starting data generation using CSTR Plant...")

plant = CSTRPlant()
plant.seed(42)

# Generate training data
x_train = []
u_train = []

res = plant.reset()
x_curr = res.x

for _ in range(TRAIN_STEPS):
    # Random inputs around nominal
    # q in [0.5, 1.5], Q_c in [-10, 10]
    u_rand = np.array([
        1.0 + np.random.uniform(-0.5, 0.5),
        np.random.uniform(-10.0, 10.0)
    ])

    x_train.append(x_curr)
    u_train.append(u_rand)

    res = plant.step(u_rand)
    x_curr = res.x

x_train = np.array(x_train)
u_train = np.array(u_train)

print(f"Data generation finished. Generated {len(x_train)} samples.")

# ----------------- rEDMDc Model Definition -----------------
class BaseRegressor(BaseEstimator, ABC):
    @abstractmethod
    def fit(self, x, y=None): pass
    @abstractmethod
    def predict(self, x): pass
    @property
    @abstractmethod
    def coef_(self): pass

class rEDMDc(BaseRegressor):
    def __init__(self, n_states, n_controls, lambda_reg=1.0, forgetting_factor=1.0):
        self.n_states, self.n_controls = n_states, n_controls
        self.lambda_reg, self.forgetting_factor = lambda_reg, forgetting_factor
        # Linear + Quadratic basis
        # 1 + n_states + n_states + n_states*(n_states-1)/2
        self.n_observables = 1 + n_states + n_states + n_states * (n_states - 1) // 2
        self.is_fitted_ = False
        self.state_matrix_ = None
        self.control_matrix_ = None
        self.Theta_ = None

    def _initialize(self):
        self.Theta_ = np.zeros((self.n_observables, self.n_observables + self.n_controls))
        self.P_ = np.eye(self.n_observables + self.n_controls) / max(self.lambda_reg, 1e-9)

    def _lift_state(self, x: np.ndarray) -> np.ndarray:
        x_flat = x.flatten()
        if x_flat.shape[0] != self.n_states:
            raise ValueError(f"Incorrect x dimension {x_flat.shape[0]}, expected {self.n_states}")
        psi = [1.0]
        psi.extend(x_flat.tolist())
        psi.extend([xi**2 for xi in x_flat])
        for i in range(self.n_states):
            for j in range(i + 1, self.n_states):
                psi.append(x_flat[i] * x_flat[j])
        return np.array(psi).reshape(-1, 1)

    def fit(self, x: np.ndarray, u: np.ndarray, y: np.ndarray = None):
        self._initialize()
        n_samples = x.shape[0]
        for k in range(n_samples - 1):
            x_k, u_k = x[k, :], u[k, :]
            psi_x_k, psi_x_k1 = self._lift_state(x_k), self._lift_state(x[k + 1, :])
            phi_k = np.vstack([psi_x_k, u_k.reshape(-1, 1)])

            # Recursive update
            e_k = psi_x_k1 - self.Theta_ @ phi_k
            den = self.forgetting_factor + phi_k.T @ self.P_ @ phi_k

            if np.abs(den) < 1e-12:
                continue

            K_k = (self.P_ @ phi_k) / den
            self.Theta_ += e_k @ K_k.T
            self.P_ = (self.P_ - K_k @ phi_k.T @ self.P_) / self.forgetting_factor

        self.state_matrix_ = self.Theta_[:, :self.n_observables]
        self.control_matrix_ = self.Theta_[:, self.n_observables:]
        self.is_fitted_ = True
        return self

    def predict(self, x_k_orig: np.ndarray, u_k_orig: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        psi_k = self._lift_state(x_k_orig)
        phi_k = np.vstack([psi_k, u_k_orig.reshape(-1, 1)])
        psi_k1_lifted = self.Theta_ @ phi_k
        # Extract original states from lifted vector (indices 1 to n_states+1)
        return psi_k1_lifted[1:self.n_states + 1].flatten()

    @property
    def coef_(self):
        check_is_fitted(self, "Theta_")
        return self.Theta_

# ----------------- Model Training -----------------
print("Training Koopman rEDMDc model...")
n_states = plant.nx
n_controls = plant.nu
mpc_model = rEDMDc(n_states, n_controls, LAMBDA_REG, FORGETTING_FACTOR)
mpc_model.fit(x_train, u=u_train)
print("Model training complete.")

A_lifted, B_lifted = mpc_model.state_matrix_, mpc_model.control_matrix_

# Target state (example setpoint)
x_target = np.array([0.8, 305.0]) # [C_A, T]
Q_cost = np.diag(Q_STATE_DIAG)
R_cost = np.diag(R_CONTROL_DIAG)

u_min = np.array([0.0, -50.0]) # [q, Q_c]
u_max = np.array([5.0, 50.0])

def setup_mpc_problem_deviation_constrained(psi_current_val, u_hist_baseline, x_target_val, A_l, B_l, Q_c, R_c, H_p_val, H_c_val, u_min_val, u_max_val, max_dev_abs, n_s, n_c, n_obs):
    U_optim = cp.Variable((H_c_val, n_c), name="U_mpc")
    Psi_pred = cp.Variable((H_p_val + 1, n_obs), name="Psi_lifted_mpc")
    cost = 0
    constraints = [Psi_pred[0, :] == psi_current_val.flatten()]

    if H_p_val > H_c_val:
        last_u_repeated = cp.vstack([U_optim[-1, :] for _ in range(H_p_val - H_c_val)])
        U_horizon = cp.vstack([U_optim, last_u_repeated])
    else:
        U_horizon = U_optim[:H_p_val,:]

    for k in range(H_p_val):
        constraints.append(Psi_pred[k + 1, :] == Psi_pred[k, :] @ A_l.T + U_horizon[k, :] @ B_l.T)
        # Cost on original states (indices 1 to n_s+1 in lifted vector)
        cost += cp.quad_form(Psi_pred[k + 1, 1:n_s + 1] - x_target_val, Q_c)
        if k < H_c_val:
            cost += cp.quad_form(U_optim[k, :], R_c)
            constraints.extend([U_optim[k, :] >= u_min_val, U_optim[k, :] <= u_max_val])

    # Deviation constraints from baseline (if baseline provided)
    if u_hist_baseline is not None:
        constraints.extend([
            U_optim[0, :] <= u_hist_baseline + max_dev_abs,
            U_optim[0, :] >= u_hist_baseline - max_dev_abs
        ])

    problem = cp.Problem(cp.Minimize(cost), constraints)
    return problem, U_optim

# ----------------- MPC Simulation -----------------
print("Starting MPC closed-loop simulation...")

RESULTS_DIR = "experiments/results_koopman"
os.makedirs(RESULTS_DIR, exist_ok=True)

x_sim = []
u_sim = []
x_pred_mpc = [] # One-step ahead predictions

# Reset plant for simulation
res = plant.reset()
# Start at a different initial condition if possible or drift it
plant._x = np.array([0.2, 340.0]) # Start far from target
x_curr = plant._x.copy()

# Initial guess for MPC
initial_U_guess = np.tile(np.array([1.0, 0.0]), (H_C, 1))
prev_optimal_U = None

solver_opts = {'solver': cp.OSQP, 'verbose': False, 'eps_abs': 1e-4, 'eps_rel': 1e-4}

for t in range(SIM_STEPS):
    # Lift current state
    psi_curr = mpc_model._lift_state(x_curr)

    # Baseline input (e.g., previous input or nominal)
    # Here we don't have a "historical baseline" like the user's data,
    # so we can use the previous input or just relax this constraint.
    # We'll set a loose deviation from the previous applied input to ensure smooth control.
    if t > 0:
        u_baseline = u_sim[-1]
    else:
        u_baseline = np.array([1.0, 0.0])

    max_dev_abs = MIN_ABS_DEVIATION_ALLOWANCE # Use fixed allowance

    # Setup MPC
    problem, U_var = setup_mpc_problem_deviation_constrained(
        psi_curr, u_baseline, x_target,
        A_lifted, B_lifted, Q_cost, R_cost,
        H_P, H_C, u_min, u_max,
        max_dev_abs, n_states, n_controls, mpc_model.n_observables
    )

    if prev_optimal_U is not None:
        U_var.value = np.roll(prev_optimal_U, -1, axis=0)
        U_var.value[-1, :] = prev_optimal_U[-1, :]
    else:
        U_var.value = initial_U_guess

    try:
        problem.solve(**solver_opts)
    except cp.SolverError:
        print(f"Solver error at step {t}")

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and U_var.value is not None:
        u_opt = U_var.value[0, :]
        prev_optimal_U = U_var.value.copy()
    else:
        print(f"Solver failed at step {t}, using baseline")
        u_opt = u_baseline
        prev_optimal_U = None

    # Apply control
    res = plant.step(u_opt)
    x_next = res.x

    # Store results
    x_sim.append(x_curr)
    u_sim.append(u_opt)

    # Predict next state with model for comparison
    x_pred = mpc_model.predict(x_curr, u_opt)
    x_pred_mpc.append(x_pred)

    x_curr = x_next

    if t % 20 == 0:
        print(f"Step {t}/{SIM_STEPS}: x={x_curr}, target={x_target}")

x_sim = np.array(x_sim)
u_sim = np.array(u_sim)
x_pred_mpc = np.array(x_pred_mpc)

# ----------------- Plotting -----------------
print("Plotting results...")
state_names = ['Concentration C_A', 'Temperature T']
input_names = ['Flow Rate q', 'Heat Q_c']

# 1. State Trajectories
fig, axs = plt.subplots(n_states, 1, figsize=(10, 8), sharex=True)
for i in range(n_states):
    axs[i].plot(x_sim[:, i], label='Actual', linewidth=2)
    axs[i].plot(x_pred_mpc[:, i], '--', label='One-step Prediction', alpha=0.7)
    axs[i].axhline(x_target[i], color='r', linestyle=':', label='Target')
    axs[i].set_ylabel(state_names[i])
    axs[i].legend()
    axs[i].grid(True)
axs[-1].set_xlabel('Time Step')
fig.suptitle('Koopman MPC Control of CSTR')
plt.savefig(os.path.join(RESULTS_DIR, 'states.png'))
plt.close()

# 2. Control Inputs
fig, axs = plt.subplots(n_controls, 1, figsize=(10, 8), sharex=True)
for i in range(n_controls):
    axs[i].plot(u_sim[:, i], label='Control Input', color='g')
    axs[i].axhline(u_min[i], color='k', linestyle='--', alpha=0.3)
    axs[i].axhline(u_max[i], color='k', linestyle='--', alpha=0.3)
    axs[i].set_ylabel(input_names[i])
    axs[i].legend()
    axs[i].grid(True)
axs[-1].set_xlabel('Time Step')
fig.suptitle('Control Inputs')
plt.savefig(os.path.join(RESULTS_DIR, 'inputs.png'))
plt.close()

# 3. Prediction Error
errors = x_sim[:-1] - x_pred_mpc[:-1] # Compare x_t+1 actual vs predicted
mse = np.mean(errors**2, axis=0)
print(f"Prediction MSE: {mse}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(errors[:, 0], label='Error C_A')
ax.plot(errors[:, 1], label='Error T')
ax.set_ylabel('Prediction Error')
ax.set_xlabel('Time Step')
ax.legend()
ax.grid(True)
ax.set_title(f'One-step Prediction Error (MSE: {np.mean(mse):.4e})')
plt.savefig(os.path.join(RESULTS_DIR, 'prediction_error.png'))
plt.close()

print(f"Experiment complete. Results saved in {RESULTS_DIR}")
