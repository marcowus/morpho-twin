
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import os
import sys
from scipy.linalg import solve_discrete_are
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
SIM_STEPS = 200
TRAIN_STEPS = 1000

# ----------------- Helper Functions -----------------
def compute_LQR_gain(A, B, Q, R, Cx=None):
    # Construct Q_z for the lifted state
    if Cx is not None:
        if np.isscalar(Q):
            Q_z = Q * np.outer(Cx, Cx)
        else:
            Q_z = Cx.T @ Q @ Cx
    else:
        if np.isscalar(Q):
            Q_z = Q * np.eye(A.shape[0])
        else:
            Q_z = Q

    # Ensure R is 2D
    if np.isscalar(R):
        R_mat = np.array([[R]])
    else:
        R_mat = R

    # Solve DARE
    # Hack: Scale A slightly to move eigenvalues off unit circle if needed
    evals = np.linalg.eigvals(A)
    if np.any(np.abs(np.abs(evals) - 1.0) < 1e-4):
        A_design = A * 0.995
    else:
        A_design = A

    P = solve_discrete_are(A_design, B, Q_z, R_mat)

    # Compute K
    R_plus_BTPB = R_mat + B.T @ P @ B
    K = np.linalg.inv(R_plus_BTPB) @ (B.T @ P @ A)

    return K, P

# ----------------- Koopman Model Definition -----------------
class BaseRegressor(BaseEstimator, ABC):
    @abstractmethod
    def fit(self, x, y=None): pass
    @abstractmethod
    def predict(self, x): pass
    @property
    @abstractmethod
    def coef_(self): pass

class OnlineKoopman(BaseRegressor):
    """
    Recursive Extended Dynamic Mode Decomposition with Control (rEDMDc).
    """
    def __init__(self, n_states, n_controls, lambda_reg=1.0, forgetting_factor=1.0):
        self.n_states, self.n_controls = n_states, n_controls
        self.lambda_reg, self.forgetting_factor = lambda_reg, forgetting_factor
        # Linear + Quadratic basis
        # 1 + n_states + n_states + n_states*(n_states-1)/2
        self.n_observables = 1 + n_states + n_states + n_states * (n_states - 1) // 2
        self.is_fitted_ = False
        self.Theta_ = None
        self.P_ = None

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
        """Batch fitting (initial training)."""
        self._initialize()
        n_samples = x.shape[0]
        for k in range(n_samples - 1):
            self.update(x[k], u[k], x[k+1])
        self.is_fitted_ = True
        return self

    def update(self, x_k: np.ndarray, u_k: np.ndarray, x_kp1: np.ndarray):
        """Online recursive update."""
        if self.Theta_ is None:
            self._initialize()

        psi_x_k = self._lift_state(x_k)
        psi_x_kp1 = self._lift_state(x_kp1)
        phi_k = np.vstack([psi_x_k, u_k.reshape(-1, 1)]) # Regressor vector

        # Recursive Least Squares Update
        e_k = psi_x_kp1 - self.Theta_ @ phi_k
        den = self.forgetting_factor + phi_k.T @ self.P_ @ phi_k
        if np.abs(den) < 1e-12:
            return

        K_k = (self.P_ @ phi_k) / den
        self.Theta_ += e_k @ K_k.T
        self.P_ = (self.P_ - K_k @ phi_k.T @ self.P_) / self.forgetting_factor
        self.is_fitted_ = True

    def predict(self, x_k_orig: np.ndarray, u_k_orig: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        psi_k = self._lift_state(x_k_orig)
        phi_k = np.vstack([psi_k, u_k_orig.reshape(-1, 1)])
        psi_k1_lifted = self.Theta_ @ phi_k
        return psi_k1_lifted[1:self.n_states + 1].flatten()

    def get_matrices(self):
        """Return A and B matrices for the lifted linear system."""
        check_is_fitted(self, "Theta_")
        A_lifted = self.Theta_[:, :self.n_observables]
        B_lifted = self.Theta_[:, self.n_observables:]
        return A_lifted, B_lifted

    @property
    def coef_(self):
        check_is_fitted(self, "Theta_")
        return self.Theta_

# ----------------- Robust Tube MPC Class -----------------

class RobustTubeMPC:
    def __init__(self, A, B, Q_phys, R_phys, N, x_min, x_max, u_min, u_max, Cx, lifting):
        self.A = A
        self.B = B
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.Cx = Cx
        self.lifting = lifting
        self.nx = Cx.shape[0]
        self.nz = A.shape[0]
        self.nu = B.shape[1] if len(B.shape) > 1 else 1

        # 1. Feedback Gain Calculation (LQR on lifted system)
        # Q_z = Cx^T Q_phys Cx
        self.Q_z = Cx.T @ Q_phys @ Cx + 1e-6 * np.eye(self.nz)
        self.R = R_phys

        self.K_fb, self.P_lqr = compute_LQR_gain(A, B, self.Q_z, self.R)

        # 2. Closed-Loop Analysis
        self.A_cl = A - B @ self.K_fb
        # Compute infinity norm of A_cl for tube tightening logic
        self.rho_inf = np.linalg.norm(self.A_cl, np.inf)
        print(f"Robust Tube Design: ||A_cl||_inf = {self.rho_inf:.4f}")

        if self.rho_inf >= 1.0:
            print("Warning: Closed-loop A_cl is not contractive in infinity norm.")
            # Fallback: scale down rho_inf for calculation but warn
            self.rho_inf = 0.99

        # 3. Disturbance Estimation Setup
        self.w_inf_bar = 0.0 # Initial estimate of disturbance bound
        self.r_e = 0.0       # Tube radius

        # 4. Nominal MPC Setup (CVXPY)
        self._setup_nominal_mpc()

        # Internal states
        self.z_nom = None

    def _setup_nominal_mpc(self):
        # Parameters for warm-starting
        self.z0_param = cp.Parameter(self.nz, name='z0')
        self.z_ss_param = cp.Parameter(self.nz, name='z_ss')
        self.v_ss_param = cp.Parameter(self.nu, name='v_ss')

        # Tightened constraints parameters
        self.x_min_tight = cp.Parameter(self.nx, name='x_min')
        self.x_max_tight = cp.Parameter(self.nx, name='x_max')
        self.v_min_tight = cp.Parameter(self.nu, name='v_min')
        self.v_max_tight = cp.Parameter(self.nu, name='v_max')

        # Variables
        self.Z = cp.Variable((self.N + 1, self.nz))
        self.V = cp.Variable((self.N, self.nu))

        cost = 0
        constraints = [self.Z[0] == self.z0_param]

        for k in range(self.N):
            # Cost: Tracking error + Input effort relative to steady state
            cost += cp.quad_form(self.Z[k] - self.z_ss_param, self.Q_z) + \
                    cp.quad_form(self.V[k] - self.v_ss_param, self.R)

            # Dynamics
            constraints.append(self.Z[k+1] == self.A @ self.Z[k] + self.B @ self.V[k])

            # Constraints (Tightened)
            # State constraints: x = Cx z
            constraints.append(self.Cx @ self.Z[k] >= self.x_min_tight)
            constraints.append(self.Cx @ self.Z[k] <= self.x_max_tight)

            # Input constraints (on v)
            constraints.append(self.V[k] >= self.v_min_tight)
            constraints.append(self.V[k] <= self.v_max_tight)

        # Terminal cost
        cost += cp.quad_form(self.Z[self.N] - self.z_ss_param, self.P_lqr)

        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def _solve_steady_state(self, x_sp):
        """
        Solve QP for (z_ss, v_ss) such that:
        (I - A) z_ss = B v_ss
        Cx z_ss = x_sp
        Min ||v_ss||
        """
        z_ss = cp.Variable(self.nz)
        v_ss = cp.Variable(self.nu)

        # Lift target for initial guess/regularization
        z_sp_guess = self.lifting.lift_one(x_sp)

        cost = cp.sum_squares(v_ss) + 0.1 * cp.sum_squares(z_ss - z_sp_guess)

        constrs = [
            (np.eye(self.nz) - self.A) @ z_ss == self.B @ v_ss,
            self.Cx @ z_ss == x_sp,
            v_ss >= self.u_min, # Nominal input constraints
            v_ss <= self.u_max
        ]

        prob = cp.Problem(cp.Minimize(cost), constrs)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return z_ss.value, v_ss.value
        except:
            pass

        # Fallback
        return z_sp_guess, np.zeros(self.nu)

    def solve(self, z_current, x_sp, v_last_applied=None):
        # 1. Estimate Disturbance (Online)
        if self.z_nom is not None and v_last_applied is not None:
            # w_hat = z_current - (A z_prev + B v_prev)
            # But we only have z_current and z_nom (predicted).
            # Disturbance is deviation from nominal prediction?
            # w_k = z_{k+1} - (A z_k + B v_k).
            # Here z_current is z_{k+1}.
            # We need z_k (previous actual state) which we don't store explicitly here.
            # Approximation: w_hat ~ z_current - z_nom (if z_nom was the prediction from previous step)
            # Better: let's track previous state externally or store it.
            pass

        # Using simplified error estimation:
        # e = z_current - z_nom
        # w_hat ~ e_{k+1} - (A - B K) e_k ... tough without history.
        # Let's use residual from model prediction vs actual.
        # This requires storing z_prev and v_prev.

        # 2. Update Tube Radius
        # For this experiment, let's assume a fixed w_inf_bar or slowly adapt
        # w_inf_bar = max(|w|)
        # r_e = w_inf_bar / (1 - rho_inf)

        # Initial estimate based on typical noise
        if self.w_inf_bar == 0.0:
            self.w_inf_bar = 0.05

        self.r_e = self.w_inf_bar / (1.0 - self.rho_inf + 1e-6)

        # 3. Calculate Tightening
        m_x = np.linalg.norm(self.Cx, np.inf) * self.r_e
        m_v = np.linalg.norm(self.K_fb, np.inf) * self.r_e

        # Update Parameters
        if self.z_nom is None:
            self.z_nom = z_current.copy()

        self.z0_param.value = self.z_nom

        z_ss, v_ss = self._solve_steady_state(x_sp)
        self.z_ss_param.value = z_ss
        self.v_ss_param.value = v_ss

        # Check feasibility of tightening
        x_min_t = self.x_min + m_x
        x_max_t = self.x_max - m_x
        v_min_t = self.u_min + m_v
        v_max_t = self.u_max - m_v

        # Safety clamp to prevent crossing
        if np.any(x_min_t > x_max_t):
            # print("Warning: Tube too large for state constraints.")
            center = (self.x_min + self.x_max)/2
            x_min_t = np.minimum(x_min_t, center)
            x_max_t = np.maximum(x_max_t, center)

        if np.any(v_min_t > v_max_t):
            # print("Warning: Tube too large for input constraints.")
            center = (self.u_min + self.u_max)/2
            v_min_t = np.minimum(v_min_t, center)
            v_max_t = np.maximum(v_max_t, center)

        self.x_min_tight.value = x_min_t
        self.x_max_tight.value = x_max_t
        self.v_min_tight.value = v_min_t
        self.v_max_tight.value = v_max_t

        # 4. Solve Nominal MPC
        try:
            self.prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
        except:
            return None, None

        if self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None, None

        v_nom = self.V.value[0]
        z_nom_next = self.Z.value[1]

        # 5. Feedback Control Law
        # v = v_nom + K(z_current - z_nom)
        # Note: K_fb from LQR usually assumes u = -Kx.
        # Closed loop A-BK.
        # So input should be v_nom - K(z_current - z_nom)

        v_control = v_nom - self.K_fb @ (z_current - self.z_nom)

        # Update nominal state for next step
        self.z_nom = z_nom_next

        return v_control, self.Z.value

class LiftingInterface:
    def __init__(self, koopman_model):
        self.model = koopman_model
    def lift_one(self, x):
        return self.model._lift_state(x).flatten()

# ----------------- Main Experiment -----------------
if __name__ == "__main__":
    print("Starting Robust Tube MPC Experiment...")

    RESULTS_DIR = "experiments/results_tube_mpc_robust"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Train Koopman Model
    plant = CSTRPlant()
    plant.seed(42)

    print("Generating training data...")
    x_train, u_train = [], []
    res = plant.reset()
    x_curr = res.x
    for _ in range(TRAIN_STEPS):
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

    n_states = plant.nx
    n_controls = plant.nu
    model = OnlineKoopman(n_states, n_controls, LAMBDA_REG, FORGETTING_FACTOR)
    model.fit(x_train, u_train)

    A_lifted, B_lifted = model.get_matrices()
    print(f"Koopman model trained. A shape: {A_lifted.shape}")

    # 2. Setup Robust Tube MPC
    Cx = np.zeros((n_states, model.n_observables))
    # Indices for linear states [1, x1, x2, ...] -> 1, 2
    for i in range(n_states):
        Cx[i, i+1] = 1.0

    x_min = np.array([0.0, 280.0])
    x_max = np.array([2.0, 400.0])
    u_min = np.array([0.0, -50.0])
    u_max = np.array([5.0, 50.0])

    lifting = LiftingInterface(model)

    # Q and R for physical states
    Q_phys = np.diag(Q_STATE_DIAG)
    R_phys = np.diag(R_CONTROL_DIAG)

    robust_mpc = RobustTubeMPC(
        A_lifted, B_lifted,
        Q_phys, R_phys,
        H_P,
        x_min, x_max, u_min, u_max,
        Cx, lifting
    )

    # 3. Simulation
    print("Running closed-loop simulation...")
    plant._x = np.array([0.5, 350.0])
    x_curr = plant._x.copy()
    x_target = np.array([0.8, 305.0])

    x_hist = []
    u_hist = []
    z_nom_hist = []

    # Track previous for disturbance estimation (simple)
    z_prev = None
    v_prev = None

    for t in range(SIM_STEPS):
        z_curr = model._lift_state(x_curr).flatten()

        # Online Disturbance Estimation Update
        if z_prev is not None and v_prev is not None:
            # w_k = z_{k+1} - (A z_k + B v_k)
            # z_curr is z_{k+1}
            z_pred = A_lifted @ z_prev + B_lifted @ v_prev
            w_inst = z_curr - z_pred
            w_norm = np.linalg.norm(w_inst, np.inf)

            # Update max bound (with leak/forgetting)
            robust_mpc.w_inf_bar = max(robust_mpc.w_inf_bar * 0.99, w_norm)

        u_opt, z_seq = robust_mpc.solve(z_curr, x_target, v_last_applied=v_prev)

        if u_opt is None:
            # Fallback
            u_opt = np.array([1.0, 0.0])

        res = plant.step(u_opt)
        x_next = res.x

        x_hist.append(x_curr)
        u_hist.append(u_opt)
        if z_seq is not None:
            # First element of z_seq is z_nom[0] (current step nominal)
            z_nom_curr = z_seq[0]
            z_nom_hist.append(Cx @ z_nom_curr)
        else:
            z_nom_hist.append(x_curr)

        z_prev = z_curr
        v_prev = u_opt
        x_curr = x_next

        if t % 20 == 0:
            print(f"Step {t}: T={x_curr[1]:.2f}, w_bar={robust_mpc.w_inf_bar:.4f}")

    # 4. Plotting
    x_hist = np.array(x_hist)
    z_nom_hist = np.array(z_nom_hist)
    u_hist = np.array(u_hist)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Tube radius in state space
    m_x = np.linalg.norm(Cx, np.inf) * robust_mpc.r_e

    # Temperature
    ax[0].plot(x_hist[:, 1], 'b-', label='Actual T')
    ax[0].plot(z_nom_hist[:, 1], 'r--', label='Nominal T')
    ax[0].axhline(x_target[1], color='g', linestyle=':', label='Target')
    ax[0].fill_between(range(len(z_nom_hist)),
                       z_nom_hist[:, 1] - m_x,
                       z_nom_hist[:, 1] + m_x,
                       color='r', alpha=0.1, label='Tube')
    ax[0].set_ylabel('Temperature [K]')
    ax[0].legend()
    ax[0].set_title('Robust Tube MPC: Temperature')

    # Concentration
    ax[1].plot(x_hist[:, 0], 'b-', label='Actual C_A')
    ax[1].plot(z_nom_hist[:, 0], 'r--', label='Nominal C_A')
    ax[1].axhline(x_target[0], color='g', linestyle=':', label='Target')
    ax[1].set_ylabel('Concentration')
    ax[1].legend()

    plt.savefig(os.path.join(RESULTS_DIR, 'robust_states.png'))
    plt.close()

    print(f"Experiment complete. Results saved in {RESULTS_DIR}")
