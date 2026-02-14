
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
        # print("Warning: Eigenvalues on unit circle. Scaling A for LQR design.")
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

# ----------------- Tube MPC Classes -----------------

class LiftingInterface:
    def __init__(self, koopman_model):
        self.model = koopman_model

    def lift_one(self, x, u=None):
        # We only lift state for z vector. u is separate in this formulation usually?
        # TubeMPC assumes z contains u? No, z is state, v is input.
        # But wait, linear MPC uses u = K z?
        # The user's LinearMPC uses u_ss = k * x_sp / (1-x_sp) logic which is specific to their problem.
        # We need to adapt `lift_one` to return z for our Koopman basis.
        if isinstance(x, float): x = np.array([x])
        if x.size != self.model.n_states:
            # Pad or fail? Let's assume x matches n_states
            # If x is scalar but n_states > 1, this is an issue.
            pass
        return self.model._lift_state(x).flatten()

    def project(self, z, Cx=None, Cu=None):
        # Project lifted state z back to valid manifold
        # For our basis [1, x, x^2, ...], we can extract x from indices 1:n+1
        # Then re-lift.
        n = self.model.n_states
        x = z[1:n+1]
        return self.lift_one(x)

class LinearMPC:
    def __init__(self, A, B, Q, R, P, N, x_min, x_max, u_min, u_max, Cx, lifting=None):
        self.A = A
        self.B = B
        self.Q = Q # Matrix Q for z
        self.R = R
        self.P = P
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.Cx = Cx
        self.lifting = lifting

        self.nz = A.shape[0]
        self.nu = B.shape[1] if len(B.shape) > 1 else 1

    def solve(self, z0, x_sp, d_est=None):
        """
        Solve Nominal MPC.
        """
        if d_est is None:
            if self.Cx.shape[0] == 1: d_est = 0.0
            else: d_est = np.zeros(self.Cx.shape[0])

        Z = cp.Variable((self.N + 1, self.nz))
        V = cp.Variable((self.N, self.nu)) # Control inputs

        cost = 0
        constraints = [Z[0] == z0]

        # Reference for z
        # We aim for x -> x_sp.
        # z_sp should be lifted x_sp.
        if self.lifting is not None:
            # We assume steady state u is roughly 0 or unknown for now
            # Better: find (z_ss, u_ss) such that z_ss = A z_ss + B u_ss and Cx z_ss = x_sp
            # For now, just lift the target state and assume u=0 for lifting (input-independent basis)
            z_ss = self.lifting.lift_one(x_sp)
        else:
            z_ss = np.zeros(self.nz)

        # Build Cost and Constraints
        for k in range(self.N):
            # Tracking cost in Z space (using Q_z provided in __init__)
            # Q is already lifted Q_z
            cost += cp.quad_form(Z[k] - z_ss, self.Q) + cp.quad_form(V[k], self.R)

            # Dynamics
            constraints.append(Z[k+1] == self.A @ Z[k] + self.B @ V[k])

            # Input Constraints
            constraints.append(V[k] >= self.u_min)
            constraints.append(V[k] <= self.u_max)

            # State Constraints (Physical x)
            # x = Cx z
            # x_min <= Cx z + d_est <= x_max
            x_k_model = self.Cx @ Z[k]
            constraints.append(x_k_model >= self.x_min - d_est)
            constraints.append(x_k_model <= self.x_max - d_est)

        # Terminal cost
        cost += cp.quad_form(Z[self.N] - z_ss, self.P)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except:
            return None, None

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None, None

        return V.value, Z.value

class TubeMPC:
    def __init__(self, A, B, Q_phys, R_phys, N, x_min, x_max, u_min, u_max, Cx,
                 w_bar_x, w_bar_u, lifting=None):
        self.A = A
        self.B = B
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.Cx = Cx
        self.lifting = lifting

        # 1. Compute Feedback Gain K (LQR on lifted system)
        # We need Q_z and R matrices for the lifted system DARE
        # Q_phys is (nx, nx), R_phys is (nu, nu)
        # Q_z = Cx.T Q_phys Cx
        self.Q_z = Cx.T @ Q_phys @ Cx
        # Regularize Q_z to be positive definite for numerical stability
        self.Q_z += 1e-6 * np.eye(A.shape[0])
        self.R = R_phys

        self.K_fb, self.P_nominal = compute_LQR_gain(A, B, self.Q_z, self.R)
        # LQR u = -K z. So K_fb is the gain matrix.
        # Tube controller: u = v + K_fb(z - z_nom)
        # We need K_fb such that A_cl = A - B K_fb is stable.

        # 2. Compute Margins (Simplified)
        # In a real Tube MPC, we compute robust invariant sets.
        # Here we use the heuristic tightening factors from the user's snippet.
        self.A_cl = A - B @ self.K_fb
        evals = np.linalg.eigvals(self.A_cl)
        rho = np.max(np.abs(evals))

        print(f"Tube MPC Design: Spectral Radius rho={rho:.4f}")

        if rho < 0.99:
            tightening = 1.0 / (1.0 - rho)
        else:
            tightening = 10.0

        self.margin_x = w_bar_x * tightening
        self.margin_u = w_bar_u * tightening

        print(f"Margins: x={self.margin_x}, u={self.margin_u}")

        # 3. Instantiate Nominal MPC
        x_min_tight = x_min + self.margin_x
        x_max_tight = x_max - self.margin_x
        u_min_tight = u_min + self.margin_u
        u_max_tight = u_max - self.margin_u

        self.nominal_mpc = LinearMPC(
            A, B, self.Q_z, self.R, self.P_nominal, N,
            x_min_tight, x_max_tight,
            u_min_tight, u_max_tight,
            Cx, lifting=lifting
        )

        self.z_nom = None
        self.d_est = np.zeros(Cx.shape[0])

    def solve(self, z_current, x_sp):
        if self.z_nom is None:
            self.z_nom = z_current.copy()

        # Disturbance estimation (optional, simple filter)
        y_meas = self.Cx @ z_current
        y_nom = self.Cx @ self.z_nom
        d_inst = y_meas - y_nom
        self.d_est = 0.8 * self.d_est + 0.2 * d_inst

        # Solve Nominal MPC
        v_seq, z_seq = self.nominal_mpc.solve(self.z_nom, x_sp, d_est=self.d_est)

        if v_seq is None:
            print("Nominal MPC failed.")
            return None, None

        v_0 = v_seq[0]
        z_nom_next = z_seq[1]

        # Tube Feedback Law: u = v + K(z_current - z_nom)
        # Note: LQR gain K is usually defined for u = -Kx.
        # So here u = v - K(z - z_nom)
        # Check sign convention in compute_LQR_gain.
        # K = inv(...) B' P A.
        # u = -K x is the stabilizing law.

        u_control = v_0 - self.K_fb @ (z_current - self.z_nom)

        # Clip
        u_control = np.clip(u_control, self.u_min, self.u_max)

        # Update nominal state
        self.z_nom = z_nom_next

        return u_control, z_seq

# ----------------- Main Experiment -----------------
if __name__ == "__main__":
    print("Starting Koopman Tube MPC Experiment...")

    RESULTS_DIR = "experiments/results_tube_mpc"
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

    # 2. Setup Tube MPC
    # Cx maps z -> x. For our basis [1, x1, x2, ...], x is at indices 1:n+1
    # z is size n_observables. x is size n_states.
    Cx = np.zeros((n_states, model.n_observables))
    for i in range(n_states):
        Cx[i, i+1] = 1.0

    # Physical constraints
    x_min = np.array([0.0, 280.0])
    x_max = np.array([2.0, 400.0])
    u_min = np.array([0.0, -50.0])
    u_max = np.array([5.0, 50.0])

    # Disturbance bounds (estimated)
    w_bar_x = np.array([0.01, 0.5]) # Based on CSTR noise
    w_bar_u = np.array([0.01, 0.1]) # Input disturbance?

    lifting = LiftingInterface(model)

    tube_mpc = TubeMPC(
        A_lifted, B_lifted,
        np.diag(Q_STATE_DIAG), np.diag(R_CONTROL_DIAG),
        H_P,
        x_min, x_max, u_min, u_max,
        Cx,
        w_bar_x, w_bar_u,
        lifting=lifting
    )

    # 3. Simulation
    print("Running closed-loop simulation...")
    plant._x = np.array([0.5, 350.0])
    x_curr = plant._x.copy()
    x_target = np.array([0.8, 305.0])

    x_hist = []
    u_hist = []
    z_nom_hist = []

    for t in range(SIM_STEPS):
        z_curr = model._lift_state(x_curr).flatten()

        u_opt, z_seq = tube_mpc.solve(z_curr, x_target)

        if u_opt is None:
            print(f"Step {t}: MPC failed. Using fallback.")
            u_opt = np.array([1.0, 0.0])

        res = plant.step(u_opt)
        x_curr = res.x

        x_hist.append(x_curr)
        u_hist.append(u_opt)
        if z_seq is not None:
            # Map z_nom back to x space for plotting
            x_nom = Cx @ tube_mpc.z_nom
            z_nom_hist.append(x_nom)
        else:
            z_nom_hist.append(x_curr) # Fallback

        if t % 20 == 0:
            print(f"Step {t}: T={x_curr[1]:.2f}, Target={x_target[1]}")

    # 4. Plotting
    x_hist = np.array(x_hist)
    z_nom_hist = np.array(z_nom_hist)
    u_hist = np.array(u_hist)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Temperature
    ax[0].plot(x_hist[:, 1], 'b-', label='Actual T')
    ax[0].plot(z_nom_hist[:, 1], 'r--', label='Nominal T (Tube Center)')
    ax[0].axhline(x_target[1], color='g', linestyle=':', label='Target')
    # Plot tube bounds around nominal?
    margin_T = tube_mpc.margin_x[1]
    ax[0].fill_between(range(len(z_nom_hist)),
                       z_nom_hist[:, 1] - margin_T,
                       z_nom_hist[:, 1] + margin_T,
                       color='r', alpha=0.1, label='Tube')
    ax[0].set_ylabel('Temperature [K]')
    ax[0].legend()
    ax[0].set_title('Tube MPC Performance: Temperature')

    # Concentration
    ax[1].plot(x_hist[:, 0], 'b-', label='Actual C_A')
    ax[1].plot(z_nom_hist[:, 0], 'r--', label='Nominal C_A')
    ax[1].axhline(x_target[0], color='g', linestyle=':', label='Target')
    ax[1].set_ylabel('Concentration')
    ax[1].legend()

    plt.savefig(os.path.join(RESULTS_DIR, 'tube_mpc_states.png'))
    plt.close()

    # Inputs
    plt.figure(figsize=(10, 4))
    plt.plot(u_hist[:, 0], label='Flow q')
    plt.plot(u_hist[:, 1], label='Coolant Q_c')
    plt.ylabel('Input')
    plt.legend()
    plt.title('Control Inputs')
    plt.savefig(os.path.join(RESULTS_DIR, 'tube_mpc_inputs.png'))
    plt.close()

    print(f"Experiment complete. Results saved in {RESULTS_DIR}")
