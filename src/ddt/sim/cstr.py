"""Continuously Stirred Tank Reactor (CSTR) benchmark plant.

This nonlinear benchmark demonstrates real control challenges:
- Nonlinear dynamics (Arrhenius reaction rate)
- Safety-critical constraints (temperature runaway prevention)
- Parameter drift (catalyst degradation)
- Need for persistent excitation

The CSTR model represents an exothermic reaction A -> B in a cooled tank:
- States: x = [C_A, T] (concentration of A, temperature)
- Inputs: u = [q, Q_c] (flow rate, coolant heat removal)
- Parameters: theta = [k_0, E_a, dH_r] (pre-exp factor, activation energy, enthalpy)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..interfaces import Plant, StepResult


@dataclass
class CSTRPlant(Plant):
    """Continuously Stirred Tank Reactor plant.

    Dynamics (continuous time):
        dC_A/dt = (C_A0 - C_A) * q/V - k(T) * C_A
        dT/dt = (T_0 - T) * q/V + (-dH_r)/(rho*C_p) * k(T) * C_A
                + Q_c / (rho*C_p*V)

    where k(T) = k_0 * exp(-E_a / (R*T)) is the Arrhenius reaction rate.

    State constraints:
        C_A >= 0 (physical)
        T <= T_max (safety - prevent runaway)

    Attributes:
        dt: Sampling time [s]

        # Physical parameters (fixed)
        V: Reactor volume [L]
        rho: Density [kg/L]
        C_p: Heat capacity [kJ/(kg*K)]
        R_gas: Gas constant [J/(mol*K)]
        C_A0: Feed concentration [mol/L]
        T_0: Feed temperature [K]

        # Uncertain parameters (true values for simulation)
        k_0_true: Pre-exponential factor [1/s]
        E_a_true: Activation energy [J/mol]
        dH_r_true: Enthalpy of reaction [J/mol]

        # Constraints
        T_max: Maximum temperature [K]
        T_min: Minimum temperature [K]

        # Noise
        process_noise_std: Process noise standard deviation [C_A, T]
        meas_noise_std: Measurement noise standard deviation [C_A, T]

        # Initial conditions
        C_A_init: Initial concentration [mol/L]
        T_init: Initial temperature [K]
    """

    dt: float = 0.1

    # Physical parameters (fixed)
    V: float = 100.0  # [L]
    rho: float = 1.0  # [kg/L]
    C_p: float = 4.18  # [kJ/(kg*K)]
    R_gas: float = 8.314  # [J/(mol*K)]
    C_A0: float = 1.0  # [mol/L]
    T_0: float = 300.0  # [K]

    # Uncertain parameters (true values)
    k_0_true: float = 7.2e10  # [1/s]
    E_a_true: float = 72750.0  # [J/mol]
    dH_r_true: float = -50000.0  # [J/mol] (exothermic, negative)

    # Constraints
    T_max: float = 400.0  # [K]
    T_min: float = 280.0  # [K]

    # Noise
    process_noise_std: np.ndarray = field(
        default_factory=lambda: np.array([0.001, 0.5])
    )
    meas_noise_std: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 1.0])
    )

    # Initial conditions
    C_A_init: float = 0.5  # [mol/L]
    T_init: float = 320.0  # [K]

    # Nominal operating input
    q_nom: float = 1.0  # [L/s]
    Q_c_nom: float = 0.0  # [kJ/s]

    # Internal state
    _x: np.ndarray = field(init=False)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._x = np.array([self.C_A_init, self.T_init], dtype=np.float64)
        self.process_noise_std = np.atleast_1d(self.process_noise_std).astype(np.float64)
        self.meas_noise_std = np.atleast_1d(self.meas_noise_std).astype(np.float64)

    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def reset(self) -> StepResult:
        """Reset to initial conditions."""
        self._x = np.array([self.C_A_init, self.T_init], dtype=np.float64)
        v = self._rng.normal(0.0, self.meas_noise_std)
        y = self._x + v
        return StepResult(x=self._x.copy(), y=y)

    def step(self, u: np.ndarray) -> StepResult:
        """Advance one time step.

        Args:
            u: Input [q, Q_c] where
               q: flow rate [L/s]
               Q_c: coolant heat removal [kJ/s]

        Returns:
            StepResult with updated state and noisy measurement
        """
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        if len(u) == 1:
            # Single input mode: use as flow rate perturbation
            q = self.q_nom + u[0]
            Q_c = self.Q_c_nom
        else:
            q = u[0]
            Q_c = u[1]

        # Current state
        C_A, T = self._x[0], self._x[1]

        # Reaction rate (Arrhenius)
        k = self._reaction_rate(T)

        # Derivatives (continuous time)
        dC_A_dt = (self.C_A0 - C_A) * q / self.V - k * C_A
        dT_dt = (
            (self.T_0 - T) * q / self.V
            + (-self.dH_r_true) / (self.rho * self.C_p) * k * C_A / 1000.0  # Convert J to kJ
            + Q_c / (self.rho * self.C_p * self.V)
        )

        # Euler integration
        w = self._rng.normal(0.0, self.process_noise_std)
        C_A_new = C_A + dC_A_dt * self.dt + w[0]
        T_new = T + dT_dt * self.dt + w[1]

        # Physical constraints (concentration non-negative)
        C_A_new = max(0.0, C_A_new)

        # Update state
        self._x = np.array([C_A_new, T_new])

        # Noisy measurement
        v = self._rng.normal(0.0, self.meas_noise_std)
        y = self._x + v

        return StepResult(x=self._x.copy(), y=y)

    def _reaction_rate(self, T: float) -> float:
        """Compute Arrhenius reaction rate.

        k(T) = k_0 * exp(-E_a / (R*T))
        """
        return float(self.k_0_true * np.exp(-self.E_a_true / (self.R_gas * T)))

    def get_true_parameters(self) -> np.ndarray:
        """Get true parameter values for estimation testing."""
        return np.array([self.k_0_true, self.E_a_true, self.dH_r_true])

    def apply_parameter_drift(self, k_0_factor: float = 1.0) -> None:
        """Apply parameter drift (e.g., catalyst degradation).

        Args:
            k_0_factor: Multiplicative factor for k_0 (< 1 for degradation)
        """
        self.k_0_true *= k_0_factor

    def is_safe(self, margin: float = 0.0) -> bool:
        """Check if current state is safe.

        Args:
            margin: Safety margin for temperature constraint

        Returns:
            True if T <= T_max - margin and T >= T_min + margin
        """
        T = self._x[1]
        return bool((T <= self.T_max - margin) and (T >= self.T_min + margin))

    def get_constraint_violation(self) -> tuple[bool, float]:
        """Check constraint violation.

        Returns:
            Tuple of (is_violated, violation_amount)
        """
        T = self._x[1]
        C_A = self._x[0]

        violations = []
        if T > self.T_max:
            violations.append(T - self.T_max)
        if T < self.T_min:
            violations.append(self.T_min - T)
        if C_A < 0:
            violations.append(-C_A)

        if violations:
            return True, max(violations)
        return False, 0.0

    @property
    def nx(self) -> int:
        """State dimension."""
        return 2

    @property
    def nu(self) -> int:
        """Input dimension."""
        return 2

    @property
    def ny(self) -> int:
        """Output dimension."""
        return 2

    @property
    def ntheta(self) -> int:
        """Parameter dimension."""
        return 3
