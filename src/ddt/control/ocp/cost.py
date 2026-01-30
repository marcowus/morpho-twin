"""Cost function builders for OCP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

try:
    import casadi as ca
except ImportError:
    ca = None  # type: ignore[assignment]

if TYPE_CHECKING:
    pass


@dataclass
class QuadraticCost:
    """Quadratic cost function components."""

    Q: np.ndarray  # State tracking weight
    R: np.ndarray  # Input regularization weight
    Q_terminal: np.ndarray  # Terminal state weight
    lambda_info: float  # FIM probing weight


def build_tracking_cost(
    Q_diag: list[float],
    R_diag: list[float],
    Q_terminal_scale: float = 1.0,
) -> QuadraticCost:
    """Build quadratic tracking cost.

    J = Σ ||x_i - x_ref||²_Q + ||u_i||²_R + ||x_N - x_ref||²_{Q_T}

    Args:
        Q_diag: Diagonal elements of Q
        R_diag: Diagonal elements of R
        Q_terminal_scale: Scale factor for terminal cost (often > 1)

    Returns:
        QuadraticCost object
    """
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)
    Q_terminal = Q_terminal_scale * Q

    return QuadraticCost(Q=Q, R=R, Q_terminal=Q_terminal, lambda_info=0.0)


def build_dual_control_cost(
    Q_diag: list[float],
    R_diag: list[float],
    lambda_info: float = 0.01,
    Q_terminal_scale: float = 1.0,
) -> QuadraticCost:
    """Build dual-control cost with FIM probing term.

    J = Σ ||x_i - x_ref||²_Q + ||u_i||²_R + λ * tr((F + εI)^{-1})

    Args:
        Q_diag: Diagonal elements of Q
        R_diag: Diagonal elements of R
        lambda_info: Weight on A-optimal FIM cost
        Q_terminal_scale: Scale factor for terminal cost

    Returns:
        QuadraticCost object with lambda_info set
    """
    cost = build_tracking_cost(Q_diag, R_diag, Q_terminal_scale)
    cost.lambda_info = lambda_info
    return cost


def stage_cost_casadi(
    x: ca.SX,
    u: ca.SX,
    x_ref: ca.SX,
    Q: np.ndarray,
    R: np.ndarray,
) -> ca.SX:
    """Build CasADi expression for stage cost.

    l(x, u) = ||x - x_ref||²_Q + ||u||²_R
    """
    if ca is None:
        raise ImportError("CasADi required.")

    dx = x - x_ref
    cost = ca.mtimes([dx.T, Q, dx]) + ca.mtimes([u.T, R, u])
    return cost


def terminal_cost_casadi(
    x: ca.SX,
    x_ref: ca.SX,
    Q_terminal: np.ndarray,
) -> ca.SX:
    """Build CasADi expression for terminal cost.

    l_N(x) = ||x - x_ref||²_{Q_T}
    """
    if ca is None:
        raise ImportError("CasADi required.")

    dx = x - x_ref
    return ca.mtimes([dx.T, Q_terminal, dx])


def evaluate_stage_cost(
    x: np.ndarray,
    u: np.ndarray,
    x_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> float:
    """Evaluate stage cost numerically."""
    dx = x - x_ref
    return float(dx.T @ Q @ dx + u.T @ R @ u)


def evaluate_total_cost(
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    x_ref: np.ndarray,
    cost: QuadraticCost,
) -> float:
    """Evaluate total trajectory cost.

    Args:
        x_traj: State trajectory (N+1, nx)
        u_traj: Input trajectory (N, nu)
        x_ref: Reference (nx,) or (N+1, nx)
        cost: QuadraticCost object

    Returns:
        Total cost value
    """
    N = len(u_traj)
    total = 0.0

    # Handle scalar or trajectory reference
    if x_ref.ndim == 1:
        refs = np.tile(x_ref, (N + 1, 1))
    else:
        refs = x_ref

    # Stage costs
    for i in range(N):
        x_i = x_traj[i].reshape(-1, 1)
        u_i = u_traj[i].reshape(-1, 1)
        ref_i = refs[i].reshape(-1, 1)
        total += evaluate_stage_cost(x_i, u_i, ref_i, cost.Q, cost.R)

    # Terminal cost
    x_N = x_traj[N].reshape(-1, 1)
    ref_N = refs[N].reshape(-1, 1)
    dx = x_N - ref_N
    total += float(dx.T @ cost.Q_terminal @ dx)

    return total
